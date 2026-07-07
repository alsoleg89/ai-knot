"""GigaChat (Sber) LLM provider with OAuth2 token exchange.

Unlike a plain OpenAI-compatible endpoint, GigaChat does not accept a static bearer
token. You authenticate with a durable *authorization key* (the base64
``client_id:client_secret`` string from the developer cabinet), exchange it for a
short-lived access token over OAuth2, and use that token for chat completions. This
provider performs the exchange and caches the token until it nears expiry.
"""

from __future__ import annotations

import time
import uuid

import httpx

# Default GigaChat endpoints (Sber production).
GIGACHAT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"
GIGACHAT_DEFAULT_MODEL = "GigaChat"
GIGACHAT_DEFAULT_SCOPE = "GIGACHAT_API_PERS"

# Refresh the access token this many seconds before it actually expires, so a request
# never rides the edge of an expiring token.
_TOKEN_REFRESH_MARGIN_S = 60.0


class GigaChatProvider:
    """Provider for Sber's GigaChat API.

    Args:
        auth_key: GigaChat *authorization key* — the base64 ``client_id:client_secret``
            string from the developer cabinet (``GIGACHAT_API_KEY``). This is exchanged
            for a short-lived access token; it is **not** used directly as a bearer token.
        scope: API scope — ``GIGACHAT_API_PERS`` (individuals), ``GIGACHAT_API_B2B``, or
            ``GIGACHAT_API_CORP``.
        default_model: Model ID used when none is specified
            (``GigaChat`` / ``GigaChat-Pro`` / ``GigaChat-Max``).
        oauth_url: OAuth2 token endpoint.
        base_url: Chat-completions base URL (without ``/chat/completions``).
        verify_ssl: TLS verification. GigaChat serves certificates from the Russian
            Ministry of Digital Development root CA, which is not in most default trust
            stores. Pass a path to that CA bundle, or ``False`` to disable verification
            (only if you accept the risk). Default ``True`` (secure).
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        auth_key: str,
        *,
        scope: str = GIGACHAT_DEFAULT_SCOPE,
        default_model: str = GIGACHAT_DEFAULT_MODEL,
        oauth_url: str = GIGACHAT_OAUTH_URL,
        base_url: str = GIGACHAT_BASE_URL,
        verify_ssl: bool | str = True,
        timeout: float = 30.0,
    ) -> None:
        self._auth_key = auth_key
        self._scope = scope
        self._default_model = default_model
        self._oauth_url = oauth_url
        self._base_url = base_url.rstrip("/")
        self._verify_ssl = verify_ssl
        self._timeout = timeout
        self._access_token: str | None = None
        self._expires_at: float = 0.0  # epoch seconds; 0 forces a fetch on first call

    @property
    def name(self) -> str:
        return "gigachat"

    @property
    def default_model(self) -> str:
        return self._default_model

    def _token_valid(self) -> bool:
        return (
            self._access_token is not None
            and time.time() < self._expires_at - _TOKEN_REFRESH_MARGIN_S
        )

    def _fetch_token(self, timeout: float) -> None:
        """Exchange the authorization key for a short-lived access token."""
        response = httpx.post(
            self._oauth_url,
            headers={
                "Authorization": f"Basic {self._auth_key}",
                "RqUID": str(uuid.uuid4()),
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            data={"scope": self._scope},
            timeout=timeout,
            verify=self._verify_ssl,
        )
        response.raise_for_status()
        payload = response.json()
        self._access_token = str(payload["access_token"])
        # GigaChat returns ``expires_at`` as an epoch timestamp in milliseconds.
        self._expires_at = float(payload["expires_at"]) / 1000.0

    def call(
        self,
        system_prompt: str,
        user_content: str,
        model: str,
        *,
        timeout: float | None = None,
    ) -> str:
        """Send a chat completion request, refreshing the OAuth token when needed."""
        request_timeout = timeout if timeout is not None else self._timeout
        if not self._token_valid():
            self._fetch_token(request_timeout)
        response = httpx.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.0,
            },
            timeout=request_timeout,
            verify=self._verify_ssl,
        )
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"])
