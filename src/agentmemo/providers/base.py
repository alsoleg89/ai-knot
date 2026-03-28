"""LLM provider protocol and shared retry logic."""

from __future__ import annotations

import logging
import time
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)

# HTTP status codes that warrant a retry with backoff.
RETRY_STATUS_CODES = frozenset({429, 500, 502, 503})


class LLMProvider(Protocol):
    """Interface that all LLM providers must satisfy.

    A provider knows how to send a system prompt + user content to an LLM
    and return the raw text response. It does NOT handle retry logic —
    that is handled by :func:`call_with_retry`.
    """

    @property
    def name(self) -> str:
        """Human-readable provider name (e.g. "openai", "anthropic")."""
        ...

    @property
    def default_model(self) -> str:
        """Default model ID for this provider."""
        ...

    def call(self, system_prompt: str, user_content: str, model: str) -> str:
        """Send a prompt to the LLM and return the text response.

        Args:
            system_prompt: System-level instruction.
            user_content: User message content.
            model: Model identifier to use.

        Returns:
            Raw text response from the LLM.

        Raises:
            httpx.TimeoutException: On request timeout.
            httpx.HTTPStatusError: On non-2xx response.
            httpx.RequestError: On network-level failure.
        """
        ...


def call_with_retry(
    provider: LLMProvider,
    system_prompt: str,
    user_content: str,
    model: str,
    *,
    max_retries: int = 3,
) -> str:
    """Call an LLM provider with retry on transient errors.

    Retries on 429/500/502/503 with exponential backoff (2^attempt seconds).
    Returns ``""`` if all attempts fail.

    Args:
        provider: LLM provider to call.
        system_prompt: System-level instruction.
        user_content: User message content.
        model: Model identifier.
        max_retries: Maximum number of attempts.

    Returns:
        LLM response text, or ``""`` on failure.
    """
    for attempt in range(max_retries):
        try:
            return provider.call(system_prompt, user_content, model)
        except httpx.TimeoutException:
            logger.warning(
                "%s request timed out (attempt %d/%d)",
                provider.name,
                attempt + 1,
                max_retries,
            )
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in RETRY_STATUS_CODES:
                logger.warning(
                    "HTTP %d from %s, will retry (attempt %d/%d)",
                    status,
                    provider.name,
                    attempt + 1,
                    max_retries,
                )
            else:
                logger.warning("HTTP %d from %s — aborting", status, provider.name)
                return ""
        except httpx.RequestError as exc:
            logger.warning("%s network error: %s", provider.name, exc)
        except (KeyError, ValueError) as exc:
            logger.warning("Unexpected %s response format: %s", provider.name, exc)
            return ""

        if attempt < max_retries - 1:
            wait = 2**attempt
            logger.info("Retrying in %ds…", wait)
            time.sleep(wait)

    logger.warning("%s failed after %d attempts", provider.name, max_retries)
    return ""
