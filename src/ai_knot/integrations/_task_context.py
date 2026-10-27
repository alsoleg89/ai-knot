"""Bounded, stateless task hints for integration recall queries.

Only caller-supplied user messages contribute task text. Context changes retrieval,
never instructions, authorization or stored facts.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

MAX_QUERY_CHARS = 4096
MAX_USER_TURNS = 8
_MAX_CONTENT_NODES = 32
_MAX_CONTENT_DEPTH = 4

# ponytail: conservative EN/RU anaphora heuristic; other wording needs a complete
# query through an explicit-query adapter. Standalone tasks do not inherit topics.
_FOLLOWUP = re.compile(
    r"(?:(?:(?:yes|ok|okay|sure|да|хорошо|ладно)[,\s]+)?"
    r"(?:please\s+)?(?:continue|proceed|go ahead|do it|next|tell me more|"
    r"what next|what's next|explain that|продолжай(?:те)?|дальше|"
    r"расскажи(?:те)? подробнее|подробнее|что дальше)(?:\s+please)?[.!?…]*|"
    r"(?:what about|how about|and|а|а как насч[её]т)\s+.+\?)",
    re.IGNORECASE,
)


def content_text(content: object) -> str:
    """Read a small text prefix from SDK content, ignoring malformed/cyclic nodes."""
    pending = [(content, 0)]
    parts: list[str] = []
    remaining = MAX_QUERY_CHARS
    visited = 0
    while pending and remaining > 0 and visited < _MAX_CONTENT_NODES:
        node, depth = pending.pop()
        visited += 1
        if isinstance(node, str):
            text = node[:remaining].strip()
            if text:
                parts.append(text)
                remaining -= len(text) + 1
        elif depth < _MAX_CONTENT_DEPTH:
            if isinstance(node, (list, tuple)):
                pending.extend((part, depth + 1) for part in reversed(node[:_MAX_CONTENT_NODES]))
            else:
                for key in ("text", "value", "content"):
                    value = node.get(key) if isinstance(node, dict) else getattr(node, key, None)
                    if value is not None:
                        pending.append((value, depth + 1))
                        break
    return " ".join(parts)[:MAX_QUERY_CHARS]


def user_texts(messages: Sequence[object], *, limit: int = MAX_USER_TURNS) -> list[str]:
    """Read at most eight user payloads; intervening tools do not erase the task.

    Scanning role metadata is linear in the supplied history; assistant/tool
    payloads are never parsed. A host should trim history at session boundaries.
    """
    texts = []
    inspected = 0
    for message in reversed(messages):
        if isinstance(message, dict):
            if message.get("type", "message") != "message":
                continue
            role = message.get("role") or message.get("source")
        else:
            role = getattr(message, "role", None) or getattr(message, "source", None)
            if role is None and type(message).__name__ == "UserMessage":
                role = "user"
        role = getattr(role, "value", role)
        if not isinstance(role, str) or role.lower() != "user":
            continue
        content = (
            message.get("content")
            if isinstance(message, dict)
            else getattr(message, "content", None)
        )
        # Empty user turns are task boundaries, not permission to reuse older text.
        texts.append(content_text(content))
        inspected += 1
        if inspected >= max(1, min(limit, MAX_USER_TURNS)):
            break
    return list(reversed(texts))


def task_query(texts: Sequence[str], *, task_context: str | None = None) -> str:
    """Keep a standalone latest task, or pair an anaphoric turn with its task.

    Explicit host context wins over inferred history for follow-ups. No current
    user text means no query, even when the host supplies context.
    """
    if not texts:
        return ""
    latest = texts[-1][:MAX_QUERY_CHARS].strip()
    if len(latest) > 256 or not _FOLLOWUP.fullmatch(latest):
        return latest
    context = task_context[:MAX_QUERY_CHARS].strip() if task_context else ""
    if not context:
        for previous in reversed(texts[:-1]):
            if not _FOLLOWUP.fullmatch(previous):
                context = previous
                break
    if not context:
        return latest
    return f"{context[: MAX_QUERY_CHARS - len(latest) - 1]}\n{latest}"
