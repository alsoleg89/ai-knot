"""Secondary preference/affect retrieval for speculative cat3 questions.

Uses a fixed English affect lexicon to pull utterances where the focus entity
expressed preferences, feelings, plans, or opinions. Deterministic — zero LLM calls.
"""

from __future__ import annotations

from typing import Any

AFFECT_LEXICON: frozenset[str] = frozenset(
    {
        "like",
        "likes",
        "liked",
        "love",
        "loves",
        "loved",
        "hate",
        "hates",
        "hated",
        "dislike",
        "dislikes",
        "disliked",
        "want",
        "wants",
        "wanted",
        "prefer",
        "prefers",
        "preferred",
        "dream",
        "dreams",
        "dreamt",
        "think",
        "thinks",
        "thought",
        "believe",
        "believes",
        "believed",
        "hope",
        "hopes",
        "hoped",
        "fear",
        "fears",
        "feared",
        "afraid",
        "enjoy",
        "enjoys",
        "enjoyed",
        "worry",
        "worries",
        "worried",
        "miss",
        "misses",
        "missed",
        "care",
        "cares",
        "cared",
        "feel",
        "feels",
        "felt",
        "wish",
        "wishes",
        "wished",
    }
)


def retrieve_preference_episodes(
    storage: Any,
    agent_id: str,
    entities: tuple[str, ...],
    *,
    top_k: int = 20,
) -> list[Any]:
    """BM25 search using entity + affect lexicon to surface preference utterances.

    Runs the existing search_episodes_by_entities with an affect-seeded query.
    Returns up to top_k episodes. Empty list if storage does not support search.
    """
    if not entities:
        return []
    search_fn = getattr(storage, "search_episodes_by_entities", None)
    if search_fn is None:
        return []
    affect_query = " ".join(sorted(AFFECT_LEXICON))
    try:
        result: list[Any] = search_fn(
            agent_id, entities, query=affect_query, top_k=top_k, diversity=False
        )
        return result
    except TypeError:
        # Fallback: without diversity param (older storage versions)
        fallback: list[Any] = search_fn(agent_id, entities, query=affect_query, top_k=top_k)
        return fallback
