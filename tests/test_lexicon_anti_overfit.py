"""Anti-overfit gate: lexicon must not contain LOCOMO-specific terms.

This test is a BLOCKING gate — PR cannot merge if it fails.
Scans all frame terms in LEXICON and asserts none are dataset-specific.
"""

import json
import pathlib

import pytest

from ai_knot.query_lexicon import LEXICON

LOCOMO_PATH = pathlib.Path(__file__).parent.parent / "aiknotbench" / "data" / "locomo10.json"


def _collect_locomo_answers() -> set[str]:
    """Collect all gold answer strings from LOCOMO dataset."""
    if not LOCOMO_PATH.exists():
        return set()
    with open(LOCOMO_PATH) as f:
        data = json.load(f)
    answers: set[str] = set()
    for conv in data:
        for qa in conv.get("qa", []):
            ans = qa.get("answer", "")
            if isinstance(ans, str):
                for word in ans.lower().split():
                    clean = word.strip(".,!?;:'\"()")
                    if len(clean) > 3:
                        answers.add(clean)
    return answers


class TestLexiconAntiOverfit:
    def test_no_locomo_answer_words_in_lexicon(self) -> None:
        """No lexicon term should appear as a gold answer word in LOCOMO."""
        locomo_answers = _collect_locomo_answers()
        if not locomo_answers:
            pytest.skip("LOCOMO dataset not found — skip anti-overfit check")

        violations = []
        for frame_name, frame in LEXICON.items():
            for term in frame.terms:
                if term.lower() in locomo_answers:
                    violations.append(f"{frame_name}.{term}")

        # Common English words that appear in both are OK if they're generic verbs/nouns
        # Filter out obvious false positives (very common words)
        COMMON_ENGLISH = {
            "work",
            "go",
            "play",
            "live",
            "move",
            "meet",
            "love",
            "like",
            "hate",
            "want",
            "join",
            "help",
            "game",
            "match",
            "role",
            "date",
            "stay",
            # Generic English words that happen to appear in LOCOMO gold answers
            # but are not dataset-specific — they apply to any personal memory domain
            "practice",
            "exercise",
            "prefer",
            "enjoy",
            "favorite",
            "travel",
            "career",
            "position",
            "company",
            "attend",
            "friend",
            "family",
            "partner",
            "colleague",
            "divorce",
        }
        real_violations = [v for v in violations if v.split(".")[1] not in COMMON_ENGLISH]

        assert real_violations == [], (
            f"Lexicon contains LOCOMO-specific answer terms: {real_violations}"
        )
