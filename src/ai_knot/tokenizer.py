"""Shared tokenization utilities for retrieval and extraction."""

from __future__ import annotations

import re

# Pre-compiled patterns.
_CAMEL_RE = re.compile(r"([a-z])([A-Z])")
_TOKEN_RE = re.compile(r"[^\W_]+")

# Russian vowels (used by Snowball-lite stemmer for region detection).
_RU_VOWELS = frozenset("аеёиоуыэюя")


def _is_cyrillic(token: str) -> bool:
    """Check if token contains Cyrillic characters."""
    return any("\u0400" <= ch <= "\u04ff" for ch in token)


def _stem_ru(token: str) -> str:
    """Lightweight Russian suffix stemmer (Snowball-lite, zero-dep).

    Implements a simplified version of the Snowball Russian stemming
    algorithm (Krovetz-style suffix stripping).  Covers the most
    frequent inflectional suffixes to normalise morphological variants
    to a common stem.  Rules are ordered longest-suffix-first within
    each group to avoid partial matches.
    """
    if len(token) <= 3:
        return token

    # Normalize ё → е for consistent matching.
    token = token.replace("ё", "е")

    # Find R1 region (after first vowel-consonant pair).
    # Snowball Russian stems only within R1 to avoid over-stemming.
    r1 = len(token)
    for i in range(1, len(token)):
        if token[i - 1] in _RU_VOWELS and token[i] not in _RU_VOWELS:
            r1 = i + 1
            break

    def _try_remove(suffixes: tuple[str, ...]) -> str | None:
        """Try to remove a suffix that falls within R1."""
        for suf in suffixes:
            if token.endswith(suf) and len(token) - len(suf) >= r1:
                return token[: -len(suf)]
        return None

    # Step 1: Perfective gerund (-вшись, -вши, -ав, -ив, etc.)
    result = _try_remove(("ившись", "ывшись", "вшись", "авши", "ивши", "ывши", "вши", "ав", "ив"))
    if result is not None:
        return result if len(result) > 2 else token

    # Step 2: Reflexive (-ся, -сь)
    working = token
    for suf in ("ся", "сь"):
        if working.endswith(suf) and len(working) - len(suf) >= r1:
            working = working[: -len(suf)]
            break

    # Step 3: Adjectival endings (adjective + optional participle suffix).
    _adj_suffixes = (
        "ими",
        "ыми",
        "его",
        "ого",
        "ему",
        "ому",
        "ее",
        "ие",
        "ые",
        "ое",
        "ей",
        "ий",
        "ый",
        "ой",
        "ем",
        "им",
        "ым",
        "ом",
        "их",
        "ых",
        "ую",
        "юю",
        "ая",
        "яя",
        "ею",
        "ию",
    )
    _part_suffixes = (
        "ивш",
        "ывш",
        "ующ",
        "ем",
        "нн",
        "вш",
        "ющ",
        "щ",
    )

    adj_removed = False
    for adj in _adj_suffixes:
        if working.endswith(adj) and len(working) - len(adj) >= r1:
            candidate = working[: -len(adj)]
            # Try participle suffix before adjective ending.
            for part in _part_suffixes:
                if candidate.endswith(part) and len(candidate) - len(part) >= r1:
                    candidate = candidate[: -len(part)]
                    break
            working = candidate
            adj_removed = True
            break

    if not adj_removed:
        # Step 4: Verb endings.
        _verb_suffixes = (
            "ейте",
            "уйте",
            "ите",
            "ать",
            "ять",
            "ить",
            "ует",
            "уют",
            "ает",
            "ают",
            "ешь",
            "ишь",
            "ете",
            "яет",
            "ала",
            "яла",
            "ила",
            "али",
            "яли",
            "или",
            "ало",
            "яло",
            "ило",
            "ана",
            "ано",
            "ей",
            "уй",
            "ал",
            "ял",
            "ил",
            "ет",
            "ит",
            "ат",
            "ят",
            "ен",
            "на",
            "ла",
            "ли",
            "ло",
            "ть",
            "ут",
            "ют",
            "ны",
            "ну",
        )
        verb_removed = False
        for suf in _verb_suffixes:
            if working.endswith(suf) and len(working) - len(suf) >= r1:
                working = working[: -len(suf)]
                verb_removed = True
                break

        if not verb_removed:
            # Step 5: Noun endings.
            _noun_suffixes = (
                "ениям",
                "ениях",
                "ений",
                "ения",
                "ение",
                "ями",
                "ами",
                "ией",
                "иям",
                "иях",
                "ов",
                "ев",
                "ей",
                "ий",
                "ия",
                "ие",
                "ью",
                "ом",
                "ем",
                "ах",
                "ям",
                "ию",
                "ии",
                "ые",
                "ых",
                "ой",
                "ам",
                "а",
                "е",
                "и",
                "о",
                "у",
                "ы",
                "ь",
                "я",
                "ю",
            )
            for suf in _noun_suffixes:
                if working.endswith(suf) and len(working) - len(suf) >= r1:
                    working = working[: -len(suf)]
                    break

    # Step 6: Superlative (-ейш, -ейше).
    for suf in ("ейше", "ейш"):
        if working.endswith(suf) and len(working) - len(suf) >= r1:
            working = working[: -len(suf)]
            break

    # Step 7: Derivational (-ость, -ост).
    for suf in ("ость", "ост"):
        if working.endswith(suf) and len(working) - len(suf) >= r1:
            working = working[: -len(suf)]
            break

    # Step 8: Clean up trailing double consonant (нн → н).
    if len(working) >= 2 and working[-1] == working[-2] and working[-1] == "н":
        working = working[:-1]

    return working if len(working) > 2 else token


def _stem_en(token: str) -> str:
    """Lightweight English suffix stemmer (Porter 1980 subset)."""
    # -ment → remove (deployment → deploy)
    if token.endswith("ment") and len(token) > 6:
        return token[:-4]

    # -tion / -sion → remove (creation → crea, but that's ok for matching)
    if (token.endswith("tion") or token.endswith("sion")) and len(token) > 6:
        return token[:-4]

    # -ness → remove (darkness → dark)
    if token.endswith("ness") and len(token) > 6:
        return token[:-4]

    # -ity → remove (complexity → complex)
    if token.endswith("ity") and len(token) > 6:
        return token[:-3]

    # -ive → remove (adaptive → adapt)
    if token.endswith("ive") and len(token) > 5:
        return token[:-3]

    # -ence / -ance → remove (preference → prefer, performance → perform)
    if (token.endswith("ence") or token.endswith("ance")) and len(token) > 6:
        return token[:-4]

    # -al → remove (functional → function)  [before -ing/-ed to avoid clash]
    if token.endswith("al") and len(token) > 5:
        return token[:-2]

    # -ing → remove (caching → cach, running → runn → run via double-consonant)
    if token.endswith("ing") and len(token) > 5:
        stem = token[:-3]
        # Handle doubled consonant: runn → run
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            stem = stem[:-1]
        return stem

    # -ed → remove (deployed → deploy, walked → walk)
    if token.endswith("ed") and len(token) > 4:
        stem = token[:-2]
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            stem = stem[:-1]
        return stem

    # -ly → remove (quickly → quick)
    if token.endswith("ly") and len(token) > 4:
        return token[:-2]

    # -er → remove (faster → fast, formatter → formatt → format)
    if token.endswith("er") and len(token) > 4:
        stem = token[:-2]
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            stem = stem[:-1]
        return stem

    # -est → remove (fastest → fast)
    if token.endswith("est") and len(token) > 5:
        return token[:-3]

    # -ies → -y (queries → query, strategies → strategy)
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"

    # -es → remove 2 chars for -ches, -shes, -sses, -xes, -zes
    # (caches → cach, matches → match, fixes → fix)
    if token.endswith("es") and len(token) > 4:
        before = token[-3]
        if before in "chsxz" or token.endswith("sses"):
            return token[:-2]

    # -s → remove (original rule, plural stripping)
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]

    # Terminal -e: strip so base forms converge with inflected stems
    # (cache → cach like caching/cached/caches; service → servic)
    if token.endswith("e") and len(token) > 4:
        return token[:-1]

    return token


def _stem(token: str) -> str:
    """Dispatch stemming by detected script."""
    if len(token) <= 3:
        return token
    if _is_cyrillic(token):
        return _stem_ru(token)
    return _stem_en(token)


def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens with stemming.

    Splits camelCase (``FastAPI`` → ``["fast", "api"]``), applies lightweight
    suffix stemming (Porter 1980 subset), and works with any Unicode script.

    Args:
        text: Input text to tokenize.

    Returns:
        List of normalized, stemmed tokens.
    """
    text = _CAMEL_RE.sub(r"\1 \2", text)
    tokens = _TOKEN_RE.findall(text.lower())
    return [_stem(t) for t in tokens]
