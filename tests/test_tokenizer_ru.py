"""Tests for Cyrillic (Russian) stemming in ai_knot.tokenizer."""

from __future__ import annotations

from ai_knot.tokenizer import tokenize


class TestCyrillicStemmer:
    """Russian morphological forms converge to common stems."""

    # -- Noun forms ----------------------------------------------------------

    def test_noun_singular_plural(self) -> None:
        """'слово' (sg) and 'слова' (pl) stem identically."""
        assert tokenize("слово") == tokenize("слова")

    def test_noun_genitive_plural(self) -> None:
        """'слов' (gen.pl) matches 'слово' (nom.sg)."""
        assert tokenize("слов") == tokenize("слово")

    def test_noun_instrumental_plural(self) -> None:
        """'словами' (instr.pl) matches 'слово' (nom.sg)."""
        assert tokenize("словами") == tokenize("слово")

    def test_noun_dative_plural(self) -> None:
        """'клиентам' (dat.pl) matches 'клиент' (nom.sg)."""
        assert tokenize("клиентам") == tokenize("клиент")

    def test_noun_genitive_singular(self) -> None:
        """'клиента' (gen.sg) matches 'клиент' (nom.sg)."""
        assert tokenize("клиента") == tokenize("клиент")

    # -- Adjective forms ------------------------------------------------------

    def test_adj_nominative_forms(self) -> None:
        """'запрещённые' and 'запрещённых' converge."""
        assert tokenize("запрещённые") == tokenize("запрещённых")

    def test_adj_masculine_feminine(self) -> None:
        """'новый' and 'новая' converge."""
        [m] = tokenize("новый")
        [f] = tokenize("новая")
        assert m == f

    def test_adj_genitive(self) -> None:
        """'нового' matches 'новый'."""
        [gen] = tokenize("нового")
        [nom] = tokenize("новый")
        assert gen == nom

    # -- Verb forms -----------------------------------------------------------

    def test_verb_infinitive_vs_present(self) -> None:
        """'запрещать' (inf) and 'запрещает' (3sg) converge."""
        assert tokenize("запрещать") == tokenize("запрещает")

    def test_verb_past_tense(self) -> None:
        """'работала' (past.f) stems same as 'работать' (inf)."""
        [past] = tokenize("работала")
        [inf] = tokenize("работать")
        assert past == inf

    # -- ё/е normalization ----------------------------------------------------

    def test_yo_to_ye_normalization(self) -> None:
        """'запрещённые' with ё stems same as 'запрещенные' with е."""
        assert tokenize("запрещённые") == tokenize("запрещенные")

    # -- Short words (no stemming) -------------------------------------------

    def test_short_words_preserved(self) -> None:
        """Words ≤3 chars are not stemmed."""
        assert tokenize("на") == ["на"]
        assert tokenize("для") == ["для"]

    # -- Mixed text -----------------------------------------------------------

    def test_mixed_russian_english(self) -> None:
        """Mixed text: Russian words use Cyrillic stemmer, English use English."""
        tokens = tokenize("Python запрещённые deployment")
        assert "python" in tokens
        assert "deploy" in tokens
        # Russian token should be stemmed (not raw "запрещённые")
        assert "запрещённые" not in tokens

    # -- English not affected -------------------------------------------------

    def test_english_stemming_unchanged(self) -> None:
        """English rules still work correctly (single-pass suffix removal)."""
        assert tokenize("deployment") == ["deploy"]
        assert tokenize("caching") == ["cach"]
        assert tokenize("queries") == ["query"]
        assert tokenize("running") == ["run"]

    # -- Reflexive verbs ------------------------------------------------------

    def test_reflexive_suffix_removed(self) -> None:
        """'вернулся' (reflexive past) stems same as 'вернул' (past)."""
        [refl] = tokenize("вернулся")
        [base] = tokenize("вернул")
        assert refl == base

    # -- Derivational suffixes -----------------------------------------------

    def test_derivational_ost(self) -> None:
        """'сложность' (-ость) is reduced."""
        [stem] = tokenize("сложность")
        assert stem != "сложность"  # must be stemmed
        assert stem.startswith("сложн")

    # -- Integration: full sentence ------------------------------------------

    def test_russian_sentence_tokenization(self) -> None:
        """Full Russian sentence tokenizes and stems all words."""
        tokens = tokenize("Запрещённые слова клиента")
        assert len(tokens) == 3
        # All tokens should be lowercase and stemmed
        for t in tokens:
            assert t == t.lower()
            assert t.isalpha()
