"""V2 LOCOMO answer-accuracy runner — bridges v2 retrieval with LLM answerer + judge.

Mirror of aiknotbench TypeScript runner but uses v2 MemoryAPI + render_pack_eswp.
Produces cat1-4 % directly comparable to v1 aiknotbench results.

Allowed in bench/ per CLAUDE.md §1 (LLM forbidden only in core/ ops/ store/ api/).

Usage:
    OPENAI_API_KEY=... .venv/bin/python -m ai_knot_v2.bench.v2_locomo_answer_runner \\
        --convs 2 --data /path/to/locomo10.json --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from ai_knot_v2.api.product import MemoryAPI
from ai_knot_v2.api.sdk import EpisodeIn, LearnRequest, RecallRequest
from ai_knot_v2.bench._prompts import ANSWER_SYSTEM, JUDGE_SYSTEM, verify_prompt_parity
from ai_knot_v2.bench.ccb.render import render_pack_eswp
from ai_knot_v2.bench.cwp.lineage_render import render_pack_cwp, render_pack_raw_annotated
from ai_knot_v2.bench.cwp.persistence import compute_pct_signatures
from ai_knot_v2.bench.v2_locomo_runner import LocomoConvData, parse_locomo_json
from ai_knot_v2.core.episode import RawEpisode

_DEFAULT_DATA = Path("/Users/alsoleg/Documents/github/ai-knot/aiknotbench/data/locomo10.json")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize(s: object) -> str:
    return _CONTROL_RE.sub("", str(s))


# ---------------------------------------------------------------------------
# Raw-episode BM25 retrieval (mirror of v1 BM25-on-turns approach)
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_BM25_K1 = 1.5
_BM25_B = 0.75


def _tokenize(s: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(s) if len(t) > 1]


def _build_bm25_index(episodes: list[RawEpisode]) -> dict[str, Any]:
    """Build BM25 index over episode texts. Returns dict with tf, doc_len, avgdl, idf."""
    docs = [(ep.episode_id, _tokenize(ep.text)) for ep in episodes]
    n = max(1, len(docs))
    df: Counter[str] = Counter()
    doc_len: dict[str, int] = {}
    for ep_id, toks in docs:
        doc_len[ep_id] = len(toks)
        for term in set(toks):
            df[term] += 1
    avgdl = sum(doc_len.values()) / max(1, n)
    idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
    return {
        "docs": {ep_id: toks for ep_id, toks in docs},
        "tf": {ep_id: Counter(toks) for ep_id, toks in docs},
        "doc_len": doc_len,
        "avgdl": avgdl,
        "idf": idf,
    }


def _bm25_search(index: dict[str, Any], query: str, top_k: int = 12) -> list[str]:
    q_terms = _tokenize(query)
    if not q_terms:
        return []
    scores: dict[str, float] = {}
    for ep_id, tf in index["tf"].items():
        dl = index["doc_len"][ep_id]
        denom_const = _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / max(1.0, index["avgdl"]))
        s = 0.0
        for t in q_terms:
            if t not in tf:
                continue
            idf = index["idf"].get(t, 0.0)
            f = tf[t]
            s += idf * (f * (_BM25_K1 + 1)) / (f + denom_const)
        if s > 0:
            scores[ep_id] = s
    return [
        ep_id for ep_id, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    ]


# ---------------------------------------------------------------------------
# Intent routing (lexicon-driven; was hardcoded EN — now per-language file)
# ---------------------------------------------------------------------------


_AGGREGATION_TOKENS = frozenset(
    {
        "list",
        "all",
        "every",
        "various",
        "different",
        "describe",
        "enumerate",
        "overview",
        "summary",
        "summariz",
        "summar",
    }
)
_AGGREGATION_PHRASES = (
    "how many",
    "tell me about",
    "what are",
    "what does",
    "what did",
    "what has",
    "what have",
    "what do",
    "what were",
    "know about",
)


def _is_aggregate_query(query: str) -> bool:
    """Detect 'aggregation' intent (need ALL mentions, not single best fact)."""
    q_lower = query.lower()
    if any(p in q_lower for p in _AGGREGATION_PHRASES):
        return True
    toks = set(_tokenize(query))
    return bool(toks & _AGGREGATION_TOKENS)


# ---------------------------------------------------------------------------
# M2-lite: Mention Graph (entity expansion)
# ---------------------------------------------------------------------------

_ENTITY_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_STOPWORDS_CAP = frozenset(
    {
        "What",
        "When",
        "Where",
        "Why",
        "Who",
        "How",
        "Which",
        "Did",
        "Does",
        "Do",
        "Was",
        "Were",
        "Is",
        "Are",
        "Will",
        "Would",
        "Could",
        "Should",
        "Has",
        "Have",
        "Had",
        "The",
        "And",
        "But",
        "Or",
        "Not",
        "Yes",
        "No",
        "Can",
        "May",
        "Might",
        "Must",
        "Shall",
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "January",
        "February",
        "March",
        "April",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    }
)


def _load_common_english_words() -> frozenset[str]:
    """Load /usr/share/dict/words as a lowercase frozenset (cached)."""
    p = Path("/usr/share/dict/words")
    if not p.exists():
        return frozenset()
    try:
        return frozenset(w.strip().lower() for w in p.read_text().splitlines() if w.strip())
    except OSError:
        return frozenset()


_COMMON_WORDS: frozenset[str] = _load_common_english_words()


def _extract_entities(text: str) -> list[str]:
    """Extract capitalized entity-like tokens (proper-noun heuristic).

    Filters out:
      - Sentence-initial common English words (via /usr/share/dict/words).
      - Question/auxiliary stopwords (_STOPWORDS_CAP).
    Keeps:
      - Acronyms (all-caps), known proper nouns, names not in common-vocab.
    """
    found = []
    for m in _ENTITY_RE.finditer(text):
        tok = m.group(0)
        if tok in _STOPWORDS_CAP:
            continue
        # Drop common English words that happen to be capitalized (sentence-start, etc.)
        if _COMMON_WORDS and tok.lower() in _COMMON_WORDS:
            continue
        found.append(tok)
    # dedupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for t in found:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _build_mention_index(
    episodes: list[RawEpisode],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Build (entity → ep_ids) and (speaker → ep_ids) indices.

    Episodes are listed in order of appearance for cheap "first/recent" picking later.
    """
    ent_idx: dict[str, list[str]] = defaultdict(list)
    speaker_idx: dict[str, list[str]] = defaultdict(list)
    for ep in episodes:
        for ent in _extract_entities(ep.text):
            ent_idx[ent].append(ep.episode_id)
        speaker_label = ep.metadata.get("speaker_name") or ep.user_id or ep.speaker
        if speaker_label:
            speaker_idx[str(speaker_label)].append(ep.episode_id)
    return ent_idx, speaker_idx


def _expand_with_mentions(
    bm25_eps: list[str],
    query: str,
    ent_idx: dict[str, list[str]],
    speaker_idx: dict[str, list[str]],
    bm25_index: dict[str, Any],
    per_entity_cap: int = 6,
) -> list[str]:
    """Augment BM25 results with episodes mentioning query entities or by speaker.

    Strategy:
      1. Extract entities from query.
      2. For each entity: pull explicit-mention episodes (top by BM25 score within query).
      3. Also pull speaker episodes when entity matches a known speaker.
      4. Append to BM25 list (BM25-first ordering preserved; entities widen recall).
    """
    entities = _extract_entities(query)
    if not entities:
        return bm25_eps

    seen: set[str] = set(bm25_eps)
    augmented: list[str] = list(bm25_eps)

    q_terms = _tokenize(query)

    def _score(ep_id: str) -> float:
        tf = bm25_index["tf"].get(ep_id, {})
        dl = bm25_index["doc_len"].get(ep_id, 0)
        denom_const = _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / max(1.0, bm25_index["avgdl"]))
        s = 0.0
        for t in q_terms:
            f = tf.get(t, 0)
            if f == 0:
                continue
            idf = bm25_index["idf"].get(t, 0.0)
            s += idf * (f * (_BM25_K1 + 1)) / (f + denom_const)
        return s

    for ent in entities:
        candidates: set[str] = set(ent_idx.get(ent, [])) | set(speaker_idx.get(ent, []))
        ranked = sorted(
            (e for e in candidates if e not in seen),
            key=_score,
            reverse=True,
        )[:per_entity_cap]
        for e in ranked:
            if e not in seen:
                augmented.append(e)
                seen.add(e)

    return augmented


# ---------------------------------------------------------------------------
# M3-lite: Evidence Ribbons (parent-context ±1 turn within session)
# ---------------------------------------------------------------------------


def _build_session_order(episodes: list[RawEpisode]) -> dict[str, list[str]]:
    """Map session_id → ordered list of ep_ids (by timestamp)."""
    by_session: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for ep in episodes:
        by_session[ep.session_id].append((ep.timestamp, ep.episode_id))
    out: dict[str, list[str]] = {}
    for sid, items in by_session.items():
        items.sort()
        out[sid] = [e for _, e in items]
    return out


def _expand_with_ribbons(
    ep_ids: list[str],
    episodes_lookup: dict[str, RawEpisode],
    session_order: dict[str, list[str]],
    radius: int = 1,
) -> list[str]:
    """For each ep_id, also include ±radius neighbors within same session.

    Preserves original ordering for primary hits; appends new neighbors after.
    """
    seen: set[str] = set(ep_ids)
    out: list[str] = list(ep_ids)
    for ep_id in ep_ids:
        ep = episodes_lookup.get(ep_id)
        if ep is None:
            continue
        sess = session_order.get(ep.session_id, [])
        if ep_id not in sess:
            continue
        idx = sess.index(ep_id)
        for off in range(1, radius + 1):
            for j in (idx - off, idx + off):
                if 0 <= j < len(sess):
                    nb = sess[j]
                    if nb not in seen:
                        out.append(nb)
                        seen.add(nb)
    return out


def _render_episodes_v1_style(
    ep_ids: list[str],
    episodes_lookup: dict[str, RawEpisode],
    char_budget: int = 22_000,
    per_turn_max: int = 1200,
) -> str:
    parts: list[str] = []
    total_chars = 0
    for i, ep_id in enumerate(ep_ids, start=1):
        ep = episodes_lookup.get(ep_id)
        if ep is None:
            continue
        try:
            d = datetime.fromtimestamp(ep.timestamp, tz=UTC).date().isoformat()
            date_part = f"[{d}] "
        except (OSError, OverflowError, ValueError):
            date_part = ""
        speaker_label = ep.metadata.get("speaker_name") or ep.user_id or ep.speaker
        raw = ep.text[:per_turn_max]
        line = f"[{i}] {date_part}{speaker_label}: {raw}"
        if total_chars + len(line) > char_budget:
            break
        parts.append(line)
        total_chars += len(line) + 1
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# OpenAI calls (with simple rate-limit retry)
# ---------------------------------------------------------------------------


def _complete_with_retry(
    client: OpenAI,
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    max_attempts: int = 6,
) -> str:
    delay = 5.0
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            msg = str(e)
            is_rate = "rate" in msg.lower() or "429" in msg
            if not is_rate or attempt == max_attempts:
                raise
            wait = delay
            print(
                f"  [rate limit] waiting {wait:.1f}s (attempt {attempt}/{max_attempts})",
                file=sys.stderr,
            )
            time.sleep(wait)
            delay = min(delay * 2, 60.0)
    raise RuntimeError("unreachable")


def _answer(client: OpenAI, model: str, context: str, question: str) -> str:
    return _complete_with_retry(
        client,
        model=model,
        system=ANSWER_SYSTEM,
        user=f"Context:\n{_sanitize(context)}\n\nQuestion: {_sanitize(question)}",
        max_tokens=256,
    )


def _parse_verdict(raw: str) -> str:
    """Parse CORRECT/WRONG verdict from raw LLM output (mirrors evaluator.ts)."""
    trimmed = raw.strip()
    try:
        parsed = json.loads(trimmed)
        v = parsed.get("verdict")
        if v == "CORRECT":
            return "CORRECT"
        if v == "WRONG":
            return "WRONG"
    except Exception:
        pass
    if re.search(r"\bCORRECT\b", trimmed, re.I):
        return "CORRECT"
    if re.search(r"\bWRONG\b", trimmed, re.I):
        return "WRONG"
    return "WRONG"


def _judge(client: OpenAI, model: str, question: str, candidate: str, gold: str) -> str:
    raw = _complete_with_retry(
        client,
        model=model,
        system=JUDGE_SYSTEM,
        user=(
            f"Question: {_sanitize(question)}\n"
            f"Candidate answer: {_sanitize(candidate)}\n"
            f"Gold answer: {_sanitize(gold)}"
        ),
        max_tokens=32,
    )
    return _parse_verdict(raw)


# ---------------------------------------------------------------------------
# Per-question result
# ---------------------------------------------------------------------------


@dataclass
class QaResult:
    conv_idx: int
    qa_idx: int
    category: int
    question: str
    gold: str
    candidate: str
    verdict: str
    rendered_chars: int
    atoms_retrieved: int


@dataclass
class ConvResult:
    conv_idx: int
    total_atoms: int
    qa_results: list[QaResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-conversation run
# ---------------------------------------------------------------------------


def run_conversation(
    conv: LocomoConvData,
    client: OpenAI,
    answer_model: str,
    judge_model: str,
    max_atoms: int = 60,
    max_tokens: int = 8000,
    render_mode: str = "raw",
    retrieval_mode: str = "atoms",
    bm25_top_k: int = 12,
    mention_expand: bool = False,
    ribbon_radius: int = 0,
    final_top_k: int = 18,
) -> ConvResult:
    """Ingest conv, then for each QA: recall → render → answer → judge.

    retrieval_mode: "atoms"     → v2 MemoryAPI.recall, atom-based
                    "raw-bm25"  → BM25 over raw episodes (mirrors v1)
                    "hybrid"    → union of atoms-derived + raw-bm25 episodes
    render_mode:    "raw"       → raw episode turns + dates (v1-style)
                    "atoms"     → sparse triples (S1 default)
    """
    api = MemoryAPI(db_path=":memory:")

    episodes: list[EpisodeIn] = []
    for turn in conv.turns:
        if not turn.text.strip():
            continue
        episodes.append(
            EpisodeIn(
                text=turn.text,
                speaker="user",
                user_id=turn.speaker,
                session_id=f"conv-{conv.conv_idx}-s{turn.session_num}",
                timestamp=turn.timestamp,
            )
        )
    learn_resp = api.learn(LearnRequest(episodes=episodes))

    all_atoms = api._library.all_atoms()  # noqa: SLF001
    total_atoms = len(all_atoms)
    atom_by_id = {a.atom_id: a for a in all_atoms}

    # Build episode lookup (always — used by raw render and BM25)
    episodes_lookup: dict[str, RawEpisode] = {}
    for ep_id in learn_resp.episode_ids:
        ep = api._store.get_episode(ep_id)  # noqa: SLF001
        if ep is not None:
            episodes_lookup[ep_id] = ep

    bm25_index = None
    if retrieval_mode in ("raw-bm25", "hybrid") or mention_expand:
        bm25_index = _build_bm25_index(list(episodes_lookup.values()))

    ent_idx: dict[str, list[str]] = {}
    speaker_idx: dict[str, list[str]] = {}
    if mention_expand:
        ent_idx, speaker_idx = _build_mention_index(list(episodes_lookup.values()))

    session_order: dict[str, list[str]] = {}
    if ribbon_radius > 0:
        session_order = _build_session_order(list(episodes_lookup.values()))

    pct_signatures = None
    if render_mode == "lineage":
        pct_signatures = compute_pct_signatures(all_atoms)

    cv = ConvResult(conv_idx=conv.conv_idx, total_atoms=total_atoms)
    pad_total = len(conv.qa_pairs)

    for qi, qa in enumerate(conv.qa_pairs):
        atoms: list[Any] = []
        ep_ids_for_render: list[str] = []

        # For lineage mode, force atom retrieval so we have claims to render trees for.
        retr_eff = retrieval_mode
        if render_mode == "lineage" and retrieval_mode == "raw-bm25":
            retr_eff = "hybrid"

        if retr_eff in ("atoms", "hybrid"):
            recall_resp = api.recall(
                RecallRequest(query=qa.question, max_atoms=max_atoms, max_tokens=max_tokens)
            )
            atoms = [atom_by_id[a.atom_id] for a in recall_resp.atoms if a.atom_id in atom_by_id]
            atom_eps: list[str] = []
            seen: set[str] = set()
            for atom in sorted(atoms, key=lambda a: a.regret_charge * a.credence, reverse=True):
                for eid in atom.evidence_episodes:
                    if eid not in seen and eid in episodes_lookup:
                        atom_eps.append(eid)
                        seen.add(eid)
            ep_ids_for_render = atom_eps[:bm25_top_k]

        if retr_eff in ("raw-bm25", "hybrid"):
            assert bm25_index is not None
            bm25_eps = _bm25_search(bm25_index, qa.question, top_k=bm25_top_k)
            if retr_eff == "raw-bm25":
                ep_ids_for_render = bm25_eps
            else:  # hybrid: union, BM25 first (higher precision for keyword queries)
                merged: list[str] = []
                seen_h: set[str] = set()
                for eid in bm25_eps + ep_ids_for_render:
                    if eid not in seen_h:
                        merged.append(eid)
                        seen_h.add(eid)
                ep_ids_for_render = merged[:bm25_top_k]

        if mention_expand and bm25_index is not None and _is_aggregate_query(qa.question):
            # Intent gate (port of v1's aggregation routing): only expand for
            # broad/aggregate queries where breadth > precision. Single-fact
            # queries (most cat1) keep BM25-only precision.
            ep_ids_for_render = _expand_with_mentions(
                ep_ids_for_render,
                qa.question,
                ent_idx,
                speaker_idx,
                bm25_index,
            )

        if ribbon_radius > 0 and session_order:
            ep_ids_for_render = _expand_with_ribbons(
                ep_ids_for_render,
                episodes_lookup,
                session_order,
                radius=ribbon_radius,
            )

        ep_ids_for_render = ep_ids_for_render[:final_top_k]

        if render_mode == "lineage":
            if atoms:
                rendered = render_pack_cwp(
                    atoms,
                    episodes_lookup,
                    qa.question,
                    pct_signatures=pct_signatures,
                    max_atoms=final_top_k,
                )
            else:
                rendered = _render_episodes_v1_style(ep_ids_for_render, episodes_lookup)
        elif render_mode == "raw-annotated":
            # Raw episodes primary (v1-style power), atoms as light annotation
            atoms_for_anno = atoms if atoms else list(atom_by_id.values())
            rendered = render_pack_raw_annotated(ep_ids_for_render, episodes_lookup, atoms_for_anno)
        elif render_mode == "raw" or retrieval_mode == "raw-bm25":
            rendered = _render_episodes_v1_style(ep_ids_for_render, episodes_lookup)
        else:
            rendered = render_pack_eswp(atoms, qa.question)

        if not rendered.strip():
            candidate = "No information."
        else:
            candidate = _answer(client, answer_model, rendered, qa.question)

        verdict = _judge(client, judge_model, qa.question, candidate, qa.answer)

        cv.qa_results.append(
            QaResult(
                conv_idx=conv.conv_idx,
                qa_idx=qi,
                category=qa.category,
                question=qa.question,
                gold=qa.answer,
                candidate=candidate,
                verdict=verdict,
                rendered_chars=len(rendered),
                atoms_retrieved=len(atoms),
            )
        )

        icon = "✓" if verdict == "CORRECT" else "✗"
        q_short = qa.question[:55] + ("…" if len(qa.question) > 55 else "")
        print(
            f"  [conv {conv.conv_idx} qa {qi + 1:3d}/{pad_total}] {icon} {verdict} "
            f'(cat {qa.category}) "{q_short}"'
        )

    return cv


# ---------------------------------------------------------------------------
# Aggregation + report
# ---------------------------------------------------------------------------


def aggregate(convs: list[ConvResult]) -> dict[str, Any]:
    by_cat: dict[int, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    for cv in convs:
        for qr in cv.qa_results:
            by_cat[qr.category]["total"] += 1
            if qr.verdict == "CORRECT":
                by_cat[qr.category]["correct"] += 1

    total_14 = 0
    correct_14 = 0
    out_by_cat: dict[int, dict[str, float]] = {}
    for cat, s in sorted(by_cat.items()):
        acc = s["correct"] / s["total"] if s["total"] else 0.0
        out_by_cat[cat] = {"total": s["total"], "correct": s["correct"], "accuracy": acc}
        if cat in (1, 2, 3, 4):
            total_14 += s["total"]
            correct_14 += s["correct"]
    cat14_acc = correct_14 / total_14 if total_14 else 0.0
    return {
        "by_cat": out_by_cat,
        "cat1_4": {"total": total_14, "correct": correct_14, "accuracy": cat14_acc},
    }


def report(convs: list[ConvResult], agg: dict[str, Any]) -> None:
    print("\n" + "─" * 52)
    cat14 = agg["cat1_4"]
    acc = cat14["accuracy"] * 100
    print(f"  cat 1-4 accuracy : {acc:.1f}%  ({cat14['correct']}/{cat14['total']})")
    cat_names = {
        1: "single-hop",
        2: "multi-hop",
        3: "temporal",
        4: "open-domain",
        5: "adversarial",
    }
    for cat in sorted(agg["by_cat"].keys()):
        s = agg["by_cat"][cat]
        label = cat_names.get(cat, f"cat{cat}")
        print(
            f"  cat {cat} ({label:10s}) : {s['accuracy'] * 100:.1f}%  ({s['correct']}/{s['total']})"
        )
    print("─" * 52)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="ai-knot v2 LOCOMO answer-accuracy runner")
    parser.add_argument("--convs", type=int, default=2)
    parser.add_argument("--data", type=Path, default=_DEFAULT_DATA)
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini", help="Model id for both answer and judge"
    )
    parser.add_argument(
        "--answer-model", type=str, default=None, help="Override answer model (defaults to --model)"
    )
    parser.add_argument(
        "--judge-model", type=str, default=None, help="Override judge model (defaults to --model)"
    )
    parser.add_argument("--max-atoms", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument(
        "--render-mode",
        choices=["raw", "atoms", "lineage", "raw-annotated"],
        default="raw",
        help=(
            "raw: episode turns + dates (v1-style); atoms: sparse triples (S1); "
            "lineage: CWP derivation tree (claim ← supporting raw observations); "
            "raw-annotated: raw episodes primary, atoms as light annotation"
        ),
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["atoms", "raw-bm25", "hybrid"],
        default="atoms",
        help="atoms: v2 MemoryAPI; raw-bm25: BM25 over raw episodes; hybrid: union",
    )
    parser.add_argument("--bm25-top-k", type=int, default=12)
    parser.add_argument(
        "--mention-expand",
        action="store_true",
        help="M2-lite: expand BM25 results with episodes mentioning query entities",
    )
    parser.add_argument(
        "--ribbon-radius",
        type=int,
        default=0,
        help="M3-lite: include ±N turn neighbors within session (default 0=off)",
    )
    parser.add_argument("--final-top-k", type=int, default=18)
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional path to write JSON report"
    )
    args = parser.parse_args()

    answer_model = args.answer_model or args.model
    judge_model = args.judge_model or args.model

    if not args.data.exists():
        print(f"ERROR: data file not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    ok, msg = verify_prompt_parity()
    if not ok:
        print(f"ERROR: prompt parity check failed: {msg}", file=sys.stderr)
        sys.exit(1)

    client = OpenAI()

    print(f"Loading {args.convs} conversation(s) from {args.data} ...")
    locomo_convs = parse_locomo_json(args.data, limit=args.convs)

    print(f"Models: answer={answer_model}  judge={judge_model}")
    print(f"Recall budget: max_atoms={args.max_atoms}  max_tokens={args.max_tokens}\n")

    results: list[ConvResult] = []
    wall_start = time.time()
    for conv in locomo_convs:
        print(f"Conv {conv.conv_idx}: {len(conv.turns)} turns, {len(conv.qa_pairs)} QA")
        cv = run_conversation(
            conv,
            client,
            answer_model,
            judge_model,
            max_atoms=args.max_atoms,
            max_tokens=args.max_tokens,
            render_mode=args.render_mode,
            retrieval_mode=args.retrieval_mode,
            bm25_top_k=args.bm25_top_k,
            mention_expand=args.mention_expand,
            ribbon_radius=args.ribbon_radius,
            final_top_k=args.final_top_k,
        )
        results.append(cv)
        print(f"  → {cv.total_atoms} atoms ingested\n")

    agg = aggregate(results)
    report(results, agg)
    print(f"\nWall-clock: {(time.time() - wall_start):.1f}s")

    if args.output:
        out = {
            "answer_model": answer_model,
            "judge_model": judge_model,
            "convs": [
                {
                    "conv_idx": cv.conv_idx,
                    "total_atoms": cv.total_atoms,
                    "qa": [
                        {
                            "qa_idx": q.qa_idx,
                            "category": q.category,
                            "question": q.question,
                            "gold": q.gold,
                            "candidate": q.candidate,
                            "verdict": q.verdict,
                            "atoms_retrieved": q.atoms_retrieved,
                            "rendered_chars": q.rendered_chars,
                        }
                        for q in cv.qa_results
                    ],
                }
                for cv in results
            ],
            "summary": agg,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Report written: {args.output}")


if __name__ == "__main__":
    main()
