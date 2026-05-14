from __future__ import annotations

import time
import uuid
from typing import Any

from deep_research.roles.base import BaseRole, RoleContext, RoleOutput
from deep_research.tools import fetch_arxiv_text, search_arxiv

_FULL_TEXT_TOP_N = 2  # fetch full PDF for the top-N most relevant hits


class ScoutRole(BaseRole):
    name = "scout"

    def run(self, ctx: RoleContext) -> RoleOutput:
        arxiv_hits = search_arxiv(ctx.focus, max_results=5)

        # Enrich top-N hits with full PDF text
        enriched: list[dict[str, str]] = []
        for hit in arxiv_hits:
            idx = arxiv_hits.index(hit)
            full_text = fetch_arxiv_text(hit["arxiv_id"]) if idx < _FULL_TEXT_TOP_N else ""
            enriched.append({**hit, "full_text": full_text})

        arxiv_block = ""
        if enriched:
            lines: list[str] = []
            for h in enriched:
                header = f"[{h['arxiv_id']}] {h['title']} ({h['authors']})"
                if h["full_text"]:
                    body = h["full_text"][:2000]
                    lines.append(f"- {header}\n  FULL TEXT (excerpt):\n  {body}")
                else:
                    lines.append(f"- {header}: {h['summary'][:200]}")
            arxiv_block = (
                "arXiv results (with full text for top hits):\n" + "\n\n".join(lines) + "\n\n"
            )

        existing = ctx.recall(ctx.focus, k=3, stream="sources")
        existing_block = ""
        if existing:
            titles = [
                str(r.get("entry", {}).get("content", r.get("text_preview", "")))[:80]
                for r in existing
            ]
            existing_block = (
                "Already in corpus (target gaps, avoid duplicates):\n"
                + "\n".join(f"- {t}" for t in titles)
                + "\n\n"
            )
        system = (
            "You are Scout, a multilingual research retriever specializing in multi-agent memory. "
            "Find sources across arXiv (English and Chinese), technical reports, algorithm papers, "
            "and memory system repositories. Be specific: title, authors, year, key claim."
        )
        user = (
            f"Research focus: {ctx.focus!r}.\n{arxiv_block}{existing_block}"
            "Identify 2-3 sources to retrieve next (can include the arXiv hits above or new ones). "
            "For each: NAME | TYPE (paper/repo/concept) | YEAR | KEY_CLAIM | RELEVANCE"
        )
        resp = self.llm.chat(system, user)
        full_text_count = sum(1 for h in enriched if h["full_text"])
        entry: dict[str, Any] = {
            "tick": ctx.tick,
            "ts": time.time(),
            "source_id": str(uuid.uuid4())[:8],
            "focus": ctx.focus,
            "arxiv_hits": enriched,
            "full_text_fetched": full_text_count,
            "content": resp.content,
        }
        ctx.corpus.append_source(entry)
        return RoleOutput(
            role_name=self.name,
            summary=f"Scout: {resp.content[:120]}",
            tokens_used=resp.total_tokens,
            data={
                "source_id": entry["source_id"],
                "arxiv_count": len(arxiv_hits),
                "full_text_fetched": full_text_count,
            },
        )
