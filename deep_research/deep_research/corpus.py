from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SUBDIRS = (
    "sources",
    "theory",
    "theory_population",
    "proofs",
    "critique",
    "experiments",
    "journal",
)


@dataclass
class CampaignState:
    campaign_id: str
    config_hash: str
    brief: str = ""
    tick: int = 0
    wall_start: float = field(default_factory=time.time)
    tokens_used: int = 0
    status: str = "running"
    focus: str = "initial exploration"
    last_role: str = ""
    approval_queue: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CampaignState:
        return cls(
            campaign_id=str(d["campaign_id"]),
            config_hash=str(d["config_hash"]),
            brief=str(d.get("brief", "")),
            tick=int(d["tick"]),
            wall_start=float(d["wall_start"]),
            tokens_used=int(d["tokens_used"]),
            status=str(d["status"]),
            focus=str(d["focus"]),
            last_role=str(d["last_role"]),
            approval_queue=list(d.get("approval_queue", [])),
        )


class Corpus:
    def __init__(self, root: Path) -> None:
        self.root = root

    def initialize(
        self,
        campaign_id: str,
        config_hash: str,
        brief: str = "",
        focus: str | None = None,
    ) -> CampaignState:
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in SUBDIRS:
            (self.root / sub).mkdir(exist_ok=True)
        state = CampaignState(
            campaign_id=campaign_id,
            config_hash=config_hash,
            brief=brief,
            focus=focus if focus is not None else (brief or "initial exploration"),
        )
        self._write_state(state)
        return state

    def load_state(self) -> CampaignState:
        return CampaignState.from_dict(json.loads((self.root / "state.json").read_text()))

    def save_state(self, state: CampaignState) -> None:
        self._write_state(state)

    def _write_state(self, state: CampaignState) -> None:
        path = self.root / "state.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state.to_dict(), indent=2))
        tmp.replace(path)

    # ── Journal ──────────────────────────────────────────────────────────────

    def append_journal(self, entry: dict[str, Any]) -> None:
        self._append_jsonl("journal/journal.jsonl", entry)

    def read_journal(self, last_n: int = 20) -> list[dict[str, Any]]:
        path = self.root / "journal" / "journal.jsonl"
        if not path.exists():
            return []
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        return [json.loads(ln) for ln in lines[-last_n:]]

    # ── Theory ───────────────────────────────────────────────────────────────

    def write_theory(self, content: str) -> None:
        (self.root / "theory" / "theory.md").write_text(content)

    def read_theory(self) -> str:
        p = self.root / "theory" / "theory.md"
        return p.read_text() if p.exists() else "(no theory yet)"

    # ── Domain appenders ─────────────────────────────────────────────────────

    def append_source(self, entry: dict[str, Any]) -> None:
        self._append_jsonl("sources/sources.jsonl", entry)

    def append_proof(self, entry: dict[str, Any]) -> None:
        self._append_jsonl("proofs/proofs.jsonl", entry)

    def append_critique(self, entry: dict[str, Any]) -> None:
        self._append_jsonl("critique/critique.jsonl", entry)

    def append_experiment(self, entry: dict[str, Any]) -> None:
        self._append_jsonl("experiments/experiments.jsonl", entry)

    def write_prototype(self, proto_id: str, code: str) -> Path:
        path = self.root / "experiments" / f"prototype_{proto_id}.py"
        path.write_text(code)
        return path

    def list_prototypes(self) -> list[Path]:
        return sorted((self.root / "experiments").glob("prototype_*.py"))

    def append_theory_candidate(self, entry: dict[str, Any]) -> None:
        self._append_jsonl("theory_population/candidates.jsonl", entry)

    def read_theory_candidates(self, last_n: int = 5) -> list[dict[str, Any]]:
        path = self.root / "theory_population" / "candidates.jsonl"
        if not path.exists():
            return []
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        return [json.loads(ln) for ln in lines[-last_n:]]

    def append_fitness_record(self, entry: dict[str, Any]) -> None:
        self._append_jsonl("theory_population/fitness.jsonl", entry)

    def read_fitness_index(self) -> list[dict[str, Any]]:
        """Return latest fitness record per candidate_id, sorted by fitness descending."""
        path = self.root / "theory_population" / "fitness.jsonl"
        if not path.exists():
            return []
        records: dict[str, dict[str, Any]] = {}
        for line in path.read_text().splitlines():
            if line.strip():
                rec: dict[str, Any] = json.loads(line)
                cid = str(rec.get("candidate_id", ""))
                records[cid] = rec
        return sorted(records.values(), key=lambda r: float(r.get("fitness", 0.0)), reverse=True)

    def read_proofs(self, last_n: int = 5) -> list[dict[str, Any]]:
        path = self.root / "proofs" / "proofs.jsonl"
        if not path.exists():
            return []
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        return [json.loads(ln) for ln in lines[-last_n:]]

    def read_disproved_hypotheses(self, last_n: int = 20) -> list[str]:
        """Return short content excerpts of all disproved proof entries (last_n max)."""
        path = self.root / "proofs" / "proofs.jsonl"
        if not path.exists():
            return []
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        disproved = [
            json.loads(ln) for ln in lines if '"disproved"' in ln or "'disproved'" in ln
        ]
        return [p.get("content", "")[:200] for p in disproved[-last_n:]]

    # ── Approval queue ────────────────────────────────────────────────────────

    def add_to_approval_queue(self, exp_id: str, description: str) -> None:
        state = self.load_state()
        state.approval_queue.append(
            {"exp_id": exp_id, "description": description[:200], "status": "pending"}
        )
        self.save_state(state)

    def approve_experiment(self, exp_id: str) -> bool:
        return self._set_exp_status(exp_id, "approved")

    def reject_experiment(self, exp_id: str) -> bool:
        return self._set_exp_status(exp_id, "rejected")

    def _set_exp_status(self, exp_id: str, new_status: str) -> bool:
        state = self.load_state()
        for item in state.approval_queue:
            if item["exp_id"] == exp_id:
                item["status"] = new_status
                self.save_state(state)
                return True
        return False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _append_jsonl(self, rel_path: str, entry: dict[str, Any]) -> None:
        path = self.root / rel_path
        with path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
