"""Strict frozen authored inputs, never executed on the host."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ai_knot.always_on.codec import decode_value, loads_strict
from ai_knot.always_on.models import context_label
from tests.eval.team_repos.cases import Case


@dataclass(frozen=True)
class Corpus:
    format_version: int
    provenance: dict[str, Any]
    cases: tuple[Case, ...]


def load_corpus(path: Path, *, expected_sha256: str) -> tuple[Case, ...]:
    raw = path.read_bytes()
    if len(raw) > 512000 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("corpus freeze mismatch")
    corpus = decode_value(loads_strict(raw), Corpus)
    if corpus.format_version != 1 or len(corpus.cases) != 3:
        raise ValueError("expected three cases in corpus version1")
    ids = set()
    scopes = set()
    for case in corpus.cases:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", case.id) or case.id in ids:
            raise ValueError("unique unambiguous case IDs required")
        context_label(case.task_key, "task_key")
        if not case.task_key or case.task_key in scopes or len(case.stages) != 3:
            raise ValueError("unique task scopes and three stages required")
        ids.add(case.id)
        scopes.add(case.task_key)
        if not 1 <= len(case.requirements) <= 5 or any(not q.strip() for q in case.requirements):
            raise ValueError("one to five public requirements required")
        stage_ids = set()
        for stage in case.stages:
            if not stage.id.strip() or stage.id in stage_ids:
                raise ValueError("unique stage IDs required")
            stage_ids.add(stage.id)
            paths = set()
            for doc in stage.documents:
                path = PurePosixPath(doc.path)
                if (
                    not doc.path
                    or str(path) != doc.path
                    or path.is_absolute()
                    or ".." in path.parts
                    or doc.path == "."
                    or doc.path in paths
                ):
                    raise ValueError("unique canonical relative document paths required")
                context_label(doc.task_key, "task_key")
                paths.add(doc.path)
    return corpus.cases
