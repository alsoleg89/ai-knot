"""Guardrail for potentially sensitive content in repository text files.

This is a heuristic safety net for repository text content in a public repo. It
does not replace legal review and should be adjusted to the team's compliance
requirements.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST_PATH = REPO_ROOT / "tests" / "compliance_allowlist.txt"
SCAN_ROOTS = (REPO_ROOT,)
SKIP_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
}
SKIP_FILE_NAMES = {"compliance_allowlist.txt", "test_legal_compliance_content.py"}
TEXT_EXTENSIONS = {".py", ".md", ".txt", ".json", ".yaml", ".yml"}

# Keep patterns explicit to avoid false positives like matching "transfer" for "trans".
BANNED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\blgbt[qia+]*\b", re.IGNORECASE),
    re.compile(r"\blesbian\b", re.IGNORECASE),
    re.compile(r"\bgay\b", re.IGNORECASE),
    re.compile(r"\bqueer\b", re.IGNORECASE),
    re.compile(r"\btransgender\b", re.IGNORECASE),
    re.compile(r"\btranssexual\b", re.IGNORECASE),
    re.compile(r"\bnon[- ]binary\b", re.IGNORECASE),
    re.compile(r"\bsame[- ]sex\b", re.IGNORECASE),
    re.compile(r"\bsexual orientation\b", re.IGNORECASE),
    re.compile(r"\bgender identity\b", re.IGNORECASE),
    re.compile(r"смен[аы]\s+пола", re.IGNORECASE),
    re.compile(r"нетрадицион\w*\s+сексуаль\w*\s+отношен\w*", re.IGNORECASE),
    re.compile(r"\bлгбт\b", re.IGNORECASE),
)


def _iter_text_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in SKIP_DIR_NAMES for part in path.parts):
                continue
            if path.name in SKIP_FILE_NAMES:
                continue
            if path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            files.append(path)
    return sorted(files)


def _load_allowlist() -> tuple[str, ...]:
    if not ALLOWLIST_PATH.exists():
        return ()

    entries: list[str] = []
    for raw_line in ALLOWLIST_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line.casefold())
    return tuple(entries)


def test_repository_text_has_no_blocklisted_legal_terms() -> None:
    allowlist = _load_allowlist()
    violations: list[str] = []

    for path in _iter_text_files():
        content = path.read_text(encoding="utf-8")

        for line_number, line in enumerate(content.splitlines(), start=1):
            lowered_line = line.casefold()
            if any(allowed in lowered_line for allowed in allowlist):
                continue

            for pattern in BANNED_PATTERNS:
                match = pattern.search(line)
                if match is None:
                    continue

                violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{line_number}: "
                    f"matched `{pattern.pattern}` in `{line.strip()}`"
                )
                break

    assert not violations, (
        "Potentially sensitive legal-compliance terms found in repository text.\n"
        "If a match is deliberate and reviewed, add a precise substring to "
        "tests/compliance_allowlist.txt.\n\n" + "\n".join(violations)
    )
