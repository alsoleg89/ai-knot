"""Release guard: the version string must be identical in all shipped sources.

The package version lives in four files that ship independently or guide release
machinery:
  - ``pyproject.toml``            (the Python distribution)
  - ``src/ai_knot/__init__.py``  (``__version__``, the runtime/API surface)
  - ``npm/package.json``         (the TypeScript client)
  - ``npm/package-lock.json``    (the npm release/install lockfile)

A release bumps them together; drift between them ships a client that disagrees
with the server it talks to, a wheel whose ``__version__`` lies, or a lockfile
that still advertises an older package version. This test fails the moment they
diverge, so the mismatch is caught in CI, not after publish.
"""

from __future__ import annotations

import json
import pathlib
import re
import tomllib

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _pyproject_version() -> str:
    data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def _init_version() -> str:
    text = (_REPO_ROOT / "src" / "ai_knot" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    assert m is not None, "__version__ not found in src/ai_knot/__init__.py"
    return m.group(1)


def _npm_version() -> str:
    data = json.loads((_REPO_ROOT / "npm" / "package.json").read_text(encoding="utf-8"))
    return str(data["version"])


def _npm_lock_version() -> str:
    data = json.loads((_REPO_ROOT / "npm" / "package-lock.json").read_text(encoding="utf-8"))
    return str(data["version"])


def test_version_is_synced_across_pyproject_init_npm_and_lockfile() -> None:
    py = _pyproject_version()
    init = _init_version()
    npm = _npm_version()
    npm_lock = _npm_lock_version()
    assert py == init == npm == npm_lock, (
        f"version drift: pyproject={py!r} __init__={init!r} npm={npm!r} "
        f"npm-lock={npm_lock!r}; bump them together."
    )


def test_version_is_pep440_semver() -> None:
    # MAJOR.MINOR.PATCH with optional pre-release/build suffix — keeps the three
    # ecosystems (PyPI, npm) parseable by the same string.
    version = _init_version()
    assert re.match(r"^\d+\.\d+\.\d+([.-].+)?$", version), f"non-semver version: {version!r}"
