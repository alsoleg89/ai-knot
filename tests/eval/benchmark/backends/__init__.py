"""Benchmark backend implementations.

Basic backends (no optional deps required):
  BaselineBackend, AiKnotBackend, QdrantEmulator, Mem0Emulator

Extended backends (require `pip install -e '.[benchmark]'` + services running):
  QdrantRealBackend  — qdrant-client + Qdrant on localhost:6333
  Mem0RealBackend    — mem0ai + Ollama on localhost:11434
"""

from tests.eval.benchmark.backends.ai_knot_backend import AiKnotBackend
from tests.eval.benchmark.backends.baseline import BaselineBackend
from tests.eval.benchmark.backends.mem0_emulator import Mem0Emulator
from tests.eval.benchmark.backends.qdrant_emulator import QdrantEmulator

__all__ = ["AiKnotBackend", "BaselineBackend", "Mem0Emulator", "QdrantEmulator"]


def get_qdrant_real() -> "QdrantRealBackend":  # noqa: F821
    from tests.eval.benchmark.backends.qdrant_real import QdrantRealBackend

    return QdrantRealBackend()


def get_mem0_real() -> "Mem0RealBackend":  # noqa: F821
    from tests.eval.benchmark.backends.mem0_real import Mem0RealBackend

    return Mem0RealBackend()
