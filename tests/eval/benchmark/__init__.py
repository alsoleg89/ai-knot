"""Comparative benchmark suite: ai-knot vs Qdrant emulator vs mem0 emulator vs baseline.

Run via CLI:
    python -m tests.eval.benchmark.runner

Or a specific subset:
    python -m tests.eval.benchmark.runner --backends ai_knot,qdrant --scenarios s1,s4

Requires Ollama running at http://localhost:11434 with llama3.2:3b.
Use --mock-judge for offline / CI runs.
"""
