"""S26 — Many-Agent Sparse Assembly with Near-Miss Hubs.

Tests fan-in retrieval across N specialist agents (N configurable, default
10 / 100 / 1000).  Each agent publishes 1 unique shard fact with rare domain
markers and 1 distractor fact with generic vocabulary.  10% of agents are
hub agents with an extra near-miss fact that partially overlaps query facets.

Each query requires assembling 3 shards from 3 different agents chosen
from distinct domain clusters.  Evaluation uses both origin_agent_id
from retrieved Fact objects and rare-marker text matching.

Metrics (per scale point N):
  target_shard_recall_at_N — fraction of target shards found in top-k
  all_shards_covered_at_N  — fraction of queries with all 3/3 shards
  distractor_rate_at_N     — fraction of non-target facts in top-k
  agent_diversity_at_N     — mean normalised count of distinct target agents
  p95_retrieve_ms_at_N     — 95th-percentile retrieval latency
  last_target_rank_at_N    — mean rank of deepest target shard (lower = better)

Only runs against MultiAgentMemoryBackend.
"""

from __future__ import annotations

import time

from tests.eval.benchmark._eval_utils import percentile
from tests.eval.benchmark.base import MultiAgentMemoryBackend, ScenarioResult
from tests.eval.benchmark.judge import BaseJudge

SCENARIO_ID = "s26_sparse_assembly"
TOP_K = 10
_DEFAULT_SCALE_POINTS = [10, 100, 1000]
_DEFAULT_NUM_QUERIES = 5
_QUERIER = "querier"

# ---------------------------------------------------------------------------
# 20 technical domains: (domain_name, shard_concept, query_concept)
#
# shard_concept  — appears verbatim in the agent's shard fact.
# query_concept  — used in queries; shares the TOPIC but differs in LEXICAL form
#                  to prevent trivial BM25 shortcuts.
# ---------------------------------------------------------------------------
_DOMAINS: list[tuple[str, str, str]] = [
    (
        "cryptographic-protocols",
        "homomorphic lattice reduction over ring-LWE",
        "encrypted computation without decryption",
    ),
    (
        "graph-databases",
        "hypergraph bisection with spectral Fiedler vectors",
        "partitioning interconnected data structures",
    ),
    (
        "satellite-telemetry",
        "Doppler-compensated CCSDS frame synchronisation",
        "orbital data stream alignment",
    ),
    (
        "quantum-error-correction",
        "surface-code stabiliser measurement cycling",
        "fault-tolerant qubit state preservation",
    ),
    (
        "bioinformatics-pipelines",
        "de Bruijn assembly with paired-end scaffolding",
        "genome reconstruction from short reads",
    ),
    (
        "supply-chain-logistics",
        "multi-echelon stochastic lot-sizing with lead-time variance",
        "inventory optimisation across distribution tiers",
    ),
    (
        "embedded-firmware",
        "RTOS preemption-priority ceiling protocol",
        "real-time task scheduling on microcontrollers",
    ),
    (
        "acoustic-modelling",
        "parabolic-equation split-step Fourier propagation",
        "underwater sound field prediction",
    ),
    (
        "cfd-simulation",
        "immersed-boundary lattice-Boltzmann voxelisation",
        "fluid dynamics around complex geometries",
    ),
    (
        "edge-computing",
        "serverless cold-start mitigation via checkpoint-restore",
        "low-latency function execution at the network edge",
    ),
    (
        "robotics-perception",
        "LiDAR SLAM with scan-context loop closure",
        "autonomous spatial mapping from point clouds",
    ),
    (
        "compiler-optimisation",
        "polyhedral loop tiling with isl scheduling",
        "automatic parallelisation of nested loops",
    ),
    (
        "network-security",
        "eBPF XDP packet classification at line rate",
        "high-speed traffic filtering in kernel space",
    ),
    (
        "climate-modelling",
        "spectral-element dynamical core with semi-implicit time-stepping",
        "atmospheric circulation numerical integration",
    ),
    (
        "financial-risk",
        "nested Monte-Carlo CVA with wrong-way risk",
        "counterparty credit exposure estimation",
    ),
    (
        "nlp-information-extraction",
        "span-level contrastive entity linking",
        "mapping text mentions to knowledge base entries",
    ),
    (
        "digital-twins",
        "physics-informed neural operator surrogate modelling",
        "real-time virtual replica of physical assets",
    ),
    (
        "photonics-design",
        "FDTD with perfectly matched layer boundary truncation",
        "electromagnetic wave propagation simulation",
    ),
    (
        "drug-discovery",
        "molecular docking with AutoDock Vina scoring function",
        "protein-ligand binding affinity prediction",
    ),
    (
        "autonomous-vehicles",
        "occupancy-grid fusion from radar and camera sensors",
        "multi-modal environment perception for self-driving",
    ),
]

# Unique rare-marker prefixes (combined with agent index for global uniqueness).
_RARE_PREFIXES = [
    "Zeph",
    "Qorx",
    "Blyn",
    "Vynt",
    "Krel",
    "Drix",
    "Phex",
    "Juno",
    "Wrax",
    "Tyve",
    "Nylx",
    "Crix",
    "Fael",
    "Omyx",
    "Lurv",
    "Svek",
    "Ghal",
    "Pryn",
    "Exul",
    "Hzik",
]


# ---------------------------------------------------------------------------
# Deterministic data generators
# ---------------------------------------------------------------------------


def _agent_id(i: int) -> str:
    return f"specialist_{i:04d}"


def _rare_marker(i: int) -> str:
    return f"{_RARE_PREFIXES[i % len(_RARE_PREFIXES)]}{i:04d}"


def _domain(i: int) -> tuple[str, str, str]:
    return _DOMAINS[i % len(_DOMAINS)]


def _shard_fact(i: int) -> str:
    domain_name, shard_concept, query_concept = _domain(i)
    marker = _rare_marker(i)
    return (
        f"The {domain_name} module {marker} implements {shard_concept} "
        f"enabling {query_concept} with adaptive convergence tuning."
    )


def _distractor_fact(i: int) -> str:
    domain_name = _domain(i)[0]
    return (
        f"Standard {domain_name} operations require regular performance "
        f"monitoring and quarterly security audits."
    )


def _near_miss_fact(i: int, query_concepts: list[str]) -> str:
    """Hub agent's near-miss: overlaps with 2 query concepts, lacks shard specifics."""
    c0 = query_concepts[i % len(query_concepts)]
    c1 = query_concepts[(i + 1) % len(query_concepts)]
    domain_name = _domain(i)[0]
    return (
        f"A {domain_name} overview covers {c0} and {c1} "
        f"at a conceptual level without implementation specifics."
    )


def _select_targets(n: int, q: int) -> list[int]:
    """Select 3 target agent indices for query *q*, spread across the agent range."""
    stride = max(1, n // 3)
    t0 = (q * 7) % n  # prime multiplier for spread
    t1 = (t0 + stride) % n
    t2 = (t0 + 2 * stride) % n
    # Guarantee distinctness.
    if t1 == t0:
        t1 = (t0 + 1) % n
    if t2 in (t0, t1):
        t2 = (max(t0, t1) + 1) % n
    return [t0, t1, t2]


def _build_query(targets: list[int]) -> str:
    """Build a query from the query_concepts of 3 target agents."""
    concepts = [_domain(t)[2] for t in targets]
    return (
        f"How can a system integrate {concepts[0]}, "
        f"{concepts[1]}, and {concepts[2]} into a unified pipeline?"
    )


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


async def _measure_at_checkpoint(
    backend: MultiAgentMemoryBackend,
    current_n: int,
    num_queries: int,
) -> dict[str, float]:
    """Run queries against the current pool state and return metric values."""
    queries_targets: list[tuple[str, list[int]]] = []
    for q in range(num_queries):
        targets = _select_targets(current_n, q)
        queries_targets.append((_build_query(targets), targets))

    shard_hits_per_query: list[int] = []
    latencies_ms: list[float] = []
    last_ranks: list[int] = []
    diversities: list[int] = []
    distractor_counts: list[int] = []
    total_retrieved = 0

    for query, targets in queries_targets:
        target_markers = {_rare_marker(t) for t in targets}
        target_agent_ids = {_agent_id(t) for t in targets}

        t0 = time.perf_counter()
        result = await backend.pool_retrieve(_QUERIER, query, top_k=TOP_K)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)

        targets_found: set[str] = set()
        last_rank = 0
        n_distractors = 0

        if result.facts:
            total_retrieved += len(result.facts)
            for rank_0, fact in enumerate(result.facts):
                is_target = fact.origin_agent_id in target_agent_ids
                has_marker = any(m in fact.content for m in target_markers)
                if is_target and has_marker:
                    targets_found.add(fact.origin_agent_id)
                    last_rank = rank_0 + 1
                else:
                    n_distractors += 1
        else:
            total_retrieved += len(result.texts)
            for rank_0, text in enumerate(result.texts):
                if any(m in text for m in target_markers):
                    targets_found.add(f"text_hit_{rank_0}")
                    last_rank = rank_0 + 1
                else:
                    n_distractors += 1

        n_found = len(targets_found)
        shard_hits_per_query.append(n_found)
        diversities.append(n_found)
        distractor_counts.append(n_distractors)
        last_ranks.append(last_rank if n_found > 0 else TOP_K + 1)

    nq = num_queries
    return {
        "target_shard_recall": sum(shard_hits_per_query) / (3 * nq),
        "all_shards_covered": sum(1 for h in shard_hits_per_query if h >= 3) / nq,
        "distractor_rate": (sum(distractor_counts) / total_retrieved if total_retrieved else 1.0),
        "agent_diversity": (sum(diversities) / nq) / 3.0,
        "p95_retrieve_ms": percentile(latencies_ms, 95),
        "last_target_rank": sum(last_ranks) / nq,
    }


async def _insert_agents(
    backend: MultiAgentMemoryBackend,
    start: int,
    end: int,
    all_query_concepts: list[str],
    total_n: int,
) -> None:
    """Insert and publish agents [start, end) into the pool."""
    n_hubs = max(1, total_n // 10)
    for i in range(start, end):
        aid = _agent_id(i)
        await backend.insert_for_agent(aid, _shard_fact(i))
        await backend.insert_for_agent(aid, _distractor_fact(i))
        if i < n_hubs:
            await backend.insert_for_agent(aid, _near_miss_fact(i, all_query_concepts))
        await backend.publish_to_pool(aid)


def _query_concepts_for_n(n: int, num_queries: int) -> list[str]:
    """Collect all query_concepts used by queries targeting *n* agents."""
    concepts: list[str] = []
    for q in range(num_queries):
        targets = _select_targets(n, q)
        concepts.extend(_domain(t)[2] for t in targets)
    return concepts


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------


async def run(
    backend: MultiAgentMemoryBackend,
    judge: BaseJudge,
    *,
    scale_points: list[int] | None = None,
    duration_s: float | None = None,
    checkpoint_every: int = 10,
) -> ScenarioResult:
    """Measure fan-in retrieval quality at increasing agent counts.

    Two modes of operation:

    **scale_points mode** (default): Independent measurements at each N with
    full reset between them.  Fast, good for comparison.

    **duration mode** (``duration_s`` set): Continuous pool growth without
    reset.  Agents are inserted incrementally; every *checkpoint_every*
    agents the queries run and metrics are recorded together with wall-clock
    time.  Shows the degradation curve as the pool grows over real time.

    Args:
        scale_points: Agent counts to test (default [10, 100, 1000]).
            Ignored when *duration_s* is set.
        duration_s: Time budget in seconds for continuous-growth mode.
        checkpoint_every: Agents between checkpoints in duration mode
            (default 10).
    """
    num_queries = _DEFAULT_NUM_QUERIES
    judge_scores: dict[str, list[float]] = {}

    if duration_s is not None:
        return await _run_duration(backend, judge_scores, num_queries, duration_s, checkpoint_every)

    return await _run_scale_points(
        backend,
        judge_scores,
        num_queries,
        scale_points if scale_points is not None else _DEFAULT_SCALE_POINTS,
    )


async def _run_scale_points(
    backend: MultiAgentMemoryBackend,
    judge_scores: dict[str, list[float]],
    num_queries: int,
    points: list[int],
) -> ScenarioResult:
    """Independent measurements at discrete scale points (reset between)."""
    for n in points:
        await backend.reset()
        concepts = _query_concepts_for_n(n, num_queries)
        await _insert_agents(backend, 0, n, concepts, n)
        metrics = await _measure_at_checkpoint(backend, n, num_queries)
        for key, val in metrics.items():
            judge_scores[f"{key}_at_{n}"] = [val]

    first, last = points[0], points[-1]
    notes = (
        f"mode=scale_points, points={points}, queries={num_queries}, "
        f"top_k={TOP_K}, "
        f"recall@{first}="
        f"{judge_scores.get(f'target_shard_recall_at_{first}', [0])[0]:.2f}, "
        f"recall@{last}="
        f"{judge_scores.get(f'target_shard_recall_at_{last}', [0])[0]:.2f}, "
        f"p95@{first}="
        f"{judge_scores.get(f'p95_retrieve_ms_at_{first}', [0])[0]:.0f}ms, "
        f"p95@{last}="
        f"{judge_scores.get(f'p95_retrieve_ms_at_{last}', [0])[0]:.0f}ms"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores=judge_scores,
        insert_result=None,
        retrieval_result=None,
        notes=notes,
    )


async def _run_duration(
    backend: MultiAgentMemoryBackend,
    judge_scores: dict[str, list[float]],
    num_queries: int,
    duration_s: float,
    checkpoint_every: int,
) -> ScenarioResult:
    """Continuous pool growth with periodic checkpoints over *duration_s*."""
    await backend.reset()

    start_wall = time.perf_counter()
    inserted = 0
    checkpoints: list[int] = []

    while True:
        elapsed = time.perf_counter() - start_wall
        if elapsed >= duration_s:
            break

        # Insert next batch of agents.
        batch_end = inserted + checkpoint_every
        concepts = _query_concepts_for_n(batch_end, num_queries)
        await _insert_agents(backend, inserted, batch_end, concepts, batch_end)
        inserted = batch_end

        # Measure at this checkpoint.
        checkpoint_wall = time.perf_counter() - start_wall
        metrics = await _measure_at_checkpoint(backend, inserted, num_queries)
        for key, val in metrics.items():
            judge_scores[f"{key}_at_{inserted}"] = [val]
        judge_scores[f"wall_s_at_{inserted}"] = [checkpoint_wall]
        checkpoints.append(inserted)

    first, last = checkpoints[0], checkpoints[-1]
    total_wall = time.perf_counter() - start_wall
    notes = (
        f"mode=duration, budget={duration_s}s, wall={total_wall:.1f}s, "
        f"agents={last}, checkpoints={len(checkpoints)}, "
        f"recall@{first}="
        f"{judge_scores.get(f'target_shard_recall_at_{first}', [0])[0]:.2f}, "
        f"recall@{last}="
        f"{judge_scores.get(f'target_shard_recall_at_{last}', [0])[0]:.2f}"
    )

    return ScenarioResult(
        scenario_id=SCENARIO_ID,
        backend_name=backend.name,
        judge_scores=judge_scores,
        insert_result=None,
        retrieval_result=None,
        notes=notes,
    )
