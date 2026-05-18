from __future__ import annotations

from pathlib import Path

from deep_research.config import CampaignConfig
from deep_research.corpus import Corpus
from deep_research.llm import LLMClient, LLMResponse, MockLLMClient
from deep_research.orchestrator import Orchestrator


class _RecordingLLM(LLMClient):
    def __init__(self) -> None:
        self.users: list[str] = []

    @property
    def model(self) -> str:
        return "recording"

    def chat(self, system: str, user: str) -> LLMResponse:
        self.users.append(user)
        return LLMResponse(content="next focus", input_tokens=1, output_tokens=1)


def _make_orch(corpus: Corpus, tick_budget: int = 3) -> Orchestrator:
    config = CampaignConfig(
        brief_text="test brief",
        tick_budget=tick_budget,
        wall_clock_seconds=3600,
        token_budget=1_000_000,
    )
    return Orchestrator(config=config, corpus=corpus, llm=MockLLMClient())


def test_ticks_run_e2e_on_mock(tmp_path: Path) -> None:
    """P0 gate: N ticks run e2e on mock LLM, corpus and state.json written."""
    corpus = Corpus(tmp_path / "campaign")
    config = CampaignConfig(
        brief_text="test brief",
        tick_budget=3,
        wall_clock_seconds=3600,
        token_budget=1_000_000,
    )
    corpus.initialize("id-1", config.hash())
    orch = Orchestrator(config=config, corpus=corpus, llm=MockLLMClient())
    orch.run()

    final = corpus.load_state()
    assert final.tick == 3
    assert final.status == "exhausted"
    assert len(corpus.read_journal()) == 3


def test_roles_receive_campaign_brief(tmp_path: Path) -> None:
    """The root brief must stay in role prompts even after focus evolves."""
    corpus = Corpus(tmp_path / "campaign")
    config = CampaignConfig(
        brief_text="recover one-off facts in long dialogues",
        tick_budget=1,
        wall_clock_seconds=3600,
        token_budget=1_000_000,
    )
    corpus.initialize("id-1", config.hash(), brief=config.brief_text)
    llm = _RecordingLLM()
    orch = Orchestrator(config=config, corpus=corpus, llm=llm)
    orch.run()

    assert len(llm.users) >= 2
    assert all("recover one-off facts in long dialogues" in user for user in llm.users)


def test_restart_continues_from_checkpoint(tmp_path: Path) -> None:
    """Crash-resumable: re-creating Orchestrator continues from saved checkpoint."""
    corpus = Corpus(tmp_path / "campaign")
    config = CampaignConfig(
        brief_text="test brief",
        tick_budget=4,
        wall_clock_seconds=3600,
        token_budget=1_000_000,
    )
    corpus.initialize("id-1", config.hash())
    orch1 = Orchestrator(config=config, corpus=corpus, llm=MockLLMClient())

    # Run 2 ticks manually, simulating a mid-run checkpoint
    for _ in range(2):
        orch1._tick()

    mid = corpus.load_state()
    assert mid.tick == 2

    # Re-create orchestrator (simulates process restart) and run to completion
    orch2 = Orchestrator(config=config, corpus=corpus, llm=MockLLMClient())
    orch2.run()
    final = corpus.load_state()
    assert final.tick == 4
    assert final.status == "exhausted"


def test_token_budget_stops_loop(tmp_path: Path) -> None:
    """Token budget exhaustion stops the loop."""
    corpus = Corpus(tmp_path / "campaign")
    config = CampaignConfig(
        brief_text="test",
        tick_budget=1000,
        wall_clock_seconds=3600,
        token_budget=1,  # exhausted immediately after first tick
    )
    corpus.initialize("id-1", config.hash())
    orch = Orchestrator(config=config, corpus=corpus, llm=MockLLMClient(tokens=5))
    orch.run()
    final = corpus.load_state()
    assert final.status == "exhausted"
    assert final.tick < 1000


def test_stopped_status_halts_loop(tmp_path: Path) -> None:
    """status='stopped' in state.json prevents any ticks from running."""
    corpus = Corpus(tmp_path / "campaign")
    config = CampaignConfig(
        brief_text="test",
        tick_budget=100,
        wall_clock_seconds=3600,
        token_budget=1_000_000,
    )
    state = corpus.initialize("id-1", config.hash())
    state.status = "stopped"
    corpus.save_state(state)

    orch = Orchestrator(config=config, corpus=corpus, llm=MockLLMClient())
    orch.run()
    assert corpus.load_state().tick == 0


def test_journal_written_per_tick(tmp_path: Path) -> None:
    corpus = Corpus(tmp_path / "campaign")
    corpus.initialize("id-1", "hash")
    orch = _make_orch(corpus, tick_budget=2)
    orch.run()
    entries = corpus.read_journal()
    assert len(entries) == 2
    assert all("tick" in e and "role" in e and "summary" in e for e in entries)


def test_role_cycle_rotates(tmp_path: Path) -> None:
    """Worker roles rotate through the cycle each tick."""
    corpus = Corpus(tmp_path / "campaign")
    corpus.initialize("id-1", "hash")
    orch = _make_orch(corpus, tick_budget=6)
    orch.run()
    roles = [e["role"] for e in corpus.read_journal()]
    assert roles == ["scout", "analyst", "theorist", "prover", "critic", "experimenter"]


def test_prover_emits_fitness_record(tmp_path: Path) -> None:
    """P2 gate: Prover tick produces a fitness record in the fitness index."""
    corpus = Corpus(tmp_path / "campaign")
    corpus.initialize("id-1", "hash")
    # Seed a theory candidate so Prover can attribute its proof
    corpus.append_theory_candidate({"candidate_id": "seed-01", "content": "seed theory"})
    orch = _make_orch(corpus, tick_budget=4)  # tick 3 = prover
    orch.run()
    index = corpus.read_fitness_index()
    assert len(index) >= 1
    record = index[0]
    assert "candidate_id" in record
    assert "fitness" in record
    assert 0.0 <= float(record["fitness"]) <= 1.0


def test_full_cycle_writes_fitness_and_proofs(tmp_path: Path) -> None:
    """One full 6-tick cycle writes proof entries and fitness records."""
    corpus = Corpus(tmp_path / "campaign")
    corpus.initialize("id-1", "hash")
    orch = _make_orch(corpus, tick_budget=6)
    orch.run()
    assert len(corpus.read_proofs()) >= 1
    assert len(corpus.read_fitness_index()) >= 1


def test_p3_experimenter_prototype_in_full_cycle(tmp_path: Path) -> None:
    """P3 gate: full 6-tick cycle with code-returning LLM writes a prototype file."""
    corpus = Corpus(tmp_path / "campaign")
    corpus.initialize("id-1", "hash")
    corpus.append_theory_candidate({"candidate_id": "seed-01", "content": "seed"})
    code = 'print("p3 prototype")'
    llm = MockLLMClient(response=f"```python\n{code}\n```\nTYPE: free")
    config = CampaignConfig(
        brief_text="test",
        tick_budget=6,
        wall_clock_seconds=3600,
        token_budget=1_000_000,
    )
    orch = Orchestrator(config=config, corpus=corpus, llm=llm)
    orch.run()
    # tick 5 = experimenter role
    assert len(corpus.list_prototypes()) >= 1
