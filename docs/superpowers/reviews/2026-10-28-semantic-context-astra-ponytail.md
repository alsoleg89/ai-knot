What this change does: It adds an explicit, budgeted Python semantic selector that checks readable, current source evidence before embedding and again before returning candidates. It also makes the multi-agent acceptance gate reject incomplete verification and adds a metered, fixed-protocol reader comparison.

No unresolved correctness, security, or minimality finding remains in the reviewed scope. The three concrete findings below were fixed by the owning agents and independently rechecked before live evaluation. This review changed no repository source, fixture, gold label, credential, or original checkout.

Resolved findings:

1. **A selected scenario with no measurements could pass** (`tests/eval/benchmark/base.py:75-88`).
   - **What this is:** The gate tracks selected scenarios separately from their results.
   - **Problem:** A passing S10 result plus a selected S8 result with `judge_scores={}` was accepted as `1/1 PASS`; S8 has no binding threshold, so its missing evidence disappeared.
   - **Fix:** The shared scenario check now rejects empty measurements and non-finite samples. The reproduced case now reports `Missing metrics`, renders FAIL, and exits 1; valid partial runs still leave absent, unselected thresholds n/a.
   - **If we skip it:** A selected scenario can execute no useful verification while the overall run claims success. This fix is present and verified.

2. **Cross-system-only reports could claim ai-knot acceptance** (`tests/eval/benchmark/report.py:354-360`).
   - **What this is:** Markdown and CLI should apply the same ai-knot acceptance scope.
   - **Problem:** A `mem0_multi_agent` S10 result with `cas_correctness=[1.0]` produced a PASS scorecard while the CLI correctly exited 1 because no ai-knot verification existed.
   - **Fix:** The report now uses the CLI's `ai_knot_multi_agent` prefix and has an explicit no-verification failure; the fallback to comparison backends is removed. The original counterexample now fails in both surfaces.
   - **If we skip it:** A saved report can contradict the enforced gate. This fix is present and verified.

3. **Cleanup failures erased already incurred provider charges** (`tests/eval/semantic_context_runner.py:333-360`).
   - **What this is:** Each evaluation arm must retain the reader and embedding request ledger even when the run fails.
   - **Problem:** After a mocked reader request charged 40 tokens, a storage cleanup error replaced the row with zero calls and zero tokens. Client close failures had the same path.
   - **Fix:** The runner retains its row and client through cleanup, marks the row failed, and preserves the meter. Independent storage-exit and client-exit reproductions now both retain one request and 40 tokens with sanitized errors.
   - **If we skip it:** Failed runs understate paid work. This fix is present and verified.

Verification performed:

- Final focused command: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /tmp/ai-knot-pr-ci/py312/bin/python -m pytest tests/always_on/test_semantic_context.py tests/test_ma_gate_failures.py tests/eval/test_semantic_context_runner.py -k 'not postgres' -q -p no:cacheprovider -o addopts=''` — **84 passed, 22 PostgreSQL cases deselected**.
- Host recipe subset: `tests/always_on/test_team_cycle.py -k 'not postgres and (semantic or unchanged_missing_context)'` with the same pytest configuration — **17 passed, 54 deselected**.
- Separate deterministic reproductions checked S10 plus empty S8, cross-system-only verification, and both cleanup-failure paths. These used SQLite and `httpx.MockTransport`, never a live provider.
- Reviewed source admission, exact revision checks, ACL redaction, support and active time, lineage/freshness, namespace cache identity, unchanged-text caching, callback timeout/failure accounting, whole-fact limits, omission semantics, snapshot fast path, and host action guards.
- Reviewed reader input projection, strict citation parsing, evidence-group scoring, shared reader/embedding meter, source/manifest fingerprints, and report persistence. Reader prompts do not receive gold labels or source IDs.
- The AST of the declared `GATE` threshold tuple is identical to HEAD; no target, operator, advisory flag, or threshold membership was weakened.
- The frozen semantic implementation, lazy exports, and semantic tests still exactly match the pre-fixture source freeze.

Interpretation and load limits:

- Runtime review assumes one owner/helper/event loop with at most 128 admitted candidates, as documented. The cap bounds paid work; namespace scanning and evidence checks are not a large-corpus index. No fleet or sustained-load claim follows.
- `review.complete` concerns admitted candidates and positive-score omissions only. An empty review or `status="ok"` does not establish task completeness, entailment, inference permission, or action authority. The new documentation states this accurately.
- The comparison is one authored, frozen batch of 16 closed-choice cases, across four domains with English/Russian cases. It is not an independent population estimate or an open-ended agent benchmark.
- The scoped-bundle control selects canonical IDs first; editorial low-ID placement can favor relevant records. Preserve and disclose that order bias rather than changing gold, IDs, or protocol after results.
- `grounded_correct` means a correct choice plus citation of every required evidence group (or correctly empty-basis abstention). It does not penalize extra irrelevant citations or prove causal faithfulness of the model's answer.
- Per-arm ceilings are equal, and one client meters embedding plus reader work. Cached/snapshot behavior changes actual spend; reserved and actual usage should remain separate in claims.

Verdict: Ship the reviewed implementation to the bounded live validation step. No unresolved review blocker remains; this verdict does not assert live model quality.

Not checked: live OpenAI behavior, PostgreSQL execution in this independent review, full CI, package release integration, external transports, or long-running/multi-process capacity. The parent owns those checks. No credentials were accessed and no provider requests were made.

Review completed: 2026-10-28T09:12:49.607613+00:00

Repository HEAD: `cf5c21062f21aaf398214f6acfbb21098f7cb964`

Pre-fixture freeze: `/tmp/ai-knot-product-quality/semantic-source-freeze.json`

Frozen independent manifest: `f1cdd8136013d62f4f83e94492d6faa2ee589c942871ce4481210430331907e4`

Source SHA-256 values at final review:

| File | SHA-256 |
|---|---|
| `src/ai_knot/always_on/semantic.py` | `bac58b11b6153bbda26b5a130b92dbdd05e8c7a3306bbdcf51545ff1b47faf06` |
| `src/ai_knot/always_on/__init__.py` | `5ea2da7e4ed86e263539957ee2888ecdc19412b3368f99f120383c1846d2ee44` |
| `tests/always_on/test_semantic_context.py` | `dba5575b0b76ec4439afc11b487b7f4a151b804fdafeb0d5375d070f50a660c0` |
| `examples/always_on_team.py` | `bfc9eee61f13fd6bea71eff334cf39ca625a4044a10b975654668881315ba162` |
| `tests/always_on/test_team_cycle.py` | `e95a71187a7c37fe06aa69731a00c4dd537f44050ae2ebcc33e4b29b2156d5df` |
| `docs/semantic-context.md` | `7c48bbed617e5f27c95ccd7c7bc6be170efd762ad1a8b52db8082fe168ae6776` |
| `docs/always-on-team-cycle.md` | `347f70a30cb5e7ce97a6d7beabfa536d685b7d331acbd3826787202219d78de2` |
| `tests/eval/benchmark/base.py` | `74affcca88457b551a7f182401bca35e2fd59ba197c4d53725856f3f305ebb12` |
| `tests/eval/benchmark/ma_gate.py` | `a35d1e5d165d029fbc9e10603dd75cbe7a2b8485b6f1c3c8b30e2d556623646e` |
| `tests/eval/benchmark/report.py` | `2c631aae9f475982c6dffb0cb6043b2f2668dbd0decd394e2012ff90590d2ca6` |
| `tests/eval/benchmark/runner.py` | `6b3e4f46bc7c62e8201dbbc347141714e29f5850b14369cfd8c372e8277f1972` |
| `tests/test_ma_gate_failures.py` | `65b614e9a0300d723d690aa274d888981dbd3fec411c2c8d8b7a0c7bdfc1c405` |
| `tests/eval/semantic_context_runner.py` | `0311d5dd3979eecde881955341ed62587cbd95a5ba5ca169bf01d7b835ad6f76` |
| `tests/eval/test_semantic_context_runner.py` | `1c0febaa7a5522754abdca6d48ef6e9652e7a7fd08c7e784eb26e4617bbbd5ff` |
| `tests/eval/semantic_context_cases.py` | `7ffc0b91db717026ac18602e74e189d99cec090221cfbeeeb5eda5bdb816562c` |
| `tests/eval/test_semantic_context_cases.py` | `7b6d1502d89bb768d796fd038e803278bd311cafc8c66579ad9905075de31b1f` |
| `tests/eval/team_decisions/openai.py` | `0d16e646eb5d37d50dd28c0462801695c80fd4418b67fa3af7131671982533b9` |
| `tests/eval/task_context/runner.py` | `6e6d90b6b0b248839230ff80ca9048e518b894860e9813f1260985d20f8dc3a3` |
| `src/ai_knot/always_on/memory.py` | `85e9838ff60cd70d4f3528f1a63693d2dcf77b6f805de9181adbf8f8c0f78769` |
| `src/ai_knot/always_on/lineage.py` | `ab8b91cf3e81457543f9994dce60d22371d3fea8293520662e20f210349fe5a8` |
| `src/ai_knot/always_on/changes.py` | `5a2aaff067694f8af764fc12bba1de970e6bc608612e628052cf3383315aea24` |
| `src/ai_knot/always_on/_state.py` | `c238d8a2ccf7f867f39c9e1bb6970fb6be2602cafd79c6176e43093e0b12fb5c` |
| `src/ai_knot/multi_agent/assembly.py` | `609ce1a88c2bc488dfd6a1963c6f7be1ea082a5713bc698e9b77df03b7a9b47a` |


Live-result audit addendum

The saved SQLite and PostgreSQL artifacts were independently read after the parent completed each single pass. No source, gold labels, IDs, retrieval protocol, reader protocol, or result rows were changed for this audit. No provider requests or model retries were made by the reviewer.

**Both runs are failed runs, not green evaluations.** Each artifact retains `status="failed"` after processing all 48 comparison records. The frozen CLI maps that status to exit 1. In each backend, the lexical `maintenance-prerequisites-ru` response is exactly `{"basis":[1],"choice":"ABSTAIN"}`. The parser rejects it with `ValueError` because abstention requires an empty basis; its choice and grounded-success flags remain false. This observed failure is retained without a protocol change or a retry.

The rejected response was still a paid, successful provider request: HTTP 200, request-ledger status `ok`, model `gpt-4.1-mini-2025-04-14`, 298 prompt tokens and 11 completion tokens. Its row retains **309 charged tokens**, replacing the initial 1,895-token reservation with reported actual usage. All 56 provider requests per backend have HTTP 200 and ledger status `ok`; the evaluation failure is reader-output validation, not an HTTP failure. Every row's `spent_tokens` equals the sum of its retained call charges.

The following values are **per backend**; SQLite and PostgreSQL match exactly:

| Arm | Grounded choices | All required groups visible on positive cases | Reader calls | Embedding calls | Charged tokens | Row execution status |
|---|---:|---:|---:|---:|---:|---|
| Lexical task context | 4/16 | 1/12 | 16 | 0 | 4,529 | 15 complete, 1 failed |
| Canonical scoped bundle | 15/16 | 12/12 | 16 | 0 | 5,068 | 16 complete |
| Semantic context | 14/16 | 12/12 | 16 | 8 | 6,111 | 16 complete |

The four abstention cases have no required evidence groups, so their `all_required_groups_visible` value is vacuously true. They are excluded from the 12-case visibility denominator above. Correct abstentions are 3/4 in every arm. Correct grounded positive choices are 1/12 lexical, 12/12 scoped bundle, and 11/12 semantic.

In the semantic arm, eight compact cases returned `snapshot` with zero embedding calls; the eight crowded cases each made one embedding call. All records fit the frozen ceilings of three whole facts, 600 content characters, two provider requests, and 12,000 charged/reserved-budget tokens per arm. The semantic arm used eight additional requests and 1,043 additional reported tokens compared with the strongest control, per backend. Token counts here combine reader and embedding usage; they are not a dollar-price comparison.

The remaining scored negatives also stay visible. Semantic selection returned all required source groups for `release-two-prerequisites`, but its reader chose B instead of frozen answer A. For `release-expired-approval`, both the scoped bundle and the semantic snapshot chose B instead of the required abstention. Candidate availability therefore did not ensure a correct decision or safe abstention, even when source filtering and output budgets held.

All 48 paired reader outputs, displayed contexts, selection diagnostics, row statuses, score fields, and token totals are identical between backends. All 48 paired reader request hashes match. Five of eight paired embedding request hashes match; three differ (`incident-manual-recovery-ru`, `release-two-prerequisites`, and `release-notification-ru`), so this audit does not claim byte-identical complete provider traffic. The source, manifest and protocol hashes match between backends, and both artifacts report `source_unchanged_during_run=true`. The stored runtime hashes also matched the reviewed files when checked. Across both runs there are 96 retained rows, 112 requests and 31,416 charged tokens.

**Interpretation:** On this authored batch, semantic candidates improved on lexical matching, but did not beat the canonical scoped-bundle control: 14/16 versus 15/16 grounded choices with more requests. Both nonlexical approaches exposed all required positive-case evidence groups. These results support retaining an optional experimental semantic path; they do not support replacing the cheap scoped bundle or claiming semantic superiority over the strongest tested control.

The original limitations remain binding: 16 editorial closed-choice cases, one pass per backend, one reader/embedding model pair, canonical-ID ordering bias, and no open-ended action or production trial. The two SQL backends reuse the same cases and yielded identical reader outputs; they are not 32 independent quality cases or an estimate of model variance. `grounded_correct` checks required-group citation, not causal faithfulness or absence of irrelevant extra citations. General accuracy, fleet capacity and production reliability remain unestablished. Defaults, gold and protocol should remain unchanged in response to these results.

No new runtime correctness blocker was found in the artifact audit. This does not convert the failed live evaluations into passing quality gates. The pre-live implementation verdict remains limited to the reviewed mechanics, and the live status and negatives must accompany any product-quality summary.

Audit completed: 2026-10-28T09:20:19.389886+00:00

Prior review SHA-256 before this append: `8d7105a7bb2b44dfd715514331924548ec3acd3122a1e0ea158f514f4f5efa4a`

Frozen protocol SHA-256: `c6cdbb9b97ce885f006f336274b27700c47aba2989124a2b2f79235b6318c0ce`

Frozen manifest SHA-256: `f1cdd8136013d62f4f83e94492d6faa2ee589c942871ce4481210430331907e4`

| Saved result artifact | SHA-256 |
|---|---|
| `/tmp/ai-knot-product-quality/semantic-live-sqlite.json` | `885c565974835a2bdf6d200d94af0bb1231563a36b6a3568fc17c953d5f10c4c` |
| `/tmp/ai-knot-product-quality/semantic-live-postgres.json` | `6f6594ca6756a45937ee13971481ae3b53b8ae87c4cba54a94b87d06936cb7db` |
