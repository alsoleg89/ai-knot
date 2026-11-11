What this change does: It adds an explicit read of all current admissible task and generic facts, with whole text and exact namespaced revisions. The read returns the entire fitting corpus or raises the existing SnapshotTooLarge error, and uses the same source admission as SemanticContext.

Looks good. Ship. There are no unresolved findings in the reviewed source snapshot. The two check failures found during this review were fixed and verified before this report was finalized.

**Review scope and expected use**

This was an independent Astra + Ponytail read-only review of the uncommitted change on HEAD `813daa5fe332359eb98461336a96b84cc825c356`. I read the implementation plan, every changed production file, TypeScript transport paths, strict decoding/error handling, evidence/lineage/freshness checks, SQL read transaction implementations, connected context and experience tests, README, host examples, and final documentation. I assumed explicitly invoked reads for a team using SQLite or PostgreSQL. Output bounds do not bound the total retained namespace scan; this review makes no fleet-throughput or large-corpus capacity claim.

The code preserves exact labels and strict integer limits, requires task_key explicitly, distinguishes private/shared IDs, sorts the result by namespace and fact ID, and raises before returning any partial result. Source rejection happens before budget accounting. The read uses a read-only SQL transaction and adds no inference, statistics, intent, lease, or subscription writes. The scope deduplication for an owner named __shared__ is present, and its regression test passed.

The semantic extraction retains candidate order, pre/post-I/O admission, vector/cache behavior, ranking, and cost reservations. The lexical change is limited to shared scope/applicability helpers; its optimized index and assembly behavior remain intact. The HTTP/MCP additions use the existing configured-owner strict decoder and error codec. TypeScript forwards the new operation through the existing HTTP/MCP paths and preserves structured overflow errors.

**Evidence from this review**

- Python 3.12.13 SQLite gate: 90 passed, 98 deselected across task snapshot, frozen semantic equivalence, HTTP/MCP, lexical context, and semantic context tests. PostgreSQL cases were explicitly excluded with `-k sqlite`.
- After the final slot/outcome/example additions: task snapshot plus existing examples returned 24 passed and one exact-output assertion failure. After its expected output was fixed, the failing subprocess example test passed in isolation. The other 24 checks were already green.
- Focused Ruff checks on every changed Python production/test file and the example now pass with `--no-cache`. The initial failure and its fix are recorded below.
- Independent SQLite probes passed with provider HTTP and embedding calls blocked: 130 denied shared facts of 70,000 characters each do not consume a one-fact/one-character output budget; equal-value slot revisions return only the newest fact; conflicting slot values return none; a corrected action outcome removes both its recorded lesson and a derived copy; restoring old raw facts does not restore their eligibility or cause overflow.
- The frozen equivalence source comes from a tracked committed artifact. Its semantic.py content exactly equals `git show 813daa5:src/ai_knot/always_on/semantic.py`; both have SHA-256 `bac58b11b6153bbda26b5a130b92dbdd05e8c7a3306bbdcf51545ff1b47faf06`. The comparison intentionally uses current shared dependencies to isolate the extraction. It is a deterministic behavior/cost-input check, not a model-quality experiment.

Test command for the main gate:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /tmp/ai-knot-pr-ci/py312/bin/python -m pytest --no-cov -o cache_dir=/tmp/ai-knot-product-quality/task-snapshot-astra-pytest-cache -k sqlite tests/always_on/test_task_snapshot.py tests/always_on/test_task_snapshot_equivalence.py tests/always_on/test_task_snapshot_surface.py tests/always_on/test_surfaces.py tests/always_on/test_mcp_surface.py tests/always_on/test_task_context.py tests/always_on/test_semantic_context.py
```

**Resolved findings**

1. **New overflow wording failed the required lint gate** (`src/ai_knot/always_on/codec.py`).
   - **What this is:** One shared message now applies to both subscription and task snapshots.
   - **Problem:** The new line was 112 characters, exceeding the configured 100-character Ruff limit. The focused command returned E501.
   - **Fix:** Parenthesize the same string over multiple lines. The implementation now does so, and Ruff passes.
   - **If skipped:** Required CI fails despite valid runtime behavior.
2. **The existing subprocess example test expected the old output** (`tests/always_on/test_example.py:53`).
   - **What this is:** The runnable task-context example now demonstrates successful whole-corpus loading and explicit overflow handling.
   - **Problem:** Its older exact-dictionary assertion omitted the new snapshot_facts and snapshot_overflow_handled fields. Running it reproduced one failure.
   - **Fix:** Add the two expected values to that assertion. The parent applied this change, and the exact failing test now passes.
   - **If skipped:** Standard CI fails when running the existing example tests.

**Documentation and source-claim check**

The final task-snapshot guide explains empty results, Unicode code-point counts, all-or-error overflow, explicit selector fallback, and retained claim/review/checkpoint/begin_action guards. The host-flow addition does not change inspect_turn readiness or scheduling. The earlier 15/16 canonical-prefix result is explicitly excluded as an accuracy result for this new contract.

I independently opened all five primary sources from the adoption note on 2026-10-28. The note correctly distinguishes [legacy Letta blocks](https://docs.letta.com/v1-sdk/memory/memory-blocks) from [current Git repository shared memory](https://docs.letta.com/concepts/shared-memory); attachment lifetime and content-hash preconditions are supported by [Letta's SDK repository guide](https://docs.letta.com/agent-sdk/repositories). The scope and loading comparisons are supported by [LangGraph memory concepts](https://docs.langchain.com/oss/python/concepts/memory) and [Deep Agents memory](https://docs.langchain.com/oss/python/deepagents/memory). Those sources support mechanism comparisons, not accuracy superiority.

**Reviewed source hashes**

Captured at 2026-10-28T09:42:31.113890+00:00. These hashes identify the exact working files reviewed; a later edit requires its own verification.

| File | SHA-256 |
|---|---|
| `docs/always-on-team-cycle.md` | `ae28ae5e4f4c8676e69fe2814dd5faf5220aa8177f0ebadf0c0602df4dbcc8bb` |
| `docs/semantic-context.md` | `f297cf25a058f8b16401b6d2ea35ca1b97054ac8a396538f27c24521fcf9a5eb` |
| `docs/task-context.md` | `1e3efb94a4846acd216f881d0587c54eb721548b526e92e119e5b49c454441ec` |
| `docs/task-snapshot.md` | `3cbd6cc9100a07958e20df1f53198b9823d798a8763404bccb6fc27cac8ede91` |
| `docs/research/2026-10-28-task-bundle-adoption.md` | `3fc732c827d2c7c2db42de71b8646814353704d744bc46c8709c76d4e9a524b3` |
| `docs/superpowers/plans/2026-10-28-task-snapshot.md` | `d3975f3ab60950908aa9f8cb7d4e8332f9541b04d7069a47a5a41feb0b443125` |
| `examples/task_context.py` | `a1a8849e048d5971a2db6e2a6aa94f1ebd00c8720169f99fee26037206eaf4e2` |
| `npm/src/__tests__/always_on.test.ts` | `ec5bd77202989f3bfb120240ef4a89b883e83421c2ed6a5e449006d8a1669835` |
| `npm/src/always_on.ts` | `db972bec2944f7c161827c8577c7ec29e5b070828112340f05ed1203b33a9b9a` |
| `src/ai_knot/_always_on_tools.py` | `1e67cdb33ad9a95a94c0c9155bb228cead9c38d45656c1a978710d73e658f1ee` |
| `src/ai_knot/always_on/codec.py` | `b774c448cf3dc072583f2c7076bbc09a254adde8b8c51ad77a7a57f0d0d522d5` |
| `src/ai_knot/always_on/context.py` | `c419532530ddf226de2390760ff78436c1e35e2fab3be517502a58fe3d2db21d` |
| `src/ai_knot/always_on/memory.py` | `e2958caa58ddea30bc566e97e32d263b69b6641535d32cbd05e3f72c24170194` |
| `src/ai_knot/always_on/semantic.py` | `b28234582cba9d8a975854fd70830e305a172ab88d187efd758bff03613925e2` |
| `src/ai_knot/server/always_on.py` | `a9829c1ee8481f3aa4fa1ec814c56a89ffb3f637008b7d011db5d9ec3f5a7f0d` |
| `tests/always_on/test_example.py` | `7ab85fa6434a59b194bbf9aa60a30752e5738b2ac2c615cbbda233cd7e15bdc5` |
| `tests/always_on/test_mcp_surface.py` | `989cb4cc8cfddc508f83dc780d983899a05d170d9dcb59b1688a1b6ccaa165eb` |
| `tests/always_on/test_surfaces.py` | `c2270958e62ffeb1d3336418eb308b74f2f4c78c6328f8b27be31e4bb24af02b` |
| `tests/always_on/test_task_snapshot.py` | `9524781334623de27579935ba7b7f3f6f411ca33c5aa22af5f0a7b27313aa513` |
| `tests/always_on/test_task_snapshot_equivalence.py` | `94acada3ae11b4963981a6a788cf8bf94d75ce0b4d2ff623b97a1a083354d174` |
| `tests/always_on/test_task_snapshot_surface.py` | `4c7412b67947aa6bcf070b16b562afd7f7d1b076249653cda9b831c72d6d1a67` |
| `docs/performance/semantic-context-runtime.json.gz` | `d39dad74a0774b8cec02f4362b4dd98b6b375ac8b7308be47953076cdc014abe` |

Verdict: Ship.

Not checked: full CI, the PostgreSQL runtime gate, cross-version Python gates, and TypeScript execution/build are owned by the parent or another reviewer. No provider/model-quality rerun, load test, credentials, Keychain access, original-checkout writes, release action, or production mutation was performed. The verdict is a code-review result, not a substitute for those required final gates.
