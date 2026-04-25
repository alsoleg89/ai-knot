# Extraction-Sufficient Witness Program: A Continuation of Invariant Witness Theory

[established] Date: 2026-04-25

[theoretical] Status: theoretical research program, not an engineering specification

[theoretical] Scope: persistent artificial agents reasoning across sessions under bounded memory and bounded computation, with emphasis on the reader-side failure that IWT does not formalize.

[theoretical] Relation to prior work: This document is a **continuation of Invariant Witness Theory** (`research/invariant_witness_theory_program.md`, 2026-04-24). IWT Phase 0–8 is not reproduced here; it is cited explicitly. ESWP extends IWT by adding a second protection criterion — extraction-sufficiency — that IWT's cohomological regret charge does not capture.

[theoretical] Label convention: Every substantive prose claim is marked [established], [theoretical], or [speculative].

[speculative] Core novelty claim: a witness-pack is **extraction-sufficient** for query Q under reader-model π_R if and only if the mutual information between the correct answer and the reader's output on the rendered pack exceeds (1 − ε) · H(y*_Q). IWT protects individual witnesses from being forgotten; ESWP protects **packs** from being rendered in ways that the reader cannot decode. The two criteria are orthogonal: a witness can survive forgetting (IWT-compliant) yet be extraction-insufficient (ESWP violation), and vice versa.

## Source Anchors

[established] IWT (Invariant Witness Theory) defines the witness primitive, cohomological regret charge, and four operations WRITE/READ/FORGET/CONSOLIDATE: `research/invariant_witness_theory_program.md`

[established] Empirical decomposition of cat1 failures on LoCoMo benchmark (9/30 WRONG with gold-item recall = 1.0 — gold in context but LLM extracts incorrectly): `research/cat1_55_investigation_20260423.md`

[established] Shannon's mutual information and channel capacity: C. E. Shannon, *A Mathematical Theory of Communication*, 1948.

[established] Landauer's principle: R. Landauer, *Irreversibility and Heat Generation in the Computing Process*, 1961.

[established] Bennett's reversible computing and the Landauer limit: C. H. Bennett, *Logical Reversibility of Computation*, 1973.

[established] Friston's free-energy principle: K. Friston, *The free-energy principle: a unified brain theory?*, 2010.

[established] Predictive coding: R. P. N. Rao and D. H. Ballard, *Predictive coding in the visual cortex*, 1999.

[established] Persistent homology and topological data analysis: H. Edelsbrunner and J. Harer, *Computational Topology*, 2010.

[established] Sheaf theory and data fusion: M. Robinson, *Topological Signal Processing*, 2014.

[established] RAG-Ragas and context-precision metrics: Shahul Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation*, 2023.

[established] LLM-as-judge evaluation: Lianmin Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, 2023.

---

# Phase 0: Extended Ontological Audit

## 0.1 What Is Memory? (IWT §0.1 — unchanged)

[theoretical] IWT defines memory as a physically instantiated macrostate M_t = φ(X_t) whose value counterfactually depends on H_t and whose possession can change future action loss. This definition is adopted unchanged.

[theoretical] The minimal test: ∃ h, h', q such that φ(X_t(h)) ≠ φ(X_t(h')) ∧ L(π(q, φ(X_t(h)))) ≠ L(π(q, φ(X_t(h')))).

## 0.2 What Is an Agent? (IWT §0.2 — unchanged)

[theoretical] A minimal formal agent A = (O, U, M, η, π). Definitions adopted from IWT §0.2 unchanged.

## 0.3 What Is Memory Failure? — Extended

[theoretical] IWT §0.3 defines memory failure as violation of **decision-sufficiency**:
∃ h, h' ∈ H : M(h) = M(h') ∧ d_Π(π*(·|h), π*(·|h')) > ε.

[speculative] ESWP adds a second primitive failure: violation of **extraction-sufficiency**. Even when M(h) correctly captures the decision-relevant constraint, the reader π_R may fail to extract the answer from render(M(h), Q).

[theoretical] Formally, extraction-sufficiency failure is:
I(y*_Q ; π_R(Q, render(W_Q))) < (1 − ε) · H(y*_Q)

where y*_Q is the correct answer, render is the deterministic context assembly operator, and π_R is the reader language model.

[speculative] The two failures are logically independent. A memory state can satisfy decision-sufficiency (it contains the right constraints) and yet violate extraction-sufficiency (the reader cannot decode them from the rendered context). This independence is the central empirical finding that motivated ESWP: 9 out of 30 cat1 WRONG answers on LoCoMo have gold-item recall = 1.0, meaning the constraints ARE in the rendered context, but the reader fails to extract them. IWT addresses the first failure; ESWP addresses both.

[theoretical] A composite failure taxonomy therefore has four cells: (decision-sufficient, extraction-sufficient), (decision-sufficient, extraction-insufficient), (decision-insufficient, extraction-sufficient — vacuously, since the right answer cannot be extracted from wrong content), and (decision-insufficient, extraction-insufficient).

[theoretical] Practical implication: improving witness retention (IWT improvements) cannot close the reader-extraction failure bucket. A separate lever — the render operator and reader-probe validator — is required.

---

# Phase 1: Deep Anomaly Map

## A1–A8 (IWT §1 — adopted with precise citations)

[theoretical] Anomalies A1 (Rare-Critical Fact Survival), A2 (Temporal Validity vs. Storage), A3 (Consolidation as Lossy Compression), A4 (Overload Inversion), A5 (Entity Identity), A6 (Write Policy Blindness), A7 (Cross-Session Causal Dependency), and A8 (Memory Interference) are adopted from IWT Phase 1 without modification. See IWT §A1–A8 for full analysis.

## A9. Reader-Extraction Blindness

[established] Falsifiable observation: an agent can fail to answer correctly even when all decision-relevant constraints are present in the rendered context. The failure is in the reader's ability to extract an answer from the context, not in memory retention or retrieval.

[established] Empirical grounding: on the LoCoMo benchmark with the ai-knot v2 pipeline (gpt-4o-mini reader, 2-conversation subset), 9 of 30 cat1-WRONG answers have gold-item recall = 1.0 — that is, every expected answer token is present somewhere in the rendered context, yet the reader produces an incorrect answer. `research/cat1_55_investigation_20260423.md`, Part 1.

[established] Case analysis: Q15 (activities, gold: pottery/camping/painting/swimming) — rendered context contains camping but reader answers pottery/running/hiking; Q60 (instruments, gold: violin) — context contains violin but reader returns clarinet; Q23 (books, gold: two titles) — titles present in claims but reader ignores them due to surrounding noise.

[theoretical] Root cause analysis: the reader is a noisy information channel. When the rendered context is long, structurally ambiguous, or contains competing claims at the same position, the reader's effective signal-to-noise ratio drops below the threshold needed to extract the target. The failure is not random: it is systematically tied to (a) context dilution by low-relevance claims, (b) claim ordering placing the target fact after several distractors, and (c) answer format mismatch between how the fact is stored and how the question asks for it.

[theoretical] Existing systems fail because they treat the read operation as complete once a candidate set is assembled. No system in the literature (Mem0, A-MEM, LightMem, Zep/Graphiti, MemGPT) formalizes the reader as a noisy decoder with measurable extraction capacity.

[theoretical] RAG-Ragas and similar metrics measure extraction quality post hoc as an evaluation signal but do not use it as a constructive principle for pack assembly. The distinction is important: ESWP uses extraction-sufficiency as a design constraint on the pack, not as a grading rubric.

[theoretical] Root cause: no existing memory system constrains context assembly by the reader's extraction capacity. They optimize for retrieval recall (getting the right atoms into the pool) and evidence diversity (covering different action aspects), but not for reader-decodability.

[theoretical] Required science: information-theoretic coding theory (rate and capacity), psychophysics of attention and extraction in LLMs (empirical), distortion bounds under noisy decoders.

[theoretical] Rating: critical, because it caps the achievable accuracy of any memory system regardless of retrieval quality.

---

# Phase 2: Cross-Domain Archaeology

[theoretical] Three mechanisms are excavated. Mechanisms 1 (Immune Clonal Memory) and 3 (Decision-Theoretic VoI / Active Inference) are adopted from IWT Phase 2 with extensions. Mechanism 2 (Sheaf Cohomology) is adopted. A fourth mechanism — Error-Correcting Code / Channel Capacity — is introduced to address A9.

## Mechanism 0 (IWT Mechanisms 1–3 — adopted)

[theoretical] IWT Phase 2 excavates three mechanisms: Immune Danger-Gated Clonal Memory (→ regret charge), Sheaf Cohomology (→ cohomological witness class), and Decision-Theoretic VoI / Active Inference (→ regret certificate). These are adopted in full. See IWT §2.1–2.4 for each.

## Mechanism 4: Error-Correcting Code and Channel Capacity

### 2.1 Native Mechanism

[established] A channel is a conditional probability distribution P(Y | X) over output Y given input X. The capacity C = max_{P(X)} I(X; Y) is the maximum mutual information achievable over a channel.

[established] An error-correcting code encodes a message m into a codeword c = enc(m) chosen so that the Hamming distance d(c, c') between codewords exceeds the channel's expected error count. The decoder dec(y) recovers m from noisy channel output y.

[established] Landauer's principle: erasing one bit of information in a system at temperature T costs at least kB T ln 2 joules. Bennett's reversal: a computation can be made logically reversible, reducing erasure cost to the minimum thermodynamic bound.

[established] The data-processing inequality: for any Markov chain X → Y → Z, I(X; Z) ≤ I(X; Y). Post-processing a channel output cannot increase information about the source.

[established] Shannon's source-channel separation theorem: for a memoryless source and channel, the optimal strategy is to first compress the source to its entropy rate (source coding) and then apply a code at the channel capacity rate. The coding operation is separable from the content.

[theoretical] Pressure: transmit a finite-precision message through a noisy channel with minimum redundancy while guaranteeing reliable decoding.

### 2.2 Structural Analogy

[theoretical] The isomorphism: the evidence pack W_Q is a codeword; the render operator ρ is the encoding function; the reader π_R is the noisy channel; the answer y*_Q is the message; and decoding is the reader's answer generation.

[theoretical] Formal mapping:
- source message m → correct answer y*_Q
- encoder enc → render operator ρ(W_Q)
- noisy channel P(Y | X) → reader π_R(Q, context)
- channel output Y → reader's answer π_R(Q, ρ(W_Q))
- decoding dec(Y) → judge evaluation 1[correct]

[theoretical] Extraction-sufficiency is the condition that the rate I(y*_Q; π_R(Q, ρ(W_Q))) exceeds the entropy of the correct answer: the channel provides enough capacity to convey the answer.

[theoretical] Where the analogy holds: it predicts that adding irrelevant claims to the pack is equivalent to adding noise to a codeword — it reduces effective signal but does not improve it. This prediction is confirmed empirically: the LoCoMo shift A preflight showed that prepending noisy claim sheets to a context decreased, not increased, correct extraction (Q66 marshmallows, Q15 activities — `cat1_55_investigation_20260423.md` Part 5).

[theoretical] Where the analogy breaks: natural language is not a discrete alphabet; the reader is not a memoryless channel; and the "codeword" (rendered context) affects the reader's prior over possible answers in ways that depend on the full context. Nevertheless, the channel-capacity framing gives the first principled reason why pack cardinality should have a non-monotone effect on extraction quality.

[theoretical] What is lost: algebraic code construction; what is recovered: a formal reason why optimal pack size is finite and why dilution harms extraction.

### 2.3 AI Primitive Implied

[speculative] The primitive implied is the **reader-conditioned render operator ρ(W_Q, π_R)**: a deterministic function that maps a witness pack and a query to a rendered context string that is optimized for extraction by π_R.

[theoretical] The render operator has two design levers: (a) **claim ordering** — placing high-regret-charge witnesses near the beginning of the context where reader attention is strongest; (b) **answer-format alignment** — rendering each witness in the surface form that maximizes I(y*_Q; π_R(Q, ·)).

[theoretical] A toy implementation is possible today: sort witnesses by regret_charge × credence descending; format each as a factual assertion rather than an episodic transcript snippet; prefix the pack with a one-line "key facts for this query" header.

[theoretical] The extraction-sufficiency probe is a behavioral validator: given pack W_Q and query Q, run π_R once on ρ(W_Q) and score the output. If I(y*_Q; output) < threshold, the pack is extraction-insufficient — either expand it or restructure ρ.

### 2.4 Plausibility

[theoretical] Theoretical soundness: medium-high. Channel capacity is formally exact; the claim that rendered context acts as a noisy codeword is empirically supported by the shift-A preflight result.

[theoretical] Empirical testability: high. The extraction-sufficiency probe (one LLM call per pack) is directly measurable.

[theoretical] Architectural feasibility: high. The render operator is a deterministic string function; the probe is a single reader call confined to bench/, not core/.

---

# Phase 3: Unified Theory — Extraction-Sufficient Witness Program

## 3.1 Name, Postulates, and Relation to IWT

[speculative] The theory is **Extraction-Sufficient Witness Program** (ESWP).

[speculative] ESWP states that a memory system achieves reliable answer generation only if it satisfies both IWT's cohomological regret charge criterion (witness retention) AND ESWP's extraction-sufficiency criterion (pack decodability). Neither criterion alone is sufficient.

[theoretical] ESWP adopts all five IWT postulates (see IWT §3.1) and adds two:

[speculative] Postulate 6 (Extraction): A witness-pack W_Q assembled for query Q under reader π_R is valid only if I(y*_Q; π_R(Q, render(W_Q))) ≥ (1 − ε) · H(y*_Q).

[speculative] Postulate 7 (Reader-Conditioned Render): The render operator ρ is a first-class component of the memory system, not a post-hoc formatter. ρ must be designed relative to a declared reader model π_R; changing π_R requires revalidating ρ.

[theoretical] IWT reduces to ESWP as a special case when extraction-sufficiency is trivially satisfied (e.g., perfect reader with zero noise). ESWP reduces to a pure coding-theoretic problem when regret charges are uniform. The non-trivial regime is bounded readers and non-uniform regret charges.

## 3.2 The New Primitive: Extraction-Sufficient Witness Pack (ESWP)

[speculative] An extraction-sufficient witness pack for query Q under reader π_R is a triple:
W_Q = (witnesses, render_operator, extraction_invariant)

where:
- witnesses ⊆ M_T is a finite subset of the current memory
- render_operator ρ: witnesses × query → context_string is deterministic
- extraction_invariant: I(y*_Q; π_R(Q, ρ(witnesses, Q))) ≥ (1 − ε) · H(y*_Q)

[theoretical] The extraction invariant is the key new constraint. IWT's minimal sufficient set S*(Q, M_T) is chosen by counterfactual separability (omitting a witness changes the safe action). ESWP adds: the chosen set must also be extraction-sufficient — omitting an extraction-necessary claim reduces the reader's ability to decode the answer.

[theoretical] Formally, an atom ω is extraction-necessary for query Q under reader π_R if:
I(y*_Q; π_R(Q, ρ(W_Q, Q))) − I(y*_Q; π_R(Q, ρ(W_Q \ {ω}, Q))) > δ_extract

[theoretical] An atom ω can be regret-charge-protected (IWT) but extraction-dispensable: it survives in memory but does not contribute to the reader's answer. Conversely, an atom can be extraction-necessary but low regret-charge: the reader needs its surface form to decode the answer even if its absence would not change optimal policy in a rational agent.

[theoretical] The relationship is: the set of IWT-protected witnesses is not in general a subset or superset of ESWP-extraction-necessary witnesses. The pack must satisfy both.

## 3.3 The Four Operations — Extended

### WRITE (extends IWT §3.3 WRITE)

[theoretical] IWT WRITE includes a witness if its cohomological regret charge q(ω) > λ · c(ω).

[speculative] ESWP WRITE adds an expected extraction-necessity score to the write criterion:
score_write(ω) = q(ω) + α_ex · E_{Q ~ P_t}[1_{ω is extraction-necessary for Q under π_R}]

[theoretical] The extraction-necessity expectation cannot be computed exactly at write time (the future query is unknown). Level-3 approximation: proxy by the render-format compatibility of ω with the declared reader π_R's documented sensitivity profile. For LLM readers: prefer factual assertion format over transcript fragment format; prefer subject-predicate-object triples over implicit claims.

[theoretical] Practically, WRITE remains structurally identical to IWT WRITE; the addition is a render-format tag on each atom: factual_assertion, transcript_fragment, or inferred. Atoms tagged transcript_fragment get a small extraction-necessity penalty at read time.

[theoretical] WRITE failure modes: all IWT modes plus: rendering atoms in formats that are systematically extraction-insufficient for π_R.

### READ (extends IWT §3.3 READ)

[theoretical] IWT READ solves: S*(Q, M_T) = argmin_{S ⊆ M_T} [K(S) + β|S|] subject to counterfactual separability.

[speculative] ESWP READ adds extraction-sufficiency as a second constraint:
S*_ESWP(Q, M_T) = argmin_{S ⊆ M_T} [K(S) + β|S|]
subject to: (1) counterfactual separability (IWT)
           (2) I(y*_Q; π_R(Q, ρ(S, Q))) ≥ (1 − ε) · H(y*_Q) (ESWP)

[theoretical] Level-3 implementation: greedy-utility selection (existing `planner.py`) + extraction-sufficiency probe at the end. If the probe score < threshold, the planner expands the pack by adding the highest-utility atom not yet selected, and re-probes. Termination: pack is extraction-sufficient OR budget is exhausted.

[theoretical] The probe is a single reader call on the rendered pack with the actual query Q. Cost: one LLM call per pack assembly per query. This is affordable in bench/ settings; in production READ it can be batched or replaced by a lighter-weight proxy (e.g., BM25 coverage of query tokens in pack).

[theoretical] READ failure modes: all IWT modes plus: greedy utility-selection producing extraction-insufficient packs when high-utility atoms are individually useful but collectively noisy.

### CONSOLIDATION (extends IWT §3.3 CONSOLIDATION)

[speculative] ESWP CONSOLIDATION preserves the extraction-sufficiency invariant across merges. Formally: for every query Q in the declared query distribution P_T, if pre-consolidation M_T is extraction-sufficient for Q, then post-consolidation M'_T = h(M_T) must also be extraction-sufficient for Q.

[theoretical] Level-3 implementation: the existing interval-merge (`consolidate.py`) already preserves evidence pointers. ESWP adds: when merging two atoms, the merged atom inherits the render-format tag of the higher-credence constituent. If a transcript_fragment and a factual_assertion are merged, the merged atom is tagged factual_assertion (upward promotion preserves extraction quality).

[theoretical] CONSOLIDATION failure modes: all IWT modes plus: downgrading render format tags during merge (e.g., merged atom gets transcript_fragment tag when constituent was factual_assertion).

### FORGETTING (extends IWT §3.3 FORGETTING)

[speculative] ESWP FORGETTING adds an extraction-necessity term to the protection energy ODE:
dE_μ/dt = −α E_μ + β VoI_μ + χ D_μ + ζ C_μ − ψ R_μ − ξ c_μ + γ EN_μ

where EN_μ is the empirical extraction-necessity score for μ: the fraction of recent queries for which μ was extraction-necessary (measured during bench/ probe evaluations, not in core/).

[theoretical] Level-3 approximation: EN_μ is proxied by the atom's synthesis_method tag and render-format tag. Atoms tagged factual_assertion with high credence get EN_μ ≈ 0.7; transcript_fragment atoms get EN_μ ≈ 0.3; inferred atoms get EN_μ ≈ 0.1. These are initializers; the true EN_μ is updated offline from probe results.

[theoretical] FORGETTING failure modes: all IWT modes plus: forgetting extraction-necessary atoms that have low regret charge because their individual counterfactual regret contribution is small but their pack-level contribution to extraction-sufficiency is large.

## 3.4 Temporal Semantics (IWT §3.4 — unchanged)

[theoretical] Tri-temporal representation: valid time τ^v, transaction time τ^x, belief time τ^b. Adopted from IWT §3.4 unchanged. ESWP adds: the render operator must faithfully propagate temporal uncertainty (staleness hazard) into the rendered context string. An atom with τ^v = [156, ∞) but a staleness hazard must be rendered as "as of session 156, [fact] — may be outdated" rather than as a timeless assertion.

## 3.5 Identity Model (IWT §3.5 — unchanged)

[theoretical] Groupoid transport maps, orbit equivalence, holonomy detection adopted from IWT §3.5. ESWP adds: the render operator must use surface entity names that are resolution-consistent with the reader's training priors. An orbit ID ("entity:alice_chen") must be rendered as a human-readable name ("Alice Chen") in the context string.

## 3.6 Boundary Conditions

[theoretical] ESWP inherits all IWT boundary conditions (IWT §3.6).

[theoretical] Additional ESWP-specific boundary conditions:
1. Reader drift: if π_R changes (model upgrade), extraction-sufficiency invariants measured under the old reader may no longer hold. Re-probing is required.
2. Query distribution shift: extraction-necessity scores EN_μ are calibrated on P_T(Q). If the query distribution shifts significantly, re-calibration is needed.
3. Probe budget: the extraction-sufficiency probe adds one LLM call per READ call in bench/ mode. In production, the probe can be replaced by a cheaper proxy (coverage heuristic); the system is then extraction-approximately-sufficient with unknown precision.

---

# Phase 4: Mathematical Architecture

## 4.1 Extended Memory State Space

[theoretical] IWT's state space is M_B = {M = (W, Γ, I, P, E) : W ⊂ Ω, Σ_ω c(ω) ≤ B}.

[speculative] ESWP extends to M_B × Π_R, where Π_R is the space of reader model parameters:
M_B^ESWP = {(M, π_R) : M ∈ M_B, π_R ∈ Π_R}

[theoretical] The natural decision-regret pseudometric on M_B^ESWP is:
d_Π^ESWP(M, N) = sup_{Q ∈ Q} w(Q) · [D_TV(π_Q^M, π_Q^N) + λ_ex · ES(M, Q, π_R) − ES(N, Q, π_R)]

where ES(M, Q, π_R) = I(y*_Q; π_R(Q, ρ(M, Q))) / H(y*_Q) is the extraction-sufficiency score.

[theoretical] The ESWP distance is strictly larger than the IWT distance when ES differs between M and N. Two memories that are equidistant under IWT may be far apart under ESWP if one produces extraction-sufficient packs and the other does not.

## 4.2 Dynamics — Extraction-Aware Fokker-Planck

[theoretical] IWT Phase 4.2 derives a birth-death-consolidation master equation. ESWP extends the drift term:

[theoretical] Let Φ(M, t) = E_{Q ~ P_t}[−F(M, Q, π_R)] be the expected negative free energy under reader π_R, where:
F(M, Q, π_R) = E_π_R[log π_R(y | Q, ρ(M, Q))] − β · VoI(M, Q)

[speculative] The extraction-aware write dynamics become:
w^+_ω(M) ∝ exp(−ΔF(ω, M, π_R) / T_write)

where ΔF(ω, M, π_R) = F(M ∪ {ω}) − F(M) is the marginal free-energy gain of adding atom ω.

[theoretical] This ΔF-driven write differs from IWT's regret-charge threshold in two ways: (1) it uses the reader's log-likelihood, not just action regret, so it captures extraction-relevant facts even when their regret contribution is low; (2) it is dynamic relative to the current pack M_t, so write decisions are path-dependent.

## 4.3 Reader-Conditioned Rate-Distortion

[speculative] The reader-conditioned rate-distortion function is:
R_{π_R}(D) = min_{P(M|H) : E[d_act^ESWP(H, M, π_R)] ≤ D} I(H; M)

where the distortion d_act^ESWP(H, M, π_R) = regret_distortion(H, M) + λ_ex · extraction_regret(H, M, π_R).

[theoretical] Extraction regret: extraction_regret(H, M, π_R) = E_{Q ~ P}[H(y*_Q) − I(y*_Q; π_R(Q, ρ(S*(Q, M), Q)))].

[theoretical] The rate-distortion bound is tighter than IWT's bound because extraction failure is an additional source of distortion over and above retrieval failure.

## 4.4 Computational Complexity

[theoretical] ESWP READ (with extraction probe) is NP-hard for the same reason as IWT READ (set cover), plus: the extraction-sufficiency constraint is an oracle call to π_R whose complexity grows with pack cardinality. Level-3 approximation is O(k · n · C_probe) where k = iterations, n = candidates, C_probe = one LLM call per pack (not per candidate — probe is called on the final pack, not on each addition). This is O(k · n) deterministic work plus O(k) LLM calls, which is affordable.

[theoretical] The key claim is that C_probe scales with pack size, not with candidate set size. This answers the complexity attack in Phase 7: the probe is not O(n) LLM calls; it is O(1) per pack assembly.

## 4.5 Approximation Hierarchy

[theoretical] Level 0: exact ESWP — Bayes-optimal POMDP with exact VoI, exact cohomology, exact extraction-sufficiency with known reader model. Intractable.

[theoretical] Level 1: tractable approximation — typed witnesses, interval logic, identity groupoids, structural causal dependency, greedy submodular selection, extraction probe on final pack, reader model approximated as zero-shot LLM.

[theoretical] Level 2: deployable today — deterministic validators assign fields; greedy-utility planner selects pack; extraction probe called on assembled pack; render operator formats by synthesis_method tag; Landauer-bounded ODE protection energy.

[theoretical] **Level 3 (v2-deploy target)**: the following six changes to the current ai-knot v2 implementation bring it from Sprint-1-placeholder to Level 3:
1. ΔF-write scorer replacing `regret_charge = risk_severity * 1.0`
2. Sheaf-section gluing replacing greedy-utility (adds render-format partition)
3. Landauer-ODE forget replacing simple exponential decay
4. RG-flow consolidate replacing interval-merge
5. Bitemporal sheaf extending Allen relations to valid_time × transaction_time
6. Reader-probe validator added to bench/ (extraction-sufficiency probe)

---

# Phase 5: Five Falsifiable Hypotheses

## H1': Write Policy — ΔF-Driven vs. Salience-Driven

HYPOTHESIS: [speculative] A write policy based on estimated marginal free energy ΔF(ω, M_t, π_R) will retain rare-critical atoms better than the current salience-based `risk_severity * 1.0` placeholder, with equal or better extraction-sufficiency.

MECHANISM: [theoretical] The ΔF scorer treats each atom as a message that reduces reader uncertainty about future queries. Rare-critical atoms have high D_ω (danger severity) and high action-affect bits, which translate to large ΔF even when their salience at write time is low. The current placeholder conflates risk_severity with regret_charge and sets irreducibility_score = 1.0 for all atoms.

PREDICTION: [theoretical] On a 700-session rare-critical benchmark, Level-3 ΔF-write will achieve risk-weighted survival rate ≥ 0.95 at 10% budget, with ≥ 20 percentage points over the current placeholder.

BASELINE: [theoretical] Current ai-knot v2 Sprint-1 placeholder (regret_charge = risk_severity * 1.0), plus Mem0, A-MEM, LightMem, and full-context-LLM baselines.

FALSIFICATION: [theoretical] If the current placeholder matches Level-3 ΔF-write within 5 percentage points at equal budget across 3 seeds, H1' fails.

CONFIDENCE: [theoretical] Medium-high. The causal mechanism is direct; the risk is that risk_severity classification noise dominates ΔF estimation differences.

## H2': Read Policy — Sheaf-Section Gluing vs. Greedy Utility

HYPOTHESIS: [speculative] Sheaf-section gluing — partitioning candidates by (orbit, valid_interval, action_class) and selecting one section from each partition — will reduce ContextDilutionRate by ≥ 30% relative to current greedy-utility selection at equal RequiredAtomRecall.

MECHANISM: [theoretical] Greedy-utility selection may select multiple atoms from the same (orbit, action_class) partition, filling the budget with correlated witnesses and leaving extraction-necessary atoms from other partitions unselected. Sheaf-section gluing enforces partition coverage before filling individual partitions.

PREDICTION: [theoretical] ContextDilutionRate = |irrelevant atoms in pack| / |pack size| drops by ≥ 30%; RequiredAtomRecall = |gold atoms in pack| / |gold atoms| stays ≥ current level.

BASELINE: [theoretical] Current `planner.py` greedy-utility with Allen-relation temporal bonus.

FALSIFICATION: [theoretical] If ContextDilutionRate difference is < 10% across 10-conversation LOCOMO runs with 3 seeds, H2' fails.

CONFIDENCE: [theoretical] Medium. The partition structure reduces dilution in theory; in practice, orbit assignment quality determines whether the partition is meaningful.

## H3': Forgetting — Landauer-ODE vs. Simple Exponential

HYPOTHESIS: [speculative] A Landauer-bounded ODE with multi-term dynamics (VoI, danger, curvature, access rate, resource cost) will produce memory that grows sublinearly with session count while maintaining rare-critical recall ≥ 0.95.

MECHANISM: [theoretical] The simple exponential `E(t) = E0 * exp(-k * t)` in the current `forget.py` decays all atoms at a rate determined only by risk_severity. It has no Landauer floor (minimum energy required to erase = kB T ln 2) and no access-rate boost term. High-danger, frequently accessed atoms will be over-forgotten by the simple ODE; the Landauer floor prevents this.

PREDICTION: [theoretical] At 700 sessions, |M_T| / T is sublinear (< α * T^{0.5}) for Level-3 FORGET, while the simple exponential shows linear growth or linear drop (either hoarding or excessive pruning).

BASELINE: [theoretical] Current `forget.py` simple exponential.

FALSIFICATION: [theoretical] If simple exponential achieves the same sublinear growth rate with equivalent rare-critical recall on 3 seeds, H3' fails.

CONFIDENCE: [theoretical] Medium. Sublinear memory growth is a clean falsification target, but the Landauer floor effect may be small unless the model runs very long (700+ sessions).

## H4': Temporal — Bitemporal Sheaf vs. Valid-Time Only

HYPOTHESIS: [speculative] Bitemporal sheaf-gluing (valid_time × transaction_time grid) will reduce wrong-current-state answers by ≥ 15 percentage points on temporal update questions compared to valid-time-only representation.

MECHANISM: [theoretical] The current `temporal.py` and `atom.py` store valid_from/valid_until (valid-time) and observation_time (transaction-time), but READ planning uses only Allen relations on valid-time. Bitemporal semantics distinguishes "when was this true in the world" from "when did we learn it" — critical for stale knowledge and delayed updates. A fact learned in session 12 that was true from session 5 to session 20 requires both axes to reason correctly about whether it was "current" at session 15 (when the reader didn't yet know it) vs. session 22 (when the reader knows the interval has ended).

PREDICTION: [theoretical] On LongMemEval temporal reasoning + LoCoMo update questions, ESWP bitemporal achieves ≥ 15pp gain over valid-time-only on "what is X's current [state]" questions.

BASELINE: [theoretical] Current v2 valid-time-only planning; timestamp-only baselines (Mem0, A-MEM).

FALSIFICATION: [theoretical] If valid-time-only achieves the same score within 5pp on temporal update questions across 3 seeds, H4' fails.

CONFIDENCE: [theoretical] High. Temporal database theory directly predicts the failure of single-time representations; the mechanism is well-established.

## H5': Reader-Extraction Probe — Closing the LLM-Fail Bucket

HYPOTHESIS: [speculative] The extraction-sufficiency probe (one reader call on the assembled pack) will close ≥ 60% of the LLM-fail bucket on LoCoMo cat1 — specifically, the 9 questions where gold-item recall = 1.0 but the reader fails to extract the answer.

MECHANISM: [theoretical] When the probe detects extraction-insufficiency (I < threshold), the pack is expanded or restructured by the reader-probe validator. The validator identifies which atoms are extraction-necessary (marginal information gain to the reader) and ensures they are placed prominently in the rendered context. The 9/30 LLM-fail cases are expected to partially close because (a) ~5 involve context dilution fixable by reordering, (b) ~2 involve format mismatch (transcript vs. assertion) fixable by render-format promotion, and (c) ~2 are fundamental semantic gaps not fixable by pack adjustment.

PREDICTION: [theoretical] Among the 9 LLM-fail cat1-WRONG questions in the LoCoMo 2-conv run, ≥ 5 become CORRECT after extraction-probe-guided pack adjustment. Expected LOCOMO cat1 improvement: +5 questions → 18/43 = 41.9% (up from 30.2% baseline).

BASELINE: [theoretical] Current baseline cat1 = 30.2% (13/43) on gpt-4o-mini reader, p1-1b-2conv.

FALSIFICATION: [theoretical] If the probe closes fewer than 3 of the 9 LLM-fail questions, H5' fails. If adding the probe degrades non-LLM-fail questions, H5' is also partially falsified.

CONFIDENCE: [theoretical] Medium. The mechanism is direct but depends on the probe correctly identifying which atoms are extraction-necessary. If the probe simply returns "not sufficient" without actionable guidance, its corrective power is limited.

---

# Phase 6: Experimental Program

## 6.1 Benchmark Gap Analysis

[established] LoCoMo (LoCoMo, Maharana et al. 2024) tests multi-session dialogue understanding over 300 turns and 9K tokens, with temporal, causal, and cross-session recall questions.

[theoretical] LoCoMo blind spot for ESWP: it measures answer correctness but not extraction-sufficiency as a separate dimension. A clean ESWP extension of LoCoMo would separately score (a) "gold in context" and (b) "reader correctly extracted gold", enabling isolation of the A9 Reader-Extraction Blindness bucket.

[established] LongMemEval (Wu et al. 2024) tests five memory abilities including temporal reasoning and knowledge updates.

[theoretical] LongMemEval blind spot for ESWP: no extraction-sufficiency scoring; temporal update questions do not distinguish valid-time vs. transaction-time confusion.

[established] BEAM (BEAM, 2025) scales to 10M tokens with validated questions.

[theoretical] BEAM blind spot: does not distinguish retrieval failure from extraction failure; does not measure render-format effects.

[established] MemoryArena (2026) tests interdependent multi-session agentic tasks.

[theoretical] MemoryArena ESWP extension: score each task both on final outcome AND on whether the required preconditions were in the assembled pack (pack coverage) AND whether the reader extracted them (extraction rate).

[established] MemGround (2026) tests interactive dynamic state tracking.

[theoretical] MemGround ESWP extension: require agent to output the minimal support pack for each action; score extraction sufficiency of the pack.

## 6.2 New Benchmark: Counterfactual Continuity Benchmark (CCB)

[speculative] CCB is specified in full in `research/ccb_benchmark_design.md` (Deliverable 5). Summary here.

[theoretical] CCB contains 200–1000 synthetic agent histories, each 100–700 sessions, across four domains: medical-logistics, scheduling-commitment, preference-identity, and financial-legal. Ground truth is established by a counterfactual intervention: a specific early witness is changed, and the correct late answers change correspondingly. A system passes CCB if it correctly propagates the counterfactual intervention.

[theoretical] CCB adds two extraction-specific metrics absent from all existing benchmarks:
- **Extraction-Sufficiency Rate (ESR)**: fraction of answered questions where the gold atom was both in the assembled pack and correctly extracted by the reader.
- **Risk-Weighted Counterfactual Accuracy (RWCA)**: fraction of interventional questions answered correctly, weighted by the risk_severity of the changed witness.

[theoretical] CCB is unfakable by full-context LLMs because the counterfactual intervention creates two near-identical histories that differ in one early witness; a system without witness-level indexing cannot correctly track which history it is in.

## 6.3 Three Core Experiments

### Experiment 1: Rare-Critical Survival (H1' test)

[theoretical] System: ai-knot v2 Level-3 (ΔF-write + energy-forget + dependency closure).
Dataset: CCB-RareCritical (4 domains × 5 seeds × 100-session histories, 1 rare-critical witness per 20 sessions).
Metric: RWCA + Extraction-Sufficiency Rate (ESR).
Baselines: v2 Sprint-1 placeholder, Mem0, A-MEM, LightMem, full-context LLM (truncated at 128K).
Ablations: no ΔF (placeholder), no Landauer floor, no dependency boundary, no render-format tag.
Expected: Level-3 RWCA ≥ 0.90; baselines < 0.70; ablation without ΔF drops ≥ 10pp.
Failure mode: risk classification noise dominates; detect by confusion matrix on risk_class.

### Experiment 2: Extraction-Sufficiency Probe (H5' test)

[theoretical] System: ai-knot v2 with extraction-probe validator in bench/ (reader-probe on assembled pack, no core changes).
Dataset: LoCoMo 2-conv, cat1 LLM-fail bucket (9 questions identified in `cat1_55_investigation_20260423.md`).
Metric: LOCOMO cat1 accuracy; LLM-fail bucket closure rate.
Ablations: no pack expansion after probe, no render-format reordering.
Expected: cat1 ≥ 41.9% (+5 of 9 LLM-fail questions); ESR for the 9 questions ≥ 0.55.
Failure mode: probe detects insufficiency but expansion harms other questions; detect by full cat1–cat5 scorecard gate.

### Experiment 3: Bitemporal Sheaf (H4' test)

[theoretical] System: ai-knot v2 with bitemporal extension to `core/temporal.py` and `ops/planner.py`.
Dataset: LongMemEval temporal update questions + LoCoMo temporal reasoning subset.
Metric: wrong-current-state answer rate, H4' prediction ≥ 15pp improvement.
Baselines: current valid-time-only v2; Zep/Graphiti (temporal graph baseline).
Ablations: valid-time-only, transaction-time-only, no staleness hazard.
Expected: bitemporal sheaf reduces wrong-current-state by ≥ 15pp on LongMemEval temporal.
Failure mode: most errors are semantic, not temporal-representation; detect by error type annotation.

## 6.4 Landmark Result

[speculative] The landmark result is a **rare-critical survival curve**: at 10% memory budget across 700-session CCB histories, Level-3 ESWP maintains RWCA ≥ 0.90 while all eight baselines (Mem0, A-MEM, LightMem, Zep/Graphiti, MemGPT, full-context-128K, v2-Sprint1-placeholder, recency-only) fall below RWCA = 0.70, with the gap widening as history length increases.

[theoretical] The phenomenon claimed: existing systems show memory overload inversion after a critical history length (IWT anomaly A4). ESWP Level-3 does not because ΔF-write protects rare-critical witnesses regardless of history depth, and the Landauer ODE provides sublinear memory growth.

[theoretical] Single headline number: ΔRWCA ≥ 0.20 at B/|H| = 0.10, 5 seeds, paired t-test p < 0.01.

---

# Phase 7: Hostile Tribunal

## Scientist B: 7.1 The Kill Shot on ESWP

[theoretical] The single fragile assumption of ESWP is that a reader-probe can correctly identify extraction-sufficient packs. If the probe is unreliable — returns false positive "sufficient" for packs where the reader will fail in deployment — then the extraction-sufficiency criterion is unenforceable and ESWP collapses to IWT.

[theoretical] The collapse is exact: if the probe has no power to distinguish extraction-sufficient from extraction-insufficient packs, the extraction-invariant condition becomes unverifiable, and ESWP degenerates to IWT with an extra LLM call.

## Scientist A Response

[theoretical] Concede: the probe's power is bounded by the alignment between the probe query and the deployment query. If probe queries are systematically simpler than deployment queries, the probe overestimates extraction sufficiency.

[speculative] Revise: ESWP's empirical target is specifically the identified 9/30 LLM-fail bucket in LoCoMo cat1. These are NOT hypothetical failures; they are documented cases where gold is in context and the reader fails. The probe is calibrated on these cases. A probe that fails to close any of the 9 identified cases is empirically falsified and the claim is retracted.

[theoretical] The theory survives partial probe failure: even if H5' closes only 3/9 (33%), the extraction-sufficiency lens explains 3 previously unexplained failures, which is scientifically informative.

## Scientist B: 7.2 The "Reader-Probe is LLM-in-Core" Attack

[theoretical] The probe calls π_R (a language model) to evaluate extraction sufficiency. This violates the ai-knot v2 architectural invariant: no LLM calls in core/ ops/ store/ api/ (see `src/ai_knot_v2/CLAUDE.md`).

## Scientist A Response

[theoretical] The reader-probe is NOT in core/. It is in bench/ exclusively. The extraction-sufficiency probe validates the assembled pack AFTER plan_evidence_pack() returns; it does not modify the pack assembly logic in ops/planner.py. The core remains LLM-free.

[theoretical] Formally: `bench/ccb/probe.py:validate_extraction_sufficiency(pack, query, reader)` calls π_R once. `ops/planner.py:plan_evidence_pack()` calls no LLM. The probe result can inform offline training of the render operator (adjusting render-format tags) but does not enter the real-time memory path.

[theoretical] This separation is deliberate: the architectural invariant exists to prevent memory correctness from depending on LLM reliability. The probe is an offline evaluation and improvement signal, not a correctness requirement at runtime.

## Scientist B: 7.3 Prior Art Attack

[established] RAG evaluation metrics (Ragas Context-Precision, Context-Recall) measure whether the retrieved context contains the answer.

[established] LLM-as-judge (Zheng et al. 2023) uses a separate LLM to evaluate answer quality.

[established] Self-RAG (Asai et al. 2023) uses reflection tokens to assess generation quality at inference time.

[theoretical] These approaches evaluate extraction quality post hoc. ESWP is therefore not the first system to measure whether a reader can extract an answer from a context.

## Scientist A Response

[theoretical] Concede: post-hoc measurement of extraction quality exists (Ragas, LLM-as-judge, Self-RAG).

[speculative] The novelty of ESWP is threefold: (1) extraction-sufficiency as a **design constraint on pack assembly**, not a grading rubric applied after generation; (2) formal definition of extraction-sufficiency as a mutual information condition I(y*_Q; π_R(Q, render(W_Q))) ≥ (1−ε)·H(y*_Q), which is distinct from coverage metrics (coverage measures presence, not decodability); (3) the empirical finding that 9/30 of a specific benchmark's wrong answers are extraction-failures that cannot be fixed by better retrieval — a phenomenon that Ragas/LLM-as-judge measure but do not explain or address architecturally.

## Scientist B: 7.4 Complexity Attack

[theoretical] The extraction-probe adds one LLM call per READ operation. At scale (millions of reads per day in production), this is prohibitively expensive.

## Scientist A Response

[theoretical] Concede: one reader call per production READ is too expensive at scale.

[theoretical] ESWP Level-3 is not a production serving system; it is a research prototype for 200–1000 session benchmark evaluation. At this scale, one LLM call per query assembly is affordable (~$0.001 per call × 233 LoCoMo questions × 10 conversations = ~$2.33 for a full run).

[theoretical] For production: the probe trains a lightweight extraction-sufficiency scorer (fine-tuned embedding or rule-based proxy) that replaces the LLM call at serving time. The probe provides training signal; the proxy provides runtime enforcement.

## Scientist B: 7.5 CCB Constructedness Attack

[theoretical] CCB is a synthetic benchmark constructed to favor systems that implement witness reasoning. A system could game CCB by implementing explicit risk templates and domain rules without IWT/ESWP theory.

## Scientist A Response

[theoretical] Concede: first-version CCB can be gamed by domain-specific rule tables.

[theoretical] Revise CCB (already specified in Deliverable 5 `research/ccb_benchmark_design.md`) to include: (a) held-out domains (the system has not seen training examples from these domains); (b) adversarial paraphrases (the counterfactual intervention uses different surface forms each seed); (c) scoring of minimal support pack (the system must output not just the answer but the pack of witnesses supporting it, which is scored for extraction-sufficiency). A system that passes by implementing explicit rules has in practice implemented Level-3 ESWP, which is acceptable.

---

# Phase 8: Research Roadmap

## 8.1 Minimal Publishable Unit (MPU)

[theoretical] MPU: ESWP Level-3 prototype + Experiment 2 (extraction-probe on LoCoMo LLM-fail bucket) + 200-session CCB-RareCritical variant.

[theoretical] Requirements: (a) extraction-probe validator in bench/ (≤150 LoC); (b) render-format tag on MemoryAtom synthesis_method (existing field reused — no schema change); (c) 200-history CCB-RareCritical generator (≤300 LoC, deterministic seed).

[theoretical] Result sufficient for a workshop paper: probe closes ≥ 3/9 LLM-fail questions + CCB-RareCritical RWCA ≥ 0.80 for Level-3 vs. < 0.60 for all baselines. Target venue: MemAgents @ ICLR, NeurIPS Memory workshop, or similar.

[theoretical] Total new LoC for MPU: ~450 (probe: 150, render reordering: 100, CCB generator: 200, evaluation scaffolding: ~100 spread across existing bench/).

## 8.2 Twelve-Month Program

[theoretical] **Months 1–2 (Foundations + MPU):** Formal ESWP definitions (this document), IWT-v2 audit (`research/iwt_v2_implementation_audit_20260424.md`), CCB spec (`research/ccb_benchmark_design.md`). Risk: audit reveals more Sprint-1 shortcuts than planned. Fallback: restrict MPU to extraction probe only.

[theoretical] **Months 3–4 (Level-3 Step 1–2):** Reader-probe validator (bench/ only, no core changes) + ΔF-write scorer (replacing regret_charge placeholder in atomizer.py). Gate: LOCOMO 2-conv cat1 ≥ 37%, all scorecard gates pass. Risk: ΔF scorer requires action-class probability distribution not available deterministically. Fallback: use risk_class × action_affect_mask as ΔF proxy.

[theoretical] **Months 5–6 (Level-3 Steps 3–4):** Sheaf-section gluing in planner.py + Landauer-ODE forget. Gate: LOCOMO 2-conv cat1 ≥ 41%, ContextDilutionRate ↓ ≥ 15%. Risk: sheaf partition on orbit creates too many singleton sections, reverting to single-atom packs. Fallback: only use action_class axis for partitioning (not orbit).

[theoretical] **Months 7–8 (CCB prototype + Experiment 1):** Build CCB generator + scorer; run Experiment 1 (RWCA vs. 8 baselines). Gate: CCB-RareCritical RWCA ≥ 0.80 for Level-3 vs. < 0.60 baselines. Risk: CCB generator produces unnatural histories. Fallback: use hand-crafted seed scenarios reviewed by authors.

[theoretical] **Months 9–10 (Full experiments + Level-3 Steps 5–6):** Bitemporal sheaf (H4') + RG-flow consolidate. Experiment 2 (extraction probe on LoCoMo LLM-fail bucket). Experiment 3 (bitemporal on LongMemEval temporal). Gate: GATE-F1 pass (LOCOMO 10-conv cat1 ≥ 45% — revised from 55% given empirical ceiling ≈ 40–44% from `cat1_55_investigation_20260423.md`). Risk: 10-conv run is expensive; multi-metric gate sensitive to noise. Fallback: report 2-conv averaged × 3 seeds as proxy.

[theoretical] **Months 11–12 (Paper + rebuttal pack):** Write paper, produce rebuttal pack, release code artifacts. Revised cat1 target: 41–46% (from extraction probe + ΔF-write combined), not 55% (empirically unreachable at retrieval+materializer level). Report extraction-sufficiency gain as separate contribution from retrieval gain.

## 8.3 Five-Year Vision

[speculative] If ESWP works, agent memory research gains a second axis of evaluation: not just "did the system retrieve the right atoms?" but "was the assembled pack decodable by the reader?"

[speculative] The five-year vision: memory systems ship with reader-model compatibility certificates, analogous to type signatures. A memory optimized for GPT-4o-mini has different render operators than one optimized for Claude Sonnet 4; swapping models requires re-probing extraction sufficiency.

[speculative] Reader-aligned memory becomes a product differentiator: "this assistant's memory is certified extraction-sufficient for [model class] on [domain]", measurable and verifiable, not just a benchmark claim.

[theoretical] The technical trajectory: Level-3 prototype (2026) → production proxy scorer replacing probe LLM call (2027) → reader-model family adapters for render operators (2028) → multi-reader ensemble memory (2029, memory shared across models with compatible sufficiency profiles) → full ESWP production system (2030).

## 8.4 Dead-End Protocol

[theoretical] If H5' fails completely (probe closes 0/9 LLM-fail questions across 3 seeds): concede that extraction-sufficiency is not improvable at the pack-assembly level with current reader models. Pivot to: (a) reader fine-tuning for better extraction from IWT-style context (different lever, not memory system); (b) publish CCB + the extraction-failure taxonomy (9 buckets) as standalone contribution; (c) retain all IWT Level-3 improvements (ΔF-write, sheaf read, Landauer forget) as independent contributions.

[theoretical] If H1' and H2' both fail (ΔF-write and sheaf read show no improvement over Sprint-1 placeholder): concede that risk_class-based approximation of ΔF is insufficient; fallback to Memory Substrate v1 multi-projection architecture (see `research/memory_substrate_v1_article_and_plan_20260424.md`) as the engineering direction while retaining IWT/ESWP as theoretical framework.

[theoretical] Surviving sub-hypotheses in either dead-end: H4' (bitemporal) is relatively independent of H1'/H2'/H5' and may still hold. H3' (Landauer ODE sublinear memory) is independent of reader-side changes and may provide a useful contribution on memory budget efficiency.

---

# Final Theoretical Commitments

[theoretical] We know that 9/30 cat1 WRONG answers on LoCoMo have the gold in the rendered context. The reader fails. This is not a retrieval problem.

[theoretical] We do not know whether extraction-sufficiency can be improved by pack restructuring without degrading other questions.

[theoretical] It is not unknowable: the probe directly measures I(y*_Q; π_R(Q, ρ(W_Q))) on the identified failure cases. Run the experiment.

[speculative] The bet is that correct artificial memory requires two things that are currently conflated: protecting the right constraints from forgetting (IWT), and assembling them in a form that the reader can decode (ESWP). Both are necessary; neither alone is sufficient.

[theoretical] IWT corrects memory storage. ESWP corrects memory delivery. The field has studied the first; it has ignored the second. The 9/30 failure cases are the evidence.
