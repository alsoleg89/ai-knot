# Bounded evidence selection: relevance, coverage and prerequisites

Research date: 2026-10-29. Read against `ece236a`, the current semantic selector and assembler, and the [frozen growing-context result](../growing-context-evaluation.md). This note proposes hypotheses and evaluation boundaries; it contains no new model experiment or product implementation.

## What the project evidence establishes

The retained ten-task/two-stage study supplies every required source in 20/20 full-reference contexts, versus 5/20 for both bounded selectors. Correct decisions are 17/20 and 15/20 respectively. Complete-evidence-and-correct-decision success is 17/20 versus 4/20. The stronger reference has a larger context allowance; this is not a universal comparison of retrieval algorithms. Still, answer accuracy alone conceals omitted prerequisites. The frozen fixtures and outputs must stay unchanged.

`SemanticContext` creates a facet for each supplied requirement, puts every source with positive cosine in that facet and records that cosine as its facet score. `CoverageAwareAssembler` treats every positive facet score on a selected candidate as covered. It chooses by global `final_score`, then backfills by that score after facets appear covered. With broadly positive similarities, one source can mark every query facet covered. This is coverage of a similarity-based map, not evidence that all task conditions were supplied. The existing source admission and task guard correctly solve different problems: admissibility and unchanged source membership/revisions.

## What the mathematical approaches offer

**Budgeted maximum coverage** selects sets with costs to maximize the weight of covered elements under a budget. The problem is NP-hard; Khuller, Moss and Naor give a particular approximation algorithm. For this product, sources can be sets, explicit condition labels their elements and rendered length their cost. The theorem does not apply automatically to the current heuristic, simultaneous fact/character/diversity constraints, or a learned and fallible source-to-condition map. [Original paper and publisher abstract](https://www.sciencedirect.com/science/article/pii/S0020019099000319/pdf?md5=2675ded72b99acbef7c2723141cefb31&pid=1-s2.0-S0020019099000319-main.pdf).

**Length-constrained submodular selection** optimizes a set objective under a knapsack constraint. Lin and Bilmes separate representation and diversity, and discuss diminishing returns and monotonicity conditions behind greedy guarantees. A useful candidate proxy is `sum_j w_j * min(1, sum_{i in S} a_ij)`, with nonnegative contribution `a_ij`: another paraphrase of an already represented condition adds less value. It requires meaningful condition dimensions. Optimizing this proxy well cannot repair a wrong condition map. [Original ACL paper, sections 2–3](https://aclanthology.org/P11-1052.pdf).

**MMR** trades query relevance against similarity to already selected passages. It addresses repeated topical material. The original paper also identifies extraction losing referential context, with larger passages as one possible remedy. MMR is a reasonable duplicate-control baseline; it does not identify a missing prerequisite. Inference for ai-knot: permission and its narrowly worded exception can be highly similar while both are necessary, so similarity penalties must not suppress an explicitly required companion. [Original Goldstein–Carbonell paper](https://aclanthology.org/X98-1025.pdf).

**Robust submodular selection** optimizes against several objectives rather than only their average. Krause et al.'s SATURATE algorithm truncates objectives and seeks a set satisfying all thresholds. Its stated strong guarantee allows a logarithmic increase in selected set size; it must not be advertised under ai-knot's unchanged hard eight-fact ceiling. A worst-covered-condition diagnostic is useful without importing the full algorithm or claiming that guarantee. [Original JMLR paper, section 4 and theorem 5](https://www.jmlr.org/papers/volume9/krause08b/krause08b.pdf).

## The key mismatch: complementary conditions

Own mathematical observation: if both permission `p` and exception assessment `e` are necessary, a success indicator has `f({}) = f({p}) = f({e}) = 0`, but `f({p,e}) = 1`. Adding `e` has greater marginal value after `p` than before it. That violates diminishing returns. All-prerequisite decision correctness is not generally a monotone submodular coverage objective, even if a surrogate evidence count is. Average source recall, aggregate cosine and publisher diversity can all look good while the one missing condition invalidates the decision.

There are three separate tasks:

1. Discover which conditions apply to the current decision, including generic rules, temporal exceptions and negative evidence.
2. Select observations representing them within the context budget.
3. Reason correctly from those observations.

Combinatorial packing primarily addresses step 2. Automatic step 1 remains the difficult semantic problem. Google's sufficient-context work independently studies the distinction between retrieval insufficiency and a reader failing despite sufficient context. Its autorater and selective-generation results support measuring that distinction, not importing its accuracy into ai-knot. Its extra model calls and learned decision rule would need their own budget and validation. [Original paper](https://arxiv.org/html/2411.06037v2), [first-party publication record](https://research.google/pubs/sufficient-context-a-new-lens-on-retrieval-augmented-generation-systems/).

## Lightweight candidates to evaluate

**Start with diagnostics.** For each frozen miss, retain admitted candidates, per-requirement scores, lengths, rank, packing decisions and omitted IDs. Separate source admission loss, ranking loss, an otherwise useful source failing to fit, and reader failure with sufficient supplied sources. Preserve exact shared/owner identities and output order. This diagnosis uses recorded vectors and no extra provider calls; it cannot become an additional independent quality trial.

**Try improved packing as a hypothesis.** Marginal coverage gain per rendered character, with explicit zero-gain handling and deterministic ties, can reduce repeated spending on one broad topic. Compare against score order and MMR using new independent cases before changing defaults. A positive cosine threshold is not calibrated semantic entailment; choosing a threshold from the already observed gold would be tuning, not validation. Keeping separate host-authored requirements can help only when those requirements actually distinguish conditions.

**Keep host-explicit obligations separate from relevance.** When a trusted host already knows mandatory exact sources or condition bundles, it can reserve those whole observations first, then use relevance for optional evidence. Review every ref under current ACL/lineage/freshness and fail explicitly if mandatory observations are unavailable or cannot fit. Never label the resulting context universally complete: the host's obligation list itself can be incomplete. An explicit include set is not automatic prerequisite discovery and must not be populated from evaluation gold.

**Preserve declared dependencies atomically if introduced.** A selected observation that only makes sense with a definition, exception or companion should travel with that declared bundle, or be omitted/reported unavailable as a whole. This needs honest source-authored/host-authored dependency semantics and bounded cycle/closure behavior. Existing provenance roots establish grounding, not necessarily the reader's complete explanatory dependency set. Do not silently reinterpret every source ref or every generic fact as mandatory context.

Generic rules are especially important for long-lived teams, but automatically retaining all generic records merely moves unbounded growth into the mandatory set. Old, unrelated or duplicative generic records must still pass applicability/current-slot checks. Every selected companion must respect the same namespace and source authority; a bundle is no ACL bypass. Capture the task guard before acquisition and recheck at checkpoint/action admission so a good selection cannot authorize a decision against an outdated universe.

## Acceptance before adoption

Use fresh independent tasks with new names/domains, multiple simultaneously applicable conditions, similar exceptions, conflicting observations, temporal changes, deletion/revocation, unrelated generic accumulation and real budget-infeasible mandatory sets. Keep a full-context control, lexical control and equal offered total provider budgets. Report decision accuracy, complete-required-set rate, mean recall, both-stage success, false abstention, requests/tokens and context-preparation cost separately. A larger-context reference stays labelled as such. Unknown/overflow is retained, not scored as a correct action. Existing frozen scores remain historical evidence; a replay is backend parity, not another live sample.

Recommendation: diagnose the recorded loss before changing ranking; then evaluate a small bounded packing or host-explicit obligation path. Do not add a graph service, default model judge or learned scheduler merely because submodular or sufficiency terminology sounds stronger. No source here proves that MMR, a better greedy rule or explicit bundles will improve ai-knot's task accuracy.
