"""Post-hoc rank diagnostics over frozen recordings. No provider or SQL access."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import ai_knot
from ai_knot.multi_agent.assembly import CoverageAwareAssembler
from ai_knot.multi_agent.models import CandidateFact
from ai_knot.types import Fact


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose(run_dir, cases_path):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert digest(cases_path) == manifest["cases_sha256"], "case freeze mismatch"
    repo = Path(ai_knot.__file__).resolve().parents[2]
    for name, expected in manifest["source_freeze"]["files"].items():
        assert digest(repo / name) == expected, "runtime source freeze mismatch"
    limits = manifest["context_contract"]["semantic"]
    max_facts = limits["facts"]
    max_chars = limits["rendered_evidence_chars"]
    data = json.loads(cases_path.read_text())
    rows = []
    for case in data["cases"]:
        full = json.loads((run_dir / case["case_id"] / "full/report.json").read_text())
        semantic = json.loads((run_dir / case["case_id"] / "semantic/report.json").read_text())
        for index, stage in enumerate(case["stages"]):
            actual = semantic["stages"][index]
            attempt = actual["attempts"][-1]
            source = full["stages"][index]["attempts"][-1]["review"]["accepted"]
            vectors = {}
            for call in semantic["calls"]:
                if call["call_id"] >= attempt["completion_call_id"]:
                    break
                if call["kind"] == "embed":
                    assert call["status"] == "ok"
                    for text, vector in zip(call["texts"], call["vectors"], strict=True):
                        norm = math.hypot(*vector)
                        vectors[text] = [v / norm for v in vector]
            requirements = [stage["request"], "\n".join(case["application_policy"]["rules"])]
            facets = {f"q{i}": [] for i in range(len(requirements))}
            candidates = {}
            for item in source:
                identity = json.dumps(
                    [item["ref"]["namespace"], item["ref"]["fact_id"]],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                candidate = CandidateFact(Fact(item["content"], id=identity))
                for i, requirement in enumerate(requirements):
                    score = sum(
                        q * v
                        for q, v in zip(vectors[requirement], vectors[item["content"]], strict=True)
                    )
                    if score > 0:
                        candidate.base_score = max(candidate.base_score, score)
                        candidate.facet_scores[f"q{i}"] = score
                        facets[f"q{i}"].append(candidate)
                candidates[identity] = candidate
            result = CoverageAwareAssembler().assemble(
                candidates_by_facet=facets,
                top_k=max_facts,
                max_chars=max_chars,
                n_publishers=1,
                min_backfill_score_frac=0.25,
            )
            predicted = [json.loads(c.fact.id)[1] for c in result.selected]
            assert predicted == attempt["selected_source_ids"], "selection reproduction mismatch"
            selected = set(predicted)
            required = set(actual["metrics"]["required_source_ids"])
            ranking = sorted(candidates.values(), key=lambda c: (-c.final_score, c.fact.id))
            rows.append(
                {
                    "case_id": case["case_id"],
                    "stage_id": stage["stage_id"],
                    "exact_selection_reproduced": True,
                    "admissible_count": len(source),
                    "all_positive_facet_count": sum(len(f) == len(source) for f in facets.values()),
                    "facet_count": len(facets),
                    "covered_facets": sorted(result.covered_facets),
                    "selected_count": len(selected),
                    "selected_chars": sum(len(c.fact.content) for c in result.selected),
                    "required_count": len(required),
                    "required_chars": sum(
                        len(c.fact.content)
                        for c in candidates.values()
                        if json.loads(c.fact.id)[1] in required
                    ),
                    "decision_correct": actual["metrics"]["decision_correct"],
                    "required_sources_present": actual["metrics"]["required_sources_present"],
                    "ranked_candidates": [
                        {
                            "source_id": json.loads(c.fact.id)[1],
                            "rank": rank,
                            "score": c.final_score,
                            "facet_scores": c.facet_scores,
                            "chars": len(c.fact.content),
                            "required_by_frozen_gold": json.loads(c.fact.id)[1] in required,
                            "selected": json.loads(c.fact.id)[1] in selected,
                        }
                        for rank, c in enumerate(ranking, 1)
                    ],
                }
            )
    return {
        "kind": "post_hoc_diagnosis_only",
        "quality_rescored": False,
        "provider_calls": 0,
        "case_sha256": manifest["cases_sha256"],
        "rows": rows,
        "aggregate": {
            "stages": len(rows),
            "all_facets_positive_for_every_source": sum(
                r["all_positive_facet_count"] == r["facet_count"] for r in rows
            ),
            "all_facets_reported_covered": sum(
                len(r["covered_facets"]) == r["facet_count"] for r in rows
            ),
            "missing_required_despite_covered_facets": sum(
                not r["required_sources_present"] and len(r["covered_facets"]) == r["facet_count"]
                for r in rows
            ),
            "gold_required_set_fits_budget": sum(
                r["required_count"] <= max_facts and r["required_chars"] <= max_chars for r in rows
            ),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("cases_path", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = diagnose(args.run_dir, args.cases_path)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["aggregate"], indent=2))
