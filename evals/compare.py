"""
Regression test: is the new version better, worse, or statistically similar?

  python -m evals.compare v1 v2                 # offline results
  python -m evals.compare v1 v2 --mode live
"""
import argparse
import json
import sys

from evals.run_eval import result_path
from evals.stats import aggregate, paired_compare


def load(config: str, mode: str) -> list[dict]:
    with open(result_path(config, mode)) as f:
        return json.load(f)["rows"]


def release_gate(old: list[dict], new: list[dict]) -> dict:
    overall = paired_compare(old, new)
    status, reasons = "PASS", []

    # 1) Hard gate: any critical safety failure blocks, regardless of averages.
    crit = [r["case"]["id"] for r in new if r["diagnosis"] == "safety_violation"]
    if crit:
        status = "BLOCK"
        reasons.append(f"critical safety failures: {crit}")

    # 2) Overall statistically worse.
    if overall["verdict"] == "WORSE":
        status = "BLOCK"
        reasons.append("overall task_success is significantly worse")

    # 3) Per-category slices: overall gains often hide subgroup regressions.
    cats = {}
    for c in sorted({r["case"]["category"] for r in old}):
        res = paired_compare([r for r in old if r["case"]["category"] == c],
                             [r for r in new if r["case"]["category"] == c])
        cats[c] = res
        if res["verdict"] == "WORSE":
            status = "BLOCK"
            reasons.append(f"category '{c}' is significantly worse")
        elif res["broken"] and status == "PASS":
            status = "WARN"
            reasons.append(f"category '{c}' broke cases that passed before: {res['broken']}")

    # 4) Cost / latency guardrails.
    a_old, a_new = aggregate(old), aggregate(new)
    if a_new["avg_tokens"] > 1.3 * a_old["avg_tokens"] and status == "PASS":
        status = "WARN"
        reasons.append("average tokens per task up more than 30%")
    return {"status": status, "reasons": reasons, "overall": overall, "by_category": cats}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--mode", default="offline", choices=["offline", "live"])
    args = p.parse_args()

    old, new = load(args.old, args.mode), load(args.new, args.mode)
    g = release_gate(old, new)
    o = g["overall"]
    print(f"\n{args.old}  ->  {args.new}   (mode={args.mode}, n={o['n']})")
    print(f"OVERALL task_success: {o['verdict']}   mean_diff={o['mean_diff']:+.3f}   "
          f"95% CI=({o['ci95'][0]:+.3f}, {o['ci95'][1]:+.3f})")
    print(f"  fixed : {o['fixed']}")
    print(f"  broken: {o['broken']}")
    print("\nBy category:")
    for c, r in g["by_category"].items():
        print(f"  {c:20s} {r['verdict']:8s} diff={r['mean_diff']:+.2f}  n={r['n']:2d}  "
              f"fixed={len(r['fixed'])} broken={len(r['broken'])}")
    print(f"\nRELEASE GATE: {g['status']}")
    for reason in g["reasons"]:
        print(f"  - {reason}")
    sys.exit(1 if g["status"] == "BLOCK" else 0)   # non-zero exit lets CI fail the build


if __name__ == "__main__":
    main()
