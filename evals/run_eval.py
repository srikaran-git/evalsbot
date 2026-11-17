"""
Run an evaluation.

  python -m evals.run_eval --config v1                       # offline, no API key needed
  python -m evals.run_eval --config v2 --suite adversarial
  python -m evals.run_eval --config v2 --mode live --judge llm   # real Claude (needs API key)
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict

from app.agent import Agent, CONFIGS
from app.llm import AnthropicLLM, OfflineLLM
from evals.dataset import SUITES, load_cases
from evals.evaluators import diagnose, score_case
from evals.judge import LLMJudge
from evals.stats import aggregate, by_group

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def result_path(config: str, mode: str) -> str:
    return os.path.join(RESULTS_DIR, f"{config}__{mode}.json")


def run(config_name: str, mode: str, suite: str, judge_kind: str) -> list[dict]:
    cfg = CONFIGS[config_name]
    llm = AnthropicLLM() if mode == "live" else OfflineLLM()
    judge = LLMJudge() if judge_kind == "llm" else None
    agent = Agent(cfg, llm)
    rows = []
    for case in load_cases(suite):
        trace = agent.run(case.input, case.user_id)
        scores = score_case(case, trace, judge)
        rows.append({"case": asdict(case), "trace": trace, "scores": scores,
                     "diagnosis": diagnose(case, scores, trace)})
    return rows


def print_report(rows: list[dict], title: str) -> None:
    agg = aggregate(rows)
    print(f"\n{'=' * 70}\n{title}   ({len(rows)} cases)\n{'=' * 70}")
    print("\nMetrics (mean over cases where the metric applies):")
    for k, v in agg.items():
        if v is not None:
            print(f"  {k:22s} {v:8.3f}")
    print("\ntask_success by suite:")
    for g, (v, n) in by_group(rows, "suite").items():
        print(f"  {g:22s} {v:5.2f}  (n={n})")
    print("\ntask_success by category:")
    for g, (v, n) in sorted(by_group(rows, "category").items()):
        print(f"  {g:22s} {v:5.2f}  (n={n})")
    diag = Counter(r["diagnosis"] for r in rows if r["diagnosis"] != "pass")
    print("\nFailure breakdown (what to fix first):")
    for d, n in diag.most_common() or [("none", 0)]:
        print(f"  {d:22s} {n}")
    crit = [r for r in rows if r["diagnosis"] == "safety_violation"]
    if crit:
        print(f"\n!! CRITICAL SAFETY FAILURES: {[r['case']['id'] for r in crit]}  -> BLOCK RELEASE")
    print("\nFailed cases:")
    for r in [r for r in rows if r["diagnosis"] != "pass"]:
        print(f"  [{r['case']['id']}] {r['diagnosis']:20s} Q: {r['case']['input'][:55]!r}")
        print(f"        A: {r['trace']['answer'][:90]!r}  retrieved={[h['id'] for h in r['trace']['retrieved']]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="v1", choices=list(CONFIGS))
    p.add_argument("--mode", default="offline", choices=["offline", "live"])
    p.add_argument("--suite", default="all", choices=SUITES + ["all"])
    p.add_argument("--judge", default="none", choices=["none", "llm"])
    args = p.parse_args()

    rows = run(args.config, args.mode, args.suite, args.judge)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = result_path(args.config, args.mode)
    with open(path, "w") as f:
        json.dump({"config": args.config, "mode": args.mode, "rows": rows}, f, indent=1)
    print_report(rows, f"config={args.config} mode={args.mode}")
    print(f"\nSaved traces + scores -> {path}")


if __name__ == "__main__":
    main()
