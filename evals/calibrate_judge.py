"""
Validate your evaluator against human labels (Day 2: the evaluator itself must be evaluated).

  1) python -m evals.calibrate_judge export v2          -> writes evals/results/calibration.csv
  2) Open the CSV, fill the `human_pass` column with 1 (answer acceptable) or 0 (not acceptable).
  3) python -m evals.calibrate_judge score              -> agreement + Cohen's kappa

`auto_pass` is task_success from your automated evaluators (fact checks, and the LLM judge if you ran
with --judge llm). Low kappa means: fix the rubric or the evaluator before trusting its scores.
"""
import csv
import json
import os
import random
import sys
from collections import Counter

from evals.run_eval import RESULTS_DIR, result_path

CSV_PATH = os.path.join(RESULTS_DIR, "calibration.csv")


def cohen_kappa(a: list, b: list) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in set(a) | set(b))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def export(config: str, mode: str = "offline", n: int = 20):
    with open(result_path(config, mode)) as f:
        rows = json.load(f)["rows"]
    rows = random.Random(0).sample(rows, min(n, len(rows)))
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "question", "answer", "auto_pass", "human_pass"])
        for r in rows:
            w.writerow([r["case"]["id"], r["case"]["input"], r["trace"]["answer"],
                        int(r["scores"]["task_success"]), ""])
    print(f"Wrote {len(rows)} rows -> {CSV_PATH}. Fill the human_pass column (1/0), then run `score`.")


def score():
    with open(CSV_PATH) as f:
        rows = [r for r in csv.DictReader(f) if r["human_pass"].strip() in ("0", "1")]
    if len(rows) < 10:
        sys.exit("Label at least 10 rows first (30+ is much better).")
    auto = [int(r["auto_pass"]) for r in rows]
    human = [int(r["human_pass"]) for r in rows]
    agree = sum(a == h for a, h in zip(auto, human)) / len(rows)
    print(f"n={len(rows)}  raw agreement={agree:.2f}  Cohen's kappa={cohen_kappa(auto, human):.2f}")
    print("Disagreements (inspect these first):")
    for r, a, h in zip(rows, auto, human):
        if a != h:
            print(f"  {r['case_id']}: auto={a} human={h}  answer={r['answer'][:80]!r}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "export":
        export(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "offline")
    elif len(sys.argv) >= 2 and sys.argv[1] == "score":
        score()
    else:
        print(__doc__)
