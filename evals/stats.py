import random
import statistics
from collections import defaultdict


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def aggregate(rows: list[dict]) -> dict:
    metrics = sorted({m for r in rows for m in r["scores"]})
    agg = {m: mean(r["scores"].get(m) for r in rows) for m in metrics}
    lat = sorted(r["trace"]["latency_s"] for r in rows)
    agg["p95_latency_s"] = lat[int(0.95 * (len(lat) - 1))]
    agg["avg_tokens"] = mean(r["trace"]["input_tokens"] + r["trace"]["output_tokens"] for r in rows)
    agg["error_rate"] = mean(1.0 if r["trace"]["error"] else 0.0 for r in rows)
    return agg


def by_group(rows: list[dict], key: str, metric: str = "task_success") -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[r["case"][key]].append(r)
    return {g: (mean(x["scores"].get(metric) for x in rs), len(rs)) for g, rs in groups.items()}


def paired_compare(rows_old, rows_new, metric="task_success", n_boot=5000, seed=0) -> dict:
    """Paired bootstrap on per-case differences (new - old). Returns verdict + fixed/broken ids."""
    a = {r["case"]["id"]: r["scores"].get(metric) for r in rows_old}
    b = {r["case"]["id"]: r["scores"].get(metric) for r in rows_new}
    ids = [i for i in a if i in b and a[i] is not None and b[i] is not None]
    if not ids:
        return {"verdict": "NO_DATA", "n": 0, "mean_diff": 0.0, "ci95": (0.0, 0.0), "fixed": [], "broken": []}
    diffs = [b[i] - a[i] for i in ids]
    rng = random.Random(seed)
    boots = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    verdict = "BETTER" if lo > 0 else "WORSE" if hi < 0 else "SIMILAR"
    return {"verdict": verdict, "n": len(ids), "mean_diff": statistics.mean(diffs), "ci95": (lo, hi),
            "fixed": [i for i, d in zip(ids, diffs) if d > 0],
            "broken": [i for i, d in zip(ids, diffs) if d < 0]}
