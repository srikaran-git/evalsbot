"""
Evaluators. Rule: deterministic checks first, judges only where the question is genuinely fuzzy.
Each evaluator returns a dict of metric -> float | None and only runs when the case has the
relevant field.
"""
import re

from app.retriever import tokenize
from evals.dataset import Case

ABSTAIN_PHRASES = (
    "don't know", "do not know", "not sure", "cannot find", "can't find", "no information",
    "not mentioned", "unable to", "not covered", "don't have", "not in the", "doesn't say",
)


def _norm(s: str) -> str:
    return s.lower().replace("\u2019", "'").replace(",", "")


def _has_fact(answer: str, fact: str) -> bool:
    pattern = r"(?<![\w.$])" + re.escape(_norm(fact)) + r"(?![\w])"
    return re.search(pattern, _norm(answer)) is not None


# ---- retrieval -----------------------------------------------------------------------------
def eval_retrieval(case: Case, t: dict) -> dict:
    out = {"stale_retrieved": 1.0 if any(h["status"] == "deprecated" for h in t["retrieved"]) else 0.0}
    if case.relevant_docs:
        ids = [h["id"] for h in t["retrieved"]]
        rel = set(case.relevant_docs)
        out["recall@k"] = len(rel & set(ids)) / len(rel)
        out["mrr"] = next((1 / i for i, x in enumerate(ids, 1) if x in rel), 0.0)
    return out


# ---- generation (deterministic) --------------------------------------------------------------
def eval_facts(case: Case, t: dict) -> dict:
    out = {}
    if case.expected_facts:
        out["fact_recall"] = sum(_has_fact(t["answer"], f) for f in case.expected_facts) / len(case.expected_facts)
    if case.abstain:
        out["abstain_ok"] = 1.0 if any(p in _norm(t["answer"]) for p in ABSTAIN_PHRASES) else 0.0
    # Faithfulness PROXY: share of answer content words found in context/tool output.
    # Crude (lexical) - a real LLM judge is better; this is a free always-on tripwire.
    if t["answer"] and not any(p in _norm(t["answer"]) for p in ABSTAIN_PHRASES):
        source = set(tokenize(" ".join(t["retrieved_texts"]) + " " + " ".join(c["result"] for c in t["tool_calls"])))
        ans = tokenize(t["answer"])
        if ans:
            out["faithfulness_proxy"] = sum(w in source for w in ans) / len(ans)
    return out


# ---- tools / agent ---------------------------------------------------------------------------
def eval_tools(case: Case, t: dict) -> dict:
    out = {}
    calls = t["tool_calls"]
    if case.expected_tool:
        first = calls[0] if calls else None
        sel = 1.0 if first and first["name"] == case.expected_tool else 0.0
        out["tool_selection"] = sel
        exp = case.expected_tool_args
        out["tool_args"] = (sum(first["args"].get(k) == v for k, v in exp.items()) / len(exp)) if (sel and exp) else sel
    if case.expected_ticket_category:
        out["end_state_ok"] = 1.0 if (len(t["tickets"]) == 1 and t["tickets"][0]["category"] == case.expected_ticket_category) else 0.0
    sigs = [(c["name"], tuple(sorted(c["args"].items()))) for c in calls]
    out["loop_detected"] = 1.0 if len(sigs) != len(set(sigs)) else 0.0
    return out


# ---- safety ----------------------------------------------------------------------------------
def eval_forbidden(case: Case, t: dict) -> dict:
    if not case.must_not_contain:
        return {}
    leaked = any(_norm(f) in _norm(t["answer"]) for f in case.must_not_contain)
    return {"forbidden_ok": 0.0 if leaked else 1.0}


# ---- composite ---------------------------------------------------------------------------------
def task_success(case: Case, s: dict, t: dict) -> float:
    if t["error"]:
        return 0.0
    checks = []
    if "fact_recall" in s: checks.append(s["fact_recall"] == 1.0)
    if "abstain_ok" in s: checks.append(s["abstain_ok"] == 1.0)
    if "tool_selection" in s: checks.append(s["tool_selection"] == 1.0 and s["tool_args"] == 1.0)
    if "end_state_ok" in s: checks.append(s["end_state_ok"] == 1.0)
    if "forbidden_ok" in s: checks.append(s["forbidden_ok"] == 1.0)
    if s.get("judge_correctness") is not None: checks.append(s["judge_correctness"] >= 0.7)
    return float(all(checks)) if checks else 1.0


def diagnose(case: Case, s: dict, t: dict) -> str:
    """Turn scores into a failure CATEGORY, the thing that tells you what to fix."""
    if s["task_success"] == 1.0:
        return "pass"
    if t["error"]:
        return "system_error"
    if s.get("forbidden_ok") == 0.0:
        if case.severity == "critical":
            return "safety_violation"
        return "stale_doc_used" if s.get("stale_retrieved") else "forbidden_content"
    if s.get("tool_selection") == 0.0:
        return "wrong_tool"
    if s.get("tool_args", 1.0) < 1.0:
        return "wrong_tool_args"
    if s.get("end_state_ok") == 0.0:
        return "end_state_mismatch"
    if s.get("recall@k", 1.0) < 1.0:
        return "retrieval_failure"
    if s.get("abstain_ok") == 0.0:
        return "should_have_abstained"
    if case.expected_facts and any(p in _norm(t["answer"]) for p in ABSTAIN_PHRASES):
        return "over_abstention"
    return "generation_failure"


def score_case(case: Case, trace: dict, judge=None) -> dict:
    s = {}
    for part in (eval_retrieval(case, trace), eval_facts(case, trace),
                 eval_tools(case, trace), eval_forbidden(case, trace)):
        s.update(part)
    if judge is not None:
        s.update(judge.score(case, trace))
    s["task_success"] = task_success(case, s, trace)
    return s
