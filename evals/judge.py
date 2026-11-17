"""
LLM-as-a-Judge (optional, live mode only). Remember Day 2: an uncalibrated judge is a guess.
Run evals/calibrate_judge.py against your own human labels before trusting any score from here.
"""
import json
import os
import re

from evals.dataset import Case

CORRECTNESS_PROMPT = """You are grading an HR/IT assistant's answer.

Question: {question}
Required facts (all must be stated correctly): {facts}
Assistant answer: {answer}

Score 1.0 if every required fact is stated correctly and nothing contradicts them,
0.5 if some are correct or the answer is partially right, 0.0 if wrong, missing, or contradicted.
Do NOT reward length or politeness.
Respond ONLY with JSON: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}"""

ABSTAIN_PROMPT = """The assistant should have said it does NOT know the answer, since the information is not available.

Question: {question}
Assistant answer: {answer}

Score 1.0 if it clearly says it doesn't know / can't find it and invents nothing, else 0.0.
Respond ONLY with JSON: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}"""


def parse_judge_json(raw: str):
    """Strip code fences / preamble. Returns float score or None (never silently 0)."""
    text = raw.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return max(0.0, min(1.0, float(json.loads(m.group(0))["score"])))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


class LLMJudge:
    def __init__(self, model: str | None = None):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("JUDGE_MODEL", os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"))

    def _ask(self, prompt: str):
        resp = self.client.messages.create(model=self.model, max_tokens=300, temperature=0,
                                           messages=[{"role": "user", "content": prompt}])
        return parse_judge_json("".join(b.text for b in resp.content if b.type == "text"))

    def score(self, case: Case, trace: dict) -> dict:
        try:
            if case.expected_facts:
                return {"judge_correctness": self._ask(CORRECTNESS_PROMPT.format(
                    question=case.input, facts=", ".join(case.expected_facts), answer=trace["answer"]))}
            if case.abstain:
                return {"judge_correctness": self._ask(ABSTAIN_PROMPT.format(
                    question=case.input, answer=trace["answer"]))}
        except Exception:
            return {"judge_correctness": None}   # judge failure is tracked, not counted as a fail
        return {}
