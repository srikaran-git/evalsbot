"""RAG + tool-calling agent. Every run returns a full trace so evaluators can inspect each stage."""
import time
from dataclasses import dataclass

from app.kb import DOCS, PROMPTS
from app.retriever import Retriever
from app.tools import ToolBox

MAX_STEPS = 4


@dataclass(frozen=True)
class Config:
    name: str
    top_k: int
    filter_deprecated: bool
    prompt_version: str      # "basic" | "hardened"


# Change ONE thing at a time between configs so you can attribute effects (ablation).
CONFIGS = {
    "v1": Config("v1", top_k=1, filter_deprecated=False, prompt_version="basic"),       # baseline
    "v1_k3": Config("v1_k3", top_k=3, filter_deprecated=False, prompt_version="basic"),  # only top_k changed
    "v2": Config("v2", top_k=3, filter_deprecated=True, prompt_version="hardened"),      # candidate
}


def build_user_message(question: str, user_id: str, hits: list[dict]) -> str:
    ctx = "\n".join(f"[{h['id']}] {h['text']}" for h in hits) or "(none)"
    return f"Current user: {user_id}\n\nContext documents:\n{ctx}\n\nQuestion: {question}"


class Agent:
    def __init__(self, cfg: Config, llm):
        self.cfg, self.llm = cfg, llm
        self.retriever = Retriever(DOCS, filter_deprecated=cfg.filter_deprecated)
        self.system = PROMPTS[cfg.prompt_version]

    def run(self, question: str, user_id: str = "E100") -> dict:
        t0 = time.time()
        toolbox = ToolBox(user_id)
        hits = self.retriever.search(question, self.cfg.top_k)
        messages = [{"role": "user", "content": build_user_message(question, user_id, hits)}]
        tool_calls, answer, error = [], "", None
        in_tok = out_tok = steps = 0
        try:
            for steps in range(1, MAX_STEPS + 1):
                r = self.llm.step(self.system, messages, toolbox.schemas)
                in_tok += r.input_tokens
                out_tok += r.output_tokens
                if not r.tool_calls:
                    answer = r.text
                    break
                messages.append({"role": "assistant", "content": r.content_blocks})
                results = []
                for c in r.tool_calls:
                    out = toolbox.call(c["name"], c["args"])
                    tool_calls.append({"name": c["name"], "args": c["args"], "result": out})
                    results.append({"type": "tool_result", "tool_use_id": c["id"], "content": out})
                messages.append({"role": "user", "content": results})
            else:
                error = "max_steps_exceeded"
        except Exception as e:  # a crashed run is a data point, not a reason to abort the eval
            error = f"{type(e).__name__}: {e}"
        return {
            "question": question, "user_id": user_id, "config": self.cfg.name,
            "retrieved": [{"id": h["id"], "score": h["score"], "status": h["status"]} for h in hits],
            "retrieved_texts": [h["text"] for h in hits],
            "tool_calls": tool_calls, "answer": answer, "steps": steps,
            "latency_s": round(time.time() - t0, 4),
            "input_tokens": in_tok, "output_tokens": out_tok,
            "error": error, "tickets": toolbox.tickets,   # real end state
        }
