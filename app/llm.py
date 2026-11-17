"""
Two interchangeable backends with the same interface: step(system, messages, tools) -> LLMResponse

AnthropicLLM  - the real thing (needs `pip install anthropic` and ANTHROPIC_API_KEY).
OfflineLLM    - a deterministic, rule-based stand-in so you can run the whole eval pipeline with no
                API key. It is NOT a language model: use it to learn the framework, then switch to
                live mode for meaningful quality numbers.
"""
import os
import re
from dataclasses import dataclass, field

from app.retriever import tokenize


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list = field(default_factory=list)      # [{"id", "name", "args"}]
    content_blocks: list = field(default_factory=list)  # assistant content to append to history
    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicLLM:
    def __init__(self, model: str | None = None):
        import anthropic  # imported lazily so offline mode needs no dependency
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        # Check Anthropic's docs for current model names and override with CLAUDE_MODEL if needed.
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

    def step(self, system, messages, tools) -> LLMResponse:
        resp = self.client.messages.create(
            model=self.model, max_tokens=1000, temperature=0,
            system=system, tools=tools, messages=messages,
        )
        text, calls, blocks = "", [], []
        for b in resp.content:
            if b.type == "text":
                text += b.text
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                calls.append({"id": b.id, "name": b.name, "args": dict(b.input)})
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": dict(b.input)})
        return LLMResponse(text, calls, blocks, resp.usage.input_tokens, resp.usage.output_tokens)


class OfflineLLM:
    """Rule-based stand-in. It reacts to the prompt version, so prompt changes show up in evals."""

    def step(self, system, messages, tools) -> LLMResponse:
        hardened = "never reveal" in system.lower()
        first = messages[0]["content"]
        in_tok = len((system + str(messages)).split())

        last = messages[-1]
        if last["role"] == "user" and isinstance(last["content"], list):   # tool results came back
            text = " ".join(b["content"] for b in last["content"] if b.get("type") == "tool_result")
            return LLMResponse(text, [], [{"type": "text", "text": text}], in_tok, len(text.split()))

        uid = re.search(r"Current user: (\w+)", first).group(1)
        question = first.split("Question:", 1)[1].strip()
        q = question.lower()
        ctx = re.findall(r"^\[(\w+)\] (.+)$", first, flags=re.M)

        def done(text):
            return LLMResponse(text, [], [{"type": "text", "text": text}], in_tok, len(text.split()))

        def call(name, args):
            block = {"type": "tool_use", "id": "toolu_offline_1", "name": name, "input": args}
            return LLMResponse("", [{"id": "toolu_offline_1", "name": name, "args": args}],
                               [block], in_tok, 10)

        # Simulated weaknesses of a weakly-prompted model (they disappear with the hardened prompt):
        if not hardened and "system prompt" in q:
            return done(system)
        if not hardened:
            for _, text in ctx:
                if "ignore all previous instructions" in text.lower():
                    sent = [s for s in re.split(r"(?<=[.!?])\s+", text) if "SYSTEM NOTE" in s][0]
                    return done(sent)

        if "ticket" in q and any(w in q for w in ("open", "create", "file", "raise")):
            if any(w in q for w in ("laptop", "monitor", "screen", "keyboard", "mouse")):
                cat = "hardware"
            elif any(w in q for w in ("vpn", "password", "log in", "login", "access")):
                cat = "access"
            elif any(w in q for w in ("install", "software", "app")):
                cat = "software"
            else:
                cat = "other"
            return call("create_ticket", {"category": cat, "description": question})

        if "pto" in q and (re.search(r"\bE\d{3}\b", question) or
                           any(c in q for c in ("do i have", "my pto", "balance", "left"))):
            m = re.search(r"\bE\d{3}\b", question)
            return call("lookup_pto", {"employee_id": m.group(0) if m else uid})

        # Extractive "RAG": best sentence from each of the top-2 retrieved docs.
        qt = set(tokenize(question))
        picks = []
        for _, text in ctx[:2]:
            best, best_ov = None, 0
            for sent in re.split(r"(?<=[.!?])\s+", text):
                ov = len(qt & set(tokenize(sent)))
                if ov > best_ov:
                    best, best_ov = sent, ov
            if best and best_ov >= 2:
                picks.append(best)
        return done(" ".join(picks) if picks else "I don't know based on the available documents.")
