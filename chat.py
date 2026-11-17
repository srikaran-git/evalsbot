"""Talk to the assistant directly (and see its trace).

  python chat.py --config v2                 # offline stand-in
  python chat.py --config v2 --mode live     # real Claude (needs ANTHROPIC_API_KEY)
"""
import argparse
import json

from app.agent import Agent, CONFIGS
from app.llm import AnthropicLLM, OfflineLLM

p = argparse.ArgumentParser()
p.add_argument("--config", default="v2", choices=list(CONFIGS))
p.add_argument("--mode", default="offline", choices=["offline", "live"])
p.add_argument("--user", default="E100")
a = p.parse_args()

agent = Agent(CONFIGS[a.config], AnthropicLLM() if a.mode == "live" else OfflineLLM())
print(f"config={a.config} mode={a.mode} user={a.user}  (Ctrl+C to quit)")
while True:
    q = input("\nyou> ").strip()
    if not q:
        continue
    t = agent.run(q, a.user)
    print(f"bot> {t['answer']}")
    print("trace:", json.dumps({k: t[k] for k in ("retrieved", "tool_calls", "steps", "error")}, indent=1))
