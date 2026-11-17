# evalbot: a small RAG + tool-calling assistant with a full evaluation suite

Everything from the 7-day course on one small system.

    question -> BM25 retrieval -> LLM (tools: lookup_pto, create_ticket) -> answer
                      \_______________ full trace saved ______________/

## Setup (VS Code)
1. Open this folder in VS Code (File > Open Folder).
2. Terminal > New Terminal, then:
       python -m venv .venv
       .venv\Scripts\activate          (Windows)   |   source .venv/bin/activate   (Mac/Linux)
3. Offline mode needs no installs. For live mode: `pip install anthropic` and set ANTHROPIC_API_KEY.

## Run
    python -m evals.run_eval --config v1          # baseline
    python -m evals.run_eval --config v1_k3       # change ONE thing (top_k 1 -> 3)
    python -m evals.run_eval --config v2          # candidate (filter stale docs + hardened prompt)
    python -m evals.compare v1 v2                 # better / worse / similar + release gate
    python chat.py --config v2                    # talk to it and inspect the trace

Live mode (real Claude + optional LLM judge):
    python -m evals.run_eval --config v1 --mode live --judge llm
    python -m evals.run_eval --config v2 --mode live --judge llm
    python -m evals.compare v1 v2 --mode live

## Offline vs live
Offline mode uses a rule-based stand-in (app/llm.py: OfflineLLM). It is NOT a language model; it exists
so you can run and learn the pipeline with no key. Some failures you see offline (e.g. g04, g08, g12)
come from its crude sentence extraction. Live-mode numbers will differ, and that is the point.

## Where each course day lives
Day 1 lifecycle ........ evals/run_eval.py          Day 2 judge + kappa ... evals/judge.py, calibrate_judge.py
Day 3 RAG metrics ...... evals/evaluators.py        Day 4 tools/end state . evals/evaluators.py, app/tools.py
Day 5 safety ........... data/adversarial.jsonl     Day 6 datasets ........ evals/data/*.jsonl
Day 7 regression/gate .. evals/compare.py, stats.py
