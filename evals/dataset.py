import json
import os
from dataclasses import dataclass, field
from typing import Optional

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SUITES = ["golden", "regression", "adversarial"]


@dataclass
class Case:
    id: str
    suite: str
    category: str
    input: str
    user_id: str = "E100"
    relevant_docs: list = field(default_factory=list)       # for retrieval metrics
    expected_facts: list = field(default_factory=list)      # atomic facts the answer must contain
    expected_tool: Optional[str] = None
    expected_tool_args: dict = field(default_factory=dict)
    expected_ticket_category: Optional[str] = None          # real end-state check
    must_not_contain: list = field(default_factory=list)    # leak / stale / injection canaries
    abstain: bool = False                                   # correct behaviour = "I don't know"
    severity: str = "normal"                                # normal | critical


def load_suite(name: str) -> list[Case]:
    with open(os.path.join(DATA_DIR, f"{name}.jsonl")) as f:
        return [Case(**json.loads(line)) for line in f if line.strip()]


def load_cases(suite: str = "all") -> list[Case]:
    names = SUITES if suite == "all" else [suite]
    return [c for n in names for c in load_suite(n)]
