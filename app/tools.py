"""Tools the agent can call. State changes are recorded so evals can check the REAL end state."""
from app.kb import EMPLOYEES

TOOL_SCHEMAS = [
    {
        "name": "lookup_pto",
        "description": "Look up the remaining PTO days for an employee. Only the current user's own "
                       "balance may be viewed.",
        "input_schema": {
            "type": "object",
            "properties": {"employee_id": {"type": "string", "description": "e.g. E100"}},
            "required": ["employee_id"],
        },
    },
    {
        "name": "create_ticket",
        "description": "Open an IT support ticket for the current user.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": ["hardware", "access", "software", "other"]},
                "description": {"type": "string"},
            },
            "required": ["category", "description"],
        },
    },
]


class ToolBox:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.tickets: list[dict] = []     # the real end state evaluators inspect
        self.schemas = TOOL_SCHEMAS

    def call(self, name: str, args: dict) -> str:
        if name == "lookup_pto":
            emp_id = str(args.get("employee_id", ""))
            if emp_id != self.user_id:      # access control enforced in the tool, not just the prompt
                return "ACCESS DENIED: you can only view your own PTO balance."
            emp = EMPLOYEES.get(emp_id)
            if not emp:
                return f"ERROR: employee {emp_id} not found."
            return f"{emp['name']} ({emp_id}) has {emp['pto_days']} PTO days remaining."
        if name == "create_ticket":
            category = args.get("category", "other")
            ticket = {"id": f"IT-{1001 + len(self.tickets)}", "category": category,
                      "description": args.get("description", ""), "user": self.user_id}
            self.tickets.append(ticket)
            return f"Ticket {ticket['id']} created (category: {category})."
        return f"ERROR: unknown tool '{name}'."
