"""batched_spawn.py — M2: the batching design implementation.
Replaces single-step spawns with 3-5 step batched missions.
Usage: from batched_spawn import BatchedMission; BatchedMission(agent, steps).run()
"""
import json, os, subprocess, sys, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG = os.path.join(ROOT, "doing.md")

class BatchedMission:
    """A batched subagent mission: 3-5 related steps in one spawn context.

    The d46 forensics: 53 spawns/day × full cold-context bootstrap = 59% of all
    requests. The d47 fix: batch steps so each spawn pays the bootstrap ONCE
    for multiple outputs. Est. savings: 70-90%.
    """

    def __init__(self, agent_type: str, steps: list, dod: str = ""):
        self.agent = agent_type
        self.steps = steps  # list of step descriptions (3-5 recommended)
        self.dod = dod or f"All {len(steps)} steps complete with artifacts"

    def build_prompt(self) -> str:
        """Build a single batched prompt for all steps."""
        step_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(self.steps))
        return f"""You are a {self.agent} executing a BATCHED mission (the d47 token-efficiency fix).
Complete ALL {len(self.steps)} steps below in this ONE session — do NOT request re-spawns between steps.
Use internal loops and conditionals where appropriate. Write incrementally (skeleton first).

STEPS:
{step_text}

DoD: {self.dod}

RULES (Wayward 2.0 batching protocol):
- Write-first: your first action creates the output file skeleton
- Token-frugal: read ≤6 files per burst, batch your Bash commands
- Incremental: append to output files after each step completes
- Single context: stay in this session — the batching IS the point
"""

    def log(self, result: str):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"{ts} | BATCHED-SPAWN | {self.agent} | {len(self.steps)} steps | {result}\n")


# The prompt template for the orchestrator to use when dispatching
def format_batched_prompt(agent_type: str, steps: list, dod: str = "") -> str:
    return BatchedMission(agent_type, steps, dod).build_prompt()


# Example usage patterns (for the orchestrator's reference):
EXAMPLES = {
    "research_triple": {
        "agent": "wayward-researcher",
        "steps": [
            "Search for the primary topic (5 queries via search-mcp)",
            "Fetch 3 most relevant primary sources",
            "Write the report incrementally (skeleton → 25% → 50% → 75% → 100%)",
            "Post the summary to the msg board",
        ],
        "dod": "Report complete + board post exists",
    },
    "build_pair": {
        "agent": "master-builder",
        "steps": [
            "Read the spec for ticket A + ticket B",
            "Implement ticket A (TDD: red → green → verify)",
            "Implement ticket B (TDD: red → green → verify)",
            "Run the integration smoke test for both",
            "Update builder/doing.md with both milestones",
        ],
        "dod": "Both tickets pass their DoDs",
    },
    "audit_batch": {
        "agent": "stack-warden",
        "steps": [
            "Fleet census (docker ps on ce + lyre via ssh)",
            "Health-check the top 10 services",
            "Prune anonymous containers if found",
            "Write the fleet snapshot report",
        ],
        "dod": "Snapshot report exists + anomalies flagged",
    },
}

if __name__ == "__main__":
    print(format_batched_prompt("wayward-researcher", EXAMPLES["research_triple"]["steps"]))
