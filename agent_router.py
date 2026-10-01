"""agent_router.py — M3: the bandit-router for agent-to-model routing.
Maps each agent to its optimal FREE model (d48 matrix) using the
bayesian_router.py Thompson-sampling pattern (310 LOC, proven).

Usage: from agent_router import route; model = route("wayward-builder", "code_gen")
"""
import json, os, random

# The d48 matrix: agent → (task_type, preferred_model, fallback_models)
ROUTING_TABLE = {
    "wayward-builder":     ("code_gen",    "local-qwen3-coder",       ["or-free-qwen-qwen3-coder-free"]),
    "wayward-reviewer":    ("judge",       "local-lfm2-rag",          ["aihubmix-glm-4.7-flash-free"]),
    "wayward-contrarian":  ("eval",        "aihubmix-glm-4.7-flash-free", ["local-qwen3-coder"]),
    "wayward-researcher":  ("fact_find",   "aihubmix-glm-4.7-flash-free", ["local-lfm2-tool"]),
    "wayward-reflector":   ("reflexion",   "local-lfm2-tool",         ["aihubmix-glm-4.7-flash-free"]),
    "wayward-curator":     ("lessons",     "aihubmix-gemini-3.5-flash-lite-free", ["local-lfm2-tool"]),
    "wayward-librarian":   ("wiki",        "local-qwen3-coder",       ["aihubmix-glm-4.7-flash-free"]),
    "wayward-cataloger":   ("catalog",     "aihubmix-glm-4.7-flash-free", ["local-lfm2-tool"]),
    "wayward-hermes":      ("cleanup",     "local-lfm2-tool",         ["aihubmix-glm-4.7-flash-free"]),
    "wayward-spec":        ("spec_gen",    "aihubmix-glm-5.2-free",   ["local-qwen3-coder"]),
    "wayward-vsm":         ("audit",       "aihubmix-glm-5.2-free",   ["local-qwen3-coder"]),
    "master-builder":      ("build",       "local-qwen3-coder",       ["or-free-qwen-qwen3-coder-free"]),
    "stack-warden":        ("ops",         "aihubmix-glm-4.7-flash-free", ["local-lfm2-tool"]),
    "master-treasurer":    ("money",       "aihubmix-glm-5.2-free",   ["local-qwen3-coder"]),
    "morning-conductor":   ("brief",       "local-lfm2-tool",         ["aihubmix-glm-4.7-flash-free"]),
    "master-scheduler":    ("triage",      "aihubmix-glm-4.7-flash-free", ["local-lfm2-tool"]),
    "forge-master":        ("plugins",     "local-qwen3-coder",       ["or-free-qwen-qwen3-coder-free"]),
}

# Task-type → model-class mapping (for unknown agents)
TASK_DEFAULTS = {
    "code_gen":  "local-qwen3-coder",
    "judge":     "local-lfm2-rag",
    "eval":      "aihubmix-glm-4.7-flash-free",
    "fact_find": "aihubmix-glm-4.7-flash-free",
    "reflexion": "local-lfm2-tool",
    "cleanup":   "local-lfm2-tool",
    "audit":     "aihubmix-glm-5.2-free",
    "spec_gen":  "aihubmix-glm-5.2-free",
    "ops":       "aihubmix-glm-4.7-flash-free",
    "money":     "aihubmix-glm-5.2-free",
    "brief":     "local-lfm2-tool",
    "build":     "local-qwen3-coder",
    "plugins":   "local-qwen3-coder",
}

# Simple posterior store (in production: read/write from a JSON that the
# bayesian_router.py Thompson sampler also reads)
POSTERIOR_FILE = os.path.join(os.path.dirname(__file__), "router_posteriors.json")


def load_posteriors():
    if os.path.exists(POSTERIOR_FILE):
        return json.load(open(POSTERIOR_FILE))
    return {}


def save_posteriors(p):
    json.dump(p, open(POSTERIOR_FILE, "w"), indent=2)


def route(agent_name: str, task_hint: str = "") -> str:
    """Route an agent to its optimal model. Uses the d48 table as the prior;
    falls back to task-type defaults; logs for future GRPO fine-tuning."""
    posteriors = load_posteriors()

    if agent_name in ROUTING_TABLE:
        task_type, preferred, fallbacks = ROUTING_TABLE[agent_name]
    else:
        task_type = task_hint or "fact_find"
        preferred = TASK_DEFAULTS.get(task_type, "aihubmix-glm-4.7-flash-free")
        fallbacks = ["local-lfm2-tool"]

    # Thompson sampling: if we have posteriors for this agent, sample from them
    if agent_name in posteriors:
        alpha_beta = posteriors[agent_name].get(preferred, {"alpha": 1, "beta": 1})
        sample = random.betavariate(alpha_beta["alpha"], alpha_beta["beta"])
        if sample < 0.3 and fallbacks:  # low confidence → explore a fallback
            return random.choice(fallbacks)

    # log the decision (for GRPO training data)
    decisions = posteriors.setdefault(agent_name, {}).setdefault("_decisions", [])
    decisions.append({"model": preferred, "task": task_type})
    if len(decisions) > 100:
        decisions.pop(0)  # keep the log bounded
    save_posteriors(posteriors)

    return preferred


def record_outcome(agent_name: str, model: str, success: bool):
    """Record whether the model's output was accepted (for the posterior update)."""
    posteriors = load_posteriors()
    entry = posteriors.setdefault(agent_name, {}).setdefault(model, {"alpha": 1, "beta": 1})
    if success:
        entry["alpha"] += 1
    else:
        entry["beta"] += 1
    save_posteriors(posteriors)


if __name__ == "__main__":
    # smoke test: route every agent
    for agent in ROUTING_TABLE:
        model = route(agent)
        print(f"  {agent:24} → {model}")
    print(f"\nPosteriors saved to: {POSTERIOR_FILE}")
