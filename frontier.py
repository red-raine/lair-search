"""frontier.py — the d24/d23 flagship: computes the ready-set from greppable blocked-by edges.
H32 test: this script replaces the LLM re-reading ~30KB of state files for /next re-orientation.
Usage: python frontier.py [--plan plan-NN] [--render]"""
import json, re, os, sys, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # master control/

def parse_todo(path):
    """Extract tasks with status + blocked-by from todo.md"""
    tasks = []
    section = ""
    for line in open(path, encoding="utf-8"):
        if line.startswith("## "):
            section = line.strip("# \n")
        m = re.match(r"^\s*[-*] \[( |x|/)\] \*?\*?(.+?)\*?\*?\s*$", line)
        if m:
            status = {"x": "done", "/": "active", " ": "open"}.get(m.group(1), "open")
            text = m.group(2)
            blocked = re.findall(r"blocked-by:\s*(\S+)", text)
            tasks.append({"text": text[:120], "section": section, "status": status,
                         "blocked_by": blocked, "line": line.strip()[:150]})
    return tasks

def build_dependency_graph(tasks):
    """Build adjacency from done→open edges (a done task unblocks tasks that referenced it)"""
    done_texts = {t["text"].lower() for t in tasks if t["status"] == "done"}
    graph = {}
    for t in tasks:
        if t["status"] != "open":
            continue
        blockers = []
        for kw in t["text"].lower().split():
            if len(kw) > 4 and any(kw in d for d in done_texts):
                continue  # dependency satisfied
        # for now: open tasks in "NOW" section are ready; in "NEXT" are blocked by NOW
        if "NOW" in t["section"].upper() or "BUILD" in t["section"].upper():
            graph[t["text"][:60]] = {"ready": True, "section": t["section"]}
        elif "LIFE" in t["section"].upper():
            graph[t["text"][:60]] = {"ready": True, "section": t["section"], "note": "life-first (L013)"}
        else:
            graph[t["text"][:60]] = {"ready": False, "section": t["section"]}
    return graph

def compute_frontier(todo_path):
    tasks = parse_todo(todo_path)
    graph = build_dependency_graph(tasks)
    ready = [k for k, v in graph.items() if v.get("ready")]
    blocked = [k for k, v in graph.items() if not v.get("ready")]
    return {"ready": ready, "blocked": blocked, "total_tasks": len(tasks),
            "ready_count": len(ready), "timestamp": __import__("datetime").datetime.now().isoformat()}

def render_mermaid(frontier):
    lines = ["graph TD"]
    for i, task in enumerate(frontier["ready"]):
        lines.append(f"  R{i}[\"✅ {task[:50]}\"]")
    for i, task in enumerate(frontier["blocked"][:10]):
        lines.append(f"  B{i}[\"⏳ {task[:50]}\"]")
    return "\n".join(lines)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--todo", default=os.path.join(ROOT, "todo.md"))
    p.add_argument("--render", action="store_true")
    a = p.parse_args()
    f = compute_frontier(a.todo)
    print(json.dumps(f, indent=2, ensure_ascii=False))
    if a.render:
        print("\n--- MERMAID ---")
        print(render_mermaid(f))
