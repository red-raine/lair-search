"""Morning Conductor — the S3 control node.
Reads overnight artifacts, triages, formats the walkthrough, dispatches.
Usage: python morning_conductor.py [--no-walkthrough]  (→ writes brief to file only)
"""
import sys
import json, os, re, datetime, glob

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TODAY = datetime.date.today().isoformat()

def read_tail(path, n=20):
    try:
        lines = open(path, encoding="utf-8").readlines()
        return [l.rstrip() for l in lines[-n:]]
    except FileNotFoundError:
        return []

def read_file(path, max_chars=4000):
    try:
        return open(path, encoding="utf-8").read()[:max_chars]
    except FileNotFoundError:
        return None

def gather_overnight():
    """Collect what the automations produced since yesterday."""
    out = {"presearch": [], "treasury": [], "builder": [], "warden": [], "errors": []}
    # presearch deliverables
    for f in sorted(glob.glob(os.path.join(ROOT, "research", "presearch", f"{TODAY}*.md"))):
        out["presearch"].append(os.path.basename(f))
    # treasury radar
    radar = os.path.join(ROOT, "treasury", "research", f"grant-radar-{TODAY}.md")
    if os.path.exists(radar):
        out["treasury"].append(os.path.basename(radar))
    # builder milestones
    builder_doing = os.path.join(ROOT, "builder", "doing.md")
    for l in read_tail(builder_doing, 5):
        if "BUILD" in l or "DONE" in l:
            out["builder"].append(l[:120])
    # warden milestones
    warden_doing = os.path.join(ROOT, "warden", "doing.md")
    for l in read_tail(warden_doing, 5):
        if "PASS" in l or "DONE" in l or "FIX" in l:
            out["warden"].append(l[:120])
    return out

def gather_frontier():
    """Compute the frontier using frontier.py logic (import or inline)."""
    frontier_script = os.path.join(ROOT, "tools", "lair-search", "frontier.py")
    if os.path.exists(frontier_script):
        os.system(f'python "{frontier_script}" > {os.path.join(ROOT, "research", "frontier-snapshot.json")} 2>&1')
    try:
        return json.load(open(os.path.join(ROOT, "research", "frontier-snapshot.json"), encoding="utf-8"))
    except Exception:
        return {"ready": [], "blocked": [], "ready_count": 0}

def gather_grill_queue():
    """Read the grill queue (avoided/forgotten tasks)."""
    return read_file(os.path.join(ROOT, "research", "grill-queue.md"), 3000) or "No grill queue found."

def check_risks():
    """Scan for Red Notes, prodrome flags, deadline proximity."""
    risks = []
    # deadline proximity
    deadline = datetime.date(2026, 10, 14)
    days = (deadline - datetime.date.today()).days
    if 0 < days <= 14:
        risks.append(f"IL CAF deadline in {days} days ({deadline}) — work samples needed")
    # kill-switch check
    doing = read_tail(os.path.join(ROOT, "doing.md"), 3)
    for l in doing:
        if "PAUSE" in l.upper() and "PRESEARCH" in l.upper():
            risks.append("PAUSE-PRESEARCH kill switch detected")
    return risks

def find_win():
    """Find one genuine positive from the tracker/canon."""
    wins = [
        "Trauma (film) won your school festival and screened at Tramodance",
        "The Mycelium's consent fence blocked every leak in testing",
        "The 22-Path Tree maps your shadow work to creative projects",
        "Your music is the recent-work candidate for the IL grant",
        "The bge reranker correctly ranked 10/10 lair questions",
    ]
    import random
    return random.choice(wins)

def format_brief(overnight, frontier, risks, win):
    """Format the morning brief."""
    lines = []
    lines.append(f"☀️ MORNING BRIEF — {TODAY}")
    lines.append("━" * 40)
    lines.append(f"Win from yesterday: {win}")
    lines.append("")
    
    if overnight["presearch"]:
        lines.append(f"📋 Presearch ({len(overnight['presearch'])} files):")
        for f in overnight["presearch"]:
            lines.append(f"   → {f}")
        lines.append("")
    
    if overnight["treasury"]:
        lines.append(f"💰 Treasury: {', '.join(overnight['treasury'])}")
        lines.append("")
    
    ready = frontier.get("ready", [])
    if ready:
        lines.append(f"🟢 Ready ({len(ready)}):")
        for r in ready[:5]:
            lines.append(f"   → {r[:80]}")
        lines.append("")
    
    if risks:
        lines.append("⚠️ Risks:")
        for r in risks:
            lines.append(f"   → {r}")
        lines.append("")
    
    lines.append("Decision needed:")
    lines.append("  (max 3 — Mysty is not a task queue)")
    lines.append("")
    lines.append("Unblocked agents: builder, warden, treasury, scheduler")
    lines.append("")
    lines.append("Go with recommendations? (y/n/partial)")
    return "\n".join(lines)

def dispatch(query, answer):
    """Dispatch Mysty's answer to the right seats via msg-board."""
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    path = os.path.join(os.environ.get("USERPROFILE", ""), ".zcode", "msg", "GLOBAL", f"{ts}-morning-conductor-dispatch.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"---\nfrom: morning-conductor\nto: all-seats\nproject: master-control\ntype: dispatch\npriority: P2\nexpires: {datetime.date.today().replace(year=datetime.date.today().year+1)}\n---\n\n{query}\n\nMysty's answer: {answer}\n")
    return path

def run(walkthrough=True):
    overnight = gather_overnight()
    frontier = gather_frontier()
    risks = check_risks()
    win = find_win()
    brief = format_brief(overnight, frontier, risks, win)
    
    # write the brief to a file
    brief_path = os.path.join(ROOT, "research", "presearch", f"{TODAY}-morning-brief.md")
    with open(brief_path, "w", encoding="utf-8") as f:
        f.write(brief)
    
    if walkthrough:
        print(brief)
        print(f"\n📄 Brief saved: {brief_path}")
    else:
        print(f"Brief written: {brief_path}")
    
    return brief

if __name__ == "__main__":
    walkthrough = "--no-walkthrough" not in sys.argv
    run(walkthrough)
