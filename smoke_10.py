"""B2 smoke test: 10 CR-2 reference queries against the CR-2b HTTP service.
Usage: python smoke_10.py [port]   (default 8026 temp; use 8025 for final verify)
Grading: expected token in top-1 name (strict) / top-3 (CR-2 DoD standard).
Set reconstructed from doing.md 2026-09-22 CR-2 entry (9 named concepts) +
CR-1 test query ASTPatchEngine as the 10th.
Exit 1 if any query misses top-3.
"""
import json, sys, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8026"
CASES = [
    ("where is the thompson sampling bandit router with beta posteriors", ["bayesian_router"]),
    ("pocketflow run_core_flow core implementation", ["pocketflow_core", "run_core_flow"]),
    ("llm slot filler signature", ["slot_fill"]),
    ("experiment runner that logs metrics", ["run_experiment"]),
    ("anti-unification ast template miner", ["anti_unification"]),
    ("spacy lemmatizer mapper", ["lemma"]),
    ("verification gate check", ["verification"]),
    ("frame recognizer", ["frame_recognizer", "recognize"]),
    ("equality graph ast optimizer", ["egraph", "equality_graph"]),
    ("unified pipeline dedup step", ["unified_pipeline"]),
]

n1 = n3 = 0
for q, toks in CASES:
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/search",
        data=json.dumps({"query": q, "top": 8}).encode(),
        headers={"Content-Type": "application/json", "X-Client": "b2-smoke"})
    out = json.loads(urllib.request.urlopen(r, timeout=240).read())
    names = [f"{h['name']} {h['file'].rsplit('/', 1)[-1]}" for h in out["results"]]  # CR-2 DoD standard: correct FILES
    low = [n.lower() for n in names]
    top1 = bool(low) and any(t.lower() in low[0] for t in toks)
    top3 = any(t.lower() in n for n in low[:3] for t in toks)
    n1 += top1; n3 += top3
    show = "; ".join(f"{h['name']}({h['file'].rsplit('/',1)[-1]})" for h in out["results"][:3])
    print(f"{'PASS' if top3 else 'FAIL'} top1={int(top1)} | {q[:46]:48} | {show}")
print(f"RESULT top3={n3}/10 top1={n1}/10 (port {PORT})")
sys.exit(0 if n3 == 10 else 1)
