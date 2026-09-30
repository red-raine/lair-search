"""Lair Search CR-2: dual retrieval + weighted RRF + bge:8084 rerank + optional LLM judge.
CLI: python cr2_query.py "question" [--top 8] [--judge none|qwen|gemini] [--arm bandit|static]
DoD: 10 real lair questions return correct files.  Service wrapper (HTTP/n8n) comes after DoD."""
import json, math, re, sys, os, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
from cr1_index import embed, req, QDR, chunks_from, pid

INDEX = "E:/vibe_coding/dev/crawler/lair_wayward.json"
RERANK = os.environ.get("CR_RERANK_URL", "http://localhost:8084/rerank")   # tei-reranker (bge-reranker-v2-m3)
LITELLM = os.environ.get("CR_LITELLM_URL", "http://localhost:4000/v1/chat/completions")
KEY = None
try:
    for line in open("E:/ai-stack/.env", encoding="utf-8"):
        if line.startswith("LITELLM_MASTER_KEY="):
            KEY = line.split("=", 1)[1].strip()
except Exception:
    pass
JUDGE_MODEL = {"qwen": "or-free-qwen-qwen3-coder-free", "gemini": "gemini-flash"}

# ---------- corpus (rebuilt from index JSON; cached) ----------
def load_corpus():
    cache = os.path.join(os.path.dirname(__file__), "corpus_cache.json")
    if os.path.exists(cache):
        return json.load(open(cache, encoding="utf-8"))
    chunks = chunks_from(INDEX)
    corpus = [{"text": c["text"], **c["payload"]} for c in chunks]
    json.dump(corpus, open(cache, "w", encoding="utf-8"))
    return corpus

# ---------- BM25 over tokens (deterministic; d33 offload ruling) ----------
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")
def toks(s):
    out, prev = [], ""
    for m in TOKEN.finditer(s):          # split camelCase / snake_case
        w = m.group(0)
        parts = re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|^[a-z]+|[a-z]+$|\d+", w)
        out.extend(p.lower() for p in parts if p)
    return out

def bm25_build(corpus):
    docs = [toks(f"{c['name']} {c['file']} {c['text']}") for c in corpus]
    n, avgdl = len(docs), max(1, sum(len(d) for d in docs) // max(1, len(docs)))
    df = {}
    for d in docs:
        for t in set(d): df[t] = df.get(t, 0) + 1
    return docs, df, n, avgdl

def bm25_search(q, docs, df, n, avgdl, k=10):
    qt = toks(q); scores = []
    for i, d in enumerate(docs):
        s, tf = 0.0, {}
        for t in d: tf[t] = tf.get(t, 0) + 1
        for t in qt:
            if t in tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (1 - 0.75 + 0.75 * len(d) / avgdl))
        scores.append((s, i))
    return sorted(scores, reverse=True)[:k]

# ---------- fusion + rerank + judge ----------
def rrf(*lists, k=60, weights=None):
    weights = weights or [1] * len(lists)
    agg = {}
    for w, lst in zip(weights, lists):
        for r, i in enumerate(lst):
            agg[i] = agg.get(i, 0.0) + w / (k + r + 1)
    return [i for i, _ in sorted(agg.items(), key=lambda x: -x[1])]

def bge_rerank(query, texts, top):
    texts = [t[:1800] for t in texts]
    body = {"query": query, "texts": texts, "raw_scores": False}
    r = urllib.request.Request(RERANK, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            out = json.loads(resp.read())
    except Exception as e:
        print("  [rerank-fail, passing through RRF order]", str(e)[:80])
        return list(range(min(top, len(texts)))), {}
    pairs = [(x["index"], x["score"]) for x in (out if isinstance(out, list) else [])] or list(enumerate(out or []))
    pairs.sort(key=lambda x: -x[1])
    return [i for i, _ in pairs[:top]], dict(pairs)

def llm_judge(query, cands, model):
    body = {"model": JUDGE_MODEL[model], "temperature": 0, "messages": [
        {"role": "user", "content": (
            "You are a code-aware retrieval judge. Query:\n" + query + "\n\nCandidates:\n" +
            "\n".join(f"[{i}] {c['kind']} {c['name']} — {c['file']} (calls_in={c.get('calls_in',0)})" for i, c in enumerate(cands)) +
            "\n\nScore each 0-10 (10=directly answers the query). Reply ONLY: ranked list of indexes like '4,0,7,2'.")}]}
    r = urllib.request.Request(LITELLM, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(r, timeout=90) as resp:
        txt = json.loads(resp.read())["choices"][0]["message"]["content"]
    idxs = [int(x) for x in re.findall(r"\d+", txt)][:len(cands)]
    return idxs or list(range(len(cands)))

def search(query, top=8, judge="none"):
    corpus = load_corpus()
    docs, df, n, avgdl = bm25_build(corpus)
    v = embed([query])[0]
    from qdrant_client import QdrantClient
    qc = QdrantClient(QDR, timeout=60)
    vec_hits = qc.query_points("lair-code", query=v, limit=20, with_payload=False).points
    vec_ids = {h.id for h in vec_hits}
    id_of = {pid(c["text"]): i for i, c in enumerate(corpus)}
    vec_list = [id_of[h.id] for h in vec_hits if h.id in id_of]
    bm_list = [i for _, i in bm25_search(query, docs, df, n, avgdl, 20)]
    fused = rrf(vec_list, bm_list, weights=[1.0, 0.7])[:12]
    cands = [corpus[i] for i in fused]
    # stage 2: bge cross-encoder (the frozen default)
    order, scores = bge_rerank(query, [c["text"] for c in cands], top)
    ranked = [cands[i] for i in order]
    # stage 3 (optional): LLM judge on the ambiguous top-set — bayesian_router hook lands here (d33)
    if judge != "none":
        jj = llm_judge(query, ranked[:6], judge)
        ranked = [ranked[:6][j] for j in jj if j < 6] + ranked[6:]
    score_by_id = {}
    for pos, c in enumerate(ranked):
        orig = fused[order[pos]] if pos < len(order) else None
        score_by_id[id(c)] = scores.get(order[pos], 0.0) if pos < len(order) else 0.0
    return [{"kind": c["kind"], "name": c["name"], "file": c["file"], "score": round(score_by_id[id(c)], 4)} for c in ranked]

if __name__ == "__main__":
    q = sys.argv[1]
    judge = "none"
    if "--judge" in sys.argv: judge = sys.argv[sys.argv.index("--judge") + 1]
    for r in search(q, judge=judge):
        print(f"{r['score']:6.3f}  {r['kind']:6} {r['name'][:44]:46} {r['file'].split('/')[-1]}")
