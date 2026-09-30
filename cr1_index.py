"""Lair Search CR-1: crawler JSON -> chunks -> TEI embeddings -> qdrant 'lair-code'.
Usage: python cr1_index.py <index.json> [<index2.json> ...]   (idempotent: rebuilds collection)
DoD: then run  cr1_query()  tests below via `python cr1_index.py --test`"""
import json, sys, os, hashlib, re, gzip
import urllib.request
from qdrant_client import QdrantClient, models

TEI = os.environ.get("CR_TEI_URL", "http://localhost:8082/embed")
QDR = os.environ.get("CR_QDRANT_URL", "http://localhost:6333")

def req(url, body=None, method=None):
    r = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                               headers={"Content-Type": "application/json"}, method=method or ("POST" if body is not None else "GET"))
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} {url}: {e.read().decode(errors='ignore')[:300]}") from None

def embed(texts, batch=32):
    out = []
    for i in range(0, len(texts), batch):
        out.extend(req(TEI, {"inputs": texts[i:i+batch]}))
    return out

def chunks_from(idx_path):
    d = json.load(open(idx_path, encoding="utf-8"))
    root = d.get("root_path", "")
    syms = d.get("symbols", {})   # {"kind:name": [files]}
    calls = d.get("calls", {})    # {callee: [callers...]}
    src = idx_path.replace("\\", "/").split("/")[-1]
    chunks = []
    for sid, files in syms.items():
        kind, _, name = sid.partition(":")
        f = (files[0] if files else "?").replace("\\", "/")
        lang = f.rsplit(".", 1)[-1] if "." in f else "?"
        ncalls = len(calls.get(name, []))
        text = (f"{kind} {name} defined in {f}. Language {lang}. "
                f"Referenced by {ncalls} call sites. "
                f"Module {'/'.join(f.split('/')[-2:])}.")
        chunks.append({"text": text, "payload": {"name": name, "kind": kind, "type": "symbol", "file": f, "lang": lang, "calls_in": ncalls, "src": src}})
    # per-file inventory chunk
    byfile = {}
    for sid, files in syms.items():
        for f in files:
            byfile.setdefault(f.replace("\\", "/"), []).append(sid)
    for f, sids in byfile.items():
        lang = f.rsplit(".", 1)[-1] if "." in f else "?"
        text = f"File {f} (language {lang}) defines: {', '.join(sids[:60])}."
        chunks.append({"text": text, "payload": {"name": f.split("/")[-1], "kind": "file", "type": "symbol", "file": f, "lang": lang, "n_symbols": len(sids), "src": src}})
    # H38 (B3a): text-body chunk windows emitted by crawler (kind=text_chunk) —
    # only present in v2-format indexes; absent key -> zero text chunks (old behavior).
    for fpath, fdata in d.get("files", {}).items():
        for tc in fdata.get("text_chunks", []):
            f = fpath.replace("\\", "/")
            lang = f.rsplit(".", 1)[-1] if "." in f else "?"
            text = f"In {f} lines {tc['start_line']}-{tc['end_line']}: {tc['text']}"
            chunks.append({"text": text, "payload": {"name": f.split("/")[-1], "kind": "text_chunk", "type": "text",
                         "file": f, "lang": lang, "lines": [tc["start_line"], tc["end_line"]], "src": src}})
    return chunks

def pid(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:15], 16)

def coll():
    """Target qdrant collection. Default 'lair-code' (unchanged legacy behavior);
    override via CR_COLLECTION env or --coll <name> — e.g. lair-code-v2 for H38
    so the production collection is never rebuilt in place."""
    return os.environ.get("CR_COLLECTION", "lair-code")

def build(index_paths):
    C = coll()
    allc = []
    for p in index_paths:
        c = chunks_from(p)
        print(f"{p}: {len(c)} chunks")
        allc.extend(c)
    # H38: dedupe by chunk id BEFORE embedding — multi-index builds (e.g. old json
    # for symbol chunks + v2 json for text chunks) regenerate identical symbol
    # chunks; embedding the duplicates only wasted minutes. Output identical.
    seen_pre = set(); uniq = []
    for c in allc:
        i = pid(c["text"])
        if i in seen_pre: continue
        seen_pre.add(i); uniq.append(c)
    print(f"after pid-dedup: {len(uniq)} unique chunks (was {len(allc)})")
    allc = uniq
    key = hashlib.md5("|".join(c["text"] for c in allc).encode()).hexdigest()[:12]
    cache = f"vecs_{key}.json"
    cache_gz = cache + ".gz"
    # B2 fix: the old `gzip.open(...) if cache.endswith(".gz")` read branch was dead
    # code (cache name was always plain .json and writes were never gzipped).
    # Now symmetric: prefer .gz, keep legacy .json readable, write .gz on new embeds.
    if os.path.exists(cache_gz):
        vecs = json.load(gzip.open(cache_gz, "rt", encoding="utf-8"))
        print(f"vectors from cache (gzip): {len(vecs)}")
    elif os.path.exists(cache):
        vecs = json.load(open(cache, encoding="utf-8"))
        print(f"vectors from cache (legacy plain): {len(vecs)}")
    else:
        vecs = embed([c["text"] for c in allc])
        with gzip.open(cache_gz, "wt", encoding="utf-8") as f:
            json.dump(vecs, f)
        print("embedded:", len(vecs), "dim:", len(vecs[0]), "->", cache_gz)
    qc = QdrantClient(QDR, timeout=120)
    try: qc.delete_collection(C)
    except Exception: pass
    qc.create_collection(C, vectors_config=models.VectorParams(size=len(vecs[0]), distance=models.Distance.COSINE))
    pts, seen = [], set()
    for c, v in zip(allc, vecs):
        i = pid(c["text"])
        if i in seen: continue
        seen.add(i)
        pts.append({"id": i, "vector": v, "payload": c["payload"]})
    qc = QdrantClient(QDR, timeout=120)
    B = 256
    for i in range(0, len(pts), B):
        qc.upsert(C, points=[models.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"]) for p in pts[i:i+B]], wait=True)
    print("upserted", len(pts), "points; count:", qc.get_collection(C).points_count)
    print(f"qdrant {C}:", req(f"{QDR}/collections/{C}")["result"]["points_count"], "points")

def test():
    queries = ["ASTPatchEngine", "pocketflow core pipeline", "experiment runner metrics",
               "posterior bayesian router", "DSPy signature slot fill", "spacy parse node"]
    for q in queries:
        v = embed([q])[0]
        r = req(f"{QDR}/collections/{coll()}/points/search", {"vector": v, "limit": 3, "with_payload": True})
        hits = [(h["payload"]["kind"], h["payload"]["name"], h["payload"]["file"].split("/")[-1], round(h["score"], 3)) for h in r["result"]]
        print(f"Q: {q}\n   {hits}")

if __name__ == "__main__":
    if "--coll" in sys.argv:
        os.environ["CR_COLLECTION"] = sys.argv[sys.argv.index("--coll") + 1]
    if "--test" in sys.argv:
        test()
    else:
        # positionals = args that are neither flags nor the value of --coll
        skip = False
        pos = []
        for a in sys.argv[1:]:
            if skip:
                skip = False
                continue
            if a == "--coll":
                skip = True
                continue
            if a.startswith("--"):
                continue
            pos.append(a)
        build(pos)
