"""EXP-046 ES-7 bake-off (B1): 6-7 retrieval arms x 20 labeled queries.
Arms: A=raw-vector(lair-code) | B=RRF | C=B+bge-rerank | C-live=:8025 HTTP verify |
D=C+LLM judge on blind-ambiguous subset | A-v2 / A-v2-symbol = raw vector on
lair-code-v2 (staging; vector-only, no BM25 sidecar for v2 corpus — noted).
E=agentic-tools: pending-coordination (3 probes failed, see notes.md).
Read-only against :8025/qdrant. Grading: deterministic label match, blind to arm.
Usage: python es7_bakeoff.py <outdir>
"""
import json, math, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
EXP = os.path.join(HERE, "..", "..", "experiments", "EXP-046-20260927-es7-bakeoff")
sys.path.insert(0, EXP)  # labels.py (pre-registered, lives with the experiment)
import cr2_query
from cr1_index import embed, QDR
from labels import CASES, BASELINE_N, HOLDSET_N

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "experiments", "EXP-046-20260927-es7-bakeoff")
AMB_MARGIN = 0.10   # blind ambiguity: C rerank score(top1)-score(top2) < this
NDCG_K = 5


def fmt_hits(hits, n=None):
    return [f"{h['name']} {h['file'].rsplit('/', 1)[-1]}".lower() for h in (hits[:n] if n else hits)]


def grade(names, toks):
    """returns (top1, top3, rel_list for nDCG over ranked names)"""
    low = [t.lower() for t in toks]
    t1 = bool(names) and any(t in names[0] for t in low)
    t3 = any(t in n for n in names[:3] for t in low)
    rels = [1.0 if any(t in n for t in low) else 0.0 for n in names[:NDCG_K]]
    return int(t1), int(t3), rels


def ndcg5(rels):
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(NDCG_K, int(sum(rels)) or 1)))
    return dcg / idcg if idcg else 0.0


def main():
    corpus = cr2_query.load_corpus()
    docs, df, n, avgdl = cr2_query.bm25_build(corpus)
    qc = cr2_query.qc if hasattr(cr2_query, "qc") else __import__("qdrant_client").QdrantClient(QDR, timeout=60)
    rows = {}   # arm -> list of dicts
    lat = {}    # arm -> ms list
    amb_idx = []  # ambiguous subset indices (decided from arm C margins)

    print(f"[bakeoff] {len(CASES)} queries; corpus={len(corpus)}")
    for qi, (q, toks) in enumerate(CASES):
        v = embed([q])[0]

        # --- arm A: raw vector (lair-code) ---
        t0 = time.time()
        vh = qc.query_points("lair-code", query=v, limit=8, with_payload=True).points
        a_hits = [{"name": h.payload.get("name", ""), "file": h.payload.get("file", "")} for h in vh]
        lat.setdefault("A", []).append((time.time() - t0) * 1000)
        rows.setdefault("A", []).append({"q": q, "hits": a_hits})

        # --- arm B: weighted RRF, no rerank ---
        t0 = time.time()
        vec_ids = {h.id for h in vh}
        id_of = {cr2_query.pid(c["text"]): i for i, c in enumerate(corpus)}
        vec_list = [id_of[h.id] for h in vh if h.id in id_of]
        bm_list = [i for _, i in cr2_query.bm25_search(q, docs, df, n, avgdl, 20)]
        fused = cr2_query.rrf(vec_list, bm_list, weights=[1.0, 0.7])[:8]
        b_hits = [{"name": corpus[i]["name"], "file": corpus[i]["file"]} for i in fused]
        lat.setdefault("B", []).append((time.time() - t0) * 1000)
        rows.setdefault("B", []).append({"q": q, "hits": b_hits})

        # --- arm C: B + bge rerank (same stage functions as live :8025) ---
        t0 = time.time()
        cands = [corpus[i] for i in fused]
        order, scores = cr2_query.bge_rerank(q, [c["text"] for c in cands], 8)
        c_hits = [{"name": cands[i]["name"], "file": cands[i]["file"]} for i in order]
        lat.setdefault("C", []).append((time.time() - t0) * 1000)
        rows.setdefault("C", []).append({"q": q, "hits": c_hits})
        sc = [scores.get(i, 0.0) for i in order[:2]]
        if len(sc) == 2 and (sc[0] - sc[1]) < AMB_MARGIN:
            amb_idx.append(qi)

        # --- arm C-live: HTTP :8025 (service path verification) ---
        t0 = time.time()
        try:
            req = urllib.request.Request("http://127.0.0.1:8025/search",
                data=json.dumps({"query": q, "top": 8}).encode(),
                headers={"Content-Type": "application/json", "X-Client": "exp046"})
            out = json.loads(urllib.request.urlopen(req, timeout=240).read())
            cl_hits = [{"name": h["name"], "file": h["file"]} for h in out["results"]]
        except Exception as e:
            cl_hits, _ = [], print(f"  [C-live fail] {str(e)[:80]}")
        lat.setdefault("C-live", []).append((time.time() - t0) * 1000)
        rows.setdefault("C-live", []).append({"q": q, "hits": cl_hits})

        # --- arms A-v2 / A-v2-symbol: staging collection (vector-only) ---
        for arm, flt in (("A-v2", None), ("A-v2-symbol", {"must": [{"key": "kind", "match": {"value": "symbol"}}]})):
            t0 = time.time()
            try:
                hv = qc.query_points("lair-code-v2", query=v, limit=8, query_filter=flt, with_payload=True).points
                h2 = [{"name": h.payload.get("name", ""), "file": h.payload.get("file", "")} for h in hv]
            except Exception as e:
                h2, _ = [], print(f"  [{arm} fail] {str(e)[:80]}")
            lat.setdefault(arm, []).append((time.time() - t0) * 1000)
            rows.setdefault(arm, []).append({"q": q, "hits": h2})
        print(f"  q{qi+1:02d} done (amb={'Y' if qi in amb_idx else '-'})")

    # --- arm D: C + LLM judge, ONLY on ambiguous subset (blind: margins from C) ---
    d_rows = [dict(r) for r in rows["C"]]   # pass-through C for non-ambiguous
    dtoks = 0
    for qi in amb_idx:
        q, _ = CASES[qi]
        cands = rows["C"][qi]["hits"][:6]
        t0 = time.time()
        try:
            cand = [{"kind": "symbol", "name": h["name"], "file": h["file"], "calls_in": 0} for h in cands]
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):  # cr2_query may print
                jj = cr2_query.llm_judge(q, cands, "qwen")
            d_ranked = [cands[j] for j in jj if j < 6] + rows["C"][qi]["hits"][6:]
        except Exception as e:
            d_ranked, _ = rows["C"][qi]["hits"], print(f"  [D fail q{qi+1}] {str(e)[:80]}")
        lat.setdefault("D", []).append((time.time() - t0) * 1000)
        d_rows[qi] = {"q": q, "hits": d_ranked}
    rows["D"] = d_rows
    if amb_idx:
        print(f"[D] ambiguous subset = {[i+1 for i in amb_idx]} (C margin < {AMB_MARGIN})")

    # --- scoring ---
    metrics = {}
    for arm, rs in rows.items():
        t1 = t3 = 0
        ndcgs = []
        perq = []
        for qi, r in enumerate(rs):
            names = fmt_hits(r["hits"])
            a1, a3, rels = grade(names, CASES[qi][1])
            t1 += a1; t3 += a3; ndcgs.append(ndcg5(rels))
            perq.append({"qi": qi + 1, "top1": a1, "top3": a3, "ndcg5": round(ndcg5(rels), 3)})
        base_n = len(CASES) - HOLDSET_N if arm in ("A", "B", "C", "C-live", "D") else len(CASES)
        hs = perq[-HOLDSET_N:]
        sub = [p for p in perq[:-HOLDSET_N]] if arm in ("A", "B", "C", "C-live", "D") else perq
        l = lat[arm]
        metrics[arm] = {
            "top1": t1, "top3": t3, "n_queries": len(rs),
            "top1_ex_holdset": sum(p["top1"] for p in sub), "top3_ex_holdset": sum(p["top3"] for p in sub),
            "n_ex_holdset": len(sub),
            "ndcg5_mean": round(sum(ndcgs) / len(ndcgs), 4),
            "ndcg5_ex_holdset": round(sum(p["ndcg5"] for p in sub) / max(len(sub), 1), 4),
            "holdset_top3": f"{sum(p['top3'] for p in hs)}/{HOLDSET_N}" if arm in ("A-v2", "A-v2-symbol") else "n/a (corpus lacks bandit tree)" if arm in ("A", "B", "C", "C-live", "D") else f"{sum(p['top3'] for p in hs)}/{HOLDSET_N}",
            "latency_ms_mean": round(sum(l) / len(l), 1),
        }
        json.dump(perq, open(os.path.join(OUT, f"perq_{arm}.json"), "w"), indent=1)

    # arm D tokens ledger
    if amb_idx:
        metrics["D"]["notes"] = f"judge on {len(amb_idx)} ambiguous queries only; model or-free-qwen-qwen3-coder-free via litellm :4000"
        metrics["D"]["tokens_query_judge"] = 1  # judge prompt ~ (6 cands x name+file); token counting unavailable headless — noted as gap

    # deltas for H35/H36 verdict material
    verdict = {
        "H35_C_vs_B_top3_delta": metrics["C"]["top3"] - metrics["B"]["top3"],
        "H35_C_vs_B_ndcg5_delta": round(metrics["C"]["ndcg5_mean"] - metrics["B"]["ndcg5_mean"], 4),
        "H35_C_vs_A_top3_delta": metrics["C"]["top3"] - metrics["A"]["top3"],
        "H36_D_vs_C_on_ambiguous": round(
            sum(p["top3"] for p in json.load(open(os.path.join(OUT, "perq_D.json")))) - 0, 0),  # detailed below
        "ambiguous_idx_1based": [i + 1 for i in amb_idx],
    }
    # H36 detail: D vs C restricted to ambiguous subset
    dperq = json.load(open(os.path.join(OUT, "perq_D.json"))) if amb_idx else []
    cperq = json.load(open(os.path.join(OUT, "perq_C.json")))
    if amb_idx:
        verdict["H36_D_vs_C_ambig_top3"] = [int(dperq[i]["top3"]) - int(cperq[i]["top3"]) for i in amb_idx]
        verdict["H36_D_vs_C_ambig_ndcg5_delta"] = round(sum(dperq[i]["ndcg5"] - cperq[i]["ndcg5"] for i in amb_idx) / len(amb_idx), 4)

    json.dump({"arms": metrics, "verdict_material": verdict, "config": {
        "ndcg_k": NDCG_K, "amb_margin": AMB_MARGIN, "reranker": "bge-v2-m3 :8084",
        "judge": "qwen3-coder via litellm :4000", "collection_live": "lair-code (9579)",
        "collection_staging": "lair-code-v2 (18852, vector-only arms)",
        "holdset": "Q19-Q20 frozen pre-scoring; bandit tree only in v2 corpus"} },
        open(os.path.join(OUT, "metrics.json"), "w"), indent=1)

    # results table
    print("\n=== RESULTS (20 queries; ex-holdset = 18) ===")
    hdr = f"{'arm':12} {'top1':>5} {'top3':>5} {'t3/18':>6} {'nDCG@5':>7} {'ms/q':>7}"
    print(hdr)
    for arm in ("A", "B", "C", "C-live", "D", "A-v2", "A-v2-symbol"):
        m = metrics.get(arm)
        if not m:
            continue
        print(f"{arm:12} {m['top1']:>5} {m['top3']:>5} {m['top3_ex_holdset']:>3}/{m['n_ex_holdset']} {m['ndcg5_mean']:>7.4f} {m['latency_ms_mean']:>7.1f}")
    print("\nverdict material:", json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
