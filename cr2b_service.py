"""Lair Search CR-2b: HTTP service wrapper (stdlib only - no new deps).
POST /search {"query": "...", "top": 8, "judge": "none|qwen|gemini", "client": "tag"}  -> ranked hits
GET  /health -> {"ok": true, "corpus": n}
Run: python cr2b_service.py [port=8025]   (n8n-webhook-callable; dockerized 2026-09-27 = B2)

B2: every /search request is appended as ONE JSONL line to $CR2B_LOG
(default: lair_search.log beside this file; container: /var/log/lair_search/queries.jsonl
on a named volume). Fields: ts (UTC), client (X-Client header or body "client",
default "unknown"), event, query, top, judge, status, latency_ms, results
(name/kind/file/score per hit) or error. Log failures never kill search.
"""
import json, sys, os, time
from http.server import BaseHTTPRequestHandler, HTTPServer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cr2_query

LOG = os.environ.get("CR2B_LOG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lair_search.log"))

class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _log_query(self, rec):
        """One JSONL line per query - H40 grades from this file, not a counter."""
        try:
            with open(LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")
        except Exception as e:
            print("log-fail:", e, file=sys.stderr)  # never kill search over logging

    def do_GET(self):
        if self.path == "/health":
            n = len(cr2_query.load_corpus()) if hasattr(cr2_query, "load_corpus") else -1
            self._send(200, {"ok": True, "service": "lair_search", "corpus_chunks": n})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/search":
            return self._send(404, {"error": "not found"})
        started = time.time()
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "client": (self.headers.get("X-Client") or "").strip()[:64] or "unknown",
               "event": "search"}
        try:
            ln = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(ln) or b"{}")
            q = body.get("query", "").strip()
            rec["query"] = q
            rec["top"] = int(body.get("top", 8))
            rec["judge"] = body.get("judge", "none")
            if body.get("client"):
                rec["client"] = str(body["client"])[:64]
            if not q:
                rec["status"], rec["error"] = 400, "query required"
                return self._send(400, {"error": "query required"})
            res = cr2_query.search(q, top=rec["top"], judge=rec["judge"])
            rec["status"] = 200
            rec["latency_ms"] = round((time.time() - started) * 1000, 1)
            rec["results"] = [{"name": r["name"], "kind": r["kind"], "file": r["file"], "score": r["score"]} for r in res]
            self._send(200, {"query": q, "results": res})
        except Exception as e:
            rec["status"] = 500
            rec["error"] = str(e)[:200]
            rec["latency_ms"] = round((time.time() - started) * 1000, 1)
            self._send(500, {"error": str(e)[:200]})
        finally:
            self._log_query(rec)  # single log point: 200, 400 and 500 all land here

    def log_message(self, *a):  # HTTP access log stays quiet; JSONL log above is the record
        pass

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8025
    print(f"lair_search service on :{port} (corpus warming... log={LOG})")
    cr2_query.load_corpus()
    print("corpus warm; serving")
    HTTPServer(("0.0.0.0", port), H).serve_forever()
