# CR-2b lair-search code-search service - 0.3.3: B3a chunk-TEXT windows merged into corpus (2026-09-27)
# HTTP layer = stdlib http.server (kept from cr2b_service.py, per ticket).
# Only pip dep = qdrant_client (python client dialect, L: qdrant 1.19).
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cr1_index.py cr2_query.py cr2b_service.py corpus_cache.json ./

# Backend endpoints via host.docker.internal (replicates the old localhost
# process; qdrant/tei/tei-reranker/litellm all publish ports on the host).
# Query log: one JSONL line per /search request (H40-consumable).
ENV CR2B_LOG=/var/log/lair-search/queries.jsonl \
    CR_QDRANT_URL=http://host.docker.internal:6333 \
    CR_TEI_URL=http://host.docker.internal:8082/embed \
    CR_RERANK_URL=http://host.docker.internal:8084/rerank \
    CR_LITELLM_URL=http://host.docker.internal:4000/v1/chat/completions

VOLUME /var/log/lair-search
EXPOSE 8025

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8025/health', timeout=4).status==200 else 1)"

CMD ["python", "cr2b_service.py", "8025"]
