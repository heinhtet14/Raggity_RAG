# Raggity RAG

A small RAG (retrieval-augmented generation) microservice, written to be read. You put
documents into **collections**; a companion agent calls **search** to fetch the passages
relevant to a question, or **answer** to get a reply that cites them.

Every stage of the pipeline is a plain Python module you can open, change and compare:

```
upload ─► load ─► chunk ─► embed ─► store (SQLite: vectors + BM25)
                                         │
query ─► [rewrite / HyDE] ─► dense + keyword ─► RRF fusion ─► [rerank] ─► top k ─► [answer + citations]
```

| Stage | File | What to learn there |
|---|---|---|
| Load | `src/raggity/ingest/loaders.py` | PDF → text per page, so citations can name a page |
| Chunk | `src/raggity/ingest/chunking.py` | fixed vs recursive splitting, overlap |
| Embed | `src/raggity/providers/embeddings.py` | what an embedding is, why a collection is tied to one model |
| Store | `src/raggity/store/sqlite_store.py` | vectors as blobs, FTS5/BM25, deletes that really delete |
| Retrieve | `src/raggity/retrieval/search.py` | dense vs keyword vs hybrid, RRF, multi-query, HyDE, metadata filters |
| Rerank | `src/raggity/retrieval/rerank.py` | bi-encoder vs cross-encoder, LLM-as-reranker |
| Generate | `src/raggity/generation.py` | grounded prompts, citations, refusing when the context lacks the answer |

[`docs/concepts.md`](docs/concepts.md) explains each concept and gives an experiment to run
for each. [`docs/agent-integration.md`](docs/agent-integration.md) shows how to give the
service to your companion agent as a tool.

## Run it

```bash
uv sync                       # base install: hash embeddings + echo LLM, no keys needed
uv run uvicorn raggity.main:app --reload
open http://127.0.0.1:8000/docs   # interactive API
```

```bash
curl -X POST localhost:8000/v1/collections -H 'content-type: application/json' -d '{"name": "notes"}'
curl -X POST localhost:8000/v1/collections/notes/documents -F file=@handbook.pdf -F 'metadata={"user_id": "u1"}'
curl -X POST localhost:8000/v1/collections/notes/search -H 'content-type: application/json' \
  -d '{"query": "how many days of leave?", "mode": "hybrid", "top_k": 5}'
```

With the defaults nothing leaves your machine: embeddings are word hashes and the
"LLM" echoes back the prompt it would have sent. That is useful for seeing exactly what a
model receives, but it isn't real RAG yet. Switch providers next.

## Providers

Everything is set with `RAGGITY_*` environment variables or a `.env` file (see
[`.env.example`](.env.example)). Install only the extras you use.

| Setting | Options | Extra |
|---|---|---|
| `RAGGITY_EMBEDDING_PROVIDER` | `hash` (default), `sentence-transformers`, `openai`, `ollama` | `local`, `openai` |
| `RAGGITY_LLM_PROVIDER` | `echo` (default), `anthropic`, `openai`, `ollama` | `anthropic`, `openai` |
| `RAGGITY_RERANKER` | `none` (default), `cross-encoder`, `llm` | `local` for cross-encoder |

A good first real setup, with free local embeddings and Claude for answers:

```bash
uv sync --extra local --extra anthropic
export RAGGITY_EMBEDDING_PROVIDER=sentence-transformers   # BAAI/bge-small-en-v1.5
export RAGGITY_LLM_PROVIDER=anthropic                      # claude-opus-5-5 by default
export RAGGITY_RERANKER=cross-encoder
export ANTHROPIC_API_KEY=...                               # or `ant auth login`
```

Fully local, no keys (needs [Ollama](https://ollama.com)):

```bash
ollama pull nomic-embed-text && ollama pull llama3.2
RAGGITY_EMBEDDING_PROVIDER=ollama RAGGITY_LLM_PROVIDER=ollama uv run uvicorn raggity.main:app
```

A collection records the embedder that built it. If you change the embedder, create a new
collection and re-upload, because vectors from different models can't be compared and the
service refuses to mix them.

The Anthropic provider streams the response, sets `output_config.effort` from
`RAGGITY_LLM_EFFORT` (default `medium`), and enables the server-side refusal fallback
(`fallbacks: "default"`), so a safety-classifier decline is retried on a fallback model.

## API

| Method | Path | |
|---|---|---|
| `POST` | `/v1/collections` | create `{name, description}` |
| `GET` | `/v1/collections` · `/v1/collections/{id-or-name}` | list / get |
| `DELETE` | `/v1/collections/{id}` | delete with all its documents |
| `POST` | `/v1/collections/{id}/documents` | multipart upload: `file`, optional `metadata` (JSON), `chunker`, `chunk_size`, `chunk_overlap` |
| `POST` | `/v1/collections/{id}/documents/text` | ingest text you already have `{filename, text, metadata}` |
| `GET` · `DELETE` | `/v1/collections/{id}/documents[/{doc}]` | list / delete (removed from both indexes immediately) |
| `POST` | `/v1/collections/{id}/search` | `{query, top_k, mode, transform, rerank, filter}` → passages with per-stage scores |
| `POST` | `/v1/collections/{id}/answer` | same body → `{answer, citations, sources}` |
| `GET` | `/healthz` | which embedder and LLM are loaded |

`mode`: `dense` | `keyword` | `hybrid`. `transform`: `none` | `multi_query` | `hyde`.
`filter`: exact match on document metadata, e.g. `{"user_id": "u1"}`.

## Operating it

- **Auth.** Set `RAGGITY_API_KEY` and every `/v1` route and `/metrics` requires
  `Authorization: Bearer <key>` (or `x-api-key`). `/healthz` stays open for liveness probes.
  This is service-to-service auth for your gateway, not visitor auth. Run Raggity on a
  private network either way.
- **Timings.** Every search, answer and upload response carries `timings_ms`, with time
  per stage (`load_vectors`, `embed_query`, `dense_scoring`, `bm25`, `rerank`, `generate`…)
  and a `total`.
- **Metrics.** `/metrics` serves Prometheus histograms: `raggity_stage_seconds{stage}`
  and `raggity_request_seconds{method,route,status}`, plus counters for ingested chunks,
  empty results and LLM errors. Routes are labelled by template, so collection ids
  never blow up the label set.
- **Vector cache.** `RAGGITY_VECTOR_CACHE=true` (the default) keeps each collection's
  matrix in memory until the collection changes. Each collection has a version counter
  in SQLite, so a write from another worker process invalidates the cache too.

### Rebuilding a collection from a folder

```bash
node scripts/export_portfolio.mjs ~/hein-htet-portfolio corpus/portfolio   # site data → Markdown
RAGGITY_API_KEY=... uv run raggity-ingest corpus/portfolio --collection portfolio --recreate
```

### Evaluating retrieval

```bash
uv run raggity-eval corpus/portfolio eval/portfolio.jsonl --modes dense,keyword,hybrid
```

See [`docs/evaluation.md`](docs/evaluation.md) for the dataset format, the metrics, and the
first results.

## Develop

```bash
uv run pytest           # no network: tests use the hash embedder and a fake LLM
uv run pyright
uv run ruff check . && uv run ruff format .
docker compose up --build
```

## Roadmap

Concepts worth adding next, roughly in order. Each is a self-contained change.

1. **Evaluation**: a small question/answer set, plus hit-rate@k, MRR and answer faithfulness
   (by hand first, then RAGAS). This lets you measure every other change.
2. **Contextual chunk headers**: prepend the document title and section heading to each chunk
   before embedding (Anthropic's "contextual retrieval").
3. **Parent-child retrieval**: match small chunks, return the larger section around them.
4. **MMR**: diversify results so the top 5 aren't near-duplicates.
5. **An ANN index**: swap brute force for HNSW (Qdrant, LanceDB or pgvector) and measure the
   recall/latency trade.
6. **Async ingestion**: queue large PDFs and report status instead of blocking the request.
7. **GraphRAG**: extract entities and relations into a graph, and answer "global" questions
   from community summaries (the idea behind LightRAG and Microsoft GraphRAG).
8. **Better parsing**: tables and scanned pages (OCR) with a layout-aware parser such as
   Docling or MinerU.
