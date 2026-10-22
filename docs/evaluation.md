# Evaluating retrieval

`raggity-eval` answers three questions about a configuration: does the right document
come back, how high does it rank, and how long does it take? Run it before and after every
change. If a change doesn't move these numbers, it didn't help.

## The dataset

`eval/portfolio.jsonl` has one question per line:

```json
{"id": "robot-arm", "question": "Which robot arm did the LLM-to-robotics adapter use?",
 "relevant": ["projects/fc-robot.md"], "tags": ["exact-term"]}
```

- `relevant` lists every file that answers the question. Relevance is per **document**,
  so the dataset still works when you change chunk size or chunker, because chunk ids
  change and file names don't.
- `tags` let you slice the results:
  - `exact-term`: the question contains a name, number or acronym that appears in the
    document (SO-ARM101, TOEIC, Kubernetes). BM25 should be strong here.
  - `paraphrase`: the question avoids the document's words ("what happens when a user
    interrupts" vs "barge-in"). This is what dense embeddings are for.
  - `multi-doc`: more than one file is needed for a full answer. Watch recall@k.

The 35 questions were drafted from `src/data/profile.ts` on the portfolio site. **Review
them yourself.** A question you wouldn't ask, or a `relevant` list you disagree with,
makes every number below wrong. Add the questions real visitors ask once you have logs.

Out-of-scope questions ("what's his salary?") have no relevant document, so they can't be
scored here. They belong in the answer-level eval (roadmap): did the agent say it doesn't know?

## The metrics

| Metric | Meaning | Why you care |
|---|---|---|
| hit@1 | share of questions where the top chunk is from a relevant file | the model leans hardest on the first source |
| hit@k | at least one relevant file in the top k | if this misses, no LLM can answer |
| MRR | mean of 1/rank of the first relevant chunk | rewards ranking it high, not just including it |
| recall@k | share of the relevant files found, averaged | matters for multi-doc questions |
| p50 / p95 ms | retrieval latency per question | p95 is what a visitor notices |

The JSON in `eval/results/` also has per-stage p50/p95, and every question's retrieved
files, so you can see *why* a config lost.

## Running

```bash
node scripts/export_portfolio.mjs ~/hein-htet-portfolio corpus/portfolio

# compare modes and chunk sizes
uv run raggity-eval corpus/portfolio eval/portfolio.jsonl --modes dense,keyword,hybrid --chunk-sizes 400,800

# with a real embedder and the cross-encoder, reranking on and off
uv sync --extra local
RAGGITY_EMBEDDING_PROVIDER=sentence-transformers RAGGITY_RERANKER=cross-encoder \
  uv run raggity-eval corpus/portfolio eval/portfolio.jsonl --modes dense,hybrid --rerank both

# latency only means something with warm-up: --repeat 3 records the third run
uv run raggity-eval corpus/portfolio eval/portfolio.jsonl --modes hybrid --repeat 3
```

The eval runs in-process against a throwaway store, so it measures retrieval without
HTTP. For end-to-end latency under load, use the load test (roadmap step 3).

## First results (2026-10-22)

Portfolio corpus: 23 files exported from the site (25 chunks at size 400). 35 questions,
k = 5, MacBook CPU, `--repeat 3`. Embedder `BAAI/bge-small-en-v1.5` (384 dims), reranker
`cross-encoder/ms-marco-MiniLM-L-6-v2`. Raw output is in `eval/results/` (git-ignored).

| config (chunk 400 / overlap 120) | hit@1 | hit@5 | MRR | recall@5 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| hash "embedder", dense | 0.37 | 0.74 | 0.516 | 0.68 | 0.1 | 0.1 |
| keyword (BM25) | 0.69 | 0.97 | 0.806 | 0.90 | 0.2 | 0.3 |
| dense, bge-small | **0.97** | 1.00 | **0.986** | 0.96 | 5.9 | 6.5 |
| hybrid, bge-small | 0.94 | 1.00 | 0.967 | 0.93 | 6.3 | 6.7 |
| dense + rerank | 0.97 | 1.00 | 0.986 | 0.96 | 43.0 | 47.3 |
| keyword + rerank | 0.97 | 1.00 | 0.986 | 0.95 | 27.8 | 39.1 |
| hybrid + rerank | 0.97 | 1.00 | 0.986 | 0.96 | 42.1 | 45.4 |

At chunk size 800, dense drops to 0.89 hit@1 (paraphrase questions suffer most), while
hybrid holds at 0.94.

What this says, and what it doesn't:

- **A real embedding model is the single biggest win.** Going from word hashing to
  bge-small takes hit@1 from 0.37 to 0.97. Everything else is fine-tuning.
- **On this corpus the eval is nearly saturated.** With 23 short documents, almost every
  config finds the answer in the top 5. That means this dataset can no longer tell good
  configs apart. It needs harder questions, or a bigger corpus (blog posts, repo READMEs),
  before it's useful for comparing rerankers or hybrid weights.
- **Hybrid didn't beat dense here.** BM25 on its own ranks short, keyword-dense files
  (`projects/memory-eval.md`) first for many unrelated questions. RRF then pulls those into
  hybrid's ranking. Worth revisiting with a larger corpus, where the exact-term advantage
  should show.
- **The reranker costs 6–9× the retrieval latency for no gain over dense.** It does rescue
  plain BM25 (0.69 → 0.97), which is the cheapest stack if you can't run an embedder.
- **Latency is all model inference.** Of hybrid's 6.2 ms p50, `embed_query` is 5.8 ms.
  Loading vectors, scoring, BM25 and fusion together take under 0.3 ms. At this size
  storage and the vector index don't matter. The vector cache makes `load_vectors` 15×
  faster (p50 0.046 ms uncached, 0.003 ms cached), which still adds up to nothing against
  a 6 ms embedding. It will matter once the corpus has thousands of chunks.
- **Bias warning:** the questions were written by someone who had read the corpus, so they
  share its vocabulary more than real visitors' questions will. One label was corrected
  after the first run (`mindroid-lang`: `profile.md` does name Mindroid's language). That
  was a dataset fix, not tuning. Still, treat these numbers as an upper bound until you
  have real visitor questions.
