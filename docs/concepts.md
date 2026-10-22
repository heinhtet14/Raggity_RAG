# RAG concepts, by experiment

Each section names a concept, points at the code, and gives an experiment to try. Run the
server with `--reload`, keep `/docs` open, and use one collection of documents you know well,
so you can tell good results from bad ones.

## 1. Why retrieve at all

A model only knows what it was trained on plus what is in its prompt. RAG puts the relevant
part of *your* documents in the prompt at question time. It is cheaper than fine-tuning and
updates as soon as you upload, and the answer can cite its source.
The hard part is retrieval: if the right passage isn't in the top k, no model can answer well.

**Try:** set `RAGGITY_LLM_PROVIDER=echo` and call `/answer`. The response is exactly the
prompt a model would get. Most RAG debugging is reading this prompt.

## 2. Chunking (`ingest/chunking.py`)

Documents are split into chunks because embeddings blur over long text, and because you can
only fit so much context into a prompt. With chunks that are too small, a chunk loses the
context that makes it meaningful ("It increased by 4%"). With chunks that are too large, the
one relevant sentence is diluted by everything around it. Overlap keeps a sentence that
straddles a boundary intact in at least one chunk.

**Try:** upload the same PDF twice to two collections, once with `chunker=fixed` and once
with `chunker=recursive`, and compare chunk boundaries in `/search` results. Then try
`chunk_size` 300 vs 1500 on a question whose answer spans two paragraphs.

## 3. Embeddings and dense search (`providers/embeddings.py`, `retrieval/search.py::_dense`)

An embedding model maps text to a vector so that similar meanings land close together.
Dense search embeds the query and ranks chunks by cosine similarity. Because the vectors are
normalised, that is just a dot product, which is what `matrix @ query` does.

The `hash` embedder only matches shared words. Comparing it with a real model is the
quickest way to see what "semantic" means.

**Try:** ask a paraphrase that shares no words with the document ("time off" vs "paid
leave") with `mode=dense`, first using `hash` and then using `sentence-transformers`.

## 4. Keyword search and BM25 (`store/sqlite_store.py::keyword_search`)

BM25 scores a chunk higher when it contains the query's words, especially rare ones, and
corrects for chunk length. Dense search is weak exactly where BM25 is strong: names, part
numbers, error codes, acronyms. Here it is SQLite's FTS5 with the Porter stemmer.

**Try:** search for a product code or a surname with `mode=dense` and then `mode=keyword`.

## 5. Hybrid search and RRF (`retrieval/search.py::reciprocal_rank_fusion`)

Run both searches and merge them. BM25 scores and cosine scores are on different scales, so
Reciprocal Rank Fusion ignores the scores and uses positions: `score = Σ 1/(60 + rank)`. A
chunk ranked well by both wins. Every result includes `scores` so you can see each stage's
contribution.

**Try:** find a query where dense and keyword disagree on the top result, then look at the
`dense[0]`, `bm25[0]` and `rrf` values for each hit in hybrid mode.

## 6. Query transformation (`retrieval/search.py::_transform`, `_hypothetical_answer`)

User questions are short and phrased unlike the documents.

- **multi_query** asks the LLM for three rephrasings, searches each one, and fuses all the
  rankings with RRF.
- **HyDE** (Hypothetical Document Embeddings) asks the LLM to write a fake answer and embeds
  that. Even if the fake answer is wrong, it is phrased like a document, so it lands nearer
  to the real passages than the question does.

Both cost an LLM call per search, so they trade latency for recall.

**Try:** a vague question ("what are the rules about money?") with `transform` set to
`none`, `multi_query`, then `hyde`. The `queries` field shows what was actually searched.

## 7. Reranking (`retrieval/rerank.py`)

First-stage search embeds query and chunk separately (a bi-encoder), which is fast but
rough. A cross-encoder reads them together and scores relevance properly, but it's too slow
to run on every chunk. So the service retrieves 30 candidates cheaply, reranks them, and
keeps the top k. The `llm` reranker does the same job with a prompt.

**Try:** `RAGGITY_RERANKER=cross-encoder`, then compare `rerank: false` and `rerank: true`
on the same query. Watch which hits move.

## 8. Metadata filtering

Each document can carry metadata (`user_id`, `source`, `date`...). `filter` keeps only
documents whose metadata matches. A companion agent needs this to keep one user's notes out
of another user's answers. Here the filter is applied to candidates before ranking, so it
never returns a match from outside the filter.

## 9. Grounded generation and citations (`generation.py`)

The prompt numbers the sources, tells the model to cite them, and tells it to say "I don't
know" when the sources don't contain the answer. The service parses the citation numbers
back out, so the caller can show the sources and detect an answer with no citations.

The sources are wrapped in tags and the model is told to treat them as data, not
instructions. Documents can contain text that looks like instructions ("ignore previous
instructions..."). This is prompt injection, and it matters most when the documents come
from the web or from users.

**Try:** upload a document that contains an instruction aimed at the model, and see whether
the answer obeys it.

## 10. Storage and deletion (`store/sqlite_store.py`)

Each chunk lives in three places: the row, its vector, and the BM25 index. A delete has to
remove all three, or a deleted document keeps turning up in search results. Here an FTS
trigger and one transaction take care of that, and `test_a_deleted_document_is_no_longer_found`
checks it for every search mode.

Brute-force search reads every vector, which is fine for thousands of chunks. Vector
databases use approximate indexes (HNSW, IVF) to stay fast at millions, in exchange for
occasionally missing a true neighbour. That trade is roadmap item 5.

## 11. Measuring it

Without evaluation, every change above is a guess. The minimum version:

1. Write 20 questions about your documents, and note which chunk answers each one.
2. **Hit rate@k**: how often the right chunk is in the top k.
   **MRR**: the average of 1/rank of the right chunk.
3. For answers, check by hand (or with an LLM judge) whether each claim is supported by a
   cited source. This is *faithfulness*.

Change one thing at a time (chunk size, mode, reranker) and re-measure. This is roadmap
item 1.
