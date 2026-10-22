"""Retrieval evaluation: does the right document come back, how high, and how fast?

    uv run raggity-eval corpus/portfolio eval/portfolio.jsonl \\
        --modes dense,keyword,hybrid --chunk-sizes 400,800

For each chunk size it builds a fresh, throwaway collection from the corpus folder with
the providers from your environment (RAGGITY_EMBEDDING_PROVIDER etc.), then runs every
question under every mode/transform/rerank combination and reports:

- hit@k:    share of questions with at least one relevant document in the top k chunks
- MRR:      mean of 1/rank of the first relevant chunk (0 if none in the top k)
- recall@k: share of each question's relevant documents found in the top k, averaged
- p50/p95:  latency per stage and in total, across all questions

Relevance is judged per *document* (the file a chunk came from), not per chunk, so the
dataset stays valid when you change the chunk size. Results are also written as JSON to
eval/results/ so you can plot them or compare runs later.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from raggity.config import Settings
from raggity.ingest.loaders import UnsupportedFileType
from raggity.retrieval.search import Mode, Transform
from raggity.service import RagService
from raggity.store.sqlite_store import Store

# -- metrics (pure functions, unit-tested) ------------------------------------------


def first_relevant_rank(ranked_files: list[str], relevant: set[str]) -> int | None:
    """1-based rank of the first chunk whose document is relevant."""
    for rank, filename in enumerate(ranked_files, start=1):
        if filename in relevant:
            return rank
    return None


def recall(ranked_files: list[str], relevant: set[str]) -> float:
    return len(relevant & set(ranked_files)) / len(relevant) if relevant else 0.0


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile; with 35 questions, interpolating adds false precision."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(q / 100 * len(ordered)) - 1)
    return ordered[index]


# -- dataset ------------------------------------------------------------------------


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    relevant: frozenset[str]
    tags: tuple[str, ...] = ()


def load_dataset(path: Path) -> list[Question]:
    questions = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("relevant"):
            raise ValueError(f"{path}:{number}: a retrieval question needs 'relevant' files")
        questions.append(
            Question(row["id"], row["question"], frozenset(row["relevant"]), tuple(row["tags"]))
        )
    return questions


# -- running ------------------------------------------------------------------------


@dataclass
class Config:
    chunk_size: int
    chunk_overlap: int
    chunker: str
    mode: str
    transform: str
    rerank: bool

    def label(self) -> str:
        extra = "".join(
            [f"+{self.transform}" if self.transform != "none" else "", "+rerank" * self.rerank]
        )
        return f"{self.mode}{extra} @ {self.chunker}/{self.chunk_size}/{self.chunk_overlap}"


@dataclass
class QuestionResult:
    id: str
    tags: tuple[str, ...]
    rank: int | None
    recall: float
    retrieved: list[str]
    timings_ms: dict[str, float]


@dataclass
class ConfigResult:
    config: Config
    k: int
    questions: list[QuestionResult] = field(default_factory=list)

    def summary(self, tag: str | None = None) -> dict[str, float]:
        rows = [q for q in self.questions if tag is None or tag in q.tags]
        n = len(rows) or 1
        totals = [q.timings_ms["total"] for q in rows]
        return {
            "n": len(rows),
            "hit@1": sum(q.rank == 1 for q in rows) / n,
            f"hit@{self.k}": sum(q.rank is not None for q in rows) / n,
            "mrr": sum(1 / q.rank for q in rows if q.rank) / n,
            f"recall@{self.k}": sum(q.recall for q in rows) / n,
            "p50_ms": percentile(totals, 50),
            "p95_ms": percentile(totals, 95),
        }

    def stage_latency(self) -> dict[str, dict[str, float]]:
        stages: dict[str, list[float]] = {}
        for q in self.questions:
            for stage, ms in q.timings_ms.items():
                stages.setdefault(stage, []).append(ms)
        return {s: {"p50": percentile(v, 50), "p95": percentile(v, 95)} for s, v in stages.items()}


def build_collection(service: RagService, corpus: Path, config: Config) -> str:
    collection = service.create_collection("eval", f"eval run: {config.label()}")
    files = sorted(p for p in corpus.rglob("*") if p.is_file())
    for path in files:
        try:
            service.ingest_file(
                collection,
                path.relative_to(corpus).as_posix(),
                path.read_bytes(),
                {},
                config.chunker,
                config.chunk_size,
                config.chunk_overlap,
            )
        except UnsupportedFileType:
            print(f"  skipping {path.name}: unsupported type", file=sys.stderr)
    return collection.id


def run(
    corpus: Path,
    questions: list[Question],
    configs: list[Config],
    k: int,
    settings: Settings,
    repeat: int = 1,
) -> list[ConfigResult]:
    results: list[ConfigResult] = []
    services: dict[tuple[int, int, str], tuple[RagService, str]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        shared: RagService | None = None
        for config in configs:
            key = (config.chunk_size, config.chunk_overlap, config.chunker)
            if key not in services:
                # One store per chunking; providers (slow to load) are shared between them.
                store = Store(Path(tmp) / f"{len(services)}.sqlite3", settings.vector_cache)
                service = (
                    RagService(settings, store=store)
                    if shared is None
                    else RagService(settings, store, shared.embedder, shared.llm, shared.reranker)
                )
                shared = shared or service
                print(
                    f"ingesting {corpus} with {config.chunker}/{key[0]}/{key[1]}", file=sys.stderr
                )
                services[key] = (service, build_collection(service, corpus, config))
            service, cid = services[key]
            collection = service.store.get_collection(cid)
            outcome = ConfigResult(config, k)
            for q in questions:
                for attempt in range(repeat):
                    result = service.search(
                        collection,
                        q.question,
                        k,
                        cast(Mode, config.mode),
                        cast(Transform, config.transform),
                        config.rerank,
                        {},
                    )
                    if attempt < repeat - 1:
                        continue  # warm-up runs: only the last one is recorded
                    files = [hit.chunk.filename for hit in result.hits]
                    outcome.questions.append(
                        QuestionResult(
                            q.id,
                            q.tags,
                            first_relevant_rank(files, set(q.relevant)),
                            recall(files, set(q.relevant)),
                            files,
                            result.timings.as_ms(),
                        )
                    )
            results.append(outcome)
            print(f"  done {config.label()}", file=sys.stderr)
    return results


# -- reporting ----------------------------------------------------------------------


def render_table(results: list[ConfigResult], tags: list[str]) -> str:
    k = results[0].k
    head = f"| config | n | hit@1 | hit@{k} | MRR | recall@{k} | p50 ms | p95 ms |"
    lines = [head, "|" + "---|" * 8]
    for r in results:
        for tag in [None, *tags]:
            s = r.summary(tag)
            if tag and not s["n"]:
                continue
            name = r.config.label() if tag is None else f"  └ {tag}"
            lines.append(
                f"| {name} | {s['n']} | {s['hit@1']:.2f} | {s[f'hit@{k}']:.2f} | {s['mrr']:.3f}"
                f" | {s[f'recall@{k}']:.2f} | {s['p50_ms']:.1f} | {s['p95_ms']:.1f} |"
            )
    return "\n".join(lines)


def misses(results: list[ConfigResult]) -> str:
    lines = []
    for r in results:
        missed = [q for q in r.questions if q.rank is None]
        if missed:
            lines.append(f"{r.config.label()}: missed " + ", ".join(q.id for q in missed))
    return "\n".join(lines)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality and latency.")
    parser.add_argument("corpus", type=Path, help="folder of documents to ingest")
    parser.add_argument("dataset", type=Path, help="JSONL of {id, question, relevant, tags}")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--modes", default="dense,keyword,hybrid")
    parser.add_argument("--transforms", default="none", help="none,multi_query,hyde")
    parser.add_argument("--rerank", default="off", choices=["off", "on", "both"])
    parser.add_argument("--chunk-sizes", default="800")
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--chunker", default="recursive", choices=["recursive", "fixed"])
    parser.add_argument("--repeat", type=int, default=1, help="runs per question; last is kept")
    parser.add_argument("--out", type=Path, default=Path("eval/results"))
    args = parser.parse_args(argv)

    settings = Settings()
    questions = load_dataset(args.dataset)
    reranks = {"off": [False], "on": [True], "both": [False, True]}[args.rerank]
    configs = [
        Config(size, min(args.chunk_overlap, size // 2), args.chunker, mode, transform, rr)
        for size in [int(s) for s in args.chunk_sizes.split(",")]
        for mode in args.modes.split(",")
        for transform in args.transforms.split(",")
        for rr in reranks
    ]
    results = run(args.corpus, questions, configs, args.k, settings, args.repeat)

    tags = sorted({tag for q in questions for tag in q.tags})
    print(
        f"\n{len(questions)} questions · embedder {settings.embedding_provider}"
        f" · llm {settings.llm_provider} · reranker {settings.reranker}"
        f" · vector cache {'on' if settings.vector_cache else 'off'}\n"
    )
    print(render_table(results, tags))
    missed = misses(results)
    if missed:
        print("\n" + missed)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = args.out / f"{stamp}.json"
    path.write_text(
        json.dumps(
            {
                "created_at": stamp,
                "git_sha": _git_sha(),
                "dataset": str(args.dataset),
                "corpus": str(args.corpus),
                "k": args.k,
                "providers": {
                    "embedding": f"{settings.embedding_provider}:{settings.embedding_model}",
                    "llm": f"{settings.llm_provider}:{settings.llm_model}",
                    "reranker": settings.reranker,
                    "vector_cache": settings.vector_cache,
                },
                "results": [
                    {
                        "config": asdict(r.config),
                        "summary": r.summary(),
                        "by_tag": {tag: r.summary(tag) for tag in tags},
                        "stage_latency_ms": r.stage_latency(),
                        "questions": [asdict(q) for q in r.questions],
                    }
                    for r in results
                ],
            },
            indent=2,
        )
    )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
