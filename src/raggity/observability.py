"""Timings and Prometheus metrics.

Every search and ingest records how long each stage took. The numbers go two places:
back to the caller in the response (`timings_ms`), and into Prometheus histograms at
/metrics. The first is for looking at one slow request; the second for p50/p95 over time.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager

from prometheus_client import Counter, Histogram

# Buckets from 0.5 ms to 30 s: stages range from a dot product to an LLM call.
_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)

STAGE_SECONDS = Histogram(
    "raggity_stage_seconds", "Time spent in each pipeline stage", ["stage"], buckets=_BUCKETS
)
REQUEST_SECONDS = Histogram(
    "raggity_request_seconds",
    "HTTP request latency by route",
    ["method", "route", "status"],
    buckets=_BUCKETS,
)
CHUNKS_INGESTED = Counter("raggity_chunks_ingested_total", "Chunks written to the store")
EMPTY_RESULTS = Counter("raggity_empty_results_total", "Searches that returned no chunks", ["mode"])
LLM_ERRORS = Counter("raggity_llm_errors_total", "LLM calls that failed or were declined")


class Timings:
    """Accumulates wall-clock time per stage. A stage run several times (one dense search per
    multi-query rewrite) adds up, so the total reflects what the request really spent."""

    def __init__(self) -> None:
        self._seconds: dict[str, float] = {}
        self._start = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Generator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._seconds[name] = self._seconds.get(name, 0.0) + elapsed
            STAGE_SECONDS.labels(name).observe(elapsed)

    def as_ms(self) -> dict[str, float]:
        ms = {name: round(s * 1000, 3) for name, s in self._seconds.items()}
        ms["total"] = round((time.perf_counter() - self._start) * 1000, 3)
        return ms
