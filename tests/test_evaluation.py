from __future__ import annotations

import json
from pathlib import Path

import pytest

from raggity.config import Settings
from raggity.evaluation import (
    Config,
    first_relevant_rank,
    load_dataset,
    main,
    percentile,
    recall,
    run,
)


def test_rank_is_one_based_and_none_when_missing() -> None:
    assert first_relevant_rank(["a", "b", "c"], {"b"}) == 2
    assert first_relevant_rank(["a", "b"], {"z"}) is None


def test_recall_counts_documents_not_chunks() -> None:
    # Two chunks from the same relevant document still count as one document found.
    assert recall(["a", "a", "b"], {"a", "c"}) == 0.5


def test_percentile_uses_nearest_rank() -> None:
    values = [float(v) for v in range(1, 21)]
    assert percentile(values, 50) == 10
    assert percentile(values, 95) == 19
    assert percentile([], 50) == 0


def test_a_question_without_relevant_files_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps({"id": "x", "question": "?", "relevant": [], "tags": []}))
    with pytest.raises(ValueError, match="needs 'relevant'"):
        load_dataset(path)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    (root / "pets").mkdir(parents=True)
    (root / "pets" / "cats.md").write_text("Cats sleep twelve to sixteen hours a day.")
    (root / "pets" / "dogs.md").write_text("Dogs need daily walks and exercise.")
    (root / "parts.md").write_text("The X-200 filter has part number QF-7731.")
    return root


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    rows = [
        {
            "id": "cats",
            "question": "how long do cats sleep",
            "relevant": ["pets/cats.md"],
            "tags": ["paraphrase"],
        },
        {"id": "part", "question": "QF-7731", "relevant": ["parts.md"], "tags": ["exact-term"]},
    ]
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_run_scores_every_config(tmp_path: Path, corpus: Path, dataset: Path) -> None:
    configs = [Config(200, 20, "recursive", mode, "none", False) for mode in ("dense", "keyword")]
    results = run(corpus, load_dataset(dataset), configs, 3, Settings(data_dir=tmp_path))

    assert [r.config.mode for r in results] == ["dense", "keyword"]
    keyword = results[1].summary()
    assert keyword["hit@1"] == 1.0 and keyword["mrr"] == 1.0
    assert results[1].summary("exact-term")["n"] == 1
    assert "bm25" in results[1].questions[0].timings_ms


def test_main_writes_a_results_file(
    tmp_path: Path, corpus: Path, dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "results"
    main([str(corpus), str(dataset), "--modes", "hybrid", "--k", "2", "--out", str(out)])

    assert "| hybrid @ recursive/800/120 |" in capsys.readouterr().out
    [written] = out.glob("*.json")
    report = json.loads(written.read_text())
    assert report["results"][0]["summary"]["n"] == 2
    assert "embed_query" in report["results"][0]["stage_latency_ms"]
