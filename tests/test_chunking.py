from __future__ import annotations

import pytest

from raggity.ingest.chunking import chunk_sections, fixed_split, recursive_split
from raggity.ingest.loaders import Section

TEXT = (
    "Raggity is a RAG service. It splits documents into chunks.\n\n"
    "Each chunk is embedded. Search compares the query with every chunk.\n\n"
    "Answers cite the chunks they used, so you can check them."
)


def test_fixed_split_overlaps_by_the_requested_amount() -> None:
    pieces = fixed_split("abcdefghij", size=4, overlap=2)
    assert pieces[:3] == ["abcd", "cdef", "efgh"]


def test_recursive_split_keeps_chunks_within_size() -> None:
    for chunk in recursive_split(TEXT, size=80, overlap=20):
        assert len(chunk) <= 80


def test_recursive_split_prefers_paragraph_boundaries() -> None:
    chunks = recursive_split(TEXT, size=80, overlap=0)
    assert chunks[0].startswith("Raggity is a RAG service.")
    assert chunks[0].rstrip().endswith("chunks.")


def test_recursive_split_carries_overlap_into_the_next_chunk() -> None:
    chunks = recursive_split(TEXT, size=80, overlap=30)
    assert any(word in chunks[1] for word in chunks[0].split()[-3:])


def test_a_word_longer_than_the_chunk_size_is_hard_cut() -> None:
    chunks = recursive_split("x" * 250, size=100, overlap=0)
    assert [len(c) for c in chunks] == [100, 100, 50]


def test_chunks_keep_their_page_and_order() -> None:
    chunks = chunk_sections(
        [Section(TEXT, page=1), Section("Second page text.", page=2)], "recursive", 80, 10
    )
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[-1].page == 2


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        chunk_sections([Section(TEXT)], "fixed", 50, 50)
