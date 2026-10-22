"""Step 2 of ingestion: split text into chunks small enough to embed and to quote.

Two strategies, so you can compare them on the same document:

- fixed:     cut every `size` characters. Simple, but it cuts mid-sentence.
- recursive: try to split on paragraph breaks, then lines, then sentences, then words,
             and only fall back to a hard cut when a piece is still too long. This is the
             idea behind LangChain's RecursiveCharacterTextSplitter.

Both add `overlap` characters from the end of one chunk to the start of the next, so a
sentence that straddles a boundary still appears whole in at least one chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

from raggity.ingest.loaders import Section

SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int | None
    index: int  # position within the document


def chunk_sections(sections: list[Section], strategy: str, size: int, overlap: int) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    split = fixed_split if strategy == "fixed" else recursive_split
    chunks: list[Chunk] = []
    for section in sections:
        for piece in split(section.text, size, overlap):
            if piece.strip():
                chunks.append(Chunk(piece.strip(), section.page, len(chunks)))
    return chunks


def fixed_split(text: str, size: int, overlap: int) -> list[str]:
    step = size - overlap
    return [text[start : start + size] for start in range(0, max(len(text), 1), step)]


def recursive_split(text: str, size: int, overlap: int) -> list[str]:
    # Pieces leave room for the overlap, so the tail carried into each chunk always fits.
    pieces = _split_to_fit(text, size - overlap, SEPARATORS)
    return _merge_with_overlap(pieces, size, overlap)


def _split_to_fit(text: str, size: int, separators: list[str]) -> list[str]:
    """Break text into pieces no longer than size, preferring the earliest separator."""
    if len(text) <= size:
        return [text]
    if not separators:
        return fixed_split(text, size, 0)
    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)
    if len(parts) == 1:
        return _split_to_fit(text, size, rest)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        # Keep the separator so sentences and paragraphs read naturally when rejoined.
        part = part + sep if i < len(parts) - 1 else part
        pieces.extend(_split_to_fit(part, size, rest))
    return pieces


def _merge_with_overlap(pieces: list[str], size: int, overlap: int) -> list[str]:
    """Pack small pieces together up to size, carrying an overlap tail between chunks."""
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            # Start the overlap at a word boundary so the next chunk doesn't open mid-word.
            if " " in tail:
                tail = tail[tail.index(" ") + 1 :]
            current = tail
        current += piece
    if current:
        chunks.append(current)
    return chunks
