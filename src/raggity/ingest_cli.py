"""Rebuild a collection from a folder, over HTTP, so it works against a deployed service.

    RAGGITY_API_KEY=... uv run raggity-ingest corpus/portfolio --collection portfolio --recreate

Each file is uploaded with its path relative to the folder as the filename (so answers
cite `projects/mindroid.md`) and as `metadata.source`. `--recreate` deletes the collection
first, which makes the command safe to re-run after you edit the corpus.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

SUPPORTED = {".pdf", ".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".html"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild a collection from a folder.")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument("--url", default=os.environ.get("RAGGITY_URL", "http://localhost:8000"))
    parser.add_argument("--recreate", action="store_true", help="delete the collection first")
    parser.add_argument("--chunker", choices=["recursive", "fixed"])
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--chunk-overlap", type=int)
    args = parser.parse_args(argv)

    key = os.environ.get("RAGGITY_API_KEY")
    headers = {"authorization": f"Bearer {key}"} if key else {}
    files = sorted(
        p for p in args.folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED
    )
    if not files:
        print(f"no supported files under {args.folder}", file=sys.stderr)
        return 1

    with httpx.Client(base_url=args.url, headers=headers, timeout=300) as http:
        existing = http.get(f"/v1/collections/{args.collection}")
        if existing.status_code == 200 and args.recreate:
            http.delete(f"/v1/collections/{existing.json()['id']}").raise_for_status()
            print(f"deleted collection {args.collection}")
        elif existing.status_code == 200:
            print(
                f"collection {args.collection} exists; pass --recreate to rebuild it",
                file=sys.stderr,
            )
            return 1
        elif existing.status_code != 404:
            existing.raise_for_status()

        created = http.post(
            "/v1/collections", json={"name": args.collection, "description": args.description}
        )
        created.raise_for_status()
        cid = created.json()["id"]

        form = {
            k: str(v)
            for k, v in {
                "chunker": args.chunker,
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.chunk_overlap,
            }.items()
            if v is not None
        }
        total_chunks = 0
        for path in files:
            name = path.relative_to(args.folder).as_posix()
            form["metadata"] = json.dumps({"source": name})
            response = http.post(
                f"/v1/collections/{cid}/documents",
                files={"file": (name, path.read_bytes())},
                data=form,
            )
            if response.status_code != 201:
                print(f"  ✗ {name}: {response.status_code} {response.text}", file=sys.stderr)
                continue
            body = response.json()
            total_chunks += body["chunk_count"]
            print(f"  ✓ {name}: {body['chunk_count']} chunks, {body['timings_ms']['total']:.0f} ms")
        print(f"{len(files)} files, {total_chunks} chunks into {args.collection} ({cid})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
