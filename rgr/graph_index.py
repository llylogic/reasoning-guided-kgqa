"""Disk-backed Freebase triple index with the same induced neighborhoods as Graph."""

import argparse
from contextlib import closing
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from data_utils import graph_triples
from .graph import Graph, validated_triple


def build(source, target):
    source, target = Path(source), Path(target)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as directory:
        pending = Path(directory) / "triples.sqlite"
        with closing(sqlite3.connect(pending)) as db, db:
            db.execute(
                "CREATE TABLE triples (h TEXT, r TEXT, t TEXT, PRIMARY KEY(h,r,t)) WITHOUT ROWID"
            )
            db.executemany(
                "INSERT OR IGNORE INTO triples VALUES (?,?,?)",
                (validated_triple(edge) for edge in graph_triples(source)),
            )
            db.execute("CREATE INDEX by_tail ON triples(t,h,r)")
            (count,) = db.execute("SELECT COUNT(*) FROM triples").fetchone()
        os.link(pending, target)
    return count


def fetch(db, nodes, by_tail=False):
    nodes = list(nodes)
    column = "t" if by_tail else "h"
    for i in range(0, len(nodes), 400):
        batch = nodes[i : i + 400]
        markers = ",".join("?" for _ in batch)
        yield from db.execute(
            f"SELECT h,r,t FROM triples WHERE {column} IN ({markers})", batch
        )


class DiskGraph:
    def __init__(self, path):
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(p)
        self.uri = p.resolve().as_uri() + "?mode=ro"

    def local(self, topics, k):
        with closing(sqlite3.connect(self.uri, uri=True)) as db:
            frontier = set(topics)
            seen = set(frontier)
            for _ in range(k):
                neighbors = {t for _, _, t in fetch(db, frontier)}
                neighbors.update(h for h, _, _ in fetch(db, frontier, by_tail=True))
                frontier = neighbors - seen
                if not frontier:
                    break
                seen.update(frontier)
            return Graph((h, r, t) for h, r, t in fetch(db, seen) if t in seen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", help="JSONL canonical triples, or an archived graph.json"
    )
    parser.add_argument("output", help="new .sqlite file")
    args = parser.parse_args()
    print(f"Indexed {build(args.source, args.output)} unique triples.")
