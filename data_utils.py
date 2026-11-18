"""Local split reader for the old workspace/data/<dataset> layout and JSONL exports."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parent


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf8"))


def records(p):
    path = Path(p)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches():
            yield from batch.to_pylist()
    elif path.suffix == ".json":
        rows = read_json(path)
        if not isinstance(rows, list):
            raise ValueError(f"{p}: expected JSON array")
        yield from rows
    else:
        with path.open(encoding="utf8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = NamedTemporaryFile("w", encoding="utf8", dir=path.parent, delete=False)
    try:
        with tmp:
            for row in rows:
                tmp.write(json.dumps(row, ensure_ascii=False) + "\n")
        Path(tmp.name).replace(path)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def load_split(directory, dataset, split):
    base = Path(directory) / dataset
    jsonl = base / (split + ".jsonl")
    files = [jsonl] if jsonl.exists() else sorted(base.glob(split + "*.parquet"))
    if not files:
        # The archived CWQ directory was named webcwq. Select a single snapshot explicitly if several exist.
        aliases = [base] + ([Path(directory) / "webcwq"] if dataset == "cwq" else [])
        choices = [
            folder
            for root in aliases
            for folder in root.glob("datasets/snapshots/*/data")
            if any(folder.glob(split + "*.parquet"))
        ]
        if len(choices) > 1:
            raise ValueError(
                f"Several snapshots found. Point a separate data root at one version: {choices}"
            )
        if choices:
            files = sorted(choices[0].glob(split + "*.parquet"))
    if not files:
        raise FileNotFoundError(f"Supply {jsonl} or {base}/{split}*.parquet")
    seen = set()
    for path in files:
        for row in records(path):
            if (
                row.get("split", split) != split
                or row.get("dataset", dataset) != dataset
            ):
                raise ValueError(f"Wrong split or benchmark in {path}")
            # Legacy parquet uses q_entity / graph / answer, already linked.
            identifier = row.get("id")
            if identifier is None or str(identifier) in seen:
                raise ValueError(
                    f"Missing or duplicate question id in {path}: {identifier}"
                )
            seen.add(str(identifier))
            topics = row.get("topics", row.get("q_entity"))
            answers = row.get("answers", row.get("answer", []))
            if (
                not isinstance(topics, list)
                or not topics
                or not all(isinstance(v, str) and v for v in topics)
            ):
                raise ValueError(
                    f"{identifier}: expected a nonempty list of linked topic IDs or graph names"
                )
            if not isinstance(answers, list) or not all(
                isinstance(v, str) for v in answers
            ):
                raise ValueError(
                    f"{identifier}: supply answer labels as a list of strings"
                )
            yield dict(
                row,
                id=str(identifier),
                dataset=dataset,
                split=split,
                topics=topics,
                answers=answers,
            )


def resolved(pathfile):
    p = Path(pathfile).resolve()
    return {
        k: (str((p.parent / v).resolve()) if v is not None else None)
        for k, v in read_json(p).items()
    }


def graph_triples(path):
    """Accept original graph.json (head -> list of triples) or a JSONL triple stream."""
    if Path(path).suffix == ".json":
        doc = read_json(path)
        if isinstance(doc, dict):
            for edges in doc.values():
                yield from edges
        elif isinstance(doc, list):
            yield from doc
        else:
            raise ValueError("Expected a triple array or head-to-triples dictionary")
    else:
        yield from records(path)
