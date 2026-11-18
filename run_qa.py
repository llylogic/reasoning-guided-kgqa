"""Prepare KGQA data, run retrieval-guided inference, and evaluate predictions."""

import argparse
import json
from collections import Counter
from pathlib import Path
from data_utils import (
    ROOT,
    load_split,
    records,
    write_rows,
    read_json,
    resolved,
    graph_triples,
)
from rgr.graph import Graph
from rgr.graph_index import DiskGraph
from rgr.models import LocalLM, BGE
from rgr.pipeline import run
from rgr.corpus import distill, make_stages
from rgr.metrics import score, summary, paired


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument(
        "command", choices=["distill", "prepare", "infer", "bootstrap", "demo"]
    )
    p.add_argument("--paths", default=ROOT / "paths.json")
    p.add_argument("--config", default=ROOT / "config.json")
    p.add_argument("--dataset", choices=["webqsp", "cwq"], default="webqsp")
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--device", default="cuda")
    p.add_argument("--retriever-device", default="cpu")
    p.add_argument(
        "--teacher-device", default="auto", help="HF device_map=auto by default"
    )
    p.add_argument(
        "--teacher-trust-local-code",
        action="store_true",
        help="allow custom Python modules from your local teacher checkpoint",
    )
    p.add_argument(
        "--teacher-plans", help="offline JSONL: dataset, split=train, id, candidates"
    )
    p.add_argument(
        "--reference-evidence",
        help="training-only JSONL sidecar with grounded gold triples and required_phrases",
    )
    p.add_argument(
        "--checkpoint", help="merged local checkpoint; default is merged3 in this run"
    )
    p.add_argument("--output", help="predictions or demo output")
    p.add_argument("--f1", choices=["set", "legacy"], default="set")
    p.add_argument("--left")
    p.add_argument("--right")
    return p.parse_args()


def index(rows, ds):
    result = {}
    for r in rows:
        key = str(r["id"])
        if key in result or r["dataset"] != ds or r["split"] != "train":
            raise ValueError(f"Duplicate ID, wrong split or benchmark: {key}")
        result[key] = r
    return result


def main():
    args = arguments()
    cfg = read_json(args.config)
    if args.command == "demo":
        from rgr.demo import demo

        print(json.dumps(demo(cfg, args.output), indent=2))
        return
    if args.command == "bootstrap":
        if not (args.left and args.right):
            raise ValueError("bootstrap needs --left and --right")
        print(json.dumps(paired(records(args.left), records(args.right)), indent=2))
        return
    paths = resolved(args.paths)
    ds = args.dataset
    work = Path(paths["output_dir"]) / ds
    rows = load_split(
        paths["data_dir"], ds, args.split if args.command == "infer" else "train"
    )
    if args.reference_evidence and args.command != "infer":
        side = index(records(args.reference_evidence), ds)

        def annotated(source):
            for row in source:
                annotation = side[row["id"]]
                yield dict(
                    row,
                    **{
                        name: annotation[name]
                        for name in ("gold", "required_phrases", "answer_ids")
                        if name in annotation
                    },
                )

        rows = annotated(rows)
    meta = read_json(paths["schema"]) if paths["schema"] else {}
    global_graph = None
    if paths["graph"]:
        if Path(paths["graph"]).suffix in (".db", ".sqlite", ".sqlite3"):
            global_graph = DiskGraph(paths["graph"])
        else:
            global_graph = Graph(graph_triples(paths["graph"]))

    def graph(row):
        if global_graph is not None:
            return global_graph
        if "graph" not in row:
            raise ValueError(f"{row['id']}: supply graph triples or set paths.graph")
        return Graph(row["graph"])

    ranker = BGE(paths["bge"], args.retriever_device)
    if args.command == "distill":
        if args.teacher_plans:
            offline = index(records(args.teacher_plans), ds)

            def proposals(row):
                items = offline[row["id"]]["candidates"]
                if not isinstance(items, list) or not all(
                    isinstance(s, str) for s in items
                ):
                    raise ValueError("expected list of teacher outputs")
                return lambda prompt, i: items[i] if i < len(items) else None
        else:
            teacher = LocalLM(
                paths["teacher"],
                args.teacher_device,
                max_new=2048,
                trust_local_code=args.teacher_trust_local_code,
            )

            def proposals(row):
                return lambda prompt, i: teacher.generate_teacher(prompt)

        def manifest():
            for row in rows:
                yield distill(row, graph(row), meta, ranker, cfg, proposals(row))

        path = work / "plans.jsonl"
        write_rows(path, manifest())
        stats = Counter(
            "accepted" if row["plan"] else "rejected" for row in records(path)
        )
        print(json.dumps(dict(stats)))
        return
    if args.command == "prepare":
        from transformers import AutoTokenizer

        tk = AutoTokenizer.from_pretrained(paths["backbone"], local_files_only=True)
        plans = index(records(work / "plans.jsonl"), ds)

        def matched(source):
            seen = set()
            for r in source:
                if r["id"] not in plans:
                    raise ValueError(f"Missing teacher manifest entry for {r['id']}")
                seen.add(r["id"])
                yield r
            if seen != plans.keys():
                raise ValueError("teacher manifest contains extra training IDs")

        rows = matched(rows)
        stages = make_stages(
            rows,
            graph,
            {(ds, k): v["plan"] for k, v in plans.items()},
            meta,
            ranker,
            lambda s: len(tk.encode(s, add_special_tokens=False)),
            cfg,
        )
        for i, data in enumerate(stages, 1):
            write_rows(work / f"stage{i}.jsonl", data)
        print(json.dumps({"training_rows_including_replay": [len(s) for s in stages]}))
        return
    lm = LocalLM(args.checkpoint or work / "merged3", args.device)
    out = Path(args.output or work / f"{args.split}_predictions.jsonl")

    def predictions():
        for row in rows:
            result = run(row, graph(row), meta, lm, ranker, cfg)
            result["hit"], result["f1"] = score(
                result["prediction"], row["answers"], args.f1
            )
            result["f1_profile"] = args.f1
            yield result

    write_rows(out, predictions())
    print(json.dumps(summary(records(out)), indent=2))


if __name__ == "__main__":
    main()
