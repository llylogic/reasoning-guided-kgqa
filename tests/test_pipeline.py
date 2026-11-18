import json
import tempfile
import unittest
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from data_utils import ROOT, read_json, load_split, records, write_rows
from rgr.graph import Graph, serialize
from rgr.graph_index import DiskGraph, build
from rgr.pipeline import align
from rgr.models import Lexical
from rgr.corpus import distill, make_stages
from rgr.demo import demo, FixtureLM
from rgr.metrics import paired, score
from rgr.training_data import Encoded, training_row
from prompter import Prompter

C = read_json(ROOT / "config.json")


class PipelineTest(unittest.TestCase):
    def test_disk_induced_neighborhood(self):
        edges = [
            ("a", "r", "b"),
            ("c", "s", "b"),
            ("a", "p", "c"),
            ("b", "v", "outside"),
            ("isolated", "s", "other"),
        ]
        graph = Graph(edges)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.sqlite"
            source = Path(directory) / "triples.jsonl"
            write_rows(source, edges + edges)
            self.assertEqual(build(source, path), len(edges))
            indexed = DiskGraph(path)
            for topics in (["a"], ["other"], ["missing"], ["b", "other"]):
                for radius in (0, 1, 2, 4):
                    self.assertEqual(
                        indexed.local(topics, radius).edges,
                        graph.local(topics, radius).edges,
                    )
            self.assertRaises(FileExistsError, build, source, path)

    def test_cli_with_local_files(self):
        import run_qa

        row = next(records(ROOT / "examples/train.jsonl"))
        plan = next(records(ROOT / "examples/teacher_plans.jsonl"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = root / "paths.json"
            cfg = read_json(ROOT / "paths.json")
            cfg.update(
                data_dir="data",
                schema=str(ROOT / "examples/kg_schema.json"),
                output_dir="out",
            )
            paths.write_text(json.dumps(cfg))
            bench = root / "data/webqsp"
            reference = {
                k: row[k]
                for k in (
                    "dataset",
                    "split",
                    "id",
                    "gold",
                    "answer_ids",
                    "required_phrases",
                )
            }
            sample = {
                k: v
                for k, v in row.items()
                if k not in ("gold", "answer_ids", "required_phrases")
            }
            write_rows(bench / "train.jsonl", [sample])
            write_rows(bench / "test.jsonl", [dict(sample, split="test", id="eval")])
            write_rows(root / "reference.jsonl", [reference])
            write_rows(root / "teacher.jsonl", [plan])

            def entry(command, *args):
                with (
                    patch.object(
                        sys,
                        "argv",
                        ["run_qa.py", command, "--paths", str(paths), *args],
                    ),
                    redirect_stdout(StringIO()),
                ):
                    run_qa.main()

            tokenizer = SimpleNamespace(encode=lambda s, **kwargs: s.split())
            hf = SimpleNamespace(
                AutoTokenizer=SimpleNamespace(
                    from_pretrained=lambda *a, **kw: tokenizer
                )
            )
            student = FixtureLM("1. Book | author\n2. step 1 | home", "")
            with (
                patch.object(run_qa, "BGE", return_value=Lexical()),
                patch.object(run_qa, "LocalLM", return_value=student),
                patch.dict(sys.modules, transformers=hf),
            ):
                entry(
                    "distill",
                    "--teacher-plans",
                    str(root / "teacher.jsonl"),
                    "--reference-evidence",
                    str(root / "reference.jsonl"),
                )
                entry("prepare", "--reference-evidence", str(root / "reference.jsonl"))
                entry("infer")
            work = root / "out/webqsp"
            self.assertEqual(next(records(work / "plans.jsonl"))["id"], row["id"])
            self.assertEqual(len(list(records(work / "stage3.jsonl"))), 2)
            self.assertEqual(next(records(work / "test_predictions.jsonl"))["hit"], 1)

    def test_three_routes_and_logs(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "pred.jsonl"
            result = demo(C, out)
            rows = list(records(out))
            self.assertEqual(result["distill_attempts"], 2)
            self.assertEqual(
                [(r["fallback"], r["direct_one_hop"]) for r in rows],
                [(False, False), (True, False), (True, True)],
            )
            self.assertNotIn(["Ava", "home", "Cedar"], rows[-1]["retrieved"])
            self.assertEqual(paired(rows, rows, n=100)["ci95_pp"], [0, 0])

    def test_induced_radius_reverse_and_density(self):
        g = Graph([("a", "r", "b"), ("c", "s", "b"), ("b", "u", "d"), ("c", "v", "d")])
        local = g.local(["a"], 1)
        self.assertEqual(local.edges, {("a", "r", "b")})
        text, shown, _ = serialize(g.edges, ["b"], len, C)
        self.assertIn("r^-1", text)
        self.assertEqual(shown, g.edges)
        bounded, subset, before = serialize(
            g.edges, ["b"], len, dict(C, evidence_tokens=32)
        )
        self.assertLessEqual(len(bounded), 32)
        self.assertEqual(bounded.count("("), bounded.count(")"))
        self.assertGreater(before, len(bounded))
        self.assertLessEqual(subset, shown)

    def test_filter_split_dependencies_type_and_replay(self):
        row = next(records(ROOT / "examples/train.jsonl"))
        g = Graph(row["graph"])
        schema = read_json(ROOT / "examples/kg_schema.json")
        rank = Lexical()
        plan = "1. Book | author\n2. step 1 | home"
        bad = "1. Book | author\n2. step 2 | home"
        self.assertRaises(ValueError, align, bad, row["topics"], g, schema, rank, C)
        self.assertRaises(
            ValueError, align, "1. Book | home", row["topics"], g, schema, rank, C
        )
        self.assertRaises(
            ValueError,
            distill,
            dict(row, split="test"),
            g,
            schema,
            rank,
            C,
            lambda p, i: plan,
        )
        m = distill(
            dict(row, required_phrases=["latest"]),
            g,
            schema,
            rank,
            C,
            lambda p, i: plan,
        )
        self.assertIsNone(m["plan"])
        self.assertEqual(m["attempts"], 3)
        rows = [dict(row, id=str(i)) for i in range(20)]
        stages = make_stages(
            rows,
            lambda _: g,
            {("webqsp", r["id"]): plan for r in rows},
            schema,
            rank,
            len,
            C,
        )
        self.assertEqual([len(s) for s in stages], [60, 23, 44])
        qa = [x for x in stages[2] if not x.get("replay")]
        self.assertEqual(sum(x["task"] == "qa_question" for x in qa), 20)
        self.assertTrue(
            all(x["distractor_count"] == 1 for x in qa if x["task"] == "qa_context")
        )

    def test_old_input_names_and_target_only_loss(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cwq"
            path.mkdir()
            row = {
                "id": "q",
                "question": "why?",
                "q_entity": ["Root"],
                "graph": [],
                "answer": ["Name"],
            }
            (path / "test.jsonl").write_text(json.dumps(row))
            sample = next(load_split(d, "cwq", "test"))
            self.assertEqual(sample["topics"], ["Root"])
            self.assertEqual(sample["answers"], ["Name"])

        class Tok:
            eos_token_id = 2

            def encode(self, s, add_special_tokens=False):
                return ([1] if add_special_tokens else []) + [ord(c) for c in s]

        ds = Encoded(
            [dict(prompt="Question:", target="Yes", task="qa", id="a")],
            Tok(),
            Prompter(),
            100,
        )
        x = ds[0]
        self.assertEqual(x["labels"][-4:], [89, 101, 115, 2])
        self.assertTrue(all(v == -100 for v in x["labels"][:-4]))
        self.assertEqual(score(["Paris", "extra"], ["Paris"]), (1, 2 / 3))
        self.assertEqual(
            training_row({"input": "q", "output": "a"}, 0, "cwq")["target"], "a"
        )
        self.assertEqual(training_row({"text": "facts"}, 0, "cwq")["prompt"], "")
        self.assertRaises(
            ValueError, training_row, {"split": "test", "text": "facts"}, 0, "cwq"
        )


if __name__ == "__main__":
    unittest.main()
