"""Exercise the three alignment outcomes on a small graph using fixed plans."""

from .graph import Graph
from .models import Lexical
from .pipeline import run
from .metrics import score, summary
from .corpus import distill, make_stages
from data_utils import write_rows


class FixtureLM:
    def __init__(self, first, retry):
        self.first, self.retry = first, retry

    def count(self, s):
        return len(s.split())

    def generate(self, prompt, stop=False):
        if prompt.startswith("Simplify"):
            return self.retry
        if stop:
            return self.first
        return (
            '1. Read the evidence.\nAnswer: ["Cedar"]'
            if "Cedar" in prompt
            else "Answer: []"
        )


def demo(cfg, output=None):
    edges = [
        ("Book", "author", "Ava"),
        ("Ava", "home", "Cedar"),
        ("Other", "home", "Delta"),
    ]
    graph = Graph(edges)
    rank = Lexical()
    plan = "1. Book | author\n2. step 1 | home"
    outputs = []
    for i, (first, retry) in enumerate([(plan, plan), ("?", plan), ("?", "?")]):
        row = dict(
            id=str(i),
            dataset="demo",
            split="test",
            question="Where does the author of Book live?",
            topics=["Book"],
        )
        out = run(row, graph, {}, FixtureLM(first, retry), rank, cfg)
        out["hit"], out["f1"] = score(out["prediction"], ["Cedar"])
        outputs.append(out)
    if output:
        write_rows(output, outputs)
    train = dict(
        row,
        id="train1",
        split="train",
        gold=edges[:2],
        answers=["Cedar"],
        answer_ids=["Cedar"],
        required_phrases=[],
    )
    manifest = distill(
        train, graph, {}, rank, cfg, lambda p, i: "?" if i == 0 else plan
    )
    stages = make_stages(
        [train],
        lambda _: graph,
        {("demo", "train1"): manifest["plan"]},
        {},
        rank,
        FixtureLM("", "").count,
        cfg,
    )
    return dict(
        mode="scripted_demo",
        inference=summary(outputs),
        distill_attempts=manifest["attempts"],
        examples_per_stage=[len(s) for s in stages],
    )
