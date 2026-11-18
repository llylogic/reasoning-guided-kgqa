import json
import math
import random
import re
from .graph import Graph, serialize, triples, atom
from .models import normalize, planning, answering, ANSWER
from .pipeline import align, ranked, describe


TEACHER = """Using the question and the supplied training evidence, write a numbered retrieval plan.

For each step, identify the starting entity or an earlier step's result, the relation or constraint to retrieve, and its dependency on previous steps.

Keep intermediate and target entities as variables unless they are explicitly named in the question. Do not introduce facts or relations unsupported by the supplied evidence.

Question:
{question}

Training evidence:
{gold_evidence}

Retrieval plan:"""


def fit_evidence(steps, row, gold, meta):
    """Verify each planned traversal on supplied gold triples, including inverse hops."""
    graph = Graph(gold)
    bound = {}
    for step in steps:
        entities = bound.get(step.source, {step.source})
        if step.source not in bound and step.source not in row["topics"]:
            allowed = [step.source, *meta.get("aliases", {}).get(step.source, [])]
            if not any(
                " " + normalize(e) + " " in " " + normalize(row["question"]) + " "
                for e in allowed
            ):
                raise ValueError("named intermediate absent from question")
        reached = {
            other
            for e in entities
            for rel, other, _ in graph.adj.get(e, ())
            if rel == step.relation
        }
        if not reached:
            raise ValueError("step unsupported by gold evidence")
        bound[step.target] = reached
    if row.get("answer_ids") and not bound[steps[-1].target].intersection(
        row["answer_ids"]
    ):
        raise ValueError("terminal path misses reference entities")


def distill(row, graph, meta, ranker, cfg, generate):
    if row["split"] != "train":
        raise ValueError("distillation requires split=train")
    gold = triples(row["gold"])
    constraints = row["required_phrases"]
    if not isinstance(constraints, list) or not all(
        isinstance(p, str) and p.strip() for p in constraints
    ):
        raise ValueError("supply required_phrases from training query annotations")
    local = graph.local(row["topics"], cfg["radius"])
    prompt = TEACHER.format(
        question=row["question"],
        gold_evidence=json.dumps(sorted(gold), ensure_ascii=False),
    )
    reasons = []
    for i in range(cfg["attempts"]):
        plan = generate(prompt, i)
        if plan is None:  # saved candidates exhausted before the retry limit
            break
        try:
            steps = align(plan, row["topics"], local, meta, ranker, cfg)
            fit_evidence(steps, row, gold, meta)
            if any(
                " " + normalize(p) + " " not in " " + normalize(plan) + " "
                for p in constraints
            ):
                raise ValueError("missing annotated constraint")
            return dict(
                dataset=row["dataset"],
                split="train",
                id=row["id"],
                plan=plan,
                attempts=i + 1,
                rejections=reasons,
            )
        except ValueError as e:
            reasons.append(str(e))
    return dict(
        dataset=row["dataset"],
        split="train",
        id=row["id"],
        plan=None,
        attempts=len(reasons),
        rejections=reasons,
    )


def sample_walk(graph, start, size, rng):
    edges = []
    e = start
    for _ in range(size):
        candidates = [
            (n, edge) for r, n, edge in graph.adj.get(e, ()) if not r.endswith("^-1")
        ]
        if not candidates:
            break
        # Sample neighbor nodes by exp(deg); then choose a labeled edge to that node.
        by_node = {}
        for n, edge in candidates:
            by_node.setdefault(n, []).append(edge)
        nodes = sorted(by_node)
        degs = [len(graph.adj[n]) for n in nodes]
        peak = max(degs)
        e = rng.choices(nodes, weights=[math.exp(d - peak) for d in degs])[0]
        edges.append(rng.choice(by_node[e]))
    return edges


def make_stages(rows, getgraph, plans, meta, ranker, count, cfg):
    rng = random.Random(cfg["seed"])
    stages = [[], [], []]
    benchmark = None

    def add(i, task, row, prompt, target):
        stages[i].append(
            dict(
                task=task,
                id=row["id"],
                dataset=row["dataset"],
                prompt=prompt,
                target=target,
            )
        )

    for row in rows:
        benchmark = benchmark or row["dataset"]
        if row["dataset"] != benchmark or row["split"] != "train":
            raise ValueError("prepare one benchmark's train split at a time")
        graph = getgraph(row).local(row["topics"], cfg["radius"])
        if not graph.edges:
            continue
        h, r, t = rng.choice(sorted(graph.edges))
        # Equal template multiplicity gives equal sampling weight to the three tasks.
        add(
            0,
            "entity",
            row,
            "Recover the masked tail:\n(" + atom(h) + " (" + atom(r) + " [MASK]))",
            t,
        )
        add(
            0,
            "relation",
            row,
            "Recover the masked relation:\n(" + atom(h) + " ([MASK] " + atom(t) + "))",
            r,
        )
        path = sample_walk(graph, h, cfg["walk_length"], rng) or [(h, r, t)]
        view, shown, _ = serialize(path, [h], count, cfg)
        readable = " ".join(
            f"{a} is linked to {b} by {rel}." for a, rel, b in sorted(shown)
        )
        add(
            0, "reconstruction", row, "Describe this graph in words:\n" + view, readable
        )
        add(1, "next_token", row, "", view)
        plan = plans.get((row["dataset"], row["id"]))
        if not plan:
            continue
        gold = triples(row["gold"])
        distractors = ranked(
            row["question"], graph.edges - gold, lambda e: describe(e, meta), ranker
        )[: cfg["noise_pool"]]
        ratio = cfg["noise_fraction"]
        n = math.floor(len(gold) * ratio / (1 - ratio) + 0.5)
        chosen = distractors[:n]
        view, _, _ = serialize(gold.union(chosen), row["topics"], count, cfg)
        target = (
            plan.strip() + ANSWER + " " + json.dumps(row["answers"], ensure_ascii=False)
        )
        add(2, "qa_question", row, planning(row["question"]), target)
        add(2, "qa_context", row, answering(row["question"], view), target)
        stages[2][-1].update(gold_count=len(gold), distractor_count=len(chosen))
    if any(not s for s in stages):
        raise ValueError(
            "empty curriculum stage; check accepted plans, splits and graph coverage"
        )
    history = []
    output = []
    for stage in stages:
        extra = rng.sample(history, math.floor(cfg["replay"] * len(history) + 0.5))
        combined = stage + [dict(x, replay=True) for x in extra]
        rng.shuffle(combined)
        output.append(combined)
        history.extend(stage)
    return output
