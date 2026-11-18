import re
from dataclasses import dataclass, asdict
from .graph import Graph, serialize
from .models import planning, rewrite, answering, answers, normalize


def ranked(query, items, describe, ranker):
    seq = sorted(items)
    scores = ranker.scores(query, [describe(x) for x in seq])
    return [
        x for _, x in sorted(zip(scores, seq), key=lambda pair: (-pair[0], pair[1]))
    ]


@dataclass
class Step:
    source: str
    relation: str
    target: str
    phrase: str


def mentions(text, names):
    text = " " + normalize(text) + " "
    return [a for a in names if a and " " + a + " " in text]


def align(plan, topics, graph, meta, ranker, cfg):
    rows = re.findall(r"(?mi)^\s*(?:step\s+)?(\d+)[.):\s-]+\s*(.+?)\s*$", plan)
    if not rows or [int(n) for n, _ in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("missing or invalid numbered steps")
    schema = meta.get("relations", {})
    aliases = {}
    for ent in sorted(set(graph.adj) | set(topics)):
        for name in [ent, *meta.get("aliases", {}).get(ent, [])]:
            aliases.setdefault(normalize(name), set()).add(ent)
    rels = sorted({r for _, r, _ in graph.edges})
    if not rels:
        raise ValueError("empty relation vocabulary")
    relnames = {}
    for r in rels:
        for name in [r, r.split(".")[-1], *schema.get(r, {}).get("aliases", [])]:
            relnames.setdefault(normalize(name), set()).add(r)
    types = {e: set(t) for e, t in meta.get("types", {}).items()}
    steps = []
    for idx, (_, line) in enumerate(rows, 1):
        lhs, sep, rhs = line.partition("|")
        phrase = rhs.strip() if sep else line
        refs = re.findall(r"\b(?:step\s*|x)(\d+)\b", lhs, re.I)
        names = mentions(lhs, aliases)
        if refs:
            if len(set(refs)) != 1 or not 1 <= int(refs[0]) < idx:
                raise ValueError("invalid step dependency")
            source = f"x{int(refs[0])}"
        elif names:
            longest = max(map(len, names))
            matched = set.union(*(aliases[n] for n in names if len(n) == longest))
            if len(matched) != 1:
                raise ValueError("ambiguous mention")
            source = next(iter(matched))
        elif sep:
            raise ValueError("unresolved source " + lhs)
        elif idx > 1:
            source = f"x{idx - 1}"
        elif len(topics) == 1:
            source = topics[0]
        else:
            raise ValueError("unresolved topic")
        inverse = "^-1" in phrase or bool(
            re.search(r"\b(?:inverse|reverse)\b", phrase, re.I)
        )
        matches = mentions(phrase.replace("^-1", ""), relnames)
        if matches:
            longest = max(map(len, matches))
            ids = set.union(*(relnames[n] for n in matches if len(n) == longest))
            if len(ids) != 1:
                raise ValueError("ambiguous relation")
            r = next(iter(ids))
        else:
            scores = ranker.scores(
                phrase,
                [
                    schema.get(r, {}).get("description", r.replace(".", " "))
                    for r in rels
                ],
            )
            value = max(scores)
            if value < cfg["relation_threshold"] or scores.count(value) > 1:
                raise ValueError("unresolved relation")
            r = rels[scores.index(value)]
        domain, rng = (schema.get(r, {}).get(k, []) for k in ("domain", "range"))
        if inverse:
            domain, rng = rng, domain
        if domain and types.get(source) and types[source].isdisjoint(domain):
            raise ValueError("type mismatch")
        target = f"x{idx}"
        types[target] = set(rng)
        steps.append(Step(source, r + ("^-1" if inverse else ""), target, phrase))
    return steps


def describe(edge, meta):
    h, r, t = edge
    names = meta.get("aliases", {})
    return " | ".join(
        (" ".join([h, *names.get(h, [])]), r, " ".join([t, *names.get(t, [])]))
    )


def run(question, graph, meta, lm, ranker, cfg):
    q, topics = question["question"], question["topics"]
    local = graph.local(topics, cfg["radius"])
    original = lm.generate(planning(q), stop=True)
    plan = original
    errors = []
    steps = []
    for attempt in range(2):
        try:
            steps = align(plan, topics, local, meta, ranker, cfg)
            break
        except ValueError as e:
            errors.append(str(e))
            if attempt == 0:
                plan = lm.generate(rewrite(q), stop=True)
    if steps:
        chosen = set()
        for step in steps:
            query = f"{q} | {step.source} | {step.relation} | {step.phrase}"
            chosen.update(
                ranked(query, local.edges, lambda e: describe(e, meta), ranker)[
                    : cfg["top_k"]
                ]
            )
    else:
        incident = {edge for root in topics for _, _, edge in local.adj.get(root, [])}
        chosen = set(
            ranked(q, incident, lambda e: describe(e, meta), ranker)[: cfg["top_k"]]
        )
    kept = Graph(chosen)
    reachable = kept.reach(topics)
    evidence = sorted(edge for edge in chosen if edge[0] in reachable)
    sgs, represented, raw_size = serialize(evidence, topics, lm.count, cfg)
    raw = lm.generate(answering(q, sgs))
    return dict(
        id=question["id"],
        dataset=question["dataset"],
        split=question["split"],
        original_plan=original,
        plan=plan,
        steps=[asdict(s) for s in steps],
        fallback=bool(errors),
        direct_one_hop=not steps,
        alignment_errors=errors,
        retrieved=evidence,
        encoded=sorted(represented),
        sgs=sgs,
        raw_evidence_tokens=raw_size,
        evidence_tokens=lm.count(sgs),
        prediction=answers(raw),
        raw_answer=raw,
    )
