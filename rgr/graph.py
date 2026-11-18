"""Canonical triples, induced neighborhoods and bounded S-expressions."""

import json
from collections import defaultdict, deque


def validated_triple(row):
    if (
        not isinstance(row, (list, tuple))
        or len(row) != 3
        or not all(isinstance(x, str) and x for x in row)
    ):
        raise ValueError(f"Expected three nonempty strings: {row!r}")
    return tuple(row)


def triples(rows):
    return {validated_triple(row) for row in rows}


class Graph:
    def __init__(self, edges):
        self.edges = triples(edges)
        self.adj = defaultdict(list)
        for h, r, t in sorted(self.edges):
            self.adj[h].append((r, t, (h, r, t)))
            self.adj[t].append((r + "^-1", h, (h, r, t)))

    def reach(self, topics, radius=None):
        seen = set(topics)
        queue = deque((e, 0) for e in sorted(seen))
        while queue:
            e, depth = queue.popleft()
            if radius is not None and depth >= radius:
                continue
            for _, n, _ in self.adj.get(e, ()):
                if n not in seen:
                    seen.add(n)
                    queue.append((n, depth + 1))
        return seen

    def local(self, topics, k):
        nodes = self.reach(topics, k)
        return Graph(
            edge for e in nodes for _, n, edge in self.adj.get(e, ()) if n in nodes
        )


def atom(s):
    if any(c in s for c in '()[],:"\\\n') or s.startswith("[REF"):
        return json.dumps(s, ensure_ascii=False)
    return s


def serialize(edges, topics, count, cfg):
    """Produce balanced SGS by omitting complete branches. Count canonical facts once."""
    graph = Graph(edges)
    seen = set()

    def visit(e, depth):
        if e in seen:
            return ("[REF: " + atom(e) + "]", [])
        if depth >= cfg["depth"] or e not in graph.adj:
            return atom(e), []
        seen.add(e)
        groups = defaultdict(list)
        for r, n, edge in graph.adj[e]:
            groups[r].append((n, edge))
        children = []
        for r in sorted(groups):
            for n, edge in sorted(groups[r])[: cfg["branch"]]:
                children.append((r, edge, visit(n, depth + 1)))
        return atom(e), children

    roots = [visit(e, 0) for e in sorted(set(topics))]

    # Keep a preorder prefix of branches. An emitted REF therefore never points
    # to an expanded occurrence removed later in the traversal.
    def render(limit):
        used = 0
        represented = set()

        def emit(node):
            nonlocal used
            name, children = node
            out = defaultdict(list)
            for relation, edge, child in children:
                if used >= limit:
                    break
                used += 1
                represented.add(edge)
                out[relation].append(emit(child))
            if not out:
                return name
            return (
                "("
                + name
                + "".join(
                    " (" + atom(r) + " " + ", ".join(ts) + ")" for r, ts in out.items()
                )
                + ")"
            )

        return "\n".join(emit(root) for root in roots), represented, used

    full, represented, total = render(float("inf"))
    raw_size = count(full)
    if raw_size <= cfg["evidence_tokens"]:
        return full, represented, raw_size
    lo, hi = 0, total
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(render(mid)[0]) <= cfg["evidence_tokens"]:
            lo = mid
        else:
            hi = mid - 1
    context, represented, _ = render(lo)
    if count(context) > cfg["evidence_tokens"]:
        return "", set(), raw_size
    return context, represented, raw_size
