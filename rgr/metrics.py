from collections import Counter, defaultdict
import random
from evaluate import eval_hit, eval_f1, normalize


def score(pred, gold, profile="set"):
    if not gold:
        raise ValueError("Cannot score unanswered/unannotated benchmark row")
    a = {normalize(v) or v.casefold().strip() for v in gold if v.strip()}
    p = {normalize(v) or v.casefold().strip() for v in pred if v.strip()}
    # Legacy answer-hit normalizes articles away. Guard empty normalized golds.
    normal = [v for v in gold if normalize(v)]
    hit = eval_hit(" ".join(pred), normal) or any(
        v.casefold().strip() in p for v in gold if not normalize(v)
    )
    if profile == "legacy":
        return int(hit), eval_f1(pred, gold)[0]
    if profile != "set":
        raise ValueError(profile)
    return int(hit), 2 * len(a & p) / (len(a) + len(p)) if a or p else 0.0


def summary(rows):
    grouped = defaultdict(Counter)
    for row in rows:
        totals = grouped[row["dataset"]]
        totals["questions"] += 1
        for key in ("fallback", "direct_one_hop", "hit", "f1"):
            totals[key] += row[key]
    return {
        ds: dict(
            questions=t["questions"],
            fallback_count=t["fallback"],
            fallback_percent=100 * t["fallback"] / t["questions"],
            terminal_count=t["direct_one_hop"],
            hits_percent=100 * t["hit"] / t["questions"],
            f1_percent=100 * t["f1"] / t["questions"],
        )
        for ds, t in grouped.items()
    }


def paired(a, b, n=10000, seed=2026):
    def keyed(rows):
        d = {}
        for row in rows:
            key = (row["dataset"], row["split"], row["id"])
            if key in d:
                raise ValueError("duplicate prediction ID")
            d[key] = row["hit"]
        return d

    a, b = keyed(a), keyed(b)
    if not a or a.keys() != b.keys() or len({k[:2] for k in a}) != 1:
        raise ValueError(
            "paired bootstrap requires identical question IDs from one dataset/split"
        )
    delta = [a[k] - b[k] for k in sorted(a)]
    rng = random.Random(seed)
    vals = sorted(
        100 * sum(rng.choices(delta, k=len(delta))) / len(delta) for _ in range(n)
    )

    def quant(p):
        i = p * (n - 1)
        j = int(i)
        return vals[j] + (i - j) * (vals[min(j + 1, n - 1)] - vals[j])

    return {
        "difference_pp": 100 * sum(delta) / len(delta),
        "ci95_pp": [quant(0.025), quant(0.975)],
        "paired_questions": len(delta),
        "resamples": n,
        "seed": seed,
    }
