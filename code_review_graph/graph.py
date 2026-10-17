"""Graph algorithms: impact (reverse reachability) and module clustering."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass

from code_review_graph.store import GraphStore, Node

TEST_PATH = re.compile(r"(^|/)(tests?/|test_[^/]*\.py$|[^/]*_test\.py$|conftest\.py$)")


TESTS_CLUSTER = "tests"
CONFIDENCE_RANK = {"exact": 0, "name": 1, "ambiguous": 2}


def is_test(path: str) -> bool:
    return bool(TEST_PATH.search(path))


@dataclass(frozen=True)
class Hit:
    node: Node
    depth: int            # 0 = changed directly, 1 = direct caller, ...
    via: str | None       # the node id this one calls on the path to the change
    confidence: str       # weakest edge confidence along the path


def impact(store: GraphStore, seeds: list[str], max_depth: int = 3) -> list[Hit]:
    """Everything that (transitively) calls the seed nodes, breadth-first.

    A changed method also "changes" its class and module for callers that only
    reference those, so seeds aren't expanded upward here; callers of a class
    are found through constructor edges at resolve time.
    """
    seen: dict[str, Hit] = {}
    queue: deque[str] = deque()
    for s in seeds:
        node = store.node(s)
        if node and s not in seen:
            seen[s] = Hit(node, 0, None, "exact")
            queue.append(s)
    while queue:
        cur = queue.popleft()
        hit = seen[cur]
        if hit.depth >= max_depth:
            continue
        for caller, confidence in store.callers(cur):
            if caller in seen:
                continue
            node = store.node(caller)
            if node is None:
                continue
            weakest = max(confidence, hit.confidence, key=CONFIDENCE_RANK.__getitem__)
            seen[caller] = Hit(node, hit.depth + 1, cur, weakest)
            queue.append(caller)
    return sorted(seen.values(), key=lambda h: (h.depth, h.node.id))


def detect_clusters(store: GraphStore, rounds: int = 30) -> dict[str, str]:
    """Group modules into communities with deterministic label propagation.

    The module graph is undirected: one unit per import plus one per
    cross-module call, divided by sqrt(deg(a) * deg(b)) so hub modules (a
    package's __init__, a utils file everyone imports) don't swallow the whole
    repo into one cluster. Each module repeatedly adopts the label with the
    highest total weight among its neighbours (ties -> smallest label).

    Test modules are kept out of the propagation and grouped as "tests": they
    depend on everything, which would glue unrelated clusters together.
    """
    modules = [r["id"] for r in store.conn.execute("SELECT id FROM nodes WHERE kind = 'module'")]
    source = [m for m in modules if not is_test(m)]
    raw: dict[str, Counter] = defaultdict(Counter)
    rows = store.conn.execute(
        """SELECT s.path AS a, d.path AS b FROM edges e
           JOIN nodes s ON s.id = e.src JOIN nodes d ON d.id = e.dst
           WHERE e.kind IN ('imports', 'calls') AND e.confidence != 'ambiguous'
             AND s.path != d.path"""
    )
    for r in rows:
        if not is_test(r["a"]) and not is_test(r["b"]):
            raw[r["a"]][r["b"]] += 1
            raw[r["b"]][r["a"]] += 1
    degree = {m: sum(raw[m].values()) for m in source}
    weights = {
        a: {b: w / math.sqrt(degree[a] * degree[b]) for b, w in nbrs.items()}
        for a, nbrs in raw.items()
    }

    labels = {m: m for m in source}
    for _ in range(rounds):
        changed = False
        for m in sorted(source):
            if not weights.get(m):
                continue
            score: Counter = Counter()
            for neighbour, w in weights[m].items():
                score[labels[neighbour]] += w
            best = min(score, key=lambda lbl: (-score[lbl], lbl))
            if score[best] > score.get(labels[m], 0) + 1e-9:
                labels[m], changed = best, True
        if not changed:
            break

    # Name each cluster after its members' most common directory.
    members: dict[str, list[str]] = defaultdict(list)
    for m, lbl in labels.items():
        members[lbl].append(m)
    names: dict[str, str] = {}
    used: Counter = Counter()
    for lbl in sorted(members, key=lambda k: (-len(members[k]), k)):
        if len(members[lbl]) == 1:
            stem = members[lbl][0].removesuffix(".py").replace("/", "-")
            names[lbl] = stem
            continue
        dirs = Counter(m.rpartition("/")[0] or "root" for m in members[lbl])
        base = dirs.most_common(1)[0][0].replace("/", "-")
        used[base] += 1
        names[lbl] = base if used[base] == 1 else f"{base}-{used[base]}"
    out = {m: names[lbl] for m, lbl in labels.items()}
    out.update({m: TESTS_CLUSTER for m in modules if is_test(m)})
    return out
