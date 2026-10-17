"""Assemble the review context: the subgraph a change touches, plus its wiki pages."""

from __future__ import annotations

import json
import re
from collections import defaultdict

from code_review_graph.graph import Hit, impact, is_test
from code_review_graph.store import GraphStore
from code_review_graph.wiki import page_name

def build_context(store: GraphStore, seeds: list[str], max_depth: int = 3) -> dict:
    hits = impact(store, seeds, max_depth=max_depth)
    # Tests are worth reporting at any distance; search deeper for them.
    test_hits = [h for h in impact(store, seeds, max_depth=max_depth + 3) if is_test(h.node.path)]
    near_modules: list[str] = []
    for h in hits:
        if h.depth <= 1 and h.node.path not in near_modules:
            near_modules.append(h.node.path)
    return {"seeds": seeds, "hits": hits, "tests": test_hits, "modules": near_modules}


CONFIDENCE_ORDER = {"exact": 0, "name": 1, "ambiguous": 2}


def _top(hits: list[Hit], limit: int | None) -> tuple[list[Hit], int]:
    """Most trustworthy, nearest first; returns (kept, number left out)."""
    if limit is None or len(hits) <= limit:
        return hits, 0
    ranked = sorted(hits, key=lambda h: (CONFIDENCE_ORDER[h.confidence], h.depth, h.node.id))
    return ranked[:limit], len(hits) - limit


def render_markdown(store: GraphStore, ctx: dict, max_items: int | None = None) -> str:
    """Markdown context pack. `max_items` caps each caller hop and the test list:
    agents need the closest, surest callers, not 40 tests that all reach the change
    the same way (which also slows every agent down)."""
    hits: list[Hit] = ctx["hits"]
    out = ["# Review context", ""]
    if not hits:
        return "\n".join(out + ["No indexed code is touched by this change.", ""])

    out += ["## Changed", ""]
    out += [_line(h) for h in hits if h.depth == 0] + [""]

    by_depth: dict[int, list[Hit]] = defaultdict(list)
    for h in hits:
        if h.depth > 0 and not is_test(h.node.path):
            by_depth[h.depth].append(h)
    if by_depth:
        out += ["## Callers (blast radius)", ""]
        for depth in sorted(by_depth):
            kept, more = _top(by_depth[depth], max_items)
            out += [f"### {depth} hop{'s' if depth > 1 else ''}", ""]
            out += [_line(h) for h in kept]
            out += [f"- … and {more} more"] if more else []
            out.append("")

    tests, more = _top(ctx["tests"], max_items)
    out += ["## Tests that reach the change", ""]
    out += [_line(h) for h in tests] if tests else ["- none found: this change may be untested"]
    out += [f"- … and {more} more tests"] if more else []
    out.append("")

    out += ["## Wiki: affected modules", ""]
    for path in ctx["modules"]:
        out += [f"### {page_name(store, path)} (`{path}`)", ""]
        out += [store.summary(path) or "_not compiled_", ""]
        notes = store.notes_for_file(path)
        if notes:
            out += ["Notes from earlier reviews:"]
            out += [f"- {n['text']}" for n in notes] + [""]

    seen_clusters = []
    for path in ctx["modules"]:
        c = store.cluster_of(path)
        info = store.cluster_summary(c) if c else None
        if info and c not in seen_clusters:
            seen_clusters.append(c)
            out += [f"### Cluster: {info[0]}", "", info[1], ""]
    return "\n".join(out)


def render_json(ctx: dict) -> str:
    def hit(h: Hit) -> dict:
        return {"id": h.node.id, "kind": h.node.kind, "path": h.node.path,
                "lines": [h.node.start_line, h.node.end_line], "depth": h.depth,
                "via": h.via, "confidence": h.confidence}

    return json.dumps(
        {"seeds": ctx["seeds"], "impact": [hit(h) for h in ctx["hits"]],
         "tests": [hit(h) for h in ctx["tests"]], "modules": ctx["modules"]},
        indent=2,
    )


LEARNING = re.compile(
    r"^-\s+`?(?P<path>[\w./-]+\.\w+)(?:(?:::|:|#)(?P<qual>[\w.]+))?`?\s*:\s*(?P<fact>.+)$"
)


def parse_learnings(review_md: str) -> list[tuple[str, str]]:
    """Pull `- target: fact` lines out of the review's Learnings section.

    Targets may be `path`, `path::Qual.name`, or the `path:name` form small models
    tend to write; all are normalized to graph ids."""
    _, found, tail = review_md.partition("## Learnings")
    if not found:
        return []
    out = []
    for line in tail.splitlines():
        line = line.strip()
        if line.startswith("## "):
            break
        m = LEARNING.match(line)
        if m:
            target = f"{m.group('path')}::{m.group('qual')}" if m.group("qual") else m.group("path")
            out.append((target, m.group("fact").strip().strip("`").strip()))
    return out


def _line(h: Hit) -> str:
    n = h.node
    via = f" → calls `{h.via.partition('::')[2] or h.via}`" if h.via else ""
    guess = {"name": " _(name-match, verify)_", "ambiguous": " _(ambiguous, verify)_"}.get(h.confidence, "")
    return f"- `{n.path}:{n.start_line}` {n.kind} `{n.name}`{via}{guess}"
