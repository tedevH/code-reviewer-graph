"""Assemble the review context: the subgraph a change touches, plus its wiki pages."""

from __future__ import annotations

import json
from collections import defaultdict

from code_review_graph.graph import Hit, impact, is_test
from code_review_graph.llm import LLM
from code_review_graph.store import GraphStore
from code_review_graph.wiki import page_name

REVIEW_SYSTEM = """You are a senior engineer reviewing a pull request. You are given the \
diff, the part of the code graph the change can affect, and wiki pages the team's tooling \
compiled about the affected modules (including notes from past reviews).

Report only real defects: bugs, broken callers, violated invariants, security problems, \
missing handling that will fail in practice. For each, give file:line, a one-sentence \
statement, and a concrete failure scenario (inputs or state -> wrong result). Use the \
impact graph to check callers the diff didn't touch. Skip style and taste. If nothing \
survives scrutiny, say so in one line. End with a section `## Learnings` holding 0-3 \
durable facts about this codebase worth remembering for future reviews, each as \
`- <file path or path::qualname>: <fact>`."""


def build_context(store: GraphStore, seeds: list[str], max_depth: int = 3) -> dict:
    hits = impact(store, seeds, max_depth=max_depth)
    # Tests are worth reporting at any distance; search deeper for them.
    test_hits = [h for h in impact(store, seeds, max_depth=max_depth + 3) if is_test(h.node.path)]
    near_modules: list[str] = []
    for h in hits:
        if h.depth <= 1 and h.node.path not in near_modules:
            near_modules.append(h.node.path)
    return {"seeds": seeds, "hits": hits, "tests": test_hits, "modules": near_modules}


def render_markdown(store: GraphStore, ctx: dict) -> str:
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
            out += [f"### {depth} hop{'s' if depth > 1 else ''}", ""]
            out += [_line(h) for h in by_depth[depth]] + [""]

    tests = ctx["tests"]
    out += ["## Tests that reach the change", ""]
    out += [_line(h) for h in tests] if tests else ["- none found: this change may be untested"]
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


def review(llm: LLM, diff_text: str, context_md: str) -> str:
    prompt = f"{context_md}\n\n# Diff\n\n```diff\n{diff_text}\n```"
    return llm.complete(REVIEW_SYSTEM, prompt, max_tokens=64000)


def parse_learnings(review_md: str) -> list[tuple[str, str]]:
    """Pull `- target: fact` lines out of the review's Learnings section."""
    _, found, tail = review_md.partition("## Learnings")
    if not found:
        return []
    out = []
    for line in tail.splitlines():
        line = line.strip()
        if line.startswith("## "):
            break
        if line.startswith("- ") and ":" in line:
            target, _, fact = line[2:].partition(":")
            # path::qualname contains "::"; re-split if we cut inside it
            if fact.startswith(":"):
                rest_target, _, fact = fact[1:].partition(":")
                target = f"{target}::{rest_target}"
            target, fact = target.strip().strip("`"), fact.strip()
            if target and fact:
                out.append((target, fact))
    return out


def _line(h: Hit) -> str:
    n = h.node
    via = f" → calls `{h.via.partition('::')[2] or h.via}`" if h.via else ""
    guess = {"name": " _(name-match, verify)_", "ambiguous": " _(ambiguous, verify)_"}.get(h.confidence, "")
    return f"- `{n.path}:{n.start_line}` {n.kind} `{n.name}`{via}{guess}"
