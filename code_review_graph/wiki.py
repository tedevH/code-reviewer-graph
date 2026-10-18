"""The meaning layer: compile the graph into an LLM-written, interlinked wiki.

This follows Karpathy's "LLM wiki" idea: instead of re-reading raw source on
every question, an LLM compiles the repo once into linked pages and keeps them
current incrementally. Here the links are not guessed by the LLM: every
[[backlink]] comes from the exact graph (imports, calls, clusters). The LLM
only writes the prose, and only for modules whose source hash changed.

Pages (flat, Obsidian-compatible):
  index.md             clusters with titles and members
  cluster-<name>.md    what a group of tightly coupled modules does together
  <module.name>.md     one module: summary, definitions, deps, dependents, notes
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable

from code_review_graph.graph import TESTS_CLUSTER
from code_review_graph.agents import Job, run_pool
from code_review_graph.events import EventBus
from code_review_graph.llm import LLM
from code_review_graph.store import GraphStore

MAX_SOURCE_BYTES = 200_000

MODULE_SYSTEM = """You maintain a wiki about a codebase. Its readers are code reviewers \
who need to judge whether a change to this code is safe. Write plainly and specifically; \
never pad."""

MODULE_PROMPT = """Write the wiki entry for module `{path}`.

Cover, in 4-8 sentences or tight bullets (no headings):
- what this module is responsible for
- its key functions/classes and how the rest of the codebase uses them
- invariants, assumptions and side effects a caller relies on
- what is risky to change here, and why

Don't list every definition; the page already shows that list.

Imports from this repo: {deps}
Used by: {users}

<source path="{path}">
{source}
</source>"""

CLUSTER_PROMPT = """These modules form one tightly coupled cluster in the codebase \
(they import and call each other more than anything else).

{members}

Reply with exactly this shape:
TITLE: <3-6 word name for what this cluster does>
<one paragraph: what the cluster does as a whole, how data flows through it, \
and the cross-module invariants a reviewer should protect>"""


def page_name(store: GraphStore, path: str) -> str:
    """Python modules use their dotted name; other files their path, flattened."""
    node = store.node(path)
    return (node.name if node and node.name else path).replace("/", ".")


def compile_wiki(
    repo: Path,
    store: GraphStore,
    llm: LLM,
    limit: int | None = None,
    progress: Callable[[str], None] = print,
    bus: EventBus | None = None,
    workers: int = 1,
) -> dict[str, int]:
    """Summarize stale modules, then stale clusters, with `workers` agents in parallel."""
    bus = bus or EventBus()
    hashes = store.file_hashes()
    done = skipped = failed = 0
    jobs: list[Job] = []
    for path in store.stale_summaries()[: limit if limit else None]:
        source = (repo / path).read_bytes()
        if not source.strip():
            store.put_summary(path, hashes[path], "Empty module.", "none")
            continue
        if len(source) > MAX_SOURCE_BYTES:
            progress(f"  skip {path}: {len(source):,} bytes is over the {MAX_SOURCE_BYTES:,} limit")
            skipped += 1
            continue
        prompt = MODULE_PROMPT.format(
            path=path,
            deps=", ".join(store.imports_of(path)) or "none",
            users=", ".join(store.imported_by(path)) or "none",
            source=source.decode("utf-8", errors="replace"),
        )
        jobs.append(Job(path, path, MODULE_SYSTEM, prompt))

    for job, result in run_pool(llm, bus, jobs, workers, name="Librarian", role="Writes module pages"):
        if isinstance(result, Exception) or not result.strip():
            progress(f"  fail {job.key}: {result or 'empty response'}")
            failed += 1
            continue
        store.put_summary(job.key, hashes[job.key], result.strip(), llm.model)
        done += 1

    clusters = store.clusters()
    cluster_jobs: list[Job] = []
    for cluster in store.stale_clusters():
        members = clusters[cluster]
        summaries = [(p, store.summary(p)) for p in members]
        if cluster == TESTS_CLUSTER or len(members) < 2 or any(s is None for _, s in summaries):
            continue  # singletons need no cluster page; wait until all members are compiled
        listing = "\n\n".join(f"## {p}\n{s}" for p, s in summaries)
        cluster_jobs.append(Job(cluster, f"cluster {cluster}", MODULE_SYSTEM,
                                CLUSTER_PROMPT.format(members=listing), max_tokens=1024))

    cluster_done = 0
    for job, result in run_pool(llm, bus, cluster_jobs, workers, name="Cartographer",
                                role="Writes cluster pages"):
        if isinstance(result, Exception) or not result.strip():
            progress(f"  fail cluster {job.key}: {result or 'empty response'}")
            failed += 1
            continue
        title, _, body = result.strip().partition("\n")
        title = title.strip().strip("#* ").removeprefix("TITLE:").strip()
        store.put_cluster_summary(job.key, title or job.key, body.strip(), llm.model)
        cluster_done += 1
    return {"modules": done, "clusters": cluster_done, "skipped": skipped, "failed": failed}


def render_wiki(store: GraphStore, out_dir: Path) -> int:
    """Write markdown pages from the store. Pure function of the graph; no LLM calls."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.md"):
        old.unlink()
    clusters = store.clusters()
    written = 0

    for path, cluster in ((p, c) for c, ms in clusters.items() for p in ms):
        name = page_name(store, path)
        lines = [
            "---", f"path: {path}", f"cluster: {cluster}", "---", "",
            f"# {name}", "",
            f"`{path}` · cluster [[cluster-{cluster}]]" if len(clusters[cluster]) > 1 else f"`{path}`",
            "",
            store.summary(path) or "_Not compiled yet. Run `crg compile`._",
            "",
        ]
        defs = [n for n in store.nodes_in_file(path) if n.kind != "module"]
        if defs:
            lines += ["## Defines", ""]
            for n in sorted(defs, key=lambda n: n.start_line):
                callers = _caller_modules(store, n.id, path)
                used = f" · called from {', '.join(callers)}" if callers else ""
                lines.append(f"- `{n.name}` ({n.kind}, L{n.start_line}-{n.end_line}){used}")
            lines.append("")
        for title, paths in (("Depends on", store.imports_of(path)), ("Used by", store.imported_by(path))):
            if paths:
                lines += [f"## {title}", ""]
                lines += [f"- [[{page_name(store, p)}]]" for p in sorted(paths)] + [""]
        notes = store.notes_for_file(path, statuses=("verified", "pending"))
        if notes:
            lines += ["## Notes", ""]
            for n in notes:
                where = n["target"].partition("::")[2]
                where = f" `{where}`" if where else ""
                unverified = " _(unverified)_" if n["status"] == "pending" else ""
                lines.append(f"- {n['created_at'][:10]} ({n['source']}){where}: {n['text']}{unverified}")
            lines.append("")
        (out_dir / f"{name}.md").write_text("\n".join(lines))
        written += 1

    index = ["# Code review graph", "", "Generated by `crg`. Edit nothing here by hand; "
             "add knowledge with `crg learn`.", ""]
    for cluster, members in sorted(clusters.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(members) > 1:
            info = store.cluster_summary(cluster)
            title = info[0] if info else cluster
            page = [f"# {title}", "", f"Cluster `{cluster}`, {len(members)} modules.", "",
                    info[1] if info else "_Not compiled yet. Run `crg compile`._", "", "## Modules", ""]
            page += [f"- [[{page_name(store, m)}]]" for m in members]
            (out_dir / f"cluster-{cluster}.md").write_text("\n".join(page) + "\n")
            written += 1
            index.append(f"- [[cluster-{cluster}]]: {title} ({len(members)} modules)")
        else:
            index.append(f"- [[{page_name(store, members[0])}]]")
    (out_dir / "index.md").write_text("\n".join(index) + "\n")
    return written + 1


def _caller_modules(store: GraphStore, node_id: str, own_path: str) -> list[str]:
    by_module: dict[str, int] = defaultdict(int)
    for caller, _ in store.callers(node_id):
        caller_path = caller.partition("::")[0]
        if caller_path != own_path:
            by_module[caller_path] += 1
    return [f"[[{page_name(store, p)}]]" for p in sorted(by_module)]
