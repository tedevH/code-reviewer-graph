"""`crg` command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from code_review_graph import context as ctx_mod
from code_review_graph.diff import changed_nodes, git_diff
from code_review_graph.indexer import index_repo
from code_review_graph.llm import DEFAULT_MODEL, LLM
from code_review_graph.store import GraphStore
from code_review_graph.wiki import compile_wiki, render_wiki


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crg", description="Code review graph: a compiled, "
                                     "self-updating knowledge graph of your repo for code review.")
    parser.add_argument("--repo", type=Path, default=Path("."), help="repository root (default: .)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("index", help="parse the repo into the exact graph (no LLM)")

    p = sub.add_parser("compile", help="LLM-compile stale modules and clusters into the wiki")
    p.add_argument("--limit", type=int, help="compile at most N modules this run")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("-y", "--yes", action="store_true", help="don't ask before calling the API")

    sub.add_parser("wiki", help="re-render wiki pages from the graph (no LLM)")

    p = sub.add_parser("impact", help="what does a change affect?")
    _add_target_args(p)
    p.add_argument("--format", choices=["markdown", "json"], default="markdown")

    p = sub.add_parser("review", help="review a diff with graph + wiki context")
    _add_target_args(p, symbols=False)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--learn", action="store_true", help="write the review's learnings back into the graph")

    p = sub.add_parser("learn", help="attach a durable note to a module or symbol")
    p.add_argument("target", help="module path, path::qualname, or a symbol name")
    p.add_argument("text")

    sub.add_parser("stats", help="graph size")

    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    store = GraphStore.for_repo(repo)
    try:
        return COMMANDS[args.command](args, repo, store)
    finally:
        store.close()


def _add_target_args(p: argparse.ArgumentParser, symbols: bool = True) -> None:
    if symbols:
        p.add_argument("symbols", nargs="*", help="node ids or symbol names")
    p.add_argument("--diff", metavar="REF", help="use `git diff REF` (e.g. main, HEAD~1)")
    p.add_argument("--diff-file", metavar="PATH", help="read a unified diff from a file, or - for stdin")
    p.add_argument("--depth", type=int, default=3, help="caller hops to follow (default: 3)")


def _wiki_dir(repo: Path) -> Path:
    return repo / ".crg" / "wiki"


def cmd_index(args, repo: Path, store: GraphStore) -> int:
    stats = index_repo(repo, store)
    render_wiki(store, _wiki_dir(repo))
    print(f"Indexed {repo.name}: " + ", ".join(f"{v} {k}" for k, v in stats.items()))
    stale = len(store.stale_summaries())
    if stale:
        print(f"{stale} module(s) need compiling. Run `crg compile` to write their wiki pages.")
    return 0


def cmd_compile(args, repo: Path, store: GraphStore) -> int:
    stale = store.stale_summaries()
    todo = stale[: args.limit] if args.limit else stale
    if not todo and not store.stale_clusters():
        print("Wiki is up to date.")
        return 0
    if todo and not args.yes:
        answer = input(f"Compile {len(todo)} module(s) with {args.model}? This calls the Claude API. [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            return 1
    result = compile_wiki(repo, store, LLM(args.model, args.effort), limit=args.limit)
    pages = render_wiki(store, _wiki_dir(repo))
    print(f"Compiled {result['modules']} module(s), {result['clusters']} cluster(s); "
          f"{result['skipped']} skipped, {result['failed']} failed. Wrote {pages} pages to .crg/wiki/")
    return 0


def cmd_wiki(args, repo: Path, store: GraphStore) -> int:
    print(f"Wrote {render_wiki(store, _wiki_dir(repo))} pages to .crg/wiki/")
    return 0


def _resolve_targets(args, repo: Path, store: GraphStore) -> tuple[list[str], str] | None:
    diff_text = ""
    if args.diff_file:
        diff_text = sys.stdin.read() if args.diff_file == "-" else Path(args.diff_file).read_text()
    elif args.diff:
        diff_text = git_diff(repo, args.diff)
    seeds = changed_nodes(store, diff_text) if diff_text else []
    for sym in getattr(args, "symbols", []) or []:
        matches = store.find_nodes(sym)
        if not matches:
            print(f"No node matches {sym!r}", file=sys.stderr)
            return None
        seeds += [m.id for m in matches if m.id not in seeds]
    if not seeds and not diff_text:
        print("Give symbols, --diff REF, or --diff-file PATH.", file=sys.stderr)
        return None
    return seeds, diff_text


def cmd_impact(args, repo: Path, store: GraphStore) -> int:
    resolved = _resolve_targets(args, repo, store)
    if resolved is None:
        return 2
    ctx = ctx_mod.build_context(store, resolved[0], max_depth=args.depth)
    out = ctx_mod.render_json(ctx) if args.format == "json" else ctx_mod.render_markdown(store, ctx)
    print(out)
    return 0


def cmd_review(args, repo: Path, store: GraphStore) -> int:
    resolved = _resolve_targets(args, repo, store)
    if resolved is None:
        return 2
    seeds, diff_text = resolved
    if not diff_text.strip():
        print("The diff is empty.", file=sys.stderr)
        return 2
    ctx = ctx_mod.build_context(store, seeds, max_depth=args.depth)
    result = ctx_mod.review(LLM(args.model, args.effort), diff_text, ctx_mod.render_markdown(store, ctx))
    print(result)
    if args.learn:
        saved = 0
        for target, fact in ctx_mod.parse_learnings(result):
            if store.node(target):
                store.add_note(target, fact, source="review")
                saved += 1
        render_wiki(store, _wiki_dir(repo))
        print(f"\nSaved {saved} learning(s) to the graph.", file=sys.stderr)
    return 0


def cmd_learn(args, repo: Path, store: GraphStore) -> int:
    matches = [store.node(args.target)] if store.node(args.target) else store.find_nodes(args.target)
    if len(matches) != 1:
        options = ", ".join(m.id for m in matches[:5]) or "nothing"
        print(f"{args.target!r} matched {options}. Use an exact node id.", file=sys.stderr)
        return 2
    store.add_note(matches[0].id, args.text, source="human")
    render_wiki(store, _wiki_dir(repo))
    print(f"Noted on {matches[0].id}")
    return 0


def cmd_stats(args, repo: Path, store: GraphStore) -> int:
    for k, v in store.stats().items():
        print(f"{k:>16}  {v}")
    print(f"{'clusters':>16}  {len(store.clusters())}")
    print(f"{'stale modules':>16}  {len(store.stale_summaries())}")
    return 0


COMMANDS = {
    "index": cmd_index, "compile": cmd_compile, "wiki": cmd_wiki, "impact": cmd_impact,
    "review": cmd_review, "learn": cmd_learn, "stats": cmd_stats,
}

if __name__ == "__main__":
    raise SystemExit(main())
