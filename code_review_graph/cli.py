"""`crg` command line."""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from code_review_graph import context as ctx_mod
from code_review_graph.agents import SPECIALISTS, multi_agent_review
from code_review_graph.diff import changed_nodes, git_diff
from code_review_graph.events import EventBus, TerminalReporter
from code_review_graph.indexer import index_repo
from code_review_graph.llm import DEFAULT_PROVIDER, LLMError, make_llm, recommended_model
from code_review_graph.store import GraphStore
from code_review_graph.wiki import compile_wiki, render_wiki


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crg", description="code-reviewer-graph: a compiled, "
                                     "self-updating knowledge graph of your repo for code review.")
    parser.add_argument("--repo", type=Path, default=Path("."), help="repository root (default: .)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("index", help="parse the repo into the exact graph (no LLM)")

    p = sub.add_parser("compile", help="have agents write wiki pages for stale modules and clusters")
    p.add_argument("--limit", type=int, help="compile at most N modules this run")
    _add_model_args(p, workers=3)

    sub.add_parser("wiki", help="re-render wiki pages from the graph (no LLM)")

    p = sub.add_parser("impact", help="what does a change affect?")
    _add_target_args(p)
    p.add_argument("--format", choices=["markdown", "json"], default="markdown")

    p = sub.add_parser("review", help="multi-agent review of a diff with graph + wiki context")
    _add_target_args(p, symbols=False)
    _add_model_args(p, workers=len(SPECIALISTS))
    p.add_argument("--agents", help="comma-separated specialists to run (default: all): "
                   + ",".join(s.id for s in SPECIALISTS))
    p.add_argument("--judge-model", help="a different (usually larger) model for the judge, same provider")
    p.add_argument("--learn", action="store_true", help="write the review's learnings back into the graph")

    p = sub.add_parser("learn", help="attach a durable note to a module or symbol")
    p.add_argument("target", help="module path, path::qualname, or a symbol name")
    p.add_argument("text")

    sub.add_parser("stats", help="graph size")

    args = parser.parse_args(argv)
    # Exit cleanly on `kill` too, so a model server crg started gets stopped (atexit).
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
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


def _add_model_args(p: argparse.ArgumentParser, workers: int) -> None:
    rec, size = recommended_model()
    p.add_argument("--provider", default=DEFAULT_PROVIDER, choices=["ollama", "openai"],
                   help="ollama (default) or openai (LM Studio, llama.cpp, vLLM, MLX server)")
    p.add_argument("--model", help=f"local model (default for this machine: {rec}, {size}; "
                   "downloaded automatically if missing)")
    p.add_argument("--base-url", help="server URL (default: Ollama on localhost:11434)")
    p.add_argument("--no-pull", action="store_true", help="fail instead of downloading a missing model")
    p.add_argument("--workers", type=int, default=workers, help=f"agents running in parallel (default: {workers})")
    p.add_argument("--ui", action="store_true", help="open the live agent dashboard")
    p.add_argument("--port", type=int, default=8765, help="dashboard port (default: 8765)")
    p.add_argument("--no-open", action="store_true", help="don't open a browser tab for the dashboard")


class _Run:
    """Event bus + terminal reporter + optional dashboard for one agent run."""

    def __init__(self, args, kind: str, title: str, llm, repo: Path, meta: dict):
        self.bus = EventBus()
        self.dashboard = None
        if args.ui:
            from code_review_graph.ui import Dashboard

            self.dashboard = Dashboard(self.bus, port=args.port, open_browser=not args.no_open).start()
            print(f"Dashboard: {self.dashboard.url}", file=sys.stderr)
        self.reporter = TerminalReporter(self.bus).start()
        self.bus.emit("run", title=title, kind=kind, provider=llm.provider, model=llm.model,
                      repo=repo.name, meta=meta)

    def prepare(self, *llms) -> bool:
        """Start the local server and download missing models, showing progress."""
        for llm in llms:
            if llm is None:
                continue
            try:
                llm.prepare(lambda info: self.bus.emit("download", **info))
            except LLMError as e:
                self.bus.log(str(e), "error")
                print(f"error: {e}", file=sys.stderr)
                return False
        return True

    def finish(self) -> None:
        import time

        self.bus.emit("run_done", seconds=round(time.time() - self.bus.started, 1))
        self.reporter.stop()
        if self.dashboard:
            print(f"Done. Dashboard still open at {self.dashboard.url} (Ctrl+C to exit)", file=sys.stderr)
            self.dashboard.hold()


def _llm(args, model: str | None = None):
    try:
        return make_llm(args.provider, model or args.model, args.base_url,
                        parallel=args.workers, auto_pull=not args.no_pull)
    except LLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return None


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
    llm = _llm(args)
    if llm is None:
        return 2
    run = _Run(args, "compile", f"Compiling wiki for {repo.name}", llm, repo,
               {"modules to write": len(todo), "agents": args.workers, "tasks": len(todo)})
    if not run.prepare(llm):
        run.finish()
        return 2
    try:
        result = compile_wiki(repo, store, llm, limit=args.limit, bus=run.bus, workers=args.workers,
                              progress=run.bus.log)
        pages = render_wiki(store, _wiki_dir(repo))
        run.bus.log(f"Compiled {result['modules']} module(s), {result['clusters']} cluster(s); "
                    f"{result['skipped']} skipped, {result['failed']} failed. Wrote {pages} pages.")
    finally:
        run.finish()
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
    only = [a.strip() for a in args.agents.split(",")] if args.agents else None
    unknown = set(only or []) - {s.id for s in SPECIALISTS}
    if unknown:
        print(f"Unknown agents: {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    llm = _llm(args)
    if llm is None:
        return 2
    judge_llm = _llm(args, args.judge_model) if args.judge_model else None
    if args.judge_model and judge_llm is None:
        return 2
    ctx = ctx_mod.build_context(store, seeds, max_depth=args.depth)
    callers = sum(1 for h in ctx["hits"] if h.depth > 0)
    title = f"Reviewing {f'diff vs {args.diff}' if args.diff else 'diff'} in {repo.name}"
    run = _Run(args, "review", title, llm, repo, {
        "changed symbols": len(seeds), "callers in reach": callers, "tests in reach": len(ctx["tests"]),
        "diff lines": diff_text.count("\n"), "tasks": len(only or SPECIALISTS) + 1,
    })
    if not run.prepare(llm, judge_llm):
        run.finish()
        return 2
    try:
        result = multi_agent_review(llm, run.bus, diff_text, ctx_mod.render_markdown(store, ctx),
                                    workers=args.workers, only=only, judge_llm=judge_llm)
        run.bus.emit("result", markdown=result)
        print(result, flush=True)
        if args.learn:
            saved = 0
            for target, fact in ctx_mod.parse_learnings(result):
                if not store.node(target):
                    target = target.partition("::")[0]   # unknown symbol: attach to its module
                if store.node(target):
                    store.add_note(target, fact, source="review")
                    saved += 1
            render_wiki(store, _wiki_dir(repo))
            run.bus.log(f"Saved {saved} learning(s) to the graph.")
    except Exception as e:
        run.bus.log(f"Review failed: {e}", "error")
        print(f"error: {e}", file=sys.stderr)
        run.finish()
        return 1
    run.finish()
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
