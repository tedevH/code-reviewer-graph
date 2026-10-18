"""`crg` command line."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from code_review_graph import context as ctx_mod
from code_review_graph import evaluate, memory
from code_review_graph.agents import SPECIALISTS, multi_agent_review
from code_review_graph.consolidate import consolidate
from code_review_graph.diff import changed_nodes, git_diff
from code_review_graph.events import EventBus, TerminalReporter
from code_review_graph.indexer import index_repo
from code_review_graph.llm import DEFAULT_PROVIDER, LLMError, make_llm, recommended_model
from code_review_graph.store import GraphStore
from code_review_graph.wiki import compile_wiki, render_wiki


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crg", description="code-reviewer-graph: a compiled, "
                                     "self-improving knowledge graph of your repo for code review.")
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
    _add_agent_args(p)
    p.add_argument("--learn", action="store_true", help="save the review's learnings (used once confirmed)")

    p = sub.add_parser("feedback", help="list open findings, or mark one good/bad")
    p.add_argument("finding", nargs="?", help="finding id, e.g. f12")
    p.add_argument("verdict", nargs="?", choices=["good", "bad"])
    p.add_argument("--note", help="why, shown to future judges as team history")

    p = sub.add_parser("eval", help="replay this repo's bug fixes to measure and tune reviews")
    p.add_argument("--cases", type=int, default=5, help="bug-fix commits to replay (default: 5)")
    p.add_argument("--max-lines", type=int, default=40, help="skip fixes larger than this many lines")
    p.add_argument("--models", help="comma-separated models to compare (default: the current model)")
    p.add_argument("--save", action="store_true", help="make the best model and agent mix the default")
    p.add_argument("--grader-model", help="model that judges whether findings name the real bug "
                   "(default: the judge or review model; a larger one grades more reliably)")
    _add_model_args(p, workers=len(SPECIALISTS))
    _add_agent_args(p)

    p = sub.add_parser("learn", help="attach a durable note to a module or symbol")
    p.add_argument("target", help="module path, path::qualname, or a symbol name")
    p.add_argument("text")

    sub.add_parser("stats", help="graph size, agent track records, learnings")

    args = parser.parse_args(argv)
    # Exit cleanly on `kill` too, so a model server crg started gets stopped (atexit).
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    repo = args.repo.resolve()
    store = GraphStore.for_repo(repo)
    _apply_config(args, repo)
    try:
        return COMMANDS[args.command](args, repo, store)
    finally:
        store.close()


# --- argument helpers ----------------------------------------------------------

def _add_target_args(p: argparse.ArgumentParser, symbols: bool = True) -> None:
    if symbols:
        p.add_argument("symbols", nargs="*", help="node ids or symbol names")
    p.add_argument("--diff", metavar="REF", help="use `git diff REF` (e.g. main, HEAD~1)")
    p.add_argument("--diff-file", metavar="PATH", help="read a unified diff from a file, or - for stdin")
    p.add_argument("--depth", type=int, default=3, help="caller hops to follow (default: 3)")
    p.add_argument("--no-index", action="store_true", help="skip re-indexing before running")


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


def _add_agent_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--agents", help="comma-separated specialists to run (default: all with a decent "
                   "track record): " + ",".join(s.id for s in SPECIALISTS))
    p.add_argument("--judge-model", help="a different (usually larger) model for the judge, same provider")


def _config_path(repo: Path) -> Path:
    return repo / ".crg" / "config.json"


def _apply_config(args, repo: Path) -> None:
    """Defaults saved by `crg eval --save`; explicit flags always win."""
    try:
        config = json.loads(_config_path(repo).read_text())
    except (OSError, ValueError):
        return
    if getattr(args, "model", "absent") is None and config.get("model"):
        args.model = config["model"]
    if getattr(args, "judge_model", "absent") is None and config.get("judge_model"):
        args.judge_model = config["judge_model"]
    if getattr(args, "agents", "absent") is None and config.get("agents"):
        args.agents = ",".join(config["agents"])
        args.agents_from_config = True


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


def _reindex(args, repo: Path, store: GraphStore) -> None:
    """The graph must match the code being reviewed; indexing takes about a second."""
    if not getattr(args, "no_index", False):
        index_repo(repo, store, warn=lambda m: print(m, file=sys.stderr))


# --- commands ----------------------------------------------------------------------

def cmd_index(args, repo: Path, store: GraphStore) -> int:
    stats = index_repo(repo, store)
    render_wiki(store, _wiki_dir(repo))
    print(f"Indexed {repo.name}: " + ", ".join(f"{v} {k}" for k, v in stats.items()))
    stale = len(store.stale_summaries())
    if stale:
        print(f"{stale} module(s) need compiling. Run `crg compile` to write their wiki pages.")
    return 0


def cmd_compile(args, repo: Path, store: GraphStore) -> int:
    _reindex(args, repo, store)
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
    if args.diff or args.diff_file:
        _reindex(args, repo, store)
    resolved = _resolve_targets(args, repo, store)
    if resolved is None:
        return 2
    ctx = ctx_mod.build_context(store, resolved[0], max_depth=args.depth)
    out = ctx_mod.render_json(ctx) if args.format == "json" else ctx_mod.render_markdown(store, ctx)
    print(out)
    return 0


def _choose_agents(args, store: GraphStore, log) -> list[str] | None:
    """Explicit --agents wins. Otherwise skip specialists whose findings this team keeps ignoring."""
    if args.agents:
        only = [a.strip() for a in args.agents.split(",") if a.strip()]
        unknown = set(only) - {s.id for s in SPECIALISTS}
        if unknown:
            raise ValueError(f"Unknown agents: {', '.join(sorted(unknown))}")
        if not getattr(args, "agents_from_config", False):
            return only
    else:
        only = [s.id for s in SPECIALISTS]
    records = memory.agent_records(store)
    keep = []
    for s in SPECIALISTS:
        if s.id not in only:
            continue
        rec = records.get(s.name)
        if rec is not None and rec.weak:
            log(f"Skipping {s.name}: {rec.describe()}. Run it anyway with --agents {s.id}.")
        else:
            keep.append(s.id)
    return keep or only


def cmd_review(args, repo: Path, store: GraphStore) -> int:
    _reindex(args, repo, store)
    resolved = _resolve_targets(args, repo, store)
    if resolved is None:
        return 2
    seeds, diff_text = resolved
    if not diff_text.strip():
        print("The diff is empty.", file=sys.stderr)
        return 2
    llm = _llm(args)
    judge_llm = _llm(args, args.judge_model) if args.judge_model else None
    if llm is None or (args.judge_model and judge_llm is None):
        return 2

    learned = memory.refresh(store, repo)       # decide earlier findings before using the records
    ctx = ctx_mod.build_context(store, seeds, max_depth=args.depth)
    callers = sum(1 for h in ctx["hits"] if h.depth > 0)
    title = f"Reviewing {f'diff vs {args.diff}' if args.diff else 'diff'} in {repo.name}"
    run = _Run(args, "review", title, llm, repo, {
        "changed symbols": len(seeds), "callers in reach": callers, "tests in reach": len(ctx["tests"]),
        "diff lines": diff_text.count("\n"), "tasks": len(SPECIALISTS) + 1,
    })
    if any(learned.values()):
        run.bus.log("Learned from earlier reviews: " + ", ".join(f"{v} {k}" for k, v in learned.items() if v))
    try:
        only = _choose_agents(args, store, run.bus.log)
    except ValueError as e:
        print(e, file=sys.stderr)
        run.finish()
        return 2
    if not run.prepare(llm, judge_llm):
        run.finish()
        return 2
    try:
        review = multi_agent_review(
            llm, run.bus, diff_text, ctx_mod.render_markdown(store, ctx, max_items=8),
            workers=args.workers, only=only, judge_llm=judge_llm,
            records=memory.agent_records(store), history=memory.history_examples(store),
        )
        result, cleaned = consolidate(review.markdown, diff_text, ctx["hits"] + ctx["tests"], repo)
        if any(cleaned.values()):
            run.bus.log("Cleaned review: " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in cleaned.items() if v))
        review_id, result = memory.record_review(store, repo, result, review.specialist_findings, args.diff,
                                                 llm.model, (judge_llm or llm).model)
        outdated = memory.expire_outdated(store, result, review_id)
        if outdated:
            run.bus.log(f"Expired {len(outdated)} outdated note(s): " + ", ".join(f"n{i}" for i in outdated))
        if args.learn:
            saved = memory.save_learnings(store, repo, review_id, ctx_mod.parse_learnings(result))
            render_wiki(store, _wiki_dir(repo))
            run.bus.log(f"Saved {saved} learning(s); they're used once a finding from this review is acted on.")
        run.bus.emit("result", markdown=result)
        print(result, flush=True)
        if "[f" in result:
            print("\nMark findings with `crg feedback <id> good|bad`; fixed ones are detected automatically.",
                  file=sys.stderr)
    except Exception as e:
        run.bus.log(f"Review failed: {e}", "error")
        print(f"error: {e}", file=sys.stderr)
        run.finish()
        return 1
    run.finish()
    return 0


def cmd_feedback(args, repo: Path, store: GraphStore) -> int:
    memory.refresh(store, repo)
    if not args.finding:
        rows = store.conn.execute(
            "SELECT id, title, status FROM findings ORDER BY id DESC LIMIT 30").fetchall()
        if not rows:
            print("No findings yet. Run `crg review` first.")
        for r in rows:
            print(f"f{r['id']:<5} {r['status']:<8} {r['title']}")
        return 0
    if not args.verdict:
        print("Give a verdict: crg feedback f12 good|bad", file=sys.stderr)
        return 2
    finding_id = int(args.finding.lstrip("fF#"))
    if not memory.feedback(store, finding_id, args.verdict, args.note):
        print(f"No finding f{finding_id}.", file=sys.stderr)
        return 2
    learned = memory.update_learnings(store, repo)
    render_wiki(store, _wiki_dir(repo))
    print(f"Marked f{finding_id} {args.verdict}."
          + (f" Learnings: {learned['verified']} verified, {learned['expired']} expired." if any(learned.values()) else ""))
    return 0


def cmd_eval(args, repo: Path, store: GraphStore) -> int:
    cases = evaluate.mine_cases(repo, limit=args.cases, max_lines=args.max_lines)
    if not cases:
        print("No small bug-fix commits found in this repo's history (looked for fix/bug/regression "
              "in commit messages). A deeper clone may help: git fetch --unshallow", file=sys.stderr)
        return 2
    models = [m.strip() for m in args.models.split(",")] if args.models else [args.model]
    first = _llm(args, models[0])
    if first is None:
        return 2
    run = _Run(args, "eval", f"Evaluating on {len(cases)} bug fixes from {repo.name}", first, repo,
               {"cases": len(cases), "models": len(models), "tasks": len(cases) * len(models)})
    only = [a.strip() for a in args.agents.split(",")] if args.agents else None
    judge_llm = _llm(args, args.judge_model) if args.judge_model else None
    grader = _llm(args, args.grader_model) if args.grader_model else None
    if grader is not None and hasattr(grader, "context_tokens"):
        grader.context_tokens = 8192    # one fix diff + a few findings; keeps a big grader in RAM
    if grader is not None and not run.prepare(grader):
        return 2
    all_results, scores = [], []
    try:
        for name in models:
            llm = first if name == models[0] else _llm(args, name)
            if llm is None or not run.prepare(llm, judge_llm):
                return 2
            reviewed = []
            for i, case in enumerate(cases, 1):
                run.bus.log(f"[{llm.model}] reviewing case {i}/{len(cases)}: {case.commit[:8]} {case.subject[:60]}")
                reviewed.append(evaluate.run_case(case, repo, llm, run.bus, args.workers, judge_llm, only))
            grader_llm = grader or judge_llm or llm
            if grader_llm is not llm and hasattr(llm, "unload"):
                llm.unload()            # two models at once can push a laptop into swap
                if judge_llm is not None and judge_llm is not grader_llm:
                    judge_llm.unload()
            run.bus.log(f"Grading {len(cases)} reviews against the real fixes with {grader_llm.model}")
            results = []
            for r, rev in reviewed:
                r = evaluate.grade_case(r, rev, grader or judge_llm or llm)
                verdict = "error: " + r.error if r.error else ("caught" if r.caught else "missed")
                run.bus.log(f"[{llm.model}] {r.commit[:8]}: {verdict} ({r.on_target}/{r.findings} named the bug, "
                            f"{r.related} vaguely, {r.located} on its lines, {r.seconds}s)")
                results.append(r)
            all_results += results
            scores.append(evaluate.score(results))
    finally:
        run.reporter.stop()

    print("\ncaught = a finding identified the real bug (graded against the actual fix);"
          " precision = share of findings that did")
    print(f"\n{'model':<24} {'caught':>8} {'precision':>10} {'avg time':>9}")
    for s in scores:
        print(f"{s.model:<24} {s.caught:>3}/{s.cases:<4} {s.precision:>10.0%} {s.seconds:>8.0f}s")
    top = evaluate.best(scores)
    print(f"\nPer specialist ({top.model}): findings that identified a real bug / findings reported")
    for agent, (hit, total) in sorted(top.agents.items()):
        print(f"  {agent:<14} {hit}/{total}")
    path = evaluate.save(repo, all_results, scores)
    print(f"\nDetails: {path.relative_to(repo)}")
    if args.save:
        drop = evaluate.useless_agents(top)
        agents = [s.id for s in SPECIALISTS if s.name not in drop]
        config = {"model": top.model, "agents": agents}
        if args.judge_model:
            config["judge_model"] = args.judge_model
        _config_path(repo).parent.mkdir(parents=True, exist_ok=True)
        _config_path(repo).write_text(json.dumps(config, indent=2) + "\n")
        print(f"Saved defaults to .crg/config.json: model {top.model}"
              + (f", without {', '.join(drop)}" if drop else ", all specialists"))
    run.bus.emit("run_done", seconds=round(time.time() - run.bus.started, 1))
    if run.dashboard:
        run.dashboard.hold()
    return 0


def cmd_learn(args, repo: Path, store: GraphStore) -> int:
    matches = [store.node(args.target)] if store.node(args.target) else store.find_nodes(args.target)
    if len(matches) != 1:
        options = ", ".join(m.id for m in matches[:5]) or "nothing"
        print(f"{args.target!r} matched {options}. Use an exact node id.", file=sys.stderr)
        return 2
    store.add_note(matches[0].id, args.text, source="human",
                   snapshot=store.node_source(repo, matches[0].id))
    render_wiki(store, _wiki_dir(repo))
    print(f"Noted on {matches[0].id}")
    return 0


def cmd_stats(args, repo: Path, store: GraphStore) -> int:
    memory.refresh(store, repo)
    for k, v in store.stats().items():
        print(f"{k:>16}  {v}")
    print(f"{'clusters':>16}  {len(store.clusters())}")
    print(f"{'stale modules':>16}  {len(store.stale_summaries())}")

    outcomes = dict(store.conn.execute("SELECT status, COUNT(*) FROM findings GROUP BY status").fetchall())
    if outcomes:
        print("\nFindings: " + ", ".join(f"{v} {k}" for k, v in sorted(outcomes.items())))
        print("Specialist track records:")
        records = memory.agent_records(store)
        for s in SPECIALISTS:
            rec = records.get(s.name)
            flag = "  (skipped in reviews)" if rec and rec.weak else ""
            print(f"  {s.name:<14} {rec.describe() if rec else 'no track record yet'}{flag}")
    notes = dict(store.conn.execute("SELECT status, COUNT(*) FROM notes GROUP BY status").fetchall())
    if notes:
        print("Learnings: " + ", ".join(f"{v} {k}" for k, v in sorted(notes.items())))
    return 0


COMMANDS = {
    "index": cmd_index, "compile": cmd_compile, "wiki": cmd_wiki, "impact": cmd_impact,
    "review": cmd_review, "feedback": cmd_feedback, "eval": cmd_eval, "learn": cmd_learn,
    "stats": cmd_stats,
}

if __name__ == "__main__":
    raise SystemExit(main())
