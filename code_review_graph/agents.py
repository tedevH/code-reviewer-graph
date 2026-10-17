"""Multi-agent orchestration.

Review: five specialists read the same diff + graph context in parallel, each
looking for one class of problem. A judge then checks every finding against
the diff, drops what it can't support, merges duplicates, and writes the
final review. Small local models do much better on one narrow question than
on "review this", and the judge pass filters the noise they produce.

Compile: a pool of librarian agents takes modules off a shared queue and
writes wiki pages in parallel.

Agents are threads around one `LLM`. With Ollama, set OLLAMA_NUM_PARALLEL so
the server actually runs requests concurrently; otherwise they queue there,
and the dashboard shows agents waiting for the model.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterator

from code_review_graph.events import EventBus
from code_review_graph.llm import LLM


@dataclass(frozen=True)
class AgentSpec:
    id: str
    name: str
    role: str
    color: str
    focus: str = ""


# Analysis + 3 findings fits comfortably; uncapped, 7B models wrote 2,400 tokens each
# on a one-line ky change, which with 5 agents sharing one model took 5 minutes.
SPECIALIST_MAX_TOKENS = 1000

SPECIALIST_SYSTEM = """You are {name}, one member of a code review team. {focus}

You will be given a pull request diff, the part of the code graph the change can \
affect (callers by hop, tests that reach it), and wiki pages compiled about the \
affected modules, including notes from earlier reviews."""

# Instructions go after the code: small local models follow the end of the prompt
# best, and making them analyse each change before deciding stops them from
# reaching for "no findings" (measured: qwen2.5-coder:7b found 0 of 2 planted bugs
# with findings-only instructions, 2 of 2 with this).
SPECIALIST_TASK = """# Your task

Review the diff above for problems in your area only. Work in two steps.

## Analysis
For each changed hunk in the diff: what did the old code do, what does the new code do, \
and which callers or tests (see the context) rely on the old behaviour? Trace one concrete \
input through old and new code. Be brief: at most 120 words of analysis in total.

## Findings
For each real problem the analysis revealed in your area:

### <high|medium|low>: <file>:<line> <one-line title>
Failure: <concrete input or state -> what goes wrong (exception, wrong value)>
Evidence: <the code or caller that shows it>

At most 3 findings. Report each root cause once, at the changed line that causes it; \
name affected callers in its Evidence instead of repeating the finding for each caller. \
Write NO FINDINGS under this heading only if the analysis shows nothing in your area \
is wrong."""

SPECIALISTS = [
    AgentSpec("correctness", "Correctness", "Logic and edge cases", "#4f7cff",
              "Your area: logic errors, wrong conditions, off-by-one, null/empty handling, "
              "error paths, state mutations and concurrency."),
    AgentSpec("security", "Security", "Vulnerabilities and trust boundaries", "#e5484d",
              "Your area: injection, authn/authz gaps, secrets, unsafe deserialization, "
              "path traversal, SSRF, unvalidated input crossing a trust boundary."),
    AgentSpec("impact", "Blast Radius", "Callers the diff didn't touch", "#f5a524",
              "Your area: the callers listed under 'Callers (blast radius)'. Find callers "
              "that break because of a changed signature, return value, exception, side "
              "effect or invariant. Name the caller and why it breaks."),
    AgentSpec("performance", "Performance", "Cost, complexity, I/O", "#12a594",
              "Your area: algorithmic blowups, repeated work in loops, N+1 queries, "
              "blocking I/O on hot paths, unbounded memory or recursion."),
    AgentSpec("tests", "Tests", "Coverage of changed behaviour", "#8e4ec6",
              "Your area: whether the changed behaviour is tested. Use 'Tests that reach "
              "the change'. Report changed behaviour that no test exercises, and tests the "
              "diff makes wrong. Name the test to add."),
]

JUDGE = AgentSpec("judge", "Judge", "Verifies and merges findings", "#2b2b2b")

JUDGE_SYSTEM = """You are the lead reviewer. Specialists have each reported findings on \
a pull request. Check every finding against the diff and the context. Keep a finding \
only if the diff supports it; drop speculation ("could", "potentially" with no concrete \
input), style nits and anything outside the diff's reach. Merge duplicates. Use the \
agreement table: a location several specialists flagged independently is likely real; \
a finding only one specialist raised needs clear evidence in the diff. For each kept \
finding, trace the code to state the actual failure (which exception, which wrong \
value, for which input). Never invent new findings beyond small corrections.

Write the final review in markdown:

## Summary
<two sentences: what the change does, overall risk: low|medium|high>

## Findings
<kept findings, most severe first, each as:
### <severity>: <file>:<line> <title>
<failure scenario in one or two sentences> (from: <specialist names>)>
or "No issues found." if none survive.

## Learnings
<0-3 durable facts about this codebase worth remembering for future reviews, each as
`- <file path or path::qualname>: <fact>`>"""


class _Streamer:
    """Coalesces token deltas so the dashboard gets ~12 updates/sec per agent, not 100."""

    def __init__(self, bus: EventBus, agent_id: str, interval: float = 0.08):
        self.bus, self.agent_id, self.interval = bus, agent_id, interval
        self.buf: dict[str, list[str]] = {"text": [], "thinking": []}
        self.last = 0.0
        self.tokens = 0
        self.status = "waiting"

    def __call__(self, kind: str, delta: str) -> None:
        self.tokens += 1
        status = "thinking" if kind == "thinking" else "writing"
        if status != self.status:
            self.flush()
            self.status = status
            self.bus.emit("agent", id=self.agent_id, status=status)
        self.buf[kind].append(delta)
        now = time.monotonic()
        if now - self.last >= self.interval:
            self.flush()
            self.last = now

    def flush(self) -> None:
        for kind, parts in self.buf.items():
            if parts:
                self.bus.emit("delta", id=self.agent_id, kind=kind, text="".join(parts))
                parts.clear()


def run_agent(llm: LLM, bus: EventBus, spec: AgentSpec, system: str, prompt: str,
              task: str | None = None, max_tokens: int = 4096) -> str:
    """Run one agent turn, publishing its lifecycle and tokens. Raises on failure."""
    stream = _Streamer(bus, spec.id)
    bus.emit("agent", id=spec.id, name=spec.name, role=spec.role, color=spec.color,
             status="waiting", task=task, reset=True)
    if llm.context_tokens and (len(system) + len(prompt)) / 3.5 > llm.context_tokens * 0.9:
        bus.log(f"{spec.name}: prompt may exceed the model's {llm.context_tokens}-token context; "
                "raise CRG_NUM_CTX or use a smaller --depth", "warn")
    start = time.monotonic()
    try:
        text = llm.complete(system, prompt, max_tokens=max_tokens, on_token=stream)
    except Exception as e:
        stream.flush()
        bus.emit("agent_done", id=spec.id, status="error", tokens=stream.tokens,
                 seconds=round(time.monotonic() - start, 2), task=task, error=str(e)[:300])
        raise
    stream.flush()
    bus.emit("agent_done", id=spec.id, status="done", tokens=stream.tokens,
             seconds=round(time.monotonic() - start, 2), task=task)
    return text


def multi_agent_review(llm: LLM, bus: EventBus, diff_text: str, context_md: str,
                       workers: int = 5, only: list[str] | None = None,
                       judge_llm: LLM | None = None) -> str:
    specialists = [s for s in SPECIALISTS if not only or s.id in only]
    for s in [*specialists, JUDGE]:
        bus.emit("agent", id=s.id, name=s.name, role=s.role, color=s.color, status="queued")
    prompt = f"{context_md}\n\n# Diff\n\n```diff\n{diff_text}\n```"
    task_prompt = f"{prompt}\n\n{SPECIALIST_TASK}"

    findings: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(run_agent, llm, bus, s, SPECIALIST_SYSTEM.format(name=s.name, focus=s.focus),
                        task_prompt, max_tokens=SPECIALIST_MAX_TOKENS): s
            for s in specialists
        }
        for future, spec in futures.items():
            try:
                findings[spec.name] = findings_section(future.result())
            except Exception as e:
                findings[spec.name] = f"(agent failed: {e})"

    reports = "\n\n".join(f"# Findings from {name}\n\n{text}" for name, text in findings.items())
    agreed = consensus(findings)
    for spot in agreed:
        if len(spot["agents"]) > 1:
            bus.log(f"Consensus: {spot['file']}:{spot['line']} flagged by {len(spot['agents'])} agents")
    table = "\n".join(
        f"- {s['file']}:{s['line']} flagged by {len(s['agents'])} of {len(findings)} "
        f"({', '.join(s['agents'])}); severities: {', '.join(s['severities'])}"
        for s in agreed
    ) or "- no findings reported"
    judge_prompt = (f"{prompt}\n\n{reports}\n\n# Agreement between specialists\n\n{table}\n\n"
                    "# Your task\n\nWrite the final review now, in exactly the format from your instructions.")
    return run_agent(judge_llm or llm, bus, JUDGE, JUDGE_SYSTEM, judge_prompt, max_tokens=6000)


_SEVERITY = re.compile(r"\b(high|medium|low)\b", re.IGNORECASE)
_LOCATION = re.compile(r"`?(?P<file>[\w./-]+\.\w+)`?:(?P<line>\d+)")


def findings_section(text: str) -> str:
    """The specialist's Findings section (the judge doesn't need its analysis)."""
    _, found, tail = text.partition("## Findings")
    return (tail if found else text).strip()


def _findings(text: str) -> list[tuple[str, int, str]]:
    """(file, line, severity) per `###` finding header. Models format these loosely:
    `### high: a.py:3 x`, `### <high>: a.py:3`, `### : a.py:3 **High**: x`."""
    out = []
    for line in text.splitlines():
        if not line.lstrip().startswith("###"):
            continue
        loc, sev = _LOCATION.search(line), _SEVERITY.search(line)
        if loc:
            out.append((loc.group("file"), int(loc.group("line")), sev.group(1).lower() if sev else "medium"))
    return out


def consensus(findings: dict[str, str], window: int = 2) -> list[dict]:
    """Group specialists' findings by location (same file, lines within `window`).

    Deterministic, so the judge (often a small local model) gets a reliable signal
    of which problems several agents found independently. Most-agreed first."""
    spots: list[dict] = []
    for agent, text in findings.items():
        for file, line, sev in _findings(text):
            spot = next((s for s in spots if s["file"] == file and abs(s["line"] - line) <= window), None)
            if spot is None:
                spot = {"file": file, "line": line, "agents": [], "severities": []}
                spots.append(spot)
            if agent not in spot["agents"]:
                spot["agents"].append(agent)
                spot["severities"].append(sev)
    return sorted(spots, key=lambda s: (-len(s["agents"]), s["file"], s["line"]))


@dataclass(frozen=True)
class Job:
    key: str
    label: str
    system: str
    prompt: str
    max_tokens: int = 2048


def run_pool(llm: LLM, bus: EventBus, jobs: list[Job], workers: int,
             name: str = "Librarian", role: str = "Writes wiki pages",
             colors: tuple[str, ...] = ("#4f7cff", "#12a594", "#f5a524", "#8e4ec6", "#e5484d", "#0090ff")
             ) -> Iterator[tuple[Job, str | Exception]]:
    """N agents pull jobs from a shared queue. Yields (job, result) on the caller's thread,
    so the caller can write results to SQLite (whose connection is single-threaded)."""
    todo: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        todo.put(job)
    done: queue.Queue[tuple[Job, str | Exception]] = queue.Queue()
    n = max(1, min(workers, len(jobs)))
    specs = [AgentSpec(f"{name.lower()}-{i + 1}", f"{name} {i + 1}", role, colors[i % len(colors)])
             for i in range(n)]
    for spec in specs:
        bus.emit("agent", id=spec.id, name=spec.name, role=spec.role, color=spec.color, status="queued")

    def work(spec: AgentSpec) -> None:
        while True:
            try:
                job = todo.get_nowait()
            except queue.Empty:
                bus.emit("agent", id=spec.id, status="idle", task=None)
                return
            try:
                result: str | Exception = run_agent(llm, bus, spec, job.system, job.prompt,
                                                    task=job.label, max_tokens=job.max_tokens)
            except Exception as e:
                result = e
            done.put((job, result))

    threads = [threading.Thread(target=work, args=(s,), daemon=True) for s in specs]
    for t in threads:
        t.start()
    for _ in jobs:
        yield done.get()
    for t in threads:
        t.join()
