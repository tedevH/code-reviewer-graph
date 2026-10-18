"""`crg eval`: measure review quality on this repository's own bug fixes.

For each recent bug-fix commit F (a non-merge commit whose message says fix,
bug, regression, ...) with parent P:

  * check out P (the buggy code) in a temporary git worktree and index it
  * review `git diff F P`: the change that would *introduce* the bug, as if
    someone had just written it
  * a case is caught when a final finding lands on a line the real fix
    touched (within a few lines)

Recall says how many real bugs crg catches; precision says how much of what it
reports is on target. Comparing models on the same cases picks the best local
model for this codebase, and per-specialist hit rates show which agents earn
their time here. Test files in the fix are left out of the diff so the answer
isn't handed to the reviewers.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from code_review_graph.agents import multi_agent_review, parse_specialist_findings
from code_review_graph.consolidate import consolidate, parse_findings
from code_review_graph.context import build_context, render_markdown
from code_review_graph.diff import changed_lines, changed_nodes
from code_review_graph.events import EventBus
from code_review_graph.extract import extractor_for
from code_review_graph.graph import is_test
from code_review_graph.indexer import index_repo
from code_review_graph.store import GraphStore

FIX_SUBJECT = re.compile(r"\b(fix(e[sd])?|bug|regression|crash(e[sd])?|broken|incorrect|wrong)\b", re.I)
NOT_A_CODE_BUG = re.compile(r"\b(ci|lint(ing)?|typos?|docs?|readme|changelog|build|flaky|tests?|deps?|"
                            r"dependencies|types?|typings?|formatting)\b", re.I)
HIT_WINDOW = 3


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


@dataclass
class Case:
    commit: str
    parent: str
    subject: str
    files: list[str]
    diff: str        # fixed -> buggy (what the reviewers see)
    fix: str = ""    # buggy -> fixed (the answer; only the grader sees it)


GRADER_SYSTEM = """You grade a code reviewer. You know the real bug: the commit that \
fixed it and its message. The reviewer only saw the change that introduced the bug, \
so it may describe the bug in its own words or by its symptom. Judge each finding by \
the behaviour it describes, not by its wording."""

GRADER_PROMPT = """# The real bug
Fix commit message: {subject}

```diff
{fix}
```

# The reviewer's findings
{findings}

# Your answer
First say in one sentence what wrong behaviour the fix corrects. Then give one line per \
finding:

<number>: SAME | RELATED | DIFFERENT - <reason>

SAME: describes that wrong behaviour or its direct symptom (for example, the value that \
is missing or wrong, or the error it causes).
RELATED: points at the right code and roughly the right problem, but too vaguely to act on.
DIFFERENT: a different problem, or a wrong explanation."""

@dataclass
class CaseResult:
    commit: str
    subject: str
    model: str
    seconds: float
    caught: bool          # a finding identified the bug (graded SAME)
    findings: int
    on_target: int        # findings graded SAME
    located: int = 0      # findings that merely landed on the fixed lines
    related: int = 0      # findings graded RELATED: right area, too vague
    agents: dict[str, list[int]] = field(default_factory=dict)   # name -> [identified, total]
    texts: list[str] = field(default_factory=list)
    grade: str = ""
    error: str | None = None


def mine_cases(repo: Path, limit: int = 5, max_lines: int = 40, scan: int = 500) -> list[Case]:
    """Recent small bug fixes in indexable source files."""
    cases: list[Case] = []
    log = _git(repo, "log", "--no-merges", f"-n{scan}", "--format=%H%x09%P%x09%s")
    for row in log.splitlines():
        sha, parents, subject = (row.split("\t", 2) + ["", ""])[:3]
        if len(parents.split()) != 1 or not FIX_SUBJECT.search(subject) or NOT_A_CODE_BUG.search(subject):
            continue
        source, changed = [], 0
        for stat in _git(repo, "show", "--numstat", "--format=", sha).splitlines():
            added, deleted, path = (stat.split("\t", 2) + ["", ""])[:3]
            if added == "-" or not path or is_test(path) or extractor_for(path) is None:
                continue
            source.append(path)
            changed += int(added) + int(deleted)
        if not source or changed == 0 or changed > max_lines:
            continue
        diff = _git(repo, "diff", "-U3", sha, parents.strip(), "--", *source)
        if diff.strip():
            fix = _git(repo, "diff", "-U3", parents.strip(), sha, "--", *source)
            cases.append(Case(sha, parents.strip(), subject, source, diff, fix))
        if len(cases) >= limit:
            break
    return cases


def _on_target(truth: dict[str, set[int]], file: str | None, line: int | None) -> bool:
    if not file or line is None:
        return False
    lines = truth.get(file.strip("`").removeprefix("./"), set())
    return any(abs(line - n) <= HIT_WINDOW for n in lines)


def grade(case: Case, findings: list, grader) -> dict[int, str]:
    """Verdict per finding index: "same", "related" or "different"."""
    if not findings:
        return {}
    listing = "\n\n".join(f"{i}. {f.header.lstrip('# ')}\n" + "\n".join(f.body).strip()
                           for i, f in enumerate(findings, 1))
    reply = grader.complete(GRADER_SYSTEM, GRADER_PROMPT.format(subject=case.subject, fix=case.fix,
                                                                findings=listing), max_tokens=600)
    verdicts = {}
    for m in re.finditer(r"^\W*(\d+)\W*[:.)-]\s*\**(SAME|RELATED|DIFFERENT)", reply, re.I | re.M):
        i = int(m.group(1)) - 1
        if 0 <= i < len(findings):
            verdicts[i] = m.group(2).lower()
    return verdicts


@dataclass
class Reviewed:
    """A finished review of one case, waiting to be graded."""
    case: Case
    findings: list
    located: list[int]
    specialist_findings: dict[str, str]


def run_case(case: Case, repo: Path, llm, bus: EventBus, workers: int, judge_llm=None,
             only: list[str] | None = None) -> tuple[CaseResult, Reviewed | None]:
    """Review one case. Grading happens separately (see `grade_case`), so a larger
    grader model is loaded once per eval instead of swapping models every case."""
    tmp = Path(tempfile.mkdtemp(prefix="crg-eval-"))
    tree = tmp / "tree"
    result = CaseResult(case.commit[:10], case.subject, llm.model, 0.0, False, 0, 0)
    reviewed = None
    store = None
    try:
        _git(repo, "worktree", "add", "--detach", "--quiet", str(tree), case.parent)
        store = GraphStore(tmp / "graph.db")
        index_repo(tree, store, warn=lambda _: None)
        ctx = build_context(store, changed_nodes(store, case.diff))
        start = time.monotonic()
        review = multi_agent_review(llm, bus, case.diff, render_markdown(store, ctx, max_items=8),
                                    workers=workers, only=only, judge_llm=judge_llm)
        md, _ = consolidate(review.markdown, case.diff, ctx["hits"] + ctx["tests"], tree)
        result.seconds = round(time.monotonic() - start, 1)

        truth = changed_lines(case.diff)
        _, findings = parse_findings(md.partition("## Findings")[2].split("\n## ", 1)[0])
        result.findings = len(findings)
        result.texts = [f.render() for f in findings]
        located = [i for i, f in enumerate(findings) if _on_target(truth, f.file, f.line)]
        result.located = len(located)
        reviewed = Reviewed(case, findings, located, review.specialist_findings)
    except Exception as e:   # one broken case shouldn't sink the whole eval
        result.error = str(e)[:300]
    finally:
        if store is not None:
            store.close()
        subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(tree)],
                       capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)
    return result, reviewed


def grade_case(result: CaseResult, reviewed: Reviewed | None, grader) -> CaseResult:
    """Landing on the right lines is easy when the whole diff is the bug, so a grader
    that knows the real fix decides which findings name the actual bug."""
    if reviewed is None or not reviewed.located:
        return result
    try:
        verdicts = grade(reviewed.case, reviewed.findings, grader)
    except Exception as e:
        result.error = f"grading failed: {str(e)[:250]}"
        return result
    result.grade = ", ".join(f"{i + 1}:{v}" for i, v in sorted(verdicts.items()))
    identified = {i for i, v in verdicts.items() if v == "same"} & set(reviewed.located)
    result.related = sum(1 for i, v in verdicts.items() if v == "related" and i in reviewed.located)
    result.on_target = len(identified)
    result.caught = bool(identified)
    hits = [reviewed.findings[i] for i in identified]
    for agent, text in reviewed.specialist_findings.items():
        raw = parse_specialist_findings(text)
        credited = sum(any(h.file and h.file.strip("`") == f and h.line and abs(h.line - ln) <= HIT_WINDOW
                           for h in hits) for f, ln, _ in raw)
        result.agents[agent] = [credited, len(raw)]
    return result


@dataclass
class ModelScore:
    model: str
    cases: int
    caught: int
    findings: int
    on_target: int
    seconds: float
    agents: dict[str, list[int]]

    @property
    def recall(self) -> float:
        return self.caught / self.cases if self.cases else 0.0

    @property
    def precision(self) -> float:
        return self.on_target / self.findings if self.findings else 0.0


def score(results: list[CaseResult]) -> ModelScore:
    ok = [r for r in results if r.error is None]
    agents: dict[str, list[int]] = {}
    for r in ok:
        for a, (hit, total) in r.agents.items():
            agents.setdefault(a, [0, 0])
            agents[a][0] += hit
            agents[a][1] += total
    return ModelScore(results[0].model if results else "", len(ok), sum(r.caught for r in ok),
                      sum(r.findings for r in ok), sum(r.on_target for r in ok),
                      round(sum(r.seconds for r in ok) / len(ok), 1) if ok else 0.0, agents)


def best(scores: list[ModelScore]) -> ModelScore:
    return max(scores, key=lambda s: (s.recall, s.precision, -s.seconds))


def useless_agents(s: ModelScore, min_findings: int = 5) -> list[str]:
    """Specialists that reported plenty and never hit a real bug on this repo."""
    return [a for a, (hit, total) in s.agents.items() if total >= min_findings and hit == 0]


def save(repo: Path, results: list[CaseResult], scores: list[ModelScore]) -> Path:
    out = repo / ".crg" / "eval" / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "results": [asdict(r) for r in results],
        "scores": [{**asdict(s), "recall": s.recall, "precision": s.precision} for s in scores],
    }, indent=2))
    return out
