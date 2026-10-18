"""Outcome memory: what happened to every finding, and what that teaches crg.

Each review is stored with its findings and a snapshot of the code each
finding points at. Later runs check what became of them:

  acted    the flagged code changed (someone fixed it)          positive
  ignored  the flagged code survived later commits, or a week    weak negative
  good     you said so: `crg feedback f12 good`                  positive
  bad      you said so: `crg feedback f12 bad`                   negative

From those outcomes crg derives:
  * a track record per specialist, which the judge sees, and which turns off
    specialists whose findings your team keeps ignoring
  * examples of findings this team acted on and ignored, for the judge
  * which learnings to trust: a review's learnings stay "pending" (hidden
    from future reviews) until one of its findings is acted on; they expire
    when the code they describe changes a lot or a later review says so
"""

from __future__ import annotations

import difflib
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from code_review_graph.agents import parse_specialist_findings
from code_review_graph.consolidate import parse_findings
from code_review_graph.store import GraphStore

POSITIVE = {"acted": 1.0, "good": 1.0}
NEGATIVE = {"ignored": 0.5, "bad": 1.0}
IGNORE_AFTER = timedelta(days=7)
PENDING_LEARNING_TTL = timedelta(days=14)
SNAPSHOT_RADIUS = 2           # lines either side of the flagged line
MATCH_WINDOW = 5
DRIFT_THRESHOLD = 0.5         # learning expires below this similarity to its snapshot
WEAK_AGENT_PRECISION = 0.2
WEAK_AGENT_MIN_DECIDED = 5


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                              check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _norm_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def snapshot(repo: Path, file: str, line: int | None) -> str | None:
    try:
        lines = (repo / file).read_text(errors="replace").splitlines()
    except OSError:
        return None
    if line is None:
        return None
    lo, hi = max(0, line - 1 - SNAPSHOT_RADIUS), min(len(lines), line + SNAPSHOT_RADIUS)
    return "\n".join(lines[lo:hi]) or None


def _still_there(snap: str, text: str) -> bool:
    """Is the snapshot still in the file, allowing for lines moving and re-indentation?"""
    want, have = _norm_lines(snap), _norm_lines(text)
    if not want:
        return True
    n = len(want)
    return any(have[i:i + n] == want for i in range(len(have) - n + 1))


# --- recording ----------------------------------------------------------------

def record_review(store: GraphStore, repo: Path, review_md: str, specialist_findings: dict[str, str],
                  base: str | None, model: str, judge_model: str) -> tuple[int, str]:
    """Store a finished review. Returns its id and the markdown with finding ids added."""
    head = _git(repo, "rev-parse", "HEAD").strip() or None
    with store.conn:
        review_id = store.conn.execute(
            "INSERT INTO reviews (head, base, model, judge_model) VALUES (?, ?, ?, ?)",
            (head, base, model, judge_model),
        ).lastrowid

    raw = [(agent, f, ln, sev) for agent, text in specialist_findings.items() for f, ln, sev in parse_specialist_findings(text)]
    head_md, found, tail = review_md.partition("## Findings")
    if not found:
        _store_agent_findings(store, review_id, raw, {})
        return review_id, review_md
    section, marker, rest = tail.partition("\n## ")
    _, findings = parse_findings(section)

    ids: dict[int, int] = {}       # index in `findings` -> row id
    for i, f in enumerate(findings):
        agents = sorted({a for a, file, ln, _ in raw
                         if f.file and file.strip("`") == f.file.strip("`") and f.line
                         and abs(ln - f.line) <= MATCH_WINDOW})
        title = re.sub(r"^#+\s*", "", f.header)
        with store.conn:
            ids[i] = store.conn.execute(
                """INSERT INTO findings (review_id, file, line, severity, title, body, agents, snapshot)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (review_id, f.file, f.line, f.severity, title, "\n".join(f.body).strip(),
                 ",".join(agents), snapshot(repo, f.file, f.line) if f.file else None),
            ).lastrowid
    _store_agent_findings(store, review_id, raw,
                          {(f.file, f.line): ids[i] for i, f in enumerate(findings) if f.file})
    _credit_named_agents(store, review_id, findings, ids)

    for i, f in enumerate(findings):
        section = section.replace(f.header, f"{f.header} [f{ids[i]}]", 1)
    out = f"{head_md}## Findings{section}"
    if marker:
        out += f"\n## {rest}"
    return review_id, out


def _store_agent_findings(store: GraphStore, review_id: int, raw: list, final: dict) -> None:
    rows = []
    for agent, file, line, sev in raw:
        match = next((fid for (ff, fl), fid in final.items()
                      if ff and ff.strip("`") == file and fl and abs(fl - line) <= MATCH_WINDOW), None)
        rows.append((review_id, agent, file, line, sev, match))
    with store.conn:
        store.conn.executemany("INSERT INTO agent_findings VALUES (?, ?, ?, ?, ?, ?)", rows)


def _credit_named_agents(store: GraphStore, review_id: int, findings: list, ids: dict[int, int]) -> None:
    """Credit specialists the judge names ("(from: A, B)") even if they reported the
    bug at a different line, e.g. Blast Radius pointing at the caller that breaks."""
    for i, f in enumerate(findings):
        m = re.search(r"\(from:\s*([^)]*)\)", "\n".join(f.body))
        if not m:
            continue
        credited = {r["agent"] for r in store.conn.execute(
            "SELECT agent FROM agent_findings WHERE finding_id = ?", (ids[i],))}
        for name in (n.strip() for n in m.group(1).split(",")):
            if name and name not in credited:
                with store.conn:
                    store.conn.execute("INSERT INTO agent_findings VALUES (?, ?, ?, ?, ?, ?)",
                                       (review_id, name, f.file, f.line, f.severity, ids[i]))
                credited.add(name)


def save_learnings(store: GraphStore, repo: Path, review_id: int, learnings: list[tuple[str, str]]) -> int:
    saved = 0
    for target, fact in learnings:
        if not store.node(target):
            target = target.partition("::")[0]   # unknown symbol: attach to its module
        if store.node(target):
            store.add_note(target, fact, source="review", status="pending", review_id=review_id,
                           snapshot=store.node_source(repo, target))
            saved += 1
    return saved


# --- outcomes -------------------------------------------------------------------

def update_outcomes(store: GraphStore, repo: Path, now: datetime | None = None) -> dict[str, int]:
    """Decide open findings from what happened to the code since the review."""
    now = now or _now()
    counts = {"acted": 0, "ignored": 0}
    rows = store.conn.execute(
        "SELECT f.*, r.created_at AS reviewed_at FROM findings f JOIN reviews r ON r.id = f.review_id "
        "WHERE f.status = 'open' AND f.file IS NOT NULL").fetchall()
    for row in rows:
        try:
            text = (repo / row["file"]).read_text(errors="replace")
        except OSError:
            text = None
        status = None
        if text is None:
            status, why = "acted", "file removed"
        elif row["snapshot"] and not _still_there(row["snapshot"], text):
            status, why = "acted", "flagged code changed"
        elif row["snapshot"]:
            reviewed = _parse_time(row["reviewed_at"])
            committed_since = _git(repo, "log", f"--since={reviewed.isoformat()}", "--format=%H",
                                   "--", row["file"]).strip()
            dirty = _git(repo, "status", "--porcelain", "--", row["file"]).strip()
            if committed_since and not dirty:
                status, why = "ignored", "file committed with the flagged code unchanged"
            elif now - reviewed >= IGNORE_AFTER:
                status, why = "ignored", f"unchanged for {IGNORE_AFTER.days} days"
        if status:
            _decide(store, row["id"], status, "auto", why)
            counts[status] += 1
    return counts


def _decide(store: GraphStore, finding_id: int, status: str, by: str, note: str | None) -> None:
    with store.conn:
        store.conn.execute(
            "UPDATE findings SET status = ?, decided_by = ?, decided_at = datetime('now'), note = ? WHERE id = ?",
            (status, by, note, finding_id),
        )


def feedback(store: GraphStore, finding_id: int, verdict: str, note: str | None = None) -> bool:
    if verdict not in ("good", "bad"):
        raise ValueError("verdict must be good or bad")
    if not store.conn.execute("SELECT 1 FROM findings WHERE id = ?", (finding_id,)).fetchone():
        return False
    _decide(store, finding_id, verdict, "human", note)
    return True


# --- what outcomes teach ------------------------------------------------------

@dataclass(frozen=True)
class Record:
    positive: float
    negative: float

    @property
    def decided(self) -> float:
        return self.positive + self.negative

    @property
    def precision(self) -> float:
        return (self.positive + 1) / (self.decided + 2)   # Laplace: new agents start at 50%

    @property
    def weak(self) -> bool:
        return self.decided >= WEAK_AGENT_MIN_DECIDED and self.precision < WEAK_AGENT_PRECISION

    def describe(self) -> str:
        if not self.decided:
            return "no track record yet"
        return f"{self.positive:g} of {self.decided:g} past findings acted on"


def agent_records(store: GraphStore) -> dict[str, Record]:
    pos: dict[str, float] = {}
    neg: dict[str, float] = {}
    rows = store.conn.execute(
        "SELECT a.agent, f.status FROM agent_findings a JOIN findings f ON f.id = a.finding_id")
    for r in rows:
        pos[r["agent"]] = pos.get(r["agent"], 0) + POSITIVE.get(r["status"], 0)
        neg[r["agent"]] = neg.get(r["agent"], 0) + NEGATIVE.get(r["status"], 0)
    return {a: Record(pos.get(a, 0), neg.get(a, 0)) for a in set(pos) | set(neg)}


def history_examples(store: GraphStore, k: int = 2) -> tuple[list[str], list[str]]:
    def pick(statuses: tuple[str, ...]) -> list[str]:
        marks = ",".join("?" * len(statuses))
        rows = store.conn.execute(
            f"SELECT title, note FROM findings WHERE status IN ({marks}) ORDER BY decided_at DESC LIMIT ?",
            (*statuses, k)).fetchall()
        return [r["title"] + (f" (team note: {r['note']})" if r["note"] else "") for r in rows]

    return pick(("good", "acted")), pick(("bad", "ignored"))


# --- learnings -------------------------------------------------------------------

def update_learnings(store: GraphStore, repo: Path, now: datetime | None = None) -> dict[str, int]:
    now = now or _now()
    counts = {"verified": 0, "expired": 0}
    for n in store.conn.execute("SELECT * FROM notes WHERE status IN ('pending', 'verified')").fetchall():
        if n["snapshot"] is not None:
            current = store.node_source(repo, n["target"])
            if current is None:
                _set(store, n, "expired", "the code it describes was removed", counts)
                continue
            ratio = difflib.SequenceMatcher(None, n["snapshot"], current).ratio()
            if ratio < DRIFT_THRESHOLD:
                _set(store, n, "expired", f"code changed substantially ({ratio:.0%} similar)", counts)
                continue
        if n["status"] != "pending" or n["review_id"] is None:
            continue
        outcomes = store.conn.execute("SELECT id, file, status FROM findings WHERE review_id = ?",
                                      (n["review_id"],)).fetchall()
        target_file = n["target"].partition("::")[0]
        confirmed = [o for o in outcomes if o["status"] in POSITIVE and o["file"] == target_file] \
            or [o for o in outcomes if o["status"] in POSITIVE]
        if confirmed:
            _set(store, n, "verified", f"finding f{confirmed[0]['id']} was acted on", counts)
        elif outcomes and all(o["status"] != "open" for o in outcomes):
            _set(store, n, "expired", "none of its review's findings were acted on", counts)
        elif now - _parse_time(n["created_at"]) >= PENDING_LEARNING_TTL:
            _set(store, n, "expired", f"not confirmed within {PENDING_LEARNING_TTL.days} days", counts)
    return counts


def _set(store: GraphStore, note, status: str, reason: str, counts: dict) -> None:
    store.set_note_status(note["id"], status, reason)
    counts[status] += 1


def expire_outdated(store: GraphStore, review_md: str, review_id: int) -> list[int]:
    """Expire notes the judge listed under "## Outdated notes" (e.g. `n12`)."""
    _, found, tail = review_md.partition("## Outdated notes")
    if not found:
        return []
    section = tail.split("\n## ", 1)[0]
    expired = []
    for nid in {int(m) for m in re.findall(r"\bn(\d+)\b", section)}:
        row = store.conn.execute("SELECT status FROM notes WHERE id = ?", (nid,)).fetchone()
        if row and row["status"] in ("verified", "pending"):
            store.set_note_status(nid, "expired", f"review {review_id} found it outdated")
            expired.append(nid)
    return expired


def refresh(store: GraphStore, repo: Path) -> dict[str, int]:
    """Run every outcome-driven update. Cheap; done before each review."""
    return {**update_outcomes(store, repo), **update_learnings(store, repo)}
