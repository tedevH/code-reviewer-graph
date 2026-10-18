"""Deterministic clean-up of the judge's review, using the graph instead of trust.

Small models restate one bug at every caller it reaches and invent details
about tests they never saw. Both are checkable:

  * root causes: a finding located in the diff. Duplicates within a few
    lines of each other are merged.
  * restatements: a finding located at a caller from the impact graph is
    folded into the root cause it descends from (via the graph path), as an
    "Affects:" list. The bug is reported once, with its blast radius.
  * unsupported test claims: a finding in a test file outside the diff is
    kept only if it quotes code that really is in that test file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from code_review_graph.diff import changed_lines
from code_review_graph.graph import Hit, is_test

_LOCATION = re.compile(r"`?(?P<file>[\w./-]+\.\w+)`?:(?P<line>\d+)")
_SEVERITY = re.compile(r"\b(high|medium|low)\b", re.IGNORECASE)
_SNIPPET = re.compile(r"`([^`\n]{6,})`")
SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}
DIFF_WINDOW = 5     # models cite old- or new-side line numbers, give or take


@dataclass
class Finding:
    header: str
    body: list[str]
    file: str | None
    line: int | None
    severity: str
    affects: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [self.header, *self.body]
        while lines and not lines[-1].strip():
            lines.pop()
        if self.affects:
            lines.append("Affects: " + ", ".join(dict.fromkeys(self.affects)))
        return "\n".join(lines)


def parse_findings(section: str) -> tuple[str, list[Finding]]:
    preamble, findings, current = [], [], None
    for line in section.splitlines():
        if line.lstrip().startswith("###"):
            loc, sev = _LOCATION.search(line), _SEVERITY.search(line)
            current = Finding(line.strip(), [], loc.group("file") if loc else None,
                              int(loc.group("line")) if loc else None,
                              sev.group(1).lower() if sev else "medium")
            findings.append(current)
        elif current is not None:
            current.body.append(line)
        else:
            preamble.append(line)
    for f in findings:   # location given only in the body
        if f.file is None and (loc := _LOCATION.search("\n".join(f.body))):
            f.file, f.line = loc.group("file"), int(loc.group("line"))
    return "\n".join(preamble).strip(), findings


def _strip(path: str) -> str:
    return path.strip("`").removeprefix("./").removeprefix("a/").removeprefix("b/")


def _quotes_real_code(f: Finding, repo: Path) -> bool:
    try:
        text = (repo / f.file).read_text(errors="replace")
    except OSError:
        return False
    norm = lambda s: re.sub(r"\s+", " ", s.replace('"', "'")).lower()  # noqa: E731
    haystack = norm(text)
    snippets = [m for m in _SNIPPET.findall("\n".join(f.body)) if f.file not in m]
    return any(norm(s) in haystack for s in snippets)


def consolidate(review_md: str, diff_text: str, hits: list[Hit], repo: Path) -> tuple[str, dict]:
    head, found, tail = review_md.partition("## Findings")
    if not found:
        return review_md, {}
    section, rest_marker, rest = tail.partition("\n## ")
    preamble, findings = parse_findings(section)

    changed = changed_lines(diff_text)
    by_id = {h.node.id: h for h in hits}

    def in_diff(f: Finding) -> bool:
        lines = changed.get(_strip(f.file or ""), set())
        return any(abs(f.line - n) <= DIFF_WINDOW for n in lines) if f.line else bool(lines)

    def caller_at(f: Finding) -> Hit | None:
        best = None
        for h in hits:
            if h.depth > 0 and h.node.path == _strip(f.file) and f.line is not None \
                    and h.node.start_line <= f.line <= h.node.end_line:
                if best is None or h.node.end_line - h.node.start_line < best.node.end_line - best.node.start_line:
                    best = h
        return best

    def seed_of(h: Hit) -> Hit:
        seen = set()
        while h.via and h.via in by_id and h.node.id not in seen:
            seen.add(h.node.id)
            h = by_id[h.via]
        return h

    roots: list[Finding] = []
    folded: list[tuple[Finding, Hit]] = []
    kept: list[Finding] = []
    stats = {"merged_duplicates": 0, "folded_into_root_cause": 0, "dropped_unsupported_tests": 0}

    for f in findings:
        if f.file is None:
            kept.append(f)
        elif in_diff(f):
            dup = next((r for r in roots if r.file == f.file and r.line and f.line
                        and abs(r.line - f.line) <= DIFF_WINDOW), None)
            if dup is None:
                roots.append(f)
            else:
                stats["merged_duplicates"] += 1
                if SEVERITY_RANK[f.severity] < SEVERITY_RANK[dup.severity]:
                    dup.severity = f.severity
                    dup.header = _SEVERITY.sub(f.severity, dup.header, count=1)
        elif (h := caller_at(f)) is not None:
            folded.append((f, h))
        elif is_test(_strip(f.file)) and not _quotes_real_code(f, repo):
            stats["dropped_unsupported_tests"] += 1
        else:
            kept.append(f)

    for f, h in folded:
        if not roots:
            kept.append(f)          # nothing in the diff to attach it to: keep it as reported
            continue
        seed = seed_of(h).node
        target = next((r for r in roots if r.file == seed.path and r.line
                       and seed.start_line - DIFF_WINDOW <= r.line <= seed.end_line + DIFF_WINDOW), None)
        target = target or next((r for r in roots if r.file == seed.path), roots[0])
        target.affects.append(f"`{h.node.name}` ({h.node.path}:{h.node.start_line})")
        stats["folded_into_root_cause"] += 1

    final = sorted(roots, key=lambda r: SEVERITY_RANK[r.severity]) + kept
    body = "\n\n".join(f.render() for f in final) if final else "No issues found."
    out = f"{head}## Findings\n\n{(preamble + chr(10) * 2) if preamble else ''}{body}\n"
    if rest_marker:
        out += f"\n## {rest}"
    return out, stats
