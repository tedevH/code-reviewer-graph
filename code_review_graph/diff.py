"""Map a unified diff onto graph nodes."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from code_review_graph.store import GraphStore

HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def git_diff(repo: Path, base: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "diff", "-U0", base],
        capture_output=True, text=True, check=True,
    ).stdout


def changed_lines(diff_text: str) -> dict[str, set[int]]:
    """New-side line numbers touched per file. Pure deletions mark the line they sat at."""
    out: dict[str, set[int]] = {}
    current: str | None = None
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            target = line[4:].strip()
            current = None if target == "/dev/null" else target.removeprefix("b/")
            if current:
                out.setdefault(current, set())
        elif current and (m := HUNK.match(line)):
            start, count = int(m.group(1)), int(m.group(2) or 1)
            out[current].update(range(start, start + count) if count else {max(start, 1)})
    return out


def changed_nodes(store: GraphStore, diff_text: str) -> list[str]:
    """Innermost function/method/class containing each changed line, else the module."""
    seeds: list[str] = []
    for path, lines in changed_lines(diff_text).items():
        nodes = store.nodes_in_file(path)
        if not nodes:
            continue
        for line in sorted(lines):
            containing = [n for n in nodes if n.start_line <= line <= n.end_line]
            if not containing:
                continue
            innermost = min(containing, key=lambda n: n.end_line - n.start_line)
            if innermost.id not in seeds:
                seeds.append(innermost.id)
    return seeds
