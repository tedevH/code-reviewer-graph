"""SQLite-backed graph store.

Two layers live in one database:
  * the exact layer (nodes + edges), rebuilt from source on every `index`
  * the meaning layer (summaries), written by an LLM and kept across rebuilds,
    keyed by file hash so only changed modules get re-summarized
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id         TEXT PRIMARY KEY,   -- "pkg/mod.py" or "pkg/mod.py::Class.method"
    kind       TEXT NOT NULL,      -- module | class | function | method
    name       TEXT NOT NULL,      -- qualified name inside the module
    path       TEXT NOT NULL,
    start_line INTEGER NOT NULL,   -- 1-based, inclusive
    end_line   INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS edges (
    src        TEXT NOT NULL,
    dst        TEXT NOT NULL,
    kind       TEXT NOT NULL,      -- calls | contains | imports
    confidence TEXT NOT NULL,      -- exact | name | ambiguous (see resolve.py)
    PRIMARY KEY (src, dst, kind)
);
CREATE INDEX IF NOT EXISTS edges_dst ON edges (dst, kind);
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS summaries (
    path       TEXT PRIMARY KEY,
    hash       TEXT NOT NULL,      -- file hash the summary was written from
    summary    TEXT NOT NULL,
    model      TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS clusters (
    path    TEXT PRIMARY KEY,
    cluster TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cluster_summaries (
    cluster    TEXT PRIMARY KEY,
    hash       TEXT NOT NULL,      -- hash over member paths + file hashes
    title      TEXT NOT NULL,
    summary    TEXT NOT NULL,
    model      TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
-- Learnings written back by reviews or people. Never rebuilt, only appended.
CREATE TABLE IF NOT EXISTS notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    target     TEXT NOT NULL,      -- node id (module path or path::qualname)
    text       TEXT NOT NULL,
    source     TEXT NOT NULL,      -- "human" | "review"
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    name: str
    path: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    kind: str
    confidence: str = "exact"


def db_path(repo: Path) -> Path:
    return repo / ".crg" / "graph.db"


class GraphStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    @classmethod
    def for_repo(cls, repo: Path) -> "GraphStore":
        return cls(db_path(repo))

    def close(self) -> None:
        self.conn.close()

    # --- exact layer -------------------------------------------------------

    def replace_graph(self, nodes: list[Node], edges: list[Edge], files: dict[str, str]) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM nodes")
            self.conn.execute("DELETE FROM edges")
            self.conn.execute("DELETE FROM files")
            self.conn.executemany(
                "INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?, ?)",
                [(n.id, n.kind, n.name, n.path, n.start_line, n.end_line) for n in nodes],
            )
            self.conn.executemany(
                "INSERT OR IGNORE INTO edges VALUES (?, ?, ?, ?)",
                [(e.src, e.dst, e.kind, e.confidence) for e in edges],
            )
            self.conn.executemany("INSERT INTO files VALUES (?, ?)", list(files.items()))
            # Summaries of deleted files are dead weight.
            self.conn.execute("DELETE FROM summaries WHERE path NOT IN (SELECT path FROM files)")

    def node(self, node_id: str) -> Node | None:
        row = self.conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return Node(**row) if row else None

    def find_nodes(self, query: str) -> list[Node]:
        """Match an exact id, "module.Qual.name", a qualified name, or a bare name suffix."""
        rows = self.conn.execute(
            """SELECT n.* FROM nodes n JOIN nodes m ON m.id = n.path
               WHERE n.id = ? OR n.name = ? OR n.name LIKE ? OR m.name || '.' || n.name = ?
               ORDER BY n.id""",
            (query, query, f"%.{query}", query),
        ).fetchall()
        return [Node(**r) for r in rows]

    def nodes_in_file(self, path: str) -> list[Node]:
        rows = self.conn.execute("SELECT * FROM nodes WHERE path = ?", (path,)).fetchall()
        return [Node(**r) for r in rows]

    def callers(self, node_id: str) -> list[tuple[str, str]]:
        rows = self.conn.execute(
            "SELECT src, confidence FROM edges WHERE dst = ? AND kind = 'calls'", (node_id,)
        ).fetchall()
        return [(r["src"], r["confidence"]) for r in rows]

    def callees(self, node_id: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT dst FROM edges WHERE src = ? AND kind = 'calls'", (node_id,)
        ).fetchall()
        return [r["dst"] for r in rows]

    def imported_by(self, path: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT src FROM edges WHERE dst = ? AND kind = 'imports'", (path,)
        ).fetchall()
        return [r["src"] for r in rows]

    def imports_of(self, path: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT dst FROM edges WHERE src = ? AND kind = 'imports'", (path,)
        ).fetchall()
        return [r["dst"] for r in rows]

    def file_hashes(self) -> dict[str, str]:
        return {r["path"]: r["hash"] for r in self.conn.execute("SELECT * FROM files")}

    def stats(self) -> dict[str, int]:
        out = {}
        for kind, count in self.conn.execute("SELECT kind, COUNT(*) FROM nodes GROUP BY kind"):
            out[kind] = count
        for kind, count in self.conn.execute("SELECT kind, COUNT(*) FROM edges GROUP BY kind"):
            out[f"{kind} edges"] = count
        out["summaries"] = self.conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
        return out

    # --- meaning layer -----------------------------------------------------

    def summary(self, path: str) -> str | None:
        row = self.conn.execute("SELECT summary FROM summaries WHERE path = ?", (path,)).fetchone()
        return row["summary"] if row else None

    def stale_summaries(self) -> list[str]:
        """Modules with no summary, or whose summary was written from older source."""
        rows = self.conn.execute(
            """SELECT f.path FROM files f LEFT JOIN summaries s ON s.path = f.path
               WHERE s.hash IS NULL OR s.hash != f.hash ORDER BY f.path"""
        ).fetchall()
        return [r["path"] for r in rows]

    def put_summary(self, path: str, file_hash: str, summary: str, model: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO summaries (path, hash, summary, model) VALUES (?, ?, ?, ?)",
                (path, file_hash, summary, model),
            )

    # --- clusters ------------------------------------------------------------

    def replace_clusters(self, assignment: dict[str, str]) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM clusters")
            self.conn.executemany("INSERT INTO clusters VALUES (?, ?)", sorted(assignment.items()))
            self.conn.execute(
                "DELETE FROM cluster_summaries WHERE cluster NOT IN (SELECT cluster FROM clusters)"
            )

    def clusters(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for r in self.conn.execute("SELECT * FROM clusters ORDER BY cluster, path"):
            out.setdefault(r["cluster"], []).append(r["path"])
        return out

    def cluster_of(self, path: str) -> str | None:
        row = self.conn.execute("SELECT cluster FROM clusters WHERE path = ?", (path,)).fetchone()
        return row["cluster"] if row else None

    def cluster_hash(self, cluster: str) -> str:
        hashes = self.file_hashes()
        members = self.clusters().get(cluster, [])
        joined = "\n".join(f"{p}:{hashes.get(p, '')}" for p in members)
        return hashlib.sha256(joined.encode()).hexdigest()

    def cluster_summary(self, cluster: str) -> tuple[str, str] | None:
        row = self.conn.execute(
            "SELECT title, summary FROM cluster_summaries WHERE cluster = ?", (cluster,)
        ).fetchone()
        return (row["title"], row["summary"]) if row else None

    def stale_clusters(self) -> list[str]:
        current = {
            r["cluster"]: r["hash"] for r in self.conn.execute("SELECT * FROM cluster_summaries")
        }
        return [c for c in self.clusters() if current.get(c) != self.cluster_hash(c)]

    def put_cluster_summary(self, cluster: str, title: str, summary: str, model: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO cluster_summaries (cluster, hash, title, summary, model) "
                "VALUES (?, ?, ?, ?, ?)",
                (cluster, self.cluster_hash(cluster), title, summary, model),
            )

    # --- notes (write-back memory) -------------------------------------------

    def add_note(self, target: str, text: str, source: str = "human") -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO notes (target, text, source) VALUES (?, ?, ?)", (target, text, source)
            )

    def notes_for_file(self, path: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM notes WHERE target = ? OR target LIKE ? ORDER BY id",
            (path, f"{path}::%"),
        ).fetchall()
