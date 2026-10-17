"""Build the exact layer: walk the repo, extract facts, resolve edges, store."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from code_review_graph.extract import FileFacts, extractor_for
from code_review_graph.graph import detect_clusters
from code_review_graph.resolve import Resolver
from code_review_graph.store import GraphStore

SKIP_DIRS = {".git", ".crg", ".venv", "venv", "node_modules", "__pycache__", "build", "dist", ".tox"}


def list_files(repo: Path) -> list[str]:
    """Repo-relative paths, honoring .gitignore when the repo is a git checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "-co", "--exclude-standard"],
            capture_output=True, text=True, check=True,
        ).stdout
        return sorted(p for p in out.splitlines() if (repo / p).is_file())
    except (subprocess.CalledProcessError, FileNotFoundError):
        found = []
        for root, dirs, files in os.walk(repo):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
            for name in files:
                found.append(str((Path(root) / name).relative_to(repo).as_posix()))
        return sorted(found)


def index_repo(repo: Path, store: GraphStore) -> dict[str, int]:
    all_facts: list[FileFacts] = []
    hashes: dict[str, str] = {}
    for rel in list_files(repo):
        extract = extractor_for(rel)
        if extract is None:
            continue
        source = (repo / rel).read_bytes()
        hashes[rel] = hashlib.sha256(source).hexdigest()
        all_facts.append(extract(rel, source))

    edges = Resolver(all_facts).edges()
    nodes = [n for f in all_facts for n in f.nodes]
    store.replace_graph(nodes, edges, hashes)
    store.replace_clusters(detect_clusters(store))
    return store.stats()
