"""Build the exact layer: walk the repo, extract facts, resolve edges, store."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from code_review_graph.extract import FileFacts, extractor_for
from code_review_graph.extract.generic import UnsupportedLanguage
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


def _go_module(repo: Path) -> str | None:
    gomod = repo / "go.mod"
    if gomod.is_file():
        for line in gomod.read_text(errors="replace").splitlines():
            if line.startswith("module "):
                return line.split()[1].strip()
    return None


MAX_FILE_BYTES = 1_000_000   # generated bundles and vendored blobs aren't worth parsing


def index_repo(repo: Path, store: GraphStore, warn=print) -> dict[str, int]:
    all_facts: list[FileFacts] = []
    hashes: dict[str, str] = {}
    failed_langs: set[str] = set()
    for rel in list_files(repo):
        extract = extractor_for(rel)
        if extract is None:
            continue
        source = (repo / rel).read_bytes()
        if len(source) > MAX_FILE_BYTES:
            continue
        try:
            facts = extract(rel, source)
        except UnsupportedLanguage as e:
            lang = str(e).split(":", 1)[0]
            if lang not in failed_langs:
                failed_langs.add(lang)
                warn(f"warning: no grammar for {e}; skipping those files")
            continue
        hashes[rel] = hashlib.sha256(source).hexdigest()
        all_facts.append(facts)

    edges = Resolver(all_facts, go_module=_go_module(repo)).edges()
    nodes = [n for f in all_facts for n in f.nodes]
    store.replace_graph(nodes, edges, hashes)
    store.replace_clusters(detect_clusters(store))
    return store.stats()
