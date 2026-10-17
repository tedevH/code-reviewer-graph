import shutil
from pathlib import Path

import pytest

from code_review_graph.indexer import index_repo
from code_review_graph.store import GraphStore

FIXTURE = Path(__file__).parent / "fixtures" / "shop"


@pytest.fixture
def repo(tmp_path):
    dest = tmp_path / "shop"
    shutil.copytree(FIXTURE, dest)
    return dest


@pytest.fixture
def store(repo):
    s = GraphStore.for_repo(repo)
    index_repo(repo, s)
    yield s
    s.close()
