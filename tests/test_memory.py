"""The self-improvement loop: outcomes, track records, verified learnings, eval."""

import argparse
import json
import os
import shutil
import subprocess
from datetime import timedelta
from pathlib import Path

import pytest

from code_review_graph import evaluate, memory
from code_review_graph.cli import _apply_config, _choose_agents, main
from code_review_graph.events import EventBus
from code_review_graph.indexer import index_repo
from code_review_graph.store import GraphStore

FIXTURE = Path(__file__).parent / "fixtures" / "shop"
REVIEW = """## Summary
x

## Findings
### high: shop/pricing.py:5 discount inverted
Failure: 10% off returns 10% of the total. (from: Correctness, Security, Blast Radius)

### low: shop/models.py:2 parse_price nit
Failure: meh.

## Learnings
- shop/pricing.py::apply_discount: pct is a percentage off, 0-100

## Outdated notes
none
"""
SPECIALISTS = {
    "Correctness": "### high: shop/pricing.py:5 inverted\n### low: shop/models.py:2 nit",
    "Security": "### medium: shop/pricing.py:4 inverted",
    "Performance": "### low: shop/orders.py:10 slow",       # filtered out by the judge
}


def git(repo, *args, date=None):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    if date:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = date
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=True, env=env).stdout


@pytest.fixture
def gitrepo(tmp_path):
    repo = tmp_path / "shop"
    shutil.copytree(FIXTURE, repo)
    git(repo, "init", "-q", "-b", "main")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "init")
    store = GraphStore.for_repo(repo)
    index_repo(repo, store)
    yield repo, store
    store.close()


def record(repo, store):
    return memory.record_review(store, repo, REVIEW, SPECIALISTS, "HEAD", "m", "m")


def test_record_review_numbers_findings_and_credits_agents(gitrepo):
    repo, store = gitrepo
    review_id, md = record(repo, store)
    assert "discount inverted [f1]" in md and "parse_price nit [f2]" in md
    rows = store.conn.execute("SELECT id, agents, snapshot FROM findings ORDER BY id").fetchall()
    assert rows[0]["agents"] == "Correctness,Security"
    assert "pct / 100" in rows[0]["snapshot"]
    perf = store.conn.execute("SELECT finding_id FROM agent_findings WHERE agent='Performance'").fetchone()
    assert perf["finding_id"] is None
    # named by the judge but reported elsewhere: still credited
    blast = store.conn.execute("SELECT finding_id FROM agent_findings WHERE agent='Blast Radius'").fetchone()
    assert blast["finding_id"] == 1


def test_fixed_code_counts_as_acted_and_untouched_as_ignored(gitrepo):
    repo, store = gitrepo
    record(repo, store)
    # f1: someone fixes the flagged line
    p = repo / "shop" / "pricing.py"
    p.write_text(p.read_text().replace("order.total() * (1 - pct / 100)", "order.total() * (100 - pct) / 100"))
    # f2: models.py gets committed later with the flagged line untouched
    m = repo / "shop" / "models.py"
    m.write_text(m.read_text() + "\n# unrelated\n")
    git(repo, "commit", "-qam", "later", date="2099-01-01T00:00:00")

    assert memory.update_outcomes(store, repo) == {"acted": 1, "ignored": 1}
    statuses = [r["status"] for r in store.conn.execute("SELECT status FROM findings ORDER BY id")]
    assert statuses == ["acted", "ignored"]


def test_open_findings_are_ignored_after_a_week(gitrepo):
    repo, store = gitrepo
    record(repo, store)
    later = memory._now() + timedelta(days=8)
    assert memory.update_outcomes(store, repo, now=later)["ignored"] == 2


def test_track_records_and_weak_agents_are_skipped(gitrepo):
    repo, store = gitrepo
    for _ in range(4):
        record(repo, store)
    for row in store.conn.execute("SELECT id, title FROM findings").fetchall():
        memory.feedback(store, row["id"], "good" if "discount" in row["title"] else "bad")
    records = memory.agent_records(store)
    assert records["Security"].precision > 0.8          # only ever flagged the real bug
    assert records["Correctness"].precision == pytest.approx(0.5)

    # An agent whose findings keep getting rejected gets switched off.
    for _ in range(6):
        memory._store_agent_findings(store, 1, [("Performance", "shop/models.py", 2, "low")],
                                     {("shop/models.py", 2): 2})
    assert memory.agent_records(store)["Performance"].weak
    args = argparse.Namespace(agents=None)
    logs = []
    chosen = _choose_agents(args, store, logs.append)
    assert "performance" not in chosen and "security" in chosen
    assert "Skipping Performance" in logs[0]
    # ...unless asked for explicitly
    assert _choose_agents(argparse.Namespace(agents="performance"), store, logs.append) == ["performance"]

    acted, ignored = memory.history_examples(store)
    assert "discount inverted" in acted[0] and "parse_price nit" in ignored[0]


def test_learnings_wait_for_confirmation(gitrepo):
    repo, store = gitrepo
    review_id, md = record(repo, store)
    memory.save_learnings(store, repo, review_id, [("shop/pricing.py::apply_discount", "pct is 0-100")])
    assert store.notes_for_file("shop/pricing.py") == []               # pending: not shown to reviews
    memory.feedback(store, 1, "good")
    assert memory.update_learnings(store, repo)["verified"] == 1
    assert store.notes_for_file("shop/pricing.py")[0]["text"] == "pct is 0-100"


def test_learnings_expire_when_rejected_or_code_rewritten(gitrepo):
    repo, store = gitrepo
    review_id, _ = record(repo, store)
    memory.save_learnings(store, repo, review_id, [("shop/pricing.py::apply_discount", "pct is 0-100")])
    memory.feedback(store, 1, "bad")
    memory.feedback(store, 2, "bad")
    assert memory.update_learnings(store, repo)["expired"] == 1

    nid = store.add_note("shop/models.py::parse_price", "strips $", snapshot=store.node_source(repo, "shop/models.py::parse_price"))
    (repo / "shop" / "models.py").write_text(
        "def parse_price(text):\n    cleaned = text.replace(',', '').replace('USD', '').strip()\n"
        "    value = decimal.Decimal(cleaned)\n    return value.quantize(decimal.Decimal('0.01'))\n"
        "\n\nclass Order:\n    def __init__(self, items):\n        self.items = items\n\n"
        "    def total(self):\n        return sum(parse_price(i['price']) for i in self.items)\n")
    index_repo(repo, store)
    memory.update_learnings(store, repo)
    row = store.conn.execute("SELECT status, reason FROM notes WHERE id = ?", (nid,)).fetchone()
    assert row["status"] == "expired" and "changed substantially" in row["reason"]


def test_judge_can_mark_notes_outdated(gitrepo):
    repo, store = gitrepo
    nid = store.add_note("shop/pricing.py::tax", "tax is 8%")
    expired = memory.expire_outdated(store, f"## Findings\nx\n\n## Outdated notes\n- n{nid}: rate changed\n", 1)
    assert expired == [nid]
    assert store.notes_for_file("shop/pricing.py") == []


def test_config_from_eval_sets_defaults_but_flags_win(tmp_path):
    (tmp_path / ".crg").mkdir()
    (tmp_path / ".crg" / "config.json").write_text(json.dumps({"model": "best:7b", "agents": ["security"]}))
    args = argparse.Namespace(model=None, judge_model=None, agents=None)
    _apply_config(args, tmp_path)
    assert args.model == "best:7b" and args.agents == "security"
    args = argparse.Namespace(model="mine:3b", judge_model=None, agents="tests")
    _apply_config(args, tmp_path)
    assert args.model == "mine:3b" and args.agents == "tests"


def test_impact_reindexes_before_running(gitrepo, capsys):
    repo, store = gitrepo
    p = repo / "shop" / "pricing.py"
    p.write_text(p.read_text() + "\n\ndef shipping(amount):\n    return tax(amount) + 5\n")
    diff = git(repo, "diff", "-U0")
    (repo / "change.diff").write_text(diff)
    assert main(["--repo", str(repo), "impact", "--diff-file", str(repo / "change.diff")]) == 0
    assert "shipping" in capsys.readouterr().out


class BugSpotter:
    """Fake model: every specialist and the judge point at the line the fix touched."""

    provider, model, context_tokens = "fake", "fake:1b", None

    def complete(self, system, prompt, max_tokens=4096, on_token=None):
        if "grade a code reviewer" in system:
            assert "Fix inverted discount" in prompt and "+    return order.total() * (1 - pct / 100)" in prompt
            return "The discount is inverted.\n1: SAME - names the inverted discount"
        if "lead reviewer" in system:
            return "## Summary\nx\n\n## Findings\n### high: shop/pricing.py:5 discount math\nFailure: y\n"
        return "## Analysis\nz\n\n## Findings\n### high: shop/pricing.py:5 discount math"


def test_eval_replays_bug_fixes(gitrepo):
    repo, store = gitrepo
    p = repo / "shop" / "pricing.py"
    p.write_text(p.read_text().replace("(1 - pct / 100)", "pct / 100"))
    git(repo, "commit", "-qam", "Change discount")
    p.write_text(p.read_text().replace("order.total() * pct / 100", "order.total() * (1 - pct / 100)"))
    git(repo, "commit", "-qam", "Fix inverted discount")
    (repo / "README").write_text("docs")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "Fix typo in docs")     # no source files: not a case

    cases = evaluate.mine_cases(repo, limit=5)
    assert [c.subject for c in cases] == ["Fix inverted discount"]
    assert "-    return order.total() * (1 - pct / 100)" in cases[0].diff   # fixed -> buggy

    llm = BugSpotter()
    result = evaluate.grade_case(*evaluate.run_case(cases[0], repo, llm, EventBus(), workers=2), llm)
    assert result.error is None and result.caught and result.on_target == 1 and result.located == 1
    assert result.agents["Correctness"] == [1, 1]
    assert "crg-eval" not in git(repo, "worktree", "list")   # cleaned up
    s = evaluate.score([result])
    assert s.recall == 1.0 and s.precision == 1.0


class OffTarget(BugSpotter):
    def complete(self, system, prompt, max_tokens=4096, on_token=None):
        if "grade a code reviewer" in system:
            return "Inverted discount.\n1: DIFFERENT - complains about style"   # right lines, wrong complaint
        return super().complete(system, prompt, max_tokens, on_token)


def test_eval_requires_identifying_the_bug_not_just_its_lines(gitrepo):
    repo, store = gitrepo
    p = repo / "shop" / "pricing.py"
    p.write_text(p.read_text().replace("(1 - pct / 100)", "pct / 100"))
    git(repo, "commit", "-qam", "Change discount")
    p.write_text(p.read_text().replace("order.total() * pct / 100", "order.total() * (1 - pct / 100)"))
    git(repo, "commit", "-qam", "Fix inverted discount")
    case = evaluate.mine_cases(repo)[0]
    llm = OffTarget()
    result = evaluate.grade_case(*evaluate.run_case(case, repo, llm, EventBus(), workers=2), llm)
    assert result.located == 1 and not result.caught and result.agents["Correctness"] == [0, 1]
