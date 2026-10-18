import threading

from code_review_graph.agents import SPECIALISTS, Job, multi_agent_review, run_pool
from code_review_graph.events import EventBus


class ScriptedLLM:
    """Streams a canned answer per agent and records which agents ran concurrently."""

    provider, model, context_tokens = "fake", "fake", None

    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.barrier = threading.Barrier(len(SPECIALISTS), timeout=5)

    def complete(self, system, prompt, max_tokens=4096, on_token=None):
        if "lead reviewer" in system:
            assert "Findings from Security" in prompt
            answer = "## Summary\nok\n\n## Findings\nNo issues found.\n\n## Learnings\n"
        else:
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            self.barrier.wait()   # proves the specialists really run in parallel
            with self.lock:
                self.active -= 1
            answer = "NO FINDINGS"
        if on_token:
            on_token("thinking", "hmm")
            for word in answer.split(" "):
                on_token("text", word + " ")
        return answer


def test_specialists_run_in_parallel_then_judge():
    bus, llm = EventBus(), ScriptedLLM()
    result = multi_agent_review(llm, bus, "diff", "context", workers=len(SPECIALISTS)).markdown
    assert "No issues found." in result
    assert llm.max_active == len(SPECIALISTS)
    events = bus.since(0, timeout=0)
    done = {e["id"] for e in events if e["type"] == "agent_done" and e["status"] == "done"}
    assert done == {s.id for s in SPECIALISTS} | {"judge"}
    assert any(e["type"] == "delta" and e["kind"] == "thinking" for e in events)


class FailingLLM:
    provider, model, context_tokens = "fake", "fake", None

    def complete(self, system, prompt, max_tokens=4096, on_token=None):
        if prompt == "boom":
            raise RuntimeError("model crashed")
        return prompt.upper()


def test_pool_returns_every_job_and_reports_errors():
    bus = EventBus()
    jobs = [Job(str(i), f"job {i}", "sys", "boom" if i == 2 else f"p{i}") for i in range(6)]
    results = dict((job.key, r) for job, r in run_pool(FailingLLM(), bus, jobs, workers=3))
    assert set(results) == {str(i) for i in range(6)}
    assert isinstance(results["2"], RuntimeError)
    assert results["0"] == "P0"
    errors = [e for e in bus.since(0, timeout=0) if e["type"] == "agent_done" and e["status"] == "error"]
    assert errors and "model crashed" in errors[0]["error"]


def test_consensus_groups_nearby_findings():
    from code_review_graph.agents import consensus

    spots = consensus({
        "Correctness": "### high: shop/models.py:2 bad parse\nFailure: x",
        "Security": "### <HIGH>: `shop/models.py`:3 unvalidated\n### low: shop/api.py:9 meh",
        "Impact": "### : `shop/models.py:1` **High**: Order.total breaks",
        "Tests": "NO FINDINGS",
    })
    assert spots[0]["file"] == "shop/models.py"
    assert spots[0]["agents"] == ["Correctness", "Security", "Impact"]
    assert spots[0]["severities"] == ["high", "high", "high"]
    assert spots[1]["agents"] == ["Security"]


def test_judge_sees_findings_not_analysis():
    from code_review_graph.agents import findings_section

    text = "## Analysis\nlong reasoning\n\n## Findings\n### high: a.py:1 x"
    assert findings_section(text) == "### high: a.py:1 x"
