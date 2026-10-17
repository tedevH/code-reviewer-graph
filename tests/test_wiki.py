from code_review_graph.wiki import compile_wiki, render_wiki


class FakeLLM:
    provider, model, context_tokens = "fake", "fake", None

    def __init__(self):
        self.calls = 0

    def complete(self, system, prompt, max_tokens=16000, on_token=None):
        self.calls += 1
        if prompt.startswith("These modules"):
            return "TITLE: Order pipeline\nOrders are priced then saved."
        return "Summary of " + prompt.split("`")[1]


def test_compile_is_incremental_and_renders_backlinks(repo, store):
    llm = FakeLLM()
    compile_wiki(repo, store, llm, progress=lambda _: None)
    first = llm.calls
    assert first > 0
    assert store.summary("shop/models.py") == "Summary of shop/models.py"

    # Nothing changed: no API calls.
    compile_wiki(repo, store, llm, progress=lambda _: None)
    assert llm.calls == first

    store.add_note("shop/models.py::parse_price", "prices arrive as '$12.00' strings")
    out = repo / ".crg" / "wiki"
    render_wiki(store, out)
    page = (out / "shop.models.md").read_text()
    assert "Summary of shop/models.py" in page
    assert "[[shop.pricing]]" in page          # used by, from the import graph
    assert "prices arrive as" in page          # write-back note
    index = (out / "index.md").read_text()
    assert "Order pipeline" in index


def test_changed_file_goes_stale(repo, store):
    from code_review_graph.indexer import index_repo

    compile_wiki(repo, store, FakeLLM(), progress=lambda _: None)
    assert store.stale_summaries() == []
    (repo / "shop" / "pricing.py").write_text("def tax(amount):\n    return amount * 0.1\n")
    index_repo(repo, store)
    assert store.stale_summaries() == ["shop/pricing.py"]
