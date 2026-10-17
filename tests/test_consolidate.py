from code_review_graph.consolidate import consolidate
from code_review_graph.context import build_context
from code_review_graph.diff import changed_nodes

DIFF = (
    "diff --git a/shop/models.py b/shop/models.py\n--- a/shop/models.py\n+++ b/shop/models.py\n"
    "@@ -2 +2 @@ def parse_price(text):\n-    return round(float(text.strip(\"$\")), 2)\n"
    "+    return round(float(text), 2)\n"
)

REVIEW = """## Summary
parse_price no longer strips "$".

## Findings
### high: shop/models.py:2 parse_price crashes on "$10"
Failure: float("$10") raises ValueError.

### medium: `shop/models.py:1` duplicate of the same bug
Failure: same thing.

### high: shop/models.py:10 Order.total breaks
Failure: calls parse_price with "$10".

### high: shop/orders.py:8 place_order breaks
Failure: total() raises.

### high: tests/test_orders.py:5 test asserts wrong total
Failure: `assert place_order([{"price": "$99"}]) == 7` fails.

### low: tests/test_orders.py:5 test uses real data
Evidence: `place_order([{"price": "$10"}])`

## Learnings
- shop/models.py::parse_price: prices arrive with a $ prefix
"""


def test_merges_restatements_into_root_cause(repo, store):
    ctx = build_context(store, changed_nodes(store, DIFF))
    out, stats = consolidate(REVIEW, DIFF, ctx["hits"] + ctx["tests"], repo)

    assert out.count("### ") == 1                                   # one finding left
    assert "parse_price crashes" in out
    assert "Affects: `Order.total` (shop/models.py:9), `place_order` (shop/orders.py:7)" in out
    assert stats == {"merged_duplicates": 1, "folded_into_root_cause": 4, "dropped_unsupported_tests": 0}
    assert out.rstrip().endswith("prices arrive with a $ prefix")  # learnings kept


def test_drops_test_claims_that_quote_nothing_real(repo, store):
    review = REVIEW.replace("tests/test_orders.py:5", "tests/test_other.py:5")
    (repo / "tests" / "test_other.py").write_text("def test_x():\n    assert True\n")
    from code_review_graph.indexer import index_repo

    index_repo(repo, store)
    ctx = build_context(store, changed_nodes(store, DIFF))
    out, stats = consolidate(review, DIFF, ctx["hits"] + ctx["tests"], repo)
    assert stats["dropped_unsupported_tests"] == 2
    assert "test_other" not in out


def test_review_without_findings_section_is_untouched(repo):
    assert consolidate("All good.", DIFF, [], repo) == ("All good.", {})
