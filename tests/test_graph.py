from code_review_graph.context import build_context, parse_learnings, render_markdown
from code_review_graph.diff import changed_lines, changed_nodes
from code_review_graph.extract.python import extract, module_name
from code_review_graph.graph import impact


def edges(store, kind="calls"):
    rows = store.conn.execute("SELECT src, dst, confidence FROM edges WHERE kind = ?", (kind,))
    return {(r["src"], r["dst"]): r["confidence"] for r in rows}


def test_module_names():
    assert module_name("shop/orders.py") == "shop.orders"
    assert module_name("shop/__init__.py") == "shop"


def test_extracts_definitions_and_nested_qualnames():
    src = b"class A:\n    class B:\n        def m(self):\n            self.n()\n"
    facts = extract("pkg/x.py", src)
    assert [n.id for n in facts.nodes] == ["pkg/x.py", "pkg/x.py::A", "pkg/x.py::A.B", "pkg/x.py::A.B.m"]
    assert facts.calls[0].enclosing_class == "A.B"


def test_resolves_calls_across_import_styles(store):
    calls = edges(store)
    # plain local call
    assert calls[("shop/models.py::Order.total", "shop/models.py::parse_price")] == "exact"
    # `from shop.models import Order` + constructor
    assert ("shop/orders.py::place_order", "shop/models.py::Order") in calls
    assert ("shop/orders.py::place_order", "shop/models.py::Order.__init__") in calls
    # `from . import pricing` then pricing.apply_discount(...)
    assert calls[("shop/orders.py::place_order", "shop/pricing.py::apply_discount")] == "exact"
    # `import shop` then shop.place_order(...), re-exported by shop/__init__.py
    assert calls[("shop/api.py::create_order_endpoint", "shop/orders.py::place_order")] == "exact"
    # order.total() on an untyped object: unique-name fallback
    assert calls[("shop/pricing.py::apply_discount", "shop/models.py::Order.total")] == "name"
    # stdlib calls produce no edges
    assert not any("dumps" in dst for _, dst in calls)


def test_import_edges(store):
    imports = edges(store, "imports")
    assert ("shop/orders.py", "shop/pricing.py") in imports
    assert ("shop/api.py", "shop/__init__.py") in imports


def test_impact_walks_callers(store):
    hits = {h.node.id: h for h in impact(store, ["shop/models.py::parse_price"], max_depth=5)}
    assert hits["shop/models.py::Order.total"].depth == 1
    assert hits["shop/pricing.py::apply_discount"].depth == 2
    assert hits["shop/pricing.py::apply_discount"].confidence == "name"
    assert hits["shop/orders.py::place_order"].depth == 3
    assert hits["shop/api.py::create_order_endpoint"].depth == 4
    assert "tests/test_orders.py::test_place_order" in hits


def test_diff_maps_to_innermost_node(store):
    diff = (
        "diff --git a/shop/models.py b/shop/models.py\n"
        "--- a/shop/models.py\n+++ b/shop/models.py\n"
        "@@ -10 +10 @@ class Order:\n-        old\n+        new\n"
    )
    assert changed_lines(diff) == {"shop/models.py": {10}}
    assert changed_nodes(store, diff) == ["shop/models.py::Order.total"]


def test_context_mentions_tests_and_callers(store):
    ctx = build_context(store, ["shop/pricing.py::tax"], max_depth=3)
    md = render_markdown(store, ctx)
    assert "place_order" in md
    assert "tests/test_orders.py" in md


def test_clusters_group_coupled_modules(store):
    clusters = store.clusters()
    shop = next(ms for ms in clusters.values() if "shop/orders.py" in ms)
    assert {"shop/models.py", "shop/pricing.py"} <= set(shop)


def test_parse_learnings():
    md = "Looks fine.\n\n## Learnings\n- shop/models.py::parse_price: prices are strings with a $ prefix\n- junk\n"
    assert parse_learnings(md) == [("shop/models.py::parse_price", "prices are strings with a $ prefix")]
