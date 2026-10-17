"""Cross-file resolution in languages other than Python.

Each fixture under tests/fixtures/poly/<lang>/ is a tiny repo where one module
calls into another through that language's import mechanism.
"""

import shutil
from pathlib import Path

import pytest

from code_review_graph.graph import impact
from code_review_graph.indexer import index_repo
from code_review_graph.store import GraphStore

POLY = Path(__file__).parent / "fixtures" / "poly"


@pytest.fixture
def index(tmp_path):
    stores = []

    def build(lang: str) -> dict[tuple[str, str], str]:
        repo = tmp_path / lang
        shutil.copytree(POLY / lang, repo)
        store = GraphStore.for_repo(repo)
        index_repo(repo, store)
        stores.append(store)
        rows = store.conn.execute("SELECT src, dst, kind, confidence FROM edges")
        build.store = store
        return {(r["src"], r["dst"], r["kind"]): r["confidence"] for r in rows}

    yield build
    for s in stores:
        s.close()


def calls(edges, src, dst):
    return edges.get((src, dst, "calls"))


def test_typescript(index):
    e = index("ts")
    assert calls(e, "src/order.ts::Order.total", "src/util.ts::fmt") == "exact"            # named import
    assert calls(e, "src/order.ts::Order.total", "src/order.ts::Order.sum") == "exact"      # this.sum()
    assert calls(e, "src/api.ts::handler", "src/order.ts::Order") == "exact"               # new Order()
    assert calls(e, "src/api.ts::handler", "src/order.ts::Order.constructor") == "exact"
    assert calls(e, "src/api.ts::handler", "src/util.ts::fmt") == "exact"                  # namespace import
    assert ("src/order.ts", "src/util.ts", "imports") in e
    hits = {h.node.id for h in impact(index.store, ["src/util.ts::fmt"], max_depth=5)}
    assert "src/api.test.ts" in {h.split("::")[0] for h in hits}


def test_go(index):
    e = index("go")
    assert calls(e, "order/order.go::Order.Total", "util/util.go::Fmt") == "exact"         # go.mod import
    assert calls(e, "order/order.go::Order.Total", "order/order.go::Order.sum") == "exact"  # receiver
    assert calls(e, "order/order.go::Order.sum", "order/helpers.go::addAll") == "exact"     # same package
    assert calls(e, "main.go::main", "order/helpers.go::New") == "exact"                    # aliased import
    assert not any("Sprintf" in dst for (_, dst, _) in e)                                   # stdlib


def test_java(index):
    e = index("java")
    base = "src/main/java/com/shop/"
    assert calls(e, base + "Order.java::Order.total", base + "Util.java::Util.fmt") == "exact"   # same package
    assert calls(e, base + "Order.java::Order.total", base + "Order.java::Order.sum") == "exact"  # implicit this
    assert calls(e, base + "api/Api.java::Api.handle", base + "Order.java::Order") == "exact"     # import + new
    assert (base + "api/Api.java", base + "Order.java", "imports") in e


def test_rust(index):
    e = index("rust")
    assert calls(e, "src/order.rs::Order.total", "src/util.rs::fmt") == "exact"            # use crate::util::fmt
    assert calls(e, "src/order.rs::Order.total", "src/order.rs::Order.sum") == "exact"     # self.sum()
    assert calls(e, "src/main.rs::main", "src/order.rs::Order.new") == "exact"             # Order::new via use


def test_c(index):
    e = index("c")
    assert calls(e, "order.c::order_total", "order.c::sum") == "exact"
    assert calls(e, "order.c::order_total", "util.c::fmt_cents") in ("exact", "name")
    assert ("order.c", "util.h", "imports") in e


def test_ruby(index):
    e = index("ruby")
    assert calls(e, "lib/order.rb::Order.total", "lib/util.rb::Util.fmt") == "exact"
    assert ("lib/order.rb", "lib/util.rb", "imports") in e


def test_php(index):
    e = index("php")
    assert calls(e, "src/Order.php::Order.total", "src/Util.php::Util.fmt") == "exact"
    assert calls(e, "src/Order.php::Order.total", "src/Order.php::Order.sum") == "exact"


def test_builtin_receivers_are_not_name_matched(tmp_path):
    repo = tmp_path / "r"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.ts").write_text(
        "export class Ky {\n  static create() { return new Ky(); }\n}\n"
        "export const clone = () => Object.create(null);\n"
        "export const make = () => Ky.create();\n"
    )
    store = GraphStore.for_repo(repo)
    index_repo(repo, store)
    rows = {(r["src"], r["dst"]) for r in store.conn.execute("SELECT src, dst FROM edges WHERE kind='calls'")}
    store.close()
    assert ("src/a.ts::make", "src/a.ts::Ky.create") in rows
    assert not any(src == "src/a.ts::clone" for src, _ in rows)
