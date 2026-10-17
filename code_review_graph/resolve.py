"""Cross-file resolution: turn per-file facts into graph edges.

Resolution is static and deliberately conservative. A call is linked:
  * "exact" when it resolves through local definitions, self./cls., or imports
  * "name"  when nothing else worked but exactly one definition in the repo
            has that name (e.g. `order.save()` with a single `save` method)
  * "ambiguous" to each of 2-3 same-named definitions when the receiver's
            type is unknown; more candidates than that is noise, so no edge
Calls into modules outside the repo (stdlib, third-party) produce no edge.
"""

from __future__ import annotations

import builtins
from collections import defaultdict

from code_review_graph.extract import CallSite, FileFacts, ImportBinding
from code_review_graph.graph import is_test
from code_review_graph.store import Edge

MAX_AMBIGUOUS = 3


class Resolver:
    def __init__(self, all_facts: list[FileFacts]):
        self.facts = {f.path: f for f in all_facts}
        self.module_to_path: dict[str, str] = {}
        for f in all_facts:
            self.module_to_path[f.module] = f.path
        self.defs: dict[str, dict[str, str]] = defaultdict(dict)   # path -> qual -> id
        self.by_name: dict[str, list[str]] = defaultdict(list)     # bare name -> ids
        for f in all_facts:
            for n in f.nodes:
                if n.kind != "module":
                    self.defs[f.path][n.name] = n.id
                    self.by_name[n.name.rsplit(".", 1)[-1]].append(n.id)
        self.bindings = {
            f.path: {b.local: b for b in f.imports} for f in all_facts
        }

    def edges(self) -> list[Edge]:
        out: list[Edge] = []
        for f in self.facts.values():
            out.extend(self._contains(f))
            out.extend(self._imports(f))
            for call in f.calls:
                for target, confidence in self.resolve_call(f, call):
                    if target != call.owner:
                        out.append(Edge(call.owner, target, "calls", confidence))
        return out

    # --- structure ---------------------------------------------------------

    def _contains(self, f: FileFacts) -> list[Edge]:
        out = []
        local = self.defs[f.path]
        for qual, node_id in local.items():
            parent_qual = qual.rpartition(".")[0]
            parent = local.get(parent_qual, f.path) if parent_qual else f.path
            out.append(Edge(parent, node_id, "contains"))
        return out

    def _imports(self, f: FileFacts) -> list[Edge]:
        out = []
        for b in f.imports:
            target = None
            if b.symbol is not None:
                target = self.module_to_path.get(f"{b.module}.{b.symbol}")
            target = target or self.module_to_path.get(b.module)
            if target and target != f.path:
                out.append(Edge(f.path, target, "imports"))
        return out

    # --- calls ---------------------------------------------------------------

    def resolve_call(self, f: FileFacts, call: CallSite) -> list[tuple[str, str]]:
        parts = call.expr.split(".")
        local = self.defs[f.path]

        if parts[0] in ("self", "cls") and call.enclosing_class and len(parts) == 2:
            target = local.get(f"{call.enclosing_class}.{parts[1]}")
            if target:
                return [(target, "exact")]
            return self._by_unique_name(parts[1], f.path)  # likely an inherited method

        if len(parts) == 1 and parts[0] in local:
            return self._with_constructor(f.path, parts[0], "exact")

        binding, rest = self._longest_binding(f.path, parts)
        if binding is not None:
            # Imports from outside the repo resolve to nothing, and we don't name-guess them.
            return self._follow_import(binding, rest)

        if len(parts) >= 2 or not hasattr(builtins, parts[0]):
            return self._by_unique_name(parts[-1], f.path)
        return []

    def _longest_binding(self, path: str, parts: list[str]) -> tuple[ImportBinding | None, list[str]]:
        bindings = self.bindings[path]
        for k in range(len(parts), 0, -1):
            b = bindings.get(".".join(parts[:k]))
            if b is not None:
                return b, parts[k:]
        return None, parts

    def _follow_import(self, b: ImportBinding, rest: list[str], depth: int = 0) -> list[tuple[str, str]]:
        """Resolve `binding.rest` to a definition, following re-exports a few levels."""
        if b.symbol is not None and f"{b.module}.{b.symbol}" in self.module_to_path:
            module, symbol_path = f"{b.module}.{b.symbol}", rest       # `from pkg import mod`
        elif b.symbol is not None:
            module, symbol_path = b.module, [b.symbol, *rest]
        else:
            module, symbol_path = b.module, rest
        path = self.module_to_path.get(module)
        if path is None or not symbol_path:
            return []
        qual = ".".join(symbol_path)
        if qual in self.defs[path]:
            return self._with_constructor(path, qual, "exact")
        # Re-export, e.g. pkg/__init__.py doing `from .orders import place_order`
        inner = self.bindings[path].get(symbol_path[0])
        if inner is not None and depth < 3:
            return self._follow_import(inner, symbol_path[1:], depth + 1)
        return []

    def _with_constructor(self, path: str, qual: str, confidence: str) -> list[tuple[str, str]]:
        target = self.defs[path][qual]
        out = [(target, confidence)]
        init = self.defs[path].get(f"{qual}.__init__")
        if init:
            out.append((init, confidence))
        return out

    def _by_unique_name(self, name: str, caller_path: str) -> list[tuple[str, str]]:
        candidates = self.by_name.get(name, [])
        if not is_test(caller_path):
            # Production code doesn't call into test files.
            candidates = [c for c in candidates if not is_test(c.partition("::")[0])]
        if len(candidates) == 1:
            return [(candidates[0], "name")]
        if len(candidates) <= MAX_AMBIGUOUS:
            return [(c, "ambiguous") for c in candidates]
        return []
