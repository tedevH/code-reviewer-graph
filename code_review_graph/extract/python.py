"""Python extractor built on tree-sitter."""

from __future__ import annotations

import tree_sitter_python
from tree_sitter import Language, Node as TSNode, Parser

from code_review_graph.extract import CallSite, FileFacts, ImportBinding
from code_review_graph.store import Node

_PARSER = Parser(Language(tree_sitter_python.language()))


def module_name(path: str) -> str:
    """"pkg/sub/mod.py" -> "pkg.sub.mod", "pkg/__init__.py" -> "pkg", "src/pkg/m.py" -> "pkg.m"."""
    parts = path[: -len(".py")].split("/")
    if parts[0] == "src" and len(parts) > 1:
        parts.pop(0)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _package(path: str, module: str) -> str:
    return module if path.endswith("__init__.py") else module.rpartition(".")[0]


def _text(node: TSNode) -> str:
    return node.text.decode("utf-8", errors="replace")


def _resolve_relative(spec: str, package: str) -> str:
    """Turn "..models" into an absolute module name relative to `package`."""
    level = len(spec) - len(spec.lstrip("."))
    rest = spec[level:]
    base = package.split(".") if package else []
    if level > 1:
        base = base[: len(base) - (level - 1)]
    return ".".join([*base, rest] if rest else base)


def extract(path: str, source: bytes) -> FileFacts:
    module = module_name(path)
    package = _package(path, module)
    tree = _PARSER.parse(source)
    root = tree.root_node
    facts = FileFacts(path=path, module=module)
    facts.nodes.append(
        Node(id=path, kind="module", name=module, path=path,
             start_line=1, end_line=root.end_point[0] + 1)
    )

    # scope: list of (kind, name) from outermost to innermost
    def walk(node: TSNode, scope: list[tuple[str, str]], owner: str) -> None:
        for child in node.named_children:
            t = child.type
            if t in ("function_definition", "class_definition"):
                name = _text(child.child_by_field_name("name"))
                qual = ".".join([s[1] for s in scope] + [name])
                if t == "class_definition":
                    kind = "class"
                elif scope and scope[-1][0] == "class":
                    kind = "method"
                else:
                    kind = "function"
                node_id = f"{path}::{qual}"
                facts.nodes.append(
                    Node(id=node_id, kind=kind, name=qual, path=path,
                         start_line=child.start_point[0] + 1, end_line=child.end_point[0] + 1)
                )
                inner_owner = owner if kind == "class" else node_id
                walk(child, [*scope, (kind, name)], inner_owner)
            elif t == "call":
                fn = child.child_by_field_name("function")
                if fn is not None and fn.type in ("identifier", "attribute"):
                    cls = _enclosing_class(scope)
                    facts.calls.append(
                        CallSite(owner=owner, expr=_text(fn),
                                 line=child.start_point[0] + 1, enclosing_class=cls)
                    )
                walk(child, scope, owner)
            elif t == "import_statement":
                _imports(child, facts)
            elif t == "import_from_statement":
                _from_imports(child, facts, package)
            else:
                walk(child, scope, owner)

    walk(root, [], path)
    return facts


def _enclosing_class(scope: list[tuple[str, str]]) -> str | None:
    """Qualified name of the innermost class in scope, e.g. "Outer.Inner"."""
    for i in range(len(scope) - 1, -1, -1):
        if scope[i][0] == "class":
            return ".".join(name for _, name in scope[: i + 1])
    return None


def _imports(node: TSNode, facts: FileFacts) -> None:
    for name in node.children_by_field_name("name"):
        if name.type == "aliased_import":
            target = _text(name.child_by_field_name("name"))
            facts.imports.append(ImportBinding(_text(name.child_by_field_name("alias")), target, None))
        else:
            # `import a.b` binds the dotted path itself for call resolution ("a.b.f()")
            target = _text(name)
            facts.imports.append(ImportBinding(target, target, None))


def _from_imports(node: TSNode, facts: FileFacts, package: str) -> None:
    spec = _text(node.child_by_field_name("module_name"))
    module = _resolve_relative(spec, package) if spec.startswith(".") else spec
    for name in node.children_by_field_name("name"):
        if name.type == "aliased_import":
            symbol = _text(name.child_by_field_name("name"))
            local = _text(name.child_by_field_name("alias"))
        else:
            symbol = local = _text(name)
        facts.imports.append(ImportBinding(local, module, symbol))
