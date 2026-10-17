"""Language extractors: source file -> facts (definitions, call sites, imports).

Extractors only describe a single file. Cross-file resolution happens in
`code_review_graph.resolve`, so adding a language means writing one extractor that
returns `FileFacts` and registering its file extensions below.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from code_review_graph.store import Node


@dataclass(frozen=True)
class CallSite:
    owner: str            # node id of the enclosing function/method/module
    expr: str             # callee expression as written, e.g. "self.save" or "utils.parse"
    line: int
    enclosing_class: str | None  # qualified class name, for resolving self./cls.


@dataclass(frozen=True)
class ImportBinding:
    local: str            # name visible in the module, may be dotted ("os.path")
    module: str           # absolute dotted module name
    symbol: str | None    # imported symbol, or None when the binding is the module itself


@dataclass
class FileFacts:
    path: str
    module: str           # dotted module name derived from the path
    nodes: list[Node] = field(default_factory=list)
    calls: list[CallSite] = field(default_factory=list)
    imports: list[ImportBinding] = field(default_factory=list)


def extractor_for(path: str):
    if path.endswith(".py"):
        from code_review_graph.extract import python

        return python.extract
    return None
