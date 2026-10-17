"""Language extractors: source file -> facts (definitions, call sites, imports).

Extractors only describe a single file. Cross-file resolution happens in
`code_review_graph.resolve`. Python has a dedicated extractor (its import
system is worth modelling exactly); every other language goes through the
tree-sitter-language-pack based extractor in `generic.py`.
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
    local: str            # name visible in the module, may be dotted ("os.path"); "" = file-level only
    module: str           # Python: absolute dotted module. Others: the import spec as written
    symbol: str | None    # imported symbol, or None when the binding is the module itself


@dataclass
class FileFacts:
    path: str
    module: str           # dotted module name (Python) or the path itself
    language: str = "python"
    nodes: list[Node] = field(default_factory=list)
    calls: list[CallSite] = field(default_factory=list)
    imports: list[ImportBinding] = field(default_factory=list)


def extractor_for(path: str):
    """Return `extract(path, source) -> FileFacts` for this file, or None to skip it."""
    if path.endswith((".py", ".pyi")):
        from code_review_graph.extract import python

        return python.extract
    from code_review_graph.extract import generic

    lang = generic.language_for(path)
    if lang is None:
        return None
    return lambda p, source: generic.extract(p, source, lang)
