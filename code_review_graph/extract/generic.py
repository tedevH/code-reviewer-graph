"""Extractor for every other language tree-sitter can parse.

Grammars come from tree-sitter-language-pack (300+ languages, downloaded on
first use and cached). Three sources of facts, from most to least precise:

  * definitions: the grammar's own `tags.scm` query when it ships one (the
    same definition tagging GitHub code navigation uses; ~70 languages),
    otherwise node-type heuristics (`function_definition`, `class_declaration`,
    `fn_item`, ...), which work across nearly all grammars
  * calls: one walker that understands the common call shapes (`f()`,
    `obj.m()`, `pkg::f()`, `obj->m()`, `new A()`, `$this->m()`, ...)
  * imports: per-family rules for JS/TS, Go, C/C++ includes, Ruby/Lua
    `require`, and a generic rule for `import`/`use`/`using` statements
"""

from __future__ import annotations

import functools
import re

import tree_sitter_language_pack as pack
from tree_sitter import Node as TSNode, Query, QueryCursor

from code_review_graph.extract import CallSite, FileFacts, ImportBinding
from code_review_graph.store import Node

# Grammars for data, markup and config: nothing to call, so skip them entirely.
NON_CODE = {
    "json", "jsonc", "json5", "yaml", "toml", "xml", "html", "css", "scss", "less", "sass",
    "markdown", "markdown_inline", "csv", "tsv", "psv", "ini", "properties", "dockerfile",
    "gitignore", "gitattributes", "gitcommit", "git_config", "git_rebase", "diff", "regex",
    "comment", "jsdoc", "requirements", "rst", "latex", "bibtex", "dtd", "xcompose",
    "ssh_config", "udev", "nginx", "hcl", "terraform", "make", "cmake", "ninja", "sql",
    "graphql", "proto", "textproto", "vue", "svelte", "astro", "angular", "embedded_template",
    "twig", "jinja", "jinja_inline", "htmldjango", "liquid", "po", "pem", "printf", "query",
    "beancount", "csv", "kdl", "ron", "editorconfig", "dotenv", "http", "mermaid", "dot",
}
JS_FAMILY = {"javascript", "typescript", "tsx", "jsx"}

# typescript/tsx tags.scm only cover TS additions; the JS query covers the rest.
EXTRA_TAGS = {"typescript": ["javascript"], "tsx": ["javascript"], "cpp": ["c"], "cuda": ["cpp", "c"],
              "arduino": ["cpp", "c"], "objc": ["c"]}

CLASS_KINDS = {"class", "interface", "struct", "enum", "trait", "module", "object",
               "protocol", "union", "type", "namespace", "record", "actor", "contract"}
FUNCTION_KINDS = {"function", "method", "constructor", "procedure", "subroutine"}

_HEURISTIC_DEF = re.compile(
    r"^(?:\w+_)?(?:(?P<cls>class|struct|interface|trait|enum|protocol|object|module|impl|"
    r"namespace|record|union)|(?P<fn>function|method|func|fn|procedure|subroutine|"
    r"constructor|def|defun|sub))(?:_(?:definition|declaration|item|statement|specifier|"
    r"def|decl|declarator))?$"
)
_NOT_DEF = re.compile(r"(?:^|_)(?:call|expression|type|parameter|parameters|body|name|signature|"
                      r"literal|modifier|keyword|list|clause|argument|reference|pointer|header|"
                      r"prototype)(?:_|$)")
_CALL = re.compile(r"(?:^|_)(?:call|invocation)(?:_expression)?$")
_NEW = {"new_expression", "object_creation_expression", "instance_creation_expression"}
_IMPORT = re.compile(r"import|include|use_declaration|using_directive|namespace_use_clause")
_IDENT = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
_SELF = {"self", "this", "cls", "Self", "me"}


class UnsupportedLanguage(RuntimeError):
    pass


def language_for(path: str) -> str | None:
    if path.endswith((".min.js", ".min.css", ".bundle.js")):
        return None
    lang = pack.detect_language_from_path(path)
    return None if not lang or lang in NON_CODE else lang


@functools.cache
def _parser(lang: str):
    try:
        return pack.get_parser(lang)
    except Exception as e:  # network down, grammar missing, ABI mismatch
        raise UnsupportedLanguage(f"{lang}: {e}") from e


@functools.cache
def _tag_queries(lang: str) -> tuple[Query, ...]:
    language = pack.get_language(lang)
    out = []
    for source_lang in [*EXTRA_TAGS.get(lang, []), lang]:
        try:
            text = pack.get_tags_query(source_lang)
            if text:
                out.append(Query(language, text))
        except Exception:
            continue  # a query that doesn't compile against this grammar version
    return tuple(out)


def _text(node: TSNode | None) -> str:
    return node.text.decode("utf-8", errors="replace") if node is not None else ""


def _walk(root: TSNode):
    """Pre-order traversal without recursion (deep ASTs blow the Python stack)."""
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.named_children))


# --- definitions ------------------------------------------------------------

def _has_body(node: TSNode) -> bool:
    if node.child_by_field_name("body") is not None:
        return True
    return any(re.search(r"block|body|compound", c.type, re.I) for c in node.named_children)


def _full_span(node: TSNode) -> TSNode:
    """Some grammars tag only a signature: C's `function_declarator`, Zig's `FnProto`.
    Widen to the enclosing node that also holds the body, so calls inside it get an owner."""
    cur = node
    for _ in range(3):
        if _has_body(cur) or cur.parent is None:
            break
        cur = cur.parent
    return cur if _has_body(cur) else node


class _Def:
    __slots__ = ("start", "end", "kind", "name", "node", "is_scope_only")

    def __init__(self, node: TSNode, kind: str, name: str, is_scope_only: bool = False):
        node = _full_span(node)
        self.start, self.end = node.start_byte, node.end_byte
        self.node, self.kind, self.name, self.is_scope_only = node, kind, name, is_scope_only

    def contains(self, other: "_Def") -> bool:
        return self.start <= other.start and other.end <= self.end and (self.start, self.end) != (other.start, other.end)


def _defs_from_tags(lang: str, root: TSNode) -> list[_Def]:
    found: dict[tuple[int, int], _Def] = {}
    rank = {"class": 0, "method": 1, "function": 2}
    for query in _tag_queries(lang):
        for _, caps in QueryCursor(query).matches(root):
            names = caps.get("name")
            if not names:
                continue
            name = _text(names[0])
            if names[0].parent is not None and names[0].parent.type == "qualified_identifier":
                # C++ out-of-class definition `void ns::K::m()`: keep the owning class
                outer = names[0].parent
                while outer.parent is not None and outer.parent.type == "qualified_identifier":
                    outer = outer.parent
                name = ".".join((normalize_callee(_text(outer)) or name).split(".")[-2:])
            for cap, nodes in caps.items():
                kind = None
                if cap.startswith("definition."):
                    sub = cap.split(".", 1)[1]
                    kind = "class" if sub in CLASS_KINDS else "method" if sub == "method" \
                        else "function" if sub in FUNCTION_KINDS else None
                    scope_only = False
                elif cap == "reference.implementation" and nodes[0].type.startswith("impl"):
                    kind, scope_only = "class", True   # Rust `impl S { ... }` scopes methods to S
                if kind is None:
                    continue
                if kind == "function" and "." in name:
                    kind = "method"
                d = _Def(nodes[0], kind, name, scope_only)
                key = (d.start, d.end)
                if key not in found or rank[kind] < rank[found[key].kind]:
                    found[key] = d
    return list(found.values())


def _heuristic_name(node: TSNode) -> str:
    named = node.child_by_field_name("name")
    if named is not None:
        return _text(named)
    for field in ("declarator", "type"):
        inner = node.child_by_field_name(field)
        depth = 0
        while inner is not None and depth < 4:
            if "identifier" in inner.type.lower() or inner.type in ("name", "constant", "word"):
                return _text(inner)
            inner = inner.child_by_field_name("declarator") or inner.child_by_field_name("name") \
                or (inner.named_children[0] if inner.named_children else None)
            depth += 1
    for child in node.named_children:
        if "identifier" in child.type.lower() or child.type in ("name", "constant", "word", "symbol"):
            return _text(child)
    return ""


def _snake(node_type: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", node_type).lower()


def _defs_from_heuristics(root: TSNode) -> list[_Def]:
    out = []
    for node in _walk(root):
        t = _snake(node.type)
        if t.endswith("_proto"):   # Zig `FnProto`: a signature whose body is a sibling
            t = t[: -len("_proto")]
        m = _HEURISTIC_DEF.match(t)
        if not m or _NOT_DEF.search(t):
            continue
        name = _heuristic_name(node).replace("::", ".")
        if not name or not _IDENT.fullmatch(name):
            continue
        kind = "class" if m.group("cls") else "method" if "." in name else "function"
        out.append(_Def(node, kind, name, node.type.startswith("impl")))
    return out


def _go_receiver(node: TSNode) -> tuple[str, str] | None:
    """Go `func (t *T) M()` -> ("t", "T"): the receiver variable acts like `self`."""
    recv = node.child_by_field_name("receiver")
    if recv is None:
        return None
    types = [n for n in _walk(recv) if n.type == "type_identifier"]
    names = [n for n in _walk(recv) if n.type == "identifier"]
    return (_text(names[0]) if names else "", _text(types[-1])) if types else None


# --- calls ------------------------------------------------------------------

def _callee(node: TSNode) -> str | None:
    fn = node.child_by_field_name("function") or node.child_by_field_name("constructor")
    if fn is None:
        name = node.child_by_field_name("name") or node.child_by_field_name("method")
        if name is not None:
            recv = (node.child_by_field_name("object") or node.child_by_field_name("receiver")
                    or node.child_by_field_name("scope"))
            return f"{_text(recv)}.{_text(name)}" if recv is not None else _text(name)
        fn = node.child_by_field_name("type") or (node.named_children[0] if node.named_children else None)
    return _text(fn) if fn is not None else None


def normalize_callee(expr: str) -> str | None:
    """`a::b::<T>`, `$obj->m`, `x?.y` -> dotted form; chained calls keep the last name."""
    prev = None
    while prev != expr:
        prev, expr = expr, re.sub(r"<[^<>]*>", "", expr)
    expr = re.sub(r"\s+", "", expr)
    for sep in ("::", "->", "?.", "!.", "\\"):
        expr = expr.replace(sep, ".")
    expr = re.sub(r"\.{2,}", ".", expr.replace("$", "").replace("@", "")).lstrip(".&*")
    if _IDENT.fullmatch(expr):
        return expr
    m = re.search(r"([A-Za-z_]\w*)\W*$", expr)
    return m.group(1) if m else None


# --- imports ----------------------------------------------------------------

def _string(node: TSNode) -> str | None:
    for n in _walk(node):
        if "string" in n.type and n.type not in ("string_content", "string_fragment") or n.type == "system_lib_string":
            return _text(n).strip("\"'`<>")
    return None


def _js_import(node: TSNode, out: list[ImportBinding]) -> None:
    spec = _string(node.child_by_field_name("source") or node)
    if not spec:
        return
    clause = next((c for c in node.named_children if c.type == "import_clause"), None)
    if clause is None:
        out.append(ImportBinding("", spec, None))
        return
    for c in clause.named_children:
        if c.type == "identifier":
            out.append(ImportBinding(_text(c), spec, "default"))
        elif c.type == "namespace_import":
            ident = next((n for n in c.named_children if n.type == "identifier"), None)
            out.append(ImportBinding(_text(ident), spec, None))
        elif c.type == "named_imports":
            for s in c.named_children:
                if s.type == "import_specifier":
                    name = _text(s.child_by_field_name("name"))
                    alias = s.child_by_field_name("alias")
                    out.append(ImportBinding(_text(alias) if alias else name, spec, name))


def _require(node: TSNode, lang: str) -> ImportBinding | None:
    """`require("x")` (JS, Lua), `require_relative "x"` (Ruby)."""
    callee = _callee(node) or ""
    if callee not in ("require", "require_relative", "load", "import"):
        return None
    spec = _string(node)
    if not spec:
        return None
    if callee == "require_relative" and not spec.startswith("."):
        spec = "./" + spec
    local = ""
    parent = node.parent
    if parent is not None and parent.type == "variable_declarator":
        name = parent.child_by_field_name("name")
        if name is not None and name.type == "identifier":
            local = _text(name)
    return ImportBinding(local, spec, None)


_KEYWORDS = re.compile(r"^(?:import|use|using|static|from|package|pub(?:\([^)]*\))?|namespace|"
                       r"include|require|open|export|extern|crate\s+|mod)\b\s*")


def _generic_import(node: TSNode, out: list[ImportBinding]) -> None:
    if node.type == "import_spec" or node.type == "preproc_include":   # Go, C/C++
        spec = _string(node.child_by_field_name("path") or node)
        if spec:
            alias = node.child_by_field_name("name")
            local = _text(alias) if alias is not None else (spec.rsplit("/", 1)[-1] if node.type == "import_spec" else "")
            out.append(ImportBinding("" if local in ("_", ".") else local, spec, None))
        return
    specs = [c for c in _walk(node) if c.type == "import_spec"]
    if specs:   # Go: import ( "a"; b "c" )
        for spec_node in specs:
            _generic_import(spec_node, out)
        return
    spec = _string(node)
    if spec is not None:
        out.append(ImportBinding("", spec, None))
        return
    text = _text(node).strip().rstrip(";").strip()
    prev = None
    while prev != text:
        prev, text = text, _KEYWORDS.sub("", text)
    text = re.sub(r"\s+", " ", text)
    alias = ""
    m = re.match(r"^(\w+)\s*=\s*(.+)$", text)               # C#: using X = A.B;
    if m:
        alias, text = m.group(1), m.group(2)
    m = re.match(r"^(.*?)(?:::|\.|\\)\{(.*)\}$", text)       # Rust/Scala/PHP lists: a::{b, c as d}
    items = [(m.group(1), i.strip()) for i in m.group(2).split(",")] if m else [(None, text)]
    for prefix, item in items:
        parts = re.split(r"\s+as\s+|\s*=>\s*", item)
        path = parts[0].strip()
        local = parts[1].strip() if len(parts) > 1 else alias
        full = f"{prefix}::{path}" if prefix is not None else path
        full = full.replace("\\", "::").removesuffix(".*").removesuffix("::*").removesuffix("._")
        if not full or full in ("self", "*"):
            continue
        last = re.split(r"::|\.", full)[-1]
        out.append(ImportBinding(local or (last if last not in ("*", "self") else ""), full, None))


# --- entry point ------------------------------------------------------------

def extract(path: str, source: bytes, lang: str) -> FileFacts:
    tree = _parser(lang).parse(source)
    root = tree.root_node
    facts = FileFacts(path=path, module=path, language=lang)
    facts.nodes.append(Node(id=path, kind="module", name=path, path=path,
                            start_line=1, end_line=root.end_point[0] + 1))

    # Tags queries are precise but often incomplete; heuristics fill the gaps.
    defs = _defs_from_tags(lang, root) if _tag_queries(lang) else []
    spans = {(d.start, d.end) for d in defs}
    defs += [d for d in _defs_from_heuristics(root) if (d.start, d.end) not in spans]
    defs.sort(key=lambda d: (d.start, -d.end))

    # Qualified names from containment: a def's enclosing defs (and impl scopes).
    qual_of: dict[int, str] = {}
    def_ids: list[tuple[_Def, str]] = []
    stack: list[_Def] = []
    seen_ids: set[str] = set()
    receivers: dict[int, str] = {}
    for d in defs:
        while stack and not stack[-1].contains(d):
            stack.pop()
        parent = stack[-1] if stack else None
        kind = d.kind
        prefix = qual_of[id(parent)] if parent is not None else ""
        if kind == "function" and parent is not None and parent.kind == "class":
            kind = "method"
        if kind == "method" and parent is None and (recv := _go_receiver(d.node)):
            prefix = recv[1]
            if recv[0]:
                receivers[id(d)] = recv[0]
        qual = f"{prefix}.{d.name}" if prefix else d.name
        qual_of[id(d)] = qual
        stack.append(d)
        if d.is_scope_only:
            continue
        node_id = f"{path}::{qual}"
        if node_id in seen_ids:   # overloads, or a Rust struct and its impl block
            continue
        seen_ids.add(node_id)
        d.kind = kind
        def_ids.append((d, node_id))
        facts.nodes.append(Node(id=node_id, kind=kind, name=qual, path=path,
                                start_line=d.node.start_point[0] + 1, end_line=d.node.end_point[0] + 1))

    callables = [(d, nid) for d, nid in def_ids if d.kind in ("function", "method")]
    classes = [(d, qual_of[id(d)]) for d in defs if d.kind == "class"]

    for node in _walk(root):
        t = node.type
        if _IMPORT.search(t) and not (node.parent is not None and _IMPORT.search(node.parent.type)):
            if lang in JS_FAMILY and t == "import_statement":
                _js_import(node, facts.imports)
            else:
                _generic_import(node, facts.imports)
            continue
        zig_args = next((c for c in node.named_children if c.type == "FnCallArguments"), None)
        if zig_args is not None:   # Zig: SuffixExpr(callee..., FnCallArguments)
            raw = _text(node)[: zig_args.start_byte - node.start_byte]
        elif not (_CALL.search(t) or t in _NEW):
            continue
        else:
            raw = None
        req = None if zig_args is not None else _require(node, lang)
        if req is not None:
            facts.imports.append(req)
            continue
        raw = raw or _callee(node)
        expr = normalize_callee(raw) if raw else None
        if not expr:
            continue
        start = node.start_byte
        owner_def, owner = min(
            (pair for pair in callables if pair[0].start <= start < pair[0].end),
            key=lambda pair: pair[0].end - pair[0].start, default=(None, path),
        )
        recv = receivers.get(id(owner_def))
        if recv and expr.startswith(recv + "."):
            expr = "self" + expr[len(recv):]
        enclosing = min(
            ((d, q) for d, q in classes if d.start <= start < d.end),
            key=lambda pair: pair[0].end - pair[0].start, default=(None, None),
        )[1]
        facts.calls.append(CallSite(owner=owner, expr=expr, line=node.start_point[0] + 1,
                                    enclosing_class=enclosing))
    return facts
