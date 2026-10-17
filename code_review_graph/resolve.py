"""Cross-file resolution: turn per-file facts into graph edges.

Resolution is static and deliberately conservative. A call is linked:
  * "exact" when it resolves through local definitions, self/this, the
            enclosing class, imports, or same-package visibility
  * "name"  when nothing else worked but exactly one definition in the repo
            (in the same language family) has that name
  * "ambiguous" to each of 2-3 same-named definitions when the receiver's
            type is unknown; more candidates than that is noise, so no edge
Calls into code outside the repo (stdlib, packages) produce no edge.

Python imports are modelled exactly (dotted modules, packages, re-exports).
Every other language maps import specs to files with per-family rules:
relative paths for JS/TS, Ruby, Lua and C includes; `go.mod` for Go; and
path-suffix matching for the languages whose imports mirror the directory
layout (Java, Kotlin, Scala, PHP, Rust, Dart, ...).
"""

from __future__ import annotations

import builtins
import posixpath
import re
from collections import defaultdict

from code_review_graph.extract import CallSite, FileFacts, ImportBinding
from code_review_graph.graph import is_test
from code_review_graph.store import Edge

MAX_AMBIGUOUS = 3
MAX_PACKAGE_FILES = 200
SELF_NAMES = {"self", "this", "cls", "Self", "me"}
CONSTRUCTORS = ("__init__", "constructor", "initialize", "init", "new")

FAMILY = {
    "javascript": "js", "typescript": "js", "tsx": "js", "jsx": "js",
    "c": "c", "cpp": "c", "cuda": "c", "objc": "c", "arduino": "c",
    "java": "jvm", "kotlin": "jvm", "scala": "jvm", "groovy": "jvm",
    "python": "python", "cython": "python",
}
FAMILY_EXTS = {
    "js": (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".d.ts"),
    "c": (".h", ".hpp", ".hh", ".c", ".cc", ".cpp"),
}
HEADER_EXTS = (".h", ".hh", ".hpp", ".hxx")
# Languages where everything in the same directory/package is visible unqualified.
PACKAGE_SCOPED = {"go", "java", "kotlin", "scala", "csharp", "swift", "fsharp"}


def family(lang: str) -> str:
    return FAMILY.get(lang, lang)


class Resolver:
    def __init__(self, all_facts: list[FileFacts], go_module: str | None = None):
        self.facts = {f.path: f for f in all_facts}
        self.go_module = go_module
        self.module_to_path = {f.module: f.path for f in all_facts if f.language == "python"}
        self.defs: dict[str, dict[str, str]] = defaultdict(dict)       # path -> qual -> id
        self.kind: dict[str, str] = {}
        self.by_name: dict[tuple[str, str], list[str]] = defaultdict(list)  # (family, name) -> ids
        self.by_stem: dict[str, list[str]] = defaultdict(list)         # lower file stem -> paths
        self.by_dir: dict[str, list[str]] = defaultdict(list)          # lower dir path -> paths
        for f in all_facts:
            fam = family(f.language)
            for n in f.nodes:
                self.kind[n.id] = n.kind
                if n.kind != "module":
                    self.defs[f.path][n.name] = n.id
                    self.by_name[(fam, n.name.rsplit(".", 1)[-1])].append(n.id)
            stem = posixpath.basename(f.path).split(".")[0].lower()
            self.by_stem[stem].append(f.path)
            self.by_dir[posixpath.dirname(f.path).lower()].append(f.path)
        self.bindings = {
            f.path: {b.local: b for b in f.imports if b.local} for f in all_facts
        }
        self._targets: dict[tuple[str, str, str | None], tuple[list[str], str | None]] = {}
        self._visible: dict[str, list[str]] = {}

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

    def resolve_call(self, f: FileFacts, call: CallSite) -> list[tuple[str, str]]:
        if f.language == "python":
            return self._resolve_python(f, call)
        return self._resolve_generic(f, call)

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
        targets: set[str] = set()
        for b in f.imports:
            if f.language == "python":
                target = None
                if b.symbol is not None:
                    target = self.module_to_path.get(f"{b.module}.{b.symbol}")
                target = target or self.module_to_path.get(b.module)
                if target:
                    targets.add(target)
            else:
                targets.update(self.import_targets(f, b)[0])
        targets.discard(f.path)
        return [Edge(f.path, t, "imports") for t in sorted(targets)]

    # --- shared helpers -----------------------------------------------------

    def _with_constructor(self, target: str) -> list[str]:
        out = [target]
        if self.kind.get(target) == "class":
            path, _, qual = target.partition("::")
            simple = qual.rsplit(".", 1)[-1]
            for name in (*CONSTRUCTORS, simple):     # Java/C#/C++ constructors share the class name
                init = self.defs[path].get(f"{qual}.{name}")
                if init:
                    out.append(init)
        return out

    def _graded(self, ids: list[str], confidence: str) -> list[tuple[str, str]]:
        ids = list(dict.fromkeys(ids))
        if not ids or len(ids) > MAX_AMBIGUOUS:
            return []
        if len(ids) > 1 and confidence == "exact":
            confidence = "ambiguous"
        return [(t, confidence) for i in ids for t in self._with_constructor(i)]

    def _by_unique_name(self, name: str, f: FileFacts) -> list[tuple[str, str]]:
        candidates = self.by_name.get((family(f.language), name), [])
        if not is_test(f.path):
            # Production code doesn't call into test files.
            candidates = [c for c in candidates if not is_test(c.partition("::")[0])]
        if len(candidates) == 1:
            return [(candidates[0], "name")]
        return self._graded(candidates, "ambiguous")

    # --- Python ---------------------------------------------------------------

    def _resolve_python(self, f: FileFacts, call: CallSite) -> list[tuple[str, str]]:
        parts = call.expr.split(".")
        local = self.defs[f.path]

        if parts[0] in ("self", "cls") and call.enclosing_class and len(parts) == 2:
            target = local.get(f"{call.enclosing_class}.{parts[1]}")
            if target:
                return [(target, "exact")]
            return self._by_unique_name(parts[1], f)  # likely an inherited method

        if len(parts) == 1 and parts[0] in local:
            return [(t, "exact") for t in self._with_constructor(local[parts[0]])]

        binding, rest = self._longest_binding(f.path, parts)
        if binding is not None:
            # Imports from outside the repo resolve to nothing, and we don't name-guess them.
            return self._follow_python_import(binding, rest)

        if len(parts) >= 2 and self._external_type(f, parts[0]):
            return []
        if len(parts) >= 2 or not hasattr(builtins, parts[0]):
            return self._by_unique_name(parts[-1], f)
        return []

    def _longest_binding(self, path: str, parts: list[str]) -> tuple[ImportBinding | None, list[str]]:
        bindings = self.bindings[path]
        for k in range(len(parts), 0, -1):
            b = bindings.get(".".join(parts[:k]))
            if b is not None:
                return b, parts[k:]
        return None, parts

    def _follow_python_import(self, b: ImportBinding, rest: list[str], depth: int = 0) -> list[tuple[str, str]]:
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
            return [(t, "exact") for t in self._with_constructor(self.defs[path][qual])]
        # Re-export, e.g. pkg/__init__.py doing `from .orders import place_order`
        inner = self.bindings[path].get(symbol_path[0])
        if inner is not None and depth < 3:
            return self._follow_python_import(inner, symbol_path[1:], depth + 1)
        return []

    # --- every other language -------------------------------------------------

    def _resolve_generic(self, f: FileFacts, call: CallSite) -> list[tuple[str, str]]:
        parts = call.expr.split(".")
        local = self.defs[f.path]
        cls = call.enclosing_class or self._owner_class(call.owner)

        if parts[0] in SELF_NAMES and len(parts) == 2:
            if cls and f"{cls}.{parts[1]}" in local:
                return [(local[f"{cls}.{parts[1]}"], "exact")]
            return self._by_unique_name(parts[1], f)   # inherited, or defined in another file

        if len(parts) == 1:
            name = parts[0]
            if cls and f"{cls}.{name}" in local:       # sibling method called without `this`
                return [(local[f"{cls}.{name}"], "exact")]
            if name in local:
                return self._graded([local[name]], "exact")
            b = self.bindings[f.path].get(name)
            if b is not None:
                files, symbol = self.import_targets(f, b)
                if not files:
                    return []                          # imported from a package
                wanted = name if symbol in (None, "default") else symbol
                hits = self._find(files, wanted)
                return self._graded(hits, "exact") if hits else self._by_unique_name(wanted, f)
            hits = self._find(self._visible_files(f), name, top_level=True)
            if hits:
                return self._graded(hits, "exact")
            return self._by_unique_name(name, f)

        b, rest = self._longest_binding(f.path, parts)
        if b is not None:
            files, symbol = self.import_targets(f, b)
            if not files:
                return []
            qual = ".".join(([symbol] if symbol not in (None, "default") else []) + rest)
            hits = self._find(files, qual) if qual else []
            return self._graded(hits, "exact") if hits else self._by_unique_name(parts[-1], f)

        # `Type.method()` / `Namespace.fn()` where Type is defined here or visible here.
        hits = self._find([f.path, *self._visible_files(f)], ".".join(parts))
        if hits:
            return self._graded(hits, "exact")
        if self._external_type(f, parts[0]):
            return []
        return self._by_unique_name(parts[-1], f)

    def _external_type(self, f: FileFacts, head: str) -> bool:
        """`Object.create`, `JSON.parse`, `String.format`: a capitalized receiver that no
        file in the repo defines is a stdlib/third-party type, so don't name-guess."""
        return head[:1].isupper() and not self.by_name.get((family(f.language), head))

    def _owner_class(self, owner: str) -> str | None:
        if self.kind.get(owner) == "method":
            qual = owner.partition("::")[2]
            return qual.rpartition(".")[0] or None
        return None

    def _find(self, files: list[str], qual: str, top_level: bool = False) -> list[str]:
        """Definitions named `qual` in these files: exact qualified name, else a suffix match."""
        exact, suffix = [], []
        for path in self._with_c_sources(files):
            defs = self.defs.get(path, {})
            if qual in defs:
                exact.append(defs[qual])
            elif not top_level:
                suffix += [i for q, i in defs.items() if q.endswith("." + qual)]
        hits = exact or suffix
        # C/C++: a header only declares; prefer the definition in a source file.
        in_source = [h for h in hits if not h.partition("::")[0].endswith(HEADER_EXTS)]
        return in_source or hits

    def _with_c_sources(self, files: list[str]) -> list[str]:
        """Add `util.c` next to an included `util.h`, where the definitions live."""
        out = list(files)
        for path in files:
            if path.endswith(HEADER_EXTS):
                stem = posixpath.splitext(path)[0]
                out += [stem + e for e in (".c", ".cc", ".cpp", ".cxx", ".m", ".cu") if stem + e in self.facts]
        return list(dict.fromkeys(out))

    def _visible_files(self, f: FileFacts) -> list[str]:
        if f.path in self._visible:
            return self._visible[f.path]
        out: list[str] = []
        for b in f.imports:
            out += self.import_targets(f, b)[0]
        if f.language in PACKAGE_SCOPED:
            here = posixpath.dirname(f.path).lower()
            siblings = [p for p in self.by_dir.get(here, []) if self.facts[p].language == f.language]
            out += siblings[:MAX_PACKAGE_FILES]
        self._visible[f.path] = [p for p in dict.fromkeys(out) if p != f.path]
        return self._visible[f.path]

    def import_targets(self, f: FileFacts, b: ImportBinding) -> tuple[list[str], str | None]:
        """Repo files an import refers to, plus the symbol it names inside them (if any)."""
        key = (f.path if b.module.startswith(".") else family(f.language), b.module, b.symbol)
        if key in self._targets:
            return self._targets[key]
        files = self._spec_files(f, b.module)
        symbol = b.symbol
        if not files and b.symbol is None and family(f.language) not in ("js", "c", "go"):
            # `use a::b::func`, `import static a.B.method`: the last segment is a symbol
            parent, _, last = re.sub(r"::|\\", ".", b.module).rpartition(".")
            if parent:
                files = self._spec_files(f, parent)
                symbol = last if files else None
        result = (files[:MAX_PACKAGE_FILES], symbol)
        self._targets[key] = result
        return result

    def _spec_files(self, f: FileFacts, spec: str) -> list[str]:
        fam = family(f.language)
        if spec.startswith((".", "/")) or fam == "c":   # relative, or a C include (relative first)
            base = posixpath.normpath(posixpath.join(posixpath.dirname(f.path), spec))
            hits = self._with_exts(base, fam, f.path)
            if hits or spec.startswith("."):
                return hits

        if fam == "js":
            if spec.startswith(("@/", "~/")):
                return self._suffix_match(spec[2:], fam, f.path, as_path=True)
            return []                                    # a package from node_modules
        if fam == "go":
            mod = self.go_module
            if mod and (spec == mod or spec.startswith(mod + "/")):
                d = spec[len(mod):].strip("/").lower()
                return [p for p in self.by_dir.get(d, []) if p.endswith(".go") and not p.endswith("_test.go")]
            return []                                    # stdlib or third-party
        if "/" in spec or re.search(r"\.\w{1,4}$", spec) and fam in ("c", "dart", "ruby", "lua"):
            return self._suffix_match(spec.split(":", 1)[-1], fam, f.path, as_path=True)
        return self._suffix_match(spec, fam, f.path, as_path=False)

    def _with_exts(self, base: str, fam: str, from_path: str) -> list[str]:
        exts = FAMILY_EXTS.get(fam) or (posixpath.splitext(from_path)[1],)
        bases = [base]
        if fam == "js" and re.search(r"\.[mc]?jsx?$", base):
            bases.append(re.sub(r"\.[mc]?jsx?$", "", base))   # ESM: "./x.js" written for x.ts
        candidates = [c for b in bases for c in (b, *(b + e for e in exts),
                      *(f"{b}/index{e}" for e in exts), *(f"{b}/mod{e}" for e in exts))]
        return [c for c in dict.fromkeys(candidates) if c in self.facts]

    def _suffix_match(self, spec: str, fam: str, from_path: str, as_path: bool) -> list[str]:
        """Files whose path ends with the spec's segments: `com.x.Foo` -> `.../com/x/Foo.java`."""
        spec = spec.strip("/")
        if as_path:
            stem = re.sub(r"\.\w{1,5}$", "", spec)
            segs = [s for s in stem.split("/") if s not in ("", ".", "..")]
        else:
            segs = [s for s in re.split(r"::|\.|\\|/", spec)
                    if s and s not in ("crate", "self", "super", "*", "_")]
        if not segs:
            return []
        candidates = [p for p in self.by_stem.get(segs[-1].lower(), [])
                      if family(self.facts[p].language) == fam]
        files: list[str] = []
        # Longest matching suffix wins. Leading segments may not exist on disk
        # (PSR-4 maps `App\` to `src/`, Java has `src/main/java/` in front).
        for k in range(len(segs), 0, -1):
            tail = "/".join(segs[-k:]).lower()
            files = [p for p in candidates
                     if ("/" + posixpath.splitext(p)[0].lower()).endswith("/" + tail)]
            if files or as_path:
                break
        if not files:
            tail = "/".join(segs).lower()
            # A package/namespace directory: `import com.x.*`, `use crate::util`
            files = [p for d, ps in self.by_dir.items()
                     if ("/" + d).endswith("/" + tail) for p in ps
                     if family(self.facts[p].language) == fam]
        return sorted(files)
