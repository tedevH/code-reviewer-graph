# code-review-graph

**Compile your repo into a self-updating knowledge graph, then review pull requests with the blast radius in view.**

Most AI code review tools paste a diff into a model and hope. The model never sees the callers three files away, doesn't know the module had the same bug last month, and forgets everything the moment the review ends.

`code-review-graph` (`crg`) takes the other approach, inspired by Andrej Karpathy's *LLM wiki* idea: **compile knowledge once, keep it current, and query it.** It doesn't re-read raw source on every review. It builds a persistent graph of your codebase, has Claude write an interlinked wiki on top of it, and feeds each review the small subgraph a change actually touches. Reviews also write back what they learn, so review #500 knows things review #1 didn't.

```
             ┌──────────────── exact layer (tree-sitter, no LLM) ────────────────┐
  source ──▶ │ nodes: modules, classes, functions, methods                       │
             │ edges: calls · imports · contains   (+ confidence on every edge) │
             └───────────────────────────────┬───────────────────────────────────┘
                                             │ clusters (label propagation)
             ┌───────────────────────────────▼────── meaning layer (Claude) ──────┐
             │ module pages · cluster pages · [[backlinks]] from the graph        │
             │ notes written back by reviews and people                           │
             └───────────────────────────────┬───────────────────────────────────┘
                                             │
  PR diff ──▶ changed nodes ──▶ reverse-reachable callers + tests ──▶ wiki pages ──▶ review
                                                                                   │
                                          learnings written back into the graph ◀──┘
```

## Why a graph?

Code is a graph whether you treat it as one or not. "What breaks if I change `parse_price`?" is a graph question: start at the node and walk the `calls` edges backwards. Plain text search can't answer it, and a model reading a diff can't either.

| | Chat wrapper | code-review-graph |
|---|---|---|
| Context | the diff | the diff + every caller within N hops + tests that reach it |
| Memory | none | persistent graph + wiki + notes from past reviews |
| Cost per review | re-reads files | reads ~a few KB of compiled pages |
| Freshness | n/a | incremental: only modules whose hash changed get re-compiled |
| Trust | model's guess | every edge is labeled `exact`, `name` or `ambiguous` |

## Install

```bash
pip install git+https://github.com/<you>/code-review-graph
```

Requires Python 3.10+. LLM features use the Claude API: set `ANTHROPIC_API_KEY` or run `ant auth login`.

## Quickstart

```bash
cd your-repo
crg index                     # parse into the graph (seconds, free, no LLM)
crg impact --diff main        # what does my branch affect?
crg compile                   # Claude writes the wiki (asks before calling the API)
crg review --diff main --learn  # review with graph + wiki context, save learnings
```

Everything lives in `.crg/`: `graph.db` (SQLite) and `wiki/` (markdown). Open `.crg/wiki/` in Obsidian to browse the graph. Add `.crg/graph.db` to `.gitignore`. Commit `.crg/wiki/` if you want the team to share the compiled knowledge.

## Commands

| Command | What it does | LLM? |
|---|---|---|
| `crg index` | Parse the repo into nodes and edges, detect clusters, render wiki skeleton | no |
| `crg impact <symbol>…` | Callers (by hop), tests that reach the symbol, affected wiki pages | no |
| `crg impact --diff REF` | Same, starting from the functions a `git diff REF` touches | no |
| `crg impact --diff-file -` | Same, reading a unified diff from stdin (handy in CI) | no |
| `crg compile [--limit N]` | Write/refresh wiki pages for stale modules and clusters | yes |
| `crg wiki` | Re-render the wiki from the store | no |
| `crg review --diff REF [--learn]` | Review a diff with the impact subgraph + wiki as context | yes |
| `crg learn <target> "<fact>"` | Attach a durable note to a module or symbol | no |
| `crg stats` | Graph size, clusters, stale modules | no |

Symbols can be given as `path::Qual.name`, `module.Qual.name`, or a bare name.

### Example: `crg impact` on [pallets/click](https://github.com/pallets/click)

```
$ crg impact click.parser._unpack_args --depth 4

## Changed
- `src/click/parser.py:51` function `_unpack_args`

## Callers (blast radius)
### 1 hop
- `src/click/parser.py:316` method `_OptionParser._process_args_for_args` → calls `_unpack_args`
### 2 hops
- `src/click/parser.py:298` method `_OptionParser.parse_args` → calls `_OptionParser._process_args_for_args`
### 3 hops
- `src/click/core.py:1396` method `Command.parse_args` → calls `_OptionParser.parse_args` _(ambiguous, verify)_
- `src/click/core.py:2038` method `Group.parse_args` → calls `_OptionParser.parse_args` _(ambiguous, verify)_
...
## Tests that reach the change
- `tests/test_basic.py:131` function `test_return_values` → calls `Command.make_context` _(ambiguous, verify)_
```

Indexing all of click (91 modules, ~2,000 definitions, ~3,000 call edges) takes about one second.

## How it works

**Exact layer.** [tree-sitter](https://tree-sitter.github.io/) parses each file into definitions, call sites and imports. A resolver links call sites across files through local definitions, `self.`/`cls.`, `import x`, `from x import y`, relative imports and package re-exports. When static resolution fails it falls back to name matching and says so:

- `exact`: resolved through scope or imports
- `name`: exactly one definition in the repo has that name
- `ambiguous`: 2–3 candidates (receiver type unknown), each linked and flagged

Calls into the stdlib or third-party packages produce no edges.

**Clusters.** Modules are grouped by label propagation over the import and call graph, with degree-normalized weights so hub modules don't swallow everything. A cluster is a set of modules that change together. It gets its own wiki page describing the invariants that span those modules.

**Meaning layer.** `crg compile` sends each module's source plus its graph neighbourhood to Claude and stores the summary keyed by file hash. Unchanged files are never re-sent. The wiki pages mix LLM prose with deterministic structure: every `[[link]]` comes from the graph, not from the model.

**Write-back.** `crg review --learn` asks the model for durable facts ("prices arrive as `$`-prefixed strings") and attaches them to the relevant nodes. `crg learn` lets humans do the same. Notes show up in the wiki and in every future review that touches that code.

## Using it in CI

```yaml
# .github/workflows/review.yml
on: pull_request
jobs:
  review:
    runs-on: ubuntu-latest
    permissions: { pull-requests: write, contents: read }
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install git+https://github.com/<you>/code-review-graph
      - run: crg index
      - run: crg review --diff origin/${{ github.base_ref }} --effort high > review.md
        env: { ANTHROPIC_API_KEY: "${{ secrets.ANTHROPIC_API_KEY }}" }
      - run: gh pr comment ${{ github.event.number }} --body-file review.md
        env: { GH_TOKEN: "${{ github.token }}" }
```

Cache `.crg/` between runs (`actions/cache`) so `compile` only re-summarizes changed modules.

## Limitations

- **Python only** for now. The extractor interface (`code_review_graph/extract/`) is small, so adding TypeScript, Go and others is the top roadmap item.
- Static resolution can't see dynamic dispatch, monkey-patching or callbacks passed as values. Those show up as missing or `ambiguous` edges, never as false `exact` ones.
- Very large files (>200 KB) are skipped by `compile` rather than silently truncated.

## Roadmap

- [ ] TypeScript/JavaScript and Go extractors
- [ ] Git-history edges (`modified_in`, `fixed_bug`) so reviews know which code has broken before
- [ ] Eval harness: replay historical bug-introducing PRs and measure precision/recall
- [ ] GitHub App with inline comments
- [ ] MCP server so coding agents can query the graph directly

## Contributing

```bash
git clone https://github.com/<you>/code-review-graph && cd code-review-graph
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Tests run against a small fixture repo in `tests/fixtures/shop/` and never call the API.

## License

MIT
