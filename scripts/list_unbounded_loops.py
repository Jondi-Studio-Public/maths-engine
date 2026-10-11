"""List the unbounded `while` loops in `mathgen/generators/`, and keep their allowlist current.

A rejection-sampling loop with no attempt cap can spin forever on an unlucky seed or odd
params. New generators use `mathgen.core.draw_until` (or `core.attempts`), which raise
`VerificationError` once the budget runs out. The loops that predate the helper are recorded
in `tests/unbounded_loops_allowlist.yaml`; `tests/test_no_unbounded_loops.py` fails on any
loop that is neither bounded nor listed there, and on any listed loop that is gone, so the
list only shrinks.

What counts as a finding (a static scan; it reads source, runs nothing):

- `while_true`: `while True:` (or any constant-true test) with no visible bound.
- `rng_redraw`: `while <cond>:` whose test or body uses `rng` (calls a method on it or passes it to
  a helper) with no visible bound. Rejection sampling written as `while b == a: b = rng.randint(..)`.

A visible bound is a counter: a name incremented or decremented in the body and compared in an `if`
whose branch breaks, returns or raises (`tries += 1; if tries >= 10: break`), or, for a conditional
loop, a name that is both in the test and stepped in the body (`while n < limit: n += 1`).

Entries are keyed by (module, function, index), the index counting that function's findings in
source order, so edits elsewhere in the file do not move them. A finding that is not a
rejection loop at all (gcd-style, deterministic) is recorded with kind `algorithmic` and a reason,
set in ALGORITHMIC below.

Usage:
    python scripts/list_unbounded_loops.py            # print the findings
    python scripts/list_unbounded_loops.py --write    # rewrite tests/unbounded_loops_allowlist.yaml
    python scripts/list_unbounded_loops.py --check    # exit 1 if the allowlist differs from the tree

Converting a loop to `draw_until` shrinks the list: run `--write` and commit the smaller file.
`--write` keeps the reason of every `algorithmic` entry it already knows and refuses to add one.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATORS = REPO_ROOT / "mathgen" / "generators"
ALLOWLIST = REPO_ROOT / "tests" / "unbounded_loops_allowlist.yaml"

KINDS = ("while_true", "rng_redraw", "algorithmic")

# (module, function) -> why the loop terminates without a cap. Deterministic loops only.
ALGORITHMIC = {
    ("p6b_mst_flow_build", "_max_flow_edmonds_karp"): "Edmonds-Karp: each pass finds an augmenting path and "
    "the flow is bounded by the capacities",
    ("pv0a_geometric_threshold_build", "_first_crossing"): "monotone geometric sequence passes the threshold "
    "in finitely many steps (no rng)",
    ("pv0a_geometric_threshold_build", "_first_crossing_two_stage"): "monotone two-stage sequence passes the "
    "threshold in finitely many steps (no rng)",
    ("pv0a_geometric_threshold_build", "_first_crossing_linear"): "monotone linear sequence passes the "
    "threshold in finitely many steps (no rng)",
}


def _own_nodes(nodes):
    """Walk statements without entering nested function or class definitions."""
    stack = list(nodes)
    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                stack.append(child)


def _names(node) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _uses_rng(nodes) -> bool:
    for node in nodes:
        for n in _own_nodes([node]):
            if isinstance(n, ast.Call):
                f = n.func
                while isinstance(f, ast.Attribute):
                    f = f.value
                if isinstance(f, ast.Name) and "rng" in f.id.lower():
                    return True
                for arg in [*n.args, *(k.value for k in n.keywords)]:
                    if isinstance(arg, ast.Name) and "rng" in arg.id.lower():
                        return True
    return False


def _stepped(body) -> set[str]:
    """Names incremented or decremented by an augmented assignment in the loop body."""
    return {
        n.target.id
        for n in _own_nodes(body)
        if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name) and isinstance(n.op, (ast.Add, ast.Sub))
    }


def _exits(stmts) -> bool:
    return any(isinstance(n, (ast.Break, ast.Return, ast.Raise)) for n in _own_nodes(stmts))


def is_bounded(loop: ast.While) -> bool:
    """True when the loop has a visible counter (see the module docstring)."""
    counters = _stepped(loop.body)
    if not counters:
        return False
    if not (isinstance(loop.test, ast.Constant) and loop.test.value):
        if counters & _names(loop.test):
            return True
    for n in _own_nodes(loop.body):
        if isinstance(n, ast.If):
            compared = {name for c in ast.walk(n.test) if isinstance(c, ast.Compare) for name in _names(c)}
            if counters & compared and (_exits(n.body) or _exits(n.orelse)):
                return True
    return False


def _is_constant_true(test) -> bool:
    return isinstance(test, ast.Constant) and bool(test.value)


class _Scanner(ast.NodeVisitor):
    def __init__(self):
        self.scope: list[str] = []
        self.found: list[tuple[str, str]] = []  # (function, kind)

    def _scoped(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scoped

    def visit_While(self, node: ast.While):
        if not is_bounded(node):
            if _is_constant_true(node.test):
                self.found.append((".".join(self.scope) or "<module>", "while_true"))
            elif _uses_rng([node.test, *node.body]):
                self.found.append((".".join(self.scope) or "<module>", "rng_redraw"))
        self.generic_visit(node)


def scan_source(source: str) -> list[tuple[str, int, str]]:
    """(function, index, kind) for each unbounded loop in `source`, in source order."""
    scanner = _Scanner()
    scanner.visit(ast.parse(source))
    seen: dict[str, int] = {}
    out = []
    for function, kind in scanner.found:
        index = seen.get(function, 0)
        seen[function] = index + 1
        out.append((function, index, kind))
    return out


def scan_tree(root: Path = GENERATORS) -> list[dict]:
    """Every finding under `root`, sorted by (module, function, index). `algorithmic` is applied here."""
    rows = []
    for path in sorted(root.rglob("*.py")):
        module = path.relative_to(root).with_suffix("").as_posix().replace("/", ".")
        for function, index, kind in scan_source(path.read_text(encoding="utf-8")):
            row = {"module": module, "function": function, "index": index, "kind": kind}
            if kind == "while_true" and (module, function) in ALGORITHMIC:
                row["kind"] = "algorithmic"
                row["reason"] = ALGORITHMIC[(module, function)]
            rows.append(row)
    return sorted(rows, key=lambda r: (r["module"], r["function"], r["index"]))


HEADER = """\
# Generated by scripts/list_unbounded_loops.py --write; do not hand-edit.
# The unbounded `while` loops in mathgen/generators/ that predate core.draw_until.
# tests/test_no_unbounded_loops.py fails on a loop missing from this list and on an entry whose loop
# is gone, so the list only shrinks. Convert a loop to core.draw_until, then rerun --write.
# kind: while_true | rng_redraw (rejection sampling, no cap) | algorithmic (deterministic, with a reason)
"""


def render(rows: list[dict]) -> str:
    lines = [HEADER, "loops:"]
    for r in rows:
        item = f"  - {{module: {r['module']}, function: {r['function']}, index: {r['index']}, kind: {r['kind']}"
        if r.get("reason"):
            item += ", reason: " + _quote(r["reason"])
        lines.append(item + "}")
    return "\n".join(lines) + "\n"


def _quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def load_allowlist(path: Path = ALLOWLIST) -> list[dict]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(data.get("loops") or [])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="rewrite the allowlist from the tree")
    mode.add_argument("--check", action="store_true", help="exit 1 if the allowlist differs from the tree")
    args = parser.parse_args(argv)
    rows = scan_tree()
    text = render(rows)
    if args.write:
        ALLOWLIST.write_text(text, encoding="utf-8")
        counts = {k: sum(r["kind"] == k for r in rows) for k in KINDS}
        print(f"wrote {ALLOWLIST.relative_to(REPO_ROOT)}: {len(rows)} loops {counts}")
        return 0
    if args.check:
        current = ALLOWLIST.read_text(encoding="utf-8") if ALLOWLIST.exists() else ""
        if current != text:
            print("allowlist is out of date: run `python scripts/list_unbounded_loops.py --write`")
            return 1
        print("allowlist is current")
        return 0
    for r in rows:
        print(f"{r['module']}:{r['function']}[{r['index']}] {r['kind']}")
    print(f"{len(rows)} unbounded loops")
    return 0


if __name__ == "__main__":
    sys.exit(main())
