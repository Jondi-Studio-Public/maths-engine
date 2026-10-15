"""Find the generic checks copied into per-generator test files that `tests/test_generator_contract.py` now covers.

Hundreds of `tests/test_*.py` files repeat the same few checks for one generator each: `test_deterministic`,
`test_answer_space`, `test_no_text_defects`, `test_prompt_variety_and_new_shapes`, `test_source_wired`,
`test_unknown_tier_is_rejected` (and `test_shape`, which is mostly topic-specific). The shared contract test checks
them once for the whole library, so the copies can go. This script says which can, and which must stay.

    python3 scripts/prune_generic_tests.py                       # dry run: print the report
    python3 scripts/prune_generic_tests.py --report out.txt      # ... and save it
    python3 scripts/prune_generic_tests.py --write               # delete the removable functions and empty files
    python3 scripts/prune_generic_tests.py tests/test_foo.py ... # only these files

A function is REMOVABLE only when every statement in it is a recognised generic check, the (topic, tier) it
exercises is one the shared test covers, and the shared threshold is not weaker than the copy's (seeds,
distinct answers, share of the commonest answer, prompt shapes, text-defect seeds). Anything else is KEPT, with
the reason: a stricter threshold than the shared one, a topic-specific assertion, or a target the script could not
resolve (it never guesses). What counts as covered is read from `tests/test_generator_contract.py` and the library
(`mathgen.skills`), so re-running after the shared test changes gives the new answer.

A file is DELETABLE when no test is left in it after the removals and no other test module imports from it.
`--write` removes the functions (with their decorators) and deletes those files; imports left unused are harmless
(ruff's F401 is ignored in pyproject.toml). It never touches a file with a function it cannot classify, and it
never edits anything without `--write`.
"""

from __future__ import annotations

import argparse
import ast
import collections
import inspect
import re
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TESTS = REPO / "tests"

FAMILY_PATTERNS = {
    "deterministic": r"test_deterministic",
    "answer_space": r"test_answer_space",
    "no_text_defects": r"test_no_text_defects",
    "shape": r"test_shapes?(_|$)",
    "source_wired": r"test_source_wired",
    "prompt_variety": r"test_prompt_variety",
    "unknown_tier": r"test_unknown_tier",
}


# ----------------------------------------------------------------------------------------------------------------
# What the rest of the library covers (read at run time)
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class Library:
    """Facts about the library and the shared test, loaded once."""

    shared: object
    registry: dict
    presets: dict
    exam_topics: set
    exam_source_of_skill: dict  # skill id -> exam topic
    exam_skill_of_topic: dict  # topic -> [skill id]
    escalation_topics: set  # exam topics test_tier_escalation covers (exam source of a qualifying skill)
    live_sources: set  # (topic, difficulty, frozenset(params items)) every live skill source, skill contract's reach
    tier_topics: set  # generators with a `tier` parameter
    tier_rejects: set  # ... that the shared test holds to raising ValueError

    @classmethod
    def load(cls):
        from mathgen.core import PRESETS, REGISTRY
        from mathgen.skills import load_skills
        from tests import generator_contract_cases as shared

        _domains, skills = load_skills()
        exam_source_of_skill, exam_skill_of_topic, live_sources, escalation = (
            {},
            collections.defaultdict(list),
            set(),
            set(),
        )
        from scripts.measure_tiers import qualifying_skills

        qualifying = {s.id for s in qualifying_skills(skills)}
        for s in skills:
            for src in s.sources:
                if s.status == "active":
                    live_sources.add(
                        (src.topic, src.difficulty, frozenset((k, repr(v)) for k, v in (src.params or {}).items()))
                    )
                if src.difficulty == "exam":
                    exam_source_of_skill[s.id] = src.topic
                    exam_skill_of_topic[src.topic].append(s.id)
                    if s.id in qualifying and s.sources[-1] is src:
                        escalation.add(src.topic)
        tier_topics = {t for t, (_l, fn) in REGISTRY.items() if "tier" in inspect.signature(fn).parameters}
        return cls(
            shared=shared,
            registry=REGISTRY,
            presets=PRESETS,
            exam_topics=set(shared.EXAM_TOPICS),
            exam_source_of_skill=exam_source_of_skill,
            exam_skill_of_topic=dict(exam_skill_of_topic),
            escalation_topics=escalation,
            live_sources=live_sources,
            tier_topics=tier_topics,
            tier_rejects=tier_topics - set(shared.TIER_BASELINE),
        )

    def effective(self, topic, difficulty, extra):
        """The parameters a `make(topic, difficulty=..., **extra)` call runs the generator with (defaults included)."""
        fn = self.registry[topic][1]
        merged = {
            name: p.default
            for name, p in inspect.signature(fn).parameters.items()
            if p.default is not inspect.Parameter.empty
        }
        if difficulty is not None:
            merged.update(self.presets.get(topic, {}).get(difficulty, {}))
        merged.update(extra)
        return merged

    def is_exam_tier(self, target):
        """True when `target` draws from the same distribution as the exam tier of its topic."""
        if target.topic not in self.registry or target.starred:
            return False
        if target.difficulty == "exam":
            return not target.extra or self.effective(target.topic, "exam", target.extra) == self.effective(
                target.topic, "exam", {}
            )
        if target.difficulty is None and target.topic in self.exam_topics:
            return self.effective(target.topic, None, target.extra) == self.effective(target.topic, "exam", {})
        return False


# ----------------------------------------------------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class Target:
    """What a copy generates: `make(topic, difficulty=..., **extra)`."""

    topic: str | None
    difficulty: str | None = None
    extra: dict = field(default_factory=dict)
    starred: bool = False  # a `**params` splat the script cannot see through
    why: str = ""  # why topic is None


@dataclass
class Fact:
    """One generic check a statement performs."""

    kind: str  # deterministic | text | shapes | newshape | answer_space | wired | unknown_tier | keep
    target: Target | None = None
    seeds: int | None = None
    minimum: int | None = None  # distinct answers / prompt shapes
    share: float | None = None  # largest share of one answer
    working: bool = False  # find_text_defects(include_working=True)
    note: str = ""  # for "keep": what is not generic
    skill: str | None = None
    flags: set = field(default_factory=set)  # wired: {"topic","stages","params","last"}


class Ctx:
    """A parsed test module."""

    def __init__(self, path: Path):
        self.path = path
        self.source = path.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)
        self.consts: dict = {}
        self.funcs: dict[str, ast.FunctionDef] = {}
        self.fixtures: dict[str, ast.FunctionDef] = {}
        self.imports: dict[str, tuple[str, str | None]] = {}  # local name -> (module stem, attr or None)
        self.assigns: dict[str, ast.expr] = {}
        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef):
                self.funcs[node.name] = node
                if any("fixture" in ast.unparse(d) for d in node.decorator_list):
                    self.fixtures[node.name] = node
            elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                tgt = node.targets[0]
                if isinstance(tgt, ast.Name):
                    self.assigns[tgt.id] = node.value
                    value = self._literal(node.value)
                    if value is not _MISSING:
                        self.consts[tgt.id] = value
                elif isinstance(tgt, ast.Tuple):
                    for i, elt in enumerate(tgt.elts):
                        if isinstance(elt, ast.Name):
                            self.assigns[elt.id] = ast.Subscript(node.value, ast.Constant(i))
            elif isinstance(node, ast.ImportFrom):
                stem = (node.module or "").split(".")[-1]
                for alias in node.names:
                    self.imports[alias.asname or alias.name] = (stem, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    self.imports[(alias.asname or alias.name).split(".")[0]] = (alias.name.split(".")[-1], None)

    def _literal(self, node):
        try:
            return ast.literal_eval(node)
        except Exception:
            pass
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "range"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
        ):
            return range(node.args[0].value)
        return _MISSING

    def const(self, node, bindings=None):
        """The value of a constant-ish expression, or _MISSING."""
        bindings = bindings or {}
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in bindings:
                return bindings[node.id]
            if node.id in self.consts:
                return self.consts[node.id]
            return _MISSING
        if isinstance(node, ast.JoinedStr):
            return _MISSING
        value = self._literal(node)
        return value


_MISSING = object()


def seed_count(node, ctx: Ctx, bindings=None) -> int | None:
    """How many seeds an iterable such as `range(60)`, `SEEDS`, `(0, 1, 7, 99)`, `questions[:60]` draws."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
        if len(node.args) == 1:
            value = ctx.const(node.args[0], bindings)
            return value if isinstance(value, int) else None
        if len(node.args) == 2:
            lo, hi = ctx.const(node.args[0], bindings), ctx.const(node.args[1], bindings)
            return hi - lo if isinstance(lo, int) and isinstance(hi, int) else None
    if isinstance(node, (ast.Tuple, ast.List)):
        return len(node.elts)
    if isinstance(node, ast.Name):
        value = ctx.const(node, bindings)
        if isinstance(value, range):
            return len(value)
        if isinstance(value, (tuple, list)):
            return len(value)
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice) and node.slice.upper is not None:
        value = ctx.const(node.slice.upper, bindings)
        if isinstance(value, int):
            return value
    return None


def comprehension_seeds(node, ctx: Ctx, bindings=None) -> int | None:
    """Seeds drawn by a list/set/generator comprehension: the product of its iterables' sizes."""
    if not isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        return None
    total = 1
    for gen in node.generators:
        n = seed_count(gen.iter, ctx, bindings)
        if n is None:
            n = fixture_length(gen.iter, ctx)
        if n is None:
            return None
        total *= n
    return total


def fixture_length(node, ctx: Ctx) -> int | None:
    """Seeds in a `questions` fixture iterated whole."""
    if isinstance(node, ast.Name) and node.id in ctx.fixtures:
        for ret in ast.walk(ctx.fixtures[node.id]):
            if isinstance(ret, ast.Return) and ret.value is not None:
                value = ret.value
                if isinstance(value, ast.Name):  # return qs
                    value = next(
                        (
                            a.value
                            for a in ast.walk(ctx.fixtures[node.id])
                            if isinstance(a, ast.Assign) and a.targets[0] == value
                        ),
                        value,
                    )
                return comprehension_seeds(value, ctx)
    return None


def find_make_calls(node):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and _is_make(n.func)]


def _is_make(func):
    return (isinstance(func, ast.Name) and func.id in ("make", "_make_raw")) or (
        isinstance(func, ast.Attribute) and func.attr == "make"
    )


def target_of_make(call: ast.Call, ctx: Ctx, bindings=None) -> Target:
    bindings = bindings or {}
    if not call.args:
        return Target(None, why="make() without a topic")
    topic = ctx.const(call.args[0], bindings)
    if not isinstance(topic, str):
        return Target(None, why=f"topic is not a constant ({ast.unparse(call.args[0])})")
    difficulty, extra, starred = None, {}, False
    for kw in call.keywords:
        if kw.arg is None:
            starred = True
        elif kw.arg == "seed":
            continue
        elif kw.arg == "difficulty":
            value = ctx.const(kw.value, bindings)
            if value is _MISSING:
                return Target(None, why=f"difficulty is not a constant ({ast.unparse(kw.value)})")
            difficulty = value
        else:
            value = ctx.const(kw.value, bindings)
            extra[kw.arg] = ast.unparse(kw.value) if value is _MISSING else value
    return Target(topic, difficulty, extra, starred)


def resolve_target(expr, ctx: Ctx, bindings=None, depth=0) -> Target:
    """The (topic, tier) an expression generates: a `make(...)` call, or a module helper / fixture that wraps one."""
    bindings = bindings or {}
    if depth > 4:
        return Target(None, why="helper chain too deep")
    calls = [n for n in ast.walk(expr) if isinstance(n, ast.Call)]
    for call in calls:
        if _is_make(call.func):
            return target_of_make(call, ctx, bindings)
    for call in calls:
        if isinstance(call.func, ast.Name) and call.func.id in ctx.funcs:
            helper = ctx.funcs[call.func.id]
            params = [a.arg for a in helper.args.args]
            sub = dict(bindings)
            for name, arg in zip(params, call.args):
                value = ctx.const(arg, bindings)
                if value is not _MISSING:
                    sub[name] = value
                else:
                    sub.pop(name, None)
            for kw in call.keywords:
                if kw.arg:
                    value = ctx.const(kw.value, bindings)
                    if value is not _MISSING:
                        sub[kw.arg] = value
            inner = [n for n in ast.walk(helper) if isinstance(n, ast.Return) and n.value is not None]
            for ret in inner:
                tgt = resolve_target(ret.value, ctx, sub, depth + 1)
                if tgt.topic or tgt.why:
                    return tgt
    names = [n.id for n in ast.walk(expr) if isinstance(n, ast.Name)]
    for name in names:
        if name in ctx.fixtures:
            for ret in (n for n in ast.walk(ctx.fixtures[name]) if isinstance(n, ast.Return) and n.value is not None):
                value = ret.value
                if isinstance(value, ast.Name):
                    for a in ast.walk(ctx.fixtures[name]):
                        if (
                            isinstance(a, ast.Assign)
                            and isinstance(a.targets[0], ast.Name)
                            and a.targets[0].id == value.id
                        ):
                            value = a.value
                return resolve_target(value, ctx, bindings, depth + 1)
    return Target(None, why="no make() call or known helper in the expression")


# ----------------------------------------------------------------------------------------------------------------
# Statement recognisers
# ----------------------------------------------------------------------------------------------------------------


class Analyser:
    """Turns one test function into the list of generic `Fact`s it asserts, or explains why it cannot."""

    def __init__(self, ctx: Ctx, fn: ast.FunctionDef):
        self.ctx = ctx
        self.fn = fn
        self.sym: dict[str, ast.expr] = {}
        self.loops: list[tuple[str, int | None]] = []  # (loop var, seeds)
        self.facts: list[Fact] = []
        self.blockers: list[str] = []
        self.params = [a.arg for a in fn.args.args]
        self.default_seeds: int | None = None

    # -- driver -------------------------------------------------------------------------------------------------

    def run(self):
        body = self.fn.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]
        for p in self.params:
            if p in self.ctx.fixtures:
                self.sym[p] = ast.Name(p)
        self.walk(body, None)
        return self

    def walk(self, stmts, loop_seeds):
        for st in stmts:
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                self.sym[st.targets[0].id] = st.value
                self.note_seeds(st.value)
            elif isinstance(st, ast.Assign):
                for tgt in st.targets:
                    if isinstance(tgt, ast.Name):
                        self.sym[tgt.id] = st.value
            elif isinstance(st, ast.AugAssign):
                self.blockers.append(f"augmented assignment: {ast.unparse(st)[:60]}")
            elif isinstance(st, ast.For):
                n = seed_count(st.iter, self.ctx)
                if n is None:
                    n = self.sliced_fixture_seeds(st.iter)
                if isinstance(st.target, ast.Name):
                    self.sym[st.target.id] = ast.Name("<loopvar>")
                self.loops.append((ast.unparse(st.target), n))
                self.walk(st.body, n if n is not None else loop_seeds)
                self.loops.pop()
                if st.orelse:
                    self.blockers.append("for/else")
            elif isinstance(st, ast.Assert):
                self.classify_assert(st, loop_seeds)
            elif isinstance(st, ast.Expr):
                if isinstance(st.value, ast.Constant):
                    continue
                self.classify_call_statement(st.value)
            elif isinstance(st, ast.With):
                self.classify_with(st)
            elif isinstance(st, ast.Return) and st.value is None:
                continue
            else:
                self.blockers.append(f"{type(st).__name__}: {ast.unparse(st).splitlines()[0][:70]}")

    def note_seeds(self, value):
        n = comprehension_seeds(value, self.ctx)
        if n is not None and self.default_seeds is None:
            self.default_seeds = n

    def sliced_fixture_seeds(self, it):
        if isinstance(it, ast.Subscript) and isinstance(it.slice, ast.Slice):
            n = seed_count(it, self.ctx)
            if n is not None:
                return n
        if isinstance(it, ast.Name) and it.id in self.ctx.fixtures:
            return fixture_length(it, self.ctx)
        return None

    def inline(self, node, depth=0):
        """`node` with the names this function assigned replaced by their right-hand sides."""
        if depth > 6:
            return node

        class _Sub(ast.NodeTransformer):
            def visit_Name(inner, n):
                if (
                    isinstance(n.ctx, ast.Load)
                    and n.id in self.sym
                    and not (isinstance(self.sym[n.id], ast.Name) and self.sym[n.id].id in ("<loopvar>",))
                ):
                    return self.inline(self.sym[n.id], depth + 1)
                return n

        return _Sub().visit(ast.parse(ast.unparse(node), mode="eval").body)

    # -- asserts ------------------------------------------------------------------------------------------------

    def classify_assert(self, st: ast.Assert, loop_seeds):
        test = st.test
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
            for v in test.values:
                self.classify_test(v, loop_seeds, st)
        else:
            self.classify_test(test, loop_seeds, st)

    def classify_test(self, test, loop_seeds, st):
        text = ast.unparse(test)
        # determinism: identical call on both sides of ==
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
            left, right = test.left, test.comparators[0]
            if self.is_defects_call(left) and isinstance(right, ast.List) and not right.elts:
                return self.text_fact(left, loop_seeds)
            li, ri = self.inline(left), self.inline(right)
            if ast.dump(li) == ast.dump(ri) and isinstance(li, ast.Call | ast.Attribute):
                return self.determinism_fact(li, loop_seeds)
            if self.is_wired_text(text):
                return self.wired_fact(text)
        if self.is_wired_text(text):
            return self.wired_fact(text)
        if self.shape_assert(test, loop_seeds):
            return
        if self.answer_assert(test, loop_seeds):
            return
        self.blockers.append(f"assertion not generic: {text[:90]}")

    # determinism
    def determinism_fact(self, call, loop_seeds):
        fn = call.func if isinstance(call, ast.Call) else call
        attr = fn.attr if isinstance(fn, ast.Attribute) else ""
        if attr not in ("to_json", "to_dict", "text", "answer_text"):
            self.blockers.append(f"equality of two calls, not a question: {ast.unparse(call)[:60]}")
            return
        base = call.func.value if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) else call
        target = resolve_target(base, self.ctx)
        self.facts.append(Fact("deterministic", target, seeds=loop_seeds))

    # text defects
    def is_defects_call(self, node):
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "find_text_defects"

    def text_fact(self, call, loop_seeds):
        working = any(kw.arg == "include_working" and ast.unparse(kw.value) == "True" for kw in call.keywords)
        arg = call.args[0]
        target = self.target_of_item(arg)
        seeds = self.item_seeds(arg, loop_seeds)
        self.facts.append(Fact("text", target, seeds=seeds, working=working))

    def target_of_item(self, arg):
        inl = self.inline(arg)
        if isinstance(arg, ast.Name) and arg.id in self.sym and isinstance(self.sym[arg.id], ast.Name):
            # a loop variable: find the iterable it came from
            return self.loop_target(arg.id)
        return resolve_target(inl, self.ctx)

    def loop_target(self, var):
        for st in ast.walk(self.fn):
            if isinstance(st, ast.For) and isinstance(st.target, ast.Name) and st.target.id == var:
                return resolve_target(st.iter, self.ctx)
        return Target(None, why="loop variable of unknown origin")

    def item_seeds(self, arg, loop_seeds):
        if loop_seeds is not None:
            return loop_seeds
        return None

    # shapes
    def shape_set(self, node):
        """If `node` (a Name or set comprehension) is `{prompt_shape(x) for ...}`: (target, seeds)."""
        inl = self.inline(node)
        if (
            isinstance(inl, ast.SetComp)
            and isinstance(inl.elt, ast.Call)
            and getattr(inl.elt.func, "id", "") == "prompt_shape"
        ):
            target = resolve_target(inl.elt.args[0], self.ctx)
            if target.topic is None and inl.generators:
                it = inl.generators[0].iter
                if isinstance(inl.elt.args[0], ast.Name):
                    target = resolve_target(it, self.ctx)
            return target, comprehension_seeds(inl, self.ctx)
        return None

    def shape_assert(self, test, loop_seeds):
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.GtE):
            left = test.left
            if isinstance(left, ast.Call) and getattr(left.func, "id", "") == "len" and left.args:
                k = self.ctx.const(test.comparators[0])
                inner = left.args[0]
                if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Sub):
                    a = self.shape_set(inner.left)
                    if a and isinstance(k, int):
                        self.facts.append(Fact("newshape", a[0], seeds=a[1], minimum=k))
                        return True
                    return False
                found = self.shape_set(inner)
                if found and isinstance(k, int):
                    self.facts.append(Fact("shapes", found[0], seeds=found[1], minimum=k))
                    return True
        if isinstance(test, ast.BinOp) and isinstance(test.op, ast.Sub):
            a = self.shape_set(test.left)
            if a:
                self.facts.append(Fact("newshape", a[0], seeds=a[1], minimum=1))
                return True
        return False

    # answer space
    def answer_assert(self, test, loop_seeds):
        if not (isinstance(test, ast.Compare) and len(test.ops) == 1):
            return False
        op, left, right = test.ops[0], test.left, test.comparators[0]
        if (
            isinstance(op, ast.GtE)
            and isinstance(left, ast.Call)
            and getattr(left.func, "id", "") == "len"
            and left.args
        ):
            minimum = self.ctx.const(right)
            subject = self.answer_subject(left.args[0])
            if subject is None or not isinstance(minimum, int):
                return False
            self.facts.append(Fact("answer_space", subject[0], seeds=subject[1], minimum=minimum, note=subject[2]))
            return True
        if isinstance(op, ast.LtE):
            share = self.top_share(left, right)
            if share is not None:
                self.facts.append(Fact("answer_space", None, share=share, note="top"))
                return True
        return False

    def answer_subject(self, node):
        """(target, seeds, 'composed' | 'part') for the collection an answer-space count is taken over."""
        inl = self.inline(node)
        text = ast.unparse(inl)
        if "answer_text" not in text and "answer_tuple" not in text and "canonical" not in text.lower():
            return None
        kind = "part" if re.search(r"parts\[\s*-?\d+\s*\]|\[LABELS\[|\)\[(label|lab)\]|\[label\]", text) else "composed"
        comp = None
        for n in ast.walk(inl):
            if isinstance(n, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
                comp = n
                break
        seeds = comprehension_seeds(comp, self.ctx) if comp is not None else None
        if seeds is None and comp is not None:
            seeds = self.slice_seed(comp)
        target = resolve_target(inl, self.ctx)
        if target.topic is None and comp is not None:
            target = resolve_target(comp.generators[0].iter, self.ctx)
        return target, seeds, kind

    def slice_seed(self, comp):
        for n in ast.walk(comp):
            if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice):
                v = seed_count(n, self.ctx)
                if v:
                    return v
        return None

    def top_share(self, left, right):
        """The ceiling on the commonest answer's share, from `max(...) <= c`, `max(...)/n <= c`, ..."""
        inl_left = self.inline(left)
        text = ast.unparse(inl_left)
        if "max(" not in text and "most_common" not in text and "top" not in ast.unparse(left):
            return None
        value = self.ctx.const(right)
        seeds = self.default_seeds or 1
        if isinstance(right, ast.BinOp) and isinstance(right.op, ast.Mult):
            c = self.ctx.const(right.left)
            if isinstance(c, (int, float)) and isinstance(right.right, (ast.Call, ast.Name)):
                return float(c)
        if value is _MISSING:
            return None
        if isinstance(inl_left, ast.BinOp) and isinstance(inl_left.op, ast.Div):
            return float(value)
        if isinstance(value, (int, float)):
            return value / seeds if value >= 1 else float(value)
        return None

    # wired
    WIRED_RES = [
        (r"^\[\w+\.topic for \w+ in \w+\] == \[(\w+|'[^']*'|f'[^']*')\]$", "topic"),
        (r"^\[\w+\.topic for \w+ in .*\] == \[(\w+|'[^']*'|f'[^']*')\]$", "topic"),
        (r"^[\w\[\]()]+\.stages == \['exam'\]$", "stages"),
        (r"^list\([\w\[\]]+\.stages\) == \['exam'\]$", "stages"),
        (r"^\[\w+\.stages for \w+ in \w+\] == \[\['exam'\]\]$", "stages"),
        (r"^[\w\[\]]+\.params == \{'tier': 'exam'\}$", "params"),
        (r"^dict\([\w\[\]]+\.params\) == \{'tier': 'exam'\}$", "params"),
        (r"^len\(\w+\) == 1$", "topic"),
        (r"^\w+\.topic == (\w+|'[^']*')$", "topic"),
        (r"^\w+\.difficulty == 'exam'$", "topic"),
        (r"^[\w.]+\.sources\[-1\] is [\w\[\]]+$", "last"),
        (r"^sum\(\(1 for \w+ in [\w.]+ if \w+\.difficulty == 'exam'\)\) == 1$", "topic"),
        (r"^\(\w+\.topic, \w+\.stages\) == \(\w+, \['exam'\]\)$", "topic+stages"),
        (
            r"^\(\w+\.topic, \w+\.difficulty, \w+\.stages, \w+\.params\) == \(\w+, 'exam', \['exam'\], \{'tier': 'exam'\}\)$",
            "all",
        ),
        (r"^\[\(\w+\.topic, \w+\.stages\) for \w+ in \w+\] == \[\(\w+, \['exam'\]\)\]$", "topic+stages"),
        (
            r"^\[\(\w+\.topic, \w+\.stages, \w+\.params\) for \w+ in \w+\] == \[\(\w+, \['exam'\], \{'tier': 'exam'\}\)\]$",
            "all",
        ),
        (r"^\(\w+\.topic, \w+\.stages, \w+\.params\) == \(\w+, \['exam'\], \{'tier': 'exam'\}\)$", "all"),
        (
            r"^\(\w+\.topic, \w+\.stages, \w+\.params\) == \(\w+, \['exam'\], \{'tier': 'exam'\}\) and \w+\.sources\[-1\] is \w+$",
            "all",
        ),
    ]

    def is_wired_text(self, text):
        return any(re.match(rx, text) for rx, _ in self.WIRED_RES) or "and" in text and self._split_and_wired(text)

    def _split_and_wired(self, text):
        parts = re.split(r" and ", text)
        return len(parts) > 1 and all(any(re.match(rx, p.strip()) for rx, _ in self.WIRED_RES) for p in parts)

    def is_wired_compare(self, test, text):
        return self.is_wired_text(text)

    def wired_fact(self, text):
        flags = set()
        for part in re.split(r" and ", text):
            for rx, flag in self.WIRED_RES:
                if re.match(rx, part.strip()):
                    flags |= {"topic", "stages", "params", "last"} if flag == "all" else set(flag.split("+"))
        topic = self.wired_topic()
        self.facts.append(Fact("wired", topic, flags=flags))

    def wired_topic(self):
        """The topic a wiring assertion pairs with: TOPIC / EXAM_TOPIC constants of the module."""
        for name in ("TOPIC", "EXAM_TOPIC"):
            if isinstance(self.ctx.consts.get(name), str):
                return Target(self.ctx.consts[name], "exam")
        for st in ast.walk(self.fn):
            if isinstance(st, ast.Constant) and isinstance(st.value, str) and st.value in REGISTRY_NAMES():
                return Target(st.value, "exam")
        return Target(None, why="wired topic is not a module constant")

    # -- other statements ---------------------------------------------------------------------------------------

    def classify_with(self, st: ast.With):
        item = st.items[0].context_expr
        text = ast.unparse(item)
        if text.startswith("pytest.raises(ValueError") and len(st.body) == 1:
            call = st.body[0]
            calls = [n for n in ast.walk(call) if isinstance(n, ast.Call) and _is_make(n.func)]
            if len(calls) == 1 and any(kw.arg == "tier" for kw in calls[0].keywords):
                target = target_of_make(calls[0], self.ctx)
                self.facts.append(Fact("unknown_tier", Target(target.topic, None)))
                return
        self.blockers.append(f"with block: {text[:60]}")

    def classify_call_statement(self, call):
        if not isinstance(call, ast.Call):
            self.blockers.append(f"expression statement: {ast.unparse(call)[:60]}")
            return
        facts = delegate_facts(call, self.ctx)
        if facts is None:
            self.blockers.append(f"call is not a known generic check: {ast.unparse(call)[:70]}")
        else:
            self.facts.extend(facts)


def REGISTRY_NAMES():
    from mathgen.core import REGISTRY

    return REGISTRY


# ----------------------------------------------------------------------------------------------------------------
# Delegating copies: `check_answer_space(TOPIC)` and friends, read from the support modules' own bodies.
# Each entry: (support module, function) -> (argument roles, facts). Verified against the module source on
# 2026-10-13; a function not listed here is not recognised, so the file keeps its test.
# ----------------------------------------------------------------------------------------------------------------

# Fact templates: (kind, seeds, minimum, share, working). `None` seeds -> the helper's default.
_SH = ("shapes", 60, 3, None, False)
_NS = ("newshape", 60, 1, None, False)
_TX = ("text", 60, None, None, False)
_DT = ("deterministic", None, None, None, False)
_AS = ("answer_space", 300, 100, 0.05, False)
_AS15 = ("answer_space", 300, 100, 0.05, False)
_WI = ("wired", None, None, None, False)

DELEGATES = {
    # tests/_exd_m5_common.py: answer space is per part, so it stays.
    ("_exd_m5_common", "check_shapes"): (("topic", "skill"), [_SH, _NS, _TX]),
    ("_exd_m5_common", "check_source_wired"): (("topic", "skill"), [_WI]),
    ("_exd_m5_common", "check_deterministic"): (("topic",), [_DT]),
    ("exd_x_m3_support", "check_source_wired"): (("skill", "topic"), [_WI]),
    ("exd_x_m3_support", "check_shapes"): (("skill", "topic"), [_SH, _NS]),
    ("exd_x_pr6_support", "check_wired"): (("topic", "skill"), [_WI]),
    ("exd_x_pr6_support", "check_variety"): (("topic", "skill"), [_SH, _NS]),
    ("exd_x_pr6_support", "check_answer_space"): (("topic",), [_AS]),
    ("exd_x_pr6_support", "check_deterministic_and_clean"): (("topic",), [("text", 40, None, None, False), _DT]),
    ("exd_x_r2a_support", "check_wired"): (("skill", "topic"), [_WI]),
    ("exd_x_r2a_support", "check_shapes"): (("skill", "topic"), [_SH, _NS]),
    ("exd_x_r2a_support", "check_answer_space"): (("topic",), [_AS]),
    ("exd_x_r2b_support", "assert_clean_and_deterministic"): (("topic",), [_TX, _DT]),
    ("exd_x_r2b_support", "assert_shapes"): (("skill", "topic"), [_SH, _NS]),
    ("exd_x_r2b_support", "assert_answer_space"): (("topic",), [_AS]),
    ("exd_x_r2c_support", "check_wired"): (("skill", "topic"), [_WI]),
    ("exd_x_r2c_support", "check_shapes"): (("topic",), [_SH, _NS]),
    ("exd_x_r2c_support", "check_answer_space"): (("topic",), [_AS]),
    ("exd_x_r2j_kit", "check_shapes"): (("skill",), [_SH, _NS]),
    ("exd_x_r2j_kit", "check_answer_space"): (("skill",), [_AS]),
    ("exd_x_r2k_kit", "check_shapes"): (("skill",), [_SH, _NS]),
    ("exd_x_r2k_kit", "check_answer_space"): (("skill",), [_AS]),
    ("exd_x_sc1_support", "check_shapes"): (("skill",), [_SH, _NS]),
    ("exd_x_sc1_support", "check_no_text_defects"): (("skill",), [_TX]),
    ("exd_x_sc1_support", "check_deterministic"): (("skill",), [_DT]),
    ("exd_x_sc1_support", "check_answer_space"): (("skill",), [_AS]),
    ("exd_x_sc2_support", "check_no_text_defects"): (("topic",), [_TX]),
    ("exd_x_sc2_support", "check_variety"): (("topic",), [_SH, _NS]),
    ("exd_x_sc2_support", "check_wired"): (("skill", "topic"), [_WI]),
    ("exd_x_sc2_support", "check_deterministic"): (("topic",), [_DT]),
    ("exd_x_sc3_support", "assert_clean_and_deterministic"): (("topic",), [_TX, _DT]),
    ("exd_x_sc3_support", "assert_shapes"): (("skill", "topic"), [_SH, _NS]),
    ("exd_x_sc3_support", "assert_answer_space"): (("topic",), [_AS]),
    ("exd_x_sc6_support", "assert_source_wired"): (("skill_obj", "topic"), [_WI]),
    ("exd_x_sc6_support", "assert_new_shapes"): (("skill_obj",), [_SH, _NS]),
    ("exd_x_sc6_support", "assert_no_text_defects"): (("skill_obj",), [_TX]),
    ("exd_x_sc6_support", "assert_answer_space"): (("skill_obj",), [_AS]),
    ("exd_x_u4_support", "check_variety"): (("topic",), [_SH, _NS]),
    ("exd_x_u4_support", "check_no_text_defects"): (("topic",), [_TX]),
    ("exd_x_u4_support", "check_answer_space"): (("topic",), [_AS]),
    ("exd_x_u5_support", "check_shapes"): (("skill",), [_SH, _NS]),
    ("exd_x_u5_support", "check_no_text_defects"): (("skill",), [_TX]),
    ("exd_x_c3_support", "assert_clean_text_and_deterministic"): (("topic",), [_TX, _DT]),
    ("exd_x_c3_support", "assert_shapes"): (("skill", "topic"), [_SH, _NS]),
    # Present in the support modules but NOT listed, so the file keeps its test: _exd_m5_common.check_answer_space
    # (one part), exd_x_m3_support.check_answer_space (adds a per-part clause), exd_x_sc2_support.check_answer_space
    # (adds a per-label share clause), exd_x_r2a_support.check_clean_and_deterministic and
    # exd_x_r2c_support.check_clean_and_deterministic (add part-level assertions).
}

# The `CHECKS = standard_checks(...)` families: key -> facts. Shared by r2g, r2h and r2i support modules.
STANDARD_CHECKS = {
    "wired": [_WI],
    "shapes": [_SH, _NS],
    "answer_space": [_AS],
    "deterministic": [("deterministic", None, None, None, False)],
}
# `shape`, `oracle_agrees`, `composed_full_marks`, `distractors_rejected` stay: they check topic-specific data.
STANDARD_SUPPORT = {"exd_x_r2g_support", "exd_x_r2h_support", "exd_x_r2i_support"}
C1_UNPACK = {0: [_SH, _NS], 3: [("text", 60, None, None, False), _DT]}  # exd_x_c1_support.standard_checks()


def _fact_from_template(template, target, skill=None):
    kind, seeds, minimum, share, working = template
    return Fact(kind, target, seeds=seeds, minimum=minimum, share=share, working=working, skill=skill)


def delegate_facts(call: ast.Call, ctx: Ctx):
    """Facts for a call such as `check_answer_space(TOPIC)`, `CHECKS["answer_space"]()` or `variety()`."""
    func = call.func
    if isinstance(func, ast.Subscript) and isinstance(func.value, ast.Name) and func.value.id == "CHECKS":
        key = ctx.const(func.slice)
        origin = ctx.assigns.get("CHECKS")
        support = _call_module(origin, ctx) if isinstance(origin, ast.Call) else None
        if support in STANDARD_SUPPORT and key in STANDARD_CHECKS:
            topic = ctx.const(origin.args[0]) if origin.args else _MISSING
            skill = ctx.const(origin.args[1]) if len(origin.args) > 1 else _MISSING
            target = Target(topic, "exam") if isinstance(topic, str) else Target(None, why="CHECKS topic not constant")
            extra = _standard_kwargs(origin, ctx)
            facts = [
                _fact_from_template(t, target, skill if isinstance(skill, str) else None) for t in STANDARD_CHECKS[key]
            ]
            if key == "answer_space" and "max_share" in extra:
                facts[0].share = extra["max_share"] / 300
            return facts
        return None
    if isinstance(func, ast.Name) and func.id in ctx.assigns and not call.args:
        origin = ctx.assigns[func.id]  # variety()/clean_and_deterministic() unpacked from c1's standard_checks
        if (
            isinstance(origin, ast.Subscript)
            and isinstance(origin.value, ast.Call)
            and _call_module(origin.value, ctx) == "exd_x_c1_support"
        ):
            index = ctx.const(origin.slice)
            if index in C1_UNPACK:
                topic = ctx.const(origin.value.args[0]) if origin.value.args else _MISSING
                target = Target(topic, "exam") if isinstance(topic, str) else Target(None, why="topic not constant")
                return [_fact_from_template(t, target) for t in C1_UNPACK[index]]
        return None
    module, name = _callee(func, ctx)
    entry = DELEGATES.get((module, name))
    if entry is None:
        return None
    roles, templates = entry
    target, skill_id = Target(None, why="no topic argument"), None
    for role, arg in zip(roles, call.args):
        value = ctx.const(arg)
        if role == "topic" and isinstance(value, str):
            target = Target(value, "exam")
        elif role in ("skill", "skill_obj"):
            skill_id = (
                value
                if isinstance(value, str)
                else ctx.const(ctx.assigns.get(getattr(arg, "id", ""), arg))
                if isinstance(arg, ast.Name)
                else None
            )
            if not isinstance(skill_id, str):
                skill_id = _skill_id_from_name(arg, ctx)
            topic = LIB().exam_source_of_skill.get(skill_id) if skill_id else None
            if target.topic is None:
                target = Target(topic, "exam") if topic else Target(None, why=f"skill {skill_id!r} has no exam source")
    facts = [_fact_from_template(t, target, skill_id) for t in templates]
    for fact in facts:  # keyword overrides such as n=100
        for kw in call.keywords:
            value = ctx.const(kw.value)
            if kw.arg in ("n", "seeds") and isinstance(value, int):
                fact.seeds = value
            elif kw.arg in ("minimum", "distinct") and isinstance(value, int) and fact.kind == "answer_space":
                fact.minimum = value
            elif kw.arg in ("share", "ceiling", "top") and isinstance(value, float) and fact.kind == "answer_space":
                fact.share = value
    return facts


def _skill_id_from_name(arg, ctx: Ctx):
    """`SKILL` may be a string constant or `_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)`."""
    if isinstance(arg, ast.Name):
        origin = ctx.assigns.get(arg.id)
        if origin is not None:
            for n in ast.walk(origin):
                if isinstance(n, ast.Compare) and isinstance(n.comparators[0], ast.Name):
                    value = ctx.consts.get(n.comparators[0].id)
                    if isinstance(value, str):
                        return value
                if isinstance(n, ast.Constant) and isinstance(n.value, str):
                    pass
            for n in ast.walk(origin):
                if (
                    isinstance(n, ast.Call)
                    and getattr(n.func, "id", "") in ("skill", "skill_of", "load_skill")
                    and n.args
                ):
                    value = ctx.const(n.args[0])
                    if isinstance(value, str):
                        return value
    return None


def _call_module(call, ctx: Ctx):
    module, _name = _callee(call.func, ctx)
    return module


def _callee(func, ctx: Ctx):
    """(support module stem, function name) of a call's target, following the file's own imports."""
    if isinstance(func, ast.Name):
        origin = ctx.imports.get(func.id)
        return (origin[0], origin[1]) if origin else (None, func.id)
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        origin = ctx.imports.get(func.value.id)
        if origin:
            module = origin[1] if origin[1] else origin[0]
            return module, func.attr
    return (None, None)


def _standard_kwargs(call: ast.Call, ctx: Ctx):
    out = {}
    for kw in call.keywords:
        value = ctx.const(kw.value)
        if kw.arg and value is not _MISSING:
            out[kw.arg] = value
    return out


_LIB = None


def LIB() -> Library:
    global _LIB
    if _LIB is None:
        _LIB = Library.load()
    return _LIB


# ----------------------------------------------------------------------------------------------------------------
# Deciding
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class Verdict:
    removable: bool
    reasons: list[str]  # why it stays, or what covers it
    stricter: bool = False  # recognised as generic, but stricter than the shared test somewhere


def judge_fact(fact: Fact, lib: Library) -> tuple[bool, bool, str]:
    """(covered, stricter, note): is this generic check covered by the shared test, at least as strictly?"""
    shared = lib.shared
    target = fact.target
    kind = fact.kind
    if kind == "answer_space" and fact.note == "top":
        return True, False, "share ceiling (judged with the distinct-count assertion)"
    if target is None or target.topic is None:
        return False, False, f"unresolved target: {target.why if target else 'none'}"
    topic = target.topic
    if topic not in lib.registry:
        return False, False, f"topic {topic!r} is not registered"
    if kind == "answer_space":
        if fact.note == "part":
            return False, False, "counts one part's answers, not the composed answer"
        if not lib.is_exam_tier(target):
            return False, False, f"target is not the exam tier ({target.difficulty!r}, {target.extra})"
        if topic not in lib.exam_topics:
            return False, False, "topic has no exam preset"
        minimum, share = shared.answer_space_for(topic)
        seeds = fact.seeds or shared.ANSWER_SEEDS
        if seeds < shared.ANSWER_SEEDS and (fact.minimum or 0) / seeds > minimum / shared.ANSWER_SEEDS:
            return True, True, f"copy needs {fact.minimum} distinct in {seeds} seeds, a denser answer space than shared"
        copy_share = fact.share if fact.share is not None else 1.0
        if seeds > shared.ANSWER_SEEDS:
            return True, True, f"copy draws {seeds} seeds, shared {shared.ANSWER_SEEDS}"
        if (fact.minimum or 0) > minimum:
            return True, True, f"copy needs {fact.minimum} distinct, shared {minimum} for this topic"
        if copy_share < share:
            return True, True, f"copy caps the commonest answer at {copy_share:.3f}, shared at {share:.3f}"
        return True, False, "answer space"
    if kind in ("text", "deterministic"):
        if kind == "deterministic":
            covered = (
                lib.is_exam_tier(target)
                and topic in lib.exam_topics
                or target.difficulty is None
                and not target.extra
                or (topic, target.difficulty, frozenset((k, repr(v)) for k, v in target.extra.items()))
                in lib.live_sources
                or topic in lib.presets
                and target.difficulty in lib.presets[topic]
                and not target.extra
            )
            return (
                (True, False, "determinism")
                if covered
                else (False, False, f"tier {target.difficulty!r}/{target.extra} not covered")
            )
        if not lib.is_exam_tier(target) or topic not in lib.exam_topics:
            return False, False, f"target is not the exam tier ({target.difficulty!r}, {target.extra})"
        if fact.working and not shared.TEXT_INCLUDES_WORKING:
            return True, True, "copy also checks the worked-solution lines"
        if (fact.seeds or 0) > shared.TEXT_SEEDS:
            return True, True, f"copy draws {fact.seeds} seeds, shared {shared.TEXT_SEEDS}"
        if topic in shared.TEXT_EXEMPT:
            return False, False, "topic exempt from the shared text check"
        return True, False, "text defects"
    if kind == "shapes":
        if not lib.is_exam_tier(target) or topic not in lib.exam_topics:
            return False, False, f"target is not the exam tier ({target.difficulty!r})"
        if (fact.minimum or 0) > shared.MIN_SHAPES:
            return True, True, f"copy needs {fact.minimum} shapes, shared {shared.MIN_SHAPES}"
        if fact.seeds is not None and fact.seeds < shared.SHAPE_SEEDS:
            return (
                True,
                True,
                f"copy draws {fact.seeds} seeds, shared {shared.SHAPE_SEEDS} (fewer seeds is the harder bar)",
            )
        if topic in shared.SHAPES_PINNED:
            return False, False, "topic is pinned below the shared shapes bar"
        return True, False, "prompt shapes"
    if kind == "newshape":
        if not lib.is_exam_tier(target) or topic not in lib.exam_topics:
            return False, False, f"target is not the exam tier ({target.difficulty!r})"
        if (fact.minimum or 1) > 1:
            return True, True, f"copy needs {fact.minimum} new shapes; test_tier_escalation needs 1"
        if (fact.seeds or 0) > 60:
            return True, True, f"copy draws {fact.seeds} seeds, test_tier_escalation 60"
        if topic not in lib.escalation_topics:
            return (
                False,
                False,
                "topic is not the last source of a qualifying skill, so test_tier_escalation does not reach it",
            )
        return True, False, "new shape (test_tier_escalation)"
    if kind == "wired":
        if topic not in lib.exam_skill_of_topic:
            return False, False, "topic is not wired as an exam source anywhere"
        if "params" in fact.flags:
            pinned = __import__("tests.test_generator_contract", fromlist=["x"]).expected_exam_params(topic)
            if topic.startswith(shared.EXAM_PREFIXES) is False or pinned != {"tier": "exam"}:
                return (
                    False,
                    False,
                    f"exam source params are {pinned} (or the topic is not exd_/exam_), not the {{'tier': 'exam'}} the copy asserts",
                )
        if topic in shared.UNWIRED_BASELINE:
            return False, False, "topic is in the shared unwired baseline"
        if len(lib.exam_skill_of_topic[topic]) > 1 and "topic" in fact.flags:
            pass
        return True, False, "wiring"
    if kind == "unknown_tier":
        if topic not in lib.tier_topics:
            return False, False, "generator has no tier parameter"
        if topic in shared.TIER_BASELINE:
            return False, False, "generator does not reject an unknown tier with ValueError (shared baseline)"
        return True, False, "unknown tier"
    return False, False, f"unhandled fact {kind}"


def judge_function(fn: ast.FunctionDef, ctx: Ctx, lib: Library) -> Verdict:
    ana = Analyser(ctx, fn).run()
    decorators = " ".join(ast.unparse(d) for d in fn.decorator_list)
    reasons = list(ana.blockers)
    if "parametrize" in decorators and not ana.facts:
        reasons.append("parametrised target")
    if not ana.facts and not reasons:
        reasons.append("no recognisable generic assertion")
    stricter = False
    # An answer-space share assertion follows its distinct-count assertion; fold it into the previous fact.
    folded: list[Fact] = []
    for fact in ana.facts:
        if fact.kind == "answer_space" and fact.note == "top" and folded and folded[-1].kind == "answer_space":
            folded[-1].share = fact.share
        else:
            folded.append(fact)
    notes = []
    for fact in folded:
        covered, strict, note = judge_fact(fact, lib)
        if not covered:
            reasons.append(f"{fact.kind}: {note}")
        elif strict:
            stricter = True
            reasons.append(f"{fact.kind}: stricter than shared ({note})")
        else:
            notes.append(f"{fact.kind}")
    if "parametrize" in decorators:
        targets = {f.target.topic for f in folded if f.target and f.target.topic}
        if not targets or len(ana.params) > 0 and not folded:
            reasons.append("parametrised target")
    if reasons:
        return Verdict(False, reasons, stricter and all("stricter" in r or r.startswith("") for r in reasons))
    return Verdict(True, [f"covered: {', '.join(sorted(set(notes)))}"])


# ----------------------------------------------------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------------------------------------------------


def family_of(name: str):
    for fam, rx in FAMILY_PATTERNS.items():
        if re.match(rx, name):
            return fam
    return None


@dataclass
class FileReport:
    path: Path
    removable: list[ast.FunctionDef] = field(default_factory=list)
    kept: list[tuple[str, Verdict, str]] = field(default_factory=list)  # (name, verdict, family)
    leftover_tests: list[str] = field(default_factory=list)
    removed_names: list[str] = field(default_factory=list)
    covered_by: dict[str, str] = field(default_factory=dict)  # removable function -> what covers it
    ctx: Ctx | None = None

    @property
    def stricter(self):
        return [k for k in self.kept if k[1].stricter]


def analyse_file(path: Path, lib: Library) -> FileReport:
    ctx = Ctx(path)
    report = FileReport(path, ctx=ctx)
    for node in ctx.tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
            fam = family_of(node.name)
            if fam is None:
                report.leftover_tests.append(node.name)
                continue
            verdict = judge_function(node, ctx, lib)
            if verdict.removable:
                report.removable.append(node)
                report.removed_names.append(node.name)
                report.covered_by[node.name] = verdict.reasons[0]
            else:
                report.kept.append((node.name, verdict, fam))
                report.leftover_tests.append(node.name)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            report.leftover_tests.append(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id.startswith("test_"):
                    report.leftover_tests.append(tgt.id)
    return report


def importers(files: list[Path]) -> dict[str, list[str]]:
    """module stem -> test files that import it (other than itself), so a 'deletable' file is not one others use."""
    stems = {p.stem for p in files}
    used: dict[str, list[str]] = collections.defaultdict(list)
    for p in sorted(TESTS.glob("*.py")):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.ImportFrom):
                mods = [(node.module or "").split(".")[-1]] + [a.name for a in node.names]
            elif isinstance(node, ast.Import):
                mods = [a.name.split(".")[-1] for a in node.names]
            for m in mods:
                if m in stems and m != p.stem:
                    used[m].append(p.name)
    return used


def remove_functions(path: Path, nodes: list[ast.FunctionDef]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    spans = []
    for fn in nodes:
        start = min([fn.lineno] + [d.lineno for d in fn.decorator_list]) - 1
        end = fn.end_lineno
        while end < len(lines) and not lines[end].strip():
            end += 1
        spans.append((start, end))
    for start, end in sorted(spans, reverse=True):
        del lines[start:end]
    path.write_text("".join(lines).rstrip("\n") + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="test files to look at (default: every tests/test_*.py)")
    parser.add_argument("--write", action="store_true", help="remove the removable functions and delete empty files")
    parser.add_argument("--report", help="also save the report to this path")
    args = parser.parse_args(argv)

    lib = LIB()
    files = [Path(f).resolve() for f in args.files] or sorted(TESTS.glob("test_*.py"))
    skip = {"test_generator_contract.py", *(f"test_generator_contract_exam_part_{i}.py" for i in range(8))}
    files = [f for f in files if f.name not in skip]
    reports = [analyse_file(f, lib) for f in files]
    reports = [r for r in reports if r.removable or r.kept]
    used = importers([r.path for r in reports])

    out: list[str] = []
    fam_total = collections.Counter()
    fam_removed = collections.Counter()
    fam_kept = collections.Counter()
    delete, keep_files, partial = [], [], []
    reasons_count: collections.Counter = collections.Counter()
    for r in reports:
        for node in r.removable:
            fam_removed[family_of(node.name)] += 1
            fam_total[family_of(node.name)] += 1
        for name, verdict, fam in r.kept:
            fam_kept[fam] += 1
            fam_total[fam] += 1
            for reason in verdict.reasons[:1]:
                reasons_count[(fam, re.sub(r"\(.*?\)|'[^']*'|\d+", "#", reason)[:80])] += 1
        if r.removable and not r.leftover_tests:
            if used.get(r.path.stem):
                keep_files.append(
                    (r, f"would be empty but imported by {', '.join(sorted(set(used[r.path.stem]))[:3])}")
                )
            else:
                delete.append(r)
        elif r.removable:
            partial.append(r)
    n_removable = sum(len(r.removable) for r in reports)
    stricter_files = [r for r in reports if r.stricter]
    relies_on_escalation = sum(1 for r in reports for why in r.covered_by.values() if "newshape" in why)

    out.append("prune_generic_tests dry run" if not args.write else "prune_generic_tests --write")
    out.append(f"files scanned with a family function: {len(reports)}")
    out.append(
        f"family functions: {sum(fam_total.values())}   removable: {n_removable}   kept: {sum(fam_kept.values())}"
    )
    out.append(f"files deletable (no test left, not imported): {len(delete)}")
    out.append(f"files that lose functions but keep other tests: {len(partial)}")
    out.append(f"files that would be empty but are imported elsewhere: {len(keep_files)}")
    out.append(f"files kept because a copy is STRICTER than the shared test: {len(stricter_files)}")
    out.append(
        f"removable functions that lean on test_tier_escalation for 'exam adds a shape' (a different baseline "
        f"from the copy's own hard-tier set): {relies_on_escalation}"
    )
    out.append("")
    out.append("per family (removable / kept):")
    for fam in FAMILY_PATTERNS:
        out.append(f"  {fam:16s} {fam_removed[fam]:4d} / {fam_kept[fam]:4d}")
    out.append("")
    out.append("why functions stay (first reason, grouped):")
    for (fam, reason), n in sorted(reasons_count.items(), key=lambda kv: (-kv[1], kv[0])):
        out.append(f"  {n:4d}  {fam:16s} {reason}")
    out.append("")
    out.append("== STRICTER than the shared test: these keep their check ==")
    for r in stricter_files:
        for name, verdict, fam in r.stricter:
            out.append(f"  {r.path.name}::{name}: " + "; ".join(x for x in verdict.reasons if "stricter" in x))
    out.append("")
    out.append("== files deletable ==")
    for r in delete:
        out.append(f"  {r.path.name}  ({', '.join(r.removed_names)})")
    out.append("")
    out.append("== files that would be empty but are imported ==")
    for r, why in keep_files:
        out.append(f"  {r.path.name}: {why}")
    out.append("")
    out.append("== functions removable, files that keep other tests ==")
    for r in partial:
        out.append(f"  {r.path.name}: remove {', '.join(r.removed_names)}; keeps {', '.join(r.leftover_tests[:6])}")
    out.append("")
    out.append("== functions kept (file::name: reasons) ==")
    for r in reports:
        for name, verdict, fam in r.kept:
            out.append(f"  {r.path.name}::{name}: " + " | ".join(verdict.reasons[:3]))
    text = "\n".join(out) + "\n"
    print(text if not args.report else "\n".join(out[:20]))
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")

    if args.write:
        for r in reports:
            if not r.removable:
                continue
            if r in delete:
                r.path.unlink()
            else:
                remove_functions(r.path, r.removable)
        print(f"wrote: {n_removable} functions removed, {len(delete)} files deleted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
