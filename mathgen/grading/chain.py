"""Typed-step proofs: a chain of expressions, each line checked against the one before it.

A proof whose working is a chain of equal expressions ("show that a^2 + b^2 - 2ab = (a - b)^2") is marked by the
engine, line by line. The learner types one expression per line (or writes them joined by ``=``); the grader checks

* the first line is the stated start, written the way the question writes it (same structure, not just the same
  value: ``a^2 + b^2 - 2ab`` and ``(a - b)^2`` are equal as values, and the point of the proof is the second form);
* every line is equal in value to the line before it, on the domain the question names for its symbols;
* the last line is the stated target, in the target's form.

``Question.answer`` is the chain-proof marking dict of ``mathgen/generators/_chain_proof_kit.py``:

    {"start", "target", "symbols": {name: kind}, "reference_lines", ["given": {name: expression}], ...}

``kind`` is ``real``, ``positive``, ``integer`` or ``nonzero``. ``given`` is a hypothesis the proof is working
under (``{"p": "q*sqrt(2)"}`` for "suppose p = q sqrt 2"): every line is read with it substituted, so two lines
that are equal only under the hypothesis count as equal. Equality in value is decided by exact rational probe
points, never by evaluating anything the learner wrote beyond sympy's restricted parser (``safe_parse``).
"""

from __future__ import annotations

import functools
import re

import sympy

from .safe_parse import TooCostly, parse_expression_safely, sympify_safely
from .typed import _TRANSFORMATIONS, GradeResult, _multi_part_aware, _sympy_text

MAX_LINES = 12
MAX_LINE_LENGTH = 300
MIN_POINTS = 6

_GRIDS = {
    "real": ("-7/3", "-2", "-1/2", "1/3", "3/2", "2", "5", "11/4", "-5", "7/5"),
    "nonzero": ("-7/3", "-2", "-1/2", "1/3", "3/2", "2", "5", "11/4", "-5", "7/5"),
    "positive": ("1/3", "1/2", "2/3", "3/2", "2", "5/2", "3", "7/3", "5", "11/4"),
    "integer": ("-5", "-3", "-2", "2", "3", "4", "6", "7", "9", "12"),
}
SYMBOL_KINDS = frozenset(_GRIDS)
_RELATION_RE = re.compile(r"[<>≤≥≠]|!=")


def _split_lines(given: str) -> list[str]:
    """The lines of a reply: one per line break, and a line written as ``A = B = C`` is three lines."""
    lines: list[str] = []
    for raw in given.replace("−", "-").split("\n"):
        if _RELATION_RE.search(raw):
            return []
        for piece in raw.split("="):
            piece = piece.strip()
            if piece:
                lines.append(piece)
    return lines


def _spell_roots(line: str) -> str:
    """``√2``, ``√x`` and ``√(...)`` as ``sqrt(...)``, with the product sign a root written after a factor leaves out
    (``q√2`` is ``q*sqrt(2)``). The shared reader splits ``(q√(2))`` into single letters, so roots are spelled here."""
    text = re.sub(r"√\s*(\d+(?:\.\d+)?|[A-Za-z])", r"sqrt(\1)", line)
    text = re.sub(r"√\s*\(", "sqrt(", text)
    return re.sub(r"([A-Za-z0-9)])\s*sqrt\(", r"\1*sqrt(", text)


@functools.lru_cache(maxsize=2048)
def _parse(line: str, *, evaluate: bool):
    if not line or len(line) > MAX_LINE_LENGTH:
        return None
    text = _spell_roots(line)
    candidate = _sympy_text(text)
    if candidate is None:
        return None
    try:
        parsed = parse_expression_safely(candidate, transformations=_TRANSFORMATIONS, evaluate=evaluate)
    except TooCostly:
        return None
    except (sympy.SympifyError, SyntaxError, TypeError, ValueError, IndexError, AttributeError):
        return None
    except Exception:  # noqa: BLE001 - tokenizer errors and the like: an unreadable line, not a crash
        return None
    return parsed if isinstance(parsed, sympy.Expr) else None


def _tree(expr):
    """`expr` as a nested tuple that ignores order and bracketing of sums and products, nothing else."""
    if expr.is_Atom:
        return sympy.srepr(expr)
    if isinstance(expr, sympy.Abs):  # |x - y| and |y - x| are the same bars; sympy flips the argument's sign itself
        inner = sympy.expand(expr.args[0])
        return ("Abs", min(sympy.srepr(inner), sympy.srepr(sympy.expand(-inner))))
    kids = [_tree(a) for a in expr.args]
    if isinstance(expr, (sympy.Add, sympy.Mul)):
        flat = []
        for arg, kid in zip(expr.args, kids):
            if isinstance(arg, type(expr)):
                flat.extend(kid[1])
            else:
                flat.append(kid)
        return (type(expr).__name__, tuple(sorted(flat, key=repr)))
    return (type(expr).__name__, tuple(kids))


def same_form(left: str, right: str) -> bool:
    """Whether two typed expressions have the same structure, ignoring the order of terms and factors and the use of
    brackets that only regroup: ``2ab + c`` and ``c + 2 * a * b`` match; ``(a - b)^2`` and ``a^2 - 2ab + b^2`` do not."""
    a, b = _parse(left, evaluate=False), _parse(right, evaluate=False)
    return a is not None and b is not None and _tree(a) == _tree(b)


def _probe_points(names: list[str], kinds: dict[str, str]):
    for p in range(len(_GRIDS["real"])):
        env = {}
        for i, name in enumerate(names):
            grid = _GRIDS[kinds.get(name, "real")]
            env[name] = sympy.Rational(grid[(p + 3 * i) % len(grid)])
        yield env


def _value(expr, env):
    try:
        value = complex(sympy.N(expr.xreplace(env), 15))
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    return value if value == value and abs(value) != float("inf") else None


def lines_equal(left, right, kinds: dict[str, str], given: dict | None = None) -> bool:
    """Whether two parsed expressions are the same value at every probe point of the symbols' domains (at least
    `MIN_POINTS` points where both are defined; a point where only one is defined is a difference)."""
    return _equal(left, right, tuple(sorted(kinds.items())), tuple(sorted((given or {}).items())))


@functools.lru_cache(maxsize=4096)
def _equal(left, right, kinds_items: tuple, given_items: tuple) -> bool:
    kinds = dict(kinds_items)
    sub = {sympy.Symbol(k): v for k, v in given_items}
    left, right = left.xreplace(sub), right.xreplace(sub)
    allowed = set(kinds)
    names = sorted({str(s) for s in left.free_symbols | right.free_symbols})
    if any(n not in allowed for n in names):
        return False
    usable = 0
    for env in _probe_points(names, kinds):
        point = {sympy.Symbol(n): v for n, v in env.items()}
        a, b = _value(left, point), _value(right, point)
        if a is None and b is None:
            continue
        if a is None or b is None or abs(a - b) > 1e-9 * (1 + abs(a) + abs(b)):
            return False
        usable += 1
    return usable >= MIN_POINTS


def _spec(question):
    spec = question.answer
    if not isinstance(spec, dict) or not spec.get("start") or not spec.get("target"):
        return None
    symbols = spec.get("symbols") or {}
    if not isinstance(symbols, dict) or any(k not in SYMBOL_KINDS for k in symbols.values()):
        return None
    return spec


def grade_step_chain(question, given: str) -> GradeResult:
    """Mark a typed chain. Wrong replies carry a `note` saying which line failed (never the answer)."""
    canonical = question.answer_text
    spec = _spec(question)

    def wrong(note=None, normalised=None):
        return GradeResult(
            correct=False,
            canonical=canonical,
            given_normalised=normalised,
            note=note,
            note_kind="chain_step" if note else None,
        )

    if spec is None or not isinstance(given, str):
        return wrong()
    lines = _split_lines(given)
    if not lines:
        return wrong()
    if len(lines) > MAX_LINES:
        return wrong(f"Use at most {MAX_LINES} lines.")
    kinds = dict(spec.get("symbols") or {})
    hypothesis = {}
    for name, text in (spec.get("given") or {}).items():
        hypothesis[name] = sympify_safely(text)
    kinds_all = {**kinds, **{k: kinds.get(k, "real") for k in hypothesis}}
    parsed = []
    for i, line in enumerate(lines, 1):
        expr = _parse(line, evaluate=True)
        if expr is None:
            return wrong(f"Line {i} could not be read as an expression.", normalised=None)
        parsed.append(expr)
    normalised = " = ".join(str(e) for e in parsed)
    if len(lines) < 2:
        return wrong("Show at least one step: write the start and the result on separate lines.", normalised)
    if not same_form(lines[0], spec["start"]):
        return wrong("The first line is not the expression the working starts from.", normalised)
    for i in range(1, len(parsed)):
        if not lines_equal(parsed[i - 1], parsed[i], kinds_all, hypothesis):
            return wrong(f"Line {i + 1} is not equal to line {i}.", normalised)
    if not same_form(lines[-1], spec["target"]):
        return wrong("The last line is not in the form the question asks you to reach.", normalised)
    return GradeResult(correct=True, canonical=canonical, given_normalised=normalised)


grade_step_chain = _multi_part_aware(grade_step_chain)

__all__ = ["grade_step_chain", "lines_equal", "same_form", "SYMBOL_KINDS"]
