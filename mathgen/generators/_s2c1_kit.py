"""Shared helpers for the second-year ordinary-differential-equation skills (the `s2c1_*` generators).

Everything here is plain Python and SymPy; nothing is registered as a topic.

* `plain` writes a SymPy expression the way a learner types it (``x e^x (ln(x) - 1)``, ``e^(2x)/3``); the grader reads
  that spelling back, which the tests check for every canonical answer.
* `say_marked` writes a sentence and its LaTeX together: the maths sits between « and ».
* `expr_options` and `number_options` turn named wrong answers into tagged `Distractor` lists, dropping any wrong
  answer that is secretly equal to the right one.
* `solves_ode` and `same_function` check a candidate solution numerically, a route independent of how it was built.
* `solve_constants` fits ``y = c1 f1 + c2 f2 + yp`` to ``y(x0)`` and ``y'(x0)`` exactly.
"""

from __future__ import annotations

import cmath
import functools
import math
import re
from fractions import Fraction

import sympy

from ..answers import Distractor
from ..core import _escape_prose, check, pick_distractors

X = sympy.Symbol("x")
T = sympy.Symbol("t")

# ---------------------------------------------------------------------------------------------------------------
# Writing expressions


def _matching(text: str, open_at: int) -> int:
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"unbalanced brackets in {text!r}")


@functools.lru_cache(maxsize=8192)
def plain(expr) -> str:
    """`expr` as a learner types it: ``2x``, ``x^2``, ``e^(2x)``, ``ln(x)``, ``x e^x``, ``sqrt(x)``."""
    s = sympy.sstr(sympy.sympify(expr))
    s = s.replace("**", "^")
    s = re.sub(r"(?<![A-Za-z])E(?![A-Za-z(])", "e", s)
    for src, dst in (("log(", "ln("), ("asin(", "arcsin("), ("acos(", "arccos("), ("atan(", "arctan(")):
        s = re.sub(r"(?<![A-Za-z])" + re.escape(src), dst, s)
    while "exp(" in s:
        i = s.rfind("exp(")
        j = _matching(s, i + 3)
        inner = s[i + 4 : j]
        s = s[:i] + ("e^" + inner if re.fullmatch(r"[A-Za-z]|\d+", inner) else "e^(" + inner + ")") + s[j + 1 :]
    s = re.sub(r"(?<![\^\d.])(\d+)\*(?=[A-Za-z(])", r"\1", s)
    s = re.sub(r"(?<![A-Za-z])([xy])\*([xy])(?![A-Za-z(])", r"\1\2", s)
    s = re.sub(r"(?<=[A-Za-z0-9)])\*(?=\()", "*", s)
    s = re.sub(r"(?<=[A-Za-z0-9)])\*(?=[A-Za-z])", " ", s)
    s = s.replace("*", " ") if "*(" not in s else s
    return re.sub(
        r"(?<![A-Za-z0-9^)])(\d*)([A-Za-z])\^\(-(\d+)\)", lambda m: f"{m.group(1) or 1}/{m.group(2)}^{m.group(3)}", s
    )


@functools.lru_cache(maxsize=8192)
def machine(expr) -> str:
    """`expr` in SymPy syntax: the form stored as a question's `answer`."""
    return sympy.sstr(sympy.sympify(expr))


_FUNCTIONS = r"(arcsin|arccos|arctan|sin|cos|tan|sec|csc|cot|ln)"


def tex(plain_text: str) -> str:
    """LaTeX for the plain maths this package writes."""
    s = plain_text.replace("{", r"\{").replace("}", r"\}")
    # sqrt(...) with balanced brackets
    while "sqrt(" in s:
        i = s.rfind("sqrt(")
        j = _matching(s, i + 4)
        s = s[:i] + r"\sqrt{" + s[i + 5 : j] + "}" + s[j + 1 :]
    s = re.sub(r"(?<![A-Za-z\\])d([xyt])(?![A-Za-z(])", r"\\,\\mathrm{d}\1", s)
    s = re.sub(r"\^\(([^()]*)\)", lambda m: "^{" + m.group(1) + "}", s)
    s = re.sub(r"\^(-?\d+)", r"^{\1}", s)
    s = re.sub(r"\^([A-Za-z])", r"^{\1}", s)
    s = re.sub(r"(?<![A-Za-z\\])" + _FUNCTIONS + r"(?=[( ^])", lambda m: "\\" + m.group(1), s)
    for symbol, command in (("ω", r"\omega "), ("π", r"\pi "), ("θ", r"\theta "), ("≤", r"\le "), ("≥", r"\ge ")):
        s = s.replace(symbol, command)
    s = s.replace("²", "^{2}").replace("³", "^{3}")
    return s


_MARKED = re.compile(r"«(.*?)»", re.S)


def say_marked(template: str) -> tuple[str, str]:
    """``(text, latex)`` for a sentence whose maths fragments sit between « and »."""
    pieces = _MARKED.split(template)
    text: list[str] = []
    latex: list[str] = []
    for i, piece in enumerate(pieces):
        if not piece:
            continue
        if i % 2:
            text.append(piece)
            latex.append(tex(piece) + "{}")
        else:
            text.append(piece)
            latex.append(r"\text{" + _escape_prose(piece) + "}")
    return "".join(text), "".join(latex)


def prompt_part(template: str, **slots) -> dict:
    """The `prompt` and `prompt_latex` of a part."""
    text, latex = say_marked(template.format(**slots) if slots else template)
    return dict(prompt=text, prompt_latex=latex)


# ---------------------------------------------------------------------------------------------------------------
# Independent numerical checks

_POINTS = (Fraction(7, 10), Fraction(13, 10), Fraction(19, 10))

_NUMERIC_NAMES = {
    "sec": lambda z: 1 / cmath.cos(z),
    "csc": lambda z: 1 / cmath.sin(z),
    "cot": lambda z: 1 / cmath.tan(z),
    "Abs": abs,
}


@functools.lru_cache(maxsize=2048)
def _compile(expr, var):
    """`expr` as a Python function of `var` over complex numbers (an independent, fast numerical route)."""
    return sympy.lambdify(var, sympy.sympify(expr), modules=[_NUMERIC_NAMES, "cmath"])


def _values(expr, var, points):
    fn = _compile(expr, var)
    return [complex(fn(float(p))) for p in points]


def same_function(left, right, var=X, points=_POINTS) -> bool:
    """Whether two expressions agree at every probe point (a numerical test, independent of SymPy's simplifier)."""
    try:
        lv, rv = _values(left, var, points), _values(right, var, points)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError, NameError, KeyError):
        return False
    for a, b in zip(lv, rv):
        if not (math.isfinite(a.real) and math.isfinite(b.real) and math.isfinite(a.imag) and math.isfinite(b.imag)):
            return False
        if abs(a - b) > 1e-9 * max(1.0, abs(a), abs(b)):
            return False
    return True


#: The tests switch this on and sweep many seeds, so every drawn question is still verified by a second route (the
#: numerical checks below, independent of how the closed form was built). At serving time the closed forms are
#: trusted and the checks are skipped, which keeps `make()` fast.
SECOND_ROUTE = False


def second_route() -> bool:
    return SECOND_ROUTE


def agree(left, right, var=X, points=_POINTS) -> bool:
    """Second-route check: whether two expressions agree numerically (always true when the second route is off)."""
    return same_function(left, right, var, points) if SECOND_ROUTE else True


def solves_ode(y, coefficients, forcing, var=X, points=_POINTS) -> bool:
    """Whether ``a2 y'' + a1 y' + a0 y = forcing`` holds at every probe point; `coefficients` is (a2, a1, a0), each
    a constant or an expression in `var`."""
    if not SECOND_ROUTE:
        return True
    a2, a1, a0 = (sympy.sympify(c) for c in coefficients)
    y = sympy.sympify(y)
    lhs = a2 * sympy.diff(y, var, 2) + a1 * sympy.diff(y, var) + a0 * y
    return same_function(lhs, forcing, var, points)


def _exact(value):
    """`value` as an exact number: itself when already rational, else rationalised."""
    value = sympy.sympify(value)
    return value if value.is_Rational else sympy.nsimplify(value)


def solve_constants(f1, f2, particular, x0, y0, v0, var=X):
    """The exact constants c1, c2 with ``y = c1 f1 + c2 f2 + particular`` meeting ``y(x0) = y0`` and ``y'(x0) = v0``
    (Cramer's rule on the 2 x 2 system)."""
    sub = {var: x0}
    f1, f2, p = sympy.sympify(f1), sympy.sympify(f2), sympy.sympify(particular)
    a, b = _exact(f1.subs(sub)), _exact(f2.subs(sub))
    c, d = _exact(sympy.diff(f1, var).subs(sub)), _exact(sympy.diff(f2, var).subs(sub))
    det = a * d - b * c
    check(det != 0, "the two basis functions are dependent at the initial point")
    r1 = _exact(sympy.sympify(y0) - p.subs(sub))
    r2 = _exact(sympy.sympify(v0) - sympy.diff(p, var).subs(sub))
    return (r1 * d - b * r2) / det, (a * r2 - c * r1) / det


def small_fraction(value, max_den: int = 6, max_num: int = 40) -> bool:
    """Whether `value` is a rational whose denominator and numerator are small enough to read comfortably."""
    v = sympy.sympify(value)
    if not v.is_Rational:
        v = sympy.nsimplify(v)
    return bool(v.is_Rational) and int(v.q) <= max_den and abs(int(v.p)) <= max_num


# ---------------------------------------------------------------------------------------------------------------
# Distractors


def readable(expr):
    """The learner-style spelling of `expr` that the answer parser reads back as the same function, or None.

    The parser refuses an expression it judges too costly, which happens to a long sum of repeated
    logarithms and exponentials; a factored spelling of the same function is often short enough."""
    from ..grading.typed import _parse_expression

    expr = sympy.sympify(expr)
    tries = [expr, sympy.factor_terms(expr), sympy.factor_terms(sympy.expand(expr))]
    for candidate in tries:
        text = plain(candidate)
        try:
            parsed = _parse_expression(text)
        except Exception:  # noqa: BLE001 - an unreadable spelling is simply not used
            parsed = None
        if not isinstance(parsed, sympy.Basic):
            continue
        parsed = parsed.xreplace({sympy.Symbol("e"): sympy.E})
        if agree(parsed, expr):
            return text
    return None


@functools.lru_cache(maxsize=4096)
def tidy(expr):
    """`expr` expanded and simplified, for a short wrong answer whose raw form is cluttered."""
    expanded = sympy.expand(expr)
    if len(str(expanded)) > 110:
        # SymPy's simplifier is the slow step and rarely shortens a long sum of exponentials or logarithms.
        return expanded
    trig = expanded.has(sympy.sin, sympy.cos, sympy.tan, sympy.cot, sympy.sec, sympy.csc)
    if trig:
        # The trigonometric simplifiers cost far more than the rest of a question, so only a short sum gets them.
        return expanded
    else:
        # No trigonometry, so skip the (slow) trigonometric simplifier.
        options = [expanded, sympy.powsimp(expanded)]
    if expanded.has(sympy.sin, sympy.cos):
        options.append(sympy.trigsimp(expanded))
    options = [o for o in options if not o.has(sympy.pi)]
    return min(options, key=lambda o: len(plain(o)))


def expr_options(answer, wrongs, n: int = 4, var=X, simplify: bool = False) -> list[Distractor]:
    """Tagged distractors for an expression answer. `wrongs` is a list of ``(misconception id, expression)`` in
    priority order. A wrong answer equal to the right one, or to an earlier option, is dropped."""
    kept: list[tuple[str, object]] = []
    for tag, w in wrongs:
        if w is None:
            continue
        w = tidy(w) if simplify else sympy.sympify(w)
        if w == 0 or same_function(w, answer, var):
            continue
        if any(same_function(w, k, var) for _, k in kept):
            continue
        kept.append((tag, w))
    candidates = [Distractor(plain(w), tag, machine(w)) for tag, w in kept]
    return pick_distractors(plain(answer), candidates, n)


def number_options(answer: float, dp: int, wrongs, n: int = 4, gap: float | None = None) -> list[Distractor]:
    """Tagged distractors for a number answer stated to `dp` places (every option has the same number of places).
    `wrongs` is a list of ``(misconception id, value)``. A value whose rounding lands within two units of the last
    place of the answer, or of an earlier option, is dropped."""
    unit = 10.0**-dp
    gap = 2 * unit if gap is None else gap
    out: list[Distractor] = []
    seen = [round(answer, dp)]
    for tag, value in wrongs:
        if value is None or not math.isfinite(value) or abs(value) > 1e6:
            continue
        r = round(float(value), dp)
        if any(abs(r - s) < gap for s in seen):
            continue
        seen.append(r)
        text = f"{r:.{dp}f}"
        out.append(Distractor(text, tag, r))
    return pick_distractors(f"{answer:.{dp}f}", out, n)


def fraction_options(answer, wrongs, n: int = 4) -> list[Distractor]:
    """Tagged distractors for an exact rational answer (`fraction` kind)."""
    from ..answers import exact_value
    from ..core import rational_text

    ans = Fraction(int(sympy.Rational(answer).p), int(sympy.Rational(answer).q))
    out = []
    kept: list[Fraction] = []
    for tag, value in wrongs:
        if value is None:
            continue
        v = sympy.nsimplify(value)
        if not v.is_Rational:
            continue
        f = Fraction(int(v.p), int(v.q))
        if f == ans or any(f == seen for seen in kept):
            continue
        kept.append(f)
        out.append(Distractor(rational_text(f), tag, exact_value(f)))
    return pick_distractors(rational_text(ans), out, n)


@functools.lru_cache(maxsize=4096)
def neat(expr):
    """`expr` in the shortest of a few equivalent spellings, avoiding negative powers (``x^(-2)``) where a quotient
    reads better. The caller checks the result against the original numerically."""
    expr = sympy.sympify(expr)
    options = [expr]
    for fn in (sympy.together, sympy.cancel):
        try:
            options.append(fn(expr))
        except (sympy.PolynomialError, ValueError, TypeError):
            continue
    clean = [o for o in options if "^(-" not in plain(o)] or options
    return min(clean, key=lambda o: len(plain(o)))
