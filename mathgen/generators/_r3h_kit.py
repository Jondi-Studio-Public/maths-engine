"""Shared helpers for the `r3h_*` generators (specialist differential equations and Methods applied calculus).

Everything here is plain Python and SymPy; nothing is registered as a topic.

* `plain` writes a SymPy expression the way a learner types it (``x e^x (ln(x) - 1)``, ``e^(2x)/3``); the grader reads
  that spelling back, which the tests check for every canonical answer.
* `say_marked` writes a sentence and its LaTeX together: the maths sits between « and ».
* `expr_options` and `number_options` turn named wrong answers into tagged `Distractor` lists, dropping any wrong
  answer that is secretly equal to the right one.
* `same_function` checks two expressions numerically, a route independent of SymPy's simplifier.
* `number_part`, `expr_part` and `fraction_part` build the dicts of a multi-part exam chain.
"""

from __future__ import annotations

import cmath
import functools
import math
import re
from fractions import Fraction

import sympy

from ..answers import Distractor
from ..core import _escape_prose, attempts, pick_distractors

X = sympy.Symbol("x")
T = sympy.Symbol("t")
#: A constant that may stay symbolic in an answer (a decay constant ``k``); numerical checks give it a probe value.
PARAM = sympy.Symbol("k")
_PARAM_PROBE = 0.37

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
    s = re.sub(r"(?<![A-Za-z])([xytk])\*([xytk])(?![A-Za-z(])", r"\1\2", s)
    s = re.sub(r"(?<=[A-Za-z0-9)])\*(?=\()", "*", s)
    s = re.sub(r"(?<=[A-Za-z0-9)])\*(?=[A-Za-z])", " ", s)
    s = s.replace("*", " ") if "*(" not in s else s
    return re.sub(
        r"(?<![A-Za-z0-9^)])(\d*)([A-Za-z])\^\(-(\d+)\)", lambda m: f"{m.group(1) or 1}/{m.group(2)}^{m.group(3)}", s
    )


def machine(expr) -> str:
    """`expr` in SymPy syntax: the form stored as a question's `answer`."""
    return sympy.sstr(sympy.sympify(expr))


_FUNCTIONS = r"(arcsin|arccos|arctan|sin|cos|tan|sec|csc|cot|ln)"


def tex(plain_text: str) -> str:
    """LaTeX for the plain maths this package writes."""
    s = plain_text.replace("{", r"\{").replace("}", r"\}")
    s = re.sub(r"d([A-Za-z])/d([xyt])", r"\\frac{\\mathrm{d}\1}{\\mathrm{d}\2}", s)
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


@functools.lru_cache(maxsize=8192)
def _compile(expr, var):
    """`expr` as a Python function of `var` over complex numbers (an independent, fast numerical route)."""
    return sympy.lambdify((var, PARAM), sympy.sympify(expr), modules=[_NUMERIC_NAMES, "cmath"])


def real_on(expr, var, points, bound: float = 1e4) -> bool:
    """Whether `expr` is a finite real number (within `bound`) at every point."""
    try:
        values = _values(expr, var, points)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError, NameError, KeyError):
        return False
    return all(
        math.isfinite(v.real) and abs(v.imag) < 1e-10 * max(1.0, abs(v.real)) and abs(v.real) < bound for v in values
    )


@functools.lru_cache(maxsize=1024)
def antiderivative(expr, var=X):
    """The antiderivative of `expr` with no constant term (cached: SymPy's integrator is the slow step)."""
    return sympy.integrate(expr, var)


def _values(expr, var, points):
    fn = _compile(expr, var)
    return [complex(fn(float(p), _PARAM_PROBE)) for p in points]


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


def small_fraction(value, max_den: int = 6, max_num: int = 40) -> bool:
    """Whether `value` is a rational whose denominator and numerator are small enough to read comfortably."""
    v = sympy.nsimplify(value)
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
        if same_function(parsed, expr):
            return text
    return None


def tidy(expr):
    """`expr` expanded and simplified, for a short wrong answer whose raw form is cluttered."""
    expanded = sympy.expand(expr)
    if len(str(expanded)) > 110:
        # SymPy's simplifier is the slow step and rarely shortens a long sum of exponentials or logarithms.
        return expanded
    trig = expanded.has(sympy.sin, sympy.cos, sympy.tan, sympy.cot, sympy.sec, sympy.csc)
    if trig:
        # The trigonometric simplifiers cost far more than the rest of a question, so only a short sum gets them.
        options = [expanded] if len(str(expanded)) > 50 else [expanded, sympy.trigsimp(expanded)]
        return min((o for o in options if not o.has(sympy.pi)), key=lambda o: len(plain(o)))
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


# ---------------------------------------------------------------------------------------------------------------
# Exam-chain parts


class Redraw(Exception):
    """A drawn question is unsuitable (too few distractors, an ambiguous rounding): draw again."""


def build_with_redraws(rng, build, tries: int = 80):
    """The first dict `build(rng)` returns without raising `Redraw`."""
    for _ in attempts(tries, "a question with enough distractors"):
        try:
            return build(rng)
        except Redraw:
            continue
    raise AssertionError("unreachable")


def need(distractors, minimum: int = 2):
    if len(distractors) < minimum:
        raise Redraw
    return distractors


def number_part(
    prompt, value, dp, marks, working, wrongs, *, follow_through=None, sensible_range=None, gap=None, **slots
):
    """A `number` part stated to `dp` places. `wrongs` is a list of ``(misconception id, value)``."""
    value = float(value)
    answer = round(value, dp)
    part = dict(
        **prompt_part(prompt, **slots),
        answer_kind="number",
        answer=answer,
        answer_text=f"{answer:.{dp}f}",
        marks=marks,
        working=list(working),
        distractors=need(number_options(value, dp, wrongs, gap=gap)),
    )
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range:
        part["sensible_range"] = sensible_range
    return part


def expr_part(prompt, expr, marks, working, wrongs, *, follow_through=None, var=X, options=None, **slots):
    """An `expression` part. `wrongs` is a list of ``(misconception id, expression)``; `options` is a ready list of
    distractors that replaces them."""
    part = dict(
        **prompt_part(prompt, **slots),
        answer_kind="expression",
        answer=machine(expr),
        answer_text=plain(expr),
        marks=marks,
        working=list(working),
        distractors=need(options if options is not None else expr_options(expr, wrongs, 4, var)),
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


def fraction_part(prompt, value, marks, working, wrongs, *, follow_through=None, **slots):
    """A `fraction` part (an exact rational). `wrongs` is a list of ``(misconception id, value)``."""
    from ..answers import exact_value
    from ..core import rational_text

    value = Fraction(int(sympy.Rational(value).p), int(sympy.Rational(value).q))
    part = dict(
        **prompt_part(prompt, **slots),
        answer_kind="fraction",
        answer=exact_value(value),
        answer_text=rational_text(value),
        marks=marks,
        working=list(working),
        distractors=need(fraction_options(value, wrongs, 4)),
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


def frac_text(value) -> str:
    value = sympy.Rational(value)
    return str(value.p) if value.q == 1 else f"{value.p}/{value.q}"


def tidy_form(expr):
    """`expr`, rewritten with `neat` only when it has a negative power to remove (the rewrite is the slow step)."""
    expr = sympy.sympify(expr)
    return neat(expr) if "^(-" in plain(expr) else expr
