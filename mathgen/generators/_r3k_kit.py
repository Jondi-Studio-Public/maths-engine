"""Shared helpers for the second-year probability and statistics generators (`r3k_*`).

Each generator in the packet works with exact rationals and closed forms, so the hand route is plain `Fraction`
arithmetic or SymPy, and every generator re-derives its answer a second way (a direct sum, an integral, a simulation
of the defining formula, a numerical derivative). Nothing here draws random numbers.

* answers: `frac_fields`, `expr_fields`, `num_fields` build the answer fields of a `fraction`, `expression` and
  `number` answer (the display text, the machine form and the LaTeX);
* distractors: `frac_distractors` and `expr_distractors` turn `(value, misconception)` pairs into tagged
  `Distractor`s that the matching grader rejects, with no repeats; `num_distractors` does the same for decimals
  that keep the answer's number of places;
* `need` fails the draw loudly when too few distractors survive; `scenario` picks a plain-English setting.
"""

from __future__ import annotations

import functools
import re
from fractions import Fraction

import sympy

from ..answers import Distractor, exact_value
from ..core import VerificationError, check, latex_answer, rational_text

#: The slow independent cross-checks (numerical quadrature, symbolic simplification of a second derivation) run only
#: when this is true. The tests switch it on and sweep many seeds, so every generator is still verified by a second
#: route; at serving time the exact first route (closed forms in exact fractions) is what runs.
SECOND_ROUTE = False


def second_route() -> bool:
    return SECOND_ROUTE


_SAMPLE_POINTS = (Fraction(7, 10), Fraction(13, 10), Fraction(19, 20), Fraction(23, 10), Fraction(1, 3))


def rat(value) -> sympy.Rational:
    """`value` (an int, a Fraction or a SymPy Rational) as a SymPy Rational."""
    if isinstance(value, sympy.Basic):
        return sympy.nsimplify(value)
    value = Fraction(value)
    return sympy.Rational(value.numerator, value.denominator)


def fr(value) -> Fraction:
    """`value` (an int, a Fraction or a SymPy number) as a Fraction."""
    if isinstance(value, (int, Fraction)):
        return Fraction(value)
    if isinstance(value, sympy.Rational):
        return Fraction(int(value.p), int(value.q))
    value = sympy.nsimplify(value)
    check(value.is_Rational, f"{value} is not rational")
    return Fraction(int(value.p), int(value.q))


# -- answer fields --------------------------------------------------------------------------------------------


def frac_fields(value) -> dict:
    """The answer fields of an exact rational (`fraction` kind)."""
    value = fr(value)
    text = rational_text(rat(value))
    return dict(answer_kind="fraction", answer=exact_value(value), answer_text=text, answer_latex=latex_answer(text))


def pretty(expr) -> str:
    """A SymPy expression written the way a learner types it: `3*x^2/2`, `exp(-2*t)`, `ln(2)`."""
    return sympy.sstr(expr).replace("**", "^").replace("log(", "ln(")


def expr_fields(expr) -> dict:
    """The answer fields of an exact expression (`expression` kind); the machine form keeps `log`, the text says `ln`."""
    expr = sympy.sympify(expr)
    return dict(
        answer_kind="expression",
        answer=sympy.sstr(expr).replace("**", "^"),
        answer_text=pretty(expr),
        answer_latex=sympy.latex(expr, ln_notation=True),
    )


def num_fields(value: float, places: int) -> dict:
    """The answer fields of a decimal answer stated to `places` decimal places (`number` kind)."""
    text = f"{value:.{places}f}"
    return dict(answer_kind="number", answer=float(text), answer_text=text, answer_latex=text)


# -- distractors ----------------------------------------------------------------------------------------------


def frac_distractors(
    true, candidates, *, limit: int = 3, positive: bool = False, at_most: Fraction | None = None, avoid=()
):
    """Tagged exact-rational distractors from `(value, misconception)` pairs; none equals `true`.

    `avoid` is a set of values that must not appear (the follow-through values of an earlier part's distractors);
    `positive` drops a non-positive value; `at_most` drops a value above that bound (1 for a probability).
    """
    true = Fraction(true)
    seen = {true} | {Fraction(v) for v in avoid}
    pool = []
    for value, tag in candidates:
        if value is None:
            continue
        value = fr(value)
        if value in seen or (positive and value <= 0) or (at_most is not None and value > at_most):
            continue
        seen.add(value)
        pool.append((value, tag))
    chosen = pool[:limit]
    # the answer must not be the only option of its shape (a whole number among fractions, or the reverse)
    same = [item for item in pool if (item[0].denominator == 1) == (true.denominator == 1)]
    if same and not any(item in same for item in chosen):
        chosen = chosen[: limit - 1] + [same[0]]
    return [Distractor(rational_text(rat(v)), tag, exact_value(v)) for v, tag in chosen]


def same_function(left, right, symbols=()) -> bool:
    """Whether two expressions agree as functions (compared exactly at a spread of points for each symbol)."""
    left, right = sympy.sympify(left), sympy.sympify(right)
    free = sorted(left.free_symbols | right.free_symbols | set(symbols), key=str)
    if not free:
        return sympy.simplify(left - right) == 0
    for k, point in enumerate(_SAMPLE_POINTS):
        values = {s: rat(point) + sympy.Rational(i + k, 7) for i, s in enumerate(free)}
        diff = (left - right).subs(values)
        if diff.is_Rational:
            # exact rational arithmetic (no transcendental part): one nonzero point settles it
            if diff != 0:
                return False
            continue
        diff = sympy.N(diff, 15)
        if diff.is_real is False or abs(diff) > sympy.Float("1e-9") * (1 + abs(sympy.N(left.subs(values), 15))):
            return False
    return True


def expr_distractors(true, candidates, *, symbols=(), limit: int = 3, avoid=()):
    """Tagged closed-form distractors from `(expression, misconception)` pairs.

    A candidate is dropped when it is the same function as `true` or as an earlier candidate, or is not a finite real
    expression. The expression grader compares exactly, so a candidate that differs at all is rejected by it.
    """
    true = sympy.sympify(true)
    kept = [true]
    avoid = [sympy.sympify(a) for a in avoid]
    out = []
    for expr, tag in candidates:
        if expr is None:
            continue
        expr = sympy.sympify(expr)
        if expr.free_symbols and expr.is_polynomial(*expr.free_symbols):
            factored = sympy.factor(sympy.expand(expr))
            if len(sympy.sstr(factored)) <= len(sympy.sstr(sympy.expand(expr))):
                expr = factored
            else:
                expr = sympy.expand(expr)
        if expr == 0:
            continue
        if expr.has(sympy.zoo, sympy.nan, sympy.oo, -sympy.oo) or expr.is_real is False:
            continue
        if any(same_function(expr, k, symbols) for k in kept + avoid):
            continue
        kept.append(expr)
        out.append(Distractor(pretty(expr), tag, sympy.sstr(expr).replace("**", "^")))
        if len(out) == limit:
            break
    return out


def num_distractors(true: float, candidates, places: int, *, limit: int = 3, avoid=()):
    """Tagged decimal distractors with the same number of places as the answer, each well clear of it.

    A typed number is graded to its own stated precision, so a candidate within two units of the last place of the true
    answer (or of an earlier candidate) is dropped.
    """
    unit = 10.0**-places
    kept = [round(true, places)] + [round(float(a), places) for a in avoid]
    out = []
    for value, tag in candidates:
        if value is None:
            continue
        value = round(float(value), places)
        if value <= 0 or any(abs(value - k) < 2.5 * unit for k in kept):
            continue
        kept.append(value)
        text = f"{value:.{places}f}"
        out.append(Distractor(text, tag, float(text)))
        if len(out) == limit:
            break
    return out


def stored(distractor):
    """A distractor's stored value as a Fraction, a float or a SymPy expression (for follow-through arithmetic)."""
    value = distractor.value
    if isinstance(value, dict):
        return Fraction(value["p"], value["q"])
    if isinstance(value, str):
        return sympy.sympify(value.replace("^", "**"))
    if isinstance(value, int) and not isinstance(value, bool):
        return Fraction(value)
    return value


def need(distractors, count: int = 2):
    """Fail the draw loudly when fewer than `count` distractors survived."""
    check(len(distractors) >= count, f"only {len(distractors)} distractors")
    return distractors


_TIDY = re.compile(r"(?<=[\w)])\*(?=[A-Za-z(])|(?<=[A-Za-z)])\*(?=\d)")
_TIMES = re.compile(r"(?<=\d)\*(?=\d)")


def tidy(text: str) -> str:
    """`3*x^2` written `3x^2` in a stem or a working line (never in an answer); `2*3` written `2 × 3`."""
    return _TIMES.sub(" × ", _TIDY.sub("", text))


def clean_working(fn):
    """Write the working lines of a generator's question the way a learner reads them (no `*` operators)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        if isinstance(out, dict):
            if out.get("working"):
                out["working"] = [tidy(line) for line in out["working"]]
            for part in out.get("parts") or ():
                if part.get("working"):
                    part["working"] = [tidy(line) for line in part["working"]]
        return out

    return wrapper


# -- fast exact calculus on polynomials -----------------------------------------------------------------------


@functools.lru_cache(maxsize=8192)
def pint(expr, limits):
    """The definite integral of a polynomial (or a polynomial with coefficients in other symbols) over `(var, lo, hi)`.

    Much quicker than `sympy.integrate` for the polynomial densities used here; limits may be expressions in the other
    variables. The result is expanded.
    """
    var, lo, hi = limits
    poly = sympy.Poly(sympy.expand(expr), var)
    anti = poly.integrate().as_expr()
    return sympy.expand(anti.subs(var, hi) - anti.subs(var, lo))


@functools.lru_cache(maxsize=8192)
def rational(expr):
    """`expr` as a single reduced fraction: content pulled out of numerator and denominator, the numerator's leading
    coefficient positive (so `3x²/(8 - y³)`, not `-3x²/((y - 2)(y² + 2y + 4))`)."""
    num, den = sympy.fraction(sympy.cancel(sympy.together(expr)))
    num, den = sympy.factor_terms(num), sympy.factor_terms(den)
    symbols = sorted(num.free_symbols, key=str)
    try:
        lead = sympy.Poly(sympy.expand(num), *symbols).LC() if symbols else num
    except sympy.PolynomialError:
        lead = 1
    if lead < 0:
        num, den = sympy.factor_terms(-num), sympy.factor_terms(-den)
    return num / den


def redraw(fn):
    """Run a generator body again with the same random stream when a draw leaves too few distractors.

    A distractor that equals the follow-through value of an earlier wrong part is dropped (the learner who followed
    through correctly must never meet it), and a few draws are left with fewer than two. Drawing again is cheaper than
    padding the list with weak distractors.
    """

    @functools.wraps(fn)
    def wrapper(rng, *args, **kwargs):
        for _ in range(60):
            try:
                return fn(rng, *args, **kwargs)
            except VerificationError as error:
                if not str(error).startswith("only "):
                    raise
        raise VerificationError("no draw left enough distractors")

    return wrapper
