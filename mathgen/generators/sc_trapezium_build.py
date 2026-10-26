"""Generates `sc_trapezium` questions (easy, medium, hard) and the exam chain `sc_trapezium_chain`.

Skill `trapezium-rule-estimate` (answer `number`): estimate a definite integral with the trapezium rule on 2 to 5
equal strips. With ordinates y0 ... yn and strip width h, the estimate is (h/2)(y0 + 2y1 + ... + 2y(n-1) + yn).

* easy: 2 to 4 strips on a short list of given ordinates, or 2 or 3 strips on a polynomial whose ordinates are whole
  numbers; the answer is an exact decimal.
* medium: either a measured table (a river's depth, a pump's flow, a rainfall rate, a solar panel's output) with a
  unit and a stated precision, on 3 to 5 strips; or 2 to 4 strips on e^x, 1/x, sqrt(x) or sin(x) (radians), where
  the ordinates are irrational and the answer is stated to 3 decimal places.
* hard: one of three tasks that need the rule turned round. Recover a missing ordinate from a stated estimate (its
  weight is 1 at an end and 2 inside); find the number of strips from the amount by which the rule overestimates
  the integral of a quadratic; or find "exact value minus estimate" for e^x, 1/x, sqrt(x) or sin(x), whose sign
  shows whether the rule over- or under-estimated (concavity), as a number rather than a coded choice.
* exam: 2 to 5 strips on one of those four curves: the strip width, the weighted sum, the estimate, exact value minus
  estimate and the absolute error, each part following through from the last.

Every key is re-derived by an independent route (SymPy's exact sum over the ordinates, with each weight read from its
position, and the sign of every error checked against the curve's concavity). The polynomial tiers use whole-number
ordinates, so no rounding of a working value can move an answer; the curve tiers keep every key clear of a rounding
tie, also when the ordinates are first rounded to six decimal places.
"""

from __future__ import annotations

import random
from fractions import Fraction

import sympy

from ..core import attempts, check, generator, join_terms, paren_negative, phrase, round_half_up
from ._company_kit import keep_company, numeric_pool
from .exd_x_r2b_support import GRADER_TOLERANCE, decimal_places
from .exd_x_r2b_support import exact_number_part as _exact_number_part
from .exd_x_r2b_support import fraction_part as _fraction_part
from .exd_x_r2b_support import number_part as _number_part
from .r4e_common import (
    Redraw,
    clear_of_boundary,
    definite,
    draw,
    real,
    single,
)

x = sympy.Symbol("x")

_MULT = "trapezium-multiplier-h-not-half"
_INNER1 = "trapezium-inner-weight-1"
_ENDS2 = "trapezium-ends-doubled"
_OMIT = "trapezium-end-ordinate-omitted"
_COUNT = "trapezium-strips-counted-as-ordinates"
_LENGTH = "trapezium-uses-interval-length-as-h"
_SIMPSON = "trapezium-simpson-weights"
_MISSING = "trapezium-missing-ordinate-weight"
_REVERSED = "trapezium-error-sign-reversed"
_ABS = "trapezium-sign-dropped"
_SQUARED = "trapezium-strips-from-error-squared"
_WIDTH = "trapezium-strip-width-for-strips"
_OFF = "trapezium-strips-off-by-one"


#: the question being built draws from this, so the choices `_company` makes follow the seed
_RNG: list = [None]


def _company(key, wrong, places, *, lo=None, whole=False, decimals=False, skew=False):
    """`wrong` re-ordered, and padded from a pool of plain near misses, so the key is never the only option with some
    feature a learner can see (decimal style, parity, how many other options lie close). `places` is how many decimal
    places the options are printed to (None: as whole numbers or fractions)."""
    key = Fraction(key)
    tol = max(GRADER_TOLERANCE, Fraction(1, 10**places)) if places is not None else Fraction(0)
    unit = Fraction(1, 10**places) if places else Fraction(1)

    def snap(v):
        v = Fraction(v)
        return round_half_up(v, places) if places is not None else v

    def usable(v):
        return abs(snap(v) - key) > tol and (not whole or Fraction(v).denominator == 1)

    ordered, seen = [], {key}
    for v, tag in wrong:
        if v is None:
            continue
        v = snap(v)
        if v not in seen and usable(v):
            seen.add(v)
            ordered.append((v, tag))
    step = Fraction(1, 10) if abs(key) < 1 else Fraction(1, 2) if abs(key) < 3 else Fraction(1)
    if places is not None:
        step = max(step, unit)
    pool = [(snap(v), t) for v, t in numeric_pool(key, step, lo, None, usable)]
    if abs(key) >= 10:  # wider misses of the same shape, so the key can share its digit count with a distant option
        seen_pool = {v for v, _ in pool} | seen
        for d in (3, 5, 7, 12, 25, 40):
            for v in (key + d * step * 2, key - d * step * 2):
                if v not in seen_pool and usable(v) and (lo is None or v >= lo):
                    seen_pool.add(v)
                    pool.append((snap(v), "near-miss-value"))
    # a uniformly random rank for the key among its options: choose how many of the three shown wrong options lie below
    # it, taking the tagged slips first and plain near misses from the pool to fill the sides they leave short
    rng = _RNG[0] or random.Random(repr(("rank", str(key), [str(v) for v, _ in ordered])))
    below = [c for c in ordered if c[0] < key]
    above = [c for c in ordered if c[0] > key]
    below_pool = [c for c in pool if c[0] < key and c[0] not in {v for v, _ in ordered}]
    above_pool = [c for c in pool if c[0] > key and c[0] not in {v for v, _ in ordered}]

    def picked(t):
        return (below + below_pool)[:t] + (above + above_pool)[: 3 - t]

    def sign(v):
        return (v > 0) - (v < 0)

    feasible = [
        t
        for t in range(4)
        if len(below) + len(below_pool) >= t
        and len(above) + len(above_pool) >= 3 - t
        and (key == 0 or any(sign(c[0]) == sign(key) for c in picked(t)))  # not the only option on its side of zero
    ]
    if feasible:
        t = rng.choices(feasible, weights=[(1.6, 0.6, 1.0, 1.0)[t] if skew else 1.0 for t in feasible])[0]
        pick = picked(t)
        if places is None and not decimals and not whole:
            # fractions: the key shares being whole (or not) with one of its options, replacing a pick on the same side
            same = [c for c in ordered + pool if (c[0].denominator == 1) == (key.denominator == 1) and c not in pick]
            if same and not any((c[0].denominator == 1) == (key.denominator == 1) for c in pick):
                i = rng.randrange(3)
                twins = [c for c in same if (c[0] < key) == (pick[i][0] < key)] or same
                pick[i] = twins[0]
        ordered = pick + [c for c in ordered if c not in pick]
    custom = None
    if decimals and places is None:
        # exact decimals print to as many places as they need: measure what the learner reads, not the fraction
        def read(v):
            n = decimal_places(Fraction(v), 8)
            return f"{float(v):.{n}f}" if n is not None else str(v)

        def digits(v):
            return min(sum(c.isdigit() for c in read(v)), 8)

        def style(v):
            tail = read(v).partition(".")[2]
            return "none" if not tail else "zeros" if not tail.strip("0") else "nonzero"

        custom = {"ndigits": digits, "decimal_style": style}
    chosen = keep_company(key, ordered, pool, count=3, places=places, custom=custom, shift_cost=1.5)
    # a key far smaller than every wrong option is the one that stands out: give it a close, plainly wrong companion
    shown = chosen[:3]
    if key != 0 and all(abs(v) > 4 * abs(key) + tol for v, _ in shown):
        near = [c for c in pool if tol < abs(c[0] - key) < 8 * max(tol, abs(key)) and c[0] not in {v for v, _ in shown}]
        if near:
            shown[rng.randrange(3)] = near[rng.randrange(min(len(near), 4))]
            chosen = shown + chosen[3:]
    return chosen


def number_part(prompt, value, places, marks, working, wrong, **kw):
    key = round_half_up(Fraction(value), places)
    return _number_part(
        prompt,
        value,
        places,
        marks,
        working,
        _company(key, wrong, places, lo=kw.pop("lo", None), skew=kw.pop("skew", False)),
        **kw,
    )


def exact_number_part(prompt, value, marks, working, wrong, **kw):
    value = Fraction(value)
    places = decimal_places(value, 8)
    wrong = [
        (v, t) for v, t in wrong if v is not None and decimal_places(Fraction(v), kw.get("max_places", 8)) is not None
    ]
    whole = kw.pop("whole", False)
    fixed = kw.get("min_places") if kw.get("min_places") == kw.get("max_places") else None
    wrong = _company(value, wrong, fixed, lo=kw.pop("lo", None), whole=whole, decimals=True)
    return _exact_number_part(prompt, value, marks, working, wrong, **kw)


def fraction_part(prompt, value, marks, working, wrong, **kw):
    return _fraction_part(prompt, value, marks, working, _company(value, wrong, None, lo=Fraction(1, 100)), **kw)


def weights(n: int) -> list[int]:
    """Trapezium weights for n strips: 1, 2, ..., 2, 1."""
    return [1] + [2] * (n - 1) + [1]


def wsum(ys) -> Fraction:
    """The weighted sum y0 + 2(y1 + ... + y(n-1)) + yn."""
    return Fraction(ys[0]) + Fraction(ys[-1]) + 2 * sum((Fraction(y) for y in ys[1:-1]), Fraction(0))


def trapezium(ys, h) -> Fraction:
    """The trapezium estimate (h/2)(y0 + 2y1 + ... + yn), exactly."""
    return Fraction(h) * wsum(ys) / 2


def _check_trapezium(ys, h, value: Fraction) -> None:
    """Independent route: SymPy's exact sum of the area of each trapezium, one strip at a time."""
    total = sympy.Integer(0)
    hq = sympy.Rational(Fraction(h).numerator, Fraction(h).denominator)
    for left, right in zip(ys, ys[1:]):
        lq = sympy.Rational(Fraction(left).numerator, Fraction(left).denominator)
        rq = sympy.Rational(Fraction(right).numerator, Fraction(right).denominator)
        total += hq * (lq + rq) / 2
    check(sympy.Rational(value.numerator, value.denominator) == total, "trapezium estimate disagrees")


def _wrong_estimates(ys, h):
    """(value, tag) for the errors a learner makes with the weights, the multiplier and the strip count."""
    ys = [Fraction(y) for y in ys]
    h = Fraction(h)
    n = len(ys) - 1
    length = h * n
    out = [
        (h * wsum(ys), _MULT),
        (h * sum(ys) / 2, _INNER1),
        (h * sum(ys), _ENDS2),
        (h * (wsum(ys) - ys[-1]) / 2, _OMIT),
        (length / (n + 1) * wsum(ys) / 2, _COUNT),
        (length * wsum(ys) / 2, _LENGTH),
    ]
    if n % 2 == 0:
        simpson = sum((w * y for w, y in zip([1] + [4 if i % 2 else 2 for i in range(1, n)] + [1], ys)), Fraction(0))
        out.append((h * simpson / 3, _SIMPSON))
    return out


def _poly_text(coeffs) -> str:
    """Coefficients from the highest power down, as a sum."""
    deg = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        p = deg - i
        terms.append((c, "" if p == 0 else ("x" if p == 1 else f"x^{p}")))
    return join_terms(terms)


def _poly_values(coeffs, start, h, n):
    deg = len(coeffs) - 1
    return [sum(Fraction(c) * (start + i * h) ** (deg - j) for j, c in enumerate(coeffs)) for i in range(n + 1)]


def _bracketed(v, places=None) -> str:
    """A number as text that may follow a minus sign: negative values in brackets."""
    text = decimal_text(v) if places is None else f"{float(v):.{places}f}"
    return f"({text})" if v < 0 else text


def decimal_text(v: Fraction) -> str:
    places = decimal_places(v, 8) or 0
    return f"{float(v):.{places}f}"


# -- easy ---------------------------------------------------------------------------------------------------------

_TABLE_STEMS = (
    "A function g has the values {vals} at x = {xs}. Use the trapezium rule with {n} strips to estimate the integral of g from x = {a} to x = {b}.",
    "The y-values of a curve y = g(x) are {vals} at x = {xs} respectively. Approximate the area under the curve between x = {a} and x = {b} using the trapezium rule with {n} strips.",
    "Readings of a function g are {vals} at x = {xs}. Estimate the integral of g from x = {a} to x = {b} by the trapezium rule on {n} equal strips.",
)

_POLY_STEMS = (
    "Use the trapezium rule with {n} strips to approximate the integral of f(x) = {f} from x = {a} to x = {b}.",
    "Approximate the area under the curve y = {f} between x = {a} and x = {b} using the trapezium rule with {n} strips.",
    "The function f is given by f(x) = {f}. Use the trapezium rule with {n} equal strips to estimate the integral of f from {a} to {b}.",
)


def _easy_table(rng):
    for _ in attempts(500, "a table with a clean trapezium estimate"):
        n = rng.choice((2, 3, 4, 4))
        h = rng.choice((1, 2, 2, 3, 5))
        start = rng.choice((0, 0, 1, 2, -1))
        ys = [Fraction(rng.randint(1, 24)) for _ in range(n + 1)]
        if len(set(ys)) >= n:
            break
    est = trapezium(ys, h)
    _check_trapezium(ys, h, est)
    xs = [start + i * h for i in range(n + 1)]
    stem = phrase(
        rng,
        _TABLE_STEMS,
        vals=", ".join(str(int(y)) for y in ys),
        xs=", ".join(str(v) for v in xs),
        n=n,
        a=xs[0],
        b=xs[-1],
    )
    working = [
        f"h = {h}; weights 1, 2, ..., 2, 1 give the sum {int(wsum(ys))}.",
        f"Estimate = ({h}/2) x {int(wsum(ys))} = {decimal_text(est)}.",
    ]
    return single(
        stem, exact_number_part("Give your answer as an exact decimal.", est, 1, working, _wrong_estimates(ys, h))
    )


def _easy_poly(rng):
    for _ in attempts(500, "a polynomial with a clean trapezium estimate"):
        n = rng.choice((2, 3))
        degree = rng.choice((2, 3))
        coeffs = [rng.choice((1, 2, 3)) * rng.choice((1, 1, -1))]
        coeffs += [rng.choice((-4, -3, -2, -1, 0, 1, 2, 3, 5)) for _ in range(degree - 1)]
        coeffs.append(rng.choice((1, 2, 3, 4, 6, 9)))
        h = rng.choice((1, 1, 2))
        start = rng.choice((0, 0, 1, -1))
        ys = _poly_values(coeffs, start, h, n)
        est = trapezium(ys, h)
        if min(ys) > 0 and len(set(ys)) >= n and decimal_places(est, 6) is not None:
            break
    _check_trapezium(ys, h, est)
    stem = phrase(rng, _POLY_STEMS, f=_poly_text(coeffs), n=n, a=start, b=start + n * h)
    working = [
        f"h = ({start + n * h} - {paren_negative(start)})/{n} = {h}; the ordinates are {', '.join(str(int(y)) for y in ys)}.",
        f"Estimate = ({h}/2) x {int(wsum(ys))} = {decimal_text(est)}.",
    ]
    return single(
        stem, exact_number_part("Give your answer as an exact decimal.", est, 1, working, _wrong_estimates(ys, h))
    )


# -- medium -------------------------------------------------------------------------------------------------------


def _walk(rng, n, lo, hi, step, decimals):
    """n + 1 irregular readings between lo and hi (in units of 10^-decimals), each a small step from the last."""
    v = rng.randint(lo, hi)
    out = [v]
    for _ in range(n):
        v = min(hi, max(lo, v + rng.randint(-step, step)))
        out.append(v)
    return [Fraction(v, 10**decimals) for v in out]


def _fmt_table(ys, decimals) -> str:
    return ", ".join(f"{float(y):.{decimals}f}" if decimals else str(int(y)) for y in ys)


# (stem, lo, hi, step, decimals, step label); {h} is the gap between readings, {w} the total span
_CONTEXTS = (
    (
        "A surveyor measures the depth of a river every {h} m across its {w} m width. The depths, in metres, are {vals}. "
        "Use the trapezium rule with {n} strips to estimate the area of the river's cross-section, in square metres.",
        8,
        55,
        14,
        1,
    ),
    (
        "The flow rate of water into a tank, in litres per minute, is recorded every {h} minutes for {w} minutes: {vals}. "
        "Use the trapezium rule with {n} strips to estimate the volume of water, in litres, that flowed into the tank.",
        110,
        330,
        70,
        0,
    ),
    (
        "During a storm the rainfall rate, in millimetres per hour, is recorded every {h} hours for {w} hours: {vals}. "
        "Use the trapezium rule with {n} strips to estimate the total rainfall, in millimetres.",
        120,
        980,
        260,
        2,
    ),
    (
        "A solar panel's power output, in kilowatts, is read every {h} hours over a {w} hour period: {vals}. "
        "Use the trapezium rule with {n} strips to estimate the energy produced, in kilowatt-hours.",
        20,
        480,
        150,
        2,
    ),
    (
        "A paddock is {w} m long. Its width, in metres, is measured every {h} m along its length: {vals}. "
        "Use the trapezium rule with {n} strips to estimate the area of the paddock, in square metres.",
        180,
        520,
        70,
        1,
    ),
)


def _medium_table(rng):
    for _ in attempts(500, "a table whose trapezium estimate is clear of a rounding tie"):
        n = rng.choice((3, 4, 5, 5))
        h = rng.choice((2, 3, 4, 5, 6))
        stem_t, lo, hi, step, decimals = rng.choice(_CONTEXTS)
        ys = _walk(rng, n, lo, hi, step, decimals)
        est = trapezium(ys, h)
        if len(set(ys)) >= n and clear_of_boundary(est, 2, Fraction(1, 12)):
            break
    _check_trapezium(ys, h, est)
    stem = stem_t.format(h=h, w=n * h, n=n, vals=_fmt_table(ys, decimals))
    working = [
        f"Strip width h = {h}. Weights 1, 2, ..., 2, 1 give the sum {decimal_text(wsum(ys))}.",
        f"Estimate = ({h}/2) x {decimal_text(wsum(ys))} = {float(est):.4f}.",
    ]
    return single(
        stem,
        number_part("Give the estimate correct to 2 decimal places.", est, 2, 1, working, _wrong_estimates(ys, h)),
    )


# (text, python function, antiderivative difference, concave?, spans (a, b))
def _spans(starts, widths):
    return tuple((Fraction(a), Fraction(a + w)) for a in starts for w in widths)


def _halves(lo, hi):
    pts = [Fraction(i, 2) for i in range(int(lo * 2), int(hi * 2) + 1)]
    return tuple((a, b) for a in pts for b in pts if b - a >= 1)


_CURVES = (
    ("e^x", sympy.exp, lambda a, b: sympy.exp(b) - sympy.exp(a), False, _spans((-1, 0, 1, 2), (1, 2, 3, 4))),
    ("1/x", lambda t: 1 / t, lambda a, b: sympy.log(b) - sympy.log(a), False, _spans((1, 2, 3, 4), (1, 2, 3, 4, 5))),
    (
        "sqrt(x)",
        sympy.sqrt,
        lambda a, b: sympy.Rational(2, 3) * (b ** sympy.Rational(3, 2) - a ** sympy.Rational(3, 2)),
        True,
        _spans((0, 1, 2, 3, 4, 6), (1, 2, 3, 4, 5)),
    ),
    ("sin(x)", sympy.sin, lambda a, b: sympy.cos(a) - sympy.cos(b), True, _halves(0, 3)),
)


def _curve(rng, n_choices):
    """A curve, an interval and a strip count, with the ordinates (exact values, 40 digits), the weighted sum, the
    estimate, the exact integral and the exact integral rounded to 4 decimal places."""
    text, f, exact_fn, concave, intervals = rng.choice(_CURVES)
    a, b = rng.choice(intervals)
    n = rng.choice(n_choices)
    h = (b - a) / n
    xs = [a + h * i for i in range(n + 1)]
    ys = [real(f(sympy.Rational(v.numerator, v.denominator))) for v in xs]
    total = wsum(ys)
    est = h * total / 2
    exact_expr = exact_fn(sympy.Rational(a.numerator, a.denominator), sympy.Rational(b.numerator, b.denominator))
    exact = real(exact_expr)
    exact4 = round_half_up(exact, 4)
    # independent route: SymPy sums the same trapezia symbolically
    sym = sum(w * f(sympy.Rational(v.numerator, v.denominator)) for w, v in zip(weights(n), xs))
    check(abs(real(sym) - total) < Fraction(1, 10**30), "weighted sum disagrees with SymPy")
    check(
        abs(real(sympy.Rational(h.numerator, h.denominator) * sym / 2) - est) < Fraction(1, 10**30),
        "estimate disagrees",
    )
    # concavity: the rule overestimates a convex curve and underestimates a concave one
    check((est < exact) == concave and est != exact, "the error's sign disagrees with the concavity")
    # a calculator that rounds each ordinate to six places must reach the same last digit
    rough = [round_half_up(y, 6) for y in ys]
    if abs(trapezium(rough, h) - est) >= Fraction(1, 10**5):
        raise Redraw("rounding the ordinates moves the estimate")
    return text, a, b, n, h, ys, total, est, exact, exact4, concave


def _medium_formula(rng):
    for _ in attempts(500, "a curve whose estimate is clear of a rounding tie"):
        text, a, b, n, h, ys, total, est, exact, exact4, concave = _curve(rng, (2, 3, 4))
        if clear_of_boundary(est, 3, Fraction(1, 10)) and clear_of_boundary(
            trapezium([round_half_up(y, 6) for y in ys], h), 3, Fraction(1, 20)
        ):
            break
    stem = phrase(
        rng,
        (
            "Use the trapezium rule with {n} strips to estimate the integral of {f} from x = {a} to x = {b}.{r}",
            "Estimate the area under the curve y = {f} between x = {a} and x = {b} using the trapezium rule with {n} strips.{r}",
            "The function f is given by f(x) = {f}. Use the trapezium rule with {n} equal strips to estimate the integral of f from x = {a} to x = {b}.{r}",
        ),
        f=text,
        a=decimal_text(a),
        b=decimal_text(b),
        n=n,
        r=" (x is in radians.)" if text == "sin(x)" else "",
    )
    working = [
        f"h = ({decimal_text(b)} - {_bracketed(a)})/{n} = {_frac_text(h)}; the ordinates are {', '.join(f'{float(y):.4f}' for y in ys)}.",
        f"Estimate = (h/2) x {float(total):.4f} = {float(est):.4f}.",
    ]
    return single(
        stem,
        number_part("Give the estimate correct to 3 decimal places.", est, 3, 2, working, _wrong_estimates(ys, h)),
    )


def _frac_text(v: Fraction) -> str:
    return f"{v.numerator}/{v.denominator}" if v.denominator != 1 else str(v.numerator)


# -- hard ---------------------------------------------------------------------------------------------------------


def _hard_missing(rng):
    """One ordinate is unknown; the trapezium estimate is stated; find it."""
    for _ in attempts(500, "a missing ordinate that is a short decimal"):
        n = rng.choice((3, 4, 5))
        h = rng.choice((Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2), Fraction(5, 2)))
        start = rng.choice((0, 0, 1, 2))
        ys = [Fraction(rng.randint(2, 40), 2) for _ in range(n + 1)]
        j = rng.randint(0, n)
        est = trapezium(ys, h)
        if decimal_places(est, 4) is not None and len(set(ys)) >= n and ys[j] != ys[max(j - 1, 0)]:
            break
    _check_trapezium(ys, h, est)
    xs = [start + h * i for i in range(n + 1)]
    cells = ", ".join(("k" if i == j else decimal_text(y)) for i, y in enumerate(ys))
    x_cells = ", ".join(decimal_text(v) for v in xs)
    stem = phrase(
        rng,
        (
            "A function f has the values {cells} at x = {xs}. The trapezium rule with {n} strips gives {est} as an estimate of the integral of f from x = {a} to x = {b}.",
            "The values of a function f are {cells} at x = {xs}. Using the trapezium rule with {n} strips, the integral of f from x = {a} to x = {b} is estimated to be {est}.",
        ),
        cells=cells,
        xs=x_cells,
        n=n,
        est=decimal_text(est),
        a=decimal_text(xs[0]),
        b=decimal_text(xs[-1]),
    )
    ws = weights(n)
    rest = sum((w * y for i, (w, y) in enumerate(zip(ws, ys)) if i != j), Fraction(0))
    target = 2 * est / h  # the weighted sum
    key = (target - rest) / ws[j]
    check(key == ys[j], "the recovered ordinate disagrees")
    other = 1 if ws[j] == 2 else 2
    wrong = [
        ((target - rest) / other, _MISSING),
        (target - rest, _MISSING),
        ((est - rest) / ws[j], _MULT),
        ((2 * est - rest) / ws[j], _MULT),
        ((est / h - rest) / ws[j], _MULT),
        ((target - rest + ys[j]) / ws[j], _MISSING),
    ]
    working = [
        f"The weighted sum is 2 x {decimal_text(est)} / {decimal_text(h)} = {decimal_text(target)}.",
        f"Ordinate {j} has weight {ws[j]}: {ws[j]}k = {decimal_text(target)} - {_bracketed(rest)}, so k = {decimal_text(ys[j])}.",
    ]
    wrong = [(v, t) for v, t in wrong if (Fraction(v) * 10).denominator == 1]  # one decimal place, like the key
    return single(stem, exact_number_part("Find the value of k.", ys[j], 2, working, wrong, min_places=1, max_places=1))


def _hard_strips(rng):
    """The rule overestimates a quadratic by a stated amount; find the number of strips."""
    for _ in attempts(500, "a quadratic whose overestimate is a short decimal"):
        n = rng.randint(4, 10)
        lead = rng.choice((1, 1, 2, 3, 4))
        length = rng.choice((2, 3, 4, 5, 6))
        coeffs = [lead, rng.choice((-6, -3, -2, 0, 1, 2, 4)), rng.choice((0, 1, 2, 5))]
        start = rng.choice((0, 0, 1, -1, 2))
        err = Fraction(lead * length**3, 6 * n * n)
        if decimal_places(err, 5) is not None and err >= Fraction(1, 100):
            break
    # independent route: the trapezium estimate and the exact integral, strip by strip, with SymPy
    h = Fraction(length, n)
    ys = _poly_values(coeffs, start, h, n)
    est = trapezium(ys, h)
    exact = definite(coeffs, start, start + length)
    check(est - exact == err, "the overestimate disagrees with the quadratic's error")
    check(
        sympy.integrate(sum(c * x ** (2 - i) for i, c in enumerate(coeffs)), (x, start, start + length))
        == sympy.Rational(exact.numerator, exact.denominator),
        "exact integral disagrees",
    )
    stem = phrase(
        rng,
        (
            "The trapezium rule with n equal strips is used to estimate the integral of f(x) = {f} from x = {a} to x = {b}. The estimate is {e} more than the exact value of the integral.",
            "A student estimates the integral of f(x) = {f} from x = {a} to x = {b} using the trapezium rule with n equal strips. Her estimate exceeds the exact value by {e}.",
            "For the curve y = {f}, the trapezium rule with n equal strips overestimates the area between x = {a} and x = {b} by {e}.",
        ),
        f=_poly_text(coeffs),
        a=start,
        b=start + length,
        e=decimal_text(err),
    )
    root = Fraction(length, n)
    wrong = [
        (Fraction(n * n), _SQUARED),
        (root, _WIDTH),
        (Fraction(n + 1), _OFF),
        (Fraction(n - 1), _OFF),
        (Fraction(2 * n), _SQUARED),
    ]
    coef = "" if lead == 1 else str(lead)
    times = "" if lead == 1 else f"{lead} x "
    working = [
        f"On one strip of width h the rule overestimates {coef}x^2 by {coef}h^3/6, so over n strips the overestimate is {times}{length}^3/(6n^2).",
        f"{decimal_text(err)} = {lead * length**3}/(6n^2), so n^2 = {n * n} and n = {n}.",
    ]
    return single(
        stem,
        exact_number_part(
            "Find the number of strips n.",
            Fraction(n),
            2,
            working,
            wrong,
            suffix="Give your answer as a whole number.",
            whole=True,
            lo=1,
        ),
    )


def _hard_difference(rng):
    """Exact value minus the estimate, to 4 decimal places; its sign shows over- or under-estimate."""
    for _ in attempts(500, "a difference clear of a rounding tie"):
        text, a, b, n, h, ys, total, est, exact, exact4, concave = _curve(rng, (2, 3, 4))
        diff = exact4 - est
        if clear_of_boundary(diff, 4, Fraction(1, 10)) and abs(diff) >= Fraction(1, 100):
            break
    check((diff > 0) == concave, "the sign of exact minus estimate disagrees with the concavity")
    stem = phrase(
        rng,
        (
            "The integral of f(x) = {f} from x = {a} to x = {b} is {exact} correct to 4 decimal places. The trapezium rule with {n} strips is used to estimate it.{r}",
            "Let f(x) = {f}. The exact value of the integral of f from x = {a} to x = {b} is {exact}, correct to 4 decimal places. It is estimated using the trapezium rule with {n} equal strips.{r}",
            "The area under y = {f} between x = {a} and x = {b} is {exact} to 4 decimal places. A student applies the trapezium rule with {n} strips to estimate this area.{r}",
        ),
        f=text,
        a=decimal_text(a),
        b=decimal_text(b),
        n=n,
        exact=f"{float(exact4):.4f}",
        r=" Take x to be in radians." if text == "sin(x)" else "",
    )
    wrong = [
        (est - exact4, _REVERSED),
        (est, "reports-estimate-not-error"),
        (abs(diff), _ABS),
        (exact4 - h * total, _MULT),
        (exact4 - h * sum(ys) / 2, _INNER1),
        (exact4 - h * (total - ys[-1]) / 2, _OMIT),
    ]
    working = [
        f"h = {_frac_text(h)}; the weighted sum is {float(total):.4f}, so the estimate is {float(est):.4f}.",
        f"Exact minus estimate = {float(exact4):.4f} - {_bracketed(est, 4)} = {float(diff):.4f}.",
    ]
    return single(
        stem,
        number_part(
            "Find the exact value minus the trapezium rule estimate, using the exact value given, correct to 4 decimal places.",
            diff,
            4,
            2,
            working,
            wrong,
        ),
    )


@generator(
    "sc_trapezium",
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def sc_trapezium(rng, tier: str = "easy"):
    """The trapezium rule on 2 to 5 strips; see the module docstring for the ladder."""
    _RNG[0] = rng
    if tier == "easy":
        return draw(_easy_table if rng.random() < 0.5 else _easy_poly, rng)
    if tier == "medium":
        return draw(_medium_table if rng.random() < 0.5 else _medium_formula, rng)
    if tier == "hard":
        roll = rng.random()
        return draw(_hard_missing if roll < 0.34 else (_hard_strips if roll < 0.67 else _hard_difference), rng)
    raise ValueError(f"unknown tier {tier!r}")


# -- exam chain ---------------------------------------------------------------------------------------------------

_CHAIN_STEMS = (
    "The integral of f(x) = {f} from x = {a} to x = {b} is to be estimated using the trapezium rule with {n} strips. Its exact value is {exact}, correct to 4 decimal places.",
    "Let f(x) = {f}. The trapezium rule with {n} equal strips is used to estimate the integral of f from x = {a} to x = {b}, whose exact value is {exact} to 4 decimal places.",
    "The area under y = {f} from x = {a} to x = {b} has exact value {exact}, correct to 4 decimal places. The trapezium rule with {n} strips is used to estimate it.",
)


def _sum_text(n: int) -> str:
    inner = " + ".join(f"y{i}" for i in range(1, n))
    return f"y0 + 2y1 + y2" if n == 2 else f"y0 + 2({inner}) + y{n}"


def _shuffled(rng, pool):
    """`pool` in a random order, so the key is not always the middle of the options offered."""
    pool = list(pool)
    rng.shuffle(pool)
    return pool


def _chain(rng):
    text, a, b, n, h, ys, total, est, exact, exact4, concave = _curve(rng, (2, 3, 4, 5))
    diff = exact4 - est
    error = abs(diff)
    for value in (total, est, diff):
        if not clear_of_boundary(value, 4, Fraction(1, 10)):
            raise Redraw("a key too close to a rounding tie")
    if error < Fraction(5, 1000):
        raise Redraw("an error that rounds to almost nothing")
    stem = phrase(
        rng,
        _CHAIN_STEMS,
        f=text,
        a=decimal_text(a),
        b=decimal_text(b),
        n=n,
        exact=f"{float(exact4):.4f}",
    )
    if text == "sin(x)":
        stem += " Take x to be in radians."
    length = b - a
    simpson_like = sum((w * y for w, y in zip([1] + [4 if i % 2 else 2 for i in range(1, n)] + [1], ys)), Fraction(0))
    part_a = fraction_part(
        "Find the width h of each strip."
        + (" Give your answer as a fraction in simplest form." if decimal_places(h, 8) is None else ""),
        h,
        1,
        [f"h = ({decimal_text(b)} - {_bracketed(a)})/{n} = {_frac_text(h)}."],
        _shuffled(
            rng,
            [
                (length / (n + 1), _COUNT),
                (length / (n - 1), _OFF),
                (length, _LENGTH),
                (length / (n + 2), _OFF),
                (length / (2 * n), _COUNT),
            ],
        ),
        sensible_range=(0, None),
    )
    wrong_b = [
        (sum(ys), _INNER1),
        (2 * sum(ys), _ENDS2),
        (total - ys[-1], _OMIT),
    ]
    if n % 2 == 0:
        wrong_b.append((simpson_like, _SIMPSON))
    part_b = number_part(
        f"Find {_sum_text(n)}, where y0, ..., y{n} are the values of f at the ends of the strips.",
        total,
        4,
        2,
        [f"The ordinates are {', '.join(f'{float(y):.4f}' for y in ys)}.", f"The weighted sum is {float(total):.4f}."],
        wrong_b,
        sensible_range=(0, None),
    )
    part_c = number_part(
        "Hence find the trapezium rule estimate of the integral.",
        est,
        4,
        2,
        [f"Estimate = (h/2) x {float(total):.4f} = {float(est):.4f}."],
        [
            (h * total, _MULT),
            (h * sum(ys) / 2, _INNER1),
            (h * (total - ys[-1]) / 2, _OMIT),
            (length * total / 2, _LENGTH),
        ],
        follow_through="part_a*part_b/2",
        sensible_range=(0, None),
    )
    part_d = number_part(
        "Find the exact value minus the estimate, using the exact value given.",
        diff,
        4,
        2,
        [
            f"{float(exact4):.4f} - {_bracketed(est, 4)} = {float(diff):.4f}.",
            "A negative result means the rule overestimated; a positive one, that it underestimated.",
        ],
        _shuffled(
            rng,
            [
                (est - exact4, _REVERSED),
                (error, _ABS),
                (exact4 - h * total, _MULT),
                (exact4 - h * sum(ys) / 2, _INNER1),
                (exact4 - h * (total - ys[-1]) / 2, _OMIT),
                (exact4 - length * total / 2, _LENGTH),
            ],
        ),
        follow_through=f"{float(exact4):.4f} - part_c",
    )
    part_e = number_part(
        "Find the absolute error of the estimate.",
        error,
        4,
        1,
        [f"|{float(diff):.4f}| = {float(error):.4f}."],
        [
            (-error, "error-sign-not-absolute"),
            (est, "reports-estimate-not-error"),
            (abs(exact4 - h * total), _MULT),
            (abs(exact4 - h * sum(ys) / 2), _INNER1),
        ],
        follow_through="abs(part_d)",
        skew=True,  # a tiny positive key has few options below it, and the kit moves it up a rank
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d, part_e])


@generator("sc_trapezium_chain", level=17, presets={"exam": dict(tier="exam")})
def sc_trapezium_chain(rng, tier: str = "exam"):
    """The trapezium rule on 2 to 5 strips for a stated curve: width, weighted sum, estimate, difference, error."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    _RNG[0] = rng
    return draw(_chain, rng)
