"""Newton divided differences: `r4b_newton_divided_differences` (easy, medium, hard) and the exam chain
`exd_x_r4b_newton_divided_differences`.

``uc2-newton-divided-differences`` (``numerical-and-computational-mathematics``). The data are whole-number ``y``-values
at unequally spaced whole-number nodes, so the divided differences are genuine fractions.

* easy: the second divided difference ``f[x₀, x₁, x₂]`` of three data points.
* medium: the third divided difference ``f[x₀, x₁, x₂, x₃]`` of four data points.
* hard: one ``y``-value is missing; the third divided difference is given, and the learner works back to the
  missing value.
* exam: four data points in four parts: ``f[x₀, x₁, x₂]``, ``f[x₁, x₂, x₃]``, ``f[x₀, x₁, x₂, x₃]`` from the learner's two
  second differences, and the value of the Newton-form cubic at a stated point from the learner's leading
  coefficients.

A divided difference of order k is checked against the coefficient of ``xᵏ`` in the interpolating polynomial SymPy
builds through the same points, and a polynomial value against SymPy's ``interpolate``.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..core import attempts, check, generator
from ._r4b_kit import (
    X,
    Redraw,
    as_single,
    fraction_distractors,
    fraction_part,
    fr,
    sub,
    with_redraw,
)
from ._ucv1_texsay import say_marked

M_NO_DIV = "num2-divided-difference-not-divided-by-node-gap"
M_ADJACENT = "num2-divided-difference-divided-by-adjacent-gap"
M_REVERSED = "num2-divided-difference-order-reversed"
M_EQUAL = "num2-divided-difference-equal-spacing-assumed"
M_LOWER = "num2-divided-difference-previous-order-reported"
M_W_SIGN = "num2-divided-difference-weight-sign-error"
M_NO_WEIGHT = "num2-divided-difference-missing-value-not-scaled"
M_DROP_TERM = "num2-divided-difference-known-term-dropped"
M_NEWTON_DROP = "num2-newton-form-last-term-dropped"
M_NEWTON_COEF = "num2-newton-form-wrong-coefficient"
M_NEWTON_NODES = "num2-newton-form-shifted-nodes"
M_NEWTON_SIGN = "num2-newton-form-factor-sign"
M_LEAD_FIRST = "num2-divided-difference-first-order-reported"


def dd(xs, ys):
    """The divided difference f[xs[0], ..., xs[-1]] by the recursion."""
    if len(xs) == 1:
        return fr(ys[0])
    return (dd(xs[1:], ys[1:]) - dd(xs[:-1], ys[:-1])) / (fr(xs[-1]) - fr(xs[0]))


def dd_by_weights(xs, ys):
    """The same value from the closed form sum y_i / prod_{j != i}(x_i - x_j): a route with no table in it."""
    total = Fraction(0)
    for i, (x, y) in enumerate(zip(xs, ys)):
        w = Fraction(1)
        for j, xj in enumerate(xs):
            if j != i:
                w *= fr(x) - fr(xj)
        total += fr(y) / w
    return total


def check_dd(xs, ys, value):
    k = len(xs) - 1
    check(dd_by_weights(xs, ys) == value, "closed form disagrees")
    poly = sympy.interpolate([(sympy.Integer(x), sympy.Integer(y)) for x, y in zip(xs, ys)], X)
    lead = sympy.Poly(poly, X).coeff_monomial(X**k) if k else poly
    check(lead == sympy.Rational(value.numerator, value.denominator), "SymPy's leading coefficient disagrees")


def newton_value(coeffs, xs, t, upto=None):
    """Newton form c0 + c1(t - x0) + c2(t - x0)(t - x1) + ... evaluated at t, using the first `upto` terms."""
    upto = len(coeffs) if upto is None else upto
    total, product = Fraction(0), Fraction(1)
    for i in range(upto):
        total += fr(coeffs[i]) * product
        if i < len(xs):
            product *= fr(t) - fr(xs[i])
    return total


def _nodes(rng, n, lo=-3, hi=7):
    for _ in attempts(500, "unequally spaced nodes"):
        xs = sorted(rng.sample(range(lo, hi + 1), n))
        gaps = {xs[i + 1] - xs[i] for i in range(n - 1)}
        if len(gaps) >= 2:
            return xs
    raise AssertionError("unreachable")


def _values(rng, n, lo=-6, hi=9):
    for _ in attempts(500, "y-values with no pattern"):
        ys = [rng.randint(lo, hi) for _ in range(n)]
        if len(set(ys)) >= n - 1 and len(set(ys[i + 1] - ys[i] for i in range(n - 1))) >= 2:
            return ys
    raise AssertionError("unreachable")


def _values_text(xs, ys, missing=None):
    """``f(−1) = 9, f(1) = 7 and f(7) = 7``, for use between « and »: each value is its own maths fragment and the
    joining words stay prose."""
    items = [f"f({_n(x)}) = {'c' if i == missing else _n(y)}" for i, (x, y) in enumerate(zip(xs, ys))]
    return "», «".join(items[:-1]) + "» and «" + items[-1]


def _n(v) -> str:
    return str(v).replace("-", "−")


def _nodes_text(xs) -> str:
    return "[" + ", ".join(_n(x) for x in xs) + "]"


def _slips(xs, ys):
    """Wrong values for the divided difference of all the points."""
    k = len(xs) - 1
    top = dd(xs[1:], ys[1:])
    bottom = dd(xs[:-1], ys[:-1])
    out = [
        (M_NO_DIV, top - bottom),
        (M_ADJACENT, (top - bottom) / (fr(xs[-1]) - fr(xs[-2]))),
        (M_REVERSED, bottom - top),
        (M_LOWER, top),
    ]
    # finite differences with the node gaps ignored: Delta^k y / k!
    diffs = list(map(fr, ys))
    for _ in range(k):
        diffs = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
    fact = 1
    for i in range(2, k + 1):
        fact *= i
    out.append((M_EQUAL, diffs[0] / fact))
    return out


@generator(
    "r4b_newton_divided_differences",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4b_newton_divided_differences(rng, tier="easy"):
    return with_redraw(lambda: _single(rng, tier), "a divided-difference question with enough distractors")


def _single(rng, tier):
    if tier in ("easy", "medium"):
        n = 3 if tier == "easy" else 4
        xs = _nodes(rng, n)
        ys = _values(rng, n)
        value = dd(xs, ys)
        if value.denominator == 1:
            raise Redraw
        check_dd(xs, ys, value)
        order = ["second", "third"][n - 3]
        names = ", ".join(f"x{sub(i)}" for i in range(n))
        text = say_marked(
            f"A function «f» has «{_values_text(xs, ys)}». "
            f"Find the {order} divided difference «f{_nodes_text(xs)}» as an exact fraction."
        )
        dist = fraction_distractors(value, _slips(xs, ys))
        if len(dist) < 3 or not any("/" in str(d) for d in dist):
            raise Redraw  # the answer is a fraction, so at least one option must be one too
        part = fraction_part(
            "",
            value,
            _slips(xs, ys),
            1,
            working=[
                "Build the table: first differences (f(xᵢ₊₁) − f(xᵢ))/(xᵢ₊₁ − xᵢ), then divide each later difference by the gap between the outer nodes.",
                f"f[{names}] = {value}.",
            ],
        )
        return _single_dict(text, part, dist)
    # hard: a missing y-value
    for _ in attempts(500, "a missing-value draw"):
        xs = _nodes(rng, 4)
        ys = _values(rng, 4)
        miss = rng.choice([1, 2])
        k_val = dd(xs, ys)
        if k_val.denominator == 1 or k_val.denominator > 60:
            continue
        break
    check_dd(xs, ys, k_val)
    weights = []
    for i in range(4):
        w = Fraction(1)
        for j in range(4):
            if j != i:
                w *= fr(xs[i]) - fr(xs[j])
        weights.append(w)
    known = sum(Fraction(ys[i]) / weights[i] for i in range(4) if i != miss)
    solved = (k_val - known) * weights[miss]
    check(solved == ys[miss], "solving for the missing value failed")
    other = (k_val + known) * weights[miss]
    cands = [
        (M_NO_WEIGHT, k_val - known),
        (M_W_SIGN, (k_val - known) * -weights[miss]),
        (M_DROP_TERM, k_val * weights[miss]),
        ("num2-divided-difference-weight-divided-not-multiplied", (k_val - known) / weights[miss]),
        ("num2-divided-difference-sign-of-known-terms", other),
    ]
    text = say_marked(
        f"A function «f» has «{_values_text(xs, ys, miss)}», where «c» is not known. The third divided "
        f"difference «f{_nodes_text(xs)}» equals «{_frac_text(k_val).replace('-', '−')}». Find «c»."
    )
    dist = fraction_distractors(Fraction(ys[miss]), cands)
    if len(dist) < 3:
        raise Redraw
    part = fraction_part(
        "",
        Fraction(ys[miss]),
        cands,
        1,
        working=[
            "The third divided difference is the sum of f(xᵢ) divided by the product of (xᵢ − xⱼ) over the other nodes.",
            f"Solving that linear equation for c gives c = {ys[miss]}.",
        ],
    )
    return _single_dict(text, part, dist)


def _frac_text(v) -> str:
    v = fr(v)
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def _single_dict(text, part, dist):
    return dict(
        text=text[0],
        latex=text[1],
        answer=part["answer"],
        answer_text=part["answer_text"],
        answer_kind=part["answer_kind"],
        distractors=dist,
        working=part["working"],
    )


# ---------------------------------------------------------------------------------------------------------------
# exam chain


@generator("exd_x_r4b_newton_divided_differences", level=20, presets={"exam": dict(tier="exam")})
def exd_x_r4b_newton_divided_differences(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    return with_redraw(lambda: _exam(rng), "an exam divided-difference question")


def _exam(rng):
    for _ in attempts(1000, "an exam divided-difference draw"):
        xs = _nodes(rng, 4, 0, 6)
        ys = _values(rng, 4)
        a = dd(xs[:3], ys[:3])
        b = dd(xs[1:], ys[1:])
        c = dd(xs, ys)
        if a.denominator == 1 or b.denominator == 1 or c.denominator == 1 or a == b:
            continue
        ts = [v for v in range(xs[0] + 1, xs[3]) if v not in xs]
        if not ts:
            continue
        t = rng.choice(ts)
        value = newton_value([ys[0], dd(xs[:2], ys[:2]), a, c], xs, t)
        break
    first = dd(xs[:2], ys[:2])
    check_dd(xs[:3], ys[:3], a)
    check_dd(xs[1:], ys[1:], b)
    check_dd(xs, ys, c)
    ref = sympy.interpolate([(sympy.Integer(x), sympy.Integer(y)) for x, y in zip(xs, ys)], X)
    check(ref.subs(X, t) == sympy.Rational(value.numerator, value.denominator), "the Newton form disagrees with SymPy")
    stem, latex = say_marked(
        f"A function «f» has «{_values_text(xs, ys)}». Write «x₀ = {_n(xs[0])}», «x₁ = {_n(xs[1])}», "
        f"«x₂ = {_n(xs[2])}» and «x₃ = {_n(xs[3])}»."
    )
    p_a = fraction_part(
        "Find the second divided difference «f[x₀, x₁, x₂]» as an exact fraction.",
        a,
        _slips(xs[:3], ys[:3]),
        2,
        working=[f"f[x₀, x₁, x₂] = {_frac_text(a)}."],
    )
    p_b = fraction_part(
        "Find the second divided difference «f[x₁, x₂, x₃]» as an exact fraction.",
        b,
        _slips(xs[1:], ys[1:]),
        2,
        working=[f"f[x₁, x₂, x₃] = {_frac_text(b)}."],
    )
    gap = xs[3] - xs[0]
    p_c = fraction_part(
        "Use your two second divided differences to find the third divided difference «f[x₀, x₁, x₂, x₃]» "
        "as an exact fraction.",
        c,
        [
            (M_NO_DIV, b - a),
            (M_ADJACENT, (b - a) / (xs[3] - xs[2])),
            (M_REVERSED, a - b),
            (M_EQUAL, (b - a) / 3),
        ],
        2,
        ft=f"(part_b - part_a)/{gap}",
        working=[f"f[x₀, x₁, x₂, x₃] = (f[x₁, x₂, x₃] − f[x₀, x₁, x₂])/(x₃ − x₀) = {_frac_text(c)}."],
    )
    coeffs = [ys[0], first, a, c]
    p_d = fraction_part(
        f"The cubic in Newton form is «p(x) = f(x₀) + f[x₀, x₁](x − x₀) + f[x₀, x₁, x₂](x − x₀)(x − x₁) + "
        f"f[x₀, x₁, x₂, x₃](x − x₀)(x − x₁)(x − x₂)». Use your values of «f[x₀, x₁, x₂]» and «f[x₀, x₁, x₂, x₃]» "
        f"to find «p({t})» as an exact fraction.",
        value,
        [
            (M_NEWTON_DROP, newton_value(coeffs, xs, t, 3)),
            (M_NEWTON_COEF, newton_value([ys[0], first, b, c], xs, t)),
            (M_NEWTON_NODES, newton_value(coeffs, xs[1:] + [xs[0]], t)),
            (
                M_NEWTON_SIGN,
                newton_value(coeffs, [-x for x in xs], -t)
                if False
                else newton_value(coeffs, [t + (t - x) * 0 + 0 for x in xs], t)
                if False
                else _flip(coeffs, xs, t),
            ),
        ],
        2,
        ft=f"{ys[0]} + {_ft_num(first)}*({t} - {xs[0]}) + part_a*({t} - {xs[0]})*({t} - {xs[1]}) + part_c*({t} - {xs[0]})*({t} - {xs[1]})*({t} - {xs[2]})",
        working=[f"p({t}) = {_frac_text(value)}."],
    )
    for part in (p_a, p_b, p_c, p_d):
        part["prompt"], part["prompt_latex"] = say_marked(part["prompt"])
    return dict(text=stem, latex=latex, parts=[p_a, p_b, p_c, p_d])


def _ft_num(v) -> str:
    v = fr(v)
    return f"Rational({v.numerator}, {v.denominator})"


def _flip(coeffs, xs, t):
    """Each factor written (x_i - t) instead of (t - x_i)."""
    total, product = Fraction(0), Fraction(1)
    for i, cc in enumerate(coeffs):
        total += fr(cc) * product
        if i < len(xs):
            product *= fr(xs[i]) - fr(t)
    return total
