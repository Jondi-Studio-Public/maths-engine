"""Generates `r4e_simpsons_rule` questions (easy, medium, hard) and the exam chain `r4e_simpsons_rule_chain`.

Skill `simpsons-rule` (answer `number`): estimate a definite integral with Simpson's rule on 2, 4 or 6 equal strips.
With ordinates y0 ... yn and strip width h, the estimate is (h/3)(y0 + 4y1 + 2y2 + ... + 4y(n-1) + yn): weights
1, 4, 2, 4, ..., 4, 1.

* easy: 2 strips (one parabola) on a polynomial given as a formula, so the ordinates are whole numbers; the
  answer is an exact decimal.
* medium: 4 strips. Either a polynomial formula, or a measured table (a river's depth, a tank's flow rate, a block's
  width) with a unit and a stated precision.
* hard: either a missing ordinate to recover from a stated Simpson estimate (the weight depends on where it sits),
  or 6 strips on a quartic, where the rule is not exact and the weight pattern has to be continued correctly.
* exam: 4 strips on 1/x, e^x, sqrt(x) or 1/(1 + x^2): the strip width, the weighted sum, the estimate and its
  absolute error against the stated exact value, each part following through from the last.

Every key is re-derived from the weights by an independent route (SymPy's exact sum over the ordinates), and the
polynomial tiers use whole-number ordinates, so no rounding of a working value can move an answer.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..core import attempts, check, generator, join_terms, paren_negative, phrase
from .r4e_common import (
    Redraw,
    clear_of_boundary,
    definite,
    draw,
    exact_number_part,
    fraction_part,
    number_part,
    real,
    single,
)
from .exd_x_r2b_support import decimal_places

x = sympy.Symbol("x")


def weights(n: int) -> list[int]:
    """Simpson weights for n (even) strips: 1, 4, 2, 4, ..., 4, 1."""
    return [1] + [4 if i % 2 else 2 for i in range(1, n)] + [1]


def simpson(ys, h) -> Fraction:
    """The Simpson estimate (h/3)(y0 + 4y1 + 2y2 + ... + yn), exactly."""
    return Fraction(h) * sum((w * Fraction(y) for w, y in zip(weights(len(ys) - 1), ys)), Fraction(0)) / 3


def _check_simpson(ys, h, value: Fraction) -> None:
    """Independent route: SymPy's exact sum, with the weight of each ordinate read from its position."""
    n = len(ys) - 1
    total = sympy.Integer(0)
    for i, y in enumerate(ys):
        w = 1 if i in (0, n) else (4 if i % 2 else 2)
        total += w * sympy.Rational(y.numerator, y.denominator)
    expected = sympy.Rational(h.numerator, h.denominator) / 3 * total
    check(sympy.Rational(value.numerator, value.denominator) == expected, "Simpson estimate disagrees")


def _trapezium(ys, h) -> Fraction:
    return Fraction(h) / 2 * (ys[0] + ys[-1] + 2 * sum(ys[1:-1]))


def _weighted(ys, ws, h, divisor=3) -> Fraction:
    return Fraction(h) * sum((w * Fraction(y) for w, y in zip(ws, ys)), Fraction(0)) / divisor


def _wrong_estimates(ys, h):
    """(value, tag) for the errors a learner makes with the weights, the multiplier and the strip count."""
    n = len(ys) - 1
    swapped = [1] + [2 if i % 2 else 4 for i in range(1, n)] + [1]
    inner4 = [1] + [4] * (n - 1) + [1]
    h_ordinates = Fraction(h) * n / (n + 1)  # the n + 1 ordinates mistaken for strips
    return [
        (_trapezium(ys, h), "simpson-trapezoidal-weights"),
        (_weighted(ys, swapped, h), "simpson-weights-swapped"),
        (_weighted(ys, inner4, h), "simpson-all-inner-weight-4"),
        (_weighted(ys, weights(n), h, divisor=2), "simpson-wrong-multiplier"),
        (_weighted(ys, weights(n), h_ordinates), "simpson-strips-counted-as-ordinates"),
    ]


def _poly_text(coeffs) -> str:
    """Coefficients from the highest power down, as a sum."""
    deg = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        p = deg - i
        terms.append((c, "" if p == 0 else ("x" if p == 1 else f"x^{p}")))
    return join_terms(terms)


def _poly_values(coeffs, start, h, n):
    return [
        Fraction(int(sum(c * (start + i * h) ** (len(coeffs) - 1 - j) for j, c in enumerate(coeffs))))
        for i in range(n + 1)
    ]


def _coefficients(rng, degree, lead=(1, 2, 3)):
    cs = [rng.choice(lead) * rng.choice((1, 1, -1))]
    cs += [rng.choice([-6, -4, -3, -2, -1, 0, 1, 2, 3, 5, 7]) for _ in range(degree - 1)]
    cs.append(rng.choice([1, 2, 3, 4, 5, 8, 10]))
    return cs


# -- easy ---------------------------------------------------------------------------------------------------------

_EASY_STEMS = (
    "Use Simpson's rule with 2 strips to approximate the integral of f(x) = {f} from x = {a} to x = {b}.",
    "Approximate the area under the curve y = {f} between x = {a} and x = {b} using Simpson's rule with 2 strips.",
    "The function f is given by f(x) = {f}. Use Simpson's rule with 2 equal strips to estimate the integral of f from {a} to {b}.",
)


def _easy(rng):
    for _ in attempts(500, "a terminating Simpson estimate"):
        degree = rng.choice((2, 2, 3))
        coeffs = _coefficients(rng, degree)
        h = rng.choice((1, 1, 2, 3))
        start = rng.choice((0, 0, 1, -1, -2))
        ys = _poly_values(coeffs, start, h, 2)
        est = simpson(ys, h)
        if est > 0 and decimal_places(est, 6) is not None and len(set(ys)) == 3 and min(ys) > 0:
            break
    _check_simpson(ys, Fraction(h), est)
    check(definite(coeffs, start, start + 2 * h) == est, "Simpson's rule is exact for a cubic")
    stem = phrase(rng, _EASY_STEMS, f=_poly_text(coeffs), a=start, b=start + 2 * h)
    wrong = _wrong_estimates(ys, h)
    wrong.append((Fraction(2 * h) / 3 * (ys[0] + 4 * ys[1] + ys[2]), "simpson-uses-interval-length-as-h"))
    working = [
        f"h = ({start + 2 * h} - {paren_negative(start)})/2 = {h}; the ordinates are y0 = {ys[0]}, y1 = {ys[1]}, y2 = {ys[2]}.",
        f"Estimate = ({h}/3)({ys[0]} + 4 x {ys[1]} + {ys[2]}) = {decimal_text(est)}.",
    ]
    return single(stem, exact_number_part("Give your answer as an exact decimal.", est, 1, working, wrong))


def decimal_text(v: Fraction) -> str:
    places = decimal_places(v, 8) or 0
    return f"{float(v):.{places}f}"


# -- medium -------------------------------------------------------------------------------------------------------


def _bell(rng):
    """Depths across a river: shallow banks, deepest in the middle."""
    peak = rng.randint(24, 46)
    left, right = rng.randint(2, 9), rng.randint(2, 9)
    mid_l = rng.randint(max(left + 4, peak - 14), peak - 3)
    mid_r = rng.randint(max(right + 4, peak - 14), peak - 3)
    return [Fraction(v, 10) for v in (left, mid_l, peak, mid_r, right)]


def _rising_then_easing(rng):
    """Flow rates in L/min: a pump that ramps up and then eases off."""
    base = rng.randint(120, 200)
    rates = [
        base,
        base + rng.randint(40, 90),
        base + rng.randint(100, 170),
        base + rng.randint(60, 120),
        base + rng.randint(10, 60),
    ]
    return [Fraction(v) for v in rates]


def _plot_widths(rng):
    """Widths of a block of land across its length, in metres, varying a little."""
    w = rng.randint(24, 40)
    return [Fraction(w + rng.randint(-6, 8)) for _ in range(5)]


_CONTEXTS = (
    (
        "A surveyor measures the depth of a river at {h} m intervals across its {w} m width. The depths, in metres, from one bank to the other are {vals}. "
        "Use Simpson's rule with 4 strips to estimate the area of the river's cross-section, in square metres.",
        _bell,
    ),
    (
        "The flow rate of water into a tank, in litres per minute, is recorded every {h} minutes for {w} minutes: {vals}. "
        "Use Simpson's rule with 4 strips to estimate the volume of water, in litres, that flowed into the tank over this time.",
        _rising_then_easing,
    ),
    (
        "A block of land is {w} m long. Its width, in metres, is measured every {h} m along its length: {vals}. "
        "Use Simpson's rule with 4 strips to estimate the area of the block, in square metres.",
        _plot_widths,
    ),
)


def _fmt_table(ys) -> str:
    return ", ".join(f"{float(y):.1f}" if y.denominator != 1 else str(int(y)) for y in ys)


def _medium_table(rng):
    for _ in attempts(500, "a table whose Simpson estimate is clear of a rounding tie"):
        h = rng.choice((2, 3, 4, 5))
        stem_t, make_ys = rng.choice(_CONTEXTS)
        ys = make_ys(rng)
        if len(set(ys)) < 4:
            continue
        est = simpson(ys, h)
        if clear_of_boundary(est, 2, Fraction(1, 12)):
            break
    _check_simpson(ys, Fraction(h), est)
    stem = stem_t.format(h=h, w=4 * h, vals=_fmt_table(ys))
    wrong = _wrong_estimates(ys, h)
    wsum = sum(w * y for w, y in zip(weights(4), ys))
    working = [
        f"Strip width h = {h}. Weights 1, 4, 2, 4, 1 give the sum {decimal_text(wsum)}.",
        f"Estimate = ({h}/3) x {decimal_text(wsum)} = {float(est):.4f}.",
    ]
    return single(stem, number_part("Give the estimate correct to 2 decimal places.", est, 2, 1, working, wrong))


def _medium_formula(rng):
    for _ in attempts(500, "a quartic with distinct positive ordinates"):
        coeffs = [
            rng.choice((1, 1, 2)),
            rng.choice((-2, -1, 0, 1, 2)),
            rng.choice((-3, -1, 0, 1, 2)),
            rng.choice((-2, 0, 1, 3)),
            rng.choice((1, 2, 4)),
        ]
        h = rng.choice((Fraction(1), Fraction(1), Fraction(1, 2)))
        start = rng.choice((0, 1))
        ys = _poly_values_q(coeffs, start, h, 4)
        est = simpson(ys, h)
        if len(set(ys)) >= 4 and min(ys) > 0 and clear_of_boundary(est, 2, Fraction(1, 12)):
            break
    _check_simpson(ys, h, est)
    end = start + 4 * h
    stem = f"Use Simpson's rule with 4 strips to estimate the area under y = {_poly_text(coeffs)} from x = {start} to x = {decimal_text(Fraction(end))}."
    wrong = _wrong_estimates(ys, h)
    wsum = sum(w * y for w, y in zip(weights(4), ys))
    working = [
        f"h = {decimal_text(h)}; the ordinates are {', '.join(decimal_text(y) for y in ys)}.",
        f"Estimate = ({decimal_text(h)}/3) x {decimal_text(wsum)} = {float(est):.4f}.",
    ]
    return single(stem, number_part("Give the estimate correct to 2 decimal places.", est, 2, 1, working, wrong))


def _poly_values_q(coeffs, start, h, n):
    deg = len(coeffs) - 1
    return [sum(Fraction(c) * (start + i * h) ** (deg - j) for j, c in enumerate(coeffs)) for i in range(n + 1)]


# -- hard ---------------------------------------------------------------------------------------------------------


def _hard_missing(rng):
    """One ordinate is unknown; Simpson's estimate is stated; find it."""
    for _ in attempts(500, "a missing ordinate that is a short decimal"):
        n = rng.choice((4, 4, 6))
        h = rng.choice((Fraction(3, 2), Fraction(3), Fraction(6), Fraction(3, 4)))
        ys = [Fraction(rng.randint(2, 30), 2) for _ in range(n + 1)]
        j = rng.randint(1, n - 1)
        est = simpson(ys, h)
        if decimal_places(est, 4) is not None and len(set(ys)) >= 4 and ys[j] != ys[j - 1]:
            break
    _check_simpson(ys, h, est)
    xs = [h * i for i in range(n + 1)]
    cells = ", ".join(("k" if i == j else decimal_text(y)) for i, y in enumerate(ys))
    x_cells = ", ".join(decimal_text(v) for v in xs)
    stem = (
        f"A function f takes the values {cells} at x = {x_cells}. Simpson's rule with {n} strips gives "
        f"{decimal_text(est)} as an estimate of the integral of f from x = 0 to x = {decimal_text(xs[-1])}."
    )
    ws = weights(n)
    rest = sum(w * y for i, (w, y) in enumerate(zip(ws, ys)) if i != j)
    target = est * 3 / h  # the weighted sum
    key = (target - rest) / ws[j]
    check(key == ys[j], "the recovered ordinate disagrees")
    other = 4 if ws[j] == 2 else 2
    wrong = [
        ((target - rest) / other, "simpson-missing-ordinate-weight"),
        (target - rest, "simpson-missing-ordinate-weight"),
        ((est - rest) / ws[j], "simpson-wrong-multiplier"),
        ((est * 2 / h - rest) / ws[j], "simpson-wrong-multiplier"),
        ((est * 3 / (2 * h) - rest) / ws[j], "simpson-wrong-multiplier"),
        ((est * 3 / h - rest) / (ws[j] * 2), "simpson-missing-ordinate-weight"),
        ((target - rest + ys[j]) / ws[j], "simpson-missing-ordinate-weight"),
    ]
    working = [
        f"The weighted sum is 3 x {decimal_text(est)} / {decimal_text(h)} = {decimal_text(target)}.",
        f"Ordinate {j} has weight {ws[j]}: {ws[j]}k = {decimal_text(target)} - {decimal_text(rest)}, so k = {decimal_text(ys[j])}.",
    ]
    wrong = [(v, t) for v, t in wrong if (Fraction(v) * 10).denominator == 1]  # one decimal place, like the key
    return single(stem, exact_number_part("Find the value of k.", ys[j], 2, working, wrong, min_places=1, max_places=1))


def _hard_six_quartic(rng):
    for _ in attempts(500, "a quartic with a clear Simpson estimate"):
        coeffs = [
            rng.choice((1, 1, 2)),
            rng.choice((-3, -2, -1, 0, 1, 2)),
            rng.choice((-6, -3, -1, 0, 2, 4)),
            rng.choice((-4, -2, 0, 1, 3)),
            rng.choice((1, 2, 5)),
        ]
        h = rng.choice((Fraction(1, 2), Fraction(1, 2), Fraction(1)))
        start = rng.choice((0, 1))
        ys = _poly_values_q(coeffs, start, h, 6)
        est = simpson(ys, h)
        if min(ys) > 0 and clear_of_boundary(est, 2, Fraction(1, 12)) and len(set(ys)) >= 6:
            break
    _check_simpson(ys, h, est)
    end = start + 6 * h
    f = sum(c * x ** (4 - j) for j, c in enumerate(coeffs))
    exact = sympy.integrate(f, (x, start, sympy.Rational(end.numerator, end.denominator)))
    check(sympy.Rational(est.numerator, est.denominator) != exact, "a quartic should not integrate exactly")
    stem = f"Use Simpson's rule with 6 strips to estimate the integral of f(x) = {_poly_text(coeffs)} from x = {start} to x = {decimal_text(Fraction(end))}."
    wrong = _wrong_estimates(ys, h)
    wrong.insert(0, (_weighted(ys, [1, 4, 2, 4, 4, 2, 1], h), "simpson-weights-swapped"))
    wsum = sum(w * y for w, y in zip(weights(6), ys))
    working = [
        f"h = {decimal_text(h)}; weights 1, 4, 2, 4, 2, 4, 1 on the ordinates {', '.join(decimal_text(y) for y in ys)}.",
        f"Estimate = ({decimal_text(h)}/3) x {decimal_text(wsum)} = {float(est):.4f}.",
    ]
    return single(stem, number_part("Give the estimate correct to 2 decimal places.", est, 2, 2, working, wrong))


@generator(
    "r4e_simpsons_rule",
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4e_simpsons_rule(rng, tier: str = "easy"):
    """Simpson's rule on 2, 4 or 6 strips; see the module docstring for the ladder."""
    if tier == "easy":
        return draw(_easy, rng)
    if tier == "medium":
        return draw(_medium_table if rng.random() < 0.55 else _medium_formula, rng)
    if tier == "hard":
        return draw(_hard_missing if rng.random() < 0.5 else _hard_six_quartic, rng)
    raise ValueError(f"unknown tier {tier!r}")


# -- exam chain ---------------------------------------------------------------------------------------------------


# (text of f, a function of x, exact antiderivative difference as a SymPy expression of the end points, a, b, h)
def _spans(starts, widths):
    return tuple((a, a + w) for a in starts for w in widths)


_CURVES = (
    ("1/x", lambda t: 1 / t, lambda a, b: sympy.log(b) - sympy.log(a), _spans((1, 2, 3), (2, 4, 6, 8))),
    ("e^x", sympy.exp, lambda a, b: sympy.exp(b) - sympy.exp(a), _spans((0, 1), (2, 3, 4))),
    (
        "sqrt(x)",
        sympy.sqrt,
        lambda a, b: sympy.Rational(2, 3) * (b ** sympy.Rational(3, 2) - a ** sympy.Rational(3, 2)),
        _spans((1, 4, 9), (4, 8, 12)),
    ),
    ("1/(1 + x^2)", lambda t: 1 / (1 + t**2), lambda a, b: sympy.atan(b) - sympy.atan(a), _spans((0, 1, 2), (2, 4, 6))),
    ("ln(x)", sympy.log, lambda a, b: b * sympy.log(b) - b - a * sympy.log(a) + a, _spans((1, 2, 3), (2, 4, 6, 8))),
)
_CHAIN_STEMS = (
    "The integral of f(x) = {f} from x = {a} to x = {b} is to be estimated using Simpson's rule with 4 strips. Its exact value is {exact}, correct to 4 decimal places.",
    "Let f(x) = {f}. Simpson's rule with 4 equal strips is used to estimate the integral of f from x = {a} to x = {b}, whose exact value is {exact} to 4 decimal places.",
)


def _round4(value) -> Fraction:
    from ..core import round_half_up

    return round_half_up(real(value), 4)


def _chain(rng):
    text, f, exact_fn, intervals = rng.choice(_CURVES)
    a, b = rng.choice(intervals)
    n = 4
    h = Fraction(b - a, n)
    xs = [a + h * i for i in range(n + 1)]
    ys = [real(f(sympy.Rational(v.numerator, v.denominator))) for v in xs]
    wsum = sum((w * y for w, y in zip(weights(n), ys)), Fraction(0))
    est = h * wsum / 3
    exact = real(exact_fn(sympy.Integer(a), sympy.Integer(b)))
    exact4 = _round4(exact_fn(sympy.Integer(a), sympy.Integer(b)))
    error = abs(est - exact4)
    for value in (wsum, est, error):
        if not clear_of_boundary(value, 4, Fraction(1, 10)):
            raise Redraw("a key too close to a rounding tie")
    if error < Fraction(5, 10000):
        raise Redraw("an error that rounds to almost nothing")
    # independent route: SymPy evaluates the same sum symbolically
    sym = sum(w * f(sympy.Rational(v.numerator, v.denominator)) for w, v in zip(weights(n), xs))
    check(abs(real(sym) - wsum) < Fraction(1, 10**30), "weighted sum disagrees with SymPy")
    check(
        abs(real(sympy.Rational(h.numerator, h.denominator) * sym / 3) - est) < Fraction(1, 10**30),
        "estimate disagrees",
    )
    stem = phrase(rng, _CHAIN_STEMS, f=text, a=a, b=b, exact=f"{float(exact4):.4f}")
    trap_sum = ys[0] + ys[-1] + 2 * sum(ys[1:-1])
    swapped_sum = ys[0] + ys[-1] + 4 * ys[2] + 2 * (ys[1] + ys[3])
    inner4_sum = ys[0] + ys[-1] + 4 * sum(ys[1:-1])

    part_a = fraction_part(
        "Find the width h of each strip.",
        h,
        1,
        [f"h = ({b} - {paren_negative(a)})/4 = {rational(h)}."],
        [
            (Fraction(b - a, n + 1), "simpson-strips-counted-as-ordinates"),
            (Fraction(b - a, 2), "simpson-strips-counted-as-ordinates"),
            (Fraction(b - a), "simpson-uses-interval-length-as-h"),
        ],
        sensible_range=(0, None),
    )
    part_b = number_part(
        "Find y0 + 4y1 + 2y2 + 4y3 + y4, where y0, ..., y4 are the values of f at the ends of the strips.",
        wsum,
        4,
        2,
        [f"The ordinates are {', '.join(f'{float(y):.4f}' for y in ys)}.", f"The weighted sum is {float(wsum):.4f}."],
        [
            (trap_sum, "simpson-trapezoidal-weights"),
            (swapped_sum, "simpson-weights-swapped"),
            (inner4_sum, "simpson-all-inner-weight-4"),
        ],
        sensible_range=(0, None),
    )
    part_c = number_part(
        "Hence find the Simpson's rule estimate of the integral.",
        est,
        4,
        2,
        [f"Estimate = (h/3) x {float(wsum):.4f} = {float(est):.4f}."],
        [
            (h * wsum / 2, "simpson-wrong-multiplier"),
            (h * trap_sum / 2, "simpson-trapezoidal-weights"),
            (h * swapped_sum / 3, "simpson-weights-swapped"),
        ],
        follow_through="part_a*part_b/3",
        sensible_range=(0, None),
    )
    part_d = number_part(
        "Find the absolute error of the estimate, using the exact value given.",
        error,
        4,
        2,
        [f"|{float(est):.4f} - {float(exact4):.4f}| = {float(error):.4f}."],
        [
            (-error, "error-sign-not-absolute"),
            (abs(h * trap_sum / 2 - exact4), "simpson-trapezoidal-weights"),
            (est, "reports-estimate-not-error"),
            (abs(h * swapped_sum / 3 - exact4), "simpson-weights-swapped"),
        ],
        follow_through=f"abs(part_c - {float(exact4):.4f})",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


def rational(v: Fraction) -> str:
    return f"{v.numerator}/{v.denominator}" if v.denominator != 1 else str(v.numerator)


@generator("r4e_simpsons_rule_chain", level=17, presets={"exam": dict(tier="exam")})
def r4e_simpsons_rule_chain(rng, tier: str = "exam"):
    """Simpson's rule on 4 strips for a stated curve: strip width, weighted sum, estimate, absolute error."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return draw(_chain, rng)
