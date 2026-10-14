"""Residuals from a least-squares line: `smech_residuals` (easy, medium, hard) and `smech_residuals_exam`.

Skill: least-squares-residuals. A residual is observed minus predicted. The data are five observations at five
consecutive whole-number x values, which makes the least-squares slope and intercept exact to one decimal place and
every residual a multiple of 0.1, so no answer is a rounding.

Tiers, each a new decision:

* easy: one observation and the line. Predict, then subtract in the right order.
* medium: all five observations and the line. Find every residual, then pick the one of greatest size, keeping its sign.
* hard: the sum of the residuals. Either the line is the least-squares line and one residual is missing from a table
  (the residuals of a least-squares line add to zero), or the line is drawn by eye and the sum has to be computed.
* exam: a four-part chain. Predicted value, residual (follow-through), the residual of greatest size, and the sum of
  the squares of the residuals.

## Construction

The least-squares line is computed from the data with `Fraction`s, and checked a second way: the residuals add to zero
and are uncorrelated with x (`sum(x * r) == 0`), which only the least-squares line satisfies.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator, phrase
from ._smech_kit import dec, number_fields, number_part

# (x quantity, y quantity, first x range, intercept range, slope range, noise)
_CONTEXTS = (
    ("hours of study per week", "test score", (2, 8), (35, 55), (3, 6), 4),
    ("daily maximum temperature (°C)", "ice-cream sales", (18, 28), (-60, -20), (6, 12), 8),
    ("age of the car (years)", "advertised price ($ thousand)", (2, 9), (28, 36), (-3, -1), 2),
    ("hours of sunshine", "solar output (kWh)", (3, 9), (2, 8), (2, 4), 2),
    ("training sessions per week", "5 km time (minutes)", (1, 4), (30, 38), (-3, -1), 2),
)


def _line_text(a: Fraction, b: Fraction, ctx) -> str:
    sign = "+" if b >= 0 else "-"
    return f"predicted {ctx[1]} = {dec(a)} {sign} {dec(abs(b))} × {ctx[0]}"


def _dataset(rng):
    """Five points at consecutive x, their exact least-squares line and the five residuals."""
    for _ in attempts(1000, "a data set with a least-squares line that leaves distinct non-zero residuals"):
        ctx = rng.choice(_CONTEXTS)
        x0 = rng.randint(*ctx[2])
        a0, b0 = rng.randint(*ctx[3]), rng.randint(*ctx[4])
        xs = [x0 + i for i in range(5)]
        ys = [a0 + b0 * x + rng.randint(-ctx[5], ctx[5]) for x in xs]
        if min(ys) <= 0:
            continue
        xbar, ybar = Fraction(sum(xs), 5), Fraction(sum(ys), 5)
        b = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / 10
        a = ybar - b * xbar
        res = [y - (a + b * x) for x, y in zip(xs, ys)]
        if b == 0 or any(r == 0 for r in res) or len({abs(r) for r in res}) < 5:
            continue
        break
    # Second route: only the least-squares line leaves residuals that add to zero and are uncorrelated with x.
    check(sum(res) == 0 and sum(x * r for x, r in zip(xs, res)) == 0, "not the least-squares line")
    return ctx, xs, ys, a, b, res


def _points(xs, ys) -> str:
    return ", ".join(f"({x}, {y})" for x, y in zip(xs, ys))


def _extreme(res):
    """The residual of greatest size, sign kept."""
    return max(res, key=abs)


_DEF = "A residual is the observed value minus the predicted value."


# ------------------------------------------------------------------------------------------------ easy


def _easy(rng):
    ctx, xs, ys, a, b, res = _dataset(rng)
    k = rng.randrange(5)
    pred = a + b * xs[k]
    text = (
        f"For {ctx[1]} against {ctx[0]}, the least-squares line is {_line_text(a, b, ctx)}. One observation has "
        f"{ctx[0]} equal to {xs[k]} and {ctx[1]} equal to {ys[k]}. {_DEF} Find the residual for this observation, "
        f"correct to 1 decimal place."
    )
    value = res[k]
    check(value == ys[k] - pred and value + pred == ys[k], "residual and prediction disagree")
    wrong = [
        (pred - ys[k], "residual-sign-reversed"),
        (pred, "mech-predicted-reported-as-residual"),
        (ys[k] - b * xs[k], "mech-intercept-left-out"),
        (ys[k] - a, "mech-slope-term-left-out"),
    ]
    working = [
        f"Predicted value = {dec(a)} + ({dec(b)}) × {xs[k]} = {dec(pred)}.",
        f"Residual = {ys[k]} - {dec(pred)} = {dec(value)}.",
    ]
    return dict(text=text, working=working, **number_fields(value, 1, wrong))


# ------------------------------------------------------------------------------------------------ medium


def _medium(rng):
    for _ in attempts(1000, "a data set where the largest residual is not the first"):
        ctx, xs, ys, a, b, res = _dataset(rng)
        top = _extreme(res)
        # Make the sign matter on half the questions, and the position vary.
        if res.index(top) != 0 and (top < 0) == (rng.random() < 0.5):
            break
    text = (
        f"{ctx[0][0].upper() + ctx[0][1:]} (x) and {ctx[1]} (y) are recorded for five observations: {_points(xs, ys)}. The "
        f"least-squares line is {_line_text(a, b, ctx)}. {_DEF} Find the residual of greatest size (keep its sign), correct to "
        f"1 decimal place."
    )
    wrong = [
        (abs(top), "mech-sign-of-residual-dropped"),
        (max(res) if top < 0 else min(res), "mech-wrong-extreme-chosen"),
        (-top, "residual-sign-reversed"),
        (res[0], "mech-first-residual-only"),
    ]
    working = [f"Residuals: {', '.join(dec(r) for r in res)}.", f"The residual of greatest size is {dec(top)}."]
    return dict(text=text, working=working, **number_fields(top, 1, wrong))


# ------------------------------------------------------------------------------------------------ hard


def _by_eye(rng, a, b, xs, ys):
    """A hand-drawn line with whole-number coefficients whose residuals have mixed signs and do not sum to zero."""
    for _ in attempts(1000, "a by-eye line whose residuals do not sum to zero"):
        d = round(b) + rng.choice((-1, 0, 1))
        xbar = Fraction(sum(xs), 5)
        c = round(a + (b - d) * xbar) + rng.choice((-3, -2, -1, 1, 2, 3))
        res = [y - (c + d * x) for x, y in zip(xs, ys)]
        if sum(res) != 0 and min(res) < 0 < max(res) and sum(abs(r) for r in res) != abs(sum(res)):
            return Fraction(c), Fraction(d), res
    raise AssertionError("unreachable")  # pragma: no cover


def _hard(rng):
    ctx, xs, ys, a, b, res = _dataset(rng)
    if rng.random() < 0.5:
        k = rng.randrange(5)
        shown = [(x, r) for i, (x, r) in enumerate(zip(xs, res)) if i != k]
        listed = ", ".join(f"x = {x}: {dec(r)}" for x, r in shown)
        text = (
            f"The least-squares line for {ctx[1]} against {ctx[0]} is {_line_text(a, b, ctx)}. The residuals for "
            f"four of the five observations are {listed}. Find the residual for the observation with "
            f"{ctx[0]} equal to {xs[k]}, correct to 1 decimal place."
        )
        value = res[k]
        listed_sum = sum(r for _, r in shown)
        wrong = [
            (listed_sum, "mech-sum-sign-not-reversed"),
            (Fraction(0), "mech-missing-residual-taken-as-zero"),
            (listed_sum / 4, "mech-mean-of-the-others"),
            (a + b * xs[k], "mech-predicted-reported-as-residual"),
        ]
        check(value == -listed_sum, "the residuals do not add to zero")
        working = [
            "The residuals of a least-squares line add to zero.",
            f"Missing residual = -({' + '.join(f'({dec(r)})' for _, r in shown)}) = {dec(value)}.",
        ]
        return dict(text=text, working=working, **number_fields(value, 1, wrong))
    c, d, eye = _by_eye(rng, a, b, xs, ys)
    text = (
        f"A student draws a line by eye through the data on {ctx[1]} against {ctx[0]}: {_points(xs, ys)}. "
        f"The student's line is predicted {ctx[1]} = {dec(c)} {'+' if d >= 0 else '-'} {dec(abs(d))} × "
        f"{ctx[0]}. {_DEF} Find the sum of the five residuals about the student's line."
    )
    total = sum(eye)
    wrong = [
        (Fraction(0), "mech-residuals-always-sum-to-zero"),
        (sum(abs(r) for r in eye), "mech-absolute-residuals-summed"),
        (-total, "residual-sign-reversed"),
        (sum(r * r for r in eye), "mech-squared-residuals-summed"),
    ]
    working = [f"Residuals: {', '.join(dec(r) for r in eye)}.", f"Sum = {dec(total)}."]
    return dict(text=text, working=working, **number_fields(total, 0, wrong))


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator("smech_residuals", level=17, presets={t: {"tier": t} for t in _TIERS})
def smech_residuals(rng, tier: str = "easy"):
    """Residuals of a least-squares line: one, the largest of five, then the sum."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return _TIERS[tier](rng)


# ------------------------------------------------------------------------------------------------ exam


@generator("smech_residuals_exam", level=17, presets={"exam": {}})
def smech_residuals_exam(rng):
    """Predicted value, residual, residual of greatest size, sum of squared residuals for one data set."""
    for _ in attempts(1000, "a data set whose residual of greatest size is not at the chosen observation"):
        ctx, xs, ys, a, b, res = _dataset(rng)
        k = rng.randint(1, 3)
        top = _extreme(res)
        if res.index(top) != k:
            break
    pred = a + b * xs[k]
    sse = sum(r * r for r in res)
    check(ys[k] - pred == res[k] and sse == sum((y - a - b * x) ** 2 for x, y in zip(xs, ys)), "second route disagrees")
    stem = (
        f"{ctx[0][0].upper() + ctx[0][1:]} (x) and {ctx[1]} (y) are recorded for five observations: {_points(xs, ys)}. The "
        f"least-squares line is {_line_text(a, b, ctx)}. {_DEF}"
    )
    part_a = number_part(
        f"Find the predicted {ctx[1]} when {ctx[0]} is {xs[k]}, correct to 1 decimal place.",
        pred,
        1,
        1,
        [f"{dec(a)} + ({dec(b)}) × {xs[k]} = {dec(pred)}."],
        [
            (Fraction(ys[k]), "mech-observed-used-as-predicted"),
            (b * xs[k], "mech-intercept-left-out"),
            (a + b, "mech-x-value-replaced-by-one"),
            (b + a * xs[k], "mech-intercept-and-slope-swapped"),
        ],
        sensible=(0, None),
    )
    part_b = number_part(
        f"Find the residual for the observation with {ctx[0]} equal to {xs[k]}, correct to 1 decimal place.",
        res[k],
        1,
        1,
        [f"Residual = {ys[k]} - {dec(pred)} = {dec(res[k])}."],
        [
            (pred - ys[k], "residual-sign-reversed"),
            (pred, "mech-predicted-reported-as-residual"),
            (Fraction(ys[k]), "mech-observed-reported-as-residual"),
            (ys[k] - b * xs[k], "mech-intercept-left-out"),
        ],
        follow=f"{ys[k]} - part_a",
    )
    part_c = number_part(
        "Find the residual of greatest size over all five observations (keep its sign), correct to 1 decimal place.",
        top,
        1,
        2,
        [f"Residuals: {', '.join(dec(r) for r in res)}.", f"The residual of greatest size is {dec(top)}."],
        [
            (abs(top), "mech-sign-of-residual-dropped"),
            (max(res) if top < 0 else min(res), "mech-wrong-extreme-chosen"),
            (res[k], "mech-first-residual-only"),
            (-top, "residual-sign-reversed"),
        ],
    )
    part_d = number_part(
        "Find the sum of the squares of the five residuals, correct to 2 decimal places.",
        sse,
        2,
        2,
        [f"{' + '.join(f'({dec(r)})²' for r in res)} = {dec(sse)}."],
        [
            (sum(abs(r) for r in res), "mech-absolute-not-squared"),
            (top * top, "mech-one-residual-squared"),
            (sse / 5, "mech-mean-instead-of-sum"),
            (Fraction(0), "mech-residuals-always-sum-to-zero"),
        ],
        sensible=(0, None),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
