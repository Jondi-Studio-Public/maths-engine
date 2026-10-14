"""Exam-tier source for `ucv1-regression-residual-diagnostic`: how well a fitted line describes five observations.

The stem gives five observations (x = 1 to 5) and their least-squares line. The parts carry the diagnosis through to
a number a learner can interpret:

(a) the residual of one named observation                      number, 2 marks
(b) the sum of squared residuals SSE                           number, 2 marks
(c) the total sum of squares SST about the mean of y           number, 1 mark
(d) the coefficient of determination r^2, 3 d.p.               number, 2 marks, 1 - part_b / part_c

## Construction

The observations are an integer-valued noisy line. For x = 1..5 the least-squares slope is Sxy / 10 and the
intercept is mean(y) - 3 * slope, so both are exact to one decimal place and every residual is exact to one decimal
place; SSE and SST are then exact to two. Nothing is rounded except r^2, whose canonical value is checked against
Sxy^2 / (Sxx * SST), an independent route, and kept away from a rounding tie. The residual asked for in (a) is
never zero.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator, round_half_up
from .exd_x_r3i_support import number_part

#: (x quantity, y quantity, base range, slope range, noise)
_CONTEXTS = (
    ("hours of study per week", "test mark", (38, 60), (4, 9), 5),
    ("number of training sessions per week", "5 km run time (minutes)", (30, 36), (-3, -1), 2),
    ("number of advertisements run", "weekly sales ($ hundreds)", (20, 45), (5, 12), 6),
    ("weeks since planting", "seedling height (cm)", (5, 12), (2, 5), 2),
    ("number of staff on shift", "customers served per hour", (10, 30), (6, 14), 5),
)


def _near_tie(value: Fraction, dp: int) -> bool:
    scaled = value * 10**dp
    return abs((scaled % 1) - Fraction(1, 2)) < Fraction(1, 50)


@generator("exd_x_r3i_regression_fit", level=17, presets={"exam": dict(tier="exam")})
def exd_x_r3i_regression_fit(rng, tier: str = "exam") -> dict:
    """Residual, SSE, SST and r^2 for a stated least-squares line through five points."""
    check(tier == "exam", f"unknown tier {tier!r}")
    xq, yq, (b_lo, b_hi), (s_lo, s_hi), noise = rng.choice(_CONTEXTS)
    xs = [1, 2, 3, 4, 5]
    for _ in attempts(1000, "five points with a mid-strength fit"):
        base = rng.randint(b_lo, b_hi)
        slope0 = rng.randint(s_lo, s_hi) if s_lo > 0 else -rng.randint(-s_hi, -s_lo)
        ys = [base + slope0 * x + rng.randint(-noise, noise) for x in xs]
        sy = sum(ys)
        ybar = Fraction(sy, 5)
        sxy = sum((x - 3) * y for x, y in zip(xs, ys))
        slope = Fraction(sxy, 10)
        intercept = ybar - 3 * slope
        fitted = [intercept + slope * x for x in xs]
        resid = [y - f for y, f in zip(ys, fitted)]
        sse = sum(e * e for e in resid)
        sst = sum((y - ybar) ** 2 for y in ys)
        k = rng.choice(xs)
        if sse == 0 or sst == 0 or resid[k - 1] == 0 or any(y <= 0 for y in ys):
            continue
        r2 = round_half_up(1 - sse / sst, 3)
        if not (Fraction(3, 10) <= r2 <= Fraction(995, 1000)) or _near_tie(1 - sse / sst, 3):
            continue
        break
    # Independent checks: residuals of a least-squares line sum to zero and are orthogonal to x; r^2 = Sxy^2/(Sxx SST).
    check(sum(resid) == 0 and sum(x * e for x, e in zip(xs, resid)) == 0, "the line is not the least-squares line")
    check(1 - sse / sst == Fraction(sxy * sxy, 10) / sst, "r^2 disagrees with Sxy^2 / (Sxx SST)")
    check(all((100 * e * e).denominator == 1 and (10 * e).denominator == 1 for e in resid), "residuals not exact")

    sign = "+" if slope >= 0 else "-"
    line = f"ŷ = {float(intercept):.1f} {sign} {abs(float(slope)):.1f}x"
    data = ", ".join(f"({x}, {y})" for x, y in zip(xs, ys))
    stem = (
        f"A small study records the {xq} (x) and the {yq} (y) for five cases: {data}. "
        f"The least-squares regression line for these data is {line}."
    )
    e_k = resid[k - 1]
    adjacent = resid[k % 5]
    sum_abs = sum(abs(e) for e in resid)
    part_a = number_part(
        f"Find the residual for the case with x = {k}, correct to 1 decimal place.",
        e_k,
        1,
        2,
        [
            f"Predicted y at x = {k} is {float(fitted[k - 1]):.1f}; residual = {ys[k - 1]} - {float(fitted[k - 1]):.1f} = {float(e_k):.1f}."
        ],
        [
            ("sign-of-result-reversed", -e_k),
            ("reports-value-as-residual", fitted[k - 1]),
            ("read-wrong-row-or-column", adjacent),
            ("reports-value-as-residual", Fraction(ys[k - 1])),
        ],
    )
    part_b = number_part(
        "Find the sum of the squares of the residuals for all five cases, correct to 2 decimal places.",
        sse,
        2,
        2,
        ["SSE = " + " + ".join(f"({float(e):.1f})²" for e in resid) + f" = {float(sse):.2f}."],
        [
            ("summed-sizes-not-squares", sum_abs),
            ("squares-the-sum-of-absolute-residuals", sum_abs**2),
            ("divided-sum-of-squares-by-count", sse / 5),
            ("summed-signed-values", sum(resid)),
        ],
        sensible_range=(0, None),
    )
    part_c = number_part(
        "Find the sum of the squares of the differences between each y value and the mean of the y values, correct "
        "to 2 decimal places.",
        sst,
        2,
        1,
        [f"Mean of y = {float(ybar):.1f}; SST = Σ(y - {float(ybar):.1f})² = {float(sst):.2f}."],
        [
            ("summed-sizes-not-squares", sum(abs(y - ybar) for y in ys)),
            ("divided-sum-of-squares-by-count", sst / 4),
            ("sums-squares-of-y-values-for-sst", Fraction(sum(y * y for y in ys))),
            ("summed-signed-values", sum(y - ybar for y in ys)),
        ],
        sensible_range=(0, None),
    )
    r2_true = 1 - sse / sst
    part_d = number_part(
        "Hence find the proportion of the variation in y explained by the line, r², correct to 3 decimal places.",
        r2,
        3,
        2,
        [f"r² = 1 - SSE/SST = 1 - {float(sse):.2f}/{float(sst):.2f} = {float(r2):.3f}."],
        [
            ("unexplained-proportion-for-explained", sse / sst),
            ("r-reported-as-r-squared", Fraction(round(float(r2_true) ** 0.5 * 10**6), 10**6)),
            ("summed-sizes-not-squares", 1 - sum_abs / sst),
        ],
        follow_through="1 - part_b/part_c",
        sensible_range=(0, 1),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
