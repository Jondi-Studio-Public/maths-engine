"""Exam source of `ucv1-multivariate-regression`: slopes from summary sums, then the intercept and a prediction.

The practice tiers solve the normal equations for a handful of listed points. An exam question gives the summary a
statistics course works from, the means and the centred sums of squares and products of n observations, and the
learner works through four linked answers for the fitted model y = b0 + b1 x1 + b2 x2:

(a) b1, by eliminating b2 from the two centred normal equations;
(b) b2, by substituting b1 back (follows through from (a));
(c) the intercept b0 = mean(y) - b1 mean(x1) - b2 mean(x2) (follows through from (a) and (b));
(d) the fitted value at a stated point (follows through from (a), (b) and (c)).

Built backwards. The n values of each feature are drawn with whole-number means, and the centred sums are computed
from them, so the summary describes real data. The slopes and intercept are whole numbers, and the cross sums with y
are S1y = b1 S11 + b2 S12 and S2y = b1 S12 + b2 S22, so the normal equations hold exactly (the residuals of the
underlying data are orthogonal to both features). An independent check solves the 2 by 2 system with SymPy.
"""

from __future__ import annotations

import random
from fractions import Fraction

from sympy import Matrix

from mathgen.core import attempts, check, generator, latex_answer

from .exd_x_r3d_common import tagged
from ._exd_r3_distinct import distinct_part_answers

_PROMPT_A = ("Find the value of b1.", "Calculate b1.")
_PROMPT_B = ("Hence find b2.", "Using your value of b1, find b2.")
_PROMPT_C = ("Hence find the intercept b0.", "Using your slopes, find b0.")
_PROMPT_D = (
    "Use the fitted model to predict y when x1 = {u} and x2 = {v}.",
    "Find the fitted value of y at x1 = {u}, x2 = {v}.",
)
_STEMS = (
    "A model y = b0 + b1·x1 + b2·x2 is fitted by least squares to {n} observations. The means are {m1} (for x1), "
    "{m2} (for x2) and {my} (for y). The centred sums are S11 = {s11}, S12 = {s12}, S22 = {s22}, S1y = {s1y} and "
    "S2y = {s2y}, where Sjk is the sum of (xj - mean of xj)(xk - mean of xk) over the observations and Sjy is the "
    "sum of (xj - mean of xj)(y - mean of y).",
    "Least squares is used to fit y = b0 + b1·x1 + b2·x2 to {n} data points. The mean of x1 is {m1}, the mean of "
    "x2 is {m2} and the mean of y is {my}. With deviations taken from these means, S11 = {s11}, S12 = {s12}, "
    "S22 = {s22}, S1y = {s1y} and S2y = {s2y} (Sjk is the sum of products of deviations of xj and xk, and Sjy "
    "the sum of products of deviations of xj and y).",
)


def _centred_sums(x1, x2):
    n = len(x1)
    m1, m2 = Fraction(sum(x1), n), Fraction(sum(x2), n)
    s11 = sum((a - m1) ** 2 for a in x1)
    s22 = sum((b - m2) ** 2 for b in x2)
    s12 = sum((a - m1) * (b - m2) for a, b in zip(x1, x2))
    return m1, m2, s11, s12, s22


@generator("exd_x_r3d_regression_summary_chain", level=17, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3d_regression_summary_chain(rng: random.Random, tier: str = "exam") -> dict:
    """b1, b2, b0 and a fitted value from the centred summary sums of a two-feature regression."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(4000, "a summary with whole-number slopes and distractors"):
        n = rng.choice([8, 10, 12])
        x1 = [rng.randint(1, 9) for _ in range(n)]
        x2 = [rng.randint(1, 9) for _ in range(n)]
        if sum(x1) % n or sum(x2) % n:
            continue
        m1, m2, s11, s12, s22 = _centred_sums(x1, x2)
        if m1.denominator != 1 or m2.denominator != 1:
            continue
        m1, m2 = int(m1), int(m2)
        s11, s12, s22 = int(s11), int(s12), int(s22)
        det = s11 * s22 - s12 * s12
        if det <= 0 or s12 == 0 or abs(s12) * 3 < min(s11, s22) or max(s11, s22) > 60:
            continue
        b1 = rng.choice([v for v in range(-6, 7) if v != 0])
        b2 = rng.choice([v for v in range(-6, 7) if v != 0])
        b0 = rng.choice([v for v in range(-8, 13) if v != 0])
        s1y, s2y = b1 * s11 + b2 * s12, b1 * s12 + b2 * s22
        my = b0 + b1 * m1 + b2 * m2
        u, v = rng.randint(1, 9), rng.randint(1, 9)
        if (u, v) == (m1, m2):
            continue
        fitted = b0 + b1 * u + b2 * v

        def whole(num, den):
            return num // den if den != 0 and num % den == 0 else None

        a_cands = [
            (whole(s1y, s11), "second-feature-ignored"),
            (b2, "slopes-swapped"),
            (whole(s22 * s1y + s12 * s2y, det), "sign-error-moving-term"),
            (-b1, "coefficient-sign-flipped"),
        ]
        b_cands = [
            (whole(s2y, s22), "second-feature-ignored"),
            (b1, "slopes-swapped"),
            (whole(s2y + s12 * b1, s22), "sign-error-moving-term"),
            (-b2, "coefficient-sign-flipped"),
        ]
        c_cands = [
            (my, "intercept-taken-as-mean-of-y"),
            (my - b1 * m1, "feature-term-dropped-from-intercept"),
            (my + b1 * m1 + b2 * m2, "sign-error-moving-term"),
            (my - b2 * m1 - b1 * m2, "slopes-swapped"),
        ]
        d_cands = [
            (b1 * u + b2 * v, "intercept-omitted-in-prediction"),
            (b0 + b1 * v + b2 * u, "feature-values-swapped"),
            (b0 + b1 * u, "feature-term-dropped-from-prediction"),
            (my, "prediction-by-mean"),
        ]
        a_d = tagged(str(b1), [c for c in a_cands if c[0] is not None])
        b_d = tagged(str(b2), [c for c in b_cands if c[0] is not None])
        c_d = tagged(str(b0), c_cands)
        d_d = tagged(str(fitted), d_cands)
        if min(len(a_d), len(b_d)) >= 2 and len(c_d) >= 3 and len(d_d) >= 3:
            break
    # Independent route: solve the centred normal equations with SymPy.
    solved = Matrix([[s11, s12], [s12, s22]]).LUsolve(Matrix([s1y, s2y]))
    check(list(solved) == [b1, b2], "the normal equations do not give the slopes")
    check(my - solved[0] * m1 - solved[1] * m2 == b0, "the intercept disagrees")

    stem = rng.choice(_STEMS).format(n=n, m1=m1, m2=m2, my=my, s11=s11, s12=s12, s22=s22, s1y=s1y, s2y=s2y)

    def part(prompt, value, marks, distractors, working, follow_through=None, sensible_range=None):
        out = dict(
            prompt=prompt,
            answer_kind="number",
            answer=value,
            answer_text=str(value),
            answer_latex=latex_answer(value),
            marks=marks,
            working=working,
            distractors=distractors,
        )
        if follow_through:
            out["follow_through"] = follow_through
        if sensible_range:
            out["sensible_range"] = sensible_range
        return out

    return dict(
        text=stem,
        parts=[
            part(
                rng.choice(_PROMPT_A),
                b1,
                3,
                a_d,
                [f"{s11}·b1 + {s12}·b2 = {s1y} and {s12}·b1 + {s22}·b2 = {s2y}; eliminating b2 gives b1 = {b1}."],
            ),
            part(
                rng.choice(_PROMPT_B),
                b2,
                1,
                b_d,
                [f"{s12}·({b1}) + {s22}·b2 = {s2y}, so b2 = {b2}."],
                follow_through=f"({s2y} - ({s12})*part_a)/{s22}",
            ),
            part(
                rng.choice(_PROMPT_C),
                b0,
                2,
                c_d,
                [f"b0 = {my} - ({b1})({m1}) - ({b2})({m2}) = {b0}."],
                follow_through=f"{my} - part_a*{m1} - part_b*{m2}",
            ),
            part(
                rng.choice(_PROMPT_D).format(u=u, v=v),
                fitted,
                1,
                d_d,
                [f"y = {b0} + ({b1})({u}) + ({b2})({v}) = {fitted}."],
                follow_through=f"part_c + part_a*{u} + part_b*{v}",
            ),
        ],
    )
