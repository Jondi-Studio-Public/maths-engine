"""Generates `exd_x_r3j_total_variance_three` questions: the law of total variance over three unequal cases.

Skill: `ucv1-total-variance-finite` (exam tier). The practice tiers have two equally likely cases; a first-year
probability exam has three cases with unequal probabilities and builds the answer in its two pieces:

(a) E[Y], the probability-weighted mean of the conditional means                    fraction, 1 mark
(b) E[Var(Y | X)], the weighted mean of the conditional variances                   fraction, 1 mark
(c) Var(E[Y | X]) = sum p_i mu_i^2 - E[Y]^2                                         fraction, 2 marks, S2 - part_a^2
(d) Var(Y) = (b) + (c)                                                              fraction, 1 mark,  part_b + part_c

## Construction

The probabilities are whole percentages k_i / N with N one of 4, 5, 10 or 20 (so the stem can state them as
percentages), three positive weights that are not all equal. The conditional means are small integers around a
believable level for the setting, not all equal (otherwise Var(E[Y | X]) is zero), and the conditional variances
are positive integers. Every answer is an exact rational computed with `Fraction`.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._exd_x_r3j_kit import fraction_part

_SETTINGS = [
    # (opening with {cases} and {percents}, case names, Y description, unit, base mean, mean spread, max variance)
    (
        "A call centre takes calls in three shifts: {cases}, with {percents} of calls respectively.",
        ["the morning shift", "the afternoon shift", "the night shift"],
        "the length of a call",
        "minutes",
        8,
        4,
        9,
    ),
    (
        "A courier company delivers through three depots: {cases}, which handle {percents} of parcels respectively.",
        ["the Parramatta depot", "the Geelong depot", "the Toowoomba depot"],
        "the delivery time of a parcel",
        "hours",
        30,
        9,
        16,
    ),
    (
        "A farm collects eggs from three flocks: {cases}, which supply {percents} of the eggs respectively.",
        ["the barn flock", "the free-range flock", "the pasture flock"],
        "the mass of an egg",
        "grams",
        60,
        6,
        16,
    ),
    (
        "A school bus company runs three routes: {cases}, carrying {percents} of students respectively.",
        ["the coastal route", "the town route", "the hill route"],
        "a student's journey time",
        "minutes",
        25,
        10,
        25,
    ),
]
_DENOMINATORS = [4, 5, 10, 20]


def _percent_list(weights: list[Fraction]) -> str:
    texts = [f"{int(w * 100)}%" for w in weights]
    return f"{texts[0]}, {texts[1]} and {texts[2]}"


def _case_list(names: list[str]) -> str:
    return f"{names[0]}, {names[1]} and {names[2]}"


@generator("exd_x_r3j_total_variance_three", level=17, presets={"exam": {}})
@distinct_part_answers
def exd_x_r3j_total_variance_three(rng):
    opening, names, what, unit, base, spread, vmax = rng.choice(_SETTINGS)
    for _ in attempts(1000, "three unequal probabilities and distinct conditional means"):
        n = rng.choice(_DENOMINATORS)
        k1, k2 = rng.randint(1, n - 2), rng.randint(1, n - 2)
        k3 = n - k1 - k2
        if k3 < 1 or len({k1, k2, k3}) < 2:
            continue
        means = [base + rng.randint(-spread, spread) for _ in range(3)]
        if len(set(means)) < 2:
            continue
        break
    weights = [Fraction(k, n) for k in (k1, k2, k3)]
    variances = [rng.randint(1, vmax) for _ in range(3)]

    e_y = sum(w * m for w, m in zip(weights, means))
    e_var = sum(w * v for w, v in zip(weights, variances))
    s2 = sum(w * m * m for w, m in zip(weights, means))
    var_e = s2 - e_y**2
    total = e_var + var_e
    check(var_e > 0 and total > e_var, "the variance of the conditional means must be positive")
    # independent route: Var(Y) = E[Y^2] - E[Y]^2 with E[Y^2 | case] = variance + mean^2
    second = sum(w * (v + m * m) for w, v, m in zip(weights, variances, means))
    check(second - e_y**2 == total, "law of total variance disagrees with the second-moment route")

    rotated = weights[1:] + weights[:1]
    plain_mean = Fraction(sum(means), 3)
    plain_var = Fraction(sum(variances), 3)
    var_of_means_plain = sum((m - plain_mean) ** 2 for m in means) / 3

    cases = [
        f"{n_} (mean {m} {unit}, variance {v})"
        for n_, m, v in zip(["case one", "case two", "case three"], means, variances)
    ]
    conditional = "; ".join(f"on {name} Y has mean {m} and variance {v}" for name, m, v in zip(names, means, variances))
    stem = (
        f"{opening.format(cases=_case_list(names), percents=_percent_list(weights))} "
        f"Let Y be {what}, in {unit}, and let X record which of the three applies: {conditional}."
    )
    del cases

    parts = [
        fraction_part(
            "Find E[Y].",
            e_y,
            1,
            [f"E[Y] = Σ p·μ = {e_y}."],
            [
                (plain_mean, "unweighted-instead-of-weighted"),
                (sum(w * m for w, m in zip(rotated, means)), "probabilities-misassigned-to-cases"),
                (Fraction(sum(m * k for m, k in zip(means, (k1, k2, k3)))), "weights-as-counts-not-probabilities"),
                (sum(w * m for w, m in zip(reversed(weights), means)), "probabilities-misassigned-to-cases"),
                (Fraction(sum(means)), "unweighted-instead-of-weighted"),
            ],
        ),
        fraction_part(
            "Find E[Var(Y | X)].",
            e_var,
            1,
            [f"E[Var(Y | X)] = Σ p·σ² = {e_var}."],
            [
                (plain_var, "unweighted-instead-of-weighted"),
                (sum(w * v for w, v in zip(rotated, variances)), "probabilities-misassigned-to-cases"),
                (Fraction(sum(v * k for v, k in zip(variances, (k1, k2, k3)))), "weights-as-counts-not-probabilities"),
                (sum(w * v for w, v in zip(reversed(weights), variances)), "probabilities-misassigned-to-cases"),
                (Fraction(sum(variances)), "unweighted-instead-of-weighted"),
            ],
        ),
        fraction_part(
            "Find Var(E[Y | X]).",
            var_e,
            2,
            [f"Var(E[Y | X]) = Σ p·μ² − (E[Y])² = {s2} − ({e_y})² = {var_e}."],
            [
                (s2, "forgot-to-subtract-square-of-mean"),
                (var_of_means_plain, "unweighted-instead-of-weighted"),
                (s2 - e_y, "forgot-to-square-the-mean"),
                (e_y**2, "gave-square-of-mean-for-variance"),
            ],
            follow_through=f"({s2.numerator}/{s2.denominator}) - part_a**2",
        ),
        fraction_part(
            "Hence find Var(Y).",
            total,
            1,
            [f"Var(Y) = E[Var(Y | X)] + Var(E[Y | X]) = {e_var} + {var_e} = {total}."],
            [
                (e_var, "dropped-variance-of-means"),
                (var_e, "dropped-mean-of-variances"),
                (e_var + s2, "forgot-to-subtract-square-of-mean"),
                (e_var - var_e, "subtracted-instead-of-added"),
            ],
            follow_through="part_b + part_c",
        ),
    ]
    return dict(text=stem, parts=parts)
