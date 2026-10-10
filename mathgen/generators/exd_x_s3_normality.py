"""Exam-tier source for `p8c_normality_approx_phat`: the sample whose proportion is being studied is a SUBGROUP of
a larger survey, so the learner must first work out the subgroup's own size and success count from percentages, then
apply the stated validity rule to that subgroup (not to the whole survey).

    n_sub = N * f / 100              (the subgroup's share of the survey)
    x     = n_sub * q / 100          (the subgroup members with the characteristic)

Both products to test are the integers ``x`` and ``n_sub - x``. The category is drawn first (uniformly over the four
outcomes) and the numbers are built to land in it, so every category is reachable. Neither product is ever exactly 10.
"""

from __future__ import annotations

import random

from ..core import check, generator, latex_answer, pick_distractors
from .p8c_normality_approx_build import (
    ALL_CATEGORIES,
    CATEGORY_BOTH_FAIL,
    CATEGORY_P_FAILS,
    CATEGORY_Q_FAILS,
    CATEGORY_VALID,
    RULE_SENTENCE,
)

_FRACTIONS = (10, 20, 25, 40, 50)  # subgroup share of the survey, in percent

_SETTINGS = (
    ("voters", "of the voters were under 30", "said they would vote for the new policy"),
    ("customers", "of the customers were first-time buyers", "returned the item"),
    ("students", "of the students were in the final year", "chose the elective"),
    ("residents", "of the residents lived within 1 km of the park", "used it every week"),
)


def _category(x: int, n_sub: int) -> str:
    p_ok, q_ok = x >= 10, n_sub - x >= 10
    if p_ok and q_ok:
        return CATEGORY_VALID
    if p_ok:
        return CATEGORY_Q_FAILS
    if q_ok:
        return CATEGORY_P_FAILS
    return CATEGORY_BOTH_FAIL


@generator(
    "exd_x_s3_normality",
    level=12,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_s3_normality(rng: random.Random, tier: str = "exam"):
    """Validity of the normal approximation to p-hat for a subgroup described by percentages."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    target = rng.choice(ALL_CATEGORIES)
    for _ in range(5000):
        n_sub = rng.choice(range(8, 161, 2))
        q = rng.choice(range(5, 100, 5))
        share = rng.choice(_FRACTIONS)
        if (n_sub * 100) % share:
            continue
        total = n_sub * 100 // share
        x = n_sub * q // 100
        if (n_sub * q) % 100 or x == 10 or n_sub - x == 10:
            continue
        if _category(x, n_sub) == target:
            break
    else:
        raise RuntimeError(f"no draw reached the category {target!r}")

    # Independent route: p-hat as a fraction of the subgroup, products from the percentages directly.
    prod_p = total * share * q / 10000
    prod_q = total * share * (100 - q) / 10000
    check(prod_p == x and prod_q == n_sub - x, "percentage route disagrees with the integer counts")
    check(_category(x, n_sub) == target, "category does not match the target")

    noun, group_clause, outcome_clause = rng.choice(_SETTINGS)
    text = (
        f"A survey of {total} {noun} found that {share}% {group_clause}. Of this group, {q}% {outcome_clause}. "
        f"{RULE_SENTENCE} Classify whether the normal approximation to the distribution of p-hat is valid "
        f"for the sample proportion of this group."
    )
    working = [
        f"The group has {total} × {share}/100 = {n_sub} members, of whom {q}% = {x} have the outcome.",
        f"n×p-hat = {x} and n×(1-p-hat) = {n_sub - x}; compare each with 10.",
    ]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=target,
        answer_text=target,
        answer_latex=latex_answer(target, plain=True),
        working=working,
        distractors=pick_distractors(target, [c for c in ALL_CATEGORIES if c != target]),
    )
