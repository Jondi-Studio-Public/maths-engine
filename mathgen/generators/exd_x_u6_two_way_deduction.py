"""Exam-tier source for `f10-two-way-table-relationship`: build a two-way table from totals, then use it.

The learner is not given a table or a list of records. A short report states the size of the whole group, the size
of two overlapping groups and how many are in both, and must

1. fill the first blank cell (in one group but not the other),
2. fill the second blank cell,
3. fill the cell for neither group, using the cells already found, and
4. read a conditional fraction off the completed table.

Built backwards from the four cells, so every count is a whole number of at least three.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..answers import exact_value
from ..core import attempts, check, generator, latex_answer, pick_distractors
from .exd_x_u6_support import keys_clear, tagged, within_unit_interval

# (whole group, member noun, in group A, not in group A, has the category, lacks the category)
_CONTEXTS = [
    (
        "students at a Hobart high school",
        "students",
        "are in Year 9",
        "are not in Year 9",
        "travel to school by bus",
        "do not travel to school by bus",
    ),
    (
        "members of a Geelong sports club",
        "members",
        "are juniors",
        "are seniors",
        "have paid their fees",
        "have not paid their fees",
    ),
    (
        "people surveyed at a Perth shopping centre",
        "people",
        "are under 30",
        "are 30 or older",
        "own a smartwatch",
        "do not own a smartwatch",
    ),
    (
        "cars inspected at a Brisbane depot",
        "cars",
        "are petrol cars",
        "are not petrol cars",
        "passed the safety check",
        "failed the safety check",
    ),
]
_TEMPLATES = [
    "A survey of {N} {whole} found that {rowA} of them {inA}. Also, {colX} of them {hasX}, and {J} of those who {inA} "
    "also {hasX}.",
    "Records for {N} {whole} show that {colX} {hasX}. Of the group, {rowA} {inA}, and {J} {inA} and also {hasX}.",
    "There are {N} {whole}. In all, {rowA} {inA} and {colX} {hasX}. Exactly {J} {inA} and {hasX}.",
]


def _frac_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _build(rng: random.Random):
    J = rng.randint(4, 40)  # in A and has X
    a = rng.randint(3, 40)  # in A, lacks X
    b = rng.randint(3, 40)  # not in A, has X
    c = rng.randint(3, 40)  # not in A, lacks X
    N, rowA, colX = J + a + b + c, J + a, J + b
    whole, noun, in_a, not_a, has_x, lacks_x = rng.choice(_CONTEXTS)
    cond = Fraction(b, N - rowA)

    stem = rng.choice(_TEMPLATES).format(N=N, whole=whole, rowA=rowA, colX=colX, J=J, inA=in_a, hasX=has_x)
    p1 = f"Find how many of the {noun} {in_a} but {lacks_x}."
    p2 = f"Find how many of the {noun} {not_a} but {has_x}."
    p3 = f"Using your answers to (a) and (b), find how many of the {noun} {not_a} and {lacks_x}."
    p4 = (
        f"Using your answer to (b), write the fraction of the {noun} who {not_a} that {has_x}, "
        "as a simplified fraction."
    )
    if not keys_clear(" ".join([stem, p1, p2, p3, p4]), [a, b, c, cond]):
        return None

    a_d = pick_distractors(
        str(a),
        [
            tagged(str(rowA), "used-total-as-cell", rowA),
            tagged(str(colX - J), "subtracted-from-wrong-total", colX - J),
            tagged(str(N - rowA), "used-total-as-cell", N - rowA),
        ],
        3,
    )
    b_d = pick_distractors(
        str(b),
        [
            tagged(str(colX), "used-total-as-cell", colX),
            tagged(str(rowA - J), "subtracted-from-wrong-total", rowA - J),
            tagged(str(N - colX), "used-total-as-cell", N - colX),
        ],
        3,
    )
    c_d = pick_distractors(
        str(c),
        [
            tagged(str(N - rowA - colX), "overlap-counted-twice", N - rowA - colX),
            tagged(str(N - rowA), "used-total-as-cell", N - rowA),
            tagged(str(N - colX), "used-total-as-cell", N - colX),
        ],
        3,
    )
    d_cands = within_unit_interval(
        [
            tagged(_frac_text(Fraction(b, N)), "joint-for-conditional", Fraction(b, N)),
            tagged(_frac_text(Fraction(b, colX)), "wrong-condition-group", Fraction(b, colX)),
            tagged(_frac_text(1 - cond), "gave-complement-probability", 1 - cond),
        ]
    )
    d_d = pick_distractors(_frac_text(cond), d_cands, 3)
    if min(len(a_d), len(b_d), len(c_d), len(d_d)) < 2:
        return None

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=p1,
                answer_kind="number",
                answer=a,
                answer_text=str(a),
                answer_latex=latex_answer(str(a)),
                marks=1,
                sensible_range=(0, N),
                working=[
                    f"{rowA} are in the first group and {J} of them also have the category.",
                    f"{rowA} - {J} = {a}.",
                ],
                distractors=a_d,
            ),
            dict(
                prompt=p2,
                answer_kind="number",
                answer=b,
                answer_text=str(b),
                answer_latex=latex_answer(str(b)),
                marks=1,
                sensible_range=(0, N),
                working=[f"{colX} - {J} = {b}."],
                distractors=b_d,
            ),
            dict(
                prompt=p3,
                answer_kind="number",
                answer=c,
                answer_text=str(c),
                answer_latex=latex_answer(str(c)),
                marks=2,
                sensible_range=(0, N),
                follow_through=f"{N} - {J} - part_a - part_b",
                working=[f"{N} - {J} - {a} - {b} = {c}."],
                distractors=c_d,
            ),
            dict(
                prompt=p4,
                answer_kind="fraction",
                answer=exact_value(cond),
                answer_text=_frac_text(cond),
                answer_latex=latex_answer(_frac_text(cond)),
                marks=2,
                sensible_range=(0, 1),
                follow_through=f"part_b/{N - rowA}",
                working=[
                    f"{N} - {rowA} = {N - rowA} are not in the first group.",
                    f"{b}/{N - rowA} = {_frac_text(cond)}.",
                ],
                distractors=d_d,
            ),
        ],
    )


@generator("exd_x_u6_two_way_deduction", level=10, presets={"exam": dict(tier="exam")})
def exd_x_u6_two_way_deduction(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: deduce the blank cells of a two-way table from totals and use them for a conditional fraction."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a table whose cells and distractors are all acceptable"):
        question = _build(rng)
        if question is not None:
            return question
