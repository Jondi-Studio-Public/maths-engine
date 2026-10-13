"""Exam-tier source for `f10-square-roots-perfect-squares`: make a number into a perfect square.

Instead of deciding whether a given number is a perfect square, the learner has to find what turns a number that
is not one into a perfect square, three ways chosen per seed:

* ``multiply`` -- the smallest whole number to multiply by (the square-free part of the number);
* ``add`` -- the smallest whole number to add (a formation of students that must be made square);
* ``remove`` -- the number left over when the largest square arrangement of tiles is laid.

Then (b) the perfect square that results, and (c) its square root. (b) and (c) carry follow-through.
Each case is built backwards from the root and checked against an independent search.
"""

from __future__ import annotations

import math
import random

from ..answers import Distractor
from ..core import GRADER_TOLERANCE, check, draw_until, generator, latex_answer, pick_distractors

_SQUARE_FREE = [2, 3, 5, 6, 7, 10, 11, 13, 14, 15]


def _smallest_multiplier(n: int) -> int:
    for k in range(1, n + 1):
        if math.isqrt(n * k) ** 2 == n * k:
            return k
    raise AssertionError("unreachable")


def _build_multiply(rng: random.Random):
    k0 = rng.choice(_SQUARE_FREE)
    s = rng.randint(3, 14)
    n = k0 * s * s
    check(_smallest_multiplier(n) == k0, "multiplier is not the smallest")
    product = n * k0
    root = k0 * s
    stem = f"The number {n} is not a perfect square."
    a_prompt = f"Find the smallest whole number that {n} can be multiplied by to give a perfect square."
    b_prompt = "What is the perfect square that results?"
    c_prompt = "Find the square root of that perfect square."
    a_cands = [
        Distractor(str(s), "gave-root-not-asked-value", s),
        Distractor(str(k0 * s), "gave-root-not-asked-value", k0 * s),
        Distractor(str(k0 * k0), "multiplied-by-wrong-factor", k0 * k0),
        Distractor(str(n), "multiplied-by-wrong-factor", n),
    ]
    return (
        n,
        k0,
        product,
        root,
        stem,
        a_prompt,
        b_prompt,
        c_prompt,
        a_cands,
        f"{n} = {k0} x {s}^2, so multiplying by {k0} gives {product}.",
        f"{n} x k",
    )


def _build_add(rng: random.Random):
    m = rng.randint(12, 45)
    t = rng.randint(2, 2 * m - 3)
    n = m * m - t
    check((m - 1) ** 2 < n < m * m, "not between consecutive squares")
    stem = f"{n} students are to stand in a square formation, with the same number of rows as columns, for a school photo. Some more students must join to make it possible."
    a_prompt = "What is the smallest number of extra students needed?"
    b_prompt = "How many students will be in the square formation?"
    c_prompt = "How many students are in each row?"
    a_cands = [
        Distractor(str(n - (m - 1) ** 2), "used-wrong-neighbouring-square", n - (m - 1) ** 2),
        Distractor(str((m + 1) ** 2 - n), "used-wrong-neighbouring-square", (m + 1) ** 2 - n),
        Distractor(str(m), "gave-root-not-asked-value", m),
        Distractor(str(t + 1), "off-by-one-endpoint", t + 1),
    ]
    return (
        n,
        t,
        m * m,
        m,
        stem,
        a_prompt,
        b_prompt,
        c_prompt,
        a_cands,
        f"{m - 1}^2 = {(m - 1) ** 2} < {n} < {m}^2 = {m * m}, so {m * m} - {n} = {t}.",
        f"{n} + x",
    )


def _build_remove(rng: random.Random):
    m = rng.randint(12, 45)
    t = rng.randint(2, 2 * m - 1)
    n = m * m + t
    check(m * m < n < (m + 1) ** 2, "not between consecutive squares")
    stem = f"A tiler has {n} square tiles and lays them in the largest possible square arrangement, with the same number of tiles along each side. The tiles that do not fit are put aside."
    a_prompt = "How many tiles are put aside?"
    b_prompt = "How many tiles are laid in the square?"
    c_prompt = "How many tiles are along one side of the square?"
    a_cands = [
        Distractor(str((m + 1) ** 2 - n), "used-wrong-neighbouring-square", (m + 1) ** 2 - n),
        Distractor(str(n - (m - 1) ** 2), "used-wrong-neighbouring-square", n - (m - 1) ** 2),
        Distractor(str(m), "gave-root-not-asked-value", m),
        Distractor(str(t - 1), "off-by-one-endpoint", t - 1),
    ]
    return (
        n,
        t,
        m * m,
        m,
        stem,
        a_prompt,
        b_prompt,
        c_prompt,
        a_cands,
        f"{m}^2 = {m * m} <= {n} < {(m + 1) ** 2}, so {n} - {m * m} = {t}.",
        f"{n} - x",
    )


_BUILDERS = [_build_multiply, _build_add, _build_remove]


@generator("exd_x_sc5_square_fix", level=9, presets={"exam": dict(tier="exam")})
def exd_x_sc5_square_fix(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: the adjustment that makes a number a perfect square, the square, and its root."""
    check(tier == "exam", f"unknown tier {tier!r}")
    case = rng.randrange(3)
    n, adjustment, square, root, stem, a_prompt, b_prompt, c_prompt, a_cands, a_working, _ = _BUILDERS[case](rng)
    check(math.isqrt(square) ** 2 == square and math.isqrt(square) == root, "result is not a perfect square")
    if case == 0:
        check(n * adjustment == square, "product is not the square")
        b_ft = f"{n}*part_a"
    elif case == 1:
        check(n + adjustment == square, "sum is not the square")
        b_ft = f"{n} + part_a"
    else:
        check(n - adjustment == square, "difference is not the square")
        b_ft = f"{n} - part_a"
    b_cands = [
        Distractor(str(n), "gave-the-original-number", n),
        Distractor(str(root), "gave-root-not-asked-value", root),
        Distractor(str(square + (2 * root + 1)), "used-wrong-neighbouring-square", square + 2 * root + 1),
        Distractor(str(square - (2 * root - 1)), "used-wrong-neighbouring-square", square - (2 * root - 1)),
    ]
    c_cands = [
        Distractor(str(root + 1), "off-by-one-endpoint", root + 1),
        Distractor(str(root - 1), "off-by-one-endpoint", root - 1),
        Distractor(str(math.isqrt(n)), "took-root-of-the-original-number", math.isqrt(n)),
    ]
    a_d = pick_distractors(str(adjustment), a_cands, 3, tolerance=GRADER_TOLERANCE)
    b_d = pick_distractors(str(square), b_cands, 3, tolerance=GRADER_TOLERANCE)
    c_d = pick_distractors(str(root), c_cands, 3, tolerance=GRADER_TOLERANCE)
    check(min(len(a_d), len(b_d), len(c_d)) >= 2, "too few distractors survive")
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=a_prompt,
                answer_kind="number",
                answer=adjustment,
                answer_text=str(adjustment),
                answer_latex=latex_answer(str(adjustment)),
                marks=2,
                sensible_range=(0, None),
                working=[a_working],
                distractors=a_d,
            ),
            dict(
                prompt=b_prompt,
                answer_kind="number",
                answer=square,
                answer_text=str(square),
                answer_latex=latex_answer(str(square)),
                marks=1,
                sensible_range=(0, None),
                follow_through=b_ft,
                working=[f"The result is {square}."],
                distractors=b_d,
            ),
            dict(
                prompt=c_prompt,
                answer_kind="number",
                answer=root,
                answer_text=str(root),
                answer_latex=latex_answer(str(root)),
                marks=1,
                sensible_range=(0, None),
                follow_through="sqrt(part_b)",
                working=[f"{root} x {root} = {square}, so the square root is {root}."],
                distractors=c_d,
            ),
        ],
    )
