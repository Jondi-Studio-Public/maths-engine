"""Exam-tier source for `classify-real-numbers`: how many of a list of roots are rational?

A list of roots ``sqrt(c k)`` (or cube roots ``cbrt(c k)``) for ``k = 1 .. N`` is described. A root of a whole number
is rational exactly when the number under it is a perfect square (or cube), so the learner has to see that ``c k``
is a perfect power only for a thin family of ``k`` and count them, rather than classify numbers one at a time:

* (a) how many of the numbers are rational (2 marks);
* (b) how many are irrational (1 mark, follow-through from (a));
* (c) the value of the largest rational number in the list (2 marks).

The counts are brute-forced with exact integer arithmetic, which is an independent route from the reasoning the
learner is meant to use (``c k`` a perfect power).
"""

from __future__ import annotations

import math
import random

from ..answers import Distractor
from ..core import GRADER_TOLERANCE, check, draw_until, generator, latex_answer, pick_distractors


def _is_power(n: int, m: int) -> int | None:
    """The integer ``r`` with ``r**m == n``, else None."""
    r = round(n ** (1 / m))
    for cand in (r - 1, r, r + 1):
        if cand >= 0 and cand**m == n:
            return cand
    return None


_SQUARE_MULTIPLIERS = [2, 3, 5, 6, 7, 8, 10, 12, 18, 20, 24, 27, 28, 32, 45, 48, 50]
_CUBE_MULTIPLIERS = [2, 3, 4, 9, 12, 16, 18, 20, 24, 32, 36, 48]

_STEMS = {
    2: [
        "The numbers √({c} × 1), √({c} × 2), √({c} × 3) and so on, up to √({c} × {n}), are written in a list.",
        "For each whole number k from 1 to {n}, the number √({c}k) is worked out and written in a list.",
        "A list is made of √({c}k) for every whole number k from 1 to {n}.",
    ],
    3: [
        "For each whole number k from 1 to {n}, the cube root of {c}k is worked out and written in a list.",
        "A list is made of the cube roots of {c}k for every whole number k from 1 to {n}.",
    ],
}


@generator("exd_x_sc5_rational_count", level=9, presets={"exam": dict(tier="exam")})
def exd_x_sc5_rational_count(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: count the rational numbers among roots of c k, then the irrational ones, then the largest value."""
    check(tier == "exam", f"unknown tier {tier!r}")
    power = rng.choice([2, 2, 2, 3])
    multipliers = _SQUARE_MULTIPLIERS if power == 2 else _CUBE_MULTIPLIERS

    def draw(r: random.Random):
        return r.choice(multipliers), r.randint(30, 200 if power == 2 else 300)

    def acceptable(v) -> bool:
        c, n = v
        hits = [k for k in range(1, n + 1) if _is_power(c * k, power) is not None]
        return len(hits) >= 3 and len(hits) <= n // 3

    c, n = draw_until(rng, draw, acceptable, what="a multiplier and length with a handful of rational roots")
    hits = [k for k in range(1, n + 1) if _is_power(c * k, power) is not None]
    rational = len(hits)
    irrational = n - rational
    top_k = hits[-1]
    top_value = _is_power(c * top_k, power)
    check(top_value is not None and top_value**power == c * top_k, "largest root is not exact")
    # Independent route: the rational ones are the k for which c*k is a perfect power, found by factorising.
    check(
        rational == sum(1 for k in range(1, n + 1) if round((c * k) ** (1 / power)) ** power == c * k),
        "count disagrees",
    )

    wrong_square_count = math.isqrt(n)
    a_cands = [
        Distractor(str(wrong_square_count), "ignored-multiplier-under-root", wrong_square_count),
        Distractor(str(math.isqrt(c * n)), "counted-every-square-up-to-the-largest", math.isqrt(c * n)),
        Distractor(str(irrational), "counted-the-wrong-class", irrational),
        Distractor(str(rational + 1), "off-by-one-endpoint", rational + 1),
        Distractor(str(n // c), "divided-instead-of-testing-perfect-power", n // c),
    ]
    b_cands = [
        Distractor(str(rational), "counted-the-wrong-class", rational),
        Distractor(str(n), "assumed-every-root-irrational", n),
        Distractor(str(irrational - 1), "off-by-one-endpoint", irrational - 1),
        Distractor(str(n - wrong_square_count), "ignored-multiplier-under-root", n - wrong_square_count),
    ]
    c_cands = [
        Distractor(str(c * top_k), "gave-radicand-not-root", c * top_k),
        Distractor(str(top_k), "gave-radicand-not-root", top_k),
        Distractor(
            str(round((n * c) ** (1 / power))), "took-root-of-the-original-number", round((n * c) ** (1 / power))
        ),
        Distractor(
            str(math.isqrt(top_k) if power == 2 else round(top_k ** (1 / 3))),
            "ignored-multiplier-under-root",
            math.isqrt(top_k) if power == 2 else round(top_k ** (1 / 3)),
        ),
    ]
    a_d = pick_distractors(str(rational), a_cands, 3, tolerance=GRADER_TOLERANCE)
    b_d = pick_distractors(str(irrational), b_cands, 3, tolerance=GRADER_TOLERANCE)
    c_d = pick_distractors(str(top_value), c_cands, 3, tolerance=GRADER_TOLERANCE)
    check(min(len(a_d), len(b_d), len(c_d)) >= 2, "too few distractors survive")

    what = "perfect square" if power == 2 else "perfect cube"
    return dict(
        text=rng.choice(_STEMS[power]).format(c=c, n=n),
        parts=[
            dict(
                prompt="How many of the numbers in the list are rational?",
                answer_kind="number",
                answer=rational,
                answer_text=str(rational),
                answer_latex=latex_answer(str(rational)),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"A root of a whole number is rational only when the number under it is a {what}.",
                    f"{c}k is a {what} for k = {', '.join(map(str, hits[:6]))}{', ...' if rational > 6 else ''}: {rational} values up to {n}.",
                ],
                distractors=a_d,
            ),
            dict(
                prompt="How many of the numbers in the list are irrational?",
                answer_kind="number",
                answer=irrational,
                answer_text=str(irrational),
                answer_latex=latex_answer(str(irrational)),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"{n} - part_a",
                working=[f"{n} - {rational} = {irrational}."],
                distractors=b_d,
            ),
            dict(
                prompt="Find the value of the largest rational number in the list.",
                answer_kind="number",
                answer=top_value,
                answer_text=str(top_value),
                answer_latex=latex_answer(str(top_value)),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"The largest k that works is {top_k}, and {c} x {top_k} = {c * top_k} = {top_value}^{power}.",
                    f"The value is {top_value}.",
                ],
                distractors=c_d,
            ),
        ],
    )
