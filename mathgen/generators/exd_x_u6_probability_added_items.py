"""Exam-tier source for `simple-event-probability`: change a collection until a target probability holds, then reuse it.

A bag holds items of three kinds. The learner

1. gives the probability of one kind as a simplified fraction,
2. works out how many more of that kind must be added so that its probability reaches a stated target (the total
   grows as well, so this is an equation to set up, not a subtraction), and
3. gives the probability of a second kind in the enlarged bag.

Built backwards: draw the counts and the target, and keep the draw only when the number to add is a whole number.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..answers import exact_value
from ..core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    rational_text,
    rounded_decimal_string,
)
from .exd_x_u6_support import keys_clear, tagged, within_unit_interval

_TARGETS = [(1, 2), (2, 3), (3, 4), (3, 5), (4, 5), (2, 5), (5, 8), (7, 10)]
_SETTINGS = [
    ("A bag holds {r} red, {g} green and {b} blue counters.", "counter", "counters", "bag"),
    ("A box contains {r} red, {g} green and {b} blue tiles.", "tile", "tiles", "box"),
    ("A jar holds {r} red, {g} green and {b} blue lollies.", "lolly", "lollies", "jar"),
    ("A pouch contains {r} red, {g} green and {b} blue beads.", "bead", "beads", "pouch"),
]


def _decimal(value: Fraction) -> str:
    return rounded_decimal_string(value, 2)


def _build(rng: random.Random):
    r, g, b = (rng.randint(1, 9) for _ in range(3))
    n = r + g + b
    p, q = rng.choice(_TARGETS)
    x_exact = Fraction(p * n - q * r, q - p)
    if x_exact.denominator != 1 or not 1 <= x_exact <= 30:
        return None
    x = int(x_exact)
    check(Fraction(r + x, n + x) == Fraction(p, q))
    p_red, p_blue_new = Fraction(r, n), Fraction(b, n + x)
    if p_blue_new == p_red or p_blue_new == Fraction(p, q):
        return None

    setting, one, many, holder = rng.choice(_SETTINGS)
    stem = setting.format(r=r, g=g, b=b)
    p1 = f"One {one} is taken at random. Write the probability that it is red as a simplified fraction."
    p2 = f"More red {many} are now added to the {holder} until the probability of taking a red {one} is {p}/{q}. How many red {many} are added?"
    p3 = f"Using your answer to (b), find the probability of taking a blue {one} from the {holder} now, as a simplified fraction."
    if not keys_clear(" ".join([stem, p1, p2, p3]), [x, p_red, p_blue_new]):
        return None

    a_d = pick_distractors(
        rational_text(p_red),
        within_unit_interval(
            [
                tagged(rational_text(Fraction(r, n - r)), "favourable-over-unfavourable", Fraction(r, n - r)),
                tagged(rational_text(1 - p_red), "gave-complement-probability", 1 - p_red),
                tagged(rational_text(Fraction(r, r + g)), "left-out-a-category-from-total", Fraction(r, r + g)),
            ]
        ),
        3,
    )
    forgot_total = Fraction(p * n, q) - r  # equates r + x with the target share of the OLD total
    wrong_b = [
        (forgot_total, "forgot-to-update-total"),
        (Fraction(q * r, p) - n, "forgot-to-update-total"),
        (Fraction(p * (g + b), q) - r, "favourable-over-unfavourable"),
        (Fraction(r + x), "gave-new-total-not-increase"),
    ]
    b_cands = [tagged(_decimal(v), mid, v) for v, mid in wrong_b if v > 0 and v != x]
    b_d = pick_distractors(str(x), b_cands, 3, tolerance=GRADER_TOLERANCE)
    c_d = pick_distractors(
        rational_text(p_blue_new),
        within_unit_interval(
            [
                tagged(rational_text(Fraction(b, n)), "forgot-to-update-total", Fraction(b, n)),
                tagged(rational_text(1 - p_blue_new), "gave-complement-probability", 1 - p_blue_new),
                tagged(rational_text(Fraction(b, n + x - b)), "favourable-over-unfavourable", Fraction(b, n + x - b)),
            ]
        ),
        3,
    )
    if min(len(a_d), len(b_d), len(c_d)) < 2:
        return None

    def fraction_part(prompt, value, marks, distractors, working, follow_through=None):
        entry = dict(
            prompt=prompt,
            answer_kind="fraction",
            answer=exact_value(value),
            answer_text=rational_text(value),
            answer_latex=latex_answer(rational_text(value)),
            marks=marks,
            sensible_range=(0, 1),
            working=working,
            distractors=distractors,
        )
        if follow_through:
            entry["follow_through"] = follow_through
        return entry

    return dict(
        text=stem,
        parts=[
            fraction_part(
                p1, p_red, 1, a_d, [f"Total = {r} + {g} + {b} = {n}.", f"P(red) = {r}/{n} = {rational_text(p_red)}."]
            ),
            dict(
                prompt=p2,
                answer_kind="number",
                answer=numeric_answer(Fraction(x)),
                answer_text=str(x),
                answer_latex=latex_answer(str(x)),
                marks=3,
                sensible_range=(0, None),
                working=[
                    f"Let x be the number added: ({r} + x)/({n} + x) = {p}/{q}.",
                    f"{q}({r} + x) = {p}({n} + x), so {q - p}x = {p * n - q * r}, x = {x}.",
                ],
                distractors=b_d,
            ),
            fraction_part(
                p3,
                p_blue_new,
                2,
                c_d,
                [f"The total is now {n} + {x} = {n + x}.", f"P(blue) = {b}/{n + x} = {rational_text(p_blue_new)}."],
                follow_through=f"{b}/({n} + part_b)",
            ),
        ],
    )


@generator("exd_x_u6_probability_added_items", level=10, presets={"exam": dict(tier="exam")})
def exd_x_u6_probability_added_items(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: simple probability, a target-probability equation, then a probability in the changed collection."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(2000, "counts whose target probability needs a whole number of additions"):
        question = _build(rng)
        if question is not None:
            return question
