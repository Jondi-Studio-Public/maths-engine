"""Exam question on regrouping across zeros in place value: a reservoir drawn down twice, then refilled.

The full volume has two or three zeros among its thousands, hundreds and tens places, and the first amount
released needs a borrow that has to pass through a zero. Part (a) is the first subtraction, part (b) takes a
second release from what is left, and part (c) asks how much would refill the reservoir to full. Each later part
follows from the earlier answer.

The wrong answers come from the two classic errors in a borrow that crosses a zero: taking the smaller digit from
the larger in each column, and regrouping across the zero without taking one from the digit that lent it.
"""

from __future__ import annotations

import random

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, pick_distractors

_PLACES = (
    "A reservoir near Bathurst",
    "A water storage near Toowoomba",
    "A dam in the Goulburn Valley",
    "A reservoir above Hobart",
    "A storage dam near Mildura",
)
_FIRST = (
    "In summer, {r1} ML is released for irrigation.",
    "{r1} ML is released for farms during the dry season.",
)
_SECOND = (
    "Later, {r2} ML is released for town supply.",
    "A further {r2} ML is released for a nearby town.",
)


def _digits(n: int) -> list[int]:
    return [int(c) for c in reversed(str(n))]


def _from_digits(ds: list[int]) -> int:
    return sum(d * 10**i for i, d in enumerate(ds))


def zero_borrow(top: int, bottom: int):
    """The place value of the digit that lends when the borrow in `top - bottom` has to cross a zero, else None."""
    t, b = _digits(top), _digits(bottom)
    b += [0] * (len(t) - len(b))
    crossing = None
    for i in range(len(t)):
        if t[i] < b[i]:
            k = i + 1
            while k < len(t) and t[k] == 0:
                k += 1
            if k > i + 1 and crossing is None:
                crossing = k
            t[k] -= 1
            for m in range(i + 1, k):
                t[m] = 9
            t[i] += 10
    return crossing


def digitwise(top: int, bottom: int) -> int:
    """The classic error: the smaller digit is taken from the larger in every column."""
    t, b = _digits(top), _digits(bottom)
    b += [0] * (len(t) - len(b))
    return _from_digits([abs(x - y) for x, y in zip(t, b)])


def _money(n: int) -> str:
    return f"{n:,}"


@generator("exd_x_pr4_regroup_zero_reservoir", level=6, presets={"exam": dict(tier="exam")})
def exd_x_pr4_regroup_zero_reservoir(rng: random.Random, tier: str = "exam") -> dict:
    check(tier == "exam", "exd_x_pr4_regroup_zero_reservoir: only the exam tier exists")
    for _ in attempts(1000, "a full volume and two releases that need a borrow across zeros"):
        zeros = rng.sample([1, 2, 3], k=rng.choice((2, 3)))
        digs = [rng.randint(1, 9) if i not in zeros else 0 for i in range(4)] + [rng.randint(2, 9)]
        full = _from_digits(digs)
        r1 = rng.randint(full * 12 // 100, full * 55 // 100)
        if r1 % 10 == 0 or zero_borrow(full, r1) is None:
            continue
        left1 = full - r1
        r2 = rng.randint(800, left1 * 6 // 10)
        if r2 % 10 == 0:
            continue
        left2 = left1 - r2
        refill = full - left2

        j = zero_borrow(full, r1)
        forgot_reduce = left1 + 10**j
        a_candidates = [
            Distractor(str(digitwise(full, r1)), "smaller-digit-taken-from-larger", digitwise(full, r1)),
            Distractor(str(forgot_reduce), "borrow-not-reduced-after-zero", forgot_reduce),
        ]
        j2 = zero_borrow(left1, r2)
        b_candidates = [
            Distractor(str(digitwise(left1, r2)), "smaller-digit-taken-from-larger", digitwise(left1, r2)),
            Distractor(str(full - r2), "uses-original-amount-not-remaining", full - r2),
        ]
        if j2 is not None:
            b_candidates.insert(1, Distractor(str(left2 + 10**j2), "borrow-not-reduced-after-zero", left2 + 10**j2))
        c_candidates = [
            Distractor(str(digitwise(full, left2)), "smaller-digit-taken-from-larger", digitwise(full, left2)),
            Distractor(str(full - left1), "uses-earlier-value", full - left1),
            Distractor(str(full + left2), None, full + left2),
        ]
        a_d = pick_distractors(str(left1), a_candidates, 3)
        b_d = pick_distractors(str(left2), b_candidates, 3)
        c_d = pick_distractors(str(refill), c_candidates, 3)
        if min(len(a_d), len(b_d), len(c_d)) >= 2:
            break

    # Independent route: add the answers back.
    check(left1 + r1 == full and left2 + r2 == left1 and full - left2 == r1 + r2, "subtractions disagree")

    stem = (
        f"{rng.choice(_PLACES)} holds {_money(full)} ML when full. "
        f"{rng.choice(_FIRST).format(r1=_money(r1))} {rng.choice(_SECOND).format(r2=_money(r2))}"
    )
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="How many megalitres are left after the first release?",
                answer_kind="number",
                answer=left1,
                answer_text=str(left1),
                marks=2,
                sensible_range=(0, None),
                working=[f"{full} - {r1} = {left1}, regrouping across the zeros."],
                distractors=a_d,
            ),
            dict(
                prompt="How many megalitres are left after the second release?",
                answer_kind="number",
                answer=left2,
                answer_text=str(left2),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"part_a - {r2}",
                working=[f"{left1} - {r2} = {left2}."],
                distractors=b_d,
            ),
            dict(
                prompt="How many megalitres would refill the reservoir to full?",
                answer_kind="number",
                answer=refill,
                answer_text=str(refill),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"{full} - part_b",
                working=[f"{full} - {left2} = {refill}."],
                distractors=c_d,
            ),
        ],
    )
