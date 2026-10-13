"""Exam-tier source for `f10-multidigit-add-subtract-regroup`: a three-part crowd-counting problem.

Three days of attendance are given. (a) adds three four-digit-or-less numbers (carrying), (b) compares two of the
days (a subtraction that needs a borrow), and (c) finds how far the weekend total falls short of a round target
(a subtraction from a number ending in zeros, so the borrow runs through them). Part (c) carries
`follow_through` on (a).

Built backwards from the three counts and the target; the target is a multiple of 100 above the total.
"""

from __future__ import annotations

from mathgen.core import VerificationError, attempts, generator

from ._pr6_chain_kit import check_chain, whole_part

_TOWNS = ["Hobart", "Mackay", "Broken Hill", "Fremantle", "Newcastle", "Townsville", "Ipswich", "Warrnambool"]
_SCENES = [
    dict(
        stem="A show in {town} had {x} visitors on Friday, {y} on Saturday and {z} on Sunday. The organisers hoped "
        "for {G} visitors over the weekend.",
        ask_a="How many visitors came over the whole weekend?",
        ask_b="How many more visitors came on Saturday than on Friday?",
        ask_c="How many more visitors were needed to reach the organisers' hope?",
    ),
    dict(
        stem="A zoo near {town} sold {x} tickets on Friday, {y} tickets on Saturday and {z} tickets on Sunday. It "
        "aimed to sell {G} tickets across the three days.",
        ask_a="How many tickets did the zoo sell across the three days?",
        ask_b="How many more tickets were sold on Saturday than on Friday?",
        ask_c="How many more tickets did the zoo need to sell to reach its aim?",
    ),
    dict(
        stem="Markets in {town} were visited by {x} people on Friday, {y} people on Saturday and {z} people on "
        "Sunday. The council wanted {G} people to visit over the three days.",
        ask_a="How many people visited the markets over the three days?",
        ask_b="How many more people visited on Saturday than on Friday?",
        ask_c="How many more people needed to visit to reach the council's target?",
    ),
]


def _digits(n: int, width: int) -> list[int]:
    return [int(c) for c in str(n).zfill(width)]


def _from_digits(ds: list[int]) -> int:
    return int("".join(str(d) for d in ds))


def carry_forgotten(*nums: int) -> int:
    """Column addition with every carry dropped: each column keeps only its units digit."""
    width = max(len(str(n)) for n in nums)
    cols = zip(*[_digits(n, width) for n in nums])
    return _from_digits([sum(c) % 10 for c in cols])


def smaller_from_larger(big: int, small: int) -> int:
    """Column subtraction that always takes the smaller digit from the larger, whichever is on top."""
    width = len(str(big))
    return _from_digits([abs(p - q) for p, q in zip(_digits(big, width), _digits(small, width))])


def borrow_not_paid(big: int, small: int) -> int:
    """Column subtraction that adds ten to a short column but forgets to take one from the next."""
    width = len(str(big))
    columns = zip(reversed(_digits(big, width)), reversed(_digits(small, width)))
    return sum((p + 10 - q if p < q else p - q) * 10**place for place, (p, q) in enumerate(columns))


@generator("exd_x_pr6_attendance_regroup", level=5, presets={"exam": dict(tier="exam")})
def exd_x_pr6_attendance_regroup(rng, tier: str = "exam"):
    scene = rng.choice(_SCENES)
    town = rng.choice(_TOWNS)
    for _ in attempts(1000, "an attendance draw that needs carrying and borrowing"):
        x, y, z = rng.randint(700, 2900), rng.randint(700, 2900), rng.randint(700, 2900)
        total = x + y + z
        target = (total // 100 + rng.randint(2, 15)) * 100
        if y <= x:
            continue
        needs_borrow = y % 10 < x % 10 or (y // 10) % 10 < (x // 10) % 10
        if not needs_borrow or carry_forgotten(x, y, z) == total or total % 100 == 0:
            continue
        gap, diff = target - total, y - x
        if len({x, y, z, total, target, gap, diff}) < 7:
            continue
        text = scene["stem"].format(town=town, x=x, y=y, z=z, G=target)
        try:
            check_chain(text, [total, diff, gap])
        except VerificationError:
            continue
        break
    return dict(
        text=text,
        parts=[
            whole_part(
                scene["ask_a"],
                total,
                2,
                [
                    (carry_forgotten(x, y, z), "carry-dropped"),
                    (y + z, "story-one-quantity-ignored"),
                    (x + y, "story-one-quantity-ignored"),
                ],
                working=[f"{x} + {y} = {x + y}.", f"{x + y} + {z} = {total}."],
            ),
            whole_part(
                scene["ask_b"],
                diff,
                1,
                [
                    (smaller_from_larger(y, x), "smaller-digit-taken-from-larger"),
                    (y + x, "story-operation-reversed"),
                    (borrow_not_paid(y, x), "borrow-not-taken-from-next-column"),
                ],
                working=[f"{y} − {x} = {diff}."],
            ),
            whole_part(
                scene["ask_c"],
                gap,
                2,
                [
                    (smaller_from_larger(target, total), "smaller-digit-taken-from-larger"),
                    (borrow_not_paid(target, total), "borrow-not-taken-from-next-column"),
                    (target + total, "story-operation-reversed"),
                ],
                follow_through=f"{target} - part_a",
                working=[f"{target} − {total} = {gap}."],
            ),
        ],
    )
