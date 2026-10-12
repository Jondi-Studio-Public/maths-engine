"""Exam-tier source for `f10-multiply-divide-one-digit-arrays`: a three-part array problem.

Objects are laid out in an array of rows and columns. (a) the learner finds the number in the array, (b) takes
some away, and (c) shares what is left into equal groups. Each part uses the one before, so (b) and (c) carry
`follow_through`; the learner must choose multiplication, subtraction and division in turn.

Built backwards: the group size and the number of groups are drawn, so what remains is exactly divisible, and
the number removed is whatever brings the array down to it.
"""

from __future__ import annotations

from mathgen.core import VerificationError, attempts, generator

from ._pr6_chain_kit import check_chain, whole_part

_TOWNS = ["Kalgoorlie", "Horsham", "Maryborough", "Coffs Harbour", "Whyalla", "Katherine", "Bathurst", "Hervey Bay"]
_SCENES = [
    dict(
        stem="A hall in {town} has {r} rows with {c} chairs in each row. {x} chairs are taken away to make room for "
        "a stage. The chairs that are left are stacked in piles of {s}.",
        ask_a="How many chairs were in the hall to begin with?",
        ask_b="How many chairs are left after the stage is set up?",
        ask_c="How many piles of chairs are made?",
    ),
    dict(
        stem="A car park in {town} has {r} rows of {c} parking bays. {x} bays are closed for repairs. The bays that "
        "are still open are painted in zones of {s} bays.",
        ask_a="How many parking bays does the car park have in total?",
        ask_b="How many bays are still open?",
        ask_c="How many zones are painted?",
    ),
    dict(
        stem="A baker in {town} lays out {r} trays with {c} biscuits on each tray. {x} biscuits break and are thrown "
        "out. The rest are packed into bags of {s}.",
        ask_a="How many biscuits were on the trays?",
        ask_b="How many biscuits are left after the broken ones are thrown out?",
        ask_c="How many bags are packed?",
    ),
]


@generator("exd_x_pr6_array_stacks", level=4, presets={"exam": dict(tier="exam")})
def exd_x_pr6_array_stacks(rng, tier: str = "exam"):
    scene = rng.choice(_SCENES)
    town = rng.choice(_TOWNS)
    for _ in attempts(1000, "an array draw with an exact final share"):
        r, c = rng.randint(3, 9), rng.randint(4, 9)
        if r == c:
            continue
        s = rng.randint(2, 9)
        total = r * c
        piles = rng.randint(3, total // s - 1) if total // s > 3 else 0
        if piles < 3:
            continue
        left = piles * s
        removed = total - left
        if removed < 2 or removed > total // 2 + 4 or len({r, c, s, removed, total, left, piles}) < 7:
            continue
        text = scene["stem"].format(town=town, r=r, c=c, x=removed, s=s)
        try:
            check_chain(text, [total, left, piles])
        except VerificationError:
            continue
        break
    return dict(
        text=text,
        parts=[
            whole_part(
                scene["ask_a"],
                total,
                1,
                [
                    (r + c, "added-instead-of-multiplied"),
                    (r * (c + 1), "story-off-by-one"),
                    (r * (c - 1), "story-off-by-one"),
                ],
                working=[f"{r} × {c} = {total}."],
            ),
            whole_part(
                scene["ask_b"],
                left,
                1,
                [(total + removed, "story-operation-reversed"), (removed, "given-quantity-reported")],
                follow_through=f"part_a - {removed}",
                working=[f"{total} − {removed} = {left}."],
            ),
            whole_part(
                scene["ask_c"],
                piles,
                2,
                [
                    (left * s, "multiplied-instead-of-divided"),
                    (left - s, "groups-total-minus-count"),
                    *([(total // s, "story-one-quantity-ignored")] if total % s == 0 else []),
                ],
                follow_through=f"part_b/{s}",
                working=[f"{left} ÷ {s} = {piles}, because {piles} × {s} = {left}."],
            ),
        ],
    )
