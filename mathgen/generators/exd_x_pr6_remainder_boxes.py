"""Exam-tier source for `f10-division-remainder-context`: a three-part packing problem in an Australian setting.

A number of things must be packed into containers of a fixed size, each container costing a fixed amount.
(a) the learner decides that a remainder means one more container (round up), (b) finds how many places in the
last container stay empty, and (c) prices the containers. Parts (b) and (c) use the container count from (a), so
they carry `follow_through`.

Built backwards: the quotient, the container size and a non-zero remainder are drawn, then the total is
`quotient * size + remainder`. Every answer is computed from those integers; the test oracle re-derives them
from the printed numbers alone.
"""

from __future__ import annotations

from mathgen.core import VerificationError, attempts, generator

from ._pr6_chain_kit import check_chain, whole_part

# (stem, thing plural, container singular, container plural, size choices, quotient range, price range, price unit)
_SCENES = [
    dict(
        stem="A school canteen in {town} has {n} {things} to pack. Each {box} holds {d} {things} and costs ${p}.",
        things=["sausage rolls", "oranges", "muffins", "party pies"],
        box=["tray", "crate", "box"],
        sizes=[6, 8, 9, 12],
        quotients=(7, 30),
        prices=(3, 25),
        ask_a="How many {boxes} are needed to pack all of the {things}?",
        ask_b="How many more {things} would fill the last {box}?",
        ask_c="What is the total cost of the {boxes} needed, in dollars?",
    ),
    dict(
        stem="{n} students from a school in {town} are going on an excursion. Each bus seats {d} students and "
        "costs ${p} to hire.",
        things=["students"],
        box=["bus"],
        sizes=[20, 24, 30, 40, 44, 48],
        quotients=(3, 12),
        prices=(80, 400),
        ask_a="How many buses are needed so that every student has a seat?",
        ask_b="How many seats will be empty on the last bus?",
        ask_c="What is the total cost of hiring the buses needed, in dollars?",
    ),
    dict(
        stem="A bakery in {town} must deliver {n} {things} today. Each van carries {d} {things} and costs ${p} to "
        "send out.",
        things=["pies", "loaves", "pastries"],
        box=["van"],
        sizes=[15, 18, 20, 24, 30],
        quotients=(4, 15),
        prices=(20, 90),
        ask_a="How many van trips are needed to deliver all of the {things}?",
        ask_b="How many {things} short of full is the last van?",
        ask_c="What is the total cost of the van trips needed, in dollars?",
    ),
]
_TOWNS = ["Ballarat", "Bunbury", "Toowoomba", "Launceston", "Mildura", "Geraldton", "Wagga Wagga", "Cairns"]


@generator("exd_x_pr6_remainder_boxes", level=5, presets={"exam": dict(tier="exam")})
def exd_x_pr6_remainder_boxes(rng, tier: str = "exam"):
    scene = rng.choice(_SCENES)
    town = rng.choice(_TOWNS)
    things = rng.choice(scene["things"])
    box = rng.choice(scene["box"])
    boxes = box + ("es" if box.endswith(("x", "s", "ch", "sh")) else "s")
    for _ in attempts(1000, "a packing draw whose answers are not printed in the question"):
        d = rng.choice(scene["sizes"])
        q = rng.randint(*scene["quotients"])
        rem = rng.randint(1, d - 1)
        p = rng.randint(*scene["prices"])
        n = q * d + rem
        needed = q + 1
        empty = needed * d - n
        cost = needed * p
        text = scene["stem"].format(town=town, n=n, things=things, box=box, d=d, p=p)
        try:
            check_chain(text, [needed, empty, cost])
        except VerificationError:
            continue
        break
    fmt = dict(things=things, box=box, boxes=boxes)
    return dict(
        text=text,
        parts=[
            whole_part(
                scene["ask_a"].format(**fmt),
                needed,
                2,
                [(q, "rounds-groups-down"), (rem, "division-remainder-as-answer"), (d, "group-size-given-as-count")],
                working=[f"{n} ÷ {d} = {q} remainder {rem}.", f"The {rem} left over still need a {box}: {needed}."],
            ),
            whole_part(
                scene["ask_b"].format(**fmt),
                empty,
                2,
                [
                    (rem, "leftover-given-for-empty-spaces"),
                    (empty + 1, "story-off-by-one"),
                    (empty - 1, "story-off-by-one"),
                ],
                follow_through=f"part_a*{d} - {n}",
                working=[f"{needed} × {d} = {needed * d}.", f"{needed * d} − {n} = {empty}."],
            ),
            whole_part(
                scene["ask_c"].format(**fmt),
                cost,
                1,
                [
                    (q * p, "cost-from-rounded-down-count"),
                    (needed + p, "added-instead-of-multiplied"),
                    (d * p, "multiplied-by-wrong-quantity"),
                ],
                follow_through=f"part_a*{p}",
                working=[f"{needed} × ${p} = ${cost}."],
            ),
        ],
    )
