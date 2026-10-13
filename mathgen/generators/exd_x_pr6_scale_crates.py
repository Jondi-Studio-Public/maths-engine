"""Exam-tier source for `f10-mental-mult-div-extension`: a three-part production problem using tens.

A farm or factory packs a number of tens of items into each tray and fills a number of tens of trays; a
different multiple of ten fills each crate. (a) the learner scales a basic fact up twice (tens times tens),
(b) divides the product by a multiple of ten (a related division fact, scaled back), and (c) prices the crates.
Parts (b) and (c) carry `follow_through`.

Built backwards: the single-digit facts a, b and the crate factor s are drawn so that 10ab is divisible by s,
which makes the number of crates a whole number.
"""

from __future__ import annotations

from mathgen.core import VerificationError, attempts, generator

from ._pr6_chain_kit import check_chain, whole_part

_SCENES = [
    dict(
        stem="A farm in the Riverina packs {a0} {things} into each tray and fills {b0} trays every day. The "
        "{things} are then loaded into crates that each hold {s0}. Each crate is sold for ${q}.",
        things=["apples", "oranges", "mandarins", "avocados", "plums"],
        ask_a="How many {things} are packed in one day?",
        ask_b="How many crates can be filled with one day's {things}?",
        ask_c="How much money does the farm make from selling those crates, in dollars?",
    ),
    dict(
        stem="A factory in Geelong puts {a0} {things} in each carton and packs {b0} cartons on every shift. The "
        "{things} are then moved to pallets that each hold {s0}. A pallet is sold for ${q}.",
        things=["drink bottles", "tennis balls", "bike lights", "water pistols"],
        ask_a="How many {things} are packed on one shift?",
        ask_b="How many pallets can be filled from one shift's {things}?",
        ask_c="How much money is made from selling those pallets, in dollars?",
    ),
    dict(
        stem="A nursery in Gympie pots {a0} {things} into each tray and fills {b0} trays each week. The {things} "
        "are then boxed up with {s0} in each box. Each box is sold for ${q}.",
        things=["seedlings", "herb plants", "succulents"],
        ask_a="How many {things} are potted in one week?",
        ask_b="How many boxes can be filled with one week's {things}?",
        ask_c="How much money does the nursery make from selling those boxes, in dollars?",
    ),
]


@generator("exd_x_pr6_scale_crates", level=5, presets={"exam": dict(tier="exam")})
def exd_x_pr6_scale_crates(rng, tier: str = "exam"):
    scene = rng.choice(_SCENES)
    things = rng.choice(scene["things"])
    for _ in attempts(1000, "a scaling draw with a whole number of crates"):
        a, b, s = rng.randint(3, 9), rng.randint(3, 9), rng.randint(2, 9)
        if a == b or (10 * a * b) % s != 0:
            continue
        q = rng.randint(6, 45)
        a0, b0, s0 = 10 * a, 10 * b, 10 * s
        total = a0 * b0
        crates = total // s0
        income = crates * q
        if crates < 5 or s0 in (a0, b0):
            continue
        text = scene["stem"].format(a0=a0, b0=b0, s0=s0, q=q, things=things)
        try:
            check_chain(text, [total, crates, income])
        except VerificationError:
            continue
        break
    fmt = dict(things=things)
    candidates_b = [(crates * 10, "wrong-number-of-zeros"), (crates * 100, "wrong-number-of-zeros")]
    if crates % 10 == 0:
        candidates_b.append((crates // 10, "wrong-number-of-zeros"))
    return dict(
        text=text,
        parts=[
            whole_part(
                scene["ask_a"].format(**fmt),
                total,
                2,
                [
                    (a * b, "tens-not-scaled"),
                    (a * b * 10, "one-factor-scaled"),
                    (a * b * 1000, "wrong-number-of-zeros"),
                ],
                working=[f"{a} × {b} = {a * b}.", f"{a0} × {b0} is 100 times as much: {total}."],
            ),
            whole_part(
                scene["ask_b"].format(**fmt),
                crates,
                2,
                candidates_b,
                follow_through=f"part_a/{s0}",
                working=[f"{total} ÷ {s0}: {total // 10} ÷ {s} = {crates}."],
            ),
            whole_part(
                scene["ask_c"],
                income,
                1,
                [
                    (crates + q, "added-instead-of-multiplied"),
                    (crates * q * 10, "wrong-number-of-zeros"),
                    (total * q, "multiplied-by-wrong-quantity"),
                ],
                follow_through=f"part_b*{q}",
                working=[f"{crates} × ${q} = ${income}."],
            ),
        ],
    )
