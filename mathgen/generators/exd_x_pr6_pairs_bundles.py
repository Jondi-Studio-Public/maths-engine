"""Exam-tier source for `f10-times-two-facts-inverse`: a three-part halving problem with a postage cost.

Single items come in a box and are sold in pairs; pairs are tied into bundles of two; each bundle costs $2 to
post, plus a flat fee. (a) halve the items to get pairs, (b) halve again to get bundles, (c) double the bundles
and add the fee. Each part uses the one before, so (b) and (c) carry `follow_through`.

Built backwards from the number of bundles, so both halvings are exact.
"""

from __future__ import annotations

from mathgen.core import VerificationError, attempts, generator

from ._pr6_chain_kit import check_chain, whole_part

_TOWNS = ["Perth", "Cairns", "Dubbo", "Mount Gambier", "Darwin", "Launceston", "Mandurah", "Albury"]
_SCENES = [
    dict(
        stem="A sports shop in {town} has a box of {n} {things}. The {things} are sold in pairs, and 2 pairs are "
        "tied together into a bundle. Posting each bundle costs $2, and there is a flat label fee of ${f}.",
        things=["socks", "gloves", "shoes", "shin pads"],
        ask_a="How many pairs are in the box?",
        ask_b="How many bundles can be made from the box?",
        ask_c="What is the total cost of posting all the bundles, including the label fee, in dollars?",
    ),
    dict(
        stem="A school in {town} has {n} {things} in a storeroom. They are given out in pairs, and 2 pairs are put "
        "in each carton. Sending each carton costs $2, plus a handling fee of ${f}.",
        things=["thongs", "boots", "gloves", "swim goggles"],
        ask_a="How many pairs are in the storeroom?",
        ask_b="How many cartons are needed for all the pairs?",
        ask_c="What is the total cost of sending the cartons, including the handling fee, in dollars?",
    ),
    dict(
        stem="A charity in {town} collects {n} {things}. They are matched into pairs, and 2 pairs are packed in "
        "each parcel. Posting each parcel costs $2, and the charity pays a fee of ${f} to register the shipment.",
        things=["socks", "gloves", "shoes", "mittens"],
        ask_a="How many pairs does the charity collect?",
        ask_b="How many parcels are packed?",
        ask_c="What is the total cost of posting the parcels, including the fee, in dollars?",
    ),
]


@generator("exd_x_pr6_pairs_bundles", level=3, presets={"exam": dict(tier="exam")})
def exd_x_pr6_pairs_bundles(rng, tier: str = "exam"):
    scene = rng.choice(_SCENES)
    town, things = rng.choice(_TOWNS), rng.choice(scene["things"])
    for _ in attempts(1000, "a halving draw whose answers are not printed"):
        bundles = rng.randint(3, 12)
        fee = rng.randint(3, 40)
        n = 4 * bundles
        pairs = 2 * bundles
        cost = 2 * bundles + fee
        text = scene["stem"].format(town=town, n=n, things=things, f=fee)
        try:
            check_chain(text, [pairs, bundles, cost])
        except VerificationError:
            continue
        break
    return dict(
        text=text,
        parts=[
            whole_part(
                scene["ask_a"],
                pairs,
                1,
                [(n * 2, "doubled-instead-of-halved"), (n, "halving-skipped"), (n - 2, "groups-total-minus-count")],
                working=[f"2 × {pairs} = {n}, so {n} ÷ 2 = {pairs}."],
            ),
            whole_part(
                scene["ask_b"],
                bundles,
                1,
                [
                    (pairs * 2, "doubled-instead-of-halved"),
                    (pairs, "halving-skipped"),
                    (pairs - 2, "groups-total-minus-count"),
                ],
                follow_through="part_a/2",
                working=[f"2 × {bundles} = {pairs}, so {pairs} ÷ 2 = {bundles}."],
            ),
            whole_part(
                scene["ask_c"],
                cost,
                2,
                [
                    (bundles + fee, "added-instead-of-multiplied"),
                    (2 * bundles, "story-one-quantity-ignored"),
                    (pairs * 2 + fee, "multiplied-by-wrong-quantity"),
                ],
                follow_through=f"2*part_b + {fee}",
                working=[f"{bundles} × $2 = ${2 * bundles}.", f"${2 * bundles} + ${fee} = ${cost}."],
            ),
        ],
    )
