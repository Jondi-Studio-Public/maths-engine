"""Exam-tier source for `f10-multiplication-facts-3-4-5-10`: a three-part sharing problem built on one fact family.

Items are bought in bags whose size is in the 3, 4, 5 or 10 times tables. (a) the learner multiplies to find the
total; (b) prices the bags (a second multiplication fact); (c) and (d) are two successive equal shares, each the
division fact that undoes a multiplication. Parts (c) and (d) carry `follow_through`, so a wrong total still earns the shares worked out from it.

Built backwards from the bag size, the number of bags, the bag price and the two share sizes; the draw is kept only when both
divisions are exact.
"""

from __future__ import annotations

from mathgen.core import VerificationError, attempts, generator

from ._pr6_chain_kit import check_chain, whole_part

_SCENES = [
    dict(
        stem="A stall at the {town} markets sells {things} in bags of {t}. A school buys {k} bags for sports day at ${c} a bag. "
        "The {things} are shared equally among {m} classes, and each class shares its {things} equally among "
        "{g} teams.",
        ask_a="How many {things} does the school buy?",
        ask_b="How much does the school pay for the bags, in dollars?",
        ask_c="How many {things} does each class get?",
        ask_d="How many {things} does each team get?",
    ),
    dict(
        stem="A fruit shop in {town} packs {things} in bags of {t}. A club orders {k} bags for a camp at ${c} a bag. The "
        "{things} are shared equally among {m} tables, and the {g} children at each table share their {things} "
        "equally.",
        ask_a="How many {things} does the club order?",
        ask_b="How much does the club pay for the bags, in dollars?",
        ask_c="How many {things} does each table get?",
        ask_d="How many {things} does each child get?",
    ),
    dict(
        stem="At a fete in {town}, {things} are sold in bags of {t}. A group buys {k} bags at ${c} a bag and shares all the "
        "{things} equally among {m} families. Each family shares its {things} equally among {g} children.",
        ask_a="How many {things} does the group buy?",
        ask_b="How much does the group pay for the bags, in dollars?",
        ask_c="How many {things} does each family get?",
        ask_d="How many {things} does each child get?",
    ),
]
_THINGS = ["mandarins", "plums", "lollies", "muesli bars", "apricots", "oranges", "kiwi fruit", "mini pies"]
_TOWNS = ["Armidale", "Karratha", "Burnie", "Bairnsdale", "Port Lincoln", "Cooma", "Gympie", "Tamworth"]


@generator("exd_x_pr6_market_fact_family", level=5, presets={"exam": dict(tier="exam")})
def exd_x_pr6_market_fact_family(rng, tier: str = "exam"):
    scene = rng.choice(_SCENES)
    things, town = rng.choice(_THINGS), rng.choice(_TOWNS)
    for _ in attempts(2000, "a fact-family draw with two exact shares"):
        t, k, c = rng.choice([3, 4, 5, 10]), rng.randint(3, 12), rng.choice([3, 4, 5, 10])
        m, g = rng.choice([2, 3, 4, 5, 6, 10]), rng.choice([2, 3, 4, 5, 6])
        total = t * k
        if total % m != 0 or (total // m) % g != 0:
            continue
        share, each = total // m, total // m // g
        pay = k * c
        if each < 2:
            continue
        text = scene["stem"].format(town=town, things=things, t=t, k=k, c=c, m=m, g=g)
        try:
            check_chain(text, [total, pay, share, each])
        except VerificationError:
            continue
        break
    fmt = dict(things=things)
    return dict(
        text=text,
        parts=[
            whole_part(
                scene["ask_a"].format(**fmt),
                total,
                1,
                [
                    (t + k, "added-instead-of-multiplied"),
                    (t * (k + 1), "adjacent-table-fact"),
                    (t * (k - 1), "adjacent-table-fact"),
                ],
                working=[f"{k} × {t} = {total}."],
            ),
            whole_part(
                scene["ask_b"],
                pay,
                1,
                [
                    (t * c, "multiplied-by-wrong-quantity"),
                    (k + c, "added-instead-of-multiplied"),
                    ((k + 1) * c, "adjacent-table-fact"),
                ],
                working=[f"{k} × ${c} = ${pay}."],
            ),
            whole_part(
                scene["ask_c"].format(**fmt),
                share,
                2,
                [
                    (total * m, "multiplied-instead-of-divided"),
                    (total - m, "groups-total-minus-count"),
                    *([(total // g, "divided-by-wrong-group-size")] if total % g == 0 else []),
                ],
                follow_through=f"part_a/{m}",
                working=[f"{total} ÷ {m} = {share}, because {m} × {share} = {total}."],
            ),
            whole_part(
                scene["ask_d"].format(**fmt),
                each,
                2,
                [
                    (share * g, "multiplied-instead-of-divided"),
                    (share - g, "groups-total-minus-count"),
                    *([(total // g, "divided-by-wrong-group-size")] if total % g == 0 else []),
                ],
                follow_through=f"part_c/{g}",
                working=[f"{share} ÷ {g} = {each}, because {g} × {each} = {share}."],
            ),
        ],
    )
