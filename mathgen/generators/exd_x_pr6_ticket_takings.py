"""Exam-tier source for `f10-multidigit-multiplication`: a three-part money problem built from two products.

Two kinds of ticket are sold at two different prices. (a) and (b) each need a multi-digit product chosen by the
learner from the story (quantity times price); (c) combines both with a running cost, so the learner must also
decide to add the takings and subtract the cost. Part (c) carries `follow_through` on (a) and (b).

Built backwards from the quantities and prices; the money left is kept positive.
"""

from __future__ import annotations

from mathgen.core import VerificationError, attempts, generator

from ._pr6_chain_kit import check_chain, whole_part

_TOWNS = ["Bendigo", "Albany", "Gladstone", "Port Macquarie", "Devonport", "Alice Springs", "Orange", "Shepparton"]
_SCENES = [
    dict(
        stem="A school in {town} sells tickets to its end-of-year concert. {na} adult tickets are sold at ${A} each "
        "and {nc} child tickets are sold at ${C} each. The hall costs ${H} to hire.",
        ask_a="How much money is collected from the adult tickets, in dollars?",
        ask_b="How much money is collected from the child tickets, in dollars?",
        ask_c="After the hall hire is paid, how much money is left for the school, in dollars?",
    ),
    dict(
        stem="A football club in {town} sells {na} members' tickets at ${A} each and {nc} general tickets at ${C} each "
        "for a home game. Running the ground for the day costs ${H}.",
        ask_a="How much money do the members' tickets bring in, in dollars?",
        ask_b="How much money do the general tickets bring in, in dollars?",
        ask_c="After the ground costs are paid, how much money does the club have left, in dollars?",
    ),
    dict(
        stem="At the {town} swimming carnival, {na} spectators each pay ${A} for a day pass and {nc} competitors "
        "each pay ${C} to enter. Hiring the pool for the day costs ${H}.",
        ask_a="How much money is collected from the spectators, in dollars?",
        ask_b="How much money is collected from the competitors, in dollars?",
        ask_c="After the pool hire is paid, how much money is left, in dollars?",
    ),
]


def _reverse_digits(n: int) -> int:
    return int(str(n)[::-1])


def _unshifted(q: int, price: int) -> int:
    """q x price with the tens partial product left unshifted: q x units + q x tens digit."""
    return q * (price % 10) + q * (price // 10)


def _product_candidates(q: int, price: int) -> list[tuple[int, str]]:
    return [
        (_unshifted(q, price), "partial-product-not-shifted"),
        (q * _reverse_digits(price), "digits-transposed"),
        (q + price, "added-instead-of-multiplied"),
    ]


@generator("exd_x_pr6_ticket_takings", level=5, presets={"exam": dict(tier="exam")})
def exd_x_pr6_ticket_takings(rng, tier: str = "exam"):
    scene = rng.choice(_SCENES)
    town = rng.choice(_TOWNS)
    for _ in attempts(1000, "a ticket draw with a sensible profit"):
        na, nc = rng.randint(40, 199), rng.randint(30, 160)
        a_price, c_price = rng.randint(12, 48), rng.randint(11, 29)
        hire = rng.randint(1, 12) * 50
        a, b = na * a_price, nc * c_price
        left = a + b - hire
        if left < 500 or a_price % 10 == 0 or c_price % 10 == 0 or a_price == c_price:
            continue
        if _reverse_digits(a_price) == a_price or _reverse_digits(c_price) == c_price:
            continue
        text = scene["stem"].format(town=town, na=na, nc=nc, A=a_price, C=c_price, H=hire)
        try:
            check_chain(text, [a, b, left])
        except VerificationError:
            continue
        break
    return dict(
        text=text,
        parts=[
            whole_part(
                scene["ask_a"],
                a,
                2,
                _product_candidates(na, a_price),
                working=[f"{na} × {a_price} = {a}."],
            ),
            whole_part(
                scene["ask_b"],
                b,
                2,
                _product_candidates(nc, c_price),
                working=[f"{nc} × {c_price} = {b}."],
            ),
            whole_part(
                scene["ask_c"],
                left,
                2,
                [
                    (a - hire, "story-one-quantity-ignored"),
                    (a + b, "story-one-quantity-ignored"),
                    (a + b + hire, "story-operation-reversed"),
                ],
                follow_through=f"part_a + part_b - {hire}",
                working=[f"{a} + {b} = {a + b}.", f"{a + b} − {hire} = {left}."],
            ),
        ],
    )
