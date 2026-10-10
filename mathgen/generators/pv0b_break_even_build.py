"""The break-even quantity where a cost line and a revenue line intersect.

One skill: ``senior-solve-practical-problems-that-involve-finding-the-point``
(ACMGM045), ``domain: linear-relations``, ``answer: number`` (exact whole
number, < 1000), ``prerequisites: [rate-of-change]``.

Scope: from a stated
fixed cost, cost per unit and selling price per unit, find the break-even
quantity at which total cost equals total revenue -- the intersection of the
cost line ``C = F + v*q`` and the revenue line ``R = p*q``. The graphical
method is dropped; every tier forms the two rules algebraically and equates
them.

Four tiers, genuine reasoning escalation:

- ``stated`` (easy): ``F``, ``v``, ``p`` all stated directly; the break-even
  quantity ``q = F/(p - v)`` is a whole number by construction. Decision:
  form both rules and equate them.
- ``round_up`` (medium): same three givens, but the exact break-even is NOT a
  whole number by construction; the prompt states the rounding rule ("the
  smallest whole number of units that must be sold to make a profit").
  Decision: equate, then round UP even when the fractional part is small (a
  profit needs strict inequality, not equality).
- ``two_points`` (hard): the unit cost is not stated directly -- the cost
  line is given by two (quantity, total cost) points; price is stated.
  Decision: recover the gradient (unit cost) and intercept (fixed cost) from
  the two points (the `rate-of-change` prerequisite) before equating. Whole
  number by construction.
- ``percentage_fee`` (exam): a per-sale fee, stated as a percentage of the
  selling price, is added to ``F``, ``v`` and ``p``, with the rounding rule
  stated. Decision: recognise the percentage fee as reducing the per-unit
  margin (``p*(1 - r/100) - v``), not a separate cost line, then round up.
  Exact ``Fraction`` arithmetic; the exact break-even is constructed
  non-integer and never within 0.01 of an integer.

Every tier's construction is proven a second way: the "equate the two rules"
identity is checked with `check_equivalent` against the independently
re-derived quotient, and every distractor is proven a genuinely different
value with `check_pairwise_distinct` before it is offered.
"""

from __future__ import annotations

import math
from fractions import Fraction

from mathgen.core import (
    VerificationError,
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    pick_distractors,
    round_half_up,
)

_RESAMPLE_ATTEMPTS = 400

# --- shared draw ranges -----------------------------------------------------

_F_LO = 100

# (4..60, ratio 1.2..4) independent of what the context said it was, so a
# loaf could cost $60 and sell for $145. Every context now carries its own
# plausible unit-cost range, price ceiling and fixed-cost cap, chosen so
# products genuinely span the range instead of all crowding the same few
# dollars: bakery/print-shop/phone-case at the cheap end, cakes/mugs/
# candles/t-shirts in the middle, and a handful of higher-priced,
# higher-fixed-cost products (handmade furniture, bicycles, skateboards) at
# the top so the key's answer space stays wide without any one context
# stretching past what a teacher would accept for it.
_CONTEXTS = (
    dict(
        business="A bakery",
        verb="make",
        noun_sg="loaf",
        noun_pl="loaves",
        gerund="Making",
        v_range=(2, 8),
        p_cap=15,
        F_cap=2500,
    ),
    dict(
        business="A cake shop",
        verb="make",
        noun_sg="cake",
        noun_pl="cakes",
        gerund="Making",
        v_range=(8, 30),
        p_cap=60,
        F_cap=6000,
    ),
    dict(
        business="A phone case maker",
        verb="make",
        noun_sg="case",
        noun_pl="cases",
        gerund="Making",
        v_range=(4, 15),
        p_cap=30,
        F_cap=5000,
    ),
    dict(
        business="A print shop",
        verb="produce",
        noun_sg="poster",
        noun_pl="posters",
        gerund="Producing",
        v_range=(3, 15),
        p_cap=30,
        F_cap=5000,
    ),
    dict(
        business="A pottery studio",
        verb="make",
        noun_sg="mug",
        noun_pl="mugs",
        gerund="Making",
        v_range=(4, 12),
        p_cap=25,
        F_cap=4000,
    ),
    dict(
        business="A t-shirt printer",
        verb="print",
        noun_sg="t-shirt",
        noun_pl="t-shirts",
        gerund="Printing",
        v_range=(6, 20),
        p_cap=40,
        F_cap=6000,
    ),
    dict(
        business="A candle maker",
        verb="make",
        noun_sg="candle",
        noun_pl="candles",
        gerund="Making",
        v_range=(5, 20),
        p_cap=40,
        F_cap=5000,
    ),
    dict(
        business="A furniture workshop",
        verb="build",
        noun_sg="handmade chair",
        noun_pl="handmade chairs",
        gerund="Building",
        v_range=(40, 150),
        p_cap=350,
        F_cap=15000,
    ),
    dict(
        business="A skateboard workshop",
        verb="build",
        noun_sg="skateboard",
        noun_pl="skateboards",
        gerund="Building",
        v_range=(30, 90),
        p_cap=220,
        F_cap=10000,
    ),
    dict(
        business="A garden furniture maker",
        verb="build",
        noun_sg="garden bench",
        noun_pl="garden benches",
        gerund="Building",
        v_range=(50, 180),
        p_cap=400,
        F_cap=15000,
    ),
    dict(
        business="A bicycle workshop",
        verb="build",
        noun_sg="bicycle",
        noun_pl="bicycles",
        gerund="Building",
        v_range=(80, 300),
        p_cap=700,
        F_cap=20000,
    ),
)

_CONTEXTS_BY_NOUN_SG = {c["noun_sg"]: c for c in _CONTEXTS}


def _draw_price(rng, v: int, p_cap: int | None = None) -> int | None:
    """Draw a selling price `p` with `p > v`, `p/v` in about [1.2, 4], and
    (when given) capped at the context's own plausible price ceiling."""
    lo = max(v + 1, math.ceil(Fraction(6, 5) * v))
    hi = math.floor(4 * v)
    if p_cap is not None:
        hi = min(hi, p_cap)
    if lo > hi:
        return None
    return rng.randint(lo, hi)


def _draw_context_and_price(rng):
    """Draw a context, then a unit cost `v` and price `p` inside THAT
    context's own plausible range -- never a price or
    cost drawn independently of what the context says it is."""
    ctx = _CONTEXTS[rng.randrange(len(_CONTEXTS))]
    v = rng.randint(*ctx["v_range"])
    p = _draw_price(rng, v, ctx["p_cap"])
    if p is None:
        return None, None, None
    return ctx, v, p


def _round_half_up_int(value: Fraction) -> int:
    return int(round_half_up(value, 0))


def _collides_with_printed(printed: set, *candidates: int) -> bool:
    """Whether any candidate equals a number the prompt itself prints
    (F, v, p, r, or a two-points quantity/total cost) -- guards lesson 9
    (no distractor equals a printed number)."""
    return any(c in printed for c in candidates)


# =============================================================================
# easy: stated (whole-number break-even, all three givens stated)
# =============================================================================

_EASY_TEMPLATES = [
    "{business} has fixed costs of ${F} per month. Each {noun_sg} costs ${v} to "
    "{verb} and sells for ${p}. How many {noun_pl} must be sold each month for "
    "total revenue to equal total cost (the break-even point)?",
    "{business} has fixed costs of ${F} per month. Each {noun_sg} costs ${v} to "
    "{verb} and sells for ${p}. Find the break-even number of {noun_pl} -- the "
    "point at which total revenue equals total cost.",
    "{business} has fixed costs of ${F} per month. Each {noun_sg} costs ${v} to "
    "{verb} and sells for ${p}. At what number of {noun_pl} sold does total "
    "revenue equal total cost?",
]


def _build_stated(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        ctx, v, p = _draw_context_and_price(rng)
        if ctx is None:
            continue
        diff = p - v
        q_hi = min(900, ctx["F_cap"] // diff)
        if q_hi < 12:
            continue
        q = rng.randint(12, q_hi)
        F = q * diff
        if F < _F_LO:
            continue

        Ff, pf, vf = Fraction(F), Fraction(p), Fraction(v)
        ignores_unit_cost = _round_half_up_int(Ff / pf)
        adds_unit_cost = _round_half_up_int(Ff / (pf + vf))

        try:
            check_pairwise_distinct(
                q,
                ignores_unit_cost,
                adds_unit_cost,
                msg="stated-tier misconception collides with the answer or another option",
            )
        except VerificationError:
            continue
        if not (1 <= ignores_unit_cost < 1000 and 1 <= adds_unit_cost < 1000):
            continue
        if _collides_with_printed({F, v, p}, ignores_unit_cost, adds_unit_cost):
            continue
        break
    else:
        raise ValueError("pv0b_break_even: no collision-free 'stated' draw found")

    # Independent route: equate C(q) = F + v*q with R(q) = p*q and solve.
    q_check = Ff / (pf - vf)
    check_equivalent(q_check, Fraction(q), "stated-tier: F/(p-v) must reproduce the constructed break-even")
    check(Ff + vf * q == pf * q, "stated-tier: cost and revenue must be equal at the constructed break-even")

    template = _EASY_TEMPLATES[rng.randrange(len(_EASY_TEMPLATES))]
    text = template.format(
        business=ctx["business"], F=F, noun_sg=ctx["noun_sg"], v=v, verb=ctx["verb"], p=p, noun_pl=ctx["noun_pl"]
    )
    answer_text = str(q)

    distractors = pick_distractors(answer_text, [str(ignores_unit_cost), str(adds_unit_cost)], n=2)
    check(len(distractors) == 2, "stated tier expects 2 distinct distractors")

    return dict(
        text=text,
        latex=text,
        answer=q,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Cost: C(q) = {F} + {v}q. Revenue: R(q) = {p}q. Equate: {F} + {v}q = {p}q, so q = {F} / ({p} - {v}) = {q}."
        ],
        distractors=distractors,
    )


# =============================================================================
# medium: round_up (exact break-even is not whole; round up for profit)
# =============================================================================

_ROUND_UP_TEMPLATES = [
    "{business} has fixed costs of ${F} per month. Each {noun_sg} costs ${v} to "
    "{verb} and sells for ${p}. What is the smallest whole number of {noun_pl} "
    "that must be sold each month for the business to make a profit?",
    "{business} has fixed costs of ${F} per month. Each {noun_sg} costs ${v} to "
    "{verb} and sells for ${p}. What is the smallest whole number of {noun_pl} "
    "that must be sold each month for the business to turn a profit?",
    "{business} has fixed costs of ${F} per month. Each {noun_sg} costs ${v} to "
    "{verb} and sells for ${p}. How many {noun_pl}, at minimum, must be sold "
    "each month for the business's revenue to exceed its costs?",
]


def _build_round_up(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        ctx, v, p = _draw_context_and_price(rng)
        if ctx is None:
            continue
        diff = p - v
        if diff < 2:
            continue
        q_floor_hi = min(899, (ctx["F_cap"] - 1) // diff)
        if q_floor_hi < 12:
            continue
        q_floor = rng.randint(12, q_floor_hi)
        frac_num = rng.randint(1, diff - 1)
        F = q_floor * diff + frac_num
        if not (_F_LO <= F <= ctx["F_cap"]):
            continue
        answer = q_floor + 1
        if answer >= 1000:
            continue

        Ff, pf, vf = Fraction(F), Fraction(p), Fraction(v)
        q_exact = Ff / (pf - vf)
        check(q_exact.denominator != 1, "round_up-tier: exact break-even must be non-integer by construction")

        ignores_unit_cost = math.ceil(Ff / pf)
        adds_unit_cost = math.ceil(Ff / (pf + vf))
        # Offer at most one, chosen 50/50: rounds-down (q_floor, the learner who forgets a profit
        # needs STRICT inequality) or one-extra-unit (the learner who adds one more unit "to make a
        # profit" after rounding up).
        adjacent_distractor = q_floor if rng.random() < 0.5 else answer + 1

        try:
            check_pairwise_distinct(
                answer,
                ignores_unit_cost,
                adds_unit_cost,
                adjacent_distractor,
                msg="round_up-tier misconception collides with the answer or another option",
            )
        except VerificationError:
            continue
        if not (1 <= ignores_unit_cost < 1000 and 1 <= adds_unit_cost < 1000 and 1 <= adjacent_distractor < 1000):
            continue
        if _collides_with_printed({F, v, p}, ignores_unit_cost, adds_unit_cost, adjacent_distractor):
            continue
        break
    else:
        raise ValueError("pv0b_break_even: no collision-free 'round_up' draw found")

    check_equivalent(math.ceil(q_exact), answer, "round_up-tier: ceil(F/(p-v)) must reproduce the constructed answer")

    template = _ROUND_UP_TEMPLATES[rng.randrange(len(_ROUND_UP_TEMPLATES))]
    text = template.format(
        business=ctx["business"], F=F, noun_sg=ctx["noun_sg"], v=v, verb=ctx["verb"], p=p, noun_pl=ctx["noun_pl"]
    )
    answer_text = str(answer)

    distractors = pick_distractors(
        answer_text,
        [str(ignores_unit_cost), str(adds_unit_cost), str(adjacent_distractor)],
        n=3,
    )
    check(len(distractors) >= 1, "round_up tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Cost: C(q) = {F} + {v}q. Revenue: R(q) = {p}q. Exact break-even q = {F} / "
            f"({p} - {v}) = {q_exact.numerator}/{q_exact.denominator}, which is not a whole number. "
            f"A profit needs revenue strictly greater than cost, so round UP: {answer} units."
        ],
        distractors=distractors,
    )


# =============================================================================
# hard: two_points (cost line recovered from two (quantity, total cost) points)
# =============================================================================

_TWO_POINTS_TEMPLATE = (
    "{gerund} {q1} {noun_pl} costs ${cost1} in total, and {gerund_lower} {q2} "
    "{noun_pl} costs ${cost2} in total. Each {noun_sg} sells for ${p}. Find the "
    "break-even number of {noun_pl} -- the point at which total revenue equals "
    "total cost."
)


def _build_two_points(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        ctx, v, p = _draw_context_and_price(rng)
        if ctx is None:
            continue
        diff = p - v
        q_hi = min(900, ctx["F_cap"] // diff)
        if q_hi < 12:
            continue
        q = rng.randint(12, q_hi)
        F = q * diff
        if F < _F_LO:
            continue

        q1 = rng.randint(50, 400)
        gap = rng.randint(60, 400)
        q2 = q1 + gap

        cost1 = F + v * q1
        cost2 = F + v * q2

        Ff, pf, vf = Fraction(F), Fraction(p), Fraction(v)
        avg_cost_wrong = Fraction(cost2, q2)
        margin_wrong = pf - avg_cost_wrong
        if margin_wrong <= 0:
            continue
        average_cost_as_unit_cost = _round_half_up_int(Ff / margin_wrong)

        ignores_unit_cost = _round_half_up_int(Ff / pf)
        adds_unit_cost = _round_half_up_int(Ff / (pf + vf))

        try:
            check_pairwise_distinct(
                q,
                average_cost_as_unit_cost,
                ignores_unit_cost,
                adds_unit_cost,
                msg="two_points-tier misconception collides with the answer or another option",
            )
        except VerificationError:
            continue
        if not (1 <= average_cost_as_unit_cost < 1000 and 1 <= ignores_unit_cost < 1000 and 1 <= adds_unit_cost < 1000):
            continue
        if _collides_with_printed(
            {q1, cost1, q2, cost2, p}, average_cost_as_unit_cost, ignores_unit_cost, adds_unit_cost
        ):
            continue
        break
    else:
        raise ValueError("pv0b_break_even: no collision-free 'two_points' draw found")

    # Independent route: recover the gradient (unit cost) and intercept (fixed
    # cost) from the two stated points, then equate -- never trust the
    # construction direction.
    v_computed = Fraction(cost2 - cost1, q2 - q1)
    F_computed = Fraction(cost1) - v_computed * q1
    check_equivalent(v_computed, vf, "two_points-tier: gradient recovered from the two points must reproduce v")
    check_equivalent(F_computed, Ff, "two_points-tier: intercept recovered from the two points must reproduce F")
    q_computed = F_computed / (pf - v_computed)
    check_equivalent(q_computed, Fraction(q), "two_points-tier: recovered break-even must reproduce the constructed q")

    text = _TWO_POINTS_TEMPLATE.format(
        gerund=ctx["gerund"],
        gerund_lower=ctx["gerund"].lower(),
        q1=q1,
        cost1=cost1,
        q2=q2,
        cost2=cost2,
        noun_sg=ctx["noun_sg"],
        noun_pl=ctx["noun_pl"],
        p=p,
    )
    answer_text = str(q)

    distractors = pick_distractors(
        answer_text,
        [str(average_cost_as_unit_cost), str(ignores_unit_cost), str(adds_unit_cost)],
        n=3,
    )
    check(len(distractors) >= 1, "two_points tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=text,
        answer=q,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Gradient (unit cost): v = ({cost2} - {cost1}) / ({q2} - {q1}) = {v}. "
            f"Intercept (fixed cost): F = {cost1} - {v} x {q1} = {F}. "
            f"Equate cost and revenue: {F} + {v}q = {p}q, so q = {F} / ({p} - {v}) = {q}."
        ],
        distractors=distractors,
    )


# =============================================================================
# exam: percentage_fee (a per-sale percentage fee changes the margin)
# =============================================================================

_FEE_R = [2, 3, 4, 5, 6, 8, 10, 12]

_PERCENTAGE_FEE_TEMPLATES = [
    "{business} has fixed costs of ${F} per month. Each {noun_sg} costs ${v} to "
    "{verb} and sells for ${p}, but the payment platform charges a fee equal to "
    "{r}% of the selling price on every {noun_sg} sold. What is the smallest "
    "whole number of {noun_pl} that must be sold each month for the business to "
    "make a profit?",
    "{business} has fixed costs of ${F} per month. Each {noun_sg} costs ${v} to "
    "{verb} and sells for ${p}, but a card processing fee equal to {r}% of the "
    "selling price is charged on every {noun_sg} sold. What is the smallest "
    "whole number of {noun_pl} that must be sold each month for the business to "
    "make a profit?",
    "{business} has fixed costs of ${F} per month. Each {noun_sg} costs ${v} to "
    "{verb} and sells for ${p}, but the marketplace takes a commission of {r}% "
    "of the selling price on every {noun_sg} sold. What is the smallest whole "
    "number of {noun_pl} that must be sold each month for the business to make "
    "a profit?",
]


def _build_percentage_fee(rng) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        ctx, v, p = _draw_context_and_price(rng)
        if ctx is None:
            continue
        r = rng.choice(_FEE_R)
        F = rng.randint(_F_LO, ctx["F_cap"])

        vf, pf, Ff, rf = Fraction(v), Fraction(p), Fraction(F), Fraction(r)
        margin = pf * (100 - rf) / 100 - vf
        if margin <= 0:
            continue
        q_exact = Ff / margin
        if q_exact.denominator == 1:
            continue
        # Never within 0.01 of an integer, per the boundary.
        nearest = round(q_exact)
        if abs(q_exact - nearest) <= Fraction(1, 100):
            continue

        answer = math.ceil(q_exact)
        if answer >= 1000:
            continue
        adjacent_distractor = (answer - 1) if rng.random() < 0.5 else (answer + 1)
        banned_adjacent = (answer + 1) if adjacent_distractor == answer - 1 else (answer - 1)

        margin_no_fee = pf - vf
        fee_ignored = math.ceil(Ff / margin_no_fee)

        margin_fee_on_cost = pf - vf * (100 + rf) / 100
        fee_on_cost = math.ceil(Ff / margin_fee_on_cost) if margin_fee_on_cost > 0 else None

        candidates = [adjacent_distractor, fee_ignored]
        if fee_on_cost is not None:
            candidates.append(fee_on_cost)
        candidates = [c for c in candidates if c != banned_adjacent]

        try:
            check_pairwise_distinct(
                answer, *candidates, msg="percentage_fee-tier misconception collides with the answer or another option"
            )
        except VerificationError:
            continue
        if any(not (1 <= c < 1000) for c in candidates):
            continue
        if _collides_with_printed({F, v, p, r}, *candidates):
            continue
        break
    else:
        raise ValueError("pv0b_break_even: no collision-free 'percentage_fee' draw found")

    check_equivalent(math.ceil(q_exact), answer, "percentage_fee-tier: ceil(F/margin) must reproduce the answer")

    template = _PERCENTAGE_FEE_TEMPLATES[rng.randrange(len(_PERCENTAGE_FEE_TEMPLATES))]
    text = template.format(
        business=ctx["business"], F=F, noun_sg=ctx["noun_sg"], v=v, verb=ctx["verb"], p=p, r=r, noun_pl=ctx["noun_pl"]
    )
    answer_text = str(answer)

    candidate_texts = [str(c) for c in candidates]
    distractors = pick_distractors(answer_text, candidate_texts, n=len(candidate_texts))
    check(len(distractors) >= 1, "percentage_fee tier expects at least 1 distinct distractor")

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"The fee is {r}% of the selling price, so the margin per unit is "
            f"{p} x (1 - {r}/100) - {v} = {margin.numerator}/{margin.denominator}. "
            f"Exact break-even q = {F} / margin = {q_exact.numerator}/{q_exact.denominator}, "
            f"not a whole number, so round UP: {answer} units."
        ],
        distractors=distractors,
    )


@generator(
    "pv0b_break_even",
    level=12,
    presets={
        "easy": dict(mode="stated"),
        "medium": dict(mode="round_up"),
        "hard": dict(mode="two_points"),
        "exam": dict(mode="percentage_fee"),
    },
)
def pv0b_break_even(rng, mode="stated"):
    """The break-even quantity where a linear cost rule meets a linear
    revenue rule. easy: F, v, p all stated, whole-number break-even.
    medium: same three givens, but the exact break-even is not whole; round
    UP for a profit. hard: v and F are recovered from two (quantity, total
    cost) points instead of being stated directly. exam: a percentage fee on
    every sale reduces the per-unit margin; round UP for a profit."""
    if mode == "stated":
        return _build_stated(rng)
    if mode == "round_up":
        return _build_round_up(rng)
    if mode == "two_points":
        return _build_two_points(rng)
    if mode == "percentage_fee":
        return _build_percentage_fee(rng)
    raise ValueError(f"unknown mode: {mode!r}")
