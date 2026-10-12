"""Exam tier for "use currency exchange rates to determine the cost in Australian dollars".

The practice tiers convert one amount at one rate. Here a traveller buys foreign currency at one rate, spends
part of it, and sells the rest back at a worse rate, so the learner must choose the right rate for each
direction, convert only what is left, and then find what the spending really cost. Three parts:

(a) the foreign currency received for the Australian dollars (2 marks);
(b) the Australian dollars received for the unspent part, to the nearest cent (3 marks, follow-through);
(c) what the foreign currency actually spent cost, in Australian dollars (2 marks, follow-through).

Built backwards from whole dollar amounts: the Australian dollars spent are a multiple of 50 and both rates have
two decimal places, so every foreign amount is exact; only the final conversion back is rounded, once, to the
cent. The rates are stated in the question and are not real quotations.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    check,
    check_distinct,
    draw_until,
    generator,
    latex_answer,
    numeric_answer,
    phrase,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)
from mathgen.slots import NAMES

_TOL = stated_precision_tolerance(2)

# (symbol, place, plural noun, low and high buying rate in hundredths)
_CURRENCIES = (
    ("US$", "New York", "US dollars", 58, 72),
    ("€", "Rome", "euros", 54, 66),
    ("£", "Edinburgh", "pounds", 44, 54),
    ("NZ$", "Queenstown", "New Zealand dollars", 104, 116),
    ("S$", "Singapore", "Singapore dollars", 82, 94),
)
_SPENDING = ("meals, tickets and gifts", "accommodation and tours", "shopping and transport", "food and souvenirs")
_STEM = (
    "{name} is travelling to {place} and exchanges A${aud} for {noun} at a currency exchange. To buy {noun}, "
    "A$1 = {sym}{rb}. To sell {noun} back, A$1 = {sym}{rs}. {name} spends {sym}{spent} on {what} and sells all "
    "the {noun} left over back to the exchange at the end of the trip.",
    "A currency exchange in Australia quotes A$1 = {sym}{rb} when it sells {noun} and A$1 = {sym}{rs} when it "
    "buys them back. {name} pays A${aud} for {noun} before a trip to {place}, spends {sym}{spent} on {what}, "
    "and returns the rest to the exchange after the trip.",
    "Before flying to {place}, {name} buys {noun} with A${aud}. The exchange sells at A$1 = {sym}{rb} and buys "
    "back at A$1 = {sym}{rs}. On the trip {name} spends {sym}{spent} on {what}, then sells what is left back.",
)


def _cents(v: Fraction) -> Fraction:
    return round_half_up(v, 2)


def _txt(v: Fraction) -> str:
    return stated_precision_string(_cents(v), 2)


def _hundredths(n: int) -> str:
    return f"{n // 100}.{n % 100:02d}"


def _part(prompt, key, marks, working, cands, follow_through=None):
    key = _cents(key)
    vals = [(_cents(v), tag) for v, tag in cands]
    check_distinct(key, *[v for v, _ in vals], msg="a distractor equals the answer")
    text = _txt(key)
    shown = [Distractor(_txt(v), tag, v) if tag else _txt(v) for v, tag in vals]
    out = dict(
        prompt=prompt,
        answer_kind="number",
        answer=numeric_answer(key),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        sensible_range=(0, None),
        working=working,
        distractors=pick_distractors(text, shown, 3, tolerance=_TOL),
    )
    if follow_through:
        out["follow_through"] = follow_through
    return out


def _draw(rng):
    sym, place, noun, lo, hi = rng.choice(_CURRENCIES)
    aud = rng.randrange(500, 3001, 50)
    rb = rng.randint(lo, hi)
    rs = rb + rng.randint(2, 6)
    foreign = Fraction(aud * rb, 100)
    spent = rng.randrange(int(foreign * Fraction(3, 10)) // 10 * 10, int(foreign * Fraction(8, 10)) // 10 * 10 + 1, 10)
    return sym, place, noun, aud, rb, rs, foreign, spent


def _acceptable(d) -> bool:
    _, _, _, aud, rb, rs, foreign, spent = d
    left = foreign - spent
    if spent < 100 or left < 50:
        return False
    back = left * 100 / rs
    frac = (back * 100) % 1
    return abs(frac - Fraction(1, 2)) > Fraction(1, 50) and back < aud


@generator("exd_x_n2_currency_round_trip", level=10, presets={"exam": dict(tier="exam")})
def exd_x_n2_currency_round_trip(rng, tier="exam"):
    """Buy, spend, sell back; see the module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    sym, place, noun, aud, rb, rs, foreign, spent = draw_until(
        rng, _draw, _acceptable, what="a currency round trip with a clear cent"
    )
    r_buy, r_sell = Fraction(rb, 100), Fraction(rs, 100)
    left = foreign - spent
    back = _cents(left / r_sell)
    cost = aud - back
    check(foreign == aud * r_buy and left > 0 and 0 < back < aud and cost > 0, "chain is inconsistent")

    name = rng.choice(NAMES)
    stem = phrase(
        rng,
        _STEM,
        name=name,
        place=place,
        noun=noun,
        sym=sym,
        aud=f"{aud:,}",
        rb=_hundredths(rb),
        rs=_hundredths(rs),
        spent=f"{spent:,}",
        what=rng.choice(_SPENDING),
    )
    ft_b = f"(part_a - {spent})/({rs}/100)"
    a_cands = [
        (aud / r_buy, "exchange-rate-direction-reversed"),
        (aud * r_sell, "used-the-other-rate"),
        (aud + r_buy, "adds-the-rate"),
    ]
    b_cands = [
        (left * r_sell, "exchange-rate-direction-reversed"),
        (foreign / r_sell, "converted-whole-amount-back"),
        (left / r_buy, "converted-back-at-outbound-rate"),
        (left, "left-foreign-amount-unconverted"),
    ]
    c_cands = [
        (back, "reported-money-back-not-cost"),
        (Fraction(spent), "foreign-amount-treated-as-dollars"),
        (spent / r_buy, None),
        (aud - left / r_buy, None),
    ]
    parts = [
        _part(
            f"How many {noun} does {name} receive for the A${aud:,}? Give the answer to the nearest cent.",
            foreign,
            2,
            [f"To buy {noun}, multiply by the buying rate: {aud} × {_hundredths(rb)} = {sym}{_txt(foreign)}."],
            a_cands,
        ),
        _part(
            f"How many Australian dollars does {name} receive for the {noun} sold back? Give the answer to the "
            "nearest cent.",
            back,
            3,
            [
                f"Left over = {_txt(foreign)} - {spent} = {sym}{_txt(left)}.",
                f"Selling back, divide by the selling rate: {_txt(left)} ÷ {_hundredths(rs)} = A${_txt(back)}.",
            ],
            b_cands,
            follow_through=ft_b,
        ),
        _part(
            f"What did the {sym}{spent:,} that {name} spent really cost, in Australian dollars? Give the answer "
            "to the nearest cent.",
            cost,
            2,
            [f"Cost = dollars paid - dollars received back = {aud} - {_txt(back)} = A${_txt(cost)}."],
            c_cands,
            follow_through=f"{aud} - part_b",
        ),
    ]
    return dict(text=stem, parts=parts)
