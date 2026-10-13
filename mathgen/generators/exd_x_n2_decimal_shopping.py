"""Exam tier for "operate with decimals".

The practice tiers perform one decimal operation. Here a market shop needs four decimal operations in a row,
and the learner must decide which: price per kilogram times a decimal mass, a whole number of items at a
decimal price, subtraction from a note to find the change, and then division of that decimal change by a
price per kilogram to see how much more can be bought. Three parts:

(a) the cost of the produce by mass (2 marks);
(b) the change from the note after also buying several of another item (2 marks, follow-through from (a));
(c) the mass of a second kind of produce the change buys, to 2 decimal places (2 marks, follow-through from (b)).

Built backwards from exact cents: the mass (two decimal places) and price are drawn so that their product is a
whole number of cents; the second price is drawn from those that divide the change into a mass with at most two
decimal places, so every answer is exact and nothing is rounded.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    check,
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
# (food, where, lowest and highest plausible price per kilogram in dollars)
_PRODUCE = (
    ("lamb mince", "a butcher in Orange", 14, 26),
    ("tomatoes", "a farmers' market in Bendigo", 4, 9),
    ("cheddar cheese", "a deli in Launceston", 16, 30),
    ("green beans", "a greengrocer in Mackay", 6, 12),
    ("peaches", "a roadside stall near Mildura", 5, 12),
)
_SECOND = ("apples", "pears", "mandarins", "potatoes", "bananas", "carrots")
_EXTRA = (
    ("jars of honey", "jar"),
    ("loaves of sourdough", "loaf"),
    ("bottles of cordial", "bottle"),
    ("tubs of dip", "tub"),
)
_STEMS = (
    "{name} is shopping at {where}. {name} buys {w} kg of {food} at ${p} per kilogram and {n} {extra} at ${e} each, "
    "and pays with a ${note} note.",
    "At {where}, {name} buys {w} kg of {food} priced at ${p} per kilogram, and also {n} {extra} at ${e} each. "
    "{name} hands over a ${note} note.",
    "{name} pays with a ${note} note at {where} for {n} {extra} priced at ${e} each and {w} kg of {food} priced "
    "at ${p} per kilogram.",
)


def _m(v: Fraction) -> str:
    return stated_precision_string(v, 2)


def _draw(rng):
    food, where, lo, hi = rng.choice(_PRODUCE)
    w = Fraction(rng.randint(60, 380), 100)
    p = Fraction(rng.randrange(lo * 100 + 5, hi * 100, 5), 100)
    n = rng.randint(2, 6)
    e = Fraction(rng.randrange(250, 750, 5), 100)
    a = w * p
    if a.denominator != 1 and (a * 100).denominator != 1:
        return None
    total = a + n * e
    note = next((v for v in (20, 50, 100) if 4 <= v - total <= 45), None)
    if note is None:
        return None
    change = note - total
    p2s = [Fraction(c, 100) for c in range(150, 1001, 5) if (change * 100 * 100) % c == 0]
    p2s = [q for q in p2s if Fraction(1, 5) <= change / q <= 6 and change / q != 1]
    if not p2s:
        return None
    p2 = rng.choice(p2s)
    return food, where, w, p, n, e, a, total, note, change, p2


@generator("exd_x_n2_decimal_shopping", level=10, presets={"exam": dict(tier="exam")})
def exd_x_n2_decimal_shopping(rng, tier="exam"):
    """Cost by mass, change from a note, mass bought with the change; see the module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    food, where, w, p, n, e, a, total, note, change, p2 = draw_until(
        rng, _draw, lambda d: d is not None, what="a shopping trip with exact decimal answers"
    )
    kg = change / p2
    check(kg * 100 == int(kg * 100) and kg * p2 == change and a == w * p, "shopping answers are not exact")
    name = rng.choice(NAMES)
    extra, _single = rng.choice(_EXTRA)
    second = rng.choice(_SECOND)
    stem = phrase(
        rng, _STEMS, name=name, where=where, w=_m(w), food=food, p=_m(p), n=n, extra=extra, e=_m(e), note=note
    )

    def part(prompt, key, marks, work, cands, ft=None):
        cands = [(round_half_up(v, 2), tag) for v, tag in cands]
        shown = [Distractor(_m(v), tag, v) if tag else _m(v) for v, tag in cands]
        picked = pick_distractors(_m(key), shown, 3, tolerance=_TOL)
        out = dict(
            prompt=prompt,
            answer_kind="number",
            answer=numeric_answer(key),
            answer_text=_m(key),
            answer_latex=latex_answer(_m(key)),
            marks=marks,
            sensible_range=(0, None),
            working=work,
            distractors=picked,
        )
        if ft:
            out["follow_through"] = ft
        return out

    e_total = n * e
    ft_b = f"{note} - part_a - {_m(e_total)}"
    ft_c = f"part_b/({_m(p2)})"
    parts = [
        part(
            f"How much do the {_m(w)} kg of {food} cost, in dollars? Give the answer to the nearest cent.",
            a,
            2,
            [f"Cost = {_m(w)} × {_m(p)} = ${_m(a)}."],
            [
                (a * 10, "decimal-point-one-place-too-far-right"),
                (a / 10, "decimal-point-one-place-too-far-left"),
                (w + p, "added-quantity-and-price"),
            ],
        ),
        part(
            f"How much change does {name} receive, in dollars? Give the answer to the nearest cent.",
            change,
            2,
            [
                f"{n} × {_m(e)} = {_m(e_total)}; total = {_m(a)} + {_m(e_total)} = {_m(total)}; change = {note} - {_m(total)} = ${_m(change)}."
            ],
            [
                (note - a, "forgot-second-purchase"),
                (total, "reported-total-not-change"),
                (note - a - e, None),
            ],
            ft=ft_b,
        ),
        part(
            f"{name} spends all of the change on {second} at ${_m(p2)} per kilogram. How many kilograms of {second} "
            "is that? Give the answer to the nearest hundredth.",
            kg,
            2,
            [f"Mass = change ÷ price = {_m(change)} ÷ {_m(p2)} = {_m(kg)} kg."],
            [
                (p2 / change, "divided-the-wrong-way"),
                (Fraction(note) / p2, "used-note-not-change"),
                (change * p2, "multiplied-where-division-needed"),
            ],
            ft=ft_c,
        ),
    ]
    return dict(text=stem, parts=parts)
