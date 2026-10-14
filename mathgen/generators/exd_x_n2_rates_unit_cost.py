"""Exam tier for "use rates to make comparisons".

The practice tiers compare two prices per litre. Here the learner must turn two differently packaged or
differently specified options into a common rate, then carry the comparison forward to a quantity: how much
dearer one option is over a larger purchase or a longer trip. Two settings are used, each giving three parts:

* two brands sold in different pack sizes (price per 100 g), then the saving on a larger purchase;
* two cars with different fuel consumption (fuel cost per 100 km), then the extra cost of a long trip.

(a) the rate for the first option, to the nearest cent (2 marks);
(b) the rate for the second option, to the nearest cent (2 marks);
(c) the difference in cost over the stated quantity, to the nearest cent (2 marks, follow-through from (a), (b)).

Built backwards from the rates, in whole cents, so every pack price and every answer is exact to the cent. The
prices and fuel consumption are invented for the question.
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
_FOODS = ("coffee beans", "mixed nuts", "cheddar cheese", "dried apricots", "olives", "loose-leaf tea")
_PACKS = (250, 400, 500, 600, 750, 800, 1000, 1200)
_SHOP_STEMS = (
    "{name} is comparing two brands of {food} at a supermarket in Newcastle. Brand A sells {ga} g for ${ca}. "
    "Brand B sells {gb} g for ${cb}.",
    "At a shop in Fremantle there are two brands of {food}. A {ga} g pack of Brand A costs ${ca} and a {gb} g "
    "pack of Brand B costs ${cb}.",
)
_FUEL_STEMS = (
    "{name} is choosing between two cars for a trip around Tasmania. Car A uses {fa} L of fuel per 100 km and "
    "Car B uses {fb} L per 100 km. Fuel costs ${p} per litre.",
    "Two hire cars are available in Cairns. Car A uses {fa} L per 100 km and Car B uses {fb} L per 100 km. {name} "
    "will pay ${p} for each litre of fuel.",
)


def _money(v: Fraction) -> str:
    return stated_precision_string(v, 2)


def _draw_shop(rng):
    pa, pb = rng.randrange(90, 481, 2), rng.randrange(90, 481, 2)  # cents per 100 g
    ga, gb = rng.choice(_PACKS), rng.choice(_PACKS)
    ca, cb = Fraction(pa * ga, 10000), Fraction(pb * gb, 10000)  # dollars
    return ("shop", Fraction(pa, 100), Fraction(pb, 100), ga, gb, ca, cb, rng.randint(2, 9))


def _draw_fuel(rng):
    fa, fb = Fraction(rng.randint(48, 142), 10), Fraction(rng.randint(48, 142), 10)
    p = Fraction(rng.randrange(150, 241, 10), 100)
    return ("fuel", fa * p, fb * p, fa, fb, p, None, rng.randint(2, 12) * 100)


def _draw(rng):
    return _draw_shop(rng) if rng.random() < 0.5 else _draw_fuel(rng)


def _acceptable(d) -> bool:
    ra, rb = d[1], d[2]
    if abs(ra - rb) < Fraction(35, 100) or min(ra, rb) < Fraction(1, 2):
        return False
    if d[0] == "shop":
        return d[5].denominator in (1, 2, 4, 5, 10, 20, 25, 50, 100) and d[6].denominator in (
            1,
            2,
            4,
            5,
            10,
            20,
            25,
            50,
            100,
        )
    return True


def _part(prompt, key, marks, working, cands, follow_through=None):
    cands = [(round_half_up(v, 2), tag) for v, tag in cands]
    check_distinct(key, *[v for v, _ in cands], msg="a distractor equals the answer")
    text = _money(key)
    shown = [Distractor(_money(v), tag, v) if tag else _money(v) for v, tag in cands]
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


@generator("exd_x_n2_rates_unit_cost", level=10, presets={"exam": dict(tier="exam")})
def exd_x_n2_rates_unit_cost(rng, tier="exam"):
    """Two rates, then the difference over a larger quantity; see the module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    kind, ra, rb, x1, x2, x3, x4, qty = draw_until(rng, _draw, _acceptable, what="two clearly different rates")
    name = rng.choice(NAMES)
    if kind == "shop":
        ga, gb, ca, cb, n_kg = x1, x2, x3, x4, qty
        stem = phrase(rng, _SHOP_STEMS, name=name, food=rng.choice(_FOODS), ga=ga, gb=gb, ca=_money(ca), cb=_money(cb))
        check(
            ca * 100 / ga * 100 == ra * 100 and cb * 100 / gb * 100 == rb * 100, "pack prices disagree with the rates"
        )
        units = Fraction(n_kg * 10)  # 100 g units
        a_prompt = "Find the price of Brand A per 100 g, to the nearest cent."
        b_prompt = "Find the price of Brand B per 100 g, to the nearest cent."
        c_prompt = (
            f"{name} needs {n_kg} kg. How much would be saved by buying the cheaper brand instead of the dearer "
            "one? Give the answer to the nearest cent."
        )
        a_work = [f"Price per 100 g = {_money(ca)} ÷ {ga} × 100 = ${_money(ra)}."]
        b_work = [f"Price per 100 g = {_money(cb)} ÷ {gb} × 100 = ${_money(rb)}."]
        c_work = [f"{n_kg} kg = {n_kg * 10} lots of 100 g; saving = |{_money(ra)} - {_money(rb)}| × {n_kg * 10}."]
        a_cands = [
            (Fraction(ga) / ca, "inverted-rate"),
            (ca / ga * 1000, "rate-for-wrong-unit"),
            (ca, "raw-total-used-as-rate"),
        ]
        b_cands = [
            (Fraction(gb) / cb, "inverted-rate"),
            (cb / gb * 1000, "rate-for-wrong-unit"),
            (cb, "raw-total-used-as-rate"),
        ]
        scale = units
    else:
        fa, fb, p, d_km = x1, x2, x3, qty
        stem = phrase(rng, _FUEL_STEMS, name=name, fa=float(fa), fb=float(fb), p=_money(p))
        units = Fraction(d_km, 100)
        a_prompt = "Find the fuel cost for Car A to travel 100 km, to the nearest cent."
        b_prompt = "Find the fuel cost for Car B to travel 100 km, to the nearest cent."
        c_prompt = (
            f"{name} plans a trip of {d_km} km. How much less will the fuel cost in the car that uses less fuel "
            "than in the other car? Give the answer to the nearest cent."
        )
        a_work = [f"Cost per 100 km = {float(fa)} L × ${_money(p)} = ${_money(ra)}."]
        b_work = [f"Cost per 100 km = {float(fb)} L × ${_money(p)} = ${_money(rb)}."]
        c_work = [f"{d_km} km = {d_km // 100} lots of 100 km; saving = |{_money(ra)} - {_money(rb)}| × {d_km // 100}."]
        a_cands = [
            (fa / p, "inverted-rate"),
            (ra / 100, "unit-rate-not-scaled"),
            (fa, "raw-quantity-used-as-rate"),
        ]
        b_cands = [
            (fb / p, "inverted-rate"),
            (rb / 100, "unit-rate-not-scaled"),
            (fb, "raw-quantity-used-as-rate"),
        ]
        scale = units
    saving = abs(ra - rb) * scale
    c_cands = [
        ((ra + rb) * scale, "rates-added-not-subtracted"),
        (abs(ra - rb), "difference-not-scaled-to-quantity"),
        (max(ra, rb) * scale, "total-of-dearer-not-difference"),
    ]
    ft = f"Abs(part_a - part_b)*{scale.numerator}" if scale.denominator == 1 else f"Abs(part_a - part_b)*{scale}"
    check(saving > 0 and saving.denominator in (1, 2, 4, 5, 10, 20, 25, 50, 100), "saving is not exact to the cent")
    parts = [
        _part(a_prompt, ra, 2, a_work, a_cands),
        _part(b_prompt, rb, 2, b_work, b_cands),
        _part(c_prompt, saving, 2, c_work, c_cands, follow_through=ft),
    ]
    return dict(text=stem, parts=parts)
