"""Exam source: three successive percentage changes, the overall change, and the change that undoes it.

One scenario: an item priced at $P is changed three times by whole-number percentages (a mix of rises and falls
ending below the starting price).

(a) the final price (compounding, not adding), to the nearest cent;
(b) the overall percentage decrease, from (a);
(c) the percentage increase that would take the final price back to the original price, from (a): the base is the
    reduced price, so the answer is larger than (b).

Money is exact `Fraction` arithmetic rounded half up once at display. A draw is kept only when the route through
the rounded price in (a) and the exact route agree at the displayed precision of (b) and (c).
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import attempts, generator, phrase, round_half_up
from mathgen.generators.exd_x_n1_support import clear_of_boundary, money, number_part, show

# (item, lowest price, highest price, step): each item is priced inside a range that is sensible for it.
_ITEMS = (
    ("a games console", 400, 900, 50),
    ("a mountain bike", 600, 2400, 50),
    ("a second-hand ute", 6000, 14000, 500),
    ("a flat-screen television", 500, 2500, 50),
    ("a surfboard", 400, 1000, 50),
    ("a laptop", 800, 2500, 50),
)
_STEM = (
    "The price of {item} starts at ${p}. It is then changed three times: {c1}, then {c2}, then {c3}.",
    "{Item} is advertised at ${p}. Over three months the price is changed in turn: {c1}, then {c2}, then {c3}.",
)


def _change(pct: int) -> str:
    return f"increased by {pct}%" if pct > 0 else f"reduced by {-pct}%"


@generator("exd_x_n1_repeated_change_and_reverse", level=11, presets={"exam": dict(tier="exam")})
def exd_x_n1_repeated_change_and_reverse(rng, tier="exam"):
    item, low, high, step = rng.choice(_ITEMS)
    for _ in attempts(1000, "a change sequence with a net fall and stable rounding"):
        p = Fraction(rng.randrange(low, high, step))
        changes = [rng.choice((-1, 1)) * rng.randint(4, 30) for _ in range(3)]
        if len({abs(c) for c in changes}) < 3 or all(c < 0 for c in changes) or all(c > 0 for c in changes):
            continue
        factor = Fraction(1)
        for c in changes:
            factor *= 1 + Fraction(c, 100)
        final = p * factor
        if not (Fraction(65, 100) <= factor <= Fraction(95, 100)):
            continue
        a_shown = round_half_up(final, 2)
        if not clear_of_boundary(final, 2):
            continue
        dec = (1 - factor) * 100
        inc = (1 / factor - 1) * 100
        dec_from_a = (p - a_shown) / p * 100
        inc_from_a = (p - a_shown) / a_shown * 100
        if show(dec, 1) != show(dec_from_a, 1) or show(inc, 1) != show(inc_from_a, 1):
            continue
        if not (clear_of_boundary(dec, 1) and clear_of_boundary(inc, 1)):
            continue
        break

    stem = phrase(
        rng,
        _STEM,
        item=item,
        Item=item[0].upper() + item[1:],
        p=money(p, 0),
        c1=_change(changes[0]),
        c2=_change(changes[1]),
        c3=_change(changes[2]),
    )
    f = Fraction
    f1, f2, f3 = (1 + f(c, 100) for c in changes)
    summed = p * (1 + f(sum(changes), 100))
    part_a = number_part(
        "Find the final price, correct to the nearest cent.",
        final,
        2,
        2,
        [
            "Multipliers: " + ", ".join(f"{float(m):g}" for m in (f1, f2, f3)) + ".",
            f"Final price = {money(p, 0)} × {float(f1):g} × {float(f2):g} × {float(f3):g} = ${show(final, 2)}.",
        ],
        [
            (summed, None),
            (p * f1, None),
            (p * f3, None),
            (p * f1 * f2, None),
        ],
        sensible_range=(0, None),
    )
    part_b = number_part(
        "Using your answer to (a), find the overall percentage decrease in the price, correct to 1 decimal place.",
        dec_from_a,
        1,
        1,
        [
            f"Decrease = {money(p, 0)} - {show(a_shown, 2)} = {show(p - a_shown, 2)}; ÷ {money(p, 0)} × 100 = {show(dec, 1)}%."
        ],
        [
            (abs(f(sum(changes))), None),
            ((p - a_shown) / a_shown * 100, "divides-by-final"),
            (p - a_shown, "gave-change-not-percent"),
            (a_shown / p * 100, "new-as-percent-of-original"),
        ],
        follow_through=f"({int(p)} - part_a)/{int(p)}*100",
        sensible_range=(0, 100),
    )
    part_c = number_part(
        "Using your answer to (a), find the percentage increase that would take the final price back to the "
        "original price, correct to 1 decimal place.",
        inc_from_a,
        1,
        2,
        [f"Increase needed = {show(p - a_shown, 2)} ÷ {show(a_shown, 2)} × 100 = {show(inc, 1)}%."],
        [
            (dec_from_a, None),
            (a_shown / p * 100, "new-as-percent-of-original"),
            (p - a_shown, "gave-change-not-percent"),
            (100 - dec_from_a, None),
        ],
        follow_through=f"({int(p)} - part_a)/part_a*100",
        sensible_range=(0, None),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c])
