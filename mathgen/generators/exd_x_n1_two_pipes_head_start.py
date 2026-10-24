"""Exam source: two rates working together after one has a head start.

One scenario: a tank is filled by pipe A alone for a while, then pipe B is opened as well.

(a) the fraction of the tank pipe A fills during its head start (fraction);
(b) the fraction of the tank the two pipes fill together in one hour (fraction);
(c) the total time from the start until the tank is full, from (a) and (b).

The learner has to turn each "fills it in n hours" statement into a rate, add rates, and split the job into two
phases. Part (c) is graded to 1 decimal place; a draw is kept only when its exact value is clear of a rounding
boundary, and the route through the rounded learner-facing answers agrees with the exact route.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import attempts, check, generator, phrase
from mathgen.generators.exd_x_n1_support import clear_of_boundary, fraction_part, number_part, show

_STEM = (
    "A rainwater tank is filled by two pipes. Pipe A can fill the empty tank on its own in {a} hours, and pipe B "
    "can fill it on its own in {b} hours. Pipe A is turned on first and runs for {h} hours before pipe B is also "
    "turned on. Both pipes then keep running until the tank is full.",
    "A swimming-club pool is filled by two hoses. Hose A alone takes {a} hours to fill the empty pool and hose B "
    "alone takes {b} hours. Hose A is started first, and after {h} hours hose B is started as well. Both hoses "
    "stay on until the pool is full.",
    "A farm dam trough is topped up by two pumps. Pump A alone would fill the empty trough in {a} hours and "
    "pump B alone in {b} hours. Pump A starts first and runs for {h} hours before pump B joins it. Both pumps "
    "then run until the trough is full.",
)
_THING = {"tank": "pipe", "pool": "hose", "trough": "pump"}


@generator("exd_x_n1_two_pipes_head_start", level=11, presets={"exam": dict(tier="exam")})
def exd_x_n1_two_pipes_head_start(rng, tier="exam"):
    for _ in attempts(1000, "a pipe draw with a head start and a total time clear of a rounding boundary"):
        a = rng.choice((6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 24))
        b = rng.choice((4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 20))
        if a == b:
            continue
        h = rng.randint(2, a - 1)
        first = Fraction(h, a)
        together = Fraction(1, a) + Fraction(1, b)
        total = h + (1 - first) / together
        if not clear_of_boundary(total, 1):
            continue
        # learner's route: from the (exact, lowest-terms) fractions the parts ask for
        if show(total, 1) != show(h + (1 - first) * Fraction(a * b, a + b), 1):
            continue
        break

    stem_pattern = rng.choice(_STEM)
    noun = next(n for n in _THING if n in stem_pattern)
    device = _THING[noun]
    stem = phrase(rng, (stem_pattern,), a=a, b=b, h=h)
    check(f"{a} hours" in stem and f"{h} hours" in stem, "stem lost a number")

    after_b = (1 - first) / together
    part_a = fraction_part(
        f"What fraction of the {noun} has been filled by the time the second {device} is turned on?",
        first,
        1,
        [
            f"{device.capitalize()} A fills 1/{a} of the {noun} each hour, so in {h} hours it fills {h}/{a}{'' if f'{h}/{a}' == f'{first.numerator}/{first.denominator}' else f' = {first.numerator}/{first.denominator}'}."
        ],
        [
            (Fraction(1, a), None),
            (Fraction(a - h, a), None),
            (Fraction(h, a + b), None),
            (Fraction(h, b), None),
            (Fraction(h, a + h), None),
        ],
        sensible_range=(0, 1),
    )
    part_b = fraction_part(
        f"What fraction of the {noun} do the two {device}s fill together in one hour?",
        together,
        2,
        [f"Rates add: 1/{a} + 1/{b} = {together.numerator}/{together.denominator} of the {noun} per hour."],
        [(Fraction(1, a + b), None), (Fraction(1, a * b), None), (Fraction(2, a + b), None)],
        sensible_range=(0, None),
    )
    part_c = number_part(
        f"Using your answers to (a) and (b), find the total time from the start until the {noun} is full, in "
        "hours, correct to 1 decimal place.",
        total,
        1,
        3,
        [
            f"Still to fill after {h} hours: 1 - {first.numerator}/{first.denominator} = {(1 - first).numerator}/{(1 - first).denominator}.",
            f"Time together = ({(1 - first).numerator}/{(1 - first).denominator}) ÷ ({together.numerator}/{together.denominator}) = {show(after_b, 2)} hours.",
            f"Total = {h} + {show(after_b, 2)} = {show(total, 1)} hours (1 d.p.).",
        ],
        [
            (after_b, None),
            (Fraction(a * b, a + b), None),
            (h + (1 - first) * a, None),
            (h + (1 - first) / (Fraction(1, b)), None),
        ],
        follow_through=f"{h} + (1 - part_a)/part_b",
        sensible_range=(0, None),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c])
