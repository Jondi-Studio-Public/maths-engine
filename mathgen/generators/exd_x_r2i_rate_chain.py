"""Exam-tier chain for `senior-interpret-the-derivative-as-the-instantaneous-rate-of`: average against instantaneous.

A quantity follows a cubic in time. Over an interval ``[p, q]`` the chain asks for

(a) the average rate of change,
(b) the instantaneous rate of change at the end of the interval,
(c) the time inside the interval at which the instantaneous rate equals the average (the mean value theorem), and
(d) by what percentage the rate at the end exceeds the average rate.

## Construction

Coefficients are drawn, then rejected unless the average rate is positive, the end rate is larger, and
``f'(t) = average`` has exactly one root strictly inside ``(p, q)`` (the other root is outside, so (c) is unique).
The root is displayed to 2 decimal places and kept clear of a rounding boundary; (d) to 1 decimal place likewise.
"""

from __future__ import annotations

import random
from fractions import Fraction

import math

from mathgen.core import attempts, check, generator, join_terms

from .exd_x_r2i_support import clear_of_boundary, number_part, shown

# (function letter, quantity, unit, time unit). Every quantity but the profit only ever grows, so its rate stays positive.
_CONTEXTS = [
    ("N", "the number of bacteria in a culture, in thousands,", "thousand per hour", "hours"),
    ("V", "the volume of water in a tank, in litres,", "litres per minute", "minutes"),
    ("S", "the number of tickets sold for a school concert,", "tickets per day", "days"),
    ("D", "the distance a cyclist has travelled, in metres,", "metres per second", "seconds"),
    ("P", "the profit of a market stall, in dollars,", "dollars per hour", "hours"),
]


def _poly_text(a: int, b: int, c: int, d: int) -> str:
    terms = [(a, "t^3"), (b, "t^2"), (c, "t"), (d, "")]
    return join_terms([(coef, var) for coef, var in terms if coef != 0])


@generator("exd_x_r2i_rate_chain", level=20, presets={"exam": {"tier": "exam"}})
def exd_x_r2i_rate_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: average rate, end rate, the time the two agree, and the percentage by which they differ."""
    check(tier == "exam", f"unknown tier {tier!r}")
    letter, quantity, unit, tunit = rng.choice(_CONTEXTS)
    for _ in attempts(5000, "a cubic whose rate equals its average rate once inside the interval"):
        a = rng.choice([1, 1, 2, 3])
        b = rng.randint(-6, 6)
        c = rng.randint(-6, 12)
        d = rng.randint(0, 20)
        p = rng.randint(0, 3)
        q = p + rng.randint(2, 5)
        avg = a * (p * p + p * q + q * q) + b * (p + q) + c
        end = 3 * a * q * q + 2 * b * q + c
        if avg <= 0 or end <= avg:
            continue
        if (letter != "P") and any(3 * a * x * x + 2 * b * x + c < 0 for x in (i / 4 for i in range(4 * (q + 2) + 1))):
            continue  # a count, volume or distance never decreases, so its rate is never negative
        disc = b * b - 3 * a * (c - avg)
        if disc <= 0:
            continue
        roots = [(-b + s * math.sqrt(disc)) / (3 * a) for s in (1, -1)]
        inside = [s for s, r in zip((1, -1), roots) if p < r < q]
        if len(inside) != 1:
            continue
        sign = inside[0]
        root = (-b + sign * math.sqrt(disc)) / (3 * a)
        root_f = Fraction(root).limit_denominator(10**9)
        pct = Fraction(100 * (end - avg), avg)
        if clear_of_boundary(root_f, 2) and clear_of_boundary(pct, 1):
            break

    # Independent checks, by evaluating the function itself rather than the derivative formulas above.
    def f(x):
        return a * x**3 + b * x**2 + c * x + d

    check(Fraction(f(q) - f(p), q - p) == avg, "average rate disagrees")
    h = 1e-6
    check(abs((f(q + h) - f(q - h)) / (2 * h) - end) < 1e-3, "end rate disagrees with a numerical slope")
    check(abs((f(root + h) - f(root - h)) / (2 * h) - avg) < 1e-3, "the chosen time does not give the average rate")

    function = _poly_text(a, b, c, d)
    stem = (
        f"{quantity.capitalize()} t {tunit} after the start is modelled by {letter}(t) = {function}, "
        f"for 0 ≤ t ≤ {q + 2}. Consider the interval from t = {p} to t = {q}."
    )

    other = (-b - sign * math.sqrt(disc)) / (3 * a)
    other_f = Fraction(other).limit_denominator(10**9)
    f_p, f_q = f(p), f(q)
    rate_p = 3 * a * p * p + 2 * b * p + c
    ft_c = f"(-({b}) + ({sign})*sqrt(({b})**2 - 3*{a}*({c} - part_a)))/(3*{a})"

    parts = [
        number_part(
            f"Find the average rate of change of {letter} over the interval, in {unit}.",
            Fraction(avg),
            0,
            2,
            [
                f"{letter}({q}) − {letter}({p}) = {f_q} − {f_p} = {f_q - f_p}.",
                f"Average rate = {f_q - f_p} ÷ {q - p} = {avg}.",
            ],
            [
                (Fraction(f_q - f_p), "change-not-rate"),
                (Fraction(f_p + f_q, 2), "averaged-endpoint-values"),
                (Fraction(rate_p + end, 2), "averaged-endpoint-rates"),
                (Fraction(f_q), "evaluated-function-not-derivative"),
            ],
            sensible_range=(0, None),
        ),
        number_part(
            f"Find the instantaneous rate of change of {letter} at t = {q}, in {unit}.",
            Fraction(end),
            0,
            2,
            [f"{letter}′(t) = {_poly_text(0, 3 * a, 2 * b, c)}.", f"{letter}′({q}) = {end}."],
            [
                (Fraction(f_q), "evaluated-function-not-derivative"),
                (Fraction(rate_p), "evaluated-at-wrong-point"),
                (Fraction(a * q * q + b * q + c), "dropped-power-rule-coefficient"),
                (Fraction(-end), "answer-sign-reversed"),
            ],
            sensible_range=(0, None),
        ),
        number_part(
            f"At what time between t = {p} and t = {q} is the instantaneous rate of change equal to the average "
            "rate of change from (a)? Give your answer to 2 decimal places.",
            root_f,
            2,
            3,
            [
                f"Solve {letter}′(t) = {avg}: {_poly_text(0, 3 * a, 2 * b, c - avg)} = 0.",
                f"The root in the interval is t = {root:.2f}.",
            ],
            [
                (Fraction(p + q, 2), "assumed-midpoint"),
                (other_f, "took-the-wrong-root"),
                (Fraction(-b, 3 * a), "left-out-the-square-root"),
            ],
            follow_through=ft_c,
            sensible_range=(0, None),
        ),
        number_part(
            "By what percentage is the instantaneous rate at the end of the interval greater than the average rate? "
            "Use your answers to (a) and (b), and give the percentage to 1 decimal place.",
            pct,
            1,
            2,
            [f"({end} − {avg}) ÷ {avg} × 100 = {shown(pct, 1)}%."],
            [
                (Fraction(100 * (end - avg), end), "divides-by-final"),
                (Fraction(100 * end, avg), "new-as-percent-of-original"),
                (Fraction(end - avg), "gave-change-not-percent"),
            ],
            follow_through="(part_b - part_a)/part_a*100",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, parts=parts)
