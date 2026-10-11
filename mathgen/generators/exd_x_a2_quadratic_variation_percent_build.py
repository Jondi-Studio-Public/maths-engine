"""Exam-tier source for quadratically related variables, posed in percentage terms.

The practice tiers give a scale factor on the input and ask for the output factor, or invert a stated
proportion. The exam question never mentions a multiplier. A quantity is related to another by
``y = k x^2`` and the change is stated as a PERCENTAGE, so the learner must convert percentages to multipliers,
apply the square (or its inverse), and convert back. Three shapes:

- ``forward``: ``x`` changes by ``p`` per cent; by what percentage does ``y`` change?
- ``inverse``: ``y`` changes by ``q`` per cent; by what percentage did ``x`` change?
- ``value``: ``y`` has a stated starting value and ``x`` changes by ``p`` per cent; find the new ``y``.

Built backwards: ``p`` is always a multiple of ten, so ``(100 +- p)^2 / 100`` is an integer and every
percentage and value in the question and the key is a whole number. The output multiplier is
``(1 +- p/100)^2``, rebuilt in the checks with exact fractions as a second route.

Distractors are named slips: treating the relation as direct proportion (the same percentage), doubling
the percentage and forgetting the second-order term, reporting the new value as a percentage of the original
instead of the change, halving or square-rooting in the inverse direction.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, generator, phrase, pick_distractors

# (relation sentence, quantity y, quantity x, unit of y)
_CONTEXTS = (
    ("The area of a circle is proportional to the square of its radius.", "area", "radius", "cm^2"),
    (
        "The kinetic energy of a moving object is proportional to the square of its speed.",
        "kinetic energy",
        "speed",
        "J",
    ),
    (
        "The braking distance of a vehicle is proportional to the square of its speed.",
        "braking distance",
        "speed",
        "m",
    ),
    (
        "The power dissipated by a resistor is proportional to the square of the current through it.",
        "power",
        "current",
        "W",
    ),
)

_PERCENT_STEPS = (10, 20, 30, 40, 50)


def _scaled(y0: Fraction, sign: int, p: int) -> Fraction:
    """y0 * (1 + sign*p/100)^2, by exact arithmetic."""
    return y0 * (1 + Fraction(sign * p, 100)) ** 2


def _exam(rng: random.Random) -> dict:
    relation, y_name, x_name, unit = rng.choice(_CONTEXTS)
    shape = rng.choice(["forward", "inverse", "value"])
    up = rng.choice([True, False])
    sign = 1 if up else -1
    p = rng.choice(_PERCENT_STEPS)
    verb = "increased" if up else "decreased"
    # percentage change in y when x changes by sign*p per cent
    q = _scaled(Fraction(100), sign, p) - 100
    check(q.denominator == 1, "output percentage is not a whole number")
    q = int(q)
    q_mag = abs(q)

    if shape == "forward":
        text = phrase(
            rng,
            (
                "{rel} The {x} is {verb} by {p}%. By what percentage does the {y} change? "
                "Give the size of the change as a percentage, without the sign.",
                "{rel} If the {x} is {verb} by {p}%, find the percentage {chg} in the {y}.",
            ),
            rel=relation,
            x=x_name,
            y=y_name,
            verb=verb,
            p=p,
            chg="increase" if up else "decrease",
        )
        key = q_mag
        # direct proportion; doubling the percentage; the new value as a percentage of the original
        wrong = [p, 2 * p, 100 + q]
    elif shape == "inverse":
        chg = "increase" if up else "decrease"
        text = (
            f"{relation} The {y_name} undergoes a {q_mag}% {chg}. "
            f"By what percentage did the {x_name} {'increase' if up else 'decrease'}?"
        )
        key = p
        # same percentage (direct proportion); half the percentage; square root of the new percentage of the original
        wrong = [q_mag, q_mag // 2 if q_mag % 2 == 0 else 0, round((100 + q) ** 0.5)]
    else:
        y0 = rng.choice([100, 200, 300, 400, 500, 600, 800])
        key_f = _scaled(Fraction(y0), sign, p)
        check(key_f.denominator == 1, "new value is not a whole number")
        key = int(key_f)
        text = (
            f"{relation} The {y_name} is measured as {y0} {unit}. "
            f"The {x_name} is then {verb} by {p}%. Find the new {y_name} in {unit}."
        )
        wrong = [
            int(y0 * (1 + Fraction(sign * p, 100))),  # treated as direct proportion
            int(y0 * (1 + Fraction(sign * 2 * p, 100))),  # percentage simply doubled
            int(y0 * (1 + sign * Fraction(p * p, 100 * 100))),  # only the square of the change applied
        ]

    # independent second route via exact multipliers
    if shape == "forward":
        check(abs(_scaled(Fraction(1), sign, p) - 1) * 100 == key, "forward key disagrees with multiplier")
    elif shape == "inverse":
        check(_scaled(Fraction(1), sign, key) == 1 + Fraction(q, 100), "inverse key disagrees with multiplier")
    else:
        check(Fraction(key, y0) == (1 + Fraction(sign * p, 100)) ** 2, "value key disagrees with multiplier")

    answer_text = str(key)
    cands = [str(w) for w in wrong if w != key and w > 0]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=key,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=pick_distractors(answer_text, cands, tolerance=Fraction(51, 1000)),
    )


@generator("exd_x_a2_quadratic_variation_percent", level=11, presets={"exam": dict(tier="exam")})
def exd_x_a2_quadratic_variation_percent(rng: random.Random, tier: str = "exam") -> dict:
    """Quadratic variation posed as percentage changes, forwards, backwards, or on a stated value."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _exam(rng)
