"""Order-of-magnitude options whose correct answer can sit at any value rank.

The estimate-a-quantity MCQs used to offer the key with its x10, /10 and x100 scalings, so the key was always the
second-smallest of four. `scaled_options` instead draws the key's rank (1st, 2nd or 3rd smallest) uniformly and fills
the other slots with powers of ten above and below it. Not a generator.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, check_pairwise_distinct, fixed_point_string, pick_distractors

_DOWN = (Fraction(1, 10), Fraction(1, 100))
_UP = (Fraction(10), Fraction(100), Fraction(1000))


def _places_needed(value: Fraction) -> int:
    places = 0
    while (value * (10**places)).denominator != 1:
        places += 1
    return places


def scaled_options(rng, correct: Fraction, suffix: str) -> tuple[str, list[str]]:
    """Return ``(correct_text, distractors)`` with the key at a random rank among the four values.

    ``suffix`` is appended to each number verbatim (for example ``" kg"``). All four values are printed to the same
    number of decimal places so the digit count never marks the key.
    """
    below = rng.choice((0, 1, 2))  # how many options sit below the key
    down = list(_DOWN)
    rng.shuffle(down)
    up = list(_UP)
    rng.shuffle(up)
    factors = down[:below] + up[: 3 - below]
    values = [correct * f for f in factors]
    check_pairwise_distinct(correct, *values, msg="an order-of-magnitude distractor collided with another option")

    places = _places_needed(min([correct, *values]))

    def fmt(value: Fraction) -> str:
        scaled = value * (10**places)
        check(scaled.denominator == 1, f"{value} does not terminate exactly within {places} places")
        return f"{fixed_point_string(int(scaled), places)}{suffix}"

    correct_text = fmt(correct)
    texts = [fmt(v) for v in values]
    distractors = pick_distractors(correct_text, texts)
    check(len(distractors) == 3, "an order-of-magnitude distractor collapsed onto another option")
    return correct_text, distractors
