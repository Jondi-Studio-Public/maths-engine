"""Shared helpers for the exam-tier consumer-arithmetic generators (`exd_x_g5_*`).

This module registers no topic. It holds the three things every one of those generators needs: a way to
show an exact `Fraction` as a dollar amount in a prompt, a way to build the `number` answer fields with a
stated precision, and a distractor filter that checks each wrong value against that precision.

Answers are built backwards from exact `Fraction` arithmetic, so the key is exact. A typed numeric answer
is graded to the precision its own `answer_text` displays, so `answer_text` always shows exactly `dp`
places (`stated_precision_string`), and a wrong value is kept only when it differs from the key by more
than twice the grading tolerance at that precision.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    check,
    currency_from_cents,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)


def usd(amount) -> str:
    """A dollar amount for a prompt: `$2500` when whole, `$2512.50` otherwise. The amount must be on the cent
    grid (a construction bug otherwise)."""
    amount = Fraction(amount)
    cents = amount * 100
    check(cents.denominator == 1, f"{amount} is not a whole number of cents")
    if amount.denominator == 1:
        return f"${amount.numerator}"
    return currency_from_cents(int(cents))


def number_fields(text: str, answer: Fraction, dp: int, wrong: list, working: list[str]) -> dict:
    """The generator-return dict for a `number` answer shown at exactly `dp` decimal places.

    `answer` must already lie on the `10**-dp` grid. `wrong` is a list of `Fraction` values in
    misconception-priority order; each is rounded half-up to `dp` places for display, so a wrong value always has
    the same number of decimal places as the key (an integer key gets whole-number wrong values).
    Candidates inside twice the grading tolerance of the key, non-positive ones, and duplicates are dropped.
    """
    answer = Fraction(answer)
    shown = stated_precision_string(answer, dp)
    margin = 2 * stated_precision_tolerance(dp) if dp > 0 else Fraction(0)
    candidates = []
    for value in wrong:
        value = Fraction(value)
        if value <= 0:
            continue
        places = dp
        candidates.append(stated_precision_string(round_half_up(value, places), places))
    distractors = pick_distractors(shown, candidates, 3, tolerance=margin)
    check(len(distractors) == 3, "fewer than three usable distractors")
    return dict(
        text=text,
        answer=numeric_answer(answer),
        answer_text=shown,
        answer_latex=shown,
        working=working,
        distractors=distractors,
    )
