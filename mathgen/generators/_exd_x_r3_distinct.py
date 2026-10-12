"""One rule for the r3 exam chains: no two parts of a question share an answer.

A learner who types one number into every part must never earn a later part's marks by luck, and a part whose answer
repeats an earlier part's is a free mark for anyone who guessed the earlier one. `distinct_answers` wraps a generator
and draws again (the same `rng`, so a seed is still reproducible) until every part's answer differs from every other's
as a number, or as text when the answer is not a number.
"""

from __future__ import annotations

import functools
from fractions import Fraction

from ..core import attempts


def answer_key(part):
    """The part's answer as a value two parts can be compared by: a Fraction when its text reads as a number."""
    text = part["answer_text"] if isinstance(part, dict) else part.answer_text
    cleaned = str(text).strip().replace(" ", "")
    try:
        return Fraction(cleaned)
    except (ValueError, ZeroDivisionError):
        return cleaned


def distinct_answers(fn):
    """Decorator for a generator function `fn(rng, ...)` that returns a dict with `parts`."""

    @functools.wraps(fn)
    def wrapper(rng, *args, **kwargs):
        for _ in attempts(200, "a draw whose parts have distinct answers"):
            drawn = fn(rng, *args, **kwargs)
            keys = [answer_key(p) for p in drawn["parts"]]
            if len(set(keys)) == len(keys):
                return drawn
        raise AssertionError("unreachable")  # pragma: no cover - attempts raises first

    return wrapper
