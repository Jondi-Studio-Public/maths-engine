"""Shared guard for the round-3 exam chains: no two parts of one question may have the same answer.

A learner who has just typed 21 for (b) should not find 21 again as the key for (c): the repeat is a free mark for a
guess, and it lets a chain be answered by copying one part into the next. `distinct_part_answers` wraps a
multi-part generator so that a draw whose parts share an answer (compared by value for numbers, by text otherwise) is
thrown away and the generator is run again on the same random stream. The stream is consumed, never reseeded, so
every seed still gives the same question every time.
"""

from __future__ import annotations

import functools
from fractions import Fraction

from mathgen.core import VerificationError

_TRIES = 400


def _key(part) -> object:
    text = str(part["answer_text"] if isinstance(part, dict) else part.answer_text).strip()
    try:
        return Fraction(text.replace(" ", ""))
    except (ValueError, ZeroDivisionError):
        return text.replace(" ", "")


def answers_distinct(parts) -> bool:
    keys = [_key(p) for p in parts]
    return len(set(keys)) == len(keys)


def distinct_part_answers(fn):
    """Re-run the generator `fn` until the parts of the question it builds have pairwise different answers."""

    @functools.wraps(fn)
    def wrapper(rng, *args, **kwargs):
        for _ in range(_TRIES):
            built = fn(rng, *args, **kwargs)
            if answers_distinct(built["parts"]):
                return built
        raise VerificationError(f"{fn.__name__}: no draw with pairwise different part answers in {_TRIES} tries")

    return wrapper
