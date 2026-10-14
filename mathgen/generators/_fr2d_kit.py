"""Shared plumbing for the chance-and-sampling builds (`fr2d_*`).

Not a generator (the name starts with an underscore). It holds the small pieces every module in the packet needs:
tagged number and fraction options, the properness test for a probability option, and a redraw wrapper for a draw
that cannot give three distinct wrong options.
"""

from __future__ import annotations

import functools
from fractions import Fraction

from mathgen.answers import Distractor, exact_value
from mathgen.core import attempts, rational_text

FRACTION_SUFFIX = "Give your answer as a fraction in simplest form."


def frac_d(value, tag: str) -> Distractor:
    """A fraction option: spelled in simplest form, valued exactly."""
    value = Fraction(value)
    return Distractor(rational_text(value), tag, exact_value(value))


def int_d(value, tag: str) -> Distractor:
    """A whole-number option."""
    return Distractor(str(int(value)), tag, int(value))


def is_proper(d: Distractor) -> bool:
    """True when the option is a probability strictly between 0 and 1."""
    v = d.value
    value = Fraction(v["p"], v["q"]) if isinstance(v, dict) else Fraction(v)
    return 0 < value < 1


class Short(Exception):
    """A draw that cannot give enough distinct wrong options; the generator redraws."""


def need(*lists, n: int = 3) -> None:
    if any(len(x) < n for x in lists):
        raise Short


def redraw(fn):
    """Re-run a generator body on the next draws when it raised `Short`."""

    @functools.wraps(fn)
    def wrapper(rng, *args, **kwargs):
        for _ in attempts(300, "a draw with enough distinct wrong options"):
            try:
                return fn(rng, *args, **kwargs)
            except Short:
                continue

    return wrapper
