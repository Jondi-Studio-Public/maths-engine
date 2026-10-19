"""A guard shared by the second-year generators whose answers are exact numbers.

A multiple-choice option set in which the answer is the only integer (or the only fraction, or the only decimal) can
be answered by shape alone: with the answer `-27` among `45/2`, `189/2` and `27/2`, nobody needs to do the arithmetic.
`keep_answer_shape` wraps a generator and draws again whenever that happens.
"""

from __future__ import annotations

import re
import zlib
from functools import wraps

from ..core import attempts

_SHAPES = (
    ("integer", re.compile(r"^-?\d+$")),
    ("decimal", re.compile(r"^-?\d+\.\d+$")),
    ("fraction", re.compile(r"^-?\d+/\d+$")),
)


def option_shape(text) -> str | None:
    """`integer`, `decimal` or `fraction` for a plain number written as an option, otherwise None."""
    for name, pattern in _SHAPES:
        if pattern.match(str(text).strip()):
            return name
    return None


def answer_shape_is_unique(question: dict) -> bool:
    """Whether the answer of a generated question is the only option of its shape (numeric option sets only)."""
    options = [question.get("answer_text"), *(getattr(d, "text", d) for d in question.get("distractors") or [])]
    if len(options) < 3:
        return False
    shapes = [option_shape(o) for o in options]
    return None not in shapes and shapes.count(shapes[0]) == 1


def keep_answer_shape(make):
    """Wrap a generator `make(rng, ...)` so that it never returns a question whose answer is alone in its shape."""

    @wraps(make)
    def wrapper(rng, *args, **kwargs):
        for _ in attempts(60, "an option set in which the answer is not the only one of its shape"):
            question = make(rng, *args, **kwargs)
            if not answer_shape_is_unique(question):
                return question
        raise AssertionError("unreachable")  # pragma: no cover

    return wrapper


def spot_check(*key, one_in: int = 4) -> bool:
    """Whether the slow second-route check should run for the question identified by `key`.

    A generator that re-derives its answer with SymPy or numerical quadrature can run that check for one question in
    `one_in` instead of every one. The choice depends only on the values in `key` (never on the random stream), so
    the same question is always checked or never checked, and a sweep over many seeds still covers every code path.
    """
    return zlib.crc32(repr(key).encode()) % one_in == 0
