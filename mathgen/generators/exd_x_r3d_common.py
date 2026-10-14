"""Small helpers shared by the multi-part exam generators whose module names start `exd_x_r3d_`.

Nothing here registers a topic. `tagged` turns (value, misconception id) pairs into `Distractor` objects spelled the
way the part spells its own answer, dropping any that the real grader could accept as the key.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import pick_distractors


def plain(value):
    """`value` as an int when whole, else a Fraction (a `Distractor` value must never be a float)."""
    value = Fraction(value)
    return int(value) if value.denominator == 1 else value


def int_text(value) -> str:
    return str(int(value))


def tagged(
    key_text: str,
    candidates: Iterable[tuple[object, str]],
    fmt: Callable[[object], str] = int_text,
    n: int = 3,
    tolerance=0,
) -> list[Distractor]:
    """Up to `n` tagged distractors from `(value, misconception id)` pairs, in priority order.

    A `None` value is skipped (the error does not apply to this draw). `fmt` spells a value like the key;
    `pick_distractors` drops duplicates and anything within `tolerance` of the key."""
    built = []
    for value, tag in candidates:
        if value is None:
            continue
        built.append(Distractor(fmt(value), tag, plain(value)))
    return pick_distractors(key_text, built, n, tolerance=tolerance)
