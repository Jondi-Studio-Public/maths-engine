"""Shared guard for the exam chains with follow-through: no follow-through result may equal another option's text.

A part that follows through is marked on its own earlier replies. When the learner's earlier reply is a wrong answer
and the follow-through formula, applied correctly to it, lands on one of the part's own tagged wrong options, the
grader sees a tagged wrong answer rather than carried work, and the learner loses a mark they earned. This wrapper
re-runs a multi-part generator until, for every follow-through part, every follow-through result reached from the
canonical answers and tagged wrong options of the parts it names (and from the results of earlier follow-through
parts) differs from each of that part's own tagged wrong options.

The stream of random numbers is consumed, never reseeded, so every seed still gives the same question every time.
This module registers no topics.
"""

from __future__ import annotations

import functools
import itertools
import math
import re
from fractions import Fraction

from mathgen.core import VerificationError

_TRIES = 400
_NUMBER = re.compile(r"(?<![\w.])\d+(?![\w.])")


def _fraction(text):
    try:
        return Fraction(str(text).replace(" ", ""))
    except (ValueError, ZeroDivisionError):
        return None


def _evaluate(follow_through: str, values: dict):
    """The follow-through formula on exact fractions, or None where it cannot be evaluated."""
    source = _NUMBER.sub(lambda m: f"Fraction({m.group(0)})", follow_through)
    namespace = {
        "Fraction": Fraction,
        "factorial": lambda x: Fraction(math.factorial(int(x))) if x == int(x) and 0 <= x <= 60 else None,
        "__builtins__": {},
    }
    try:
        result = eval(source, namespace, values)  # noqa: S307 - the formula is written by the generator itself
    except (ArithmeticError, TypeError, ValueError):
        return None
    return result if isinstance(result, Fraction) else None


def collisions(parts):
    """The first (part label, replies, result) where a follow-through result lands on one of its own part's tagged
    wrong options, or None."""
    candidates: list[set] = []
    for part in parts:
        values = {_fraction(part["answer_text"])}
        values |= {_fraction(d.text) for d in part.get("distractors", ())}
        candidates.append({v for v in values if v is not None})
    for j, part in enumerate(parts):
        formula = part.get("follow_through")
        if not formula:
            continue
        names = sorted(set(re.findall(r"part_([a-z])", formula)))
        pools = [sorted(candidates[ord(n) - 97]) for n in names]
        own = {_fraction(d.text) for d in part.get("distractors", ())}
        carried = set()
        for combo in itertools.product(*pools):
            got = _evaluate(formula, {f"part_{n}": v for n, v in zip(names, combo)})
            if got is None:
                continue
            carried.add(got)
            if got in own:
                return (chr(97 + j), dict(zip(names, combo)), got)
        candidates[j] |= carried
    return None


def follow_through_clean(fn):
    """Re-run the generator `fn` until `collisions` is false for the parts it builds."""

    @functools.wraps(fn)
    def wrapper(rng, *args, **kwargs):
        last = None
        for _ in range(_TRIES):
            built = fn(rng, *args, **kwargs)
            last = collisions(built["parts"])
            if last is None:
                return built
        raise VerificationError(
            f"{fn.__name__}: no draw whose follow-through avoids its own wrong options in {_TRIES} tries; last: {last}"
        )

    return wrapper
