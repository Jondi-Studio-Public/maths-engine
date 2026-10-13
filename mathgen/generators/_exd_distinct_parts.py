"""Shared guard for the exam chains: no two parts of one question may have the same answer.

A learner who has just typed 21 for (b) should not find 21 again as the key for (c): the repeat is a free mark for a
guess, and it lets a chain be answered by copying one part into the next. `distinct_part_answers` wraps a
multi-part generator so that a draw whose parts share an answer (compared by value for numbers, by text otherwise) is
thrown away and the generator is run again on the same random stream. The stream is consumed, never reseeded, so
every seed still gives the same question every time.

The same wrapper also throws away a draw where a follow-through part would accept a junk reply: a "1", "2" or
"-1" typed into every box, or one part answered correctly and the rest "1", must not be what some part's
follow-through formula works out to (`junk_safe`).
"""

from __future__ import annotations

import functools
from fractions import Fraction
from types import SimpleNamespace

import sympy

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


_JUNK = (1, 2, -1)  # "0" is left to the grader, which never carries a zero reply into a later part
_NEAR = 0.06  # a little more than the grader's own tolerance on a typed number


def _value(kind: str, text: str):
    from mathgen.grading.parts import reply_value

    return reply_value(kind, text)


def junk_safe(parts) -> bool:
    """False when some follow-through part would take a junk reply (see the module docstring) as correct work."""
    from mathgen.grading.parts import VALUE_KINDS, follow_through_value, named_parts

    holders = [SimpleNamespace(label=chr(97 + i), follow_through=p.get("follow_through")) for i, p in enumerate(parts)]
    if not any(h.follow_through for h in holders):
        return True
    kinds = [p["answer_kind"] for p in parts]
    canon = [p["answer_text"] for p in parts]
    scenarios = [[str(v)] * len(parts) for v in _JUNK]
    for i in range(len(parts)):
        scenarios.append([canon[i] if j == i else "1" for j in range(len(parts))])
    for replies in scenarios:
        values = {}
        for h, kind, reply in zip(holders, kinds, replies):
            if kind in VALUE_KINDS:
                values[h.label] = _value(kind, reply)
        for j, h in enumerate(holders):
            if not h.follow_through or replies[j] == canon[j] or kinds[j] not in VALUE_KINDS:
                continue
            carried = follow_through_value(h, {k: v for k, v in values.items() if k < h.label})
            reply = values.get(h.label)
            if carried is None or reply is None or not (carried.is_real and reply.is_real):
                continue
            # The grader never carries a result that is just an earlier reply typed again, so that case is safe.
            used = [values.get(name) for name in named_parts(h)]
            try:
                if any(u is not None and u.is_real and abs(float(u) - float(carried)) < 1e-9 for u in used):
                    continue
                if abs(float(carried) - float(reply)) <= _NEAR:
                    return False
            except (TypeError, ValueError):
                continue
    return True


def distinct_part_answers(fn):
    """Re-run the generator `fn` until the parts of the question it builds have pairwise different answers."""

    @functools.wraps(fn)
    def wrapper(rng, *args, **kwargs):
        for _ in range(_TRIES):
            built = fn(rng, *args, **kwargs)
            if answers_distinct(built["parts"]) and junk_safe(built["parts"]):
                return built
        raise VerificationError(
            f"{fn.__name__}: no draw with pairwise different, junk-proof part answers in {_TRIES} tries"
        )

    return wrapper
