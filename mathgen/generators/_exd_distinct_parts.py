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


class DrawRejected(VerificationError):
    """Raised inside a generator wrapped by `distinct_part_answers` to throw the current draw away, as an unequal
    or junk-unsafe draw is: the wrapper runs the generator again on the same, already consumed, stream. Use it for
    a draw that is correct but cannot be asked well, such as one whose slips leave a part with fewer than two
    distractors. Unwrapped, it is an ordinary `VerificationError`."""


def answer_key(part) -> object:
    text = str(part["answer_text"] if isinstance(part, dict) else part.answer_text).strip()
    try:
        return Fraction(text.replace(" ", ""))
    except (ValueError, ZeroDivisionError):
        return text.replace(" ", "")


def answers_distinct(parts) -> bool:
    keys = [answer_key(p) for p in parts]
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
    names = [h.label for h in holders]
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
            if not h.follow_through or _right(replies[j], canon[j]) or kinds[j] not in VALUE_KINDS:
                continue
            if any(
                kinds[names.index(n)] == "ordered_list"
                and (not hasattr(values.get(n), "args") or len(values[n].args) != len(parts[names.index(n)]["answer"]))
                for n in named_parts(h)
            ):
                continue  # the grader carries a list reply only at the length of the list asked for
            carried = follow_through_value(h, {k: v for k, v in values.items() if k < h.label})
            reply = values.get(h.label)
            if carried is None or reply is None or not (carried.is_real and reply.is_real):
                continue
            # The grader never carries a result that is just an earlier reply typed again, so that case is safe.
            used = [values.get(name) for name in named_parts(h)]
            if not _would_carry(parts, h, values, replies, canon):
                continue
            try:
                if any(u is not None and u.is_real and abs(float(u) - float(carried)) < 1e-9 for u in used):
                    continue
                # A typed number is marked to a tolerance; a fraction or an expression only when it is exact.
                if abs(float(carried) - float(reply)) <= (_NEAR if kinds[j] == "number" else 1e-9):
                    return False
            except (TypeError, ValueError):
                continue
    return True


def _right(reply: str, canonical: str) -> bool:
    """Whether a reply is the canonical answer, by value where both read as numbers: "1" is right for "1.00"."""
    return answer_key({"answer_text": reply}) == answer_key({"answer_text": canonical})


def _would_carry(parts, holder, values, replies, canon) -> bool:
    """Whether the grader would carry these replies into `holder` at all: it carries nothing from a wrong reply
    outside its part's `sensible_range` or more than `_CARRY_FACTOR` from its canonical answer either way."""
    from mathgen.grading.parts import VALUE_KINDS, _in_range, _plausible_size, named_parts, reply_value

    for name in named_parts(holder):
        i = ord(name) - 97
        if _right(replies[i], canon[i]) or values.get(name) is None:
            continue
        source = parts[i]
        if not _in_range(source.get("sensible_range"), values[name]):
            return False
        if source["answer_kind"] in VALUE_KINDS and not _plausible_size(
            values[name], reply_value(source["answer_kind"], canon[i])
        ):
            return False
    return True


def distinct_part_answers(fn):
    """Re-run the generator `fn` until the parts of the question it builds have pairwise different answers."""

    @functools.wraps(fn)
    def wrapper(rng, *args, **kwargs):
        for _ in range(_TRIES):
            try:
                built = fn(rng, *args, **kwargs)
            except DrawRejected:
                continue
            if answers_distinct(built["parts"]) and junk_safe(built["parts"]):
                return built
        raise VerificationError(
            f"{fn.__name__}: no draw with pairwise different, junk-proof part answers in {_TRIES} tries"
        )

    return wrapper
