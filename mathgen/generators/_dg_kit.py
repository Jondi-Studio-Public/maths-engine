"""Small helpers shared by the `dg_*` data-display and graph builds: whole-or-half-number answers with tagged
distractors, bar-height replies, and a part builder. Nothing here draws random numbers or registers a topic."""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

from ..answers import Distractor
from ..grading import GRADERS
from ..grading.interactive import bars_answer, fraction_json
from ._spread_kit import REQUIRED, check_bounds, mixed, round_extras, round_step

NEAR = "dg-near-miss-value"  # the plain near miss that fills a side of the options

__all__ = ["bars_fields", "bars_dist", "bars_text", "numd", "numf", "numtext", "set_fields", "p_number", "p_bars"]


def numtext(value) -> str:
    """A whole number, or a half-step number as a decimal: ``7`` or ``7.5`` (never ``15/2``)."""
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    if value.denominator == 2:
        return f"{float(value):.1f}"
    raise ValueError(f"{value} is not a whole or half number")


def _py(value):
    value = Fraction(value)
    return value.numerator if value.denominator == 1 else float(value)


def numf(value) -> dict:
    """``answer``, ``answer_text`` and ``answer_latex`` for a whole or half-step number."""
    return dict(answer=_py(value), answer_text=numtext(value), answer_latex=numtext(value))


def numd(
    answer, candidates, n: int = 3, *, lo=REQUIRED, hi=REQUIRED, unbounded=None, nonzero=False, step=None
) -> list[Distractor]:
    """Tagged distractors for a whole or half-step number answer. ``candidates`` is ``[(value, tag), ...]``; a value
    equal to the answer, repeated, outside ``[lo, hi]`` or not a whole or half number is skipped. Every option has
    the same number of decimal places as the answer (``7.0`` stands for 7 beside an answer of 7.5)."""
    check_bounds(lo, hi, unbounded)
    answer = Fraction(answer)
    seen = {answer}
    out: list[Distractor] = []
    # The picks below take the first usable candidates; mix the order so the key has no fixed rank among the options.
    step = (
        Fraction(step)
        if step is not None  # a reading off an axis in unit steps: a neighbour is one unit away, not a gridline
        else Fraction(round_step(answer, always=True))
        if answer.denominator == 1
        else Fraction(1, 2)
    )
    # plain near misses stay positive unless the caller allows less (`lo=0` lets a difference or a count be 0)
    floor = 0 if lo is not None and lo <= 0 else None
    # a signed answer (a difference of two readings) has signed neighbours, on both sides of 0
    signed = (lo is not None and lo < 0) or (lo is None and answer < 0)
    near = [
        (answer + d * step, NEAR)
        for d in (-3, -2, -1, 1, 2, 3)
        if signed or (answer > 0 and (answer + d * step > 0 or (floor is not None and answer + d * step >= floor)))
    ]
    near += [(v, t) for v, t in round_extras(answer, lambda v: (Fraction(v), NEAR)) if answer.denominator == 1]

    def usable(value):
        value = Fraction(value)
        if value.denominator not in (1, 2) or (answer.denominator == 1 and value.denominator != 1):
            return False  # a half beside a whole-number answer would give the answer away by its shape
        return value != answer and not (nonzero and value == 0)

    # only options that will be shown take part in the draw, so the key's rank is the one the draw gave it
    candidates = mixed(
        answer,
        [(v, t) for v, t in candidates if v is not None and usable(v)],
        fillers=[f for f in near if usable(f[0])],
        lo=lo,
        hi=hi,
        unbounded=unbounded,
    )
    for value, tag in candidates:
        value = Fraction(value)
        if (
            value in seen
            or value.denominator not in (1, 2)
            or (lo is not None and value < lo)
            or (hi is not None and value > hi)
        ):
            continue
        seen.add(value)
        text = numtext(value) if answer.denominator == 1 and value.denominator == 1 else f"{float(value):.1f}"
        if answer.denominator == 1 and value.denominator != 1:
            continue  # a half beside a whole-number answer would give the answer away by its shape
        out.append(Distractor(text, tag, _py(value)))
        if len(out) == n:
            break
    return out


# -- bar heights (the `bars` answer) ---------------------------------------------------------------------------------

TOP = 40


def bars_text(categories, heights) -> str:
    return ", ".join(f"{c} = {numtext(h)}" for c, h in zip(categories, heights))


def bars_fields(categories, heights, snap=1) -> dict:
    text = bars_text(categories, heights)
    return dict(
        answer=bars_answer(categories, [Fraction(h) for h in heights], top=TOP, snap=snap),
        answer_text=text,
        answer_latex="plain:" + text,
    )


def bars_dist(categories, heights, candidates, snap=1, n: int = 3) -> list[Distractor]:
    """Tagged distractors for a bars answer: ``candidates`` is ``[(wrong heights, tag), ...]``. Kept when the real
    grader marks the typed text wrong for the answer and right for the heights it stands for."""
    key = bars_fields(categories, heights, snap)
    probe = SimpleNamespace(answer=key["answer"], answer_text=key["answer_text"], answer_kind="bars", parts=None)
    seen = {key["answer_text"]}
    out: list[Distractor] = []
    for wrong, tag in candidates:
        if wrong is None or any(Fraction(h) < 0 or Fraction(h) > TOP for h in wrong):
            continue
        text = bars_text(categories, wrong)
        if text in seen:
            continue
        stands = bars_fields(categories, wrong, snap)
        if GRADERS["bars"](probe, text).correct:
            continue
        if not GRADERS["bars"](
            SimpleNamespace(answer=stands["answer"], answer_text=text, answer_kind="bars", parts=None), text
        ).correct:
            continue
        seen.add(text)
        out.append(Distractor(text, tag, stands["answer"]))
        if len(out) == n:
            break
    return out


def set_fields(items) -> dict:
    items = sorted(items)
    text = "{" + ", ".join(str(i) for i in items) + "}"
    return dict(answer=[str(i) for i in items], answer_text=text, answer_latex="plain:" + text)


def p_number(prompt, value, marks, working, distractors, **extra) -> dict:
    return dict(
        prompt=prompt,
        answer_kind="number",
        marks=marks,
        working=list(working),
        distractors=distractors,
        **numf(value),
        **extra,
    )


def p_bars(prompt, categories, heights, marks, working, distractors, snap=1, **extra) -> dict:
    return dict(
        prompt=prompt,
        answer_kind="bars",
        marks=marks,
        working=list(working),
        distractors=distractors,
        **bars_fields(categories, heights, snap),
        **extra,
    )


__all__ += ["fraction_json", "TOP"]
