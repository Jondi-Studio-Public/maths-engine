"""Small helpers shared by the `dg_*` data-display and graph builds: whole-or-half-number answers with tagged
distractors, bar-height replies, and a part builder. Nothing here draws random numbers or registers a topic."""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

from ..answers import Distractor
from ..grading import GRADERS
from ..grading.interactive import bars_answer, fraction_json
from ._spread_kit import mixed

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


def numd(answer, candidates, n: int = 3, *, lo=None, hi=None) -> list[Distractor]:
    """Tagged distractors for a whole or half-step number answer. ``candidates`` is ``[(value, tag), ...]``; a value
    equal to the answer, repeated, outside ``[lo, hi]`` or not a whole or half number is skipped. Every option has
    the same number of decimal places as the answer (``7.0`` stands for 7 beside an answer of 7.5)."""
    answer = Fraction(answer)
    seen = {answer}
    out: list[Distractor] = []
    # The picks below take the first usable candidates; mix the order so the key has no fixed rank among the options.
    step = Fraction(1) if answer.denominator == 1 else Fraction(1, 2)
    near = [(answer + d * step, NEAR) for d in (-3, -2, -1, 1, 2, 3) if answer > 0 and answer + d * step > 0]
    candidates = mixed(answer, [(v, t) for v, t in candidates if v is not None], fillers=near)
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
