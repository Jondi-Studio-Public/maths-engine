"""Shared builders for the data, tariff, timetable, circle and Earth skills in the `r4g_*` modules.

* `single_number` builds a one-answer question and `number_part` one part of an exam chain. Both write the key at a
  fixed number of decimal places and keep every wrong option at that same number of places.
* A slip (a wrong value plus the misconception id of the error that produces it) is dropped when it repeats the key,
  repeats an earlier slip, lies within the grader's tolerance of the key, or is accepted by the real grader.
* `hhmm` and `minutes` turn a clock time into a four-digit 24-hour number and back (times used here are 10:00 or
  later, so the four-digit form never needs a leading zero).
"""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

from ..answers import Distractor, exact_value
from ..core import (
    GRADER_TOLERANCE,
    check,
    latex_answer,
    numeric_answer,
    round_half_up,
    stated_precision_string,
)
from ..grading import GRADERS


def number_text(value, places: int) -> str:
    """`value` rounded half up to `places` places and written at exactly that many places."""
    rounded = round_half_up(Fraction(value), places)
    return stated_precision_string(rounded, places) if places else str(int(rounded))


def clear_of_tie(value, places: int) -> bool:
    """True when `value` is not within 1/1000 of a rounding boundary at `places` places."""
    scaled = Fraction(value) * 10**places
    return abs(scaled - scaled.__floor__() - Fraction(1, 2)) > Fraction(1, 100)


def hhmm(total_minutes) -> int:
    total_minutes = int(total_minutes)
    return (total_minutes // 60) * 100 + total_minutes % 60


def minutes(clock: int) -> int:
    return (clock // 100) * 60 + clock % 100


def clock_text(total_minutes) -> str:
    """12-hour-free 24-hour text for a stem: 13:05 written as `1305`."""
    return f"{hhmm(total_minutes):04d}"


def _accepted(prompt, answer, answer_text, reply) -> bool:
    probe = SimpleNamespace(
        answer=answer, answer_text=answer_text, text=prompt, answer_alt=None, answer_form=None, answer_kind="number"
    )
    return bool(GRADERS["number"](probe, reply).correct)


def _options(prompt, value, places, slips, keep):
    value = Fraction(value)
    text = number_text(value, places)
    answer = numeric_answer(round_half_up(value, places))
    options: list[Distractor] = []
    seen = {text}
    for slip, tag in slips:
        slip = Fraction(slip)
        shown = number_text(slip, places)
        if shown in seen or abs(Fraction(shown) - Fraction(text)) <= GRADER_TOLERANCE:
            continue
        if Fraction(shown) == 0 and Fraction(text) != 0:
            continue  # a bare zero is no plausible answer
        if _accepted(prompt, answer, text, shown):
            continue
        seen.add(shown)
        options.append(Distractor(shown, tag, exact_value(Fraction(shown))))
        if len(options) == keep:
            break
    check(len(options) >= 2, f"fewer than two distractors for {prompt!r}: {[s[1] for s in slips]}")
    return text, answer, options


def single_number(text, value, places, working, slips, *, keep=3) -> dict:
    shown, answer, options = _options(text, value, places, slips, keep)
    return dict(
        text=text,
        answer_kind="number",
        answer=answer,
        answer_text=shown,
        answer_latex=latex_answer(shown),
        working=working,
        distractors=options,
    )


def number_part(prompt, value, places, marks, working, slips, *, follow_through=None, sensible_range=None, keep=3):
    """One number part of an exam chain; `follow_through` is a SymPy string over `part_a`, `part_b`, ..."""
    shown, answer, options = _options(prompt, value, places, slips, keep)
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=answer,
        answer_text=shown,
        answer_latex=latex_answer(shown),
        marks=marks,
        working=working,
        distractors=options,
        sensible_range=sensible_range if sensible_range is not None else (0, None),
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


def retry(build, rng, tries: int = 60):
    """`build(rng)` again, on the stream where the last attempt left it, whenever a draw left a question or part with
    fewer than two usable distractors. Any other failed check (two routes to a key disagreeing) still raises."""
    from ..core import VerificationError

    for _ in range(tries):
        try:
            return build(rng)
        except VerificationError as err:
            if "fewer than two distractors" not in str(err):
                raise
    raise VerificationError(f"no draw with two usable distractors in {tries} tries")
