"""Shared plumbing for the senior `table` generators whose blanks are money, percentages or ratios.

`_f10_table_kit` handles tables of whole-number counts. The tables built on top of this module have blanks
that are decimal strings such as "306.00" or "4.2", and the `table` grader reads the number of decimal
places written in the canonical cell as the precision it grades to (half a unit in the last place). So every
blank is built here from an exact `Fraction`, shown at a fixed number of places, and every wrong table is
proved to be rejected by the real grader.

This module is not a generator (its name starts with an underscore, so the generators package skips it).
"""

from __future__ import annotations

import random
from fractions import Fraction
from types import SimpleNamespace

from mathgen.core import check, round_half_up, stated_precision_string
from mathgen.grading.typed import grade_table, table_answer_text

from ._f10_table_kit import build_answer, grid_text, pick_blanks, with_blanks

__all__ = [
    "blank_places",
    "build_answer",
    "cell_text",
    "finish_wrong",
    "grid_text",
    "money",
    "one_dp",
    "pick_blanks",
    "rounds_clear",
]


def cell_text(value, places: int) -> str:
    """`value` rounded half-up and shown at exactly `places` decimal places (an integer string for 0)."""
    rounded = round_half_up(Fraction(value), places)
    if places == 0:
        return str(int(rounded))
    return stated_precision_string(rounded, places)


def money(value) -> str:
    """A dollar amount to the cent, no symbol."""
    return cell_text(value, 2)


def one_dp(value) -> str:
    """A percentage or ratio to one decimal place."""
    return cell_text(value, 1)


def rounds_clear(value, places: int, margin: Fraction = Fraction(1, 10)) -> bool:
    """True when rounding `value` to `places` decimals is not close to a half-way tie.

    `margin` is a fraction of the last place: the scaled value's fractional part must be more than `margin`
    away from one half. Exact Fraction arithmetic, so a tie can never slip through as a float."""
    scaled = Fraction(value) * 10**places
    frac = scaled - (scaled.numerator // scaled.denominator)
    return abs(frac - Fraction(1, 2)) > margin


def blank_places(blank) -> int:
    """Decimal places written in a canonical blank (an int is 0)."""
    if isinstance(blank, str) and "." in blank:
        return len(blank.split(".")[1])
    return 0


def finish_wrong(
    rng: random.Random,
    answer: dict,
    options: dict[str, list[tuple[str, Fraction]]],
    want: int = 3,
    whole: list[tuple[str, dict[str, Fraction]]] | None = None,
):
    """(answer_text, distractors, option_notes) for a table question.

    `options` maps a blank key ("r,c") to the (misconception id, wrong value) pairs a learner could produce
    there; each becomes a whole table with that one blank replaced. `whole` lists (misconception id, {key: wrong
    value}) edits that change several blanks at once, for an error that runs through the whole table (the
    wrong total used for every percentage). Candidates are taken one per misconception first, whole edits
    before single ones, then any others. A candidate is dropped when a value is negative, when it leaves every
    blank equal to the right cell at that cell's own precision, or when the real `table` grader accepts it;
    every kept one is graded and must be rejected."""
    answer_text = table_answer_text(answer)
    probe = SimpleNamespace(answer=answer, answer_text=answer_text)
    check(grade_table(probe, answer_text).correct, "table grader rejects its own canonical answer")
    pool = [(mid, dict(edit)) for mid, edit in (whole or [])]
    singles = [(mid, {key: val}) for key in sorted(options) if key in answer["blanks"] for mid, val in options[key]]
    rng.shuffle(singles)
    pool += singles
    first, seen = [], set()
    for item in pool:
        if item[0] not in seen:
            seen.add(item[0])
            first.append(item)
    ordered = first + [item for item in pool if item not in first]
    distractors: list[str] = []
    notes: dict[str, str] = {}
    for mid, edit in ordered:
        if len(distractors) >= want:
            break
        changes = {}
        for key, val in edit.items():
            val = Fraction(val)
            right = answer["blanks"][key]
            places = blank_places(right)
            shown = cell_text(val, places)
            changes[key] = int(shown) if places == 0 else shown
        if any(Fraction(str(v)) < 0 for v in changes.values()):
            continue
        if all(Fraction(str(v)) == Fraction(str(answer["blanks"][k])) for k, v in changes.items()):
            continue
        text = table_answer_text(with_blanks(answer, changes))
        if text in distractors or text == answer_text:
            continue
        if grade_table(probe, text).correct:
            continue
        distractors.append(text)
        notes[text] = mid
    for text in distractors:
        check(not grade_table(probe, text).correct, "table grader accepts a deliberately wrong table")
    check(len(distractors) >= 2, "a table question needs at least two tagged wrong tables")
    return answer_text, distractors, notes
