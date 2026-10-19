"""Shared part builders for the `exd_x_r3i_*` multi-part exam sources.

Not a generator: it registers no topic. Each exam source in that family builds its parts with these helpers so
that every part carries tagged distractors (`mathgen.answers.Distractor`) laid out like the part's own key: a
typed number part shows its distractors at the key's decimal places, and every distractor is checked to be a
different value from the key at the part's stated precision.
"""

from __future__ import annotations

from fractions import Fraction

from ..answers import Distractor
from ..core import check, latex_answer, numeric_answer, rational_text, round_half_up, stated_precision_string


def _pick(entries: list[tuple[str, str, Fraction]], key_text: str, n: int) -> list[Distractor]:
    """Up to `n` distractors with distinct texts, preferring one per misconception first."""
    chosen: list[Distractor] = []
    seen = {key_text}
    used_ids: set[str] = set()
    for second_pass in (False, True):
        for mid, text, value in entries:
            if len(chosen) == n:
                break
            if text in seen or (not second_pass and mid in used_ids):
                continue
            seen.add(text)
            used_ids.add(mid)
            chosen.append(Distractor(text, mid, value))
    return chosen


def number_part(
    prompt: str,
    key: Fraction,
    dp: int,
    marks: int,
    working: list[str],
    candidates: list[tuple[str, Fraction | None]],
    *,
    follow_through: str | None = None,
    sensible_range: tuple | None = None,
    n: int = 3,
) -> dict:
    """A typed number part. `key` is exact and already on the `dp` grid; each candidate is `(misconception id, exact
    value)` and is rounded to the same grid, so every distractor shows the key's decimal places."""
    key = Fraction(key)
    key_text = stated_precision_string(key, dp) if dp else str(int(key))
    check(key == round_half_up(key, dp), "the key is not on its stated grid")
    entries = []
    for mid, value in candidates:
        if value is None:
            continue
        value = Fraction(value)
        if dp == 0 and value.denominator != 1:
            continue  # an integer key is only ever offered whole-number distractors
        rounded = round_half_up(value, dp)
        if rounded == key:
            continue
        text = stated_precision_string(rounded, dp) if dp else str(int(rounded))
        entries.append((mid, text, rounded))
    picked = _pick(entries, key_text, n)
    check(len(picked) >= 2, "fewer than two distinct distractors for a number part")
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=numeric_answer(key),
        answer_text=key_text,
        answer_latex=latex_answer(key_text),
        marks=marks,
        working=working,
        distractors=picked,
    )
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part


def fraction_part(
    prompt: str,
    key: Fraction,
    marks: int,
    working: list[str],
    candidates: list[tuple[str, Fraction | None]],
    *,
    follow_through: str | None = None,
    sensible_range: tuple | None = None,
    text_of=rational_text,
    n: int = 3,
) -> dict:
    """A part graded by exact value. `text_of` spells a value (`p/q` by default)."""
    key = Fraction(key)
    key_text = text_of(key)
    entries = []
    for mid, value in candidates:
        if value is None:
            continue
        value = Fraction(value)
        if value == key:
            continue
        entries.append((mid, text_of(value), value))
    picked = _pick(entries, key_text, n)
    check(len(picked) >= 2, "fewer than two distinct distractors for a fraction part")
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer={"p": int(key.numerator), "q": int(key.denominator)},
        answer_text=key_text,
        answer_latex=latex_answer(key_text),
        marks=marks,
        working=working,
        distractors=picked,
    )
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part
