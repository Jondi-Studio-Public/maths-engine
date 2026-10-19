"""Shared builders for the consumer-finance skills (`r3e_*`): tax, payslips, super, HELP debts, GST, cards, BNPL, bills.

Every rate, threshold and scale in these skills is STATED in the question (a short table or a sentence) and the
question says to use it: nothing here is a fact about any particular year. The tables are illustrative scales drawn
from small sets of values, so no question can be answered by remembering a real schedule.

* `single_number` builds a one-answer question; `number_part` builds one part of a multi-part exam chain. Both write
  the key at a fixed number of decimal places and keep every distractor at that same number of places.
* A slip (a wrong value with the misconception id that produces it) is dropped when it repeats the key, repeats an
  earlier slip, lies within the grader's tolerance of the key, or is accepted by the real grader for the part.
* `Banded` is a stepped-rate scale (tax brackets, HELP repayment bands): the rate on each slice of income above a
  lower threshold. It evaluates exactly, tells which band an amount falls in, and writes the matching `Min`/`Max`
  follow-through expression for an exam chain.
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


PEOPLE = [
    ("Mei", "a primary school teacher"),
    ("Tom", "an electrician"),
    ("Aisha", "a paramedic"),
    ("Liam", "a bus driver"),
    ("Chloe", "a physiotherapist"),
    ("Ben", "a surveyor"),
    ("Priya", "a pharmacist"),
    ("Jack", "a diesel mechanic"),
    ("Sophie", "an accountant"),
    ("Daniel", "a civil engineer"),
    ("Nina", "a dental assistant"),
    ("Hamid", "a warehouse supervisor"),
]


def number_text(value, places: int) -> str:
    """`value` rounded half up to `places` places and written at exactly that many places."""
    rounded = round_half_up(Fraction(value), places)
    return stated_precision_string(rounded, places) if places else str(int(rounded))


def dollars(value, places: int | None = None) -> str:
    """A dollar amount for a question stem: `$12,500` when whole, `$12,500.40` otherwise (or at `places`)."""
    value = Fraction(value)
    if places is None:
        places = 0 if value.denominator == 1 else 2
    text = number_text(value, places)
    whole, _, cents = text.partition(".")
    sign = "-" if whole.startswith("-") else ""
    whole = f"{int(whole.lstrip('-')):,}"
    return f"{sign}${whole}" + (f".{cents}" if cents else "")


def percent(value) -> str:
    """A percentage without trailing zeros: 16, 32.5, 0.05."""
    value = Fraction(value)
    if value.denominator == 1:
        return str(int(value))
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text


def dec(value) -> str:
    """A plain decimal for working lines: 1.08, 0.85, 0.0004 (never a fraction like 27/25)."""
    value = Fraction(value)
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text


def _accepted(kind: str, prompt: str, answer, answer_text: str, reply: str) -> bool:
    probe = SimpleNamespace(
        answer=answer, answer_text=answer_text, text=prompt, answer_alt=None, answer_form=None, answer_kind=kind
    )
    return bool(GRADERS[kind](probe, reply).correct)


def _options(prompt, value, places, slips, keep):
    value = Fraction(value)
    text = number_text(value, places)
    answer = numeric_answer(round_half_up(value, places))
    options: list[Distractor] = []
    seen = {text}
    for slip, tag in slips:
        slip = Fraction(slip)
        shown = number_text(slip, places)
        if shown in seen or abs(Fraction(shown) - Fraction(text)) <= Fraction(GRADER_TOLERANCE).limit_denominator(1000):
            continue
        if _accepted("number", prompt, answer, text, shown):
            continue
        seen.add(shown)
        options.append(Distractor(shown, tag, exact_value(Fraction(shown))))
        if len(options) == keep:
            break
    check(len(options) >= 2, f"fewer than two distractors for {prompt!r}: {[s[1] for s in slips]}")
    return text, answer, options


def single_number(text, value, places, working, slips, *, keep=3) -> dict:
    """A one-answer number question dict."""
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


def listing(rows: list[tuple[str, str]]) -> str:
    """Rows written as one sentence: `label: value; label: value.` (question text has no line breaks)."""
    return "; ".join(f"{label}: {value}" for label, value in rows) + "."


class Banded:
    """A stepped-rate scale: `bands` is a list of `(lower threshold, rate in percent)`, ascending, first threshold 0.

    The rate on a slice of income applies only to that slice (as with income tax brackets).
    """

    def __init__(self, bands):
        bands = [(Fraction(t), Fraction(r)) for t, r in bands]
        check(bands[0][0] == 0 and all(a[0] < b[0] for a, b in zip(bands, bands[1:])), "bands must ascend from 0")
        self.bands = bands

    def amount(self, income) -> Fraction:
        income = Fraction(income)
        total = Fraction(0)
        for i, (low, rate) in enumerate(self.bands):
            high = self.bands[i + 1][0] if i + 1 < len(self.bands) else None
            if income <= low:
                break
            top = income if high is None or income < high else high
            total += (top - low) * rate / 100
        return total

    def band_index(self, income) -> int:
        """The band whose slice contains `income` (a threshold itself belongs to the band below it)."""
        index = 0
        for i, (low, _rate) in enumerate(self.bands):
            if Fraction(income) > low:
                index = i
        return index

    def marginal_rate(self, income) -> Fraction:
        return self.bands[self.band_index(income)][1]

    def slice_amounts(self, income) -> list[Fraction]:
        """The amount charged on each band's slice of `income`."""
        income = Fraction(income)
        out = []
        for i, (low, rate) in enumerate(self.bands):
            high = self.bands[i + 1][0] if i + 1 < len(self.bands) else None
            top = income if high is None or income < high else high
            out.append(max(Fraction(0), top - low) * rate / 100)
        return out

    def expression(self, name: str) -> str:
        """A SymPy string for the amount as a function of the earlier part `name` (e.g. `part_a`).

        The amount is the sum, over the bands, of the CHANGE in rate at each threshold times the income above it:
        one term per taxed band, using max(x, 0) = (x + |x|) / 2 because `Abs` parses several times faster than `Max`.
        """
        terms = []
        previous = Fraction(0)
        for low, rate in self.bands:
            step = rate - previous
            previous = rate
            if step == 0:
                continue
            if low:
                coefficient = f"{step.numerator}/{step.denominator * 200}"
                terms.append(f"{coefficient}*({name} - {int(low)} + Abs({name} - {int(low)}))")
            else:
                terms.append(f"{step.numerator}/{step.denominator * 100}*{name}")
        return " + ".join(terms)

    def table_rows(self, label=lambda low, high: "") -> list[tuple[str, str]]:
        rows = []
        for i, (low, rate) in enumerate(self.bands):
            high = self.bands[i + 1][0] if i + 1 < len(self.bands) else None
            if high is None:
                span = f"{dollars(low + 1)} and over" if low else "Any amount"
            elif low == 0:
                span = f"{dollars(0)} to {dollars(high)}"
            else:
                span = f"{dollars(low + 1)} to {dollars(high)}"
            rows.append((span, "Nil" if rate == 0 else f"{percent(rate)} cents for each $1 over {dollars(low)}"))
        return rows
