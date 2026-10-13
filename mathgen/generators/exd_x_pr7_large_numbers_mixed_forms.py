"""Exam tier for "order large natural numbers": the numbers arrive in different written forms.

The practice tiers show numerals to compare. Here each of five visitor counts is written in a different way: as a
numeral, as "N thousand and M", in expanded form, or in decimal millions. The learner converts every one to a
numeral (keeping placeholder zeros, as in "412 thousand and 6") and then lists the numerals from smallest to
largest. The conversion is the exam step; sorting alone would not get there.

Built backwards from five drawn values of 5 to 7 digits, each with a few non-zero digits so that zeros matter.
At least one form is always one where dropping a placeholder zero changes the order, so the placeholder
distractor differs from the key.
"""

from __future__ import annotations

import random

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, pick_distractors, reversed_and_adjacent_swap

_EVENTS = [
    "The number of visitors to five Australian national parks in one year was written down in different ways.",
    "Five museums recorded their total visitors for the year in different forms.",
    "Annual visitors to five Australian tourist attractions were reported in different ways.",
]
_ASK = [
    "Write each number of visitors as a numeral, then list the numbers from smallest to largest.",
    "Convert every visitor count to a numeral and put the numbers in order, smallest first.",
]


def _draw_value(rng: random.Random) -> int:
    """5 to 7 digits with two to four non-zero digits, so there are placeholder zeros."""
    length = rng.choice([5, 6, 6, 7])
    places = rng.sample(range(length - 1), rng.randint(1, 3))
    digits = {length - 1: rng.randint(1, 9)} | {p: rng.randint(1, 9) for p in places}
    return sum(d * 10**p for p, d in digits.items())


def _expanded(v: int) -> str | None:
    parts = [f"{int(c) * 10 ** (len(str(v)) - 1 - i):,}" for i, c in enumerate(str(v)) if c != "0"]
    return " + ".join(parts) if 2 <= len(parts) <= 4 else None


def _forms(v: int) -> dict[str, str]:
    """Every written form that applies to `v`, by name."""
    forms = {"numeral": f"{v:,}"}
    if v % 1000 and v < 1_000_000:
        forms["thousand_and"] = f"{v // 1000} thousand and {v % 1000}"
    if v % 100_000 == 0 and v >= 1_000_000:
        forms["millions"] = f"{v / 1_000_000:.1f} million"
    expanded = _expanded(v)
    if expanded:
        forms["expanded"] = expanded
    return forms


def _sloppy(form: str, shown: str, v: int) -> int:
    """The value a learner reaches by dropping placeholder zeros when converting this form."""
    if form == "thousand_and":
        return int(f"{v // 1000}{v % 1000}")
    if form == "expanded":
        return int("".join(c for c in str(v) if c != "0"))
    return v


@generator("exd_x_pr7_large_numbers_mixed_forms", level=3, presets={"exam": {"tier": "exam"}})
def exd_x_pr7_large_numbers_mixed_forms(rng: random.Random, tier: str = "exam") -> dict:
    """Convert five visitor counts written in different forms to numerals and order them. See module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(2000, "five distinct values with a form where a dropped zero changes the order"):
        values = [_draw_value(rng) for _ in range(5)]
        if len(set(values)) < 5 or len({len(str(v)) for v in values}) < 2:
            continue
        chosen: list[tuple[str, str, int]] = []
        for v in values:
            forms = _forms(v)
            name = rng.choice([n for n in forms if n != "numeral"] + ["numeral"])
            chosen.append((name, forms[name], v))
        if sum(1 for n, _, _ in chosen if n == "numeral") > 2:
            continue
        ordered = sorted(values)
        sloppy = [_sloppy(n, t, v) for n, t, v in chosen]
        if sorted(sloppy) == sloppy or sorted(range(5), key=lambda i: sloppy[i]) == sorted(
            range(5), key=lambda i: values[i]
        ):
            continue
        break

    # Second route: re-read each written form into a value with string parsing, independent of the draw.
    for name, text, v in chosen:
        if name == "numeral":
            check(int(text.replace(",", "")) == v)
        elif name == "thousand_and":
            lead, rest = text.split(" thousand and ")
            check(int(lead) * 1000 + int(rest) == v)
        elif name == "millions":
            check(int(text.split(".")[0]) * 10**6 + int(text.split(".")[1].split(" ")[0]) * 10**5 == v)
        else:
            check(sum(int(p.replace(",", "")) for p in text.split(" + ")) == v)

    key = ", ".join(f"{v:,}" for v in ordered)
    shown_text = "; ".join(t for _, t, _ in chosen)
    rev, swap = reversed_and_adjacent_swap(ordered)

    def listed(vals: list[int]) -> str:
        return ", ".join(f"{v:,}" for v in vals)

    sloppy_order = [v for _, v in sorted(zip(sloppy, values))]
    by_first = [
        v for _, v in sorted(zip((int(t.replace(",", "").split()[0].split(".")[0]) for _, t, _ in chosen), values))
    ]
    candidates = [
        Distractor(listed(sloppy_order), "placeholder-zero-dropped", sloppy_order),
        Distractor(listed(by_first), "ordered-by-first-number-shown", by_first),
        Distractor(listed(rev), "reversed-order", rev),
        Distractor(listed(swap), "adjacent-swap", swap),
    ]
    distractors = pick_distractors(key, candidates, 4)
    check(len(distractors) >= 2, "too few distractors")

    return dict(
        text=f"{rng.choice(_EVENTS)} {shown_text}. {rng.choice(_ASK)}",
        answer=ordered,
        answer_kind="ordered_list",
        answer_text=key,
        answer_latex=key,
        working=[
            *(f"{t} = {v:,}." for _, t, v in chosen),
            f"Smallest to largest: {key}.",
        ],
        distractors=distractors,
    )
