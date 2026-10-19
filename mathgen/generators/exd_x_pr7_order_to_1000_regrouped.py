"""Exam tier for "order numbers to 1000": the numbers are described, not written as numerals.

The practice tiers show three-digit numerals. Here five numbers are given as regrouped place-value descriptions
("3 hundreds, 14 tens and 6 ones") or as comparisons ("100 more than 468", "10 less than 700"), plus at most one
plain numeral. The learner must work out each number and then order the numerals, which is what a Year 3 or 4
national-assessment question on place value asks.

Built backwards from five distinct values between 100 and 999. Each description is checked by re-reading it
into a value with plain arithmetic.
"""

from __future__ import annotations

import random

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, pick_distractors, reversed_and_adjacent_swap

_STEMS = [
    "A teacher at a school in {town} wrote five numbers on the board in different ways: {items}.",
    "Five library book numbers at a school in {town} are described in different ways: {items}.",
    "Five houses in a street in {town} have numbers described in different ways: {items}.",
]
_TOWNS = ["Bunbury", "Cairns", "Wagga Wagga", "Launceston", "Mildura", "Gladstone", "Albany", "Whyalla"]
_ASK = [
    "Write the numbers as numerals, from smallest to largest.",
    "Put the numbers in order from smallest to largest, writing each as a numeral.",
]


def _describe(rng: random.Random, v: int) -> tuple[str, str, int]:
    """(kind, description, the value a learner reaches by the matching slip) for one number."""
    h, t, o = v // 100, (v // 10) % 10, v % 10
    kind = rng.choice(["regroup_tens", "regroup_tens", "more100", "less10", "less1", "plain"])
    if kind == "regroup_tens" and h >= 2:
        hundreds = f"{h - 1} hundred" if h == 2 else f"{h - 1} hundreds"
        ones = "" if o == 0 else f" and {o} {'one' if o == 1 else 'ones'}"
        text = f"{hundreds}, {t + 10} tens{ones}" if o else f"{hundreds} and {t + 10} tens"
        return kind, text, int(f"{h - 1}{t + 10}{o}")
    if kind == "more100" and v - 100 >= 100:
        return kind, f"100 more than {v - 100}", v - 100 + 10
    if kind == "less10":
        return kind, f"10 less than {v + 10}", v + 10 - 100 if v + 10 <= 999 else -1
    if kind == "less1" and v + 1 <= 999:
        return kind, f"1 less than {v + 1}", v + 1 - 10
    return "plain", str(v), v


@generator("exd_x_pr7_order_to_1000_regrouped", level=2, presets={"exam": {"tier": "exam"}})
def exd_x_pr7_order_to_1000_regrouped(rng: random.Random, tier: str = "exam") -> dict:
    """Work out five described numbers, then order them. See module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(2000, "five described numbers, at most one plain and one regrouped"):
        values = rng.sample(range(120, 990), 5)
        if len({v // 100 for v in values}) > 4:
            continue
        rows = [_describe(rng, v) for v in values]
        if any(slip < 0 for _, _, slip in rows):
            continue
        kinds = [k for k, _, _ in rows]
        if kinds.count("plain") > 1 or "regroup_tens" not in kinds or len(set(kinds)) < 3:
            continue
        slips = [slip for _, _, slip in rows]
        if len(set(slips)) < 5 or sorted(range(5), key=lambda i: slips[i]) == sorted(range(5), key=lambda i: values[i]):
            continue
        break

    # Second route: re-read each description from its words.
    for (kind, text, _), v in zip(rows, values):
        words = text.replace(",", "").split()
        if kind == "regroup_tens":
            nums = [int(w) for w in words if w.isdigit()]
            check(nums[0] * 100 + nums[1] * 10 + (nums[2] if len(nums) > 2 else 0) == v)
        elif kind == "plain":
            check(int(text) == v)
        else:
            amount = int(words[0])
            check(int(words[-1]) + (amount if "more" in text else -amount) == v)

    ordered = sorted(values)
    key = ", ".join(str(v) for v in ordered)
    rev, swap = reversed_and_adjacent_swap(ordered)
    slip_order = [v for _, v in sorted(zip(slips, values))]

    def listed(vals):
        return ", ".join(str(v) for v in vals)

    regroup_slip = [
        (int(f"{v // 100 - 1}{(v // 10) % 10 + 10}{v % 10}") if k == "regroup_tens" else v)
        for (k, _, _), v in zip(rows, values)
    ]
    place_slip = [(s if k in ("more100", "less10", "less1") else v) for (k, _, s), v in zip(rows, values)]
    regroup_order = [v for _, v in sorted(zip(regroup_slip, values))]
    place_order = [v for _, v in sorted(zip(place_slip, values))]
    candidates = [
        Distractor(listed(regroup_order), "regrouped-quantity-read-as-digits", regroup_order),
        Distractor(listed(place_order), "more-less-wrong-place", place_order),
        Distractor(listed(rev), "reversed-order", rev),
        Distractor(listed(swap), "adjacent-swap", swap),
    ]
    distractors = pick_distractors(key, candidates, 4)
    check(len(distractors) >= 2, "too few distractors")
    check(listed(slip_order) != key)

    items = "; ".join(text for _, text, _ in rows)
    return dict(
        text=f"{rng.choice(_STEMS).format(town=rng.choice(_TOWNS), items=items)} {rng.choice(_ASK)}",
        answer=ordered,
        answer_kind="ordered_list",
        answer_text=key,
        answer_latex=key,
        working=[
            *(
                (f"{text} is already a number." if text == str(v) else f"{text} = {v}.")
                for (_, text, _), v in zip(rows, values)
            ),
            f"Smallest to largest: {key}.",
        ],
        distractors=distractors,
    )
