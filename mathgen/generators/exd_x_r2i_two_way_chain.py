"""Exam-tier chain for `senior-interpret-information-presented-in-two-way-tables`: complete, compare, conclude.

A survey reports a grand total and three cells of a two-by-two table. The chain asks for

(a) the missing cell,
(b) the percentage of the first group who gave the first response,
(c) the same percentage for the second group, which needs (a), and
(d) the difference between the two percentages, the figure a report would quote when comparing the groups.

The table is built backwards from four cells. Both percentages are rounded to 1 decimal place and kept clear of a
rounding boundary, and (d) is the difference of the two rounded figures, which is what the question asks for.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import attempts, check, generator, round_half_up

from .exd_x_r2i_support import clear_of_boundary, number_part

# (people, rows, columns, outcome noun phrase for the first column)
_THEMES = [
    ("voters", ("Urban", "Rural"), ("Support", "Oppose"), "support the proposal"),
    ("patients", ("Under 40", "40 or older"), ("Medication A", "Medication B"), "were given Medication A"),
    ("students", ("Year 11", "Year 12"), ("Walk or cycle", "Other"), "walk or cycle to school"),
    ("members", ("Junior", "Senior"), ("Renewed", "Did not renew"), "renewed their membership"),
    ("customers", ("Online", "In store"), ("Satisfied", "Not satisfied"), "were satisfied"),
    ("residents", ("House", "Apartment"), ("Recycle", "Do not recycle"), "recycle"),
    ("apprentices", ("First year", "Second year"), ("Passed", "Did not pass"), "passed the assessment"),
    ("campers", ("Coastal", "Bush"), ("Booked again", "Did not book again"), "booked again"),
]


@generator("exd_x_r2i_two_way_chain", level=20, presets={"exam": {"tier": "exam"}})
def exd_x_r2i_two_way_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: complete a two-way table from its total, then compare two row percentages."""
    check(tier == "exam", f"unknown tier {tier!r}")
    people, rows, cols, outcome = rng.choice(_THEMES)
    for _ in attempts(2000, "a table whose row percentages sit clear of a rounding boundary"):
        c11, c12, c21, c22 = (rng.randint(12, 90) for _ in range(4))
        n = c11 + c12 + c21 + c22
        row1 = c11 + c12
        row2 = c21 + c22
        pct1 = Fraction(100 * c11, row1)
        pct2 = Fraction(100 * c21, row2)
        if not (clear_of_boundary(pct1, 1) and clear_of_boundary(pct2, 1)):
            continue
        shown1, shown2 = round_half_up(pct1, 1), round_half_up(pct2, 1)
        if shown1 == shown2 or abs(shown1 - shown2) < 2 or c22 in (c11, c12, c21):
            continue
        break
    diff = abs(shown1 - shown2)
    check(row1 + row2 == n and c22 == n - c11 - c12 - c21, "missing cell disagrees with the total")

    cells = (
        f"{rows[0]}/{cols[0]}: {c11}, {rows[0]}/{cols[1]}: {c12}, {rows[1]}/{cols[0]}: {c21}, "
        f"{rows[1]}/{cols[1]}: not recorded"
    )
    stem = (
        f"A survey of {n} {people} recorded two categorical variables. Rows: {rows[0]}, {rows[1]}. "
        f"Columns: {cols[0]}, {cols[1]}. {cells}. The grand total of {n} is known, so the missing count can be found."
    )

    pct_c_from_total = Fraction(100 * c21, n)
    candidates_a = [
        (c11 + c12 + c21, "added-instead-of-subtracted"),
        (n - c11 - c12, "left-out-a-value"),
        (n - c21, "left-out-a-value"),
    ]
    candidates_b = [
        (Fraction(100 * c11, n), "wrong-denominator-for-proportion"),
        (Fraction(100 * c11, c11 + c21), "wrong-denominator-for-proportion"),
        (Fraction(c11), "gave-count-not-percentage"),
        (100 - pct1, "gave-complement-for-event"),
    ]
    candidates_c = [
        (pct_c_from_total, "wrong-denominator-for-proportion"),
        (Fraction(100 * c21, c11 + c21), "wrong-denominator-for-proportion"),
        (100 - pct2, "gave-complement-for-event"),
        (Fraction(c21), "gave-count-not-percentage"),
    ]
    candidates_d = [
        (shown1 + shown2, "added-instead-of-subtracted"),
        (shown2 - shown1 if shown1 > shown2 else shown1 - shown2, "reversed-difference"),
        (Fraction(abs(c11 - c21)), "compared-counts-not-percentages"),
        (Fraction(100 * abs(c11 - c21), n), "wrong-denominator-for-proportion"),
    ]

    first_label = f"{rows[0]}"
    second_label = f"{rows[1]}"
    parts = [
        number_part(
            f"Find the number of {people} in the {rows[1]} group whose response was {cols[1]}.",
            Fraction(c22),
            0,
            1,
            [f"{n} − {c11} − {c12} − {c21} = {c22}."],
            candidates_a,
            sensible_range=(0, None),
        ),
        number_part(
            f"What percentage of the {people} in the {first_label} group {outcome}? "
            "Give your answer to 1 decimal place.",
            shown1,
            1,
            2,
            [f"Row total = {c11} + {c12} = {row1}.", f"{c11} ÷ {row1} × 100 = {float(shown1):.1f}%."],
            candidates_b,
            sensible_range=(0, 100),
        ),
        number_part(
            f"What percentage of the {people} in the {second_label} group {outcome}? "
            "Give your answer to 1 decimal place.",
            shown2,
            1,
            2,
            [f"Row total = {c21} + {c22} = {row2}.", f"{c21} ÷ {row2} × 100 = {float(shown2):.1f}%."],
            candidates_c,
            follow_through=f"100*{c21}/({c21} + part_a)",
            sensible_range=(0, 100),
        ),
        number_part(
            "Find the difference between the two percentages in (b) and (c), larger minus smaller, in "
            "percentage points. Give your answer to 1 decimal place.",
            diff,
            1,
            1,
            [f"|{float(shown1):.1f} − {float(shown2):.1f}| = {float(diff):.1f} percentage points."],
            candidates_d,
            follow_through="Abs(part_b - part_c)",
            sensible_range=(0, 100),
        ),
    ]
    return dict(text=stem, parts=parts)
