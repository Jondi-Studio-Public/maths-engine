"""Exam source: one amount as a percentage of another in two years, then the change in that share.

One scenario: in each of two years a total is counted and a part of it is recorded.

(a) the part as a percentage of the total in the first year;
(b) the same in the second year;
(c) the change in the share in percentage points, from (a) and (b);
(d) the percentage increase in the NUMBER in the part (not in the share).

Part (c) is the classic trap between a percentage point and a percent. Shares are rounded to 1 decimal place and
(c) is worked from the two rounded shares, as the question tells the learner to do; a draw is kept only when (a),
(b) and (d) are clear of a rounding boundary.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import attempts, generator, phrase, round_half_up
from mathgen.generators.exd_x_n1_support import clear_of_boundary, money, number_part, show

# (stem, what is counted, which of them are recorded)
_SCENES = (
    (
        "In one year a school canteen sold {n1} items, and {k1} of them were from the healthy menu. The next year "
        "it sold {n2} items, and {k2} of those were from the healthy menu.",
        "items sold",
        "from the healthy menu",
    ),
    (
        "A library recorded {n1} loans in one year, and {k1} of them were e-books. The following year it recorded "
        "{n2} loans, and {k2} of them were e-books.",
        "loans",
        "e-books",
    ),
    (
        "A suburban swimming pool had {n1} entries in one year, and {k1} of them were concession entries. The "
        "next year it had {n2} entries, and {k2} were concession entries.",
        "entries",
        "concession entries",
    ),
)


@generator("exd_x_n1_share_and_change", level=11, presets={"exam": dict(tier="exam")})
def exd_x_n1_share_and_change(rng, tier="exam"):
    for _ in attempts(1000, "a share draw clear of rounding boundaries"):
        n1 = rng.randint(800, 4200)
        n2 = rng.randint(800, 4200)
        k1 = rng.randint(n1 * 15 // 100, n1 * 60 // 100)
        k2 = rng.randint(n2 * 15 // 100, n2 * 70 // 100)
        s1 = Fraction(k1, n1) * 100
        s2 = Fraction(k2, n2) * 100
        grow = Fraction(k2 - k1, k1) * 100
        if not (s2 > s1 + 2 and k2 > k1):
            continue
        if not all(clear_of_boundary(v, 1) for v in (s1, s2, grow)):
            continue
        a, b = round_half_up(s1, 1), round_half_up(s2, 1)
        if b - a <= 0 or n1 == n2:
            continue
        break

    pattern, things, which = rng.choice(_SCENES)
    stem = phrase(rng, (pattern,), n1=money(n1, 0), n2=money(n2, 0), k1=money(k1, 0), k2=money(k2, 0))
    points = b - a
    relative = (b - a) / a * 100
    count_relative_distractor = Fraction(k2 - k1, k2) * 100
    part_a = number_part(
        f"What percentage of the first year's {things} were {which}? Give your answer correct to 1 decimal place.",
        s1,
        1,
        1,
        [f"{k1} ÷ {n1} × 100 = {show(s1, 1)}%."],
        [(Fraction(n1, k1) * 100, None), (Fraction(k1, n1), None), (Fraction(k1, n1 - k1) * 100, None)],
        sensible_range=(0, 100),
    )
    part_b = number_part(
        f"What percentage of the second year's {things} were {which}? Give your answer correct to 1 decimal place.",
        s2,
        1,
        1,
        [f"{k2} ÷ {n2} × 100 = {show(s2, 1)}%."],
        [(Fraction(n2, k2) * 100, None), (Fraction(k2, n2), None), (Fraction(k2, n2 - k2) * 100, None)],
        sensible_range=(0, 100),
    )
    part_c = number_part(
        "Using your answers to (a) and (b), by how many percentage points did the share increase? Give your answer "
        "correct to 1 decimal place.",
        points,
        1,
        1,
        [f"{show(b, 1)} - {show(a, 1)} = {show(points, 1)} percentage points."],
        [(relative, None), (Fraction(k2 - k1, n1) * 100, None), (a - b, None), (b / a, None)],
        follow_through="part_b - part_a",
        sensible_range=(0, 100),
    )
    part_d = number_part(
        f"Find the percentage increase in the NUMBER of {things} that were {which} (not in their share), from the "
        "first year to the second, correct to 1 decimal place.",
        grow,
        1,
        2,
        [f"Increase = {k2} - {k1} = {k2 - k1}; {k2 - k1} ÷ {k1} × 100 = {show(grow, 1)}%."],
        [
            (count_relative_distractor, "divides-by-final"),
            (Fraction(k2, k1) * 100, "new-as-percent-of-original"),
            (points, None),
            (Fraction(k2 - k1), "gave-change-not-percent"),
        ],
        sensible_range=(0, None),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
