"""Exam tier for `f10-symmetry-construction`: the symmetry of a regular polygon with some vertices marked.

The hard tier names the number of lines and the order of rotation of a plain regular polygon. Here the polygon is
described only by an angle fact, and every k-th vertex carries a dot, so the figure's symmetry is smaller than the
polygon's own. The learner must

1. recover the number of sides from an interior angle, an exterior angle or the sum of the interior angles,
2. see that the dots form a smaller regular polygon (m = n / k of them) and count the lines of symmetry of the figure,
3. find the smallest turn that maps the figure onto itself, 360 / m degrees.

Built backwards from the number of dots m and the gap k. Parts (b) and (c) carry `follow_through` from the number of
sides, so a wrong count in (a) does not lose the later marks.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, generator, round_half_up
from .exd_x_sc3_support import number_part, tagged

_ORDINALS = {
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
}
_MARGIN = Fraction(1, 1000)


def _clear_of_rounding(m: int) -> bool:
    """The one-decimal-place turn 360/m does not sit on a rounding boundary."""
    offset = (Fraction(360, m) * 10) % 1
    return abs(offset - Fraction(1, 2)) > _MARGIN * 10 or offset == 0


_COMBOS = [(k, m) for k in _ORDINALS for m in range(3, 31) if k * m <= 80 and _clear_of_rounding(m)]


def _angle_fact(rng: random.Random, n: int) -> tuple[str, str]:
    """(sentence giving the polygon, which kind of fact) for an n-sided regular polygon."""
    forms = ["sum"]
    if 360 % n == 0:
        forms += ["interior", "exterior"]
    form = rng.choice(forms)
    if form == "interior":
        return f"Each interior angle of a regular polygon is {180 - 360 // n}°.", form
    if form == "exterior":
        return f"Each exterior angle of a regular polygon is {360 // n}°.", form
    return f"The interior angles of a regular polygon add to {(n - 2) * 180}°.", form


@generator("exd_x_sc3_symmetry_chain", level=10, presets={"exam": dict(tier="exam")})
def exd_x_sc3_symmetry_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Sides from an angle fact, then lines of symmetry and the smallest turn of the polygon with marked vertices."""
    check(tier == "exam", f"unknown tier {tier!r}")
    k, m = rng.choice(_COMBOS)
    n = k * m
    fact, form = _angle_fact(rng, n)
    turn = round_half_up(Fraction(360, m), 1)
    # Keep the one-decimal-place turn clear of a rounding boundary, so working from the rounded n gives the same text.
    offset = (Fraction(360, m) * 10) % 1
    check(abs(offset - Fraction(1, 2)) > _MARGIN * 10 or offset == 0, "turn angle sits on a rounding boundary")

    stem = f"{fact} Starting from one vertex, a dot is marked on every {_ORDINALS[k]} vertex."

    a_cands = [(Fraction(n, 2), "half-turn-used-for-sides")] if form != "sum" and n % 2 == 0 else []
    if form == "sum":
        a_cands += [(Fraction(n - 2), "forgets-to-add-two-to-the-sides")]
    a_cands += [(Fraction(n - 1), "sides-from-angle-slip"), (Fraction(n + 1), "sides-from-angle-slip")]
    b_cands = [
        (Fraction(n), "ignores-the-marked-vertices"),
        (Fraction(k), "reports-the-gap-as-the-count"),
        (Fraction(1), "counts-only-the-line-through-the-first-dot"),
        (Fraction(2 * m), "lines-doubled"),
    ]
    c_cands = [
        (Fraction(360, n), "ignores-the-marked-vertices"),
        (Fraction(360, k), "reports-the-gap-as-the-count"),
        (Fraction(180, m), "uses-a-half-turn-as-the-full-turn"),
        (Fraction(360), "only-the-full-turn"),
    ]
    sides_working = {
        "interior": [f"Exterior angle = 180° - {180 - 360 // n}° = {360 // n}°.", f"Sides = 360° ÷ {360 // n}° = {n}."],
        "exterior": [f"Sides = 360° ÷ {360 // n}° = {n}."],
        "sum": [f"(n - 2) × 180° = {(n - 2) * 180}°, so n - 2 = {n - 2} and n = {n}."],
    }[form]
    parts = [
        number_part(
            "How many sides does the polygon have?",
            Fraction(n),
            0,
            2,
            sides_working,
            tagged(Fraction(n), a_cands, 0, lo=0),
            sensible_range=(2, None),
        ),
        number_part(
            "How many lines of symmetry does the polygon with its dots have?",
            Fraction(m),
            0,
            1,
            [
                f"The dots are every {_ORDINALS[k]} vertex, so there are {n} ÷ {k} = {m} of them.",
                f"They form a regular {m}-sided polygon, with {m} lines of symmetry.",
            ],
            tagged(Fraction(m), b_cands, 0, lo=0),
            follow_through=f"part_a/{k}",
            sensible_range=(0, None),
        ),
        number_part(
            "What is the smallest angle, in degrees, through which the polygon with its dots can be turned about its "
            "centre to fit onto itself again? Give your answer correct to 1 decimal place.",
            turn,
            1,
            2,
            [f"The figure fits onto itself {m} times in a full turn.", f"360° ÷ {m} = {float(turn):.1f}°."],
            tagged(turn, c_cands, 1, lo=0),
            follow_through=f"360*{k}/part_a",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, parts=parts)
