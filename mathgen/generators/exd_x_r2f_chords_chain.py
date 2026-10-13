"""Exam source for "angles at the circumference subtended by the same arc": a four-part chain on crossing chords.

Chords AC and BD of a circle meet at P inside the circle. Two angles on arc BC are given as linear expressions in
x (angle BAC and angle BDC), and so is angle ACD, which stands on arc AD with angle ABD. The learner works through:

(a) solve for x from angle BAC = angle BDC                    2 marks
(b) angle BAC, by substituting x                               1 mark, follows from (a)
(c) angle ABD, which equals angle ACD (same arc AD)            2 marks, follows from (a)
(d) angle CPD: angle sum of triangle ABP, then vertically opposite   2 marks, follows from (b) and (c)

Built backwards from the integer x and the two angles, so every answer is a whole number of degrees. The four arcs
of the circle add to 360 degrees, which bounds the angles: angle APB = 180 - BAC - ABD must be positive and sensible.
"""

from __future__ import annotations

import random

from ..answers import Distractor
from ..core import attempts, check, generator, join_terms, latex_answer, pick_distractors


def _lin(a: int, b: int) -> str:
    return join_terms([(a, "x"), (b, "")])


def _draw(rng: random.Random):
    """x, angle BAC, angle ABD and the three expressions that take those values at x."""
    for _ in attempts(1000, "crossing-chord angles"):
        x = rng.randint(4, 25)
        a1, a2 = rng.sample(range(1, 6), 2)
        bac = rng.randint(24, 70)
        abd = rng.randint(20, 70)
        c = rng.randint(1, 5)
        b1, b2, d = bac - a1 * x, bac - a2 * x, abd - c * x
        if abs(b1) <= 60 and abs(b2) <= 60 and abs(d) <= 60 and bac + abd < 160 and (c, d) != (a1, b1):
            cands = _candidates(x, bac, abd, a1, b1, a2, b2, c, d)
            if _has_two_distractors(x, bac, abd, 180 - bac - abd, *cands):
                return x, bac, abd, (a1, b1), (a2, b2), (c, d)
    raise AssertionError("unreachable")  # pragma: no cover


def _candidates(x, bac, abd, a1, b1, a2, b2, c, d):
    """The tagged wrong answers for parts (a) to (d), as (tag, value) pairs."""
    cpd = 180 - bac - abd
    a_cands = [
        ("sign-error-moving-term", (b1 - b2) // (a1 - a2) if (b1 - b2) % (a1 - a2) == 0 else None),
        ("angles-added-instead-of-equated", (180 - b1 - b2) // (a1 + a2) if (180 - b1 - b2) % (a1 + a2) == 0 else None),
        ("gives-angle-not-unknown", bac),
        ("sign-error-moving-term", (b2 - b1) // (a1 + a2) if (b2 - b1) % (a1 + a2) == 0 else None),
        ("ignored-constant-term", bac // a1 if bac % a1 == 0 else None),
    ]
    b_cands = [
        ("assumes-supplementary", 180 - bac),
        ("forgets-coefficient-when-substituting", x + b1),
        ("central-angle-confused-with-circumference-angle", 2 * bac),
        ("sign-error-linear-term", -a1 * x + b1),
    ]
    c_cands = [
        ("wrong-arc-pair-equated", bac),
        ("assumes-supplementary", 180 - abd),
        ("sign-error-linear-term", c * x - d),
        ("forgets-coefficient-when-substituting", x + d),
        ("central-angle-confused-with-circumference-angle", 2 * abd),
        ("sign-error-linear-term", -c * x + d),
    ]
    d_cands = [
        ("triangle-angle-sum-not-subtracted", bac + abd),
        ("adjacent-angle-for-vertically-opposite", 180 - cpd),
        ("omits-an-angle-in-the-triangle-sum", 180 - bac),
        ("omits-an-angle-in-the-triangle-sum", 180 - abd),
    ]
    return a_cands, b_cands, c_cands, d_cands


def _has_two_distractors(x, bac, abd, cpd, a_cands, b_cands, c_cands, d_cands) -> bool:
    """Whether every part has two distinct whole-number wrong answers in range (the generator needs at least two)."""
    for key, cands, lo, hi in (
        (x, a_cands, -200, 200),
        (bac, b_cands, 1, 179),
        (abd, c_cands, 1, 179),
        (cpd, d_cands, 1, 179),
    ):
        usable = {v for _, v in cands if isinstance(v, int) and lo <= v <= hi and v != key}
        if len(usable) < 2:
            return False
    return True


def _angle_part(prompt, value, marks, working, distractors, follow_through=None, hi=180):
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=value,
        answer_text=str(value),
        answer_latex=latex_answer(str(value)),
        marks=marks,
        sensible_range=(0, hi),
        working=working,
        distractors=distractors,
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


def _tag(key: int, candidates, lo=1, hi=179):
    """Whole-number tagged distractors that are not the key."""
    tagged = [Distractor(str(v), tag, v) for tag, v in candidates if isinstance(v, int) and lo <= v <= hi and v != key]
    out = pick_distractors(str(key), tagged, 3)
    check(len(out) >= 2, "too few distractors")
    return out


@generator(
    "exd_x_r2f_chords_chain",
    level=10,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_r2f_chords_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Crossing chords: x from one pair of equal angles, then two more angles, then the angle at the crossing."""
    check(tier == "exam", f"unknown tier {tier!r}")
    x, bac, abd, (a1, b1), (a2, b2), (c, d) = _draw(rng)
    cpd = 180 - bac - abd
    check(a1 * x + b1 == a2 * x + b2 == bac and c * x + d == abd, "angles disagree at x")
    check(cpd > 0, "angle at the crossing is not positive")

    stem = (
        "Chords AC and BD of a circle meet at the point P inside the circle. "
        f"Angle BAC = ({_lin(a1, b1)})°, angle BDC = ({_lin(a2, b2)})° and angle ACD = ({_lin(c, d)})°."
    )
    a_cands, b_cands, c_cands, d_cands = _candidates(x, bac, abd, a1, b1, a2, b2, c, d)
    return dict(
        text=stem,
        parts=[
            _angle_part(
                "Find the value of x.",
                x,
                2,
                [
                    "Angles BAC and BDC stand on the same arc BC, so they are equal.",
                    f"{_lin(a1, b1)} = {_lin(a2, b2)} gives x = {x}.",
                ],
                _tag(x, [(t, v) for t, v in a_cands if v is not None], lo=-200, hi=200),
                hi=None,
            ),
            _angle_part(
                "Hence find the size of angle BAC, in degrees.",
                bac,
                1,
                [f"Angle BAC = {a1}({x}) + ({b1}) = {bac} degrees."],
                _tag(bac, b_cands),
                follow_through=f"({a1})*part_a + ({b1})",
            ),
            _angle_part(
                "Find the size of angle ABD, in degrees.",
                abd,
                2,
                [
                    "Angles ABD and ACD stand on the same arc AD, so they are equal.",
                    f"Angle ABD = {c}({x}) + ({d}) = {abd} degrees.",
                ],
                _tag(abd, c_cands),
                follow_through=f"({c})*part_a + ({d})",
            ),
            _angle_part(
                "Hence find the size of angle CPD, in degrees.",
                cpd,
                2,
                [
                    f"In triangle ABP, angle APB = 180 - {bac} - {abd} = {cpd} degrees.",
                    "Angle CPD is vertically opposite angle APB, so it is also " + f"{cpd} degrees.",
                ],
                _tag(cpd, d_cands),
                follow_through="180 - part_b - part_c",
            ),
        ],
    )
