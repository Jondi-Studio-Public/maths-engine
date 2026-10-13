"""Exam-tier source for `f10-geometric-construction-algorithm`: a refined construction that feeds a second right triangle.

A five-step construction draws a segment AB, marks the arcs' meeting point P above AB with the compasses at a stated
setting, joins P to the midpoint M of AB, marks a point Q on PM a stated distance from P, and joins A to Q. A
refinement then changes the compass setting. Three connected parts, each feeding the next:

(a) the length PM under the refined setting (Pythagoras in triangle AMP);
(b) the length MQ (follow-through from (a): PM less the stated PQ);
(c) the length AQ (follow-through from (b): Pythagoras in triangle AMQ).

Built backwards from half-length h = AM: the refined setting and the final triangle AMQ both have h as a leg, so
every length the learner finds is a whole number of millimetres. The original setting only has to be long enough for
the arcs to meet, so ignoring the refinement gives a different answer.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..answers import Distractor
from ..core import attempts, check, generator, latex_answer, pick_distractors

_MAX_HALF = 72
_MAX_HYP = 150


def _legs_with_hypotenuse(h: int) -> list[tuple[int, int]]:
    """Every (m, hyp) with h^2 + m^2 = hyp^2, m >= 3 and hyp within the compass's reach."""
    out = []
    for m in range(3, 150):
        sq = h * h + m * m
        hyp = math.isqrt(sq)
        if hyp * hyp == sq and hyp <= _MAX_HYP:
            out.append((m, hyp))
    return out


# Labels for the segment ends, the apex, the midpoint and the marked point; and two ways to word the marking step.
_LABELS = (("A", "B", "P", "M", "Q"), ("K", "L", "S", "N", "T"), ("D", "E", "F", "G", "H"), ("X", "Y", "Z", "W", "V"))
_MARK_STEPS = (
    "On {p}{m}, mark point {q} so that {p}{q} = {d} mm.",
    "Mark point {q} on {p}{m}, {d} mm from {p}.",
    "On the line {p}{m}, mark point {q} {d} mm from {p}.",
)

_TRIPLES = {h: legs for h in range(8, _MAX_HALF + 1) if len(legs := _legs_with_hypotenuse(h)) >= 2}


@generator("exd_x_sc2_construction_chain", level=9, presets={"exam": dict(tier="exam")})
def exd_x_sc2_construction_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: refined compass setting, then the distance MQ, then the length AQ."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(2000, "two right triangles sharing the half-length"):
        h = rng.choice(sorted(_TRIPLES))
        legs = _TRIPLES[h]
        (m_new, c_new), (m_q, aq) = rng.sample(legs, 2)
        if m_q >= m_new:
            continue  # Q must lie between P and M
        d = m_new - m_q  # PQ
        c_old = rng.randint(h + 3, _MAX_HYP)  # the original setting only has to be long enough for the arcs to meet
        if d < 3 or c_old == c_new:
            continue
        break
    length = 2 * h
    check(c_new * c_new == h * h + m_new * m_new and aq * aq == h * h + m_q * m_q, "triples disagree")
    check(m_new - d == m_q and c_old > h, "chain disagrees")
    old_sq = c_old * c_old - h * h
    m_old = math.isqrt(old_sq) if math.isqrt(old_sq) ** 2 == old_sq else None  # the original PM, when whole

    a, b, p, m, q = rng.choice(_LABELS)
    mark = rng.choice(_MARK_STEPS).format(p=p, m=m, q=q, d=d)
    stem = (
        "A construction algorithm has five steps. "
        f"Step 1: Draw segment {a}{b}, {length} mm long. "
        f"Step 2: Set the compasses to {c_old} mm and draw an arc centred on {a} and an arc centred on {b}, meeting "
        f"at point {p} on one side of {a}{b}. "
        f"Step 3: Mark the midpoint {m} of {a}{b} and join {p} to {m}. "
        f"Step 4: {mark} "
        f"Step 5: Join {a} to {q}. "
        f"A refinement changes Step 2 so that the compasses are set to {c_new} mm."
    )

    a_cands = [
        Distractor(str(c_new), "reports-the-compass-setting", Fraction(c_new)),
        Distractor(str(m_old), "ignores-the-refinement", Fraction(m_old)) if m_old else None,
        Distractor(str(c_old), None, Fraction(c_old)),
        Distractor(str(m_new + 2), None, Fraction(m_new + 2)),
        Distractor(str(c_new - h), "subtracts-instead-of-pythagoras", Fraction(c_new - h)),
        Distractor(str(h), "reports-one-leg", Fraction(h)),
    ]
    b_cands = [
        Distractor(str(d), "reports-the-given-length", Fraction(d)),
        Distractor(str(m_new + d), "adds-instead-of-subtracting-segment", Fraction(m_new + d)),
        Distractor(str(m_old - d), "ignores-the-refinement", Fraction(m_old - d)) if m_old and m_old > d else None,
        Distractor(str(c_new - d), None, Fraction(c_new - d)),
    ]
    c_cands = [
        Distractor(str(m_q), "reports-one-leg", Fraction(m_q)),
        Distractor(str(c_new), "reports-wrong-triangle-hypotenuse", Fraction(c_new)),
        Distractor(str(h + m_q), "adds-the-two-legs", Fraction(h + m_q)),
        Distractor(
            str(math.isqrt(h * h + d * d)) if math.isqrt(h * h + d * d) ** 2 == h * h + d * d else str(h + d),
            "uses-the-wrong-leg",
            Fraction(math.isqrt(h * h + d * d)) if math.isqrt(h * h + d * d) ** 2 == h * h + d * d else Fraction(h + d),
        ),
    ]
    a_dis = pick_distractors(str(m_new), [c for c in a_cands if c], 3)
    b_dis = pick_distractors(str(m_q), [c for c in b_cands if c], 3)
    c_dis = pick_distractors(str(aq), c_cands, 3)
    check(len(a_dis) >= 2 and len(b_dis) >= 2 and len(c_dis) >= 2, "too few distractors")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"After the refinement, how long is {p}{m}, in mm?",
                answer_kind="number",
                answer=m_new,
                answer_text=str(m_new),
                answer_latex=latex_answer(str(m_new)),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"{a}{m} = {length} / 2 = {h} mm and {a}{p} = {c_new} mm, with a right angle at {m}.",
                    f"{p}{m}² = {c_new}² − {h}² = {c_new**2 - h * h}, so {p}{m} = {m_new} mm.",
                ],
                distractors=a_dis,
            ),
            dict(
                prompt=f"Hence find the length of {m}{q}, in mm.",
                answer_kind="number",
                answer=m_q,
                answer_text=str(m_q),
                answer_latex=latex_answer(str(m_q)),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"part_a - {d}",
                working=[f"{q} is on {p}{m}, so {m}{q} = {p}{m} − {p}{q} = {m_new} − {d} = {m_q} mm."],
                distractors=b_dis,
            ),
            dict(
                prompt=f"Find the length of {a}{q}, in mm.",
                answer_kind="number",
                answer=aq,
                answer_text=str(aq),
                answer_latex=latex_answer(str(aq)),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"sqrt({h * h} + part_b**2)",
                working=[f"{a}{q}² = {a}{m}² + {m}{q}² = {h}² + {m_q}² = {aq * aq}, so {a}{q} = {aq} mm."],
                distractors=c_dis,
            ),
        ],
    )
