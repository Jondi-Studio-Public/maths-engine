"""Exam-tier source for `f10-congruence-similarity-criteria`: use the SAS similarity test to build, then use, a second triangle.

Triangle ABC is described by its three sides and the angle at B. Triangle PQR has the same angle at Q and one
stated side next to it. Three connected parts:

(a) the length the other side beside the angle must have for PQR to be similar to ABC by the SAS test, with P, Q,
    R corresponding to A, B, C (the learner must match the sides that include the angle, and scale by the ratio
    of the pair already given);
(b) the third side of PQR, by the same ratio (follow-through from (a));
(c) the perimeter of PQR (follow-through from (a) and (b)).

Every triangle is a multiple of a base triangle whose angle at B is exactly 60, 90 or 120 degrees (checked with
the cosine rule), so the sides and the stated angle agree. Triangle PQR is a different whole-number multiple of
the same base, so every length is a whole number of centimetres.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..answers import Distractor
from ..core import attempts, check, generator, latex_answer, pick_distractors

# (AB, BC, AC, angle B in degrees): the angle between the first two sides, opposite the third.
_BASES = (
    (3, 4, 5, 90),
    (5, 12, 13, 90),
    (8, 15, 17, 90),
    (6, 8, 10, 90),
    (5, 8, 7, 60),
    (3, 8, 7, 60),
    (7, 15, 13, 60),
    (8, 15, 13, 60),
    (3, 5, 7, 120),
    (7, 8, 13, 120),
    (5, 3, 7, 120),
    (8, 5, 7, 60),
    (7, 24, 25, 90),
    (20, 21, 29, 90),
    (16, 21, 19, 60),
    (5, 16, 19, 120),
    (7, 40, 37, 60),
    (7, 33, 37, 120),
)
_SCENES = (
    ("two triangular garden beds", "m"),
    ("two triangular pieces of timber framing", "cm"),
    ("two triangles on a design plan", "cm"),
    ("two triangular shade cloths", "m"),
)
_MAX_METRES = 36
_MAX_SIDE = 60


def _check_angle(ab: int, bc: int, ac: int, angle: int) -> None:
    cosine = (ab * ab + bc * bc - ac * ac) / (2 * ab * bc)
    check(abs(cosine - math.cos(math.radians(angle))) < 1e-9, "base triangle and angle disagree")


@generator("exd_x_sc2_similar_sas", level=9, presets={"exam": dict(tier="exam")})
def exd_x_sc2_similar_sas(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: the side that makes PQR similar to ABC by SAS, then the third side, then the perimeter."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(2000, "two scales of one base triangle that fit"):
        ab0, bc0, ac0, angle = rng.choice(_BASES)
        x, y = rng.randint(1, 10), rng.randint(1, 10)
        if x == y:
            continue
        big = max(ab0, bc0, ac0)
        if big * max(x, y) > _MAX_SIDE:
            continue
        break
    _check_angle(ab0, bc0, ac0, angle)
    ab, bc, ac = ab0 * x, bc0 * x, ac0 * x
    pq, qr, pr = ab0 * y, bc0 * y, ac0 * y
    ratio = Fraction(y, x)
    check(Fraction(pq, ab) == Fraction(qr, bc) == Fraction(pr, ac) == ratio, "ratios disagree")
    perimeter = pq + qr + pr

    given_first = rng.random() < 0.5  # which side beside the angle is given in triangle PQR
    scene, unit = rng.choice(_SCENES)
    if unit == "m" and max(ab, bc, ac, pq, qr, pr) > _MAX_METRES:
        scene, unit = "two triangular pieces of timber framing", "cm"
    if given_first:
        given = f"PQ = {pq} {unit}"
        ask = "QR"
        unknown_side = qr
    else:
        given = f"QR = {qr} {unit}"
        ask = "PQ"
        unknown_side = pq
    stem = (
        f"A designer is comparing {scene}. In triangle ABC, AB = {ab} {unit}, BC = {bc} {unit} and AC = {ac} {unit}, "
        f"and angle B = {angle}°. In triangle PQR, angle Q = {angle}° and {given}. The vertices P, Q and R correspond "
        "to A, B and C."
    )

    # Candidate errors, each computed from the same stem.
    given_value = pq if given_first else qr
    own = ab if given_first else bc  # the matching side of ABC
    other = bc if given_first else ab  # the side being asked for, in ABC
    a_cands = [
        Distractor(str(other + (given_value - own)), "additive-scaling", Fraction(other + (given_value - own))),
        Distractor(str(other), "assumed-congruent-copy", Fraction(other)),
        Distractor(
            str(Fraction(other * own, given_value)) if (other * own) % given_value == 0 else "0",
            "ratio-inverted",
            Fraction(other * own, given_value),
        ),
        Distractor(str(ac), None, Fraction(ac)),
    ]
    b_cands = [
        Distractor(str(ac + (given_value - own)), "additive-scaling", Fraction(ac + (given_value - own))),
        Distractor(str(ac), "assumed-congruent-copy", Fraction(ac)),
        Distractor(str(unknown_side), None, Fraction(unknown_side)),
        Distractor(str(pr + 1), None, Fraction(pr + 1)),
    ]
    c_cands = [
        Distractor(
            str(given_value + unknown_side + ac),
            "used-unscaled-side-in-perimeter",
            Fraction(given_value + unknown_side + ac),
        ),
        Distractor(str(ab + bc + ac), "gave-perimeter-of-original", Fraction(ab + bc + ac)),
        Distractor(
            str(given_value + unknown_side), "left-out-a-side-of-perimeter", Fraction(given_value + unknown_side)
        ),
        Distractor(str(perimeter + 1), None, Fraction(perimeter + 1)),
    ]
    a_cands = [c for c in a_cands if c.text != "0"]
    a_dis = pick_distractors(str(unknown_side), a_cands, 3)
    b_dis = pick_distractors(str(pr), b_cands, 3)
    c_dis = pick_distractors(str(perimeter), c_cands, 3)
    check(len(a_dis) >= 2 and len(b_dis) >= 2 and len(c_dis) >= 2, "too few distractors")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"For triangle PQR to be similar to triangle ABC by the SAS similarity test, how long must "
                f"{ask} be, in {unit}?",
                answer_kind="number",
                answer=unknown_side,
                answer_text=str(unknown_side),
                answer_latex=latex_answer(str(unknown_side)),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"SAS similarity needs the sides on either side of the {angle}° angle in the same ratio.",
                    f"{'PQ : AB' if given_first else 'QR : BC'} = {given_value} : {own}, so the ratio is {ratio}.",
                    f"{ask} = {other} x {ratio} = {unknown_side} {unit}.",
                ],
                distractors=a_dis,
            ),
            dict(
                prompt=f"With {ask} as in part (a), find the length of PR, in {unit}.",
                answer_kind="number",
                answer=pr,
                answer_text=str(pr),
                answer_latex=latex_answer(str(pr)),
                marks=2,
                sensible_range=(0, None),
                follow_through=(f"{ac}*part_a/{other}"),
                working=[f"PR : AC is also {ratio}, so PR = {ac} x {ratio} = {pr} {unit}."],
                distractors=b_dis,
            ),
            dict(
                prompt=f"Find the perimeter of triangle PQR, in {unit}.",
                answer_kind="number",
                answer=perimeter,
                answer_text=str(perimeter),
                answer_latex=latex_answer(str(perimeter)),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"{given_value} + part_a + part_b",
                working=[f"{pq} + {qr} + {pr} = {perimeter} {unit}."],
                distractors=c_dis,
            ),
        ],
    )
