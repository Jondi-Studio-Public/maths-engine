"""Exam-tier source for `senior-the-angle-at-the-centre-subtended-by-an`: a chain of angles in a circle.

A, B and C lie on a circle with centre O, with O inside triangle ABC. The stem gives angle OAB = a and a second
base angle (OCB or OBC, equal in size because OB = OC) = c, with a + c < 90 degrees. Then, with s = a + c:

(a) angle ABC = s                              two isosceles triangles (OA = OB, OB = OC)   2 marks
(b) angle AOC = 2s                             angle at the centre                          1 mark   2 * part_a
(c) angle OAC = (180 - AOC) / 2                isosceles triangle OAC                       1 mark   (180 - part_b) / 2
(d) angle BAC = OAB + OAC = 90 - c             1 mark                                       a + part_c

a + c < 90 makes triangle ABC acute, so O is inside it and B is on the major arc AC. The key is checked in the
generator against coordinates (B on the unit circle, A and C placed by their central angles).
"""

from __future__ import annotations

import math

from ..answers import Distractor
from ..core import check, generator, phrase, pick_distractors
from ._exd_r3_distinct import distinct_part_answers


def _options(key: int, candidates, n: int = 3):
    out, seen = [], {key}
    for tag, value in candidates:
        if value != int(value) or value in seen or not 0 < value < 400:
            continue
        value = int(value)
        seen.add(value)
        out.append(Distractor(str(value), tag, value=value))
    return pick_distractors(str(key), out, n)


def _angle(p, q, r) -> float:
    """Angle at q in degrees between q->p and q->r."""
    v1, v2 = (p[0] - q[0], p[1] - q[1]), (r[0] - q[0], r[1] - q[1])
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (math.hypot(*v1) * math.hypot(*v2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


@generator("exd_x_r3e_circle_centre_chain", level=13, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3e_circle_centre_chain(rng, tier: str = "exam") -> dict:
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in range(2000):
        a, c = rng.randint(8, 70), rng.randint(8, 70)
        s = a + c
        if s >= 88:
            continue
        aoc, oac, bac = 2 * s, 90 - s, 90 - c
        a_opts = _options(
            s,
            [
                ("subtracted-base-angles-from-180", 180 - s),
                ("centre-and-circumference-angles-confused", 2 * s),
                ("angle-subtracted-not-added", abs(a - c)),
                ("subtracted-base-angles-from-180", 180 - 2 * s),
            ],
        )
        b_opts = _options(
            aoc,
            [
                ("forgot-doubling-at-centre", s),
                ("reflex-angle-at-centre", 360 - aoc),
                ("subtracted-base-angles-from-180", 180 - s),
            ],
        )
        c_opts = _options(
            oac,
            [
                ("isosceles-base-angle-not-halved", 180 - aoc),
                ("centre-and-circumference-angles-confused", s),
                ("subtracted-base-angles-from-180", (180 - s) / 2),
            ],
        )
        d_opts = _options(
            bac,
            [
                ("angle-subtracted-not-added", a - oac),
                ("gave-part-of-angle", a),
                ("gave-part-of-angle", oac),
                ("angle-subtracted-not-added", 180 - a - oac),
            ],
        )
        if min(len(a_opts), len(b_opts), len(c_opts), len(d_opts)) >= 2:
            break
    else:
        check(False, "no angle pair with enough distractors")

    # Coordinates: B on the unit circle at angle 0; A and C placed from their central angles with B between them.
    theta_a, theta_c = math.radians(180 - 2 * a), -math.radians(180 - 2 * c)
    O, B = (0.0, 0.0), (1.0, 0.0)
    A, C = (math.cos(theta_a), math.sin(theta_a)), (math.cos(theta_c), math.sin(theta_c))
    for got, want in (
        (_angle(O, A, B), a),
        (_angle(O, C, B), c),
        (_angle(A, B, C), s),
        (_angle(A, O, C), aoc),
        (_angle(O, A, C), oac),
        (_angle(B, A, C), bac),
    ):
        check(abs(got - want) < 1e-6, "angle disagrees with the coordinates")

    second = rng.choice(["OCB", "OBC"])
    stem = phrase(
        rng,
        (
            "The points A, B and C lie on a circle with centre O, and O lies inside triangle ABC. "
            "Angle OAB is {a} degrees and angle {second} is {c} degrees.",
            "A, B and C are points on a circle with centre O. The centre O is inside triangle ABC, "
            "angle OAB = {a} degrees and angle {second} = {c} degrees.",
            "Triangle ABC is drawn inside a circle with centre O, with all three vertices on the circle and O "
            "inside the triangle. It is known that angle OAB is {a} degrees and that angle {second} is {c} degrees.",
        ),
        a=a,
        c=c,
        second=second,
    )
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find angle ABC, in degrees.",
                answer_kind="number",
                answer=s,
                answer_text=str(s),
                marks=2,
                working=[
                    f"OA = OB, so angle OBA = {a} degrees. OB = OC, so angle OBC = {c} degrees.",
                    f"Angle ABC = {a} + {c} = {s} degrees.",
                ],
                distractors=a_opts,
            ),
            dict(
                prompt="Hence find angle AOC, in degrees.",
                answer_kind="number",
                answer=aoc,
                answer_text=str(aoc),
                marks=1,
                working=[f"The angle at the centre is twice the angle at the circumference: 2 x {s} = {aoc} degrees."],
                distractors=b_opts,
                follow_through="2*part_a",
            ),
            dict(
                prompt="Find angle OAC, in degrees.",
                answer_kind="number",
                answer=oac,
                answer_text=str(oac),
                marks=1,
                working=[f"OA = OC, so angle OAC = (180 - {aoc})/2 = {oac} degrees."],
                distractors=c_opts,
                follow_through="(180 - part_b)/2",
            ),
            dict(
                prompt="Hence find angle BAC, in degrees.",
                answer_kind="number",
                answer=bac,
                answer_text=str(bac),
                marks=1,
                working=[f"Angle BAC = angle OAB + angle OAC = {a} + {oac} = {bac} degrees."],
                distractors=d_opts,
                follow_through=f"{a} + part_c",
            ),
        ],
    )
