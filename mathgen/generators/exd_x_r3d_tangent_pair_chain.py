"""Exam source of `senior-the-alternate-segment-theorem`: a tangent pair, solved through four linked angles.

Tangents from an external point touch a circle at A and B, and Q is a point on the major arc AB. The angle between
a tangent and the chord AB equals the angle AQB in the alternate segment, and both are given as expressions in an
unknown x. The learner finds

(a) x, from the two expressions for equal angles;
(b) the angle AQB itself, which is also the angle between each tangent and the chord;
(c) the angle between the tangents at the external point (the two tangents are equal, so triangle PAB is isosceles);
(d) the angle OAB between the radius OA and the chord AB (the radius meets the tangent at a right angle).

Parts (b), (c) and (d) follow through from the earlier answers: (b) puts x into the expression, (c) is 180 - 2 b and
(d) is 90 - b. The unknown, the expressions and the angle are drawn so every answer is a whole number of degrees.
An independent check solves the equation with SymPy and recomputes the other angles from the triangle sum.
"""

from __future__ import annotations

import random

from sympy import Symbol, solve

from mathgen.core import attempts, check, generator, join_terms, latex_answer

from .exd_x_r3d_common import tagged
from ._exd_r3_distinct import distinct_part_answers

_LETTERS = (("P", "A", "B", "Q"), ("T", "M", "N", "R"), ("E", "C", "D", "F"))
_STEMS = (
    "Tangents {p}{a} and {p}{b} are drawn from an external point {p} to a circle with centre O, touching it at {a} and {b}. "
    "{q} is a point on the major arc {a}{b}. Angle {p}{a}{b} = ({e1}) degrees and angle {a}{q}{b} = ({e2}) degrees.",
    "From a point {p} outside a circle with centre O, the tangents {p}{a} and {p}{b} touch the circle at {a} and {b}. "
    "A point {q} lies on the major arc {a}{b}. The angle between the tangent {p}{a} and the chord {a}{b} is ({e1}) degrees, "
    "and angle {a}{q}{b} is ({e2}) degrees.",
    "The tangents at {a} and {b} to a circle with centre O meet at {p}. {q} is on the major arc {a}{b}. "
    "Angle {p}{a}{b} is ({e1}) degrees and angle {a}{q}{b} is ({e2}) degrees.",
)
_PROMPT_A = ("Find the value of x.", "Determine x.")
_PROMPT_B = ("Hence find angle {a}{q}{b}, in degrees.", "Using your value of x, find angle {a}{q}{b}, in degrees.")
_PROMPT_C = ("Find angle {a}{p}{b}, in degrees.", "Hence find the angle {a}{p}{b} between the tangents, in degrees.")
_PROMPT_D = ("Find angle O{a}{b}, in degrees.", "Hence find angle O{a}{b}, in degrees.")


@generator("exd_x_r3d_tangent_pair_chain", level=14, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3d_tangent_pair_chain(rng: random.Random, tier: str = "exam") -> dict:
    """x from the alternate segment theorem, then the angles AQB, APB and OAB."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(3000, "angle expressions with whole-number answers"):
        x = rng.randint(3, 24)
        theta = rng.randint(22, 78)
        p, r = rng.sample(range(1, 7), 2)
        q, s = theta - p * x, theta - r * x
        if not (0 < abs(q) <= 70 and 0 < abs(s) <= 70):
            continue
        # Candidate wrong answers that are whole numbers.
        sum_x = (180 - q - s) / (p + r)
        comp_x = (90 - q - s) / (p + r)
        a_cands = [
            (-x, "sign-error-moving-term"),
            (s - q, "forgot-to-divide-by-coefficient"),
            (theta, "reports-the-angle-instead-of-x"),
            (int(sum_x) if sum_x == int(sum_x) and sum_x > 0 else None, "tangent-chord-angle-supplementary"),
            (int(comp_x) if comp_x == int(comp_x) and comp_x > 0 else None, "tangent-chord-angle-complementary"),
        ]
        b_cands = [
            (180 - theta, "tangent-chord-angle-supplementary"),
            (90 - theta, "tangent-chord-angle-complementary"),
            (2 * theta, "inscribed-angle-doubled"),
        ]
        c_cands = [
            (theta, "tangent-angles-all-equal"),
            (180 - theta, "isosceles-base-angle-not-doubled"),
            (2 * theta, "inscribed-angle-doubled"),
            (90 - theta, "tangent-chord-angle-complementary"),
        ]
        d_cands = [
            (theta, "radius-tangent-right-angle-ignored"),
            (90 + theta, "radius-tangent-right-angle-added"),
            (180 - 2 * theta, "tangent-pair-angle-given-for-radius-angle"),
            (2 * theta, "inscribed-angle-doubled"),
        ]
        a_d = tagged(str(x), [c for c in a_cands if c[0] is not None])
        b_d = tagged(str(theta), b_cands)
        c_d = tagged(str(180 - 2 * theta), c_cands)
        d_d = tagged(str(90 - theta), d_cands)
        if len(a_d) == 3 and len(b_d) == 3 and len(c_d) >= 3 and len(d_d) >= 3:
            break
    # Independent route: solve the equation, then the angles from the triangle sums.
    xs = Symbol("x")
    check(solve(p * xs + q - (r * xs + s), xs) == [x], "the equation must have the single root x")
    check(p * x + q == theta == r * x + s, "both expressions must give the angle")
    apb = 180 - theta - theta  # triangle PAB: PA = PB, base angles equal the tangent-chord angle
    oab = 90 - theta  # triangle OAB is isosceles with apex angle 2 theta
    check(apb + 2 * theta == 180 and (180 - 2 * theta) / 2 == oab, "triangle sums disagree")

    pl, al, bl, ql = rng.choice(_LETTERS)
    e1 = join_terms([(p, "x"), (q, "")])
    e2 = join_terms([(r, "x"), (s, "")])
    stem = rng.choice(_STEMS).format(p=pl, a=al, b=bl, q=ql, e1=e1, e2=e2)
    names = dict(p=pl, a=al, b=bl, q=ql)
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=rng.choice(_PROMPT_A),
                answer_kind="number",
                answer=x,
                answer_text=str(x),
                answer_latex=latex_answer(x),
                marks=2,
                sensible_range=(0, None),
                working=[f"Alternate segment theorem: {e1} = {e2}, so x = {x}."],
                distractors=a_d,
            ),
            dict(
                prompt=rng.choice(_PROMPT_B).format(**names),
                answer_kind="number",
                answer=theta,
                answer_text=str(theta),
                answer_latex=latex_answer(theta),
                marks=1,
                follow_through=f"{r}*part_a + ({s})",
                sensible_range=(0, 90),
                working=[f"{e2} = {theta}°."],
                distractors=b_d,
            ),
            dict(
                prompt=rng.choice(_PROMPT_C).format(**names),
                answer_kind="number",
                answer=apb,
                answer_text=str(apb),
                answer_latex=latex_answer(apb),
                marks=2,
                follow_through="180 - 2*part_b",
                sensible_range=(0, 180),
                working=[
                    f"Triangle {pl}{al}{bl} is isosceles with base angles {theta}°, so the angle at {pl} is {apb}°."
                ],
                distractors=c_d,
            ),
            dict(
                prompt=rng.choice(_PROMPT_D).format(**names),
                answer_kind="number",
                answer=oab,
                answer_text=str(oab),
                answer_latex=latex_answer(oab),
                marks=2,
                follow_through="90 - part_b",
                sensible_range=(0, 90),
                working=[
                    f"The radius O{al} is perpendicular to the tangent {pl}{al}, so angle O{al}{bl} = 90° - {theta}° = {oab}°."
                ],
                distractors=d_d,
            ),
        ],
    )
