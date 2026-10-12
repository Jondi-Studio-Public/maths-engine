"""Exam source of `vector-dot-product`: a dot product, a length, and the scalar projection they give.

The practice tiers dot two vectors written as components. An exam question gives the three vertices of a triangle,
so the learner first forms the vectors, and then uses the dot product to resolve one side along another:

(a) AB . AC from the coordinates of A, B and C;
(b) the length |AC|;
(c) the scalar projection of AB onto AC, which is (AB . AC) / |AC| and so follows through from (a) and (b);
(d) the square of the shortest distance from B to the line AC, which is |AB|^2 less the square of the projection
    (Pythagoras on the right triangle that AB makes with the line), and so follows through from (c).

Built backwards. AC is an integer vector whose length is a whole number (from a Pythagorean quadruple, with its
components permuted and signed). AB = m u + w, where u is the primitive integer vector along AC (so AC = g u), w is
perpendicular to u (a cross product of u with a small vector), and |u| is a whole number. Then the scalar projection is
the whole number m |u|, and the squared distance from B to the line is |w|^2. The independent check forms the
vectors from the printed coordinates and recomputes the three answers with exact arithmetic.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from mathgen.core import attempts, check, generator, latex_answer

from .exd_x_r3d_common import tagged
from ._exd_r3_distinct import distinct_part_answers

#: Primitive integer vectors (up to order and sign) whose length is a whole number.
_BASES = (
    (1, 2, 2),
    (2, 3, 6),
    (1, 4, 8),
    (4, 4, 7),
    (2, 6, 9),
    (6, 6, 7),
    (3, 4, 12),
    (2, 5, 14),
    (2, 10, 11),
    (0, 3, 4),
    (0, 5, 12),
    (0, 8, 15),
)

_STEMS = (
    "In triangle ABC the vertices are A{a}, B{b} and C{c}.",
    "Triangle ABC has vertices A{a}, B{b} and C{c}.",
    "The points A{a}, B{b} and C{c} are the vertices of a triangle.",
    "A surveyor records three points in space: A{a}, B{b} and C{c}.",
)
_PROMPT_A = ("Find the dot product AB · AC.", "Calculate AB · AC.")
_PROMPT_B = ("Find the length of AC.", "Calculate |AC|.")
_PROMPT_D = (
    "Hence find the square of the shortest distance from B to the line AC.",
    "Using your answer to part (c), find the square of the perpendicular distance from B to the line AC.",
)
_PROMPT_C = (
    "Hence find the scalar projection of AB onto AC, the signed length of the component of AB along AC.",
    "Using parts (a) and (b), find the scalar projection of AB onto AC (the signed length of AB measured along AC).",
)


def _dot(u, v) -> int:
    return sum(x * y for x, y in zip(u, v))


def _cross(u, v):
    return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])


def _sub(p, q):
    return tuple(x - y for x, y in zip(p, q))


def _fmt(p) -> str:
    return "(" + ", ".join(str(x) for x in p) + ")"


@generator("exd_x_r3d_vector_projection", level=16, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3d_vector_projection(rng: random.Random, tier: str = "exam") -> dict:
    """AB . AC, |AC| and the scalar projection of AB onto AC for a triangle given by its vertices."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(2000, "a triangle with small coordinates"):
        base = list(rng.choice(_BASES))
        rng.shuffle(base)
        u0 = tuple(x * rng.choice([-1, 1]) for x in base)
        len0 = math.isqrt(_dot(u0, u0))
        scale = rng.choice([1, 1, 2])
        ac = tuple(scale * x for x in u0)
        length = scale * len0
        probe = tuple(rng.randint(-2, 2) for _ in range(3))
        w = _cross(u0, probe)
        if w == (0, 0, 0):
            continue
        g = math.gcd(math.gcd(abs(w[0]), abs(w[1])), abs(w[2]))
        w = tuple(x // g for x in w)
        m = rng.choice([-3, -2, -1, 1, 2, 3])
        ab = tuple(m * x + y for x, y in zip(u0, w))
        a = tuple(rng.randint(-5, 5) for _ in range(3))
        b = tuple(x + y for x, y in zip(a, ab))
        c = tuple(x + y for x, y in zip(a, ac))
        if max(abs(x) for x in (*b, *c)) > 22:
            continue
        dot = _dot(ab, ac)
        proj = m * len0
        bc = _sub(c, b)
        d_cands = [
            (-dot, "reversed-vector-direction"),
            (_dot(b, c), "position-vectors-dotted-for-displacement"),
            (_dot(ab, bc), "wrong-vector-pair-dotted"),
            (_dot(ab, ab), "wrong-vector-pair-dotted"),
        ]
        ab_len2 = _dot(ab, ab)
        l_cands = [
            (length * length, "magnitude-without-square-root"),
            (sum(abs(x) for x in ac), "magnitude-as-sum-of-components"),
            (math.isqrt(ab_len2) if math.isqrt(ab_len2) ** 2 == ab_len2 else None, "length-of-wrong-vector"),
        ]
        ab2 = _dot(ab, ab)
        perp2 = ab2 - proj * proj
        check(perp2 == _dot(w, w), "the squared perpendicular distance must be |w|^2")
        q_cands = [
            (ab2, "perpendicular-distance-taken-as-hypotenuse"),
            (ab2 - proj, "projection-not-squared"),
            (proj * proj, "projection-squared-for-perpendicular-distance"),
            (ab2 + proj * proj, "pythagoras-added-for-leg"),
        ]
        p_cands = [
            (dot, "projection-division-by-length-omitted"),
            (
                dot // (length * length) if dot % (length * length) == 0 else None,
                "vector-coefficient-for-scalar-projection",
            ),
            (-proj, "projection-sign-lost"),
        ]
        a_d = tagged(str(dot), d_cands)
        b_d = tagged(str(length), l_cands)
        c_d = tagged(str(proj), p_cands)
        d_d = tagged(str(perp2), q_cands)
        if len(a_d) >= 3 and len(b_d) >= 2 and len(c_d) >= 2 and len(d_d) >= 3:
            break
    # Independent route: from the printed coordinates, exactly.
    ab_chk, ac_chk = _sub(b, a), _sub(c, a)
    check(Fraction(_dot(ab_chk, ac_chk)) == dot, "AB . AC disagrees with the coordinates")
    check(math.isqrt(_dot(ac_chk, ac_chk)) ** 2 == _dot(ac_chk, ac_chk), "|AC| must be a whole number")
    check(Fraction(dot, length) == proj, "scalar projection disagrees")

    stem = rng.choice(_STEMS).format(a=_fmt(a), b=_fmt(b), c=_fmt(c))
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=rng.choice(_PROMPT_A),
                answer_kind="number",
                answer=dot,
                answer_text=str(dot),
                answer_latex=latex_answer(dot),
                marks=2,
                working=[f"AB = {_fmt(ab)}, AC = {_fmt(ac)}; AB · AC = {dot}."],
                distractors=a_d,
            ),
            dict(
                prompt=rng.choice(_PROMPT_B),
                answer_kind="number",
                answer=length,
                answer_text=str(length),
                answer_latex=latex_answer(length),
                marks=1,
                sensible_range=(0, None),
                working=[f"|AC| = √({_dot(ac, ac)}) = {length}."],
                distractors=b_d,
            ),
            dict(
                prompt=rng.choice(_PROMPT_C),
                answer_kind="number",
                answer=proj,
                answer_text=str(proj),
                answer_latex=latex_answer(proj),
                marks=2,
                follow_through="part_a / part_b",
                working=[f"Scalar projection = (AB · AC) / |AC| = {dot} / {length} = {proj}."],
                distractors=c_d,
            ),
            dict(
                prompt=rng.choice(_PROMPT_D),
                answer_kind="number",
                answer=perp2,
                answer_text=str(perp2),
                answer_latex=latex_answer(perp2),
                marks=2,
                follow_through=f"{ab2} - part_c**2",
                sensible_range=(0, None),
                working=[f"|AB|² = {ab2}; the squared distance is |AB|² - {proj}² = {perp2}."],
                distractors=d_d,
            ),
        ],
    )
