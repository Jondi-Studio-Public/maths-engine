"""Exam source of `senior-represent-a-scalar-multiple-of-a-vector`: find k, then the multiplier, then a squared length.

The practice tiers scale a vector by a given scalar, or undo that scaling. An exam question hides the scalar: the
vector k u - w (or w - k u) is parallel to e, so it is a scalar multiple m e of e, and the learner works through three
linked answers:

(a) the scalar k that makes the vector parallel to e (the 2 by 2 determinant of the vector and e is zero);
(b) the multiplier m for which the vector equals m e;
(c) the squared length of the vector, which is m^2 |e|^2.

Parts (b) and (c) follow through from the earlier answers. Built backwards from u, e, k and m, with w = k u - m e (or
w = k u + m e), so every answer is a whole number. The independent check solves the parallel condition for k with
SymPy and recomputes the multiplier and the squared length from the vector itself.
"""

from __future__ import annotations

import random

from sympy import Matrix, Symbol, solve

from mathgen.core import attempts, check, generator, latex_answer, paren_negative

from .exd_x_r3d_common import tagged
from ._exd_distinct_parts import distinct_part_answers

_STEMS = (
    "Let u = {u}, w = {w} and e = {e}. For one value of k, the vector {expr} is parallel to e.",
    "The vectors are u = {u}, w = {w} and e = {e}. There is exactly one scalar k for which {expr} is parallel to e.",
    "Given u = {u}, w = {w} and e = {e}, the vector {expr} is parallel to e when k takes a particular value.",
)
_PROMPT_A = ("Find the scalar k.", "Determine the value of k.")
_PROMPT_B = (
    "Hence find the scalar m for which {expr} = m e.",
    "Using your value of k, find the scalar m such that {expr} = m e.",
)
_PROMPT_C = (
    "Using your value of m, find |{expr}|², the squared length of the vector.",
    "Hence find the squared length of the vector {expr}.",
)


def _pair(v) -> str:
    return f"({v[0]}, {v[1]})"


@generator("exd_x_r3d_vector_parallel_scalar", level=14, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3d_vector_parallel_scalar(rng: random.Random, tier: str = "exam") -> dict:
    """k, then m, then the squared length of a vector that is parallel to e."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    nonzero = [-4, -3, -2, -1, 1, 2, 3, 4]
    for _ in attempts(2000, "a parallel-vector draw"):
        u = (rng.choice(nonzero), rng.choice(nonzero))
        e = (rng.choice([-3, -2, -1, 1, 2, 3]), rng.choice([-3, -2, -1, 1, 2, 3]))
        if u[0] * e[1] - u[1] * e[0] == 0:
            continue
        k = rng.choice([-5, -4, -3, -2, 2, 3, 4, 5])
        m = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
        reverse = rng.random() < 0.4  # w - k u = m e instead of k u - w = m e
        sign = -1 if reverse else 1  # the vector is sign * (k u - w)
        # sign (k u - w) = m e  =>  w = k u - sign m e
        w = (k * u[0] - sign * m * e[0], k * u[1] - sign * m * e[1])
        if w == (0, 0) or max(abs(w[0]), abs(w[1])) > 30:
            continue
        expr = "w - ku" if reverse else "ku - w"
        length2 = m * m * (e[0] ** 2 + e[1] ** 2)
        # Independent route: solve the parallel condition for k, then read m and the length off the vector.
        ks = Symbol("k")
        vec = Matrix([sign * (ks * u[0] - w[0]), sign * (ks * u[1] - w[1])])
        roots = solve(Matrix.hstack(vec, Matrix(e)).det(), ks)
        check(roots == [k], "the parallel condition must have the single root k")
        v = [sign * (k * u[i] - w[i]) for i in range(2)]
        check(v[0] * e[1] == v[1] * e[0] and v[0] == m * e[0] and v[1] == m * e[1], "the vector must equal m e")
        check(v[0] ** 2 + v[1] ** 2 == length2, "the squared length disagrees")
        a_cands = [(-k, "sign-error-moving-term"), (m, "scalars-k-and-m-confused")]
        for num, den in ((e[0] + w[0], u[0]), (e[1] + w[1], u[1])):
            a_cands.append((num // den if num % den == 0 else None, "parallel-taken-as-equal"))
        for num, den in ((w[0], u[0]), (w[1], u[1])):
            a_cands.append((num // den if num % den == 0 else None, "parallel-condition-ignored"))
        other = sign * (k * u[1] - w[1])
        b_cands = [
            (k, "scalars-k-and-m-confused"),
            (-m, "reversed-subtraction"),
            (other // e[0] if other % e[0] == 0 else None, "used-other-component-of-vector"),
        ]
        c_cands = [
            (m * (e[0] ** 2 + e[1] ** 2), "scalar-not-squared-in-length"),
            (m * m, "direction-vector-length-ignored"),
            (length2 // (m * m) if m * m else None, "scalar-multiplier-ignored"),
            (m * m * (e[0] + e[1]) ** 2, "square-of-sum-for-sum-of-squares"),
        ]
        a_d = tagged(str(k), a_cands)
        b_d = tagged(str(m), [c for c in b_cands if c])
        c_d = tagged(str(length2), c_cands)
        if len(a_d) >= 3 and len(b_d) >= 2 and len(c_d) >= 3:
            break

    stem = rng.choice(_STEMS).format(u=_pair(u), w=_pair(w), e=_pair(e), expr=expr)
    b_ft = f"({sign}*(part_a*{u[0]} - ({w[0]})))/({e[0]})"
    c_ft = f"part_b**2 * {e[0] ** 2 + e[1] ** 2}"
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=rng.choice(_PROMPT_A),
                answer_kind="number",
                answer=k,
                answer_text=str(k),
                answer_latex=latex_answer(k),
                marks=2,
                working=[
                    f"The vector {expr} must be a multiple of e: its 2 by 2 determinant with e is zero, giving k = {k}."
                ],
                distractors=a_d,
            ),
            dict(
                prompt=rng.choice(_PROMPT_B).format(expr=expr),
                answer_kind="number",
                answer=m,
                answer_text=str(m),
                answer_latex=latex_answer(m),
                marks=1,
                follow_through=b_ft,
                working=[f"{expr} = {_pair(v)} = {m}·{_pair(e)}, so m = {m}."],
                distractors=b_d,
            ),
            dict(
                prompt=rng.choice(_PROMPT_C).format(expr=expr),
                answer_kind="number",
                answer=length2,
                answer_text=str(length2),
                answer_latex=latex_answer(length2),
                marks=2,
                follow_through=c_ft,
                sensible_range=(0, None),
                working=[f"|{expr}|² = m²|e|² = {paren_negative(m)}²·{e[0] ** 2 + e[1] ** 2} = {length2}."],
                distractors=c_d,
            ),
        ],
    )
