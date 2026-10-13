"""Exam source of `matrix-determinant-2x2`: the product rule for determinants used to recover an unknown entry.

The practice tiers evaluate one determinant or the determinant of a product of two numeric matrices. An exam
question gives the determinant of a PRODUCT, one factor with an unknown entry, and works backwards:

(a) det(A) for the numeric matrix A;
(b) det(B), from det(AB) = det(A) det(B) and the given det(AB);
(c) the unknown entry x of B, from the formula for det(B);
(d) det(c A^(-1) B), which needs det(cM) = c^2 det(M) and det(A^(-1)) = 1/det(A).

Parts (b) to (d) follow through from the earlier parts.

## Construction

Built backwards from integer matrices. det(A) is drawn (never 0 or +-1) and B is drawn with an integer x such that
det(B) is a non-zero multiple of det(A), so the answer to (d) is a whole number. An independent route rebuilds the
matrices with SymPy and recomputes every part.
"""

from __future__ import annotations

from sympy import Matrix, Symbol, latex, solve

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, latex_answer, pick_distractors
from ._exd_distinct_parts import distinct_part_answers

_X = Symbol("x")


def _mat_text(rows) -> str:
    return "[" + "; ".join(" ".join(str(v) for v in row) for row in rows) + "]"


def _ints(values, tag_pairs):
    out = []
    for v, tag in tag_pairs:
        out.append(Distractor(str(v), tag, int(v)))
    return out


def _part(prompt, key, marks, distractors, working, follow_through=None):
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=key,
        answer_text=str(key),
        answer_latex=latex_answer(str(key)),
        marks=marks,
        working=working,
        distractors=distractors,
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


def _build(rng):
    for _ in attempts(2000, "matrices with an integer unknown and a whole-number final answer"):
        a1, a2, a3, a4 = (rng.randint(-6, 6) for _ in range(4))
        det_a = a1 * a4 - a2 * a3
        if abs(det_a) < 2:
            continue
        b2, b3 = rng.randint(-6, 6), rng.randint(-6, 6)
        b4 = rng.choice([v for v in range(-6, 7) if abs(v) >= 2])
        x = rng.choice([v for v in range(-9, 10) if v != 0])
        det_b = x * b4 - b2 * b3
        if det_b == 0 or det_b % det_a != 0:
            continue
        m = det_b // det_a
        if abs(m) < 2 and rng.random() < 0.7:
            continue
        c = rng.choice([2, 3])
        det_ab = det_a * det_b
        keys = {det_a, det_b, x, c * c * m}
        shown = {a1, a2, a3, a4, b2, b3, b4, det_ab, c}
        if len(keys) < 4 or keys & shown or abs(det_ab) > 3000:
            continue
        break

    # Independent route with SymPy.
    big_a = Matrix([[a1, a2], [a3, a4]])
    big_b = Matrix([[_X, b2], [b3, b4]])
    solved = solve((big_a * big_b).det() - det_ab, _X)
    check(solved == [x], "SymPy finds a different x")
    check(big_a.det() == det_a and big_b.subs(_X, x).det() == det_b, "determinants disagree")
    check((c * big_a.inv() * big_b.subs(_X, x)).det() == c * c * m, "det(cA^-1 B) disagrees")

    a_d = pick_distractors(
        str(det_a),
        _ints(
            None,
            [
                (a1 * a4 + a2 * a3, "determinant-added-not-subtracted"),
                (a1 * a3 - a2 * a4, "determinant-columns-not-diagonals"),
                (-det_a, "determinant-sign-reversed"),
                (a1 * a4, "determinant-one-diagonal-only"),
            ],
        ),
        3,
    )
    b_d = pick_distractors(
        str(det_b),
        _ints(
            None,
            [
                (det_ab - det_a, "determinants-summed-for-product"),
                (det_ab * det_a, "determinant-product-rule-multiplied-not-divided"),
                (det_ab, "second-matrix-ignored"),
                (-det_b, "determinant-sign-reversed"),
            ],
        ),
        3,
    )
    x_cands = [
        (det_b + b2 * b3, "forgot-to-divide-by-coefficient"),
        (det_b - b2 * b3, "forgot-to-divide-by-coefficient"),
        (det_b, "ignored-constant-term"),
        (-x, "sign-error-moving-term"),
    ]
    x_d = pick_distractors(str(x), _ints(None, x_cands), 3)
    d_cands = [
        (c * m, "scalar-multiple-not-squared"),
        (m, "scalar-multiple-ignored"),
        (c * c * det_a * det_b, "inverse-determinant-not-reciprocal"),
        (c * c * det_b, "inverse-ignored-in-determinant"),
    ]
    d_d = pick_distractors(str(c * c * m), _ints(None, d_cands), 3)
    if min(len(a_d), len(b_d), len(x_d), len(d_d)) < 2:
        return None

    mat_a, mat_b = _mat_text([[a1, a2], [a3, a4]]), _mat_text([["x", b2], [b3, b4]])
    tex_a, tex_b = latex(big_a), latex(big_b)
    variant = rng.randrange(3)
    if variant == 0:
        text = f"A = {mat_a} and B = {mat_b}, where x is a constant. det(AB) = {det_ab}."
        tex = rf"A = {tex_a}\ \text{{and}}\ B = {tex_b}\text{{, where }}x\text{{ is a constant. }}\det(AB) = {det_ab}\text{{.}}"
    elif variant == 1:
        text = f"Matrices A = {mat_a} and B = {mat_b} satisfy det(AB) = {det_ab}, where x is a constant."
        tex = rf"\text{{Matrices }}A = {tex_a}\ \text{{and}}\ B = {tex_b}\text{{ satisfy }}\det(AB) = {det_ab}\text{{, where }}x\text{{ is a constant.}}"
    else:
        text = f"The matrix product AB has determinant {det_ab}, where A = {mat_a}, B = {mat_b} and x is a constant."
        tex = rf"\text{{The matrix product }}AB\text{{ has determinant }}{det_ab}\text{{, where }}A = {tex_a},\ B = {tex_b}\text{{ and }}x\text{{ is a constant.}}"
    last = rng.choice([f"det({c}A⁻¹B)", f"det({c}BA⁻¹)"])
    return dict(
        text=text,
        latex=tex,
        parts=[
            _part(
                "Find det(A).",
                det_a,
                1,
                a_d,
                [f"det(A) = {a1} × {a4} - {a2} × {a3} = {det_a}."],
            ),
            _part(
                "Using your answer to (a), find det(B).",
                det_b,
                2,
                b_d,
                [f"det(AB) = det(A) × det(B), so det(B) = {det_ab} ÷ {det_a} = {det_b}."],
                follow_through=f"{det_ab}/part_a",
            ),
            _part(
                "Using your answer to (b), find the value of x.",
                x,
                2,
                x_d,
                [f"det(B) = {b4}x - ({b2})({b3}) = {det_b}, so x = {x}."],
                follow_through=f"(part_b + ({b2})*({b3}))/{b4}",
            ),
            _part(
                f"Using your answers to (a) and (b), find {last}.",
                c * c * m,
                3,
                d_d,
                [
                    f"det({c}M) = {c}² det(M) for a 2 × 2 matrix, and det(A⁻¹) = 1 ÷ det(A).",
                    f"= {c * c} × {det_b} ÷ {det_a} = {c * c * m}.",
                ],
                follow_through=f"{c * c}*part_b/part_a",
            ),
        ],
    )


@generator("exd_x_r3a_determinant_chain", level=12, presets={"exam": {"tier": "exam"}})
@distinct_part_answers
def exd_x_r3a_determinant_chain(rng, tier: str = "exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(200, "a draw with enough distractors in every part"):
        built = _build(rng)
        if built is not None:
            return built
