"""Exam-tier source for `senior-define-and-use-multiplication-by-a-scalar-of`: combining scaled vectors, then
choosing a scalar.

p = p1 i + p2 j and q = q1 i + q2 j, and w = m p - n q. The chain:

(a) the i-coefficient of w                                  number, 1 mark    m p1 - n q1
(b) the j-coefficient of w                                  number, 1 mark    m p2 - n q2
(c) the k for which kp + w has no i-component               number, 2 marks   -part_a / p1
(d) the j-coefficient of that vector kp + w                 number, 1 mark    part_c p2 + part_b

Built backwards from the scalar k: p1, m, n and k are drawn first and q1 = (m + k) p1 / n must be a whole number, so
k = -w1/p1 is a whole number and every answer is an integer.
"""

from __future__ import annotations

from fractions import Fraction

from ..answers import Distractor
from ..core import check, generator, join_terms, pick_distractors
from ._exd_distinct_parts import distinct_part_answers

_NONZERO = [n for n in range(-6, 7) if n != 0]


def _vec(i: int, j: int) -> str:
    return join_terms([(i, "i"), (j, "j")])


def _scaled(m: int, name: str) -> str:
    return f"{m}{name}"


def _options(key: int, candidates, n: int = 3):
    out, seen = [], {key}
    for tag, value in candidates:
        value = Fraction(value)
        if value.denominator != 1 or int(value) in seen:
            continue
        seen.add(int(value))
        out.append(Distractor(str(int(value)), tag, value=int(value)))
    return pick_distractors(str(key), out, n)


@generator("exd_x_r3e_vector_combination", level=12, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3e_vector_combination(rng, tier: str = "exam") -> dict:
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in range(2000):
        p1, p2, q2 = rng.choice(_NONZERO), rng.choice(_NONZERO), rng.choice(_NONZERO)
        m, n = rng.randint(2, 5), rng.randint(2, 4)
        k = rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
        if m + k == 0 or ((m + k) * p1) % n:
            continue
        q1 = (m + k) * p1 // n
        if q1 == 0 or abs(q1) > 9:
            continue
        w1, w2 = m * p1 - n * q1, m * p2 - n * q2
        if w1 == 0 or w2 == 0 or w1 != -k * p1:
            continue
        d_key = k * p2 + w2
        if d_key == 0 or d_key == w2:
            continue
        a_opts = _options(
            w1,
            [
                ("combination-sign-slip", m * p1 + n * q1),
                ("scalar-applied-to-first-vector-only", m * p1 - q1),
                ("scalars-attached-to-wrong-vectors", n * p1 - m * q1),
                ("used-other-component-of-vector", w2),
            ],
        )
        b_opts = _options(
            w2,
            [
                ("combination-sign-slip", m * p2 + n * q2),
                ("scalar-applied-to-first-vector-only", m * p2 - q2),
                ("scalars-attached-to-wrong-vectors", n * p2 - m * q2),
                ("used-other-component-of-vector", w1),
            ],
        )
        c_opts = _options(
            k,
            [
                ("sign-error-moving-term", -k),
                ("forgot-to-divide-by-coefficient", -w1),
                ("used-other-component-of-vector", Fraction(-w2, p2)),
                ("scalars-attached-to-wrong-vectors", Fraction(-w1, q1)),
            ],
        )
        d_opts = _options(
            d_key,
            [
                ("scalar-left-off-component", w2),
                ("scalar-added-not-multiplied", k + p2 + w2),
                ("combination-sign-slip", k * p2 - w2),
                ("second-vector-left-out-of-sum", k * p2),
            ],
        )
        if min(len(a_opts), len(b_opts), len(c_opts), len(d_opts)) >= 3:
            break
    else:
        check(False, "no vector combination with enough distractors")

    check(w1 == m * p1 - n * q1 and w2 == m * p2 - n * q2, "w disagrees with its definition")
    check(k * p1 + w1 == 0, "kp + w has an i-component")
    stem = f"The vectors p and q are p = {_vec(p1, p2)} and q = {_vec(q1, q2)}. The vector w = {m}p - {n}q."
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the coefficient of i in w.",
                answer_kind="number",
                answer=w1,
                answer_text=str(w1),
                marks=1,
                working=[
                    f"w = {m}({_vec(p1, p2)}) - {n}({_vec(q1, q2)}); the i-coefficient is {m}({p1}) - {n}({q1}) = {w1}."
                ],
                distractors=a_opts,
            ),
            dict(
                prompt="Find the coefficient of j in w.",
                answer_kind="number",
                answer=w2,
                answer_text=str(w2),
                marks=1,
                working=[f"The j-coefficient is {m}({p2}) - {n}({q2}) = {w2}."],
                distractors=b_opts,
            ),
            dict(
                prompt="The vector kp + w has no i-component. Find the value of k.",
                answer_kind="number",
                answer=k,
                answer_text=str(k),
                marks=2,
                working=[f"The i-coefficient of kp + w is k({p1}) + ({w1}) = 0, so k = {k}."],
                distractors=c_opts,
                follow_through=f"-part_a/({p1})",
            ),
            dict(
                prompt="Hence find the coefficient of j in kp + w.",
                answer_kind="number",
                answer=d_key,
                answer_text=str(d_key),
                marks=1,
                working=[f"The j-coefficient is ({k})({p2}) + ({w2}) = {d_key}."],
                distractors=d_opts,
                follow_through=f"part_c*({p2}) + part_b",
            ),
        ],
    )
