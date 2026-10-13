"""Exam-tier source for `senior-use-ordered-pair-notation-and-column-vector-notation`: points built from vectors.

Point A is given, the column vector from A to B, and the column vector from C to B. One axis (x or y) is chosen and
every part asks about it, so the learner has to pick the right entry of each column. The chain:

(a) the coordinate of B                  number, 1 mark    A + AB
(b) the coordinate of C                  number, 2 marks   B - CB                  follow_through: part_a - r
(c) the component of the vector AC       number, 1 mark    C - A                   follow_through: part_b - A
(d) the coordinate of M, midpoint of AC  number, 2 marks   A + AC/2                follow_through: A + part_c/2

C - A is drawn even so the midpoint has whole-number coordinates.
"""

from __future__ import annotations

from ..answers import Distractor
from ..core import check, generator, paren_negative, phrase, pick_distractors
from ._exd_distinct_parts import distinct_part_answers

_COORD = [n for n in range(-9, 10)]
_COMP = [n for n in range(-8, 9) if n != 0]


def _options(key: int, candidates, n: int = 3):
    out, seen = [], {key}
    for tag, value in candidates:
        if value in seen:
            continue
        seen.add(value)
        out.append(Distractor(str(value), tag, value=value))
    return pick_distractors(str(key), out, n)


@generator("exd_x_r3e_vector_points", level=12, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3e_vector_points(rng, tier: str = "exam") -> dict:
    check(tier == "exam", f"unknown tier {tier!r}")
    axis = rng.choice(["x", "y"])
    t = 0 if axis == "x" else 1
    for _ in range(2000):
        A = (rng.choice(_COORD), rng.choice(_COORD))
        AB = (rng.choice(_COMP), rng.choice(_COMP))
        CB = (rng.choice(_COMP), rng.choice(_COMP))
        B = (A[0] + AB[0], A[1] + AB[1])
        C = (B[0] - CB[0], B[1] - CB[1])
        AC = (C[0] - A[0], C[1] - A[1])
        if any(v == 0 for v in AC) or AC[t] % 2 or AC[1 - t] % 2:
            continue
        M = (A[0] + AC[0] // 2, A[1] + AC[1] // 2)
        o = 1 - t
        a_key, b_key, c_key, d_key = B[t], C[t], AC[t], M[t]
        a_opts = _options(
            a_key,
            [
                ("reversed-vector-direction", A[t] - AB[t]),
                ("vector-component-as-coordinate", AB[t]),
                ("used-other-component-of-vector", B[o]),
            ],
        )
        b_opts = _options(
            b_key,
            [
                ("reversed-vector-direction", B[t] + CB[t]),
                ("vector-component-as-coordinate", -CB[t]),
                ("used-other-component-of-vector", C[o]),
            ],
        )
        c_opts = _options(
            c_key,
            [
                ("coordinate-as-vector-component", C[t]),
                ("reversed-vector-direction", A[t] - C[t]),
                ("reversed-vector-direction", AB[t] + CB[t]),
                ("used-other-component-of-vector", AC[o]),
            ],
        )
        d_opts = _options(
            d_key,
            [
                ("midpoint-without-start-point", AC[t] // 2),
                ("endpoint-for-midpoint", C[t]),
                ("vector-component-as-coordinate", AC[t]),
                ("used-other-component-of-vector", M[o]),
            ],
        )
        if min(len(a_opts), len(b_opts), len(c_opts), len(d_opts)) >= 3:
            break
    else:
        check(False, "no vector-points draw with enough distractors")

    check(B == (A[0] + AB[0], A[1] + AB[1]) and (C[0] + CB[0], C[1] + CB[1]) == B, "B or C disagrees with the vectors")
    check((A[t] + C[t]) == 2 * M[t] and M[t] - A[t] == AC[t] // 2, "midpoint disagrees")
    stem = phrase(
        rng,
        (
            "Point A is ({ax}, {ay}). The vector from A to B is the column vector ({p}, {q}) and the vector from C "
            "to B is the column vector ({r}, {s}).",
            "A is the point ({ax}, {ay}). Going from A to B is the column vector ({p}, {q}), and going from C to B "
            "is the column vector ({r}, {s}).",
            "On a coordinate grid, A is at ({ax}, {ay}). The column vector from A to B is ({p}, {q}) and the column "
            "vector from C to B is ({r}, {s}).",
        ),
        ax=A[0],
        ay=A[1],
        p=AB[0],
        q=AB[1],
        r=CB[0],
        s=CB[1],
    )
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"Find the {axis}-coordinate of B.",
                answer_kind="number",
                answer=a_key,
                answer_text=str(a_key),
                marks=1,
                working=[f"B = A + AB, so its {axis}-coordinate is {A[t]} + ({AB[t]}) = {a_key}."],
                distractors=a_opts,
            ),
            dict(
                prompt=f"Find the {axis}-coordinate of C.",
                answer_kind="number",
                answer=b_key,
                answer_text=str(b_key),
                marks=2,
                working=[f"C = B - CB, so its {axis}-coordinate is {a_key} - ({CB[t]}) = {b_key}."],
                distractors=b_opts,
                follow_through=f"part_a - ({CB[t]})",
            ),
            dict(
                prompt=f"Hence find the {axis}-component of the vector from A to C.",
                answer_kind="number",
                answer=c_key,
                answer_text=str(c_key),
                marks=1,
                working=[f"AC = C - A, so its {axis}-component is {b_key} - ({A[t]}) = {c_key}."],
                distractors=c_opts,
                follow_through=f"part_b - ({A[t]})",
            ),
            dict(
                prompt=f"M is the midpoint of AC. Find the {axis}-coordinate of M.",
                answer_kind="number",
                answer=d_key,
                answer_text=str(d_key),
                marks=2,
                working=[f"M = A + AC/2, so its {axis}-coordinate is {A[t]} + {paren_negative(c_key)}/2 = {d_key}."],
                distractors=d_opts,
                follow_through=f"({A[t]}) + part_c/2",
            ),
        ],
    )
