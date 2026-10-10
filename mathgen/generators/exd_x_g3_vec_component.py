"""Exam-tier source for the skill "Express a vector in component form using the unit vectors".

The practice tiers read a component off a described movement. This tier gives a
*direction* (a vector ``w`` that is not a unit vector) and a *magnitude*, and asks
for one component of the vector ``v`` that has that magnitude and points the same
way as ``w`` (or the opposite way).  The learner has to choose the method:
normalise ``w`` using its length, scale to the stated magnitude, then apply the
direction word.

Built backwards from the answer.  A Pythagorean triple ``(a, b, c)`` gives a
primitive direction ``d = (±a, ±b)`` (possibly with the two legs swapped).
``w = s * d`` for a small integer ``s`` (so ``|w| = s * c`` is a whole number),
``v = ±t * d`` and the stated magnitude is ``t * c``.  Every number in the prompt
is therefore an integer and the answer is an exact integer.
"""

from __future__ import annotations

import random

from sympy import Matrix, Rational, sqrt

from ..core import check, check_equivalent, generator, join_terms, pick_distractors

_TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29)]


def _vec_text(p: int, q: int) -> str:
    return join_terms([(p, "i"), (q, "j")])


def _vec_tex(p: int, q: int) -> str:
    sign = "+" if q > 0 else "-"
    first = {1: "", -1: "-"}.get(p, str(p))
    second = "" if abs(q) == 1 else str(abs(q))
    return rf"{first}\mathbf{{i}} {sign} {second}\mathbf{{j}}"


def _build(rng: random.Random) -> dict:
    for _ in range(200):
        a, b, c = rng.choice(_TRIPLES)
        if rng.random() < 0.5:
            a, b = b, a
        da = a * rng.choice([-1, 1])
        db = b * rng.choice([-1, 1])
        s = rng.choice([1, 2, 3])
        t = rng.randint(2, 6)
        opposite = rng.random() < 0.5
        want = rng.choice(["i", "j"])
        w = (s * da, s * db)
        sign = -1 if opposite else 1
        v = (sign * t * da, sign * t * db)
        magnitude = t * c
        answer = v[0] if want == "i" else v[1]
        other = v[1] if want == "i" else v[0]
        w_comp = w[0] if want == "i" else w[1]

        # Independent route: normalise w with sympy, scale, apply the direction.
        unit = Matrix(w) / sqrt(w[0] ** 2 + w[1] ** 2)
        scaled = (magnitude * sign) * unit
        check_equivalent(scaled[0 if want == "i" else 1], answer)

        cands = [
            str(-answer),  # ignores the direction word
            str(other),  # reports the other component
            str(w_comp),  # reads the component of w itself
            str(magnitude * w_comp),  # scales w by the magnitude without dividing by |w|
        ]
        distractors = pick_distractors(str(answer), cands)
        if len(distractors) < 3:
            continue
        break
    else:  # pragma: no cover
        raise ValueError("could not draw a question with three distractors")

    wtxt = _vec_text(*w)
    wtex = _vec_tex(*w)
    frame = rng.choice(["vector", "force"])
    if frame == "vector":
        direction = "in the opposite direction to" if opposite else "in the same direction as"
        latex = (
            rf"\text{{Let }}\mathbf{{w}} = {wtex}\text{{. A vector }}\mathbf{{v}}"
            rf"\text{{ has magnitude {magnitude} and points {direction} }}\mathbf{{w}}"
            r"\text{. Write }\mathbf{v}\text{ in component form using the unit vectors }"
            r"\mathbf{i}\text{ and }\mathbf{j}"
            rf"\text{{. What is the coefficient of }}\mathbf{{{want}}}\text{{ (a whole number)?}}"
        )
        text = (
            f"Let w = {wtxt}. A vector v has magnitude {magnitude} and points {direction} w. "
            f"Write v in component form using the unit vectors i and j. "
            f"What is the coefficient of {want} (a whole number)?"
        )
    else:
        direction = "opposite to" if opposite else "along"
        latex = (
            rf"\text{{A force }}\mathbf{{F}}\text{{ of magnitude {magnitude} newtons acts {direction} the vector }}{wtex}"
            r"\text{. Write }\mathbf{F}\text{ in component form using the unit vectors }"
            r"\mathbf{i}\text{ and }\mathbf{j}"
            rf"\text{{. What is the coefficient of }}\mathbf{{{want}}}\text{{ in }}\mathbf{{F}}\text{{ (a whole number)?}}"
        )
        text = (
            f"A force F of magnitude {magnitude} newtons acts {direction} the vector {wtxt}. "
            f"Write F in component form using the unit vectors i and j. "
            f"What is the coefficient of {want} in F (a whole number)?"
        )
    check(len(distractors) >= 3)
    return dict(
        text=text,
        latex=latex,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[
            f"|w| = sqrt(({w[0]})^2 + ({w[1]})^2) = {s * c}",
            f"unit vector of w = ({Rational(w[0], s * c)}, {Rational(w[1], s * c)})",
            f"v = {sign * magnitude} times that unit vector = ({v[0]}, {v[1]})",
            f"coefficient of {want} = {answer}",
        ],
        distractors=distractors,
    )


@generator(
    "exd_x_g3_vec_component",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_g3_vec_component(rng: random.Random, tier: str = "exam") -> dict:
    """Component of a vector given its magnitude and a direction vector."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
