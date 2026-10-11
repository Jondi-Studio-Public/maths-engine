"""Exam-tier source for the skill "Represent a scalar multiple of a vector".

The practice tiers scale a vector by a given scalar, or undo that scaling.  This
tier hides the scalar itself: ``k u - w`` is stated to be parallel to a third
vector ``e`` and the learner must find ``k``.  That needs two ideas together
(scalar multiples and the parallel condition) and a choice of method (compare
components, or set the cross term to zero).

Built backwards: draw ``u`` and ``e`` that are not parallel, an integer ``k`` and
an integer multiple ``m`` of ``e``, then set ``w = k u - m e``.  Then
``k u - w = m e`` is parallel to ``e`` by construction, and because ``u`` is not
parallel to ``e`` the value of ``k`` is the only one that works.
"""

from __future__ import annotations

import random

from sympy import Matrix, Symbol, solve

from ..core import check, generator, pick_distractors


def _pair(v) -> str:
    return f"({v[0]}, {v[1]})"


def _build(rng: random.Random) -> dict:
    for _ in range(500):
        u = (rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]), rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]))
        e = (rng.choice([-3, -2, -1, 1, 2, 3]), rng.choice([-3, -2, -1, 1, 2, 3]))
        if u[0] * e[1] - u[1] * e[0] == 0:
            continue
        k = rng.choice([-5, -4, -3, -2, 2, 3, 4, 5])
        m = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
        form = rng.choice(["minus", "reversed", "plus"])
        # ku - w = m e, or w - ku = m e, or ku + w = m e
        w = (
            (k * u[0] - m * e[0], k * u[1] - m * e[1])
            if form == "minus"
            else (k * u[0] + m * e[0], k * u[1] + m * e[1])
            if form == "reversed"
            else (m * e[0] - k * u[0], m * e[1] - k * u[1])
        )
        if w == (0, 0) or max(abs(w[0]), abs(w[1])) > 30:
            continue

        # Independent route: solve the parallel condition (zero 2x2 determinant) for k.
        ks = Symbol("k")
        shown = {"minus": (1, -1), "reversed": (-1, 1), "plus": (1, 1)}[form]
        diff = Matrix([shown[0] * ks * u[0] + shown[1] * w[0], shown[0] * ks * u[1] + shown[1] * w[1]])
        sols = solve(Matrix.hstack(diff, Matrix(e)).det(), ks)
        check(sols == [k], "parallel condition must have the single root k")

        cands = [-k, m]
        # sign slip: treats the condition as ku + w parallel to e (gives -k exactly)
        for num, den in ((e[0] + w[0], u[0]), (e[1] + w[1], u[1]), (w[0], u[0]), (w[1], u[1])):
            # equates a component of ku - w with that of e, or ignores e entirely
            if den != 0 and num % den == 0:
                cands.append(num // den)
        distractors = pick_distractors(str(k), [str(c) for c in cands])
        if len(distractors) < 3:
            continue
        break
    else:  # pragma: no cover
        raise ValueError("could not draw a question")
    expr, texexpr = {
        "minus": ("ku - w", r"k\mathbf{u} - \mathbf{w}"),
        "reversed": ("w - ku", r"\mathbf{w} - k\mathbf{u}"),
        "plus": ("ku + w", r"k\mathbf{u} + \mathbf{w}"),
    }[form]
    frame = rng.choice(["given", "parallel", "find"])
    given = f"u = {_pair(u)}, w = {_pair(w)} and e = {_pair(e)}"
    if frame == "given":
        text = f"Let {given}. The vector {expr} is parallel to e. Find the scalar k."
    elif frame == "parallel":
        text = f"The vectors are {given}. For one value of k, {expr} is parallel to e. What is k?"
    else:
        text = f"Given {given}, find the scalar k such that {expr} is parallel to e."
    latex = (
        rf"\text{{Let }}\mathbf{{u}} = {_pair(u)},\ \mathbf{{w}} = {_pair(w)}\text{{ and }}\mathbf{{e}} = {_pair(e)}"
        rf"\text{{. The vector }}{texexpr}\text{{ is parallel to }}\mathbf{{e}}\text{{. Find the scalar }}k\text{{.}}"
    )
    return dict(
        text=text,
        latex=latex,
        answer=k,
        answer_text=str(k),
        answer_latex=str(k),
        working=[
            f"ku - w = ({k}*{u[0]} - ({w[0]}), {k}*{u[1]} - ({w[1]})) in terms of k",
            f"parallel to e: (k*{u[0]} - ({w[0]}))*{e[1]} - (k*{u[1]} - ({w[1]}))*{e[0]} = 0",
            f"k = {k}",
        ],
        distractors=distractors[:3],
    )


@generator(
    "exd_x_g3_vec_scalar",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_g3_vec_scalar(rng: random.Random, tier: str = "exam") -> dict:
    """Find the scalar that makes ku - w parallel to e."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
