"""Exam-tier source for "the midpoints of the sides of a quadrilateral join to form a parallelogram".

The practice tiers find a missing midpoint from the other three. An exam
question runs the relation the other way and asks for a missing VERTEX: the
learner is given midpoints of consecutive sides and one vertex, and has to
rebuild the quadrilateral one reflection at a time, using that each midpoint is
the average of its two endpoints.

With P, Q, R, S the midpoints of AB, BC, CD, DA:

    B = 2P - A,    C = 2Q - B,    D = 2R - C,

so the whole quadrilateral follows from A and the midpoints. Three shapes:

* ``chain_d``: A, P, Q, R given, find D (three reflections).
* ``chain_c``: A, P, Q given, find C (two reflections); the question also
  states that PQRS is a parallelogram, which is not needed.
* ``choose``: A, P, Q, R, S all given, find C. S is redundant and the learner
  must choose the shortest route, C = 2Q - 2P + A.

All coordinates are integers. The key is rebuilt in the generator with sympy
vectors by solving the four midpoint equations simultaneously.
"""

from __future__ import annotations

import random

from sympy import Matrix, Rational, symbols

from ..core import GRADER_TOLERANCE, check, check_equivalent, generator, phrase, pick_distractors


def _pt(p) -> str:
    return f"({p[0]}, {p[1]})"


def _fmt(p) -> str:
    return f"x = {p[0]}, y = {p[1]}"


def _rand_pt(rng, lo, hi):
    return (rng.randint(lo, hi), rng.randint(lo, hi))


def _reflect(mid, pt):
    return (2 * mid[0] - pt[0], 2 * mid[1] - pt[1])


def _solve_all(a, p, q, r):
    """Independent route: solve the four midpoint equations for B, C, D with sympy."""
    out = []
    for axis in (0, 1):
        b, c, d = symbols("b c d")
        sol = {
            b: Rational(2 * p[axis] - a[axis]),
        }
        # P = (A+B)/2, Q = (B+C)/2, R = (C+D)/2, as a linear system.
        m = Matrix([[1, 0, 0], [1, 1, 0], [0, 1, 1]])
        rhs = Matrix([2 * p[axis] - a[axis], 2 * q[axis], 2 * r[axis]])
        s = m.LUsolve(rhs)
        check(s[0] == sol[b], "midpoints: B disagrees")
        out.append(tuple(s))
    return (out[0][0], out[1][0]), (out[0][1], out[1][1]), (out[0][2], out[1][2])


def _draw(rng):
    for _ in range(500):
        a = _rand_pt(rng, -8, 8)
        p, q, r = _rand_pt(rng, -8, 8), _rand_pt(rng, -8, 8), _rand_pt(rng, -8, 8)
        b = _reflect(p, a)
        c = _reflect(q, b)
        d = _reflect(r, c)
        pts = [a, b, c, d]
        if len(set(pts)) < 4:
            continue
        if any(abs(v) > 30 for pt in pts for v in pt):
            continue
        if all((pt[0] - a[0]) * (b[1] - a[1]) == (pt[1] - a[1]) * (b[0] - a[0]) for pt in pts):
            continue
        s = ((d[0] + a[0]) // 2, (d[1] + a[1]) // 2) if (d[0] + a[0]) % 2 == 0 and (d[1] + a[1]) % 2 == 0 else None
        return a, b, c, d, p, q, r, s
    raise AssertionError("no draw")  # pragma: no cover


@generator(
    "exd_x_h4_midpoints_reconstruct",
    level=10,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_h4_midpoints_reconstruct(rng: random.Random, tier: str = "exam") -> dict:
    """Rebuild a quadrilateral vertex from midpoints of its sides."""
    check(tier == "exam", f"unknown tier {tier!r}")
    kind = rng.choice(("chain_d", "chain_c", "choose"))
    if kind == "choose":
        # Need S an integer point so the redundant fourth midpoint is stated exactly.
        for _ in range(500):
            a, b, c, d, p, q, r, s = _draw(rng)
            if s is not None:
                break
        else:  # pragma: no cover
            raise AssertionError("no draw with integer S")
    else:
        a, b, c, d, p, q, r, s = _draw(rng)
    b2, c2, d2 = _solve_all(a, p, q, r)
    check_equivalent(b2[0], b[0])
    check_equivalent(c2[1], c[1])
    check_equivalent(d2[0], d[0])
    check((b2, c2, d2) == (b, c, d), "midpoints: solved vertices disagree")

    if kind == "chain_d":
        key, name = d, "D"
        text = phrase(
            rng,
            (
                "Quadrilateral ABCD has vertex A{a}. The midpoints of AB, BC and CD are P{p}, Q{q} and R{r}. "
                "Find the coordinates of D.",
                "The midpoints of the consecutive sides AB, BC and CD of a quadrilateral ABCD are P{p}, Q{q} and "
                "R{r}, and vertex A is {a}. Find the coordinates of vertex D.",
            ),
            a=_pt(a),
            p=_pt(p),
            q=_pt(q),
            r=_pt(r),
        )
        wrong = [r, _reflect(r, b), (d[1], d[0]), _reflect(q, d)]
        working = [
            f"P is the midpoint of AB, so B = 2P - A = {_pt(b)}.",
            f"Q is the midpoint of BC, so C = 2Q - B = {_pt(c)}.",
            f"R is the midpoint of CD, so D = 2R - C = {_pt(d)}.",
        ]
    elif kind == "chain_c":
        key, name = c, "C"
        text = phrase(
            rng,
            (
                "In quadrilateral ABCD, vertex A is {a}. The midpoint of AB is P{p} and the midpoint of BC is Q{q}. "
                "Find the coordinates of C.",
                "A quadrilateral ABCD has A{a}. P{p} is the midpoint of side AB and Q{q} is the midpoint of side "
                "BC. Find the coordinates of vertex C.",
            ),
            a=_pt(a),
            p=_pt(p),
            q=_pt(q),
        )
        wrong = [q, _reflect(q, a), (c[1], c[0]), _reflect(p, q)]
        working = [
            f"P is the midpoint of AB, so B = 2P - A = {_pt(b)}.",
            f"Q is the midpoint of BC, so C = 2Q - B = {_pt(c)}.",
        ]
    else:
        key, name = c, "C"
        text = phrase(
            rng,
            (
                "P, Q, R and S are the midpoints of sides AB, BC, CD and DA of a quadrilateral ABCD. "
                "P{p}, Q{q}, R{r} and S{s} are given, and vertex A is {a}. Find the coordinates of C.",
                "The midpoints of the sides AB, BC, CD and DA of quadrilateral ABCD are P{p}, Q{q}, R{r} and S{s} "
                "in that order. Vertex A is {a}. Find the coordinates of vertex C.",
            ),
            a=_pt(a),
            p=_pt(p),
            q=_pt(q),
            r=_pt(r),
            s=_pt(s),
        )
        wrong = [q, _reflect(r, a), (c[1], c[0]), _reflect(q, a)]
        working = [
            "Each midpoint is the average of two vertices, so B = 2P - A and C = 2Q - B.",
            f"B = {_pt(b)}, so C = 2Q - B = {_pt(c)}. R and S are not needed.",
        ]
    answer_text = _fmt(key)
    distractors = pick_distractors(answer_text, [_fmt(w) for w in wrong], tolerance=GRADER_TOLERANCE)
    check(len(distractors) >= 2, "too few distractors")
    check(name in text, "asked vertex missing from prompt")
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer={"x": float(key[0]), "y": float(key[1])},
        answer_text=answer_text,
        answer_latex=rf"x = {key[0]},\; y = {key[1]}",
        working=working,
        distractors=distractors,
    )
