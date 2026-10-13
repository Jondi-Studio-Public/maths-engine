"""Exam chain for vector addition: a parallelogram in three dimensions with one coordinate unknown (three parts).

``ABCD`` is a parallelogram with its vertices in order, so the diagonals share a midpoint: ``A + C = B + D`` (as position
vectors). One coordinate of one vertex is written ``k``.

(a) find ``k`` (the learner has to see which pairs of vertices are opposite);
(b) hence the dot product of the side vectors ``AB`` and ``AD`` (follow-through: the dot product as a polynomial in
    ``part_a``);
(c) hence the angle ``DAB`` to the nearest degree (follow-through ``arccos`` of ``part_b`` over the two lengths).

Three vertices and the unknown are drawn first; the fourth vertex is derived, so every coordinate is a whole number. A
draw is kept only when the sides are not parallel, the angle is not a right angle and the angle sits well clear of a
rounding boundary.
"""

from __future__ import annotations

import math
import random

from mathgen.answers import Distractor, exact_value
from mathgen.core import (
    attempts,
    check,
    generator,
    latex_answer,
    pick_distractors,
)

_AXES = ("x", "y", "z")
_STEM = (
    "In three dimensions, ABCD is a parallelogram with its vertices labelled in order. "
    "A is the point {A}, B is {B}, C is {C} and D is {D}.",
    "The quadrilateral ABCD is a parallelogram, with vertices in order around it, in three-dimensional space. "
    "A is {A}, B is {B}, C is {C} and D is {D}.",
)


def _pt(coords) -> str:
    return "(" + ", ".join(str(c) for c in coords) + ")"


def _dist(text: str, tag: str, value: int) -> Distractor:
    return Distractor(text, tag, exact_value(value))


def _int_part(prompt, key: int, marks, working, dists, ft=None, sensible=None):
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=int(key),
        answer_text=str(int(key)),
        answer_latex=latex_answer(str(int(key))),
        marks=marks,
        working=working,
        distractors=dists,
    )
    if sensible is not None:
        part["sensible_range"] = sensible
    if ft:
        part["follow_through"] = ft
    return part


def _dists(key: int, pairs, n: int = 3):
    cands = [_dist(str(int(v)), tag, int(v)) for tag, v in pairs]
    out = pick_distractors(str(int(key)), cands, n)
    check(len(out) >= 2, "too few distractors")
    return out


def _sub(u, v):
    return [a - b for a, b in zip(u, v)]


def _dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def _eval(poly, k) -> int:
    return poly[0] * k * k + poly[1] * k + poly[2]


def _poly_text(poly) -> str:
    """A quadratic (a, b, c) in part_a as text SymPy reads: "part_a**2 - 3*part_a + 12"."""
    out = ""
    for coef, term in ((poly[0], "part_a**2"), (poly[1], "part_a"), (poly[2], "")):
        if coef == 0:
            continue
        mag = abs(coef)
        body = term if (mag == 1 and term) else (f"{mag}*{term}" if term else str(mag))
        out += (("-" if coef < 0 else "") + body) if not out else (" - " if coef < 0 else " + ") + body
    return out or "0"


def _angle(u, v) -> float:
    c = _dot(u, v) / (math.sqrt(_dot(u, u)) * math.sqrt(_dot(v, v)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


@generator("exd_x_r2d_vector_addition_chain", level=12, presets={"exam": dict(tier="exam")})
def exd_x_r2d_vector_addition_chain(rng: random.Random, tier: str = "exam") -> dict:
    """A parallelogram with an unknown coordinate: find it, then a dot product and an angle."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    unknown = rng.choice("ABCD")
    axis = rng.randrange(3)
    nxt = "BCDA"["ABCD".index(unknown)]
    prv = "DABC"["ABCD".index(unknown)]
    for _ in attempts(2000, "a parallelogram with a clean angle"):
        a = [rng.randint(-6, 6) for _ in range(3)]
        b = [rng.randint(-6, 6) for _ in range(3)]
        d = [rng.randint(-6, 6) for _ in range(3)]
        c = [bi + di - ai for ai, bi, di in zip(a, b, d)]
        pts = {"A": a, "B": b, "C": c, "D": d}
        s1, s2 = _sub(pts[nxt], pts[unknown]), _sub(pts[prv], pts[unknown])
        cross = [s1[1] * s2[2] - s1[2] * s2[1], s1[2] * s2[0] - s1[0] * s2[2], s1[0] * s2[1] - s1[1] * s2[0]]
        if not any(cross) or _dot(s1, s2) == 0 or max(abs(x) for x in c) > 9 or pts[unknown][axis] == 0:
            continue
        if abs(_angle(s1, s2) % 1 - 0.5) >= 0.06:
            break
    k_true = pts[unknown][axis]

    # The stated points, with k written in place. Each coordinate is held as (constant, coefficient of k).
    shown = {name: list(p) for name, p in pts.items()}
    shown[unknown][axis] = "k"
    lin = {name: [(x, 0) for x in p] for name, p in pts.items()}
    lin[unknown][axis] = (0, 1)

    def solve_k(coefs: dict[str, int]) -> int | None:
        """Solve sum(coef * coordinate) = 0 along `axis` for k; None unless k is a unique whole number."""
        total = sum(c * lin[n][axis][0] for n, c in coefs.items())
        kc = sum(c * lin[n][axis][1] for n, c in coefs.items())
        if kc == 0 or total % kc:
            return None
        return -total // kc

    diagonals = {"A": 1, "C": 1, "B": -1, "D": -1}  # A + C - B - D = 0
    check(solve_k(diagonals) == k_true, "k is not determined by A + C = B + D")

    opp = "CDAB"["ABCD".index(unknown)]
    v_name = unknown
    u1_l = [(lin[nxt][i][0] - lin[v_name][i][0], lin[nxt][i][1] - lin[v_name][i][1]) for i in range(3)]
    u2_l = [(lin[prv][i][0] - lin[v_name][i][0], lin[prv][i][1] - lin[v_name][i][1]) for i in range(3)]
    # Product of two affine forms c + d k and e + f k, summed over the three coordinates: a quadratic in k.
    dot_poly = (
        sum(x[1] * y[1] for x, y in zip(u1_l, u2_l)),
        sum(x[0] * y[1] + x[1] * y[0] for x, y in zip(u1_l, u2_l)),
        sum(x[0] * y[0] for x, y in zip(u1_l, u2_l)),
    )
    n1_poly = (
        sum(x[1] * x[1] for x in u1_l),
        sum(2 * x[0] * x[1] for x in u1_l),
        sum(x[0] * x[0] for x in u1_l),
    )
    n2_poly = (
        sum(x[1] * x[1] for x in u2_l),
        sum(2 * x[0] * x[1] for x in u2_l),
        sum(x[0] * x[0] for x in u2_l),
    )
    u1, u2 = _sub(pts[nxt], pts[v_name]), _sub(pts[prv], pts[v_name])
    dot_true = _dot(u1, u2)
    check(_eval(dot_poly, k_true) == dot_true, "dot product disagrees with the drawn points")
    check(dot_poly[0] > 0 and n1_poly[0] + n2_poly[0] > 0, "the dot product does not depend on k")
    ang = _angle(u1, u2)
    angle_whole = int(math.floor(ang + 0.5))
    ft_b = _poly_text(dot_poly)
    ft_c = f"180*acos(part_b/(sqrt({_poly_text(n1_poly)})*sqrt({_poly_text(n2_poly)})))/pi"
    side1, side2 = f"{v_name}{nxt}", f"{v_name}{prv}"
    known = [n for n in "BDAC" if n != unknown]

    a_pairs = []
    for rel in ({"A": 1, "B": 1, "C": -1, "D": -1}, {"A": 1, "D": 1, "B": -1, "C": -1}):  # wrong pairing
        v = solve_k(rel)
        if v is not None:
            a_pairs.append(("parallelogram-vertex-mispaired", v))
    for name in known:  # leave one known vertex out of A + C = B + D
        v = solve_k({n: (0 if n == name else c) for n, c in diagonals.items()})
        if v is not None:
            a_pairs.append(("dropped-a-term", v))
    for name in known:  # a sign slip on one known vertex when moving it across
        v = solve_k({n: (-c if n == name else c) for n, c in diagonals.items()})
        if v is not None:
            a_pairs.append(("sign-error-moving-term", v))
    a_pairs.append(("sign-error-moving-term", -k_true))
    a_pairs = [(t, v) for t, v in a_pairs if v != k_true]

    diag = _sub(pts[opp], pts[v_name])
    b_pairs = [
        ("used-wrong-vector-pair", _dot(u1, diag)),  # a side with the diagonal
        ("used-wrong-vector-pair", -dot_true),  # a side with the reverse of the other
        ("used-wrong-vector-pair", _dot(pts[nxt], pts[prv])),  # position vectors, not the sides
        ("used-wrong-vector-pair", _dot(diag, u2)),
    ]
    b_pairs = [(t, v) for t, v in b_pairs if v != dot_true]
    c_pairs = [("supplementary-angle", 180 - angle_whole)]
    cos_val = dot_true / (math.sqrt(_dot(u1, u1)) * math.sqrt(_dot(u2, u2)))
    c_pairs.append(("swaps-sine-and-cosine", int(math.floor(math.degrees(math.asin(cos_val)) + 0.5))))
    c_pairs.append(("used-wrong-vector-pair", int(math.floor(_angle(u1, diag) + 0.5))))
    c_pairs.append(("used-wrong-vector-pair", int(math.floor(_angle(u2, diag) + 0.5))))
    for wrong_denominator in (_dot(u1, u1) + _dot(u2, u2), max(_dot(u1, u1), _dot(u2, u2))):  # not |u1||u2|
        if abs(dot_true) <= wrong_denominator:
            c_pairs.append(
                (
                    "forgot-to-divide-by-magnitudes",
                    int(math.floor(math.degrees(math.acos(dot_true / wrong_denominator)) + 0.5)),
                )
            )
    c_pairs = [(t, v) for t, v in c_pairs if 0 < v < 180 and v != angle_whole]

    stem = rng.choice(_STEM).format(**{n: _pt(shown[n]) for n in "ABCD"})
    axis_name = _AXES[axis]
    return dict(
        text=stem,
        parts=[
            _int_part(
                f"Find the value of k, the {axis_name}-coordinate of {unknown}.",
                k_true,
                2,
                [
                    "The diagonals of a parallelogram bisect each other, so A + C = B + D (position vectors).",
                    f"The {axis_name}-coordinates give k = {k_true}.",
                ],
                _dists(k_true, a_pairs),
            ),
            _int_part(
                f"Hence find the dot product {side1} · {side2} of the two sides at {v_name}.",
                dot_true,
                2,
                [f"{side1} = {_pt(u1)} and {side2} = {_pt(u2)}.", f"{side1} · {side2} = {dot_true}."],
                _dists(dot_true, b_pairs),
                ft=ft_b,
            ),
            _int_part(
                f"Hence find the size of angle {nxt}{v_name}{prv}, in degrees, to the nearest degree.",
                angle_whole,
                2,
                [
                    f"|{side1}|² = {_dot(u1, u1)} and |{side2}|² = {_dot(u2, u2)}.",
                    f"cos {nxt}{v_name}{prv} = {dot_true} ÷ (|{side1}||{side2}|), so the angle is {angle_whole}°.",
                ],
                _dists(angle_whole, c_pairs),
                ft=ft_c,
                sensible=(0, 180),
            ),
        ],
    )
