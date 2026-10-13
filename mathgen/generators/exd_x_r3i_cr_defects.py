"""Exam-tier source for `ucv1-cr-equations-check`: how far each Cauchy-Riemann equation is from holding at a point.

The stem gives u(x, y) and v(x, y), polynomials of degree up to three, and a point P. Instead of choosing which
equations hold from a list, the learner computes the two discrepancies and combines them:

(a) u_x - v_y at P                                            number, 2 marks
(b) u_y + v_x at P                                            number, 2 marks
(c) |u_x - v_y| + |u_y + v_x| at P                            number, 1 mark, |part_a| + |part_b|

An equation holds at P exactly when its discrepancy is 0, and the total in (c) is 0 exactly when both do.

## Construction

u and v start from the real and imaginary parts of a holomorphic cubic (for which both discrepancies are 0
everywhere), and a few extra terms are added, each of which moves one or both discrepancies by a known amount. The
case is chosen first (only the first fails, only the second fails, both fail) and the point and extra
terms are redrawn until the discrepancies at P show that case, so the three cases are all common. Both holding is not drawn: all three keys would be 0, and a reply of 0 throughout
would earn full marks. A single failing equation is drawn only when its discrepancy is negative, so (c) differs from it.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator, join_terms, phrase
from ._exd_distinct_parts import distinct_part_answers
from .exd_x_r3i_support import number_part

Poly = dict[tuple[int, int], int]  # {(power of x, power of y): coefficient}

_CASES = ("first-fails",) * 6 + ("second-fails",) * 6 + ("both-fail",) * 27


def _add(p: Poly, mono: tuple[int, int], c: int) -> None:
    p[mono] = p.get(mono, 0) + c


def _dx(p: Poly) -> Poly:
    return {(i - 1, j): c * i for (i, j), c in p.items() if i > 0}


def _dy(p: Poly) -> Poly:
    return {(i, j - 1): c * j for (i, j), c in p.items() if j > 0}


def _at(p: Poly, x: int, y: int) -> int:
    return sum(c * x**i * y**j for (i, j), c in p.items())


_NAMES = {
    (3, 0): "x^3",
    (2, 1): "x^2y",
    (1, 2): "xy^2",
    (0, 3): "y^3",
    (2, 0): "x^2",
    (1, 1): "xy",
    (0, 2): "y^2",
    (1, 0): "x",
    (0, 1): "y",
}
_ORDER = [(3, 0), (2, 1), (1, 2), (0, 3), (2, 0), (1, 1), (0, 2), (1, 0), (0, 1)]


def _text(p: Poly) -> str:
    return join_terms([(p.get(m, 0), _NAMES[m]) for m in _ORDER])


def _build(rng) -> tuple[Poly, Poly]:
    a3, a2 = rng.choice([-2, -1, 0, 1, 2]), rng.randint(-3, 3)
    a1, b1 = rng.randint(-4, 4), rng.randint(-3, 3)
    u: Poly = {}
    v: Poly = {}
    # f(z) = a3 z^3 + a2 z^2 + (a1 + i b1) z, with z = x + iy.
    for mono, c in (((3, 0), a3), ((1, 2), -3 * a3), ((2, 0), a2), ((0, 2), -a2), ((1, 0), a1), ((0, 1), -b1)):
        _add(u, mono, c)
    for mono, c in (((2, 1), 3 * a3), ((0, 3), -a3), ((1, 1), 2 * a2), ((0, 1), a1), ((1, 0), b1)):
        _add(v, mono, c)
    for target in (u, v):
        for mono in ((1, 1), (2, 0), (0, 2), (1, 0), (0, 1)):
            if rng.random() < 0.35:
                _add(target, mono, rng.choice([-2, -1, 1, 2]))
    return {m: c for m, c in u.items() if c}, {m: c for m, c in v.items() if c}


@generator("exd_x_r3i_cr_defects", level=17, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3i_cr_defects(rng, tier: str = "exam") -> dict:
    """The two Cauchy-Riemann discrepancies at a point of u + iv, and their total."""
    check(tier == "exam", f"unknown tier {tier!r}")
    case = rng.choice(_CASES)
    for _ in attempts(3000, "functions and a point showing the chosen case"):
        u, v = _build(rng)
        x0, y0 = rng.randint(-3, 3), rng.randint(-3, 3)
        if (x0, y0) == (0, 0) or len(u) < 2 or len(v) < 2:
            continue
        ux, uy, vx, vy = _at(_dx(u), x0, y0), _at(_dy(u), x0, y0), _at(_dx(v), x0, y0), _at(_dy(v), x0, y0)
        d1, d2 = ux - vy, uy + vx
        # Steer the chosen case: a term -d1 x in u cancels the first discrepancy at P (it moves u_x only), and a
        # term -d2 x in v cancels the second (it moves v_x only).
        if case in ("both-hold", "second-fails"):
            _add(u, (1, 0), -d1)
            ux, d1 = ux - d1, 0
        if case in ("both-hold", "first-fails"):
            _add(v, (1, 0), -d2)
            vx, d2 = vx - d2, 0
        u, v = {m: c for m, c in u.items() if c}, {m: c for m, c in v.items() if c}
        shown = ("both-hold", "first-fails", "second-fails", "both-fail")[(d1 != 0) + 2 * (d2 != 0)]
        # Each part needs three distinct wrong answers in the same layout, so redraw a point where they collapse.
        a_wrong = {ux + vy, -d1, uy - vx, ux, ux - uy} - {d1}
        b_wrong = {uy - vx, -d2, ux + vy, uy, ux - vx} - {d2}
        c_wrong = {
            d1 + d2,
            abs(d1 + d2),
            max(abs(d1), abs(d2)),
            abs(d1) - abs(d2),
            ux + vy,
            abs(ux) + abs(uy) + abs(vx) + abs(vy),
            uy - vx,
            abs(ux) + abs(vy),
        } - {abs(d1) + abs(d2)}
        if shown == case and min(len(a_wrong), len(b_wrong), len(c_wrong)) >= 3:
            break
    total = abs(d1) + abs(d2)

    # Independent route: finite differences. For a polynomial of degree at most three, the central difference
    # D(h) = [p(t + h) - p(t - h)] / 2h equals p'(t) + h^2 * (cubic coefficient), so (4 D(1) - D(2)) / 3 = p'(t).
    def partial(poly: Poly, wrt_x: bool) -> Fraction:
        def central(h: int) -> Fraction:
            if wrt_x:
                return Fraction(_at(poly, x0 + h, y0) - _at(poly, x0 - h, y0), 2 * h)
            return Fraction(_at(poly, x0, y0 + h) - _at(poly, x0, y0 - h), 2 * h)

        return (4 * central(1) - central(2)) / 3

    check(partial(u, True) - partial(v, False) == d1 and partial(u, False) + partial(v, True) == d2, "FD disagrees")

    stem = phrase(
        rng,
        (
            "Let f(z) = u(x, y) + i v(x, y) with u(x, y) = {u} and v(x, y) = {v}. Consider the point P where "
            "(x, y) = ({x0}, {y0}).",
            "A complex function f = u + iv has u(x, y) = {u} and v(x, y) = {v}. Let P be the point (x, y) = "
            "({x0}, {y0}).",
        ),
        u=_text(u),
        v=_text(v),
        x0=x0,
        y0=y0,
    )
    part_a = number_part(
        "Find the value of u_x - v_y at P.",
        Fraction(d1),
        0,
        2,
        [f"u_x({x0}, {y0}) = {ux} and v_y({x0}, {y0}) = {vy}, so u_x - v_y = {d1}."],
        [
            ("added-instead-of-subtracted", Fraction(ux + vy)),
            ("sign-of-result-reversed", Fraction(-d1)),
            ("cr-wrong-partials-paired", Fraction(uy - vx)),
            ("reports-one-partial-not-the-difference", Fraction(ux)),
            ("cr-wrong-partials-paired", Fraction(ux - uy)),
        ],
    )
    part_b = number_part(
        "Find the value of u_y + v_x at P.",
        Fraction(d2),
        0,
        2,
        [f"u_y({x0}, {y0}) = {uy} and v_x({x0}, {y0}) = {vx}, so u_y + v_x = {d2}."],
        [
            ("sign-slip-in-formula", Fraction(uy - vx)),
            ("sign-of-result-reversed", Fraction(-d2)),
            ("cr-wrong-partials-paired", Fraction(ux + vy)),
            ("reports-one-partial-not-the-difference", Fraction(uy)),
            ("cr-wrong-partials-paired", Fraction(ux - vx)),
        ],
    )
    part_c = number_part(
        "Hence find the value of |u_x - v_y| + |u_y + v_x| at P.",
        Fraction(total),
        0,
        1,
        [f"|{d1}| + |{d2}| = {total}."],
        [
            ("summed-signed-values", Fraction(d1 + d2)),
            ("absolute-values-taken-of-sum", Fraction(abs(d1 + d2))),
            ("larger-discrepancy-only", Fraction(max(abs(d1), abs(d2)))),
            ("subtracted-instead-of-added", Fraction(abs(d1) - abs(d2))),
            ("all-four-partials-added", Fraction(abs(ux) + abs(uy) + abs(vx) + abs(vy))),
            ("added-instead-of-subtracted", Fraction(ux + vy)),
            ("second-pair-of-partials-differenced", Fraction(uy - vx)),
            ("first-pair-of-absolute-partials-added", Fraction(abs(ux) + abs(vy))),
        ],
        follow_through="Abs(part_a) + Abs(part_b)",
        sensible_range=None,
    )
    return dict(text=stem, parts=[part_a, part_b, part_c])
