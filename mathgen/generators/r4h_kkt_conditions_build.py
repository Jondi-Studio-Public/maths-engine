"""Karush–Kuhn–Tucker conditions: `r4h_kkt_conditions` (easy, medium, hard) and the exam chain `exd_x_r4h_kkt_conditions`.

``uc2-kkt-conditions-check`` (``optimisation``). To minimise ``f`` subject to ``gᵢ ≤ 0`` the KKT conditions are
stationarity ``∇f + Σ λᵢ ∇gᵢ = 0``, ``λᵢ ≥ 0``, ``gᵢ ≤ 0`` and complementary slackness ``λᵢ gᵢ = 0``.

* easy: one linear constraint, active at a stated point; find the multiplier ``λ`` from stationarity (it can be
  negative, which shows the point is not a KKT point).
* medium: two linear constraints active at a stated point; find both multipliers (a column vector).
* hard: a circular constraint ``x² + y² ≤ r²`` and a linear objective; locate the active boundary point and find ``λ``.
* exam: minimise a weighted squared distance to a point outside a half-plane: ``λ``, the minimiser's two coordinates
  (follow-through from ``λ``) and the minimum value (follow-through from the coordinates).

Exact arithmetic with `Fraction`. Every instance is built backwards from a chosen point and multipliers.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator
from ._r4h_kit import (
    Redraw,
    dtxt,
    finish_parts,
    fraction_part,
    ft_symbol,
    lin_text,
    rat,
    shifted,
    single_from,
    tx,
    with_redraw,
)

MISC = {
    "num2t-kkt-gradient-sign": "Setting ∇f = λ∇g instead of ∇f + λ∇g = 0, so the multiplier has the wrong sign.",
    "num2t-kkt-gradient-not-doubled": "Differentiating (x − a)² as (x − a), dropping the factor 2, when forming ∇f.",
    "num2t-kkt-one-component": "Using the ratio from one component of the gradient equation without checking the other.",
    "num2t-kkt-constraint-value": "Giving the value of the constraint function or its right-hand side as the multiplier.",
    "num2t-kkt-gradient-of-g-forgotten": "Dividing the gradient of f by the constraint's coefficients without matching components.",
    "num2t-kkt-multipliers-swapped": "Swapping the multipliers of the two constraints.",
    "num2t-kkt-single-constraint": "Solving with only one of the two active constraints.",
    "num2t-kkt-system-sign": "Solving ∇f = λ₁∇g₁ + λ₂∇g₂ with the sign of ∇f reversed.",
    "num2t-kkt-system-transposed": "Setting up the two equations with the gradients' components in the wrong columns.",
    "num2t-kkt-circle-not-doubled": "Taking ∇g = (x, y) for g = x² + y² − r², dropping the factor 2, so λ comes out twice too large.",
    "num2t-kkt-radius-squared": "Using r² where r belongs in the formula for the multiplier.",
    "num2t-kkt-norm-squared": "Using the squared length of the objective gradient instead of its length.",
    "num2t-kkt-circle-point-unscaled": "Taking the boundary point as the objective gradient itself rather than scaled to the circle.",
    "num2t-kkt-lambda-sum-form": "Using (a + b)/(2r) in place of √(a² + b²)/(2r) for the multiplier.",
    "num2t-kkt-lambda-uses-unconstrained": "Giving the amount by which the unconstrained minimum violates the constraint, not divided by the gradient's squared length.",
    "num2t-kkt-lambda-weights-ignored": "Ignoring the weights on the squares when solving for the multiplier.",
    "num2t-kkt-lambda-plain-length": "Dividing the violation by the length of the constraint gradient rather than its square.",
    "num2t-kkt-coordinate-sign": "Moving from the unconstrained minimum in the direction of the constraint gradient instead of against it.",
    "num2t-kkt-coordinate-unscaled": "Moving from the unconstrained minimum by λ times the constraint coefficient without dividing by twice the weight.",
    "num2t-kkt-coordinate-unconstrained": "Giving the unconstrained minimiser's coordinate instead of the constrained one.",
    "num2t-kkt-coordinate-other": "Giving the other coordinate of the constrained minimiser.",
    "num2t-kkt-value-unconstrained": "Giving the value at the unconstrained minimiser, which is zero or the objective's constant.",
    "num2t-kkt-value-unweighted": "Leaving out the weights on the squares when evaluating the objective.",
    "num2t-kkt-value-lambda-squared": "Giving λ² in place of the objective's value at the minimiser.",
    "num2t-kkt-value-not-squared": "Forgetting to square the distances when evaluating the objective.",
}
M_SIGN = "num2t-kkt-gradient-sign"
M_HALF = "num2t-kkt-gradient-not-doubled"
M_ONE = "num2t-kkt-one-component"
M_RHS = "num2t-kkt-constraint-value"
M_SWAP = "num2t-kkt-multipliers-swapped"
M_SINGLE = "num2t-kkt-single-constraint"
M_SYS_SIGN = "num2t-kkt-system-sign"
M_SYS_T = "num2t-kkt-system-transposed"
M_CIRC = "num2t-kkt-circle-not-doubled"
M_R2 = "num2t-kkt-radius-squared"
M_NORM2 = "num2t-kkt-norm-squared"
M_SUM = "num2t-kkt-lambda-sum-form"


def _g_text(p, q, r) -> str:
    """The constraint function ``px + qy − r`` as text."""
    base = lin_text(p, q)
    return base if r == 0 else f"{base} {'−' if r > 0 else '+'} {dtxt(abs(r))}"


def _point(rng):
    return Fraction(rng.randint(-3, 4)), Fraction(rng.randint(-3, 4))


def _grad_pair(rng):
    for _ in attempts(500, "a constraint gradient"):
        p, q = rng.randint(-3, 3), rng.randint(-3, 3)
        if (p, q) != (0, 0) and p != 0 and q != 0:
            return p, q
    raise AssertionError("unreachable")


def _lam(rng, negative=False):
    v = Fraction(rng.choice([1, 2, 3, 4, 5, 6, 1, 3, 5]), rng.choice([1, 1, 2]))
    return -v if negative else v


@generator(
    "r4h_kkt_conditions",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4h_kkt_conditions(rng, tier="easy"):
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return with_redraw(lambda: {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng), "a KKT question")


def _easy(rng):
    px, py = _point(rng)
    p, q = _grad_pair(rng)
    lam = _lam(rng, negative=rng.random() < 0.3)
    cx, cy = px + lam * p / 2, py + lam * q / 2
    r = p * px + q * py
    gx, gy = 2 * (px - cx), 2 * (py - cy)
    check((gx, gy) == (-lam * p, -lam * q), "stationarity fails")
    cands = [
        (M_SIGN, -lam),
        (M_HALF, lam * 2),
        (M_RHS, Fraction(r)),
        ("num2t-kkt-gradient-of-g-forgotten", lam / 2),
    ]
    part = fraction_part("", lam, cands, 1)
    f_t = f"{shifted('x', cx)} + {shifted('y', cy)}"
    con = f"{lin_text(p, q)} ≤ {dtxt(r)}"
    P = f"({dtxt(px)}, {dtxt(py)})"
    stem = rng.choice(
        [
            f"Minimise «f(x, y) = {f_t}» subject to «{con}». The constraint is active at «P = {P}». Find the number «λ» "
            f"for which «∇f(P) + λ∇g(P) = 0», where «g(x, y) = {_g_text(p, q, r)}»; give it as a fraction.",
            f"For the problem of minimising «{f_t}» subject to «g(x, y) = {_g_text(p, q, r)} ≤ 0», the point "
            f"«P = {P}» lies on the boundary. Solve the stationarity equation «∇f + λ∇g = 0» at «P» for «λ». Give it "
            "as a fraction.",
            f"Let «f(x, y) = {f_t}» and «g(x, y) = {_g_text(p, q, r)}». At the boundary point «P = {P}», find the "
            "multiplier «λ» satisfying «∇f(P) = −λ∇g(P)», as a fraction.",
        ]
    )
    return single_from(
        part,
        stem,
        working=[
            f"∇f(P) = ({tx(gx)}, {tx(gy)}) and ∇g = ({p}, {q}).",
            f"∇f + λ∇g = 0 gives λ = {tx(lam)}.",
        ],
    )


def _medium(rng):
    for _ in attempts(1000, "two constraints"):
        p1, q1 = _grad_pair(rng)
        p2, q2 = _grad_pair(rng)
        if p1 * q2 - p2 * q1 == 0 or (p1, q1) == (-p2, -q2):
            continue
        break
    px, py = _point(rng)
    l1, l2 = _lam(rng), _lam(rng)
    if l1 == l2:
        raise Redraw
    cx, cy = px + (l1 * p1 + l2 * p2) / 2, py + (l1 * q1 + l2 * q2) / 2
    r1, r2 = p1 * px + q1 * py, p2 * px + q2 * py
    gx, gy = 2 * (px - cx), 2 * (py - cy)
    check(gx + l1 * p1 + l2 * p2 == 0 and gy + l1 * q1 + l2 * q2 == 0, "stationarity fails")
    # alternatives
    det = p1 * q2 - p2 * q1

    def solve(a, b, c, d, e, f):  # a l1 + b l2 = e ; c l1 + d l2 = f
        dd = a * d - b * c
        return (Fraction(e * d - b * f, dd), Fraction(a * f - e * c, dd)) if dd else None

    which = rng.choice([1, 2])
    value = (l1, l2)[which - 1]
    sys_sign = solve(p1, p2, q1, q2, gx, gy)
    transposed = solve(p1, q1, p2, q2, -gx, -gy)
    cands = [
        (M_SWAP, (l1, l2)[2 - which]),
        (M_SYS_SIGN, sys_sign[which - 1] if sys_sign else None),
        (M_SYS_T, transposed[which - 1] if transposed else None),
        (M_HALF, value * 2),
        (M_SIGN, -value),
    ]
    part = fraction_part("", value, cands, 1, min_options=3)
    f_t = f"{shifted('x', cx)} + {shifted('y', cy)}"
    g1, g2 = lin_text(p1, q1), lin_text(p2, q2)
    P = f"({dtxt(px)}, {dtxt(py)})"
    lam = "λ₁" if which == 1 else "λ₂"
    stem = rng.choice(
        [
            f"Minimise «f(x, y) = {f_t}» subject to «{g1} ≤ {dtxt(r1)}» and «{g2} ≤ {dtxt(r2)}». Both constraints are "
            f"active at «P = {P}». Find the multiplier «{lam}» for which «∇f(P) + λ₁∇g₁ + λ₂∇g₂ = 0», as a fraction.",
            f"The point «P = {P}» lies on both boundaries «{g1} = {dtxt(r1)}» and «{g2} = {dtxt(r2)}» of a feasible "
            f"region. Solve the stationarity equation for the multipliers of the problem minimising «{f_t}» and give "
            f"«{lam}» as a fraction.",
            f"For the objective «f(x, y) = {f_t}» with constraints «g₁ = {_g_text(p1, q1, r1)} ≤ 0» and "
            f"«g₂ = {_g_text(p2, q2, r2)} ≤ 0», both active at «P = {P}», find «{lam}» in «∇f + λ₁∇g₁ + λ₂∇g₂ = 0» "
            "at «P».",
        ]
    )
    return single_from(
        part,
        stem,
        working=[
            f"∇f(P) = ({tx(gx)}, {tx(gy)}); the two equations are {lin_text(p1, p2, 'λ₁', 'λ₂')} = {tx(-gx)} and {lin_text(q1, q2, 'λ₁', 'λ₂')} = {tx(-gy)}.",
            f"Solving gives λ₁ = {tx(l1)} and λ₂ = {tx(l2)}.",
        ],
    )


_TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (6, 8, 10), (4, 3, 5), (12, 5, 13)]


def _hard(rng):
    a0, b0, h0 = rng.choice(_TRIPLES)
    k = rng.choice([1, 2, 3])
    sa, sb = rng.choice([(1, 1), (1, -1), (-1, 1)])
    a, b, hyp = sa * a0 * k, sb * b0 * k, h0 * k
    r = rng.choice([1, 2, 3, 4, 5])
    lam = Fraction(hyp, 2 * r)
    cands = [
        (M_CIRC, Fraction(hyp, r)),
        (M_R2, Fraction(hyp, 2 * r * r)),
        (M_NORM2, Fraction(hyp * hyp, 2 * r)),
        (M_SUM, Fraction(abs(a) + abs(b), 2 * r)),
        ("num2t-kkt-circle-point-unscaled", Fraction(hyp, 2)),
    ]
    part = fraction_part("", lam, cands, 1)
    obj = lin_text(-a, -b)
    stem = rng.choice(
        [
            f"Minimise «f(x, y) = {obj}» subject to «x² + y² ≤ {r * r}». The minimum lies on the boundary circle. "
            "Find the KKT multiplier «λ ≥ 0» in «∇f + λ∇g = 0», where «g(x, y) = x² + y² − "
            + f"{r * r}», as a fraction.",
            f"The linear function «{obj}» is minimised over the disc «x² + y² ≤ {r * r}». At the minimiser the "
            "constraint is active. Find the multiplier «λ» (with «λ ≥ 0») for the constraint "
            f"«x² + y² − {r * r} ≤ 0», as a fraction.",
            f"A KKT point of the problem minimise «{obj}» over «g(x, y) = x² + y² − {r * r} ≤ 0» has «g = 0» and «λ ≥ 0». Find «λ» "
            "as a fraction.",
        ]
    )
    return single_from(
        part,
        stem,
        working=[
            f"∇f = ({-a}, {-b}) and ∇g = (2x, 2y), so (x, y) = ({tx(Fraction(a, 1) / (2 * lam))}, {tx(Fraction(b, 1) / (2 * lam))}).",
            f"On the circle x² + y² = {r * r}: ({a * a + b * b})/(4λ²) = {r * r}, so λ = {hyp}/{2 * r} = {tx(lam)}.",
        ],
    )


@generator("exd_x_r4h_kkt_conditions", level=20, presets={"exam": dict(tier="exam")})
def exd_x_r4h_kkt_conditions(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    return with_redraw(lambda: _exam(rng), "an exam KKT question")


def _exam(rng):
    al, be = rng.choice([(1, 1), (1, 2), (2, 1), (1, 4), (2, 2)])
    p, q = _grad_pair(rng)
    xs, ys = Fraction(rng.randint(-3, 3)), Fraction(rng.randint(-3, 3))
    lam = _lam(rng)
    a = xs + lam * p / (2 * al)
    b = ys + lam * q / (2 * be)
    c = p * xs + q * ys
    if p * a + q * b <= c:
        raise Redraw
    val = al * (xs - a) ** 2 + be * (ys - b) ** 2
    check(
        val == lam * lam * (Fraction(p * p, 4 * al) + Fraction(q * q, 4 * be)),
        "value disagrees with the multiplier formula",
    )
    viol = p * a + q * b - c
    denom = Fraction(p * p, 2 * al) + Fraction(q * q, 2 * be)
    check(lam == viol / denom, "multiplier formula")
    p_a = fraction_part(
        "The unconstrained minimum violates the constraint, so it is active at the solution. Find the multiplier «λ ≥ 0».",
        lam,
        [
            ("num2t-kkt-lambda-uses-unconstrained", Fraction(viol)),
            ("num2t-kkt-lambda-weights-ignored", viol / Fraction(p * p + q * q, 2)),
            ("num2t-kkt-lambda-plain-length", viol / Fraction(p * p + q * q)),
            (M_HALF, viol / Fraction(p * p + q * q)),
        ],
        1,
        working=[
            f"Stationarity gives x = a − λp/(2α), y = b − λq/(2β); substituting into the constraint gives λ = {tx(lam)}."
        ],
        value_range=(0, None),
    )
    la = ft_symbol("part_a")
    p_b = fraction_part(
        "Use your «λ» to find the constrained minimiser's «x»-coordinate.",
        xs,
        [
            ("num2t-kkt-coordinate-sign", a + lam * p / (2 * al)),
            ("num2t-kkt-coordinate-unscaled", a - lam * p),
            ("num2t-kkt-coordinate-unconstrained", a),
            ("num2t-kkt-coordinate-other", ys),
        ],
        1,
        ft=str(rat(a) - la * rat(Fraction(p, 2 * al))),
        working=[f"x = {tx(a)} − λ·{p}/{2 * al} = {tx(xs)}."],
    )
    p_c = fraction_part(
        "Use your «λ» to find the constrained minimiser's «y»-coordinate.",
        ys,
        [
            ("num2t-kkt-coordinate-sign", b + lam * q / (2 * be)),
            ("num2t-kkt-coordinate-unscaled", b - lam * q),
            ("num2t-kkt-coordinate-unconstrained", b),
            ("num2t-kkt-coordinate-other", xs),
        ],
        1,
        ft=str(rat(b) - la * rat(Fraction(q, 2 * be))),
        working=[f"y = {tx(b)} − λ·{q}/{2 * be} = {tx(ys)}."],
    )
    xb, yc = ft_symbol("part_b"), ft_symbol("part_c")
    p_d = fraction_part(
        "Find the minimum value of «f» on the feasible region, using your coordinates.",
        val,
        [
            ("num2t-kkt-value-unconstrained", Fraction(0)),
            ("num2t-kkt-value-unweighted", (xs - a) ** 2 + (ys - b) ** 2),
            ("num2t-kkt-value-lambda-squared", lam * lam),
            ("num2t-kkt-value-not-squared", al * abs(xs - a) + be * abs(ys - b)),
        ],
        2,
        ft=str(rat(al) * (xb - rat(a)) ** 2 + rat(be) * (yc - rat(b)) ** 2),
        working=[f"f = {al}({tx(xs - a)})² + {be}({tx(ys - b)})² = {tx(val)}."],
        value_range=(0, None),
    )
    f_t = f"{shifted('x', a, 2, al)} + {shifted('y', b, 2, be)}"
    cons = f"{lin_text(p, q)} ≤ {dtxt(c)}"
    stem = rng.choice(
        [
            f"Minimise «f(x, y) = {f_t}» subject to «{cons}».",
            f"A designer wants the point nearest, in the weighted sense «f(x, y) = {f_t}», to the centre «({dtxt(a)}, {dtxt(b)})» "
            f"that satisfies «{cons}».",
            f"Consider the problem: minimise «{f_t}» over the region «{cons}».",
        ]
    )
    return finish_parts([p_a, p_b, p_c, p_d], stem)
