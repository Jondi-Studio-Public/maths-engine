"""Generates `s2a1_jacobian_cov` questions (easy, medium, hard) and the exam chain `exd_x_s2a1_jacobian_cov`.

``uc2-jacobian-change-of-variables`` (``multivariable-and-vector-calculus``, answer ``expression``): evaluate a double
integral over a region in the xy-plane by a change of variables, dA = |∂(x,y)/∂(u,v)| du dv. This is the integration
the numeric-Jacobian skill ``jacobian-two-variable`` stops short of: here the learner finds the Jacobian of the
substitution (or of its inverse), maps the region, substitutes into the integrand and integrates.

* easy: a linear map x = au + bv, y = cu + dv is GIVEN together with the rectangle in the uv-plane it sends onto a
  parallelogram; the Jacobian is the constant ad - bc and may be negative.
* medium: the INVERSE is given, u = αx + βy and v = γx + δy, with the region bounded by four lines; the learner needs
  the reciprocal ∂(x,y)/∂(u,v) = 1/(αδ - βγ) and has to write x and y in terms of u and v for the integrand.
* hard: (i) the hyperbolic coordinates u = xy and v = y/x on a region bounded by two hyperbolas and two lines, where
  the Jacobian 1/(2v) is not constant and the answer involves ln; (ii) a linear map of the triangle u, v ≥ 0, u + v ≤ k;
  (iii) a region bounded by four lines with NO substitution given, so the learner must choose one.
* exam (``exd_x_s2a1_jacobian_cov``): a plate between four lines with density ρ = k + m·x: (a) the Jacobian
  ∂(x,y)/∂(u,v) (signed), (b) the area (follows through from (a)), (c) the mass (follows through from (a)), (d) the
  average density (follows through from (b) and (c)).

Each key is worked out twice. The exact route changes variables and integrates term by term. The second route lives in
tests/test_s2a1_jacobian.py: a midpoint sum on a fine grid of the region itself, tested point by point against its xy
inequalities, which never uses a Jacobian or a substitution.
"""

from __future__ import annotations

import math

import numpy as np
import sympy

from ..core import VerificationError, attempts, check, generator
from ._s2_options import keep_answer_shape
from ._s2a1_common import exact_answer, exact_part, exact_wrong, grid_integral, stem, tidy
from ._ucv1_calc_markup import tex_value

x, y, u, v = sympy.symbols("x y u v", real=True)
S = sympy.Integer


# ---------------------------------------------------------------------------------------------------------------
# exact integration over rectangles and triangles in the uv-plane


def mono_int(expr, var, lo, hi):
    """Exact integral of a sum of c·var^e terms (e rational, e = -1 gives a logarithm) between positive limits."""
    total = S(0)
    for term in sympy.Add.make_args(sympy.expand(expr, log=False)):
        coeff, dep = term.as_independent(var, as_Add=False)
        if dep == 1:
            total += coeff * (hi - lo)
            continue
        base, power = dep.as_base_exp()
        check(base == var, "a term is not a power of the variable")
        if power == -1:
            total += coeff * sympy.log(hi / lo)
        else:
            total += coeff * (hi ** (power + 1) - lo ** (power + 1)) / (power + 1)
    return total


def uv_integral(expr, u_lims, v_lims):
    """∬ expr du dv with v between v_lims (numbers or functions of u) and u between u_lims (outer)."""
    inner = mono_int(expr, v, *v_lims)
    return sympy.expand(mono_int(inner, u, *u_lims), log=False)


def lin(p, q, a="u", b="v") -> str:
    """'p·a + q·b' as text: 2u + v, -u + 3v, u - v, 3v."""
    parts = []
    for coeff, name in ((p, a), (q, b)):
        if coeff == 0:
            continue
        mag = "" if abs(coeff) == 1 else str(abs(coeff))
        if not parts:
            parts.append(("-" if coeff < 0 else "") + mag + name)
        else:
            parts.append(f"{'-' if coeff < 0 else '+'} {mag}{name}")
    return " ".join(parts)


def poly_text(expr) -> str:
    return str(sympy.expand(expr)).replace("**2", "²").replace("**3", "³").replace("*", "")


def f_text(f) -> str:
    text = poly_text(f)
    return f"({text})" if len(sympy.Add.make_args(sympy.expand(f))) > 1 else text


# ---------------------------------------------------------------------------------------------------------------
# easy: a linear map is given, with the rectangle it carries onto a parallelogram


_F_EASY = [x, y, x + y, x + 2 * y, 2 * x - y, x**2, y**2, x * y, x**2 + y**2, x + y + 1, 1]


def _easy(rng):
    for _ in attempts(200, "a map with a usable determinant"):
        a, b, c, d = rng.randint(1, 4), rng.randint(-2, 3), rng.randint(-2, 3), rng.randint(1, 4)
        D = a * d - b * c
        if D == 0 or abs(D) > 9:
            continue
        break
    if rng.random() < 0.5 and D > 0:
        for _ in attempts(200, "a map with a negative determinant"):
            a, b, c, d = rng.randint(1, 4), rng.randint(-2, 3), rng.randint(-2, 3), rng.randint(1, 4)
            D = a * d - b * c
            if D < 0 and abs(D) <= 9:
                break
    p, q = rng.choice([1, 2, 3]), rng.choice([1, 2, 3])
    xs, ys = a * u + b * v, c * u + d * v
    corners = [(0, 0), (p, 0), (p, q), (0, q)]
    verts = [(a * i + b * j, c * i + d * j) for i, j in corners]
    f = rng.choice(_F_EASY)
    f_uv = sympy.expand(f.subs({x: xs, y: ys}, simultaneous=True)) if f != 1 else S(1)
    value = sympy.Rational(abs(D)) * uv_integral_poly(f_uv, (0, p), (0, q))
    check(value > 0, "the integral is not positive")

    def uv(expr, weight=abs(D)):
        return sympy.Rational(weight) * uv_integral_poly(expr, (0, p), (0, q))

    xmin, xmax = min(i for i, _ in verts), max(i for i, _ in verts)
    ymin, ymax = min(j for _, j in verts), max(j for _, j in verts)
    cands = [
        ("jacobian-omitted", uv(f_uv, 1)),
        ("jacobian-inverted", value / (D * D)),
        ("jacobian-sign-ignored", -value if D < 0 else None),
        ("cov-integrand-not-substituted", uv(f.subs({x: u, y: v}, simultaneous=True) if f != 1 else S(1))),
        ("cov-limits-from-region", abs(D) * uv_integral_poly(f_uv, (xmin, xmax), (ymin, ymax))),
        ("cov-integrand-dropped", abs(D) * p * q) if f != 1 else None,
    ]
    cands = [item for item in cands if item is not None and item[1] is not None]
    verts_text = ", ".join(f"«({i}, {j})»" for i, j in verts[:-1]) + f" and «({verts[-1][0]}, {verts[-1][1]})»"
    if f == 1:
        text = (
            f"Let R be the parallelogram with vertices {verts_text}. The substitution «x = {lin(a, b)}», «y = {lin(c, d)}» "
            f"maps the rectangle «0 ≤ u ≤ {p}», «0 ≤ v ≤ {q}» onto R. Find the area of R."
        )
    else:
        text = (
            f"Let R be the parallelogram with vertices {verts_text}. The substitution «x = {lin(a, b)}», «y = {lin(c, d)}» "
            f"maps the rectangle «0 ≤ u ≤ {p}», «0 ≤ v ≤ {q}» onto R. Evaluate «∬ {f_text(f)} dA» over R."
        )
    working = [
        f"∂(x,y)/∂(u,v) = ({a})({d}) - ({b})({c}) = {D}, so dA = {abs(D)} du dv.",
        f"Substitute x = {lin(a, b)} and y = {lin(c, d)} into the integrand and integrate over the rectangle.",
    ]
    return text, value, cands, working


def uv_integral_poly(expr, u_lims, v_lims):
    expr = sympy.expand(expr)
    inner = (
        sum(
            (c * (v_lims[1] ** (n + 1) - v_lims[0] ** (n + 1)) / (n + 1) for (n,), c in sympy.Poly(expr, v).terms()),
            S(0),
        )
        if expr.has(v)
        else expr * (v_lims[1] - v_lims[0])
    )
    inner = sympy.expand(inner)
    if not inner.has(u):
        return sympy.nsimplify(inner * (u_lims[1] - u_lims[0]))
    return sympy.nsimplify(
        sum(
            (c * (u_lims[1] ** (n + 1) - u_lims[0] ** (n + 1)) / (n + 1) for (n,), c in sympy.Poly(inner, u).terms()),
            S(0),
        )
    )


# ---------------------------------------------------------------------------------------------------------------
# medium and hard (iii): four lines, u = αx + βy, v = γx + δy


def _line_system(rng, small=True):
    for _ in attempts(300, "a line system"):
        al, be, ga, de = (
            rng.randint(1, 3),
            rng.choice([-2, -1, 1, 2, 3]),
            rng.randint(1, 3),
            rng.choice([-2, -1, 1, 2, 3]),
        )
        det = al * de - be * ga
        if det == 0 or abs(det) > 6:
            continue
        if math.gcd(al, be) != 1 or math.gcd(ga, de) != 1:
            continue
        if (al, be) == (ga, de) or (al, be) == (-ga, -de):
            continue
        return al, be, ga, de, det
    raise VerificationError("no line system")


def _lines_problem(rng, give_substitution: bool, tier_hard: bool):
    al, be, ga, de, det = _line_system(rng)
    u0 = rng.randint(-1, 2)
    u1 = u0 + rng.randint(1, 3)
    v0 = rng.randint(-1, 2)
    v1 = v0 + rng.randint(1, 3)
    # x, y from u, v
    xs = sympy.Rational(de, det) * u - sympy.Rational(be, det) * v
    ys = -sympy.Rational(ga, det) * u + sympy.Rational(al, det) * v
    pool = (
        [x, y, x + y, x * y, x**2, y**2, x + 2 * y, 2 * x - y, x**2 + y**2]
        if tier_hard
        else [x, y, x + y, x * y, x**2, y**2, x + 2 * y]
    )
    f = rng.choice(pool)
    # sometimes make the integrand the product of the two line forms, which becomes u v at once
    if not tier_hard and rng.random() < 0.3:
        f = (al * x + be * y) * (ga * x + de * y)
    f_uv = sympy.expand(f.subs({x: xs, y: ys}, simultaneous=True))
    value = uv_integral_poly(f_uv, (u0, u1), (v0, v1)) / abs(det)
    check(value > 0, "the integral is not positive")

    wrong_inv_x = sympy.Rational(de, det) * u + sympy.Rational(be, det) * v
    wrong_inv_y = sympy.Rational(ga, det) * u + sympy.Rational(al, det) * v
    f_slip = sympy.expand(f.subs({x: wrong_inv_x, y: wrong_inv_y}, simultaneous=True))
    cands = [
        ("jacobian-inverted", value * det * det),
        ("jacobian-omitted", value * abs(det)),
        (
            "cov-integrand-not-substituted",
            uv_integral_poly(f.subs({x: u, y: v}, simultaneous=True), (u0, u1), (v0, v1)) / abs(det),
        ),
        ("cov-inversion-slip", uv_integral_poly(f_slip, (u0, u1), (v0, v1)) / abs(det)),
        ("cov-integrand-dropped", sympy.Rational((u1 - u0) * (v1 - v0), abs(det))),
        ("jacobian-sign-ignored", -value) if det < 0 else None,
    ]
    cands = [item for item in cands if item is not None]
    U_line = lin(al, be, "x", "y")
    V_line = lin(ga, de, "x", "y")
    words = f"bounded by the lines «{U_line} = {u0}», «{U_line} = {u1}», «{V_line} = {v0}» and «{V_line} = {v1}»"
    sub = f" Use the substitution «u = {U_line}», «v = {V_line}»." if give_substitution else ""
    if f == 1:
        raise VerificationError("f is 1")
    text = f"Evaluate «∬ {f_text(f)} dA», where R is the region {words}.{sub}"
    working = [
        f"∂(u,v)/∂(x,y) = ({al})({de}) - ({be})({ga}) = {det}, so ∂(x,y)/∂(u,v) = 1/({det}) and dA = du dv/{abs(det)}.",
        f"Solve for x = {poly_text(xs)} and y = {poly_text(ys)} and integrate over {u0} ≤ u ≤ {u1}, {v0} ≤ v ≤ {v1}.",
    ]
    return text, value, cands, working


def _medium(rng):
    return _lines_problem(rng, True, False)


# ---------------------------------------------------------------------------------------------------------------
# hard (i): hyperbolic coordinates u = xy, v = y/x


_F_HYP = [S(1), x * y, y**2, x**2, y / x, x**2 + y**2]


def _hyperbolic(rng):
    a, b = rng.choice([(1, 2), (1, 3), (1, 4), (2, 5), (1, 5), (2, 4), (3, 6)])
    c, d = rng.choice([(1, 2), (1, 3), (2, 3), (1, 4)])
    f = rng.choice(_F_HYP)
    # x = sqrt(u/v), y = sqrt(u v):  dA = du dv / (2 v)
    f_uv = sympy.simplify(f.subs({x: sympy.sqrt(u / v), y: sympy.sqrt(u * v)}, simultaneous=True))
    f_uv = sympy.expand(sympy.powsimp(f_uv, force=True))
    weight = lambda w: sympy.expand(f_uv * w)  # noqa: E731
    value = uv_integral(weight(1 / (2 * v)), (S(a), S(b)), (S(c), S(d)))
    check(value.is_real and float(value) > 0, "the integral is not positive")

    cands = [
        ("jacobian-omitted", uv_integral(weight(1), (S(a), S(b)), (S(c), S(d)))),
        ("jacobian-inverted", uv_integral(weight(2 * v), (S(a), S(b)), (S(c), S(d)))),
        ("jacobian-factor-slip", value * 2),
        ("jacobian-variable-swapped", uv_integral(weight(1 / (2 * u)), (S(a), S(b)), (S(c), S(d)))),
    ]
    cands = [item for item in cands if item is not None]
    if f == 1:
        text = (
            f"Let R be the region in the first quadrant bounded by the curves «xy = {a}», «xy = {b}» and the lines «y = {c if c > 1 else ''}x», «y = {d}x». "
            f"Use the substitution «u = xy», «v = y/x» to find the area of R."
        )
    else:
        text = (
            f"Evaluate «∬ {f_text(f)} dA», where R is the region in the first quadrant bounded by the curves «xy = {a}», «xy = {b}» "
            f"and the lines «y = {c if c > 1 else ''}x» and «y = {d}x». Use the substitution «u = xy», «v = y/x»."
        )
    working = [
        "x = √(u/v) and y = √(uv), so ∂(x,y)/∂(u,v) = 1/(2v).",
        f"The region becomes {a} ≤ u ≤ {b}, {c} ≤ v ≤ {d}.",
    ]
    return text, value, cands, working


# ---------------------------------------------------------------------------------------------------------------
# hard (ii): a linear map of the triangle u, v ≥ 0, u + v ≤ k


def _triangle(rng):
    for _ in attempts(300, "a triangle map"):
        a, b, c, d = rng.randint(1, 3), rng.randint(-2, 3), rng.randint(-2, 3), rng.randint(1, 3)
        D = a * d - b * c
        if D == 0 or abs(D) > 7:
            continue
        break
    k = rng.choice([1, 2, 3])
    xs, ys = a * u + b * v, c * u + d * v
    verts = [(0, 0), (a * k, c * k), (b * k, d * k)]
    f = rng.choice([x, y, x + y, x**2, y**2, x * y, x + 2 * y, x**2 + y**2])
    f_uv = sympy.expand(f.subs({x: xs, y: ys}, simultaneous=True))
    value = abs(D) * uv_integral_poly_triangle(f_uv, k)
    check(value > 0, "the integral is not positive")

    rect = abs(D) * uv_integral_poly(f_uv, (0, k), (0, k))
    cands = [
        ("jacobian-omitted", value / abs(D)),
        ("jacobian-inverted", value / (D * D)),
        ("cov-triangle-as-rectangle", rect),
        (
            "cov-integrand-not-substituted",
            abs(D) * uv_integral_poly_triangle(f.subs({x: u, y: v}, simultaneous=True), k),
        ),
        ("jacobian-sign-ignored", -value) if D < 0 else None,
    ]
    cands = [item for item in cands if item is not None]
    pts = ", ".join(f"«({i}, {j})»" for i, j in verts[:-1]) + f" and «({verts[-1][0]}, {verts[-1][1]})»"
    text = (
        f"Let R be the triangle with vertices {pts}. The substitution «x = {lin(a, b)}», «y = {lin(c, d)}» maps the triangle "
        f"«u ≥ 0», «v ≥ 0», «u + v ≤ {k}» onto R. Evaluate «∬ {f_text(f)} dA» over R."
    )
    working = [
        f"∂(x,y)/∂(u,v) = ({a})({d}) - ({b})({c}) = {D}, so dA = {abs(D)} du dv.",
        f"Integrate over the triangle u ≥ 0, v ≥ 0, u + v ≤ {k}: v runs from 0 to {k} - u.",
    ]
    return text, value, cands, working


def uv_integral_poly_triangle(expr, k):
    expr = sympy.expand(expr)
    inner = (
        sum((c * (k - u) ** (n + 1) / (n + 1) for (n,), c in sympy.Poly(expr, v).terms()), S(0))
        if expr.has(v)
        else expr * (k - u)
    )
    inner = sympy.expand(inner)
    if not inner.has(u):
        return sympy.nsimplify(inner * k)
    return sympy.nsimplify(sum((c * k ** (n + 1) / (n + 1) for (n,), c in sympy.Poly(inner, u).terms()), S(0)))


def _hard(rng):
    pick = rng.random()
    if pick < 0.4:
        return _hyperbolic(rng)
    if pick < 0.7:
        return _triangle(rng)
    return _lines_problem(rng, False, True)


# ---------------------------------------------------------------------------------------------------------------
# the generator


@generator(
    "s2a1_jacobian_cov",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@keep_answer_shape
def s2a1_jacobian_cov(rng, tier="easy"):
    build = {"easy": _easy, "medium": _medium, "hard": _hard}[tier]
    for _ in attempts(80, "a change-of-variables integral with enough distractors"):
        try:
            text, value, cands, working = build(rng)
        except VerificationError:
            continue
        value = sympy.sympify(value)
        wrong = exact_wrong(value, cands, 4, allow_negative=True)
        if len(wrong) >= 3 and float(value) > 0:
            break
    kind, machine, answer_text = exact_answer(value)
    return dict(
        **stem(text),
        answer=machine,
        answer_text=answer_text,
        answer_latex=tex_value(tidy(value)),
        answer_kind=kind,
        distractors=wrong,
        working=working,
    )


# ---------------------------------------------------------------------------------------------------------------
# exam chain


def _exam_once(rng):
    al, be, ga, de, det = _line_system(rng)
    u0 = rng.randint(0, 2)
    u1 = u0 + rng.randint(1, 3)
    v0 = rng.randint(0, 2)
    v1 = v0 + rng.randint(1, 3)
    xs = sympy.Rational(de, det) * u - sympy.Rational(be, det) * v
    ys = -sympy.Rational(ga, det) * u + sympy.Rational(al, det) * v
    k, m = rng.randint(2, 6), rng.randint(1, 3)
    which = rng.choice(["x", "y"])
    coord_uv = xs if which == "x" else ys
    J = sympy.Rational(1, det)  # ∂(x,y)/∂(u,v), signed
    uv_area = sympy.Rational((u1 - u0) * (v1 - v0))
    X = uv_integral_poly(coord_uv, (u0, u1), (v0, v1))  # ∬ coordinate du dv
    area = abs(J) * uv_area
    mass = abs(J) * (k * uv_area + m * X)
    check(mass > 0 and float(X) > 0, "the plate has no mass or a negative moment")
    avg = mass / area
    xy = [(xs.subs({u: uu, v: vv}), ys.subs({u: uu, v: vv})) for uu, vv in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]

    U_line, V_line = lin(al, be, "x", "y"), lin(ga, de, "x", "y")
    a_c = [
        ("jacobian-inverted", det),
        ("jacobian-sign-dropped", abs(J)) if J < 0 else None,
        ("jacobian-cross-term-dropped", sympy.Rational(de, det) * sympy.Rational(al, det)),
        ("jacobian-omitted", S(1)),
    ]
    a_c = [item for item in a_c if item is not None]
    b_c = [
        ("jacobian-omitted", uv_area),
        ("jacobian-inverted", uv_area * abs(det)),
        (
            "cov-limits-from-region",
            (max(i for i, _ in xy) - min(i for i, _ in xy)) * (max(j for _, j in xy) - min(j for _, j in xy)),
        ),
    ]
    b_c = [(t, sympy.nsimplify(val)) for t, val in b_c]
    c_c = [
        ("jacobian-omitted", k * uv_area + m * X),
        ("triple-integral-density-term-dropped", k * area),
        ("jacobian-inverted", abs(J) ** -1 * (k * uv_area + m * X)),
        (
            "cov-integrand-not-substituted",
            abs(J) * (k * uv_area + m * uv_integral_poly(u if which == "x" else v, (u0, u1), (v0, v1))),
        ),
        ("triple-integral-moment-not-divided", m * abs(J) * X),
    ]
    d_c = [
        ("triple-integral-mass-not-divided", mass),
        ("triple-integral-moment-inverted", area / mass),
        ("triple-integral-density-term-dropped", sympy.Integer(k)),
        ("jacobian-omitted", (k * uv_area + m * X) / area),
    ]
    ft_b = f"Abs(part_a)*{uv_area}"
    ft_c = f"Abs(part_a)*({k * uv_area + m * X})"
    parts = [
        exact_part(
            "Find the Jacobian «∂(x, y)/∂(u, v)» of the substitution.",
            J,
            2,
            [
                f"x and y in terms of u and v: «x = {poly_text(xs)}», «y = {poly_text(ys)}».",
                "The Jacobian is the determinant of the matrix of partial derivatives of x and y.",
            ],
            a_c,
            allow_negative=True,
        ),
        exact_part(
            "Find the area of the plate, in square metres.",
            area,
            2,
            ["Area = |∂(x,y)/∂(u,v)| times the area of the rectangle in the uv-plane."],
            b_c,
            follow_through=ft_b,
            sensible_range=(0, None),
        ),
        exact_part(
            "Find the mass of the plate, in kilograms.",
            mass,
            3,
            [f"Mass = |J| ∬ ({k} + {m if m > 1 else ''}{which}) du dv with {which} written in terms of u and v."],
            c_c,
            follow_through=ft_c,
            sensible_range=(0, None),
        ),
        exact_part(
            "Find the average density of the plate, in kilograms per square metre.",
            avg,
            1,
            ["Average density = mass / area."],
            d_c,
            follow_through="part_c/part_b",
            sensible_range=(0, None),
        ),
    ]
    check(all(parts), "a part has too few distractors")
    return dict(
        **stem(
            f"A thin plate occupies the region R bounded by the lines «{U_line} = {u0}», «{U_line} = {u1}», «{V_line} = {v0}» and "
            f"«{V_line} = {v1}», with lengths in metres. Its density at the point «(x, y)» is «ρ = {k} + {m if m > 1 else ''}{which}» "
            f"kilograms per square metre. Use the substitution «u = {U_line}», «v = {V_line}»."
        ),
        parts=parts,
    )


@generator("exd_x_s2a1_jacobian_cov", level=20, presets={"exam": dict(tier="exam")})
def exd_x_s2a1_jacobian_cov(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(30, "an exam change-of-variables chain"):
        try:
            return _exam_once(rng)
        except VerificationError:
            continue
    raise AssertionError("unreachable")
