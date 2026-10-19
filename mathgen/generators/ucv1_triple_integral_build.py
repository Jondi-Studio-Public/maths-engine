"""Generates `triple_integral` questions (easy, medium, hard) and the exam chain `exd_x_uc_triple_integral`.

``ucv1-triple-integral`` (``multivariable-and-vector-calculus``, answer ``fraction``): exact triple integrals.

* easy: a separable polynomial, a product of three one-variable factors, over a box.
* medium: a region whose top is a plane over a rectangle (0 ≤ z ≤ px + qy + r), or the volume of the tetrahedron cut
  from the first octant by a plane written in the form px + qy + rz = d.
* hard: a non-separable integrand over such a tetrahedron; a solid described by a chain of inequalities such as
  0 ≤ y ≤ x ≤ z ≤ c, where the learner has to write the limits; or an iterated integral whose order has to be
  changed (the value only is asked).
* exam (``exd_x_uc_triple_integral``): a solid of density ρ = k + p·w (w one of x, y, z), either the tetrahedron or
  a wedge 0 ≤ z ≤ x + y over a rectangle: (a) the volume, (b) the mass (follows through from (a)), (c) one
  coordinate of the centre of mass (follows through from (b)), (d) the average density (follows through from
  (a) and (b)).

Each answer is worked out twice. The exact route integrates polynomials step by step between the stated limits. The
second route is closed-form (the Dirichlet integral for a tetrahedron, the moments of an ordered simplex) or, for
the box and the wedge, six-point Gauss-Legendre in each variable.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, join_terms, pick_distractors, rational_text
from ._ucv1_multiint import Poly, gauss_nested
from ._ucv1_texsay import say_marked, tex

X, Y, Z = Poly.var(0), Poly.var(1), Poly.var(2)
_ZERO = Poly()
_VARS = (X, Y, Z)
_NAMES = "xyz"


def _nonzero(rng, lo: int, hi: int) -> int:
    return rng.choice([v for v in range(lo, hi + 1) if v != 0])


def _triple(f: Poly, bounds) -> Fraction:
    """Exact iterated integral: innermost z, then y, then x, each between the polynomial limits `bounds[i]`."""
    g = f
    for i in (2, 1, 0):
        g = g.integrate(i, *bounds[i])
    return g.t.get((0, 0, 0), Fraction(0))


def _numeric(f: Poly, bounds) -> float:
    return gauss_nested(lambda a, b, c: float(f.value((a, b, c))), bounds)


def _check_close(exact: Fraction, numeric: float, what: str) -> None:
    check(abs(numeric - float(exact)) <= 1e-8 * max(1.0, abs(numeric)), f"{what}: {numeric} vs {exact}")


def _const_bounds(box):
    return [(Poly.const(lo), Poly.const(hi)) for lo, hi in box]


def _tetra_bounds(a: int, b: int, c: int):
    return [
        (_ZERO, Poly.const(a)),
        (_ZERO, b * (1 - X * Fraction(1, a))),
        (_ZERO, c * (1 - X * Fraction(1, a) - Y * Fraction(1, b))),
    ]


def _dirichlet(f: Poly, a: int, b: int, c: int) -> Fraction:
    """∭ f over x/a + y/b + z/c ≤ 1 (x, y, z ≥ 0), term by term: a^(i+1) b^(j+1) c^(k+1) i! j! k! / (i+j+k+3)!"""
    total = Fraction(0)
    for (i, j, k), coef in f.t.items():
        total += coef * Fraction(
            a ** (i + 1) * b ** (j + 1) * c ** (k + 1) * math.factorial(i) * math.factorial(j) * math.factorial(k),
            math.factorial(i + j + k + 3),
        )
    return total


def _intercept_plane(a: int, b: int, c: int):
    """(p, q, r, d) with px + qy + rz = d meeting the axes at a, b, c."""
    big = math.lcm(a, b, c)
    return big // a, big // b, big // c, big


def _plane_text(a: int, b: int, c: int) -> str:
    p, q, r, d = _intercept_plane(a, b, c)
    return f"{(p * X + q * Y + r * Z).text()} = {d}"


def _f_text(f: Poly) -> str:
    text = f.text()
    return f"({text})" if len(f.t) > 1 else text


# ---------------------------------------------------------------------------------------------------------------
# easy: a separable polynomial over a box


def _factor(rng, var: Poly) -> Poly:
    shape = rng.choice(["power", "linear", "linear", "shift"])
    if shape == "power":
        return var ** rng.choice([1, 2, 2, 3])
    if shape == "linear":
        return rng.choice([1, 2, 3]) * var + _nonzero(rng, -3, 3)
    return var**2 + _nonzero(rng, -2, 2)


def _factor_text(factor: Poly) -> str:
    text = factor.text()
    return f"({text})" if len(factor.t) > 1 else text


def _one_d(p: Poly, i: int, lo: int, hi: int) -> Fraction:
    anti = p.antiderivative(i)
    point = lambda v: tuple(v if n == i else 0 for n in range(3))  # noqa: E731
    return anti.value(point(hi)) - anti.value(point(lo))


def _easy(rng):
    for _ in attempts(300, "an easy triple integral"):
        factors = [_factor(rng, v) for v in _VARS]
        box = []
        for _i in range(3):
            lo = rng.randint(-2, 2)
            box.append((lo, lo + rng.randint(1, 3)))
        parts = [_one_d(factors[i], i, *box[i]) for i in range(3)]
        value = parts[0] * parts[1] * parts[2]
        if value != 0 and len({str(f.text()) for f in factors}) == 3 and value.denominator <= 60:
            break
    f = factors[0] * factors[1] * factors[2]
    _check_close(value, _numeric(f, _const_bounds(box)), "box integral")
    # the wrong routes
    wrong_antiderivative = []
    for i in range(3):
        slip = Poly({tuple(e + 1 if n == i else e for n, e in enumerate(k)): c for k, c in factors[i].t.items()})
        wrong_antiderivative.append(slip)
    point = lambda i, v: tuple(v if n == i else 0 for n in range(3))  # noqa: E731
    slip_value = Fraction(1)
    for i in range(3):
        slip_value *= wrong_antiderivative[i].value(point(i, box[i][1])) - wrong_antiderivative[i].value(
            point(i, box[i][0])
        )
    upper_only = Fraction(1)
    for i in range(3):
        upper_only *= factors[i].antiderivative(i).value(point(i, box[i][1]))
    candidates = [
        ("triple-integral-sum-not-product", parts[0] + parts[1] + parts[2]),
        ("triple-integral-forgot-lower-bound", upper_only),
        ("triple-integral-power-rule-slip", slip_value),
        (
            "triple-integral-variable-treated-constant",
            parts[0] * parts[1] * (box[2][1] - box[2][0]) * factors[2].value((0, 0, box[2][1])),
        ),
    ]
    text = (
        f"Evaluate the triple integral «∭ {_factor_text(factors[0])}{_factor_text(factors[1])}"
        f"{_factor_text(factors[2])} dV» over the box «{box[0][0]} ≤ x ≤ {box[0][1]}, "
        f"{box[1][0]} ≤ y ≤ {box[1][1]}, {box[2][0]} ≤ z ≤ {box[2][1]}»."
    )
    return dict(text=text, value=value, candidates=candidates)


# ---------------------------------------------------------------------------------------------------------------
# medium


def _medium_top(rng):
    for _ in attempts(300, "a medium triple integral over a sloping top"):
        a, b = rng.randint(1, 3), rng.randint(1, 4)
        p, q, r = rng.choice([0, 1, 1, 2]), rng.choice([0, 1, 1, 2]), rng.choice([0, 0, 1, 2])
        if p == 0 and q == 0:
            continue
        f = rng.choice([Poly.const(1), X, Y, Z, X + Y, X * Y, X * Z, Y * Z, Z * Z, X + Z])
        top = p * X + q * Y + r
        bounds = [(_ZERO, Poly.const(a)), (_ZERO, Poly.const(b)), (_ZERO, top)]
        value = _triple(f, bounds)
        if value != 0 and value.denominator <= 60:
            break
    _check_close(value, _numeric(f, bounds), "sloping-top integral")
    # wrong routes
    f_on_top = f.compose({2: top})
    substituted = _triple(f_on_top, [(_ZERO, Poly.const(a)), (_ZERO, Poly.const(b))] + [(_ZERO, Poly.const(1))])
    height = top.value((a, b, 0))
    const_top = _triple(f, [(_ZERO, Poly.const(a)), (_ZERO, Poly.const(b)), (_ZERO, Poly.const(height))])
    swapped = _triple(f, [(_ZERO, Poly.const(b)), (_ZERO, Poly.const(a)), (_ZERO, top)])
    half_z = _triple(f, [(_ZERO, Poly.const(a)), (_ZERO, Poly.const(b)), (_ZERO, top * Fraction(1, 2))])
    candidates = [
        ("triple-integral-substituted-not-integrated", substituted),
        ("triple-integral-limit-as-constant", const_top),
        ("triple-integral-rectangle-swapped", swapped),
        ("triple-integral-height-halved", half_z),
    ]
    region = f"«0 ≤ x ≤ {a}, 0 ≤ y ≤ {b}, 0 ≤ z ≤ {top.text()}»"
    if f == Poly.const(1):
        text = rng.choice(
            [f"Find the volume of the solid E given by {region}.", f"Evaluate «∭ dV» over the solid E: {region}."]
        )
    else:
        text = rng.choice(
            [
                f"Evaluate «∭ {_f_text(f)} dV» over the solid E: {region}.",
                f"Find «∭ {_f_text(f)} dV», where E is the solid {region}.",
            ]
        )
    return dict(text=text, value=value, candidates=candidates)


def _medium_tetra(rng):
    a, b, c = rng.randint(1, 6), rng.randint(1, 6), rng.randint(1, 6)
    p, q, r, d = _intercept_plane(a, b, c)
    value = Fraction(a * b * c, 6)
    exact = _triple(Poly.const(1), _tetra_bounds(a, b, c))
    check(exact == value, "tetrahedron volume: integration disagrees with abc/6")
    candidates = [
        ("triple-integral-bounding-box", Fraction(a * b * c)),
        ("triple-integral-pyramid-factor", Fraction(a * b * c, 2)),
        ("triple-integral-base-not-halved", Fraction(a * b * c, 3)),
        ("triple-integral-coefficients-as-intercepts", Fraction(p * q * r, 6)),
    ]
    text = rng.choice(
        [
            f"Find the volume of the tetrahedron in the first octant bounded by the coordinate planes and the plane «{_plane_text(a, b, c)}».",
            f"The plane «{_plane_text(a, b, c)}» and the three coordinate planes bound a tetrahedron. Find its volume.",
        ]
    )
    return dict(text=text, value=value, candidates=candidates)


# ---------------------------------------------------------------------------------------------------------------
# hard


_HARD_F = [
    X + Y + Z,
    X * Y,
    X * Z,
    Y * Z,
    2 * X + Y * Z,
    X * Y + Z,
    X + 2 * Y + 3 * Z,
    X * Y + Y * Z,
    X * Z + Y * Y,
    X * Y * Z,
    X * X + Y * Z,
    Y * Z + 2 * X * Z,
]


def _hard_tetra(rng):
    for _ in attempts(100, "a hard tetrahedron integral"):
        a, b, c = rng.randint(1, 4), rng.randint(1, 4), rng.randint(1, 4)
        f = rng.choice(_HARD_F)
        value = _dirichlet(f, a, b, c)
        if value.denominator <= 400:
            break
    exact = _triple(f, _tetra_bounds(a, b, c))
    check(exact == value, "tetrahedron integral: iterated integration disagrees with the Dirichlet formula")
    box = _triple(f, _const_bounds([(0, a), (0, b), (0, c)]))
    no_y = _triple(
        f, [(_ZERO, Poly.const(a)), (_ZERO, Poly.const(b)), (_ZERO, c * (1 - X * Fraction(1, a) - Y * Fraction(1, b)))]
    )
    no_x = _triple(
        f, [(_ZERO, Poly.const(a)), (_ZERO, b * (1 - X * Fraction(1, a))), (_ZERO, c * (1 - X * Fraction(1, a)))]
    )
    slab = _triple(f, [(_ZERO, Poly.const(a)), (_ZERO, b * (1 - X * Fraction(1, a))), (_ZERO, Poly.const(c))])
    candidates = [
        ("triple-integral-bounding-box", box),
        ("triple-integral-limit-not-dependent", no_y),
        ("triple-integral-limit-missing-variable", no_x),
        ("triple-integral-top-not-slanted", slab),
    ]
    text = (
        f"Evaluate «∭ {_f_text(f)} dV» over the tetrahedron in the first octant bounded by the coordinate planes "
        f"and the plane «{_plane_text(a, b, c)}»."
    )
    return dict(text=text, value=value, candidates=candidates)


def _chain_moment(f: Poly, order: tuple, c: int) -> Fraction:
    """∭ f over 0 ≤ u ≤ v ≤ w ≤ c, where (u, v, w) = order (variable indices), by the closed form for monomials:
    c^(i+j+k+3) / ((i+1)(i+j+2)(i+j+k+3)) with i, j, k the exponents of u, v, w."""
    total = Fraction(0)
    for exps, coef in f.t.items():
        i, j, k = (exps[order[0]], exps[order[1]], exps[order[2]])
        total += coef * Fraction(c ** (i + j + k + 3), (i + 1) * (i + j + 2) * (i + j + k + 3))
    return total


def _chain_value(f: Poly, order: tuple, c: int) -> Fraction:
    u, v, w = order
    g = f.integrate(u, _ZERO, Poly.var(v))
    g = g.integrate(v, _ZERO, Poly.var(w))
    g = g.integrate(w, _ZERO, Poly.const(c))
    return g.t.get((0, 0, 0), Fraction(0))


def _chain_numeric(f: Poly, order: tuple, c: int) -> float:
    """Gauss-Legendre over 0 ≤ u ≤ v ≤ w ≤ c, with w outermost: w in [0, c], v in [0, w], u in [0, v]."""
    u, v, w = order

    # nested Gauss needs bounds in x, y, z order; rebuild with the chain as the integration order w, v, u
    def integrand(wv, vv, uv):
        point = [0.0, 0.0, 0.0]
        point[w], point[v], point[u] = wv, vv, uv
        return float(f.value(tuple(point)))

    # bounds expressed on a relabelled (x, y, z) = (w, v, u) system
    bounds = [(_ZERO, Poly.const(c)), (_ZERO, Poly.var(0)), (_ZERO, Poly.var(1))]
    return gauss_nested(integrand, bounds)


def _hard_chain(rng):
    for _ in attempts(100, "a chain-of-inequalities integral"):
        order = tuple(rng.sample([0, 1, 2], 3))
        c = rng.randint(1, 3)
        f = rng.choice(_HARD_F)
        value = _chain_moment(f, order, c)
        if value.denominator <= 600:
            break
    exact = _chain_value(f, order, c)
    check(exact == value, "chain integral: iterated integration disagrees with the closed form")
    _check_close(value, _chain_numeric(f, order, c), "chain integral")
    reverse = tuple(reversed(order))
    candidates = [
        ("triple-integral-limits-reversed", _chain_moment(f, reverse, c)),
        ("triple-integral-symmetry-assumed", _triple(f, _const_bounds([(0, c)] * 3)) / 6),
        ("triple-integral-bounding-box", _triple(f, _const_bounds([(0, c)] * 3))),
        ("triple-integral-chain-misordered", _chain_moment(f, (order[1], order[0], order[2]), c)),
    ]
    u, v, w = (_NAMES[i] for i in order)
    text = f"Evaluate «∭ {_f_text(f)} dV» over the solid E given by «0 ≤ {u} ≤ {v} ≤ {w} ≤ {c}»."
    return dict(text=text, value=value, candidates=candidates)


def _hard_reorder(rng):
    c = rng.randint(1, 3)
    f = rng.choice([g for g in _HARD_F if g.uses(0) and g.uses(2)])
    # the region is 0 ≤ z ≤ y ≤ x ≤ c; the iterated integral is given in the order dz dy dx
    original = [
        (_ZERO, Poly.const(c)),
        (_ZERO, X),
        (_ZERO, Y),
    ]
    value = _triple(f, original)
    # the new order dx dy dz: x from y to c, y from z to c, z from 0 to c
    g = f.integrate(0, Y, Poly.const(c))
    g = g.integrate(1, Z, Poly.const(c))
    g = g.integrate(2, _ZERO, Poly.const(c))
    rewritten = g.t.get((0, 0, 0), Fraction(0))
    check(rewritten == value, "change of order changed the value")
    _check_close(value, _numeric(f, original), "change-of-order integral")
    # not rewritten: the same limits attached to the new order (x inner 0..y, y inner 0..z, z outer 0..c)
    g = f.integrate(0, _ZERO, Y)
    g = g.integrate(1, _ZERO, Z)
    g = g.integrate(2, _ZERO, Poly.const(c))
    kept = g.t.get((0, 0, 0), Fraction(0))
    g = f.integrate(0, Y, Poly.const(c))
    g = g.integrate(1, _ZERO, Poly.const(c))
    g = g.integrate(2, _ZERO, Poly.const(c))
    y_unbounded = g.t.get((0, 0, 0), Fraction(0))
    cube = _triple(f, _const_bounds([(0, c)] * 3))
    candidates = [
        ("triple-integral-limits-not-rewritten", kept),
        ("triple-integral-limit-not-dependent", y_unbounded),
        ("triple-integral-bounding-box", cube),
        ("triple-integral-symmetry-assumed", cube / 6),
    ]
    text = (
        f"The iterated integral «I» = ∫ from x = 0 to {c} ∫ from y = 0 to x ∫ from z = 0 to y «{_f_text(f)} dz dy dx» "
        f"is to be evaluated with the order of integration changed to «dx dy dz». Find «I»."
    )
    latex = (
        rf"\text{{The iterated integral }}I = \int_{{x=0}}^{{{c}}}\int_{{y=0}}^{{x}}\int_{{z=0}}^{{y}}"
        rf"{tex(_f_text(f))}\,dz\,dy\,dx\text{{ is to be evaluated with the order of integration changed to }}"
        rf"\;dx\,dy\,dz\text{{. Find }}\;I\text{{.}}"
    )
    return dict(text=text, value=value, candidates=candidates, latex=latex)


def _hard(rng):
    pick = rng.random()
    if pick < 0.4:
        return _hard_tetra(rng)
    if pick < 0.7:
        return _hard_chain(rng)
    return _hard_reorder(rng)


def _distractors(value: Fraction, candidates, max_denominator: int = 5000):
    out = []
    for tag, v in candidates:
        v = Fraction(v)
        if v != value and v.denominator <= max_denominator:
            out.append(Distractor(rational_text(v), tag, exact_value(v)))
    return out


def _lone_shape(answer_text: str, distractors) -> bool:
    """Whether the answer is the only integer or only fraction among its options, so it can be picked without working."""
    shape = "fraction" if "/" in answer_text else "integer"
    return all(("fraction" if "/" in d.text else "integer") != shape for d in distractors)


def _bracket_fraction(text: str) -> str:
    """A fraction in brackets so it can be divided: ``(17/5)``; a whole number is left alone."""
    return f"({text})" if "/" in text else text


@generator(
    "triple_integral",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def triple_integral(rng, tier="easy"):
    for _ in attempts(30, "a triple integral with enough distractors"):
        if tier == "easy":
            d = _easy(rng)
        elif tier == "medium":
            d = _medium_top(rng) if rng.random() < 0.55 else _medium_tetra(rng)
        else:
            d = _hard(rng)
        value = d["value"]
        answer_text = rational_text(value)
        distractors = pick_distractors(answer_text, _distractors(value, d["candidates"]), 4)
        if _lone_shape(answer_text, distractors):
            # no wrong option reads like the answer: the reversed-limits slip (the negative) always does
            reversed_limits = [("triple-integral-limits-reversed", -value), *d["candidates"]]
            distractors = pick_distractors(answer_text, _distractors(value, reversed_limits), 4)
        if len(distractors) >= 3:
            break
    else:  # pragma: no cover - attempts() raises first
        raise AssertionError("unreachable")
    if "latex" in d:
        text, latex = d["text"].replace("«", "").replace("»", ""), d["latex"]
        text = text.replace("I = ∫", "I = ∫")
    else:
        text, latex = say_marked(d["text"])
    return dict(
        text=text,
        latex=latex,
        answer=exact_value(value),
        answer_text=answer_text,
        answer_kind="fraction",
        distractors=distractors,
        working=[f"Integrate in turn with respect to z, y and x between the stated limits to reach {answer_text}."],
    )


# ---------------------------------------------------------------------------------------------------------------
# exam chain: a solid with a density


_WORD = {0: "x", 1: "y", 2: "z"}


_EXACT = " Give an exact answer: a fraction in simplest form, or an integer."


def _prompt(text: str) -> dict:
    plain, latex = say_marked(text)
    return dict(prompt=plain, prompt_latex=latex)


def _stem(text: str) -> dict:
    plain, latex = say_marked(text)
    return dict(text=plain, latex=latex)


class _Redraw(Exception):
    """A draw that cannot give each part two distinct wrong answers."""


def _solid(rng):
    """(kind, region text, bounds, extents (max x, y, z), centroid moments by Dirichlet or numerics)."""
    if rng.random() < 0.55:
        a, b, c = rng.randint(1, 4), rng.randint(1, 4), rng.randint(1, 4)
        text = f"the tetrahedron in the first octant bounded by the coordinate planes and the plane «{_plane_text(a, b, c)}»"
        return "tetra", text, _tetra_bounds(a, b, c), (a, b, c), (a, b, c)
    a, b = rng.randint(1, 3), rng.randint(1, 3)
    text = f"the wedge «0 ≤ x ≤ {a}, 0 ≤ y ≤ {b}, 0 ≤ z ≤ x + y»"
    bounds = [(_ZERO, Poly.const(a)), (_ZERO, Poly.const(b)), (_ZERO, X + Y)]
    return "wedge", text, bounds, (a, b, a + b), (a, b, 0)


def _moment(kind, f: Poly, bounds, dims) -> Fraction:
    exact = _triple(f, bounds)
    if kind == "tetra":
        a, b, c = dims
        check(exact == _dirichlet(f, a, b, c), "moment: iterated integration disagrees with the Dirichlet formula")
    else:
        _check_close(exact, _numeric(f, bounds), "wedge moment")
    return exact


@generator("exd_x_uc_triple_integral", level=20, presets={"exam": dict(tier="exam")})
def exd_x_uc_triple_integral(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(30, "an exam triple integral with enough distractors"):
        try:
            return _exam_once(rng)
        except _Redraw:
            continue
    raise AssertionError("unreachable")


def _frac_ft(v: Fraction) -> str:
    return str(v.numerator) if v.denominator == 1 else f"({v.numerator}/{v.denominator})"


def _exam_once(rng):
    kind, solid_text, bounds, extents, dims = _solid(rng)
    k, p = rng.randint(1, 5), rng.randint(1, 3)
    w = rng.randint(0, 2)  # the density varies with this coordinate
    u = rng.randint(0, 2)  # the centre-of-mass coordinate asked for
    rho = k + p * _VARS[w]
    one = Poly.const(1)
    V = _moment(kind, one, bounds, dims)
    Iw = _moment(kind, _VARS[w], bounds, dims)
    M = _moment(kind, rho, bounds, dims)
    check(M == k * V + p * Iw, "mass is not k V + p ∭ w dV")
    Mu = _moment(kind, _VARS[u] * rho, bounds, dims)
    ubar = Mu / M
    avg = M / V
    check(0 < ubar < extents[u], "centre of mass outside the solid's extent")
    check(avg == k + p * Iw / V, "average density")

    others = [i for i in range(3) if i != u]
    Mother = [_moment(kind, _VARS[i] * rho, bounds, dims) for i in others]
    geometric = _moment(kind, _VARS[u], bounds, dims) / V

    if kind == "tetra":
        a_cands = [
            ("triple-integral-bounding-box", Fraction(dims[0] * dims[1] * dims[2])),
            ("triple-integral-pyramid-factor", V * 3),
            ("triple-integral-base-not-halved", V * 2),
        ]
    else:
        a_cands = [
            ("triple-integral-bounding-box", V * 2),
            ("triple-integral-limit-missing-variable", Fraction(dims[0] ** 2 * dims[1], 2)),
            ("triple-integral-height-halved", V / 2),
        ]
    b_cands = [
        ("triple-integral-density-term-dropped", k * V),
        ("triple-integral-density-maximum", (k + p * extents[w]) * V),
        ("triple-integral-density-midpoint", (k + p * Fraction(extents[w], 2)) * V),
    ]
    c_cands = [
        ("triple-integral-moment-not-divided", Mu),
        ("triple-integral-centroid-uniform", geometric),
        ("triple-integral-moment-inverted", M / Mu),
        ("triple-integral-wrong-coordinate", Mother[0] / M),
    ]
    d_cands = [
        ("triple-integral-moment-inverted", V / M),
        ("triple-integral-mass-not-divided", M),
        ("triple-integral-density-midpoint", k + p * Fraction(extents[w], 2)),
        ("triple-integral-density-maximum", Fraction(k + p * extents[w])),
    ]
    parts_dist = []
    for answer, cands in ((V, a_cands), (M, b_cands), (ubar, c_cands), (avg, d_cands)):
        parts_dist.append(pick_distractors(rational_text(answer), _distractors(answer, cands, 400), 4))
    if min(len(x) for x in parts_dist) < 2:
        raise _Redraw

    density_text = f"{k} + {_NAMES[w] if p == 1 else str(p) + _NAMES[w]}"
    prompts = {
        "a": rng.choice(["Find the volume of the solid, in cubic metres.", "Find the volume of the solid, in «m³»."]),
        "b": rng.choice(
            ["Find the mass of the solid, in kilograms.", "Hence find the mass of the solid, in kilograms."]
        ),
        "c": f"Find the «{_WORD[u]}»-coordinate of the centre of mass of the solid, in metres.",
        "d": rng.choice(
            [
                "Find the average density of the solid, in kilograms per cubic metre.",
                "Use your answers to find the average density of the solid, in «kg/m³».",
            ]
        ),
    }
    ft_b = f"{k}*part_a + {p}*{_frac_ft(Iw)}"
    ft_c = f"{_frac_ft(Mu)}/part_b"
    ft_d = "part_b/part_a"
    return dict(
        **_stem(
            f"A solid occupies {solid_text}. Lengths are in metres. "
            f"At the point «(x, y, z)» the density of the solid is «ρ = {density_text}» kilograms per cubic metre."
        ),
        parts=[
            dict(
                **_prompt(prompts["a"] + _EXACT),
                answer_kind="fraction",
                answer=exact_value(V),
                answer_text=rational_text(V),
                marks=2,
                sensible_range=(0, None),
                working=[f"The volume is ∭ dV over the solid, which is {rational_text(V)}."],
                distractors=parts_dist[0],
            ),
            dict(
                **_prompt(prompts["b"] + _EXACT),
                answer_kind="fraction",
                answer=exact_value(M),
                answer_text=rational_text(M),
                marks=3,
                follow_through=ft_b,
                sensible_range=(0, None),
                working=[
                    f"The mass is ∭ ρ dV = {join_terms([(k, 'V'), (p, f'∭ {_WORD[w]} dV')])} = {rational_text(M)}."
                ],
                distractors=parts_dist[1],
            ),
            dict(
                **_prompt(prompts["c"] + _EXACT),
                answer_kind="fraction",
                answer=exact_value(ubar),
                answer_text=rational_text(ubar),
                marks=3,
                follow_through=ft_c,
                sensible_range=(0, extents[u]),
                working=[
                    f"{_WORD[u]}-bar = (∭ {_WORD[u]}ρ dV)/mass = {_bracket_fraction(rational_text(Mu))} ÷ {_bracket_fraction(rational_text(M))}."
                ],
                distractors=parts_dist[2],
            ),
            dict(
                **_prompt(prompts["d"] + _EXACT),
                answer_kind="fraction",
                answer=exact_value(avg),
                answer_text=rational_text(avg),
                marks=1,
                follow_through=ft_d,
                sensible_range=(0, None),
                working=[f"Average density = mass/volume = {rational_text(avg)}."],
                distractors=parts_dist[3],
            ),
        ],
    )
