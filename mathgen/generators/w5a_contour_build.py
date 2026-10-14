"""Generates `w5a_contour` and `exd_x_w5a_contour` questions: contour integrals by the residue theorem.

One second-year skill, ``uc2-contour-integrals-residue-theorem`` (domain ``complex-analysis``). The contour is a circle
of stated centre and radius, taken anticlockwise (or clockwise, where stated), and the answer is an exact expression in
``i`` and ``pi``. Every question is built from chosen poles first: the radius is then chosen so that the circle
separates the poles into the enclosed and the unenclosed, with a margin of at least 0.4 between the radius and every
pole's distance from the centre, so which poles are inside is never a close call.

Tiers:

- ``easy``: two simple poles, a circle that encloses exactly one of them. Which one is not stated.
- ``medium``: three poles, one of them double, and a circle that encloses two of them (the double pole among them, or
  not) with its centre away from the origin.
- ``hard``: integrands where a shortcut or a trap decides the answer: ``e^(kz)`` over a polynomial, a circle that
  encloses every pole of ``N(z)/D(z)`` with the numerator one degree short of the denominator (the answer is
  ``2*pi*i`` times a leading coefficient, taken clockwise), and ``conj(z - c)/(z - p)``, which is not analytic but
  equals ``R^2/((z - c)(z - p))`` on the circle.
- ``exam`` (topic ``exd_x_w5a_contour``): a four-part chain on three real simple poles and a circle centred off the
  real axis: how many poles it encloses, the residue at each of two of them, then the integral, which follows through
  from the residues.

The second route, in `make()`, is the trapezoid rule on the contour itself: the mean of f(z)(z - c) over equally
spaced points of the circle is (1/2*pi*i) times the integral, and it converges geometrically with the margin above.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction as Fr

import sympy

from ..core import attempts, check, generator
from ._w5a_support import (
    C,
    Link,
    Rat,
    X,
    atext,
    circle_mean,
    cx_x,
    exam_dict,
    fac_x,
    frac_x,
    msay,
    numeric_close,
    poly_x,
    prod_x,
    qsym,
    rat_x,
    sdiv,
    single,
    zminus_x,
)

TOPIC = "w5a_contour"
EXAM_TOPIC = "exd_x_w5a_contour"

_N = 400  # points of the trapezoid rule
_MARGIN = 0.4


def _oint(f: X, circle: X, clockwise: bool = False) -> X:
    return X(f"∮_({circle.p}) {f.p} dz", rf"\oint_{{{circle.t}}} {f.t}\,\mathrm{{d}}z")


def _circle(c: C, r: Fr) -> X:
    inner = zminus_x(c)
    rr = f"{r.numerator}" if r.denominator == 1 else f"{r.numerator}/{r.denominator}"
    rt = rf"{r.numerator}" if r.denominator == 1 else rf"\frac{{{r.numerator}}}{{{r.denominator}}}"
    return X(f"|{inner.p}| = {rr}", f"|{inner.t}| = {rt}")


def _fline(body: X) -> X:
    return X("f(z) = " + body.p, "f(z) = " + body.t)


def _dist2(c: C, a: C) -> Fr:
    return (a - c).abs2()


_RADII = [Fr(k, 2) for k in range(2, 13)]


def _radius_for(c: C, poles: list, inside: set) -> Fr | None:
    """A half-integer radius whose circle about c encloses exactly the poles with those indices, with margin."""
    ds = [math.sqrt(_dist2(c, a)) for a in poles]
    ok = [
        r
        for r in _RADII
        if all((d < float(r) - _MARGIN) if i in inside else (d > float(r) + _MARGIN) for i, d in enumerate(ds))
    ]
    return ok[0] if len(ok) == 1 else (ok[(len(ok) - 1) // 2] if ok else None)


_CENTRES = [C(0), C(0), C(1), C(-1), C(0, 1), C(0, -1), C(1, 1), C(-1, 1), C(2), C(-2), C(1, -1), C(0, 2)]


def _find_circle(rng, poles: list, want: int, must_inside: set | None = None):
    """(centre, radius, inside indices) for a circle that encloses exactly `want` of the poles, or None."""
    centres = list(dict.fromkeys(_CENTRES))
    rng.shuffle(centres)
    for c in centres:
        order = sorted(range(len(poles)), key=lambda i: math.sqrt(_dist2(c, poles[i])))
        inside = set(order[:want])
        if must_inside is not None and inside != must_inside:
            continue
        r = _radius_for(c, poles, inside)
        if r is not None:
            return c, r, inside
    return None


def _contour_mean(fn, c: C, r: Fr) -> complex:
    return circle_mean(fn, complex(c), float(r), _N)


def _val(sum_res) -> sympy.Basic:
    return sympy.expand(2 * sympy.pi * sympy.I * sum_res)


def _wrongs_sum(res_in, res_out, res_all, n_in) -> list:
    """Standard wrong integrals from the residues inside, outside and in total (all SymPy values)."""
    two_pi_i = 2 * sympy.pi * sympy.I
    out = []
    s_in = sum(res_in) if res_in else sympy.Integer(0)
    out.append((sympy.expand(2 * sympy.pi * s_in), "forgets-i-in-residue-theorem"))
    out.append((sympy.expand(two_pi_i * sum(res_all)), "included-pole-outside-contour"))
    if res_out:
        out.append((sympy.expand(two_pi_i * sum(res_out)), "pole-outside-the-contour"))
    if n_in >= 2:
        out.append((sympy.expand(two_pi_i * res_in[0]), "ignores-pole-enclosed-by-larger-contour"))
        out.append((sympy.expand(two_pi_i * (res_in[0] - sum(res_in[1:]))), "residues-subtracted"))
    out.append((sympy.expand(-two_pi_i * s_in), "cx2-orientation-ignored"))
    out.append((sympy.expand(s_in), "cx2-forgot-2-pi-i"))
    return out


def _phrase(rng, fx: X, circ: X, clockwise: bool = False):
    way = "clockwise" if clockwise else "anticlockwise"
    shapes = [
        lambda: msay(
            "Let ",
            fx,
            " and let C be the circle ",
            circ,
            ", traversed once ",
            way,
            ". Evaluate the integral of f(z) round C.",
        ),
        lambda: msay(
            "Evaluate ",
            X("∮_C f(z) dz", r"\oint_{C} f(z)\,\mathrm{d}z"),
            ", where ",
            fx,
            " and C is the circle ",
            circ,
            " taken once ",
            way,
            ".",
        ),
        lambda: msay(
            "The contour C is the circle ",
            circ,
            ", traversed ",
            way,
            ". Use the residue theorem to find the exact value of ",
            X("∮_C f(z) dz", r"\oint_{C} f(z)\,\mathrm{d}z"),
            ", where ",
            fx,
            ".",
        ),
    ]
    return shapes[rng.randrange(3)]()


# -- easy ------------------------------------------------------------------------------------------------------------


def _easy(rng):
    for _ in attempts(800, "an easy contour integral"):
        a, b = rng.sample(_GAUSS_POLES, 2)
        which = rng.randrange(2)
        found = _find_circle(rng, [a, b], 1, {which})
        if found is None:
            continue
        c, r, inside = found
        num = [C(rng.randint(-4, 4)), C(rng.choice([0, 1, 2, -1, -2]))]
        if num[1].is_zero() and num[0].is_zero():
            continue
        rat = Rat(num, [(a, 1), (b, 1)])
        r_a, r_b = rat.residue(0), rat.residue(1)
        if r_a.is_zero() or r_b.is_zero():
            continue
        break
    res = [r_a, r_b]
    sym_in = [res[which].sym()]
    sym_out = [res[1 - which].sym()]
    value = _val(sym_in[0])
    fx = _fline(rat_x(rat))
    circ = _circle(c, r)
    check(c is not None, "centre")
    numeric_close(_contour_mean(rat.at, c, r) * 2j * math.pi, complex(sympy.N(value, 15)), "contour integral", 1e-6)
    wrongs = _wrongs_sum(sym_in, sym_out, [r_a.sym(), r_b.sym()], 1)
    working = [
        f"The poles are z = {cx_x(a).p} and z = {cx_x(b).p}. Their distances from the centre {cx_x(c).p} are "
        f"{_dist_text(c, a)} and {_dist_text(c, b)}, so only z = {cx_x([a, b][which]).p} is inside the circle of radius {atext(qsym(r))}.",
        f"The residue there is {atext(res[which].sym())}, so the integral is 2πi × ({atext(res[which].sym())}) = {atext(value)}.",
    ]
    return single(_phrase(rng, fx, circ), value, working, wrongs)


_GAUSS_POLES = [C(a, b) for a in (-3, -2, -1, 0, 1, 2, 3) for b in (-2, -1, 0, 1, 2)]


def _dist_text(c: C, a: C) -> str:
    d2 = _dist2(c, a)
    root = math.isqrt(d2.numerator) if d2.denominator == 1 else None
    if root is not None and root * root == d2:
        return str(root)
    return f"√{d2.numerator}"


# -- medium ----------------------------------------------------------------------------------------------------------


def _medium(rng):
    for _ in attempts(1500, "a medium contour integral"):
        a, b = rng.sample(_GAUSS_POLES, 2)
        c3 = rng.choice([p for p in _GAUSS_POLES if p not in (a, b)])
        poles = [(a, 2), (b, 1), (c3, 1)]
        want_set = rng.choice([{0, 1}, {0, 2}, {1, 2}, {0}])
        found = _find_circle(rng, [p for p, _ in poles], len(want_set), want_set)
        if found is None:
            continue
        c, r, inside = found
        num = [C(rng.randint(-4, 4)), C(rng.randint(-3, 3)), C(rng.choice([0, 0, 1, -1]))]
        if num[1].is_zero() and num[0].is_zero():
            continue
        rat = Rat(num, poles)
        res = [rat.residue(i) for i in range(3)]
        if any(r_.is_zero() for r_ in res) or not all(
            r_.re.denominator <= 60 and r_.im.denominator <= 60 for r_ in res
        ):
            continue
        break
    inside_i = sorted(inside)
    outside_i = [i for i in range(3) if i not in inside]
    sym_res = [x.sym() for x in res]
    value = _val(sum(sym_res[i] for i in inside_i))
    numeric_close(_contour_mean(rat.at, c, r) * 2j * math.pi, complex(sympy.N(value, 15)), "contour integral", 1e-6)
    fx = _fline(rat_x(rat))
    circ = _circle(c, r)
    wrongs = _wrongs_sum([sym_res[i] for i in inside_i], [sym_res[i] for i in outside_i], sym_res, len(inside_i))
    # a double pole treated as simple
    simple = sympy.expand(2 * sympy.pi * sympy.I * sum(_simple_res(rat, i, sym_res) for i in inside_i))
    wrongs.insert(0, (simple, "cx2-residue-simple-formula"))
    names = ", ".join(cx_x(poles[i][0]).p for i in inside_i)
    working = [
        f"The poles are {cx_x(a).p} (order 2), {cx_x(b).p} and {cx_x(c3).p}. The circle has centre {cx_x(c).p} and radius {atext(qsym(r))}, and it encloses z = {names}.",
        "The residues at the enclosed poles are " + ", ".join(atext(sym_res[i]) for i in inside_i) + ".",
        f"The integral is 2πi times their sum: {atext(value)}.",
    ]
    return single(_phrase(rng, fx, circ), value, working, wrongs)


def _simple_res(rat: Rat, i: int, sym_res):
    """The 'simple-pole shortcut' value at pole i: the rest of f evaluated there (for a double pole, wrong)."""
    a, m = rat.poles[i]
    if m == 1:
        return sym_res[i]
    ser = rat.laurent(i, 1)
    return ser[-m].sym()


# -- hard ------------------------------------------------------------------------------------------------------------


def _hard_exp(rng):
    for _ in attempts(1500, "an exponential contour integral"):
        a, b = rng.sample([-2, -1, 0, 1, 2], 2)
        m = rng.choice([1, 2])
        k = rng.choice([1, 2, 3, -1, -2])
        poles = [(C(a), m), (C(b), 1)]
        want_set = rng.choice([{0}, {1}, {0, 1}])
        found = _find_circle(rng, [p for p, _ in poles], len(want_set), want_set)
        if found is None:
            continue
        c, r, inside = found
        break
    # residue = e^(k p) times a rational coefficient
    coeffs = []
    for j, (p, mm) in enumerate(poles):
        gden = [C(1)]
        for i, (q, nn) in enumerate(poles):
            if i != j:
                from ._w5a_support import pmul, ppow

                gden = pmul(gden, ppow([p - q, C(1)], nn))
        expo = [C(Fr(k**t, math.factorial(t))) for t in range(mm)]
        inv = sdiv([C(1)], gden, mm)
        total = C(0)
        for t in range(mm):
            total = total + expo[t] * inv[mm - 1 - t]
        coeffs.append(total)
    res_sym = [coeffs[j].sym() * sympy.exp(k * (a if j == 0 else b)) for j in range(2)]
    inside_i = sorted(inside)
    value = sympy.expand(2 * sympy.pi * sympy.I * sum(res_sym[i] for i in inside_i))
    outside_i = [i for i in range(2) if i not in inside]

    def fn(z):
        import cmath

        return cmath.exp(k * z) / ((z - a) ** m * (z - b))

    numeric_close(_contour_mean(fn, c, r) * 2j * math.pi, complex(sympy.N(value, 15)), "contour integral", 1e-6)
    kz = "z" if k == 1 else "-z" if k == -1 else f"{k}z"
    den = prod_x([fac_x(C(a), m), fac_x(C(b))])
    quo = frac_x(X(f"e^({kz})", rf"e^{{{kz}}}"), den)
    fx = X("f(z) = " + quo.p, "f(z) = " + quo.t)
    circ = _circle(c, r)
    wrongs = _wrongs_sum([res_sym[i] for i in inside_i], [res_sym[i] for i in outside_i], res_sym, len(inside_i))
    wrongs.insert(
        0,
        (
            sympy.expand(2 * sympy.pi * sympy.I * sum(coeffs[i].sym() for i in inside_i)),
            "cx2-residue-exponential-factor-dropped",
        ),
    )
    names = ", ".join(str([a, b][i]) for i in inside_i)
    working = [
        f"The only singularities are the poles z = {a} (order {m}) and z = {b}. The circle encloses z = {names}.",
        "Each residue contains the factor e^(k p) at its pole p; the order-2 pole also needs the derivative of e^(kz).",
        f"The integral is 2πi times the sum of the enclosed residues: {atext(value)}.",
    ]
    return single(_phrase(rng, fx, circ), value, working, wrongs)


def _hard_big_circle(rng):
    """Numerator one degree short of the denominator, a circle enclosing every pole, possibly clockwise."""
    for _ in attempts(1500, "a big-circle integral"):
        poles = rng.sample(_GAUSS_POLES, 3)
        lead = rng.choice([1, 2, 3, -1, -2])
        num = [C(rng.randint(-4, 4)), C(rng.randint(-3, 3)), C(lead)]
        rat = Rat(num, [(p, 1) for p in poles])
        res = [rat.residue(i) for i in range(3)]
        if any(r.is_zero() for r in res):
            continue
        found = _find_circle(rng, poles, 3, {0, 1, 2})
        if found is None:
            continue
        c, r, inside = found
        break
    clockwise = rng.random() < 0.5
    total = sum((x for x in res), C(0))
    check(total == C(lead), "the residues of N/D sum to the leading coefficient")
    value = sympy.expand((-1 if clockwise else 1) * 2 * sympy.pi * sympy.I * lead)
    s = -1 if clockwise else 1
    sym_res = [x.sym() for x in res]
    fx = _fline(rat_x(rat))
    circ = _circle(c, r)
    ccw = sympy.expand(2 * sympy.pi * sympy.I * lead)
    wrongs = [
        (-value, "cx2-orientation-ignored"),
        (sympy.expand(2 * sympy.pi * lead), "forgets-i-in-residue-theorem"),
        (sympy.expand(2 * sympy.pi * sympy.I * s * sym_res[0]), "ignores-pole-enclosed-by-larger-contour"),
        (sympy.Integer(0), "cx2-sum-of-residues-assumed-zero"),
        (sympy.expand(s * 2 * sympy.pi * sympy.I * qsym(num[1].re)), "cx2-wrong-coefficient-for-infinity"),
        (ccw if clockwise else sympy.expand(-ccw), "cx2-orientation-ignored"),
    ]
    numeric_close(_contour_mean(rat.at, c, r) * 2j * math.pi * s, complex(sympy.N(value, 15)), "contour integral", 1e-6)
    working = [
        "All three poles lie inside the circle, so the integral is 2πi times the sum of all three residues.",
        "For large |z|, f(z) behaves like (leading coefficient of N)/z, so the residues sum to that leading coefficient "
        f"{lead}, without finding any of them.",
        f"Taken {'clockwise' if clockwise else 'anticlockwise'}, the integral is {atext(value)}.",
    ]
    return single(_phrase(rng, fx, circ, clockwise), value, working, wrongs)


def _hard_conjugate(rng):
    """Integral of conj(z - c)/(z - p) round |z - c| = R: on the circle conj(z - c) = R^2/(z - c)."""
    for _ in attempts(1000, "a conjugate integral"):
        c = rng.choice(_CENTRES)
        p = rng.choice(_GAUSS_POLES)
        if p == c:
            continue
        inside_p = rng.random() < 0.3
        r = _radius_for(c, [p], {0} if inside_p else set())
        if r is None:
            continue
        break
    r2 = r * r
    value = sympy.Integer(0) if inside_p else sympy.expand(2 * sympy.pi * sympy.I * qsym(r2) / (c - p).sym())
    zc = zminus_x(c)
    cbar = c.conj()
    top = X(f"conj({zc.p})", rf"\overline{{{zc.t}}}")
    # the conjugate of (z - c) is written as the conjugate of the whole bracket
    fx = X(f"f(z) = {top.p}/({zminus_x(p).p})", rf"f(z) = \frac{{{top.t}}}{{{zminus_x(p).t}}}")
    circ = _circle(c, r)

    def fn(z):
        return (z - complex(c)).conjugate() / (z - complex(p))

    numeric_close(_contour_mean(fn, c, r) * 2j * math.pi, complex(sympy.N(value, 15)), "contour integral", 1e-6)
    del cbar
    wrongs = [
        (
            sympy.expand(2 * sympy.pi * sympy.I * qsym(r2) / (c - p).sym()) if inside_p else sympy.Integer(0),
            "cx2-conjugate-treated-as-analytic",
        ),
        (sympy.expand(-2 * sympy.pi * sympy.I * qsym(r2) / (c - p).sym()), "calc-sign-slip"),
        (sympy.expand(2 * sympy.pi * sympy.I * qsym(r) / (c - p).sym()), "cx2-radius-not-squared"),
        (sympy.expand(2 * sympy.pi * sympy.I * qsym(r2) / (p - c).sym()), "calc-sign-slip"),
        (sympy.expand(2 * sympy.pi * sympy.I * qsym(r2)), "cx2-conjugate-treated-as-analytic"),
        (sympy.expand(2 * sympy.pi * sympy.I * qsym(r2) * 2 / (c - p).sym()), "cx2-conjugate-residues-doubled"),
    ]
    working = [
        f"On the circle, conj(z - c) = r²/(z - c) with r = {atext(qsym(r))}, so the integrand equals r²/((z - c)(z - p)).",
        f"The pole z = {cx_x(c).p} is always inside; z = {cx_x(p).p} is {'inside' if inside_p else 'outside'}.",
        "The residues at c and p are r²/(c - p) and r²/(p - c)"
        + (", which cancel." if inside_p else f"; only the first counts, giving {atext(value)}."),
    ]
    return single(
        msay("Let ", fx, ". Evaluate the integral of f(z) round the circle ", circ, " taken once anticlockwise."),
        value,
        working,
        wrongs,
    )


_TIERS = {"easy": [_easy], "medium": [_medium], "hard": [_hard_exp, _hard_big_circle, _hard_conjugate]}


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5a_contour(rng: random.Random, tier: str = "easy") -> dict:
    """The integral of a function round a circle, by residues: an exact expression in i and pi."""
    check(tier in _TIERS, f"unknown tier {tier!r}")
    return rng.choice(_TIERS[tier])(rng)


# -- exam ------------------------------------------------------------------------------------------------------------


@generator(EXAM_TOPIC, level=19, presets={"exam": dict(tier="exam")})
def exd_x_w5a_contour(rng: random.Random, tier: str = "exam") -> dict:
    """Three real simple poles and a circle off the real axis: how many poles inside, two residues, the integral."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(2000, "a contour chain"):
        poles_i = rng.sample([-4, -3, -2, -1, 0, 1, 2, 3, 4], 3)
        poles = [C(p) for p in poles_i]
        num = [C(rng.randint(-4, 4)), C(rng.randint(-3, 3)), C(rng.choice([1, 2, -1, 3]))]
        rat = Rat(num, [(p, 1) for p in poles])
        res = [rat.residue(i) for i in range(3)]
        if any(x.is_zero() for x in res) or not all(x.re.denominator <= 60 for x in res):
            continue
        found = _find_circle(rng, poles, 2)
        if found is None:
            continue
        c, r, inside = found
        if c.is_real():
            continue
        break
    inside_i = sorted(inside)
    out_i = [i for i in range(3) if i not in inside]
    p1, p2 = inside_i
    r1, r2 = res[p1].re, res[p2].re
    rout = res[out_i[0]].re
    value = sympy.expand(2 * sympy.pi * sympy.I * qsym(r1 + r2))
    numeric_close(_contour_mean(rat.at, c, r) * 2j * math.pi, complex(sympy.N(value, 15)), "contour integral", 1e-6)
    fx = _fline(rat_x(rat))
    circ = _circle(c, r)
    stem = msay("Let ", fx, " and let C be the circle ", circ, ", traversed once anticlockwise.")
    dists = [math.sqrt(_dist2(c, p)) for p in poles]
    del dists
    pole_list = ", ".join(str(poles_i[i]) for i in range(3))
    wa = Link(
        msay("How many of the poles of f lie inside C?"),
        2,
        "number",
        2,
        [
            f"The poles are {pole_list}. Their distances from the centre {cx_x(c).p} are "
            + ", ".join(_dist_text(c, p) for p in poles)
            + f", and the radius is {atext(qsym(r))}."
        ],
        [(3, "included-pole-outside-contour"), (1, "cx2-pole-count-slip"), (0, "cx2-pole-count-slip")],
        sensible=(0, 4),
    )
    wb = Link(
        msay("Find the residue of f at ", X(f"z = {poles_i[p1]}", f"z = {poles_i[p1]}"), "."),
        qsym(r1),
        "fraction",
        2,
        [f"Res = N({poles_i[p1]}) over the other two factors at {poles_i[p1]} = {atext(qsym(r1))}."],
        [
            (-qsym(r1), "residue-sign-error"),
            (qsym(r2), "residue-at-wrong-pole"),
            (qsym(rout), "residue-at-wrong-pole"),
            (qsym(sum((num[k] * poles[p1] ** k for k in range(3)), C(0)).re), "cx2-residue-numerator-only"),
        ],
    )
    wc = Link(
        msay("Find the residue of f at ", X(f"z = {poles_i[p2]}", f"z = {poles_i[p2]}"), "."),
        qsym(r2),
        "fraction",
        2,
        [f"Res = N({poles_i[p2]}) over the other two factors at {poles_i[p2]} = {atext(qsym(r2))}."],
        [
            (-qsym(r2), "residue-sign-error"),
            (qsym(r1), "residue-at-wrong-pole"),
            (qsym(rout), "residue-at-wrong-pole"),
            (qsym(sum((num[k] * poles[p2] ** k for k in range(3)), C(0)).re), "cx2-residue-numerator-only"),
        ],
    )
    wd = Link(
        msay("Hence find the exact value of the integral of f(z) round C."),
        value,
        "expression",
        3,
        [
            f"The enclosed poles are {poles_i[p1]} and {poles_i[p2]}, so the integral is 2πi({atext(qsym(r1))} + ({atext(qsym(r2))})) = {atext(value)}."
        ],
        [
            (sympy.expand(2 * sympy.pi * sympy.I * qsym(r1 + r2 + rout)), "included-pole-outside-contour"),
            (sympy.expand(2 * sympy.pi * qsym(r1 + r2)), "forgets-i-in-residue-theorem"),
            (sympy.expand(2 * sympy.pi * sympy.I * qsym(r1)), "ignores-pole-enclosed-by-larger-contour"),
            (sympy.expand(2 * sympy.pi * sympy.I * qsym(r1 - r2)), "residues-subtracted"),
        ],
        ft="2*pi*I*(part_b + part_c)",
    )
    return exam_dict(stem, [wa, wb, wc, wd])
