"""Lagrange interpolation: `r4b_lagrange_interpolation` (easy, medium, hard) and the exam chain
`exd_x_r4b_lagrange_interpolation`.

``uc2-lagrange-interpolating-polynomial`` (``numerical-and-computational-mathematics``):

* easy: three data points taken from a quadratic with whole-number coefficients, at unequal whole-number nodes; the
  answer is the interpolating polynomial in expanded form (an expression).
* medium: four data points from a cubic with whole-number coefficients; the answer is the interpolating polynomial in
  expanded form.
* hard: the data are values of ``f(x) = 1/x`` at four unequally spaced nodes, so every ``y`` is a fraction and the
  interpolating cubic has fractional coefficients; the answer is that polynomial in expanded form.
* exam: ``f(x) = 1/x`` at three nodes in four parts: the middle basis polynomial at a stated point (fraction), the
  interpolating polynomial (expression), its value at the stated point (follow-through, fraction) and the interpolation
  error ``f(t) − p(t)`` (follow-through, fraction).

Each polynomial is built from the Lagrange basis in exact `Fraction` arithmetic and checked against SymPy's
``interpolate`` and by evaluating at every node.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..answers import Distractor
from ..core import attempts, check, generator
from ._r4b_kit import (
    X,
    Redraw,
    as_single,
    fraction_distractors,
    fraction_part,
    fr,
    poly_differs,
    poly_eval,
    poly_machine,
    poly_prompt,
    poly_sympy,
    poly_text,
    sub,
    with_redraw,
)
from ._ucv1_texsay import say_marked

M_NO_DENOM = "num2-lagrange-basis-denominator-dropped"
M_MISMATCH = "num2-lagrange-y-values-paired-with-wrong-bases"
M_NODE_SIGN = "num2-lagrange-node-sign-error"
M_FACTOR = "num2-lagrange-basis-factor-dropped"
M_NODE_SIGN_DENOM = "num2-lagrange-denominator-sign-error"
M_LINEAR = "num2-lagrange-fewer-nodes-used"
M_NEAREST = "num2-lagrange-nearest-node-value"
M_NEG = "num2-lagrange-error-sign"
M_F_FOR_P = "num2-lagrange-function-value-for-polynomial"
M_BASIS_OTHER = "num2-lagrange-wrong-basis-polynomial"
M_BASIS_NUM = "num2-lagrange-basis-numerator-only"
M_BASIS_NODE = "num2-lagrange-basis-at-its-own-node"


def pmul(a, b):
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, u in enumerate(a):
        for j, v in enumerate(b):
            out[i + j] += fr(u) * fr(v)
    return out


def padd(a, b):
    n = max(len(a), len(b))
    return [fr(a[i] if i < len(a) else 0) + fr(b[i] if i < len(b) else 0) for i in range(n)]


def pscale(a, k):
    return [fr(c) * fr(k) for c in a]


def trim(a):
    a = list(a)
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a


def numerator(xs, i, signs=1):
    out = [Fraction(1)]
    for j, xj in enumerate(xs):
        if j != i:
            out = pmul(out, [-signs * fr(xj), Fraction(1)])
    return out


def denominator(xs, i, sign=1):
    out = Fraction(1)
    for j, xj in enumerate(xs):
        if j != i:
            out *= sign * (fr(xs[i]) - fr(xj))
    return out


def basis(xs, i):
    return pscale(numerator(xs, i), 1 / denominator(xs, i))


def lagrange(xs, ys):
    out = [Fraction(0)]
    for i in range(len(xs)):
        out = padd(out, pscale(basis(xs, i), ys[i]))
    return trim(out)


def check_polynomial(xs, ys, coeffs):
    """Against SymPy's interpolation, and at every node."""
    ref = sympy.interpolate(
        list(
            zip(
                [sympy.Rational(x.numerator, x.denominator) for x in map(fr, xs)],
                [sympy.Rational(y.numerator, y.denominator) for y in map(fr, ys)],
            )
        ),
        X,
    )
    check(sympy.expand(ref - poly_sympy(coeffs)) == 0, "SymPy's interpolating polynomial disagrees")
    for x, y in zip(xs, ys):
        check(poly_eval(coeffs, x) == fr(y), "the polynomial misses a data point")


def _slips(xs, ys):
    """Wrong interpolating polynomials as ``(misconception, coefficients)``."""
    n = len(xs)
    out = []
    no_denom = [Fraction(0)]
    for i in range(n):
        no_denom = padd(no_denom, pscale(numerator(xs, i), ys[i]))
    out.append((M_NO_DENOM, trim(no_denom)))
    rev = [Fraction(0)]
    for i in range(n):
        rev = padd(rev, pscale(basis(xs, i), ys[n - 1 - i]))
    out.append((M_MISMATCH, trim(rev)))
    ns = [Fraction(0)]
    for i in range(n):
        ns = padd(ns, pscale(pscale(numerator(xs, i, signs=-1), 1), Fraction(ys[i]) / denominator(xs, i)))
    out.append((M_NODE_SIGN, trim(ns)))
    sd = [Fraction(0)]
    for i in range(n):
        sd = padd(sd, pscale(numerator(xs, i), Fraction(ys[i]) / denominator(xs, i, sign=-1)))
    out.append((M_NODE_SIGN_DENOM, trim(sd)))
    fd = [Fraction(0)]
    for i in range(n):
        j = (i + 1) % n
        num = [-fr(xs[j]), Fraction(1)]
        fd = padd(fd, pscale(num, Fraction(ys[i]) / (fr(xs[i]) - fr(xs[j]))))
    out.append((M_FACTOR, trim(fd)))
    return out


def poly_distractors(answer, cands, n=4):
    seen = {poly_text(answer)}
    out = []
    for tag, c in cands:
        text = poly_text(c)
        if text in seen or not poly_differs(answer, c):
            continue
        seen.add(text)
        out.append(Distractor(text, tag, poly_machine(c)))
        if len(out) == n:
            break
    return out


def _data_from_poly(rng, degree, lo, hi, xrange, ycap):
    for _ in attempts(2000, "a polynomial with moderate data"):
        coeffs = [Fraction(rng.randint(-4, 4)) for _ in range(degree)] + [Fraction(rng.choice([-2, -1, 1, 2]))]
        if coeffs[0] == 0 and degree > 1 and coeffs[1] == 0:
            continue
        xs = sorted(rng.sample(range(xrange[0], xrange[1] + 1), degree + 1))
        ys = [poly_eval(coeffs, x) for x in xs]
        if max(abs(y) for y in ys) > ycap or len(set(ys)) < len(ys) - 0:
            continue
        # unequal spacing so the nodes do not hand over a pattern
        gaps = {xs[i + 1] - xs[i] for i in range(len(xs) - 1)}
        if len(gaps) < 2:
            continue
        return coeffs, xs, [int(y) for y in ys]
    raise AssertionError("unreachable")


def _points(xs, ys):
    return ", ".join(f"({x}, {y})" for x, y in zip(xs, ys))


@generator(
    "r4b_lagrange_interpolation",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4b_lagrange_interpolation(rng, tier="easy"):
    return with_redraw(lambda: _single(rng, tier), "a Lagrange question with enough distractors")


def _single(rng, tier):
    if tier == "easy":
        coeffs, xs, ys = _data_from_poly(rng, 2, 0, 0, (-3, 5), 25)
        p = lagrange(xs, ys)
        check(trim(p) == trim(coeffs), "interpolation did not recover the quadratic")
        check_polynomial(xs, ys, p)
        pts = _points(xs, ys)
        text = say_marked(
            rng.choice(
                [
                    f"A quadratic polynomial «p» passes through the points «{pts}». Find «p(x)» in expanded form.",
                    f"The graph of the quadratic «p» goes through the points «{pts}». Write «p(x)» in expanded form.",
                    f"Find the quadratic polynomial «p(x)», in expanded form, whose graph passes through the points «{pts}».",
                ]
            )
        )
        dist = poly_distractors(p, _slips(xs, ys))
        if len(dist) < 3:
            raise Redraw
        return _poly_question(
            text,
            p,
            dist,
            [
                "Build one basis polynomial per node: each is 1 at its own node and 0 at the others.",
                f"p(x) = {poly_text(p)}.",
            ],
        )
    if tier == "medium":
        coeffs, xs, ys = _data_from_poly(rng, 3, 0, 0, (-3, 5), 60)
        p = lagrange(xs, ys)
        check(trim(p) == trim(coeffs), "interpolation did not recover the cubic")
        check_polynomial(xs, ys, p)
        pts = _points(xs, ys)
        text = say_marked(
            rng.choice(
                [
                    f"A cubic polynomial «p» passes through the points «{pts}». Find «p(x)» in expanded form.",
                    f"The graph of the cubic «p» goes through the points «{pts}». Write «p(x)» in expanded form.",
                    f"Find the cubic polynomial «p(x)», in expanded form, whose graph passes through the points «{pts}».",
                ]
            )
        )
        dist = poly_distractors(p, _slips(xs, ys))
        if len(dist) < 3:
            raise Redraw
        return _poly_question(
            text,
            p,
            dist,
            [
                "Each of the four basis polynomials is a product of three linear factors divided by its value at its own node.",
                f"p(x) = {poly_text(p)}.",
            ],
        )
    # hard: f(x) = 1/x at four unequally spaced nodes
    for _ in attempts(500, "four nodes for 1/x"):
        xs = sorted(rng.sample(range(1, 10), 4))
        if len({xs[i + 1] - xs[i] for i in range(3)}) >= 2:
            break
    ys = [Fraction(1, x) for x in xs]
    p = lagrange(xs, ys)
    check_polynomial(xs, ys, p)
    prod = [Fraction(1)]
    for xv in xs:
        prod = pmul(prod, [-fr(xv), Fraction(1)])
    resid = padd(pmul(p, [Fraction(0), Fraction(1)]), [-1])
    ratio = resid[-1] / prod[-1]
    check(trim(resid) == trim(pscale(prod, ratio)), "x p(x) - 1 is not a multiple of the node polynomial")
    nodes = f"{xs[0]}, {xs[1]}, {xs[2]}, {xs[3]}"
    text = say_marked(
        rng.choice(
            [
                f"The function «f(x) = 1/x» is interpolated by the cubic polynomial «p» through the nodes "
                f"«x = {nodes}», so that «p» agrees with «f» at each node. Find «p(x)» in expanded form.",
                f"Let «p» be the cubic polynomial that agrees with «f(x) = 1/x» at the nodes «x = {nodes}». "
                "Find «p(x)» in expanded form.",
                f"The cubic «p» interpolates «f(x) = 1/x» at the four nodes «x = {nodes}». Write «p(x)» in expanded form.",
            ]
        )
    )
    dist = poly_distractors(p, _slips(xs, ys))
    if len(dist) < 3:
        raise Redraw
    return _poly_question(
        text,
        p,
        dist,
        [
            f"The data are f({xs[0]}) = 1/{xs[0]}, f({xs[1]}) = 1/{xs[1]}, f({xs[2]}) = 1/{xs[2]} and f({xs[3]}) = 1/{xs[3]}.",
            f"p(x) = {poly_text(p)}.",
        ],
    )


def _poly_question(text, p, dist, working):
    return dict(
        text=text[0],
        latex=text[1],
        answer=poly_machine(p),
        answer_text=poly_text(p),
        answer_kind="expression",
        distractors=dist,
        working=working,
    )


def _frac_text(v) -> str:
    v = fr(v)
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def _half(rng, xs):
    """A half-integer strictly between the first and last node that is not a node."""
    opts = [Fraction(2 * k + 1, 2) for k in range(xs[0], xs[-1])]
    opts = [v for v in opts if xs[0] < v < xs[-1]]
    return rng.choice(opts)


# ---------------------------------------------------------------------------------------------------------------
# exam chain


@generator("exd_x_r4b_lagrange_interpolation", level=20, presets={"exam": dict(tier="exam")})
def exd_x_r4b_lagrange_interpolation(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    return with_redraw(lambda: _exam(rng), "an exam Lagrange question")


def _exam(rng):
    for _ in attempts(500, "exam nodes for 1/x"):
        xs = sorted(rng.sample(range(1, 8), 3))
        if xs[1] - xs[0] == xs[2] - xs[1]:
            continue
        ts = [v for v in range(xs[0] + 1, xs[2]) if v not in xs]
        if ts:
            t = Fraction(rng.choice(ts))
            if (
                poly_eval(basis(xs, 1), t).denominator > 1
                and poly_eval(lagrange(xs, [Fraction(1, x) for x in xs]), t) > 0
            ):
                break
    ys = [Fraction(1, x) for x in xs]
    p = lagrange(xs, ys)
    check_polynomial(xs, ys, p)
    l1 = poly_eval(basis(xs, 1), t)
    check(l1.denominator > 1, "L1 at t is a whole number")
    value = poly_eval(p, t)
    err = Fraction(1) / t - value
    check(err == Fraction(1) / t - value and value != Fraction(1) / t, "the error is zero")
    # the closed form for 1/x: p(x) = (ab + bc + ca - (a + b + c)x + x^2) / (abc)
    a, b, c = xs
    closed = [Fraction(a * b + b * c + c * a, a * b * c), Fraction(-(a + b + c), a * b * c), Fraction(1, a * b * c)]
    check(trim(closed) == trim(p), "closed form for 1/x disagrees")
    tt = int(t)
    stem, latex = say_marked(
        rng.choice(
            [
                f"The function «f(x) = 1/x» is interpolated by the quadratic polynomial «p» at the nodes "
                f"«x₀ = {a}», «x₁ = {b}» and «x₂ = {c}».",
                f"Let «p» be the quadratic polynomial that agrees with «f(x) = 1/x» at the nodes "
                f"«x₀ = {a}», «x₁ = {b}» and «x₂ = {c}».",
                f"The quadratic «p» interpolates «f(x) = 1/x» at the three nodes «x₀ = {a}», «x₁ = {b}» and «x₂ = {c}».",
            ]
        )
    )
    other = [poly_eval(basis(xs, 0), t), poly_eval(basis(xs, 2), t)]
    num_only = poly_eval(numerator(xs, 1), t)
    p_a = fraction_part(
        f"Find the value at «x = {tt}» of the basis polynomial «L₁(x)», the one that is 1 at «x₁» and 0 at the "
        "other two nodes, as an exact fraction.",
        l1,
        [(M_BASIS_OTHER, other[0]), (M_BASIS_OTHER, other[1]), (M_BASIS_NUM, num_only), (M_BASIS_NODE, Fraction(1))],
        2,
        working=[f"L₁({tt}) = ({tt} − {a})({tt} − {c})/(({b} − {a})({b} − {c})) = {_frac_text(l1)}."],
    )
    p_b = dict(
        prompt="Find «p(x)» in expanded form.",
        answer_kind="expression",
        answer=poly_machine(p),
        answer_text=poly_text(p),
        marks=3,
        working=[f"p(x) = {poly_text(p)}."],
        distractors=poly_distractors(p, _slips(xs, ys)),
    )
    if len(p_b["distractors"]) < 2:
        raise Redraw
    p_c = fraction_part(
        f"Using your «p(x)», find «p({tt})» as an exact fraction.",
        value,
        [(tag, poly_eval(cc, t)) for tag, cc in _slips(xs, ys)] + [(M_F_FOR_P, Fraction(1) / t)],
        1,
        ft=f"Subs(part_b, (x,), ({tt},))",
        working=[f"p({tt}) = {_frac_text(value)}."],
    )
    p_d = fraction_part(
        f"Using your «p({tt})», find the interpolation error «f({tt}) − p({tt})» as an exact fraction.",
        err,
        [
            (M_NEG, -err),
            (M_F_FOR_P, value),
            ("num2-lagrange-error-as-ratio", Fraction(1) / t / value if value else None),
            (
                "num2-lagrange-error-vs-nearest-node",
                Fraction(1) / t - Fraction(ys[min(range(3), key=lambda i: abs(xs[i] - tt))]),
            ),
        ],
        2,
        ft=f"Rational(1, {tt}) - part_c",
        working=[f"f({tt}) − p({tt}) = 1/{tt} − {_frac_text(value)} = {_frac_text(err)}."],
    )
    for part in (p_a, p_b, p_c, p_d):
        part["prompt"], part["prompt_latex"] = say_marked(part["prompt"])
    return dict(text=stem, latex=latex, parts=[p_a, p_b, p_c, p_d])
