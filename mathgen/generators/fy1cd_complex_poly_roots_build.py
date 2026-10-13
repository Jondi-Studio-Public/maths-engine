"""Generates `ucv1_complex_poly_roots` questions and the exam chain `exd_x_fy_complex_poly_roots`.

``ucv1-complex-roots-real-polynomials`` (``complex-analysis``, answer ``number``): a cubic or quartic with real
coefficients and a stated non-real root. The conjugate root theorem gives a second root, a real quadratic factor
x² - 2px + (p² + q²), and from there the rest.

* easy: a cubic with one non-real root stated; find its real root (the roots sum to minus the x² coefficient).
* medium: a quartic with one non-real root stated and two real roots; find the product of the real roots, or the sum of
  their squares.
* hard: a cubic x³ + Ax² + kx + C with a non-real root stated; find the coefficient k (the pairwise sums of roots); or
  a quartic given only in expanded form with one non-real root stated, whose largest real root is needed.
* exam (``exd_x_fy_complex_poly_roots``): a quartic with a non-real root stated. (a) the other non-real root, (b) the
  constant of the real quadratic factor, (c) the constant of the other quadratic factor (following through from (b)),
  (d) the largest real root (following through from (c)).

Every polynomial is built from its factors; the stated root is checked against the expanded polynomial with SymPy,
and the real roots with ``real_roots``.
"""

from __future__ import annotations

import math

import sympy

from ..answers import Distractor
from ..core import attempts, check, generator, join_terms, latex_answer, pick_distractors
from ._fy1cd_kit import F, ctext, int_dist, poly_text, poly_tex, say

_X = sympy.Symbol("x")


def _expand(factors) -> list[int]:
    poly = sympy.Poly(sympy.prod(factors), _X)
    return [int(c) for c in poly.all_coeffs()]


def _check_root(coeffs: list[int], p: int, q: int) -> None:
    value = sum(c * (p + q * sympy.I) ** (len(coeffs) - 1 - i) for i, c in enumerate(coeffs))
    check(sympy.expand(value) == 0, "the stated root is not a root")


def _real_roots(coeffs: list[int]) -> list[int]:
    roots = sympy.Poly(coeffs, _X).real_roots()
    return sorted(int(r) for r in roots)


def _root_formula(p: int, q: int) -> F:
    text = ctext(p, q)
    tex = text.replace("i", r"\,i") if "i" in text else text
    return F(text, tex)


def _poly_formula(coeffs: list[int], rhs: str = " = 0") -> F:
    return F(poly_text(coeffs) + rhs, poly_tex(coeffs) + rhs)


def _draw_complex(rng, real_part_zero=True):
    p = rng.choice([v for v in range(-3, 4) if real_part_zero or v != 0])
    q = rng.randint(1, 3)
    return p, q, p * p + q * q


def _draw_real(rng, avoid=()):
    return rng.choice([v for v in range(-5, 6) if v != 0 and v not in avoid])


def _dists(answer, candidates, n: int = 4) -> list[Distractor]:
    return pick_distractors(str(answer), [int_dist(v, t) for v, t in candidates], n)


# -- easy: a cubic, one real root ---------------------------------------------------------------------------------


def _easy(rng):
    p, q, m = _draw_complex(rng, real_part_zero=False)
    r = _draw_real(rng)
    coeffs = _expand([_X - r, _X**2 - 2 * p * _X + m])
    b, d = coeffs[1], coeffs[3]
    _check_root(coeffs, p, q)
    check(_real_roots(coeffs) == [r], "SymPy's real root differs")
    stem = say(
        "The cubic ",
        _poly_formula(coeffs),
        " has real coefficients and one of its roots is ",
        _root_formula(p, q),
        ". Find its real root.",
    )
    ds = _dists(
        r,
        [
            (-b - p, "cpoly-sum-forgets-conjugate"),
            (b - 2 * p, "cpoly-vieta-sign-slip"),
            (-b, "cpoly-ignores-complex-pair"),
            (-r, "cpoly-product-sign-slip"),
            (-b + 2 * p, "cpoly-vieta-sign-slip"),
        ],
    )
    check(len(ds) >= 2, "too few distractors")
    return dict(
        text=stem[0],
        latex=stem[1],
        answer_kind="number",
        answer=r,
        answer_text=str(r),
        answer_latex=str(r),
        working=[
            f"The coefficients are real, so the conjugate {ctext(p, -q)} is also a root.",
            f"The three roots add to {-b}, and the two non-real roots add to {2 * p}, so the real root is {-b} - ({2 * p}) = {r}.",
        ],
        distractors=ds,
    )


# -- medium: a quartic, the real roots ----------------------------------------------------------------------------


def _quartic(rng):
    for _ in attempts(500, "a quartic with moderate coefficients"):
        p, q, m = _draw_complex(rng)
        r1 = _draw_real(rng)
        r2 = _draw_real(rng, avoid=(r1,))
        r1, r2 = min(r1, r2), max(r1, r2)
        coeffs = _expand([_X**2 - 2 * p * _X + m, (_X - r1) * (_X - r2)])
        if max(abs(c) for c in coeffs) <= 60:
            break
    _check_root(coeffs, p, q)
    check(_real_roots(coeffs) == [r1, r2], "SymPy's real roots differ")
    return p, q, m, r1, r2, coeffs


def _medium(rng):
    p, q, m, r1, r2, coeffs = _quartic(rng)
    e = coeffs[4]
    u = -(r1 + r2)
    v = r1 * r2
    variant = rng.choice(["product", "sumsq"])
    if variant == "product":
        ask, answer = "the product of the two real roots", v
        cands = [
            (e, "cpoly-product-not-divided"),
            (-v, "cpoly-product-sign-slip"),
            (e // (q * q) if e % (q * q) == 0 else u, "cpoly-modulus-imag-part-only"),
            (-u, "cpoly-sum-instead-of-product"),
        ]
        working = [
            f"The conjugate {ctext(p, -q)} is a root, so {poly_text([1, -2 * p, m])} is a factor.",
            f"The product of all four roots is {e} and the non-real pair has product {m}, so the real roots multiply to {e}/{m} = {v}.",
        ]
    else:
        ask, answer = "the sum of the squares of the two real roots", u * u - 2 * v
        cands = [
            (u * u, "cpoly-sumsq-forgot-cross-term"),
            (u * u + 2 * v, "cpoly-sumsq-cross-term-sign"),
            (u * u - v, "cpoly-sumsq-cross-term-not-doubled"),
            (-u, "cpoly-sum-instead-of-sumsq"),
        ]
        working = [
            f"The conjugate {ctext(p, -q)} is a root, so the real roots add to {-coeffs[1]} - 2({p}) = {-u} and multiply to {coeffs[4]}/{m} = {v}.",
            f"Their squares add to ({-u})² - 2({v}) = {answer}.",
        ]
    stem = say(
        "The quartic ",
        _poly_formula(coeffs),
        " has real coefficients and two real roots. One of its non-real roots is ",
        _root_formula(p, q),
        ". Find ",
        ask,
        ".",
    )
    ds = _dists(answer, cands)
    check(len(ds) >= 2, "too few distractors")
    return dict(
        text=stem[0],
        latex=stem[1],
        answer_kind="number",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=working,
        distractors=ds,
    )


# -- hard -------------------------------------------------------------------------------------------------------


def _hard_param(rng):
    for _ in attempts(500, "a cubic with moderate coefficients"):
        p, q, m = _draw_complex(rng)
        r = _draw_real(rng)
        coeffs = _expand([_X - r, _X**2 - 2 * p * _X + m])
        k = coeffs[2]
        if max(abs(c) for c in coeffs) <= 40 and k not in (m, 2 * p * r) and coeffs[3] != 0:
            break
    _check_root(coeffs, p, q)
    a_coef, c_coef = coeffs[1], coeffs[3]
    # Independent route: with k unknown, the stated root satisfies the cubic only for this k.
    kk = sympy.Symbol("k")
    cubic = sum(c * (p + q * sympy.I) ** (3 - i) for i, c in enumerate([1, a_coef, kk, c_coef]))
    solved = sympy.solve(sympy.expand(cubic), kk)
    check(solved == [k], "SymPy's value of k differs")
    plain = join_terms([(1, "x³"), (a_coef, "x²"), (1, "kx"), (c_coef, "")])
    tex = plain.replace("x³", "x^{3}").replace("x²", "x^{2}")
    stem = say(
        "The cubic ",
        F(plain + " = 0", tex + " = 0"),
        " has real coefficients and a non-real root ",
        _root_formula(p, q),
        ". Find k.",
    )
    ds = _dists(
        k,
        [
            (m, "cpoly-pairwise-sum-missing-term"),
            (2 * p * r, "cpoly-pairwise-sum-missing-term"),
            (m - 2 * p * r, "cpoly-pairwise-sign-slip"),
            (-k, "cpoly-vieta-sign-slip"),
            (m + p * r, "cpoly-pairwise-sum-missing-term"),
        ],
    )
    check(len(ds) >= 2, "too few distractors")
    return dict(
        text=stem[0],
        latex=stem[1],
        answer_kind="number",
        answer=k,
        answer_text=str(k),
        answer_latex=str(k),
        working=[
            f"The conjugate {ctext(p, -q)} is also a root. The pair has product {m}, so the real root is {-c_coef}/{m} = {r}.",
            f"The sum of the products of pairs of roots is k, so k = {m} + 2({p})({r}) = {k}.",
        ],
        distractors=ds,
    )


def _hard_largest(rng):
    p, q, m, r1, r2, coeffs = _quartic(rng)
    stem = say(
        "The quartic ",
        _poly_formula(coeffs),
        " has real coefficients and a non-real root ",
        _root_formula(p, q),
        ". Find its largest real root.",
    )
    u, v = -(r1 + r2), r1 * r2
    ds = _dists(
        r2,
        [
            (r1, "cpoly-wrong-real-root"),
            (-r2, "cpoly-root-sign-slip"),
            (-r1, "cpoly-root-sign-slip"),
            (v, "cpoly-product-instead-of-root"),
            (-u, "cpoly-sum-instead-of-root"),
        ],
    )
    check(len(ds) >= 2, "too few distractors")
    return dict(
        text=stem[0],
        latex=stem[1],
        answer_kind="number",
        answer=r2,
        answer_text=str(r2),
        answer_latex=str(r2),
        working=[
            f"The conjugate {ctext(p, -q)} is a root, so {poly_text([1, -2 * p, m])} is a factor.",
            f"Dividing leaves {poly_text([1, u, v])} = 0, whose roots are {r1} and {r2}.",
        ],
        distractors=ds,
    )


@generator(
    "ucv1_complex_poly_roots",
    level=18,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def ucv1_complex_poly_roots(rng, tier="easy"):
    """Roots of a real polynomial with a stated non-real root; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_param, _hard_largest])(rng)


# -- exam -------------------------------------------------------------------------------------------------------

_PI = "re(Subs(part_a, (i,), (I,)))"
_PJ = "im(Subs(part_a, (i,), (I,)))"


def _expr_dist(p: int, q: int, tag: str) -> Distractor:
    text = ctext(p, q)
    return Distractor(text, tag, text)


@generator("exd_x_fy_complex_poly_roots", level=18, presets={"exam": dict(tier="exam")})
def exd_x_fy_complex_poly_roots(rng, tier="exam"):
    """Exam chain: the conjugate root, the real quadratic factor, the other factor and the largest real root."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(200, "a quartic whose non-real pair does not have product 1"):
        p, q, m, r1, r2, coeffs = _quartic(rng)
        if m >= 2:
            break
    e = coeffs[4]
    u, v = -(r1 + r2), r1 * r2
    disc = u * u - 4 * v
    check(math.isqrt(disc) ** 2 == disc, "the real roots are not integers")
    check(m * v == e and coeffs[1] == u - 2 * p, "factor product differs from the polynomial")

    a_dist = pick_distractors(
        ctext(p, -q),
        [
            _expr_dist(-p, -q, "cpoly-conjugate-negates-both"),
            _expr_dist(-p, q, "cpoly-conjugate-negates-real"),
            _expr_dist(p, q, "cpoly-conjugate-repeats-root"),
            _expr_dist(q, p, "cpoly-conjugate-swaps-parts"),
        ],
        4,
    )
    b_cands = [
        ((p + q) ** 2, "cpoly-square-of-sum"),
        (p * p - q * q, "cpoly-i-squared-sign"),
        (q * q, "cpoly-modulus-imag-part-only"),
        (p * p, "cpoly-modulus-real-part-only"),
    ]
    b_dist = _dists(m, b_cands)
    c_cands = [
        (e, "cpoly-product-not-divided"),
        (e * m, "cpoly-product-multiplied"),
        (-v, "cpoly-product-sign-slip"),
        (e - m, "cpoly-constant-subtracted"),
    ]
    c_dist = _dists(v, c_cands)
    d_dist = _dists(
        r2,
        [
            (r1, "cpoly-wrong-real-root"),
            (-r2, "cpoly-root-sign-slip"),
            (v, "cpoly-product-instead-of-root"),
            (-u, "cpoly-sum-instead-of-root"),
        ],
    )

    # Offering the follow-through value of a learner's own earlier wrong answer as a wrong option would deny the marks
    # that follow-through exists to give, so those values are not offered.
    def own_c(b_value):
        return e / b_value if b_value else None

    own = {own_c(int(b.text)) for b in b_dist}
    c_dist = [c for c in c_dist if int(c.text) not in own]

    def own_d(c_value):
        disc_c = u * u - 4 * c_value
        return (-u + math.sqrt(disc_c)) / 2 if disc_c >= 0 else None

    own_ds = {own_d(int(c.text)) for c in c_dist} | {own_d(v)}
    d_dist = [d for d in d_dist if float(d.text) not in own_ds - {float(r2)}]
    for label, ds in (("a", a_dist), ("b", b_dist), ("c", c_dist), ("d", d_dist)):
        check(len(ds) >= 2, f"too few ({label}) distractors: {ds}")

    stem = say(
        "The quartic ",
        _poly_formula(coeffs, ""),
        " has real coefficients. One of its roots is ",
        _root_formula(p, q),
        " and the other two roots are real.",
    )
    ft_c = f"{e}/part_b"
    ft_d = f"(-({u}) + sqrt(({u})**2 - 4*part_c))/2"
    ft_b = f"{_PI}**2 + {_PJ}**2"
    return dict(
        text=stem[0],
        latex=stem[1],
        parts=[
            dict(
                prompt="Write down the other non-real root.",
                answer_kind="expression",
                answer=ctext(p, -q),
                answer_text=ctext(p, -q),
                answer_latex=latex_answer(ctext(p, -q)),
                marks=1,
                working=["The coefficients are real, so non-real roots come in conjugate pairs."],
                distractors=a_dist,
            ),
            dict(
                prompt="The two non-real roots are the roots of a quadratic factor x² + sx + m with real coefficients. Find m.",
                prompt_latex=r"\text{The two non-real roots are the roots of a quadratic factor }x^{2}+sx+m\text{ with real coefficients. Find }m\text{.}",
                answer_kind="number",
                answer=m,
                answer_text=str(m),
                answer_latex=str(m),
                marks=2,
                follow_through=ft_b,
                working=[
                    f"m is the product of the two roots: ({ctext(p, q)})({ctext(p, -q)}) = {p * p} + {q * q} = {m}."
                ],
                distractors=b_dist,
            ),
            dict(
                prompt="The other quadratic factor is x² + ux + v. Find v.",
                prompt_latex=r"\text{The other quadratic factor is }x^{2}+ux+v\text{. Find }v\text{.}",
                answer_kind="number",
                answer=v,
                answer_text=str(v),
                answer_latex=str(v),
                marks=2,
                follow_through=ft_c,
                working=[f"The constant terms multiply: {m}v = {e}, so v = {v}."],
                distractors=c_dist,
            ),
            dict(
                prompt="Hence find the largest real root of the quartic.",
                answer_kind="number",
                answer=r2,
                answer_text=str(r2),
                answer_latex=str(r2),
                marks=3,
                follow_through=ft_d,
                working=[
                    f"The x³ coefficient gives u - 2({p}) = {coeffs[1]}, so u = {u}.",
                    f"{poly_text([1, u, v])} = 0 has roots {r1} and {r2}, so the largest is {r2}.",
                ],
                distractors=d_dist,
            ),
        ],
    )
