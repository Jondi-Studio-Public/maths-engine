"""Generates `r3i_complex_factor` questions (easy, medium, hard) and the exam chain `r3i_exam_complex_factor`.

Skill ``factor-polynomial-over-complex`` (``complex-analysis``, answer ``expression``): factorising a real polynomial
into linear factors over the complex numbers, and into the real quadratic factor that holds a complex pair. Every
non-real zero comes with its conjugate, and the learner types ``i`` for the imaginary unit.

* easy: a real quadratic with no real zeros (``x² + 4x + 13``, ``2x² + 4x + 10``, ``x² + 9``) factorised over C.
* medium: a cubic with one stated real zero, either factorised completely over C or asked for its real quadratic factor
  (a polynomial with real coefficients, written out).
* hard: a quartic with one stated non-real zero, factorised completely over C (the other quadratic factor may split into
  real linear factors or hold a second conjugate pair), or ``x⁴ - a⁴`` factorised over C.
* exam (``r3i_exam_complex_factor``): a quartic with a stated non-real zero ``p + qi``; (a) the constant term of the real
  quadratic factor with that zero, (b) the constant term of the other quadratic factor (follows through from (a)),
  (c) the larger real zero of that factor (follows through from (b)), (d) the complete factorisation over C.

Each answer is expanded again with SymPy and compared with the polynomial, and the zeros of the polynomial are found
independently with a numerical root-finder and compared with the zeros the factors name.
"""

from __future__ import annotations

import re

import mpmath as mp
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from ..core import attempts, check, generator, join_terms
from ._r3i_kit import expr_dist, expr_fields, num_cands, num_fields, pick, pick_text, retry_short

X = sympy.Symbol("x")
I = sympy.I
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _sgn(n: int) -> str:
    return f" + {n}" if n >= 0 else f" - {-n}"


def _linear(re: int, im: int = 0) -> str:
    """The factor x - (re + im·i) as it is typed: (x + 2 - 3i), (x - 4), x."""
    out = "x"
    if re != 0:
        out += _sgn(-re)
    if im != 0:
        mag = abs(im)
        out += f" {'-' if im > 0 else '+'} {'' if mag == 1 else mag}i"
    return f"({out})" if (re != 0 or im != 0) else out


def _pair(re: int, im: int) -> str:
    """The two factors for the zeros re ± im·i, im > 0."""
    return _linear(re, -im) + _linear(re, im)


def _factored(lead: int, reals, pairs) -> str:
    parts = [("" if lead == 1 else str(lead))]
    for r in sorted(reals):
        parts.append(_linear(r))
    for re, im in pairs:
        parts.append(_pair(re, im))
    return "".join(parts)


def _expr(lead, reals, pairs):
    e = sympy.Integer(lead)
    for r in reals:
        e *= X - r
    for re, im in pairs:
        e *= (X - re) ** 2 + im * im
    return sympy.expand(e)


def _poly_text(expr) -> str:
    """An expanded polynomial as written in a stem: x⁴ - 2x³ + 5x² - 8x + 4."""
    poly = sympy.Poly(expr, X)
    out = ""
    for (power,), c in poly.terms():
        c = int(c)
        mag = abs(c)
        body = (str(mag) if (power == 0 or mag != 1) else "") + (
            "" if power == 0 else "x" + ("" if power == 1 else {2: "²", 3: "³", 4: "⁴"}[power])
        )
        out += (f"-{body}" if c < 0 else body) if not out else f" {'-' if c < 0 else '+'} {body}"
    return out


def _poly_ascii(expr) -> str:
    return _poly_text(expr).replace("²", "^2").replace("³", "^3").replace("⁴", "^4")


def _order(z):
    return (round(z.real, 5), round(z.imag, 5))


def _zeros_check(expr, reals, pairs):
    """The numerical zeros of the polynomial are exactly the zeros the factors name."""
    coeffs = [int(c) for c in sympy.Poly(expr, X).all_coeffs()]
    with mp.workdps(30):
        found = sorted((complex(z) for z in mp.polyroots(coeffs, maxsteps=300, extraprec=80)), key=_order)
    named = sorted(
        [complex(r, 0) for r in reals] + [complex(re, s * im) for re, im in pairs for s in (1, -1)], key=_order
    )
    check(len(found) == len(named), "number of zeros differs")
    for a, b in zip(found, named):
        check(abs(a - b) < 1e-8, f"zero {a} differs from {b}")


def _typed_to_expr(text: str):
    """The typed factorisation read back as a SymPy expression (i is the imaginary unit)."""
    text = re.sub(r"(?<![A-Za-z])i(?![A-Za-z])", "I", text)
    return parse_expr(text, local_dict={"x": X, "I": I}, transformations=_TRANSFORMS)


def _answer(lead, reals, pairs, expr) -> dict:
    _zeros_check(expr, reals, pairs)
    typed = _factored(lead, reals, pairs)
    check(sympy.expand(_typed_to_expr(typed) - expr) == 0, "the typed factorisation does not expand to the polynomial")
    return expr_fields(None, typed)


# -- easy ---------------------------------------------------------------------------------------------------


def _easy(rng):
    lead = rng.choice([1, 1, 1, 2, 3])
    shape = rng.choice(["shifted", "shifted", "pure"])
    if shape == "pure":
        p, q = 0, rng.randint(2, 9)
    else:
        p, q = rng.choice([x for x in range(-6, 7) if x != 0]), rng.randint(1, 6)
    expr = _expr(lead, [], [(p, q)])
    fields = _answer(lead, [], [(p, q)], expr)
    lead_text = "" if lead == 1 else str(lead)
    cands = [
        expr_dist(_factored(lead, [], [(-p, q)]), "r3i-factor-root-sign-read-from-coefficient") if p != 0 else None,
        expr_dist(_factored(lead, [], [(0, q)]), "r3i-factor-real-part-dropped") if p != 0 else None,
        expr_dist(_factored(1, [], [(p, q)]), "r3i-factor-leading-coefficient-dropped") if lead != 1 else None,
        expr_dist(_linear(p, -q) + _linear(p, -q), "r3i-factor-conjugate-missed"),
        expr_dist(lead_text + _linear(q) + _linear(-q), "r3i-factor-sum-of-squares-as-difference") if p == 0 else None,
        expr_dist(
            _factored(lead, [], [(p, q * q)]) if q * q <= 36 else _factored(lead, [], [(p, q + 1)]),
            "r3i-factor-imaginary-part-squared",
        ),
    ]
    stem = f"Factorise {_poly_text(expr)} into linear factors over the complex numbers. Use i for √(-1)."
    return dict(
        text=stem,
        working=[
            "The quadratic formula gives zeros " + (f"{p} ± {q}i" if p else f"±{q}i") + ".",
            f"So the polynomial equals {fields['answer_text']}.",
        ],
        distractors=pick_text(fields, cands),
        **fields,
    )


# -- medium -------------------------------------------------------------------------------------------------


def _medium(rng):
    r = rng.choice([x for x in range(-4, 5) if x != 0])
    p, q = rng.randint(-3, 3), rng.randint(1, 4)
    lead = 1
    expr = _expr(lead, [r], [(p, q)])
    quad = sympy.expand((X - p) ** 2 + q * q)
    if rng.random() < 0.55:
        fields = _answer(lead, [r], [(p, q)], expr)
        cands = [
            expr_dist(_factored(lead, [-r], [(p, q)]), "r3i-factor-root-sign-read-from-coefficient"),
            expr_dist(_factored(lead, [r], [(-p, q)]), "r3i-factor-root-sign-read-from-coefficient"),
            expr_dist(_linear(r) + _linear(p, q) + _linear(p, q), "r3i-factor-conjugate-missed"),
            expr_dist(_linear(r) + _pair(0, q), "r3i-factor-real-part-dropped") if p != 0 else None,
        ]
        working = [
            f"Dividing by (x {'-' if r > 0 else '+'} {abs(r)}) leaves the quadratic {_poly_text(quad)}, whose zeros are {p} ± {q}i.",
            f"So P(x) = {fields['answer_text']}.",
        ]
        ask = "Factorise P(x) completely into linear factors over the complex numbers. Use i for √(-1)."
    else:
        fields = dict(expr_fields(None, _poly_ascii(quad)), answer_form="expanded")
        cands = [
            expr_dist(_poly_ascii(sympy.expand((X + p) ** 2 + q * q)), "r3i-factor-root-sign-read-from-coefficient")
            if p != 0
            else None,
            expr_dist(_poly_ascii(sympy.expand((X - p) ** 2 - q * q)), "r3i-factor-quadratic-sign-of-q-squared"),
            expr_dist(_poly_ascii(sympy.expand((X - p) ** 2 + q)), "r3i-factor-imaginary-part-not-squared")
            if q > 1
            else None,
            expr_dist(_poly_ascii(sympy.expand(X**2 - 2 * p * X + p * p)), "r3i-factor-imaginary-part-lost"),
        ]
        working = [
            f"Dividing by (x {'-' if r > 0 else '+'} {abs(r)}) leaves {_poly_text(quad)}.",
            f"Its discriminant is {(-2 * p) ** 2 - 4 * (p * p + q * q)} < 0, so it has no real factors.",
        ]
        ask = "Find the quadratic factor of P(x) that has real coefficients and no real zeros. Write it out in expanded form."
    check(sympy.simplify(sympy.expand((X - r) * quad) - expr) == 0, "(x - r) times the quadratic differs from P")
    stem = f"P(x) = {_poly_text(expr)}. Given that x = {r} is a zero of P, {ask[0].lower() + ask[1:]}"
    return dict(text=stem, working=working, distractors=pick_text(fields, cands), **fields)


# -- hard ---------------------------------------------------------------------------------------------------


def _hard_quartic(rng):
    for _ in attempts(500, "a quartic whose second quadratic factor is not the first"):
        p, q = rng.randint(-3, 3), rng.randint(1, 3)
        if rng.random() < 0.6:
            r1, r2 = sorted(rng.sample([x for x in range(-4, 5) if x != 0], 2))
            reals, pairs2 = [r1, r2], []
        else:
            reals = []
            s, t = rng.randint(-3, 3), rng.randint(1, 3)
            if (s, t) == (p, q):
                continue
            pairs2 = [(s, t)]
        break
    pairs = [(p, q)] + pairs2
    expr = _expr(1, reals, pairs)
    fields = _answer(1, reals, pairs, expr)
    others = f"{reals[0]} and {reals[1]}" if reals else f"{pairs2[0][0]} ± {pairs2[0][1]}i"
    wrong = [
        expr_dist(_factored(1, reals, [(-p, q)] + pairs2), "r3i-factor-root-sign-read-from-coefficient"),
        expr_dist(_factored(1, [-x for x in reals], pairs), "r3i-factor-root-sign-read-from-coefficient")
        if reals
        else expr_dist(
            _factored(1, [], [(p, q), (-pairs2[0][0], pairs2[0][1])]), "r3i-factor-root-sign-read-from-coefficient"
        ),
        expr_dist(
            _linear(p, -q) + _linear(p, -q) + "".join(_linear(r) for r in reals) + "".join(_pair(*pr) for pr in pairs2),
            "r3i-factor-conjugate-missed",
        ),
        expr_dist(_factored(1, reals[:1], pairs), "r3i-factor-zero-dropped") if reals else None,
    ]
    return dict(
        text=(
            f"P(x) = {_poly_text(expr)} has a zero at x = {p} + {q if q != 1 else ''}i. "
            "Factorise P(x) completely into linear factors over the complex numbers. Use i for √(-1)."
        ),
        working=[
            f"P has real coefficients, so {p} - {q if q != 1 else ''}i is also a zero, giving the real factor {_poly_text(sympy.expand((X - p) ** 2 + q * q))}.",
            f"Dividing leaves a quadratic with zeros {others}.",
            f"So P(x) = {fields['answer_text']}.",
        ],
        distractors=pick_text(fields, wrong),
        **fields,
    )


def _hard_fourth_power(rng):
    a = rng.randint(1, 4)
    scale = rng.choice([1, 1, 2, 3])
    expr = sympy.expand(scale * (X**4 - a**4))
    reals, pairs = [-a, a], [(0, a)]
    fields = _answer(scale, reals, pairs, expr)
    wrong = [
        expr_dist(
            _factored(scale, reals, [(0, a * a)]) if a * a <= 16 else _factored(scale, reals, [(0, a + 1)]),
            "r3i-factor-imaginary-part-squared",
        ),
        expr_dist(
            _factored(scale, reals, []) + _linear(a * a) + _linear(-a * a), "r3i-factor-sum-of-squares-as-difference"
        ),
        expr_dist(_factored(scale, reals, [(a, a)]), "r3i-factor-real-part-dropped"),
        expr_dist(_factored(scale, [-a], [(0, a)]), "r3i-factor-zero-dropped"),
    ]
    return dict(
        text=f"Factorise {_poly_text(expr)} completely into linear factors over the complex numbers. Use i for √(-1).",
        working=[
            f"x⁴ - {a**4} = (x² - {a * a})(x² + {a * a}) = (x - {a})(x + {a})(x - {a}i)(x + {a}i).",
            f"Allowing for the factor {scale}, the result is {fields['answer_text']}.",
        ],
        distractors=pick_text(fields, wrong),
        **fields,
    )


def _hard(rng):
    return _hard_quartic(rng) if rng.random() < 0.8 else _hard_fourth_power(rng)


@generator(
    "r3i_complex_factor",
    level=14,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def r3i_complex_factor(rng, tier="easy"):
    """Factorising real polynomials over the complex numbers; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ---------------------------------------------------------------------------------------------------


@generator("r3i_exam_complex_factor", level=14, presets={"exam": dict(tier="exam")})
@retry_short
def r3i_exam_complex_factor(rng, tier="exam"):
    """Exam chain on a quartic with a stated non-real zero."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "a quartic whose other quadratic factor has two distinct integer zeros"):
        p, q = rng.choice([x for x in range(-3, 4) if x != 0]), rng.randint(1, 3)
        r1, r2 = sorted(rng.sample([x for x in range(-4, 5) if x != 0], 2))
        if r1 + r2 != 0 and r1 * r2 > -20:
            break
    m = p * p + q * q
    reals, pairs = [r1, r2], [(p, q)]
    expr = _expr(1, reals, pairs)
    poly = sympy.Poly(expr, X)
    a3, a0 = int(poly.coeff_monomial(X**3)), int(poly.coeff_monomial(1))
    e = -(r1 + r2)  # the other factor is x² + e x + f
    f = r1 * r2
    check(a3 == e - 2 * p and a0 == m * f, "quartic coefficients differ from the factors")
    larger = max(r1, r2)
    ans_d = _answer(1, reals, pairs, expr)
    f_a, f_b, f_c = num_fields(m), num_fields(f), num_fields(larger)
    wrong_d = [
        expr_dist(_factored(1, reals, [(-p, q)]), "r3i-factor-root-sign-read-from-coefficient"),
        expr_dist(_factored(1, [-r1, -r2], pairs), "r3i-factor-root-sign-read-from-coefficient"),
        expr_dist(_linear(p, -q) + _linear(p, -q) + _linear(r1) + _linear(r2), "r3i-factor-conjugate-missed"),
        expr_dist(_factored(1, reals[:1], pairs), "r3i-factor-zero-dropped"),
    ]
    return dict(
        text=(
            f"P(x) = {_poly_text(expr)} has real coefficients, and x = {p} + {q if q != 1 else ''}i is a zero of P. "
            "Write i for √(-1)."
        ),
        parts=[
            dict(
                prompt="P(x) has a quadratic factor with real coefficients and the form x² + bx + c, having this zero. Find c.",
                marks=1,
                sensible_range=(0, None),
                working=[f"The conjugate {p} - {q}i is also a zero, so c = ({p})² + {q}² = {m}."],
                distractors=pick(
                    f_a,
                    num_cands(
                        [
                            (p * p - q * q, "r3i-factor-quadratic-sign-of-q-squared"),
                            (p + q, "r3i-factor-constant-sum-not-product"),
                            (q * q, "r3i-factor-real-part-dropped"),
                            (p * p, "r3i-factor-imaginary-part-lost"),
                        ],
                        m,
                    ),
                ),
                **f_a,
            ),
            dict(
                prompt="P(x) is the product of that factor and a second quadratic x² + ex + f. Find f.",
                marks=2,
                follow_through=f"({a0})/part_a",
                sensible_range=(-100, 100),
                working=[f"Comparing constant terms, {m}f = {a0}, so f = {f}."],
                distractors=pick(
                    f_b,
                    num_cands(
                        [
                            (a0 * m, "r3i-factor-constant-multiplied"),
                            (a0 - m, "r3i-factor-constant-subtracted"),
                            (-f, "r3i-factor-constant-sign"),
                        ],
                        f,
                    ),
                ),
                **f_b,
            ),
            dict(
                prompt="The second quadratic factor has two distinct integer zeros. Find the larger one.",
                marks=2,
                follow_through=f"(-({e})+sqrt(({e})**2-4*part_b))/2",
                sensible_range=(-100, 100),
                working=[
                    f"Comparing x³ terms gives e = {e}. The factor {join_terms([(1, 'x²'), (e, 'x'), (f, '')])} has zeros {r1} and {r2}."
                ],
                distractors=pick(
                    f_c,
                    num_cands(
                        [
                            (min(r1, r2), "r3i-factor-smaller-zero-given"),
                            (-larger, "r3i-factor-root-sign-read-from-coefficient"),
                            (e, "r3i-factor-linear-coefficient-given"),
                        ],
                        larger,
                    ),
                ),
                **f_c,
            ),
            dict(
                prompt="Hence factorise P(x) completely into linear factors over the complex numbers.",
                marks=3,
                working=[f"The zeros are {r1}, {r2} and {p} ± {q}i."],
                distractors=pick_text(ans_d, wrong_d),
                **ans_d,
            ),
        ],
    )
