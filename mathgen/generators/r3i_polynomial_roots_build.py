"""Generates `r3i_poly_roots` questions (easy, medium, hard) and the exam chain `r3i_exam_poly_roots`.

Skill ``polynomial-roots-sum-product`` (``functions-sequences``, answer ``expression``): symmetric functions of the
roots of a quadratic or cubic read from its coefficients, and polynomials built from transformed roots. The roots are
never asked for, and the coefficients are chosen so that the roots are irrational or complex, so the only way through is
the relationships among the roots and the coefficients.

* easy: a quadratic ``ax² + bx + c = 0`` with roots α and β; one of α² + β², 1/α + 1/β, (α - β)² or α²β + αβ².
* medium: a cubic with roots α, β and γ; one of α² + β² + γ², 1/α + 1/β + 1/γ or (α + β)(β + γ)(γ + α), or an unknown
  coefficient found from a stated value of one of those.
* hard: the monic polynomial with integer coefficients whose roots are a stated transformation of the roots of a
  given quadratic or cubic (2α + 1, α², 1/α, α - 3, αβ-style products), or the common difference or ratio of a cubic
  whose roots are known to be in arithmetic or geometric progression.
* exam (``r3i_exam_poly_roots``): a cubic with roots α, β, γ; (a) α + β + γ, (b) αβ + βγ + γα, (c) α² + β² + γ²
  (follows through from (a) and (b)), (d) α³ + β³ + γ³ (follows through from (a) and (b)).

Every value is re-derived by solving the polynomial numerically (mpmath, 30 digits) and evaluating the asked expression
at the actual roots.
"""

from __future__ import annotations

from fractions import Fraction

import mpmath as mp
import sympy

from ..core import attempts, check, generator, phrase
from ._r3i_kit import (
    expr_dist,
    expr_fields,
    expr_num_cands,
    expr_num_fields,
    num_cands,
    num_fields,
    pick,
    pick_text,
    retry_short,
)

X = sympy.Symbol("x")


def _poly_text(coeffs, var="x") -> str:
    """coeffs from the highest power down: [2, -7, 3] -> 2x² - 7x + 3."""
    n = len(coeffs) - 1
    out = ""
    for i, c in enumerate(coeffs):
        c = Fraction(c)
        power = n - i
        if c == 0:
            continue
        mag = abs(c)
        if power == 0:
            body = str(mag.numerator) if mag.denominator == 1 else f"{mag.numerator}/{mag.denominator}"
        else:
            lead = (
                ""
                if mag == 1
                else (str(mag.numerator) if mag.denominator == 1 else f"({mag.numerator}/{mag.denominator})")
            )
            body = lead + var + ("" if power == 1 else {2: "²", 3: "³"}[power])
        sign = "-" if c < 0 else "+"
        out += (f"-{body}" if c < 0 else body) if not out else f" {sign} {body}"
    return out or "0"


def _poly_ascii(coeffs) -> str:
    """The same polynomial as typed: x^2 - 7x + 3."""
    return _poly_text(coeffs).replace("²", "^2").replace("³", "^3")


def _numeric_roots(coeffs):
    """The roots of the polynomial to 30 digits (mpmath), as complex numbers."""
    with mp.workdps(30):
        return [
            complex(r)
            for r in mp.polyroots(
                [mp.mpf(Fraction(c).numerator) / Fraction(c).denominator for c in coeffs], maxsteps=200, extraprec=60
            )
        ]


def _close(value, target) -> bool:
    return abs(complex(value) - complex(target)) < 1e-9 * max(1.0, abs(complex(target)))


def _draw_quadratic(rng, leading=(1, 2, 3, 4)):
    """Coefficients of a quadratic whose roots are real and irrational."""
    for _ in attempts(500, "a quadratic with irrational or complex roots"):
        a = rng.choice(leading)
        b, c = rng.randint(-9, 9), rng.choice([x for x in range(-9, 10) if x != 0])
        disc = b * b - 4 * a * c
        if b == 0:
            continue
        if disc <= 0 or int(disc**0.5) ** 2 == disc:
            continue
        return a, b, c
    raise AssertionError  # pragma: no cover


def _cubic_discriminant(a, b, c, d):
    return 18 * a * b * c * d - 4 * b**3 * d + b * b * c * c - 4 * a * c**3 - 27 * a * a * d * d


def _draw_cubic(rng):
    for _ in attempts(500, "a monic cubic with no rational root"):
        b, c = rng.randint(-8, 8), rng.choice([x for x in range(-8, 9) if x != 0])
        d = rng.choice([x for x in range(-9, 10) if x != 0])
        if all((r**3 + b * r * r + c * r + d) != 0 for r in range(-9, 10)) and _cubic_discriminant(1, b, c, d) != 0:
            return b, c, d
    raise AssertionError  # pragma: no cover


# -- easy ---------------------------------------------------------------------------------------------------


def _easy(rng):
    a, b, c = _draw_quadratic(rng)
    e1, e2 = Fraction(-b, a), Fraction(c, a)
    r1, r2 = _numeric_roots([a, b, c])
    kind = rng.choice(["squares", "reciprocals", "difference", "product_sum"])
    if kind == "squares":
        ans = e1**2 - 2 * e2
        check(_close(r1**2 + r2**2, float(ans)), "α² + β² differs from the numerical roots")
        ask = "α² + β²"
        cands = [
            (e1**2, "r3i-roots-square-of-sum-only"),
            (e1**2 + 2 * e2, "r3i-roots-cross-term-sign"),
            (e1 * e1 - e2, "r3i-roots-cross-term-not-doubled"),
            (Fraction(b * b - 2 * c, a), "r3i-roots-forgot-to-divide-by-leading-coefficient"),
        ]
        working = ["α + β = -b/a and αβ = c/a, and α² + β² = (α + β)² - 2αβ.", f"= ({e1})² - 2({e2}) = {ans}."]
    elif kind == "reciprocals":
        ans = e1 / e2
        check(_close(1 / r1 + 1 / r2, float(ans)), "1/α + 1/β differs from the numerical roots")
        ask = "1/α + 1/β"
        cands = [
            (e2 / e1, "r3i-roots-reciprocal-sum-inverted"),
            (-e1 / e2, "r3i-roots-sign-of-sum"),
            (1 / e1 + 1 / e2, "r3i-roots-reciprocals-taken-termwise"),
            (Fraction(-b, c), "r3i-roots-forgot-to-divide-by-leading-coefficient"),
        ]
        working = ["1/α + 1/β = (α + β)/(αβ).", f"= ({e1})/({e2}) = {ans}."]
    elif kind == "difference":
        ans = e1**2 - 4 * e2
        check(_close((r1 - r2) ** 2, float(ans)), "(α - β)² differs from the numerical roots")
        ask = "(α - β)²"
        cands = [
            (e1**2 - 2 * e2, "r3i-roots-cross-term-not-doubled"),
            (e1**2 + 4 * e2, "r3i-roots-cross-term-sign"),
            (e1 - 2 * e2, "r3i-roots-forgot-to-square-sum"),
            (Fraction(b * b - 4 * c, a), "r3i-roots-forgot-to-divide-by-leading-coefficient"),
        ]
        working = ["(α - β)² = (α + β)² - 4αβ.", f"= ({e1})² - 4({e2}) = {ans}."]
    else:
        ans = e1 * e2
        check(_close(r1 * r1 * r2 + r1 * r2 * r2, float(ans)), "α²β + αβ² differs from the numerical roots")
        ask = "α²β + αβ²"
        cands = [
            (e1 + e2, "r3i-roots-product-taken-as-sum"),
            (e1**2 * e2, "r3i-roots-factorised-incorrectly"),
            (-e1 * e2, "r3i-roots-sign-of-sum"),
            (e1 * e2 * a, "r3i-roots-forgot-to-divide-by-leading-coefficient"),
        ]
        working = ["α²β + αβ² = αβ(α + β).", f"= ({e2})({e1}) = {ans}."]
    fields = expr_num_fields(ans)
    stem = phrase(
        rng,
        (
            "The roots of {p} = 0 are α and β. Find the exact value of {ask}.",
            "α and β are the roots of the equation {p} = 0. Find the exact value of {ask}.",
        ),
        p=_poly_text([a, b, c]),
        ask=ask,
    )
    return dict(text=stem, working=working, distractors=pick_text(fields, expr_num_cands(cands, ans)), **fields)


# -- medium -------------------------------------------------------------------------------------------------


def _medium_symmetric(rng):
    b, c, d = _draw_cubic(rng)
    e1, e2, e3 = Fraction(-b), Fraction(c), Fraction(-d)
    roots = _numeric_roots([1, b, c, d])
    r1, r2, r3 = roots
    kind = rng.choice(["squares", "reciprocals", "pair_sums"])
    if kind == "squares":
        ans = e1**2 - 2 * e2
        check(_close(r1**2 + r2**2 + r3**2, float(ans)), "Σα² differs from the numerical roots")
        ask = "α² + β² + γ²"
        cands = [
            (e1**2 + 2 * e2, "r3i-roots-cross-term-sign"),
            (e1**2 - e2, "r3i-roots-cross-term-not-doubled"),
            (e1**2, "r3i-roots-square-of-sum-only"),
            (e1**2 - 2 * e3, "r3i-roots-wrong-symmetric-function-used"),
        ]
        working = [
            "α + β + γ = -b, αβ + βγ + γα = c, αβγ = -d.",
            f"α² + β² + γ² = (α + β + γ)² - 2(αβ + βγ + γα) = {ans}.",
        ]
    elif kind == "reciprocals":
        ans = e2 / e3
        check(_close(1 / r1 + 1 / r2 + 1 / r3, float(ans)), "Σ1/α differs from the numerical roots")
        ask = "1/α + 1/β + 1/γ"
        cands = [
            (e3 / e2, "r3i-roots-reciprocal-sum-inverted"),
            (-e2 / e3, "r3i-roots-sign-of-sum"),
            (e2 / (-e3), "r3i-roots-sign-of-sum"),
            (e1 / e3, "r3i-roots-wrong-symmetric-function-used"),
        ]
        working = ["1/α + 1/β + 1/γ = (αβ + βγ + γα)/(αβγ).", f"= {e2}/{e3} = {ans}."]
    else:
        ans = e1 * e2 - e3
        check(_close((r1 + r2) * (r2 + r3) * (r3 + r1), float(ans)), "(α + β)(β + γ)(γ + α) differs from the roots")
        ask = "(α + β)(β + γ)(γ + α)"
        cands = [
            (e1 * e2 + e3, "r3i-roots-expansion-sign"),
            (e1 * e2, "r3i-roots-expansion-term-missed"),
            (e1 * e2 - 2 * e3, "r3i-roots-expansion-term-miscounted"),
            (e1 * e3 - e2, "r3i-roots-wrong-symmetric-function-used"),
        ]
        working = [
            "Each pair sum is e₁ minus the third root, so the product is (e₁ - γ)(e₁ - α)(e₁ - β).",
            f"That equals e₁e₂ - e₃ = ({e1})({e2}) - ({e3}) = {ans}.",
        ]
    fields = expr_num_fields(ans)
    stem = f"α, β and γ are the roots of {_poly_text([1, b, c, d])} = 0. Find the exact value of {ask}."
    return dict(text=stem, working=working, distractors=pick_text(fields, expr_num_cands(cands, ans)), **fields)


def _medium_unknown(rng):
    for _ in attempts(500, "a cubic with an unknown coefficient"):
        e1 = rng.randint(-7, 8)
        e3 = rng.choice([x for x in range(-12, 13) if x != 0])
        k = rng.randint(-9, 12)
        if k != 0:
            break
    # x³ - e1 x² + k x - e3: roots with sum e1, pairwise sum k, product e3; Σα² = e1² - 2k
    squares = e1 * e1 - 2 * k
    quad = f" {'-' if e1 > 0 else '+'} {abs(e1)}x²" if e1 != 0 else ""
    poly = f"x³{quad} + kx {'-' if e3 > 0 else '+'} {abs(e3)}"
    ans = k
    check(squares == e1 * e1 - 2 * ans, "k differs from the sum of squares")
    cands = [
        (Fraction(e1 * e1 - squares, 1), "r3i-roots-cross-term-not-doubled"),
        ((squares - e1 * e1) // 2 if (squares - e1 * e1) % 2 == 0 else None, "r3i-roots-sign-of-sum"),
        (e1 * e1 + squares, "r3i-roots-cross-term-sign"),
        (Fraction(e1 * e1 - squares, 4), "r3i-roots-divided-by-four"),
    ]
    fields = expr_num_fields(ans)
    return dict(
        text=(f"The roots of {poly} = 0 are α, β and γ, and α² + β² + γ² = {squares}. Find the value of k."),
        working=[
            f"α + β + γ = {e1}, and α² + β² + γ² = (α + β + γ)² - 2(αβ + βγ + γα).",
            f"{squares} = {e1 * e1} - 2k, so k = {k}.",
        ],
        distractors=pick_text(fields, expr_num_cands(cands, ans)),
        **fields,
    )


def _medium(rng):
    return _medium_symmetric(rng) if rng.random() < 0.65 else _medium_unknown(rng)


# -- hard ---------------------------------------------------------------------------------------------------


def _expr_answer(coeffs):
    """A monic integer polynomial as an expression answer (expanded)."""
    text = _poly_ascii(coeffs)
    return dict(expr_fields(None, text), answer_form="expanded")


def _hard_transform_quadratic(rng):
    for _ in attempts(500, "a monic quadratic and an integer-coefficient transformation"):
        b, c = rng.randint(-9, 9), rng.choice([x for x in range(-6, 7) if x != 0])
        if b * b - 4 * c <= 0 or int((b * b - 4 * c) ** 0.5) ** 2 == b * b - 4 * c:
            continue
        e1, e2 = -b, c
        kind = rng.choice(["linear", "shift", "squares", "reciprocal"])
        if kind == "linear":
            m, s = rng.choice([2, 3, -2, 3, 4]), rng.choice([x for x in range(-4, 5) if x != 0])
            new_sum, new_prod = m * e1 + 2 * s, m * m * e2 + m * s * e1 + s * s
            what = f"{'' if m == 1 else m}α {'+' if s > 0 else '-'} {abs(s)}"
            what2 = what.replace("α", "β")
            fn = lambda r: m * r + s  # noqa: E731
        elif kind == "shift":
            s = rng.choice([x for x in range(-4, 5) if x != 0])
            new_sum, new_prod = e1 + 2 * s, e2 + s * e1 + s * s
            what, what2 = f"α {'+' if s > 0 else '-'} {abs(s)}", f"β {'+' if s > 0 else '-'} {abs(s)}"
            fn = lambda r: r + s  # noqa: E731
        elif kind == "squares":
            new_sum, new_prod = e1 * e1 - 2 * e2, e2 * e2
            what, what2 = "α²", "β²"
            fn = lambda r: r * r  # noqa: E731
        else:
            if abs(e2) != 1:
                continue
            new_sum, new_prod = Fraction(e1, e2), Fraction(1, e2)
            what, what2 = "1/α", "1/β"
            fn = lambda r: 1 / r  # noqa: E731
        if Fraction(new_sum).denominator != 1 or Fraction(new_prod).denominator != 1:
            continue
        break
    new_sum, new_prod = int(new_sum), int(new_prod)
    r1, r2 = _numeric_roots([1, b, c])
    t1, t2 = fn(r1), fn(r2)
    check(_close(t1 + t2, new_sum) and _close(t1 * t2, new_prod), "sum or product of the new roots differs")
    key = [1, -new_sum, new_prod]
    fields = _expr_answer(key)
    fields_dists = [
        ([1, new_sum, new_prod], "r3i-roots-new-polynomial-sign-of-sum"),
        ([1, -new_prod, new_sum], "r3i-roots-new-polynomial-sum-product-swapped"),
    ]
    wrong_transform = None
    if kind in ("linear", "shift"):
        m_, s_ = (m, s) if kind == "linear" else (1, s)
        # substituting the transformation into the polynomial instead of its inverse
        sub = sympy.expand((m_ * X + s_) ** 2 + b * (m_ * X + s_) + c)
        wrong_transform = [int(v) for v in sympy.Poly(sub, X).all_coeffs()]
    elif kind == "squares":
        wrong_transform = [1, -(e1**2), e2**2]
    else:
        wrong_transform = [1, -e1, e2]
    cands = []
    for coeffs, tag in fields_dists:
        cands.append(expr_dist(_poly_ascii(coeffs), tag))
    cands.append(expr_dist(_poly_ascii(wrong_transform), "r3i-roots-transformation-applied-to-coefficients"))
    cands.append(expr_dist(_poly_ascii([1, -new_sum, e2]), "r3i-roots-new-product-not-recomputed"))
    return dict(
        text=(
            f"α and β are the roots of {_poly_text([1, b, c])} = 0. Find the quadratic with integer coefficients and a leading "
            f"coefficient of 1 whose roots are {what} and {what2}."
        ),
        working=[
            f"α + β = {e1} and αβ = {e2}.",
            f"The new roots have sum {new_sum} and product {new_prod}, so the quadratic is x² - ({new_sum})x + ({new_prod}).",
        ],
        distractors=pick_text(fields, cands),
        **fields,
    )


def _hard_progression(rng):
    kind = rng.choice(["arithmetic", "geometric"])
    if kind == "arithmetic":
        for _ in attempts(500, "three roots in arithmetic progression"):
            m, d = rng.randint(-4, 6), rng.randint(1, 5)
            if m != 0 and m - d != 0 and m + d != 0:
                break
        roots = (m - d, m, m + d)
        ans = d
        stem_tail = (
            "The three roots are in arithmetic progression. Find the common difference, given that it is positive."
        )
        cands = [
            (m, "r3i-roots-middle-root-given"),
            (d * d, "r3i-roots-difference-squared-given"),
            (2 * d, "r3i-roots-common-difference-doubled"),
            (m - d, "r3i-roots-smallest-root-given"),
        ]
    else:
        for _ in attempts(500, "three roots in geometric progression"):
            m, r = rng.randint(2, 6), rng.choice([2, 3])
            if m % r == 0:
                break
        roots = (m // r, m, m * r)
        ans = r
        stem_tail = (
            "The three roots are in geometric progression. Find the common ratio, given that it is greater than 1."
        )
        cands = [
            (m, "r3i-roots-middle-root-given"),
            (r * r, "r3i-roots-ratio-squared-given"),
            (m * r, "r3i-roots-largest-root-given"),
            (Fraction(1, r), "r3i-roots-ratio-inverted"),
        ]
    p, q, s = roots
    e1, e2, e3 = p + q + s, p * q + q * s + s * p, p * q * s
    coeffs = [1, -e1, e2, -e3]
    check(sorted(sympy.roots(sympy.Poly(coeffs, X)).keys()) == sorted(roots), "roots differ")
    if kind == "arithmetic":
        check(Fraction(e1, 3) == m, "the middle root is not e1/3")
    else:
        check(e3 == m**3, "the middle root cubed is not the product")
    fields = expr_num_fields(ans)
    return dict(
        text=f"The cubic {_poly_text(coeffs)} = 0 has three real roots. {stem_tail}",
        working=(
            [
                f"Write the roots as m - d, m and m + d. Their sum is 3m = {e1}, so m = {m}.",
                f"The product m(m² - d²) = {e3} gives d² = {d * d if kind == 'arithmetic' else 0}, so d = {d}.",
            ]
            if kind == "arithmetic"
            else [
                f"Write the roots as m/r, m and mr. Their product is m³ = {e3}, so m = {m}.",
                f"The sum m(1/r + 1 + r) = {e1} gives r = {r}.",
            ]
        ),
        distractors=pick_text(fields, expr_num_cands(cands, ans)),
        **fields,
    )


def _hard(rng):
    return _hard_transform_quadratic(rng) if rng.random() < 0.6 else _hard_progression(rng)


@generator(
    "r3i_poly_roots",
    level=14,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def r3i_poly_roots(rng, tier="easy"):
    """Relationships among the roots of a polynomial; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam ---------------------------------------------------------------------------------------------------


@generator("r3i_exam_poly_roots", level=14, presets={"exam": dict(tier="exam")})
@retry_short
def r3i_exam_poly_roots(rng, tier="exam"):
    """Exam chain: Σα, Σαβ, Σα² and Σα³ for the roots of a cubic."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "a cubic with three distinct roots"):
        a = rng.choice([1, 1, 2])
        b, c = rng.randint(-8, 8), rng.randint(-8, 8)
        d = rng.choice([x for x in range(-9, 10) if x != 0])
        if _cubic_discriminant(a, b, c, d) != 0:
            break
    e1, e2, e3 = Fraction(-b, a), Fraction(c, a), Fraction(-d, a)
    roots = _numeric_roots([a, b, c, d])
    r1, r2, r3 = roots
    p2 = e1**2 - 2 * e2
    p3 = e1**3 - 3 * e1 * e2 + 3 * e3
    check(_close(r1 + r2 + r3, float(e1)) and _close(r1 * r2 + r2 * r3 + r3 * r1, float(e2)), "e1 or e2 differ")
    check(_close(r1**2 + r2**2 + r3**2, float(p2)), "Σα² differs")
    check(_close(r1**3 + r2**3 + r3**3, float(p3)), "Σα³ differs")
    f_a, f_b, f_c, f_d = num_fields(e1), num_fields(e2), num_fields(p2), num_fields(p3)
    e3_text = str(e3) if e3.denominator == 1 else f"({e3})"
    return dict(
        text=f"α, β and γ are the roots of {_poly_text([a, b, c, d])} = 0.",
        parts=[
            dict(
                prompt="Write down the value of α + β + γ.",
                marks=1,
                sensible_range=(-100, 100),
                working=[f"α + β + γ = -b/a = {e1}."],
                distractors=pick(
                    f_a,
                    num_cands(
                        [
                            (-e1, "r3i-roots-sign-of-sum"),
                            (Fraction(b), "r3i-roots-forgot-to-divide-by-leading-coefficient"),
                            (e2, "r3i-roots-wrong-symmetric-function-used"),
                        ],
                        e1,
                    ),
                ),
                **f_a,
            ),
            dict(
                prompt="Write down the value of αβ + βγ + γα.",
                marks=1,
                sensible_range=(-100, 100),
                working=[f"αβ + βγ + γα = c/a = {e2}."],
                distractors=pick(
                    f_b,
                    num_cands(
                        [
                            (-e2, "r3i-roots-sign-of-sum"),
                            (Fraction(c), "r3i-roots-forgot-to-divide-by-leading-coefficient"),
                            (e1, "r3i-roots-wrong-symmetric-function-used"),
                        ],
                        e2,
                    ),
                ),
                **f_b,
            ),
            dict(
                prompt="Hence find the value of α² + β² + γ².",
                marks=2,
                follow_through="part_a**2-2*part_b",
                sensible_range=(-100, 400),
                working=[f"(α + β + γ)² - 2(αβ + βγ + γα) = ({e1})² - 2({e2}) = {p2}."],
                distractors=pick(
                    f_c,
                    num_cands(
                        [
                            (e1**2 + 2 * e2, "r3i-roots-cross-term-sign"),
                            (e1**2 - e2, "r3i-roots-cross-term-not-doubled"),
                            (e1**2, "r3i-roots-square-of-sum-only"),
                        ],
                        p2,
                    ),
                ),
                **f_c,
            ),
            dict(
                prompt="Find the value of α³ + β³ + γ³.",
                marks=3,
                follow_through=f"part_a**3-3*part_a*part_b+3*({e3})",
                sensible_range=(-1000, 1000),
                working=[
                    "Use α³ + β³ + γ³ = e₁³ - 3e₁e₂ + 3e₃ with e₃ = αβγ = -d/a.",
                    f"= ({e1})³ - 3({e1})({e2}) + 3({e3}) = {p3}.",
                ],
                distractors=pick(
                    f_d,
                    num_cands(
                        [
                            (e1**3 - 3 * e1 * e2 - 3 * e3, "r3i-roots-cube-sum-product-sign"),
                            (e1**3 - 3 * e1 * e2, "r3i-roots-cube-sum-product-missed"),
                            (e1**3 + 3 * e1 * e2 + 3 * e3, "r3i-roots-cube-sum-cross-sign"),
                            (e1**3, "r3i-roots-cube-of-sum-only"),
                        ],
                        p3,
                    ),
                ),
                **f_d,
            ),
        ],
    )
