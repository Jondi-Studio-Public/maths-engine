"""Generates `r3b_partial_fractions` questions (easy, medium, hard) and the exam chain `r3b_exam_partial_fractions`.

Skill ``partial-fractions-repeated-irreducible`` (``integration``, answer ``pair``): partial fractions whose denominators
have a repeated linear factor or an irreducible quadratic factor, and the integrals they lead to. The answers are
labelled constants, typed as ``A = 2, B = -1``.

* easy: ``(px + q)/(x - a)^2 = A/(x - a) + B/(x - a)^2``.
* medium: ``(numerator)/((x - a)(x^2 + b)) = A/(x - a) + (Bx + C)/(x^2 + b)`` with ``b`` positive.
* hard: either ``(numerator)/((x - a)^2 (x^2 + b)) = A/(x - a) + B/(x - a)^2 + (Cx + D)/(x^2 + b)`` (four constants), or
  the coefficients in the integral ``P ln|x - a| + Q ln(x^2 + s^2) + R arctan(x/s)`` of an ``(x - a)(x^2 + s^2)`` fraction.
* exam (``r3b_exam_partial_fractions``): ``N/((x - a)(x^2 + s^2))`` with ``a`` negative: (a) A by substituting x = a, (b) B from
  the x^2 coefficient (follows from (a)), (c) C from the x coefficient (follows from (b)), (d) the exact value of the
  integral from 0 to s, which has a logarithm of a ratio, ``ln 2`` and a multiple of pi (follows from (a), (b), (c)).

Every decomposition is checked by adding the fractions back and comparing with the original at five points.
"""

from __future__ import annotations

from fractions import Fraction

from sympy import Abs, Poly, Rational, atan, expand, log, pi

from ..answers import Distractor
from ..core import attempts, check, generator, latex_from_inline
from ._r3b_kit import (
    Cand,
    Redraw,
    as_fraction,
    check_antiderivative,
    check_definite,
    exact_fields,
    frac_fields,
    part,
    pick_expr,
    pick_fractions,
    retry,
    same_function,
    say,
    shown,
    tex,
    x,
)

_FIND = ("{P} Find {Q}.", "{P} Determine {Q}.")


def _nonzero(rng, lo=-9, hi=9):
    return rng.choice([v for v in range(lo, hi + 1) if v])


def _lin(a) -> str:
    """x - a as the learner reads it."""
    return f"x {'-' if a > 0 else '+'} {abs(a)}"


# -- labelled constants -----------------------------------------------------------------------------------------------


def _val_text(v: Fraction) -> str:
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def _val_tex(v: Fraction) -> str:
    if v.denominator == 1:
        return str(v.numerator)
    return rf"{'-' if v < 0 else ''}\frac{{{abs(v.numerator)}}}{{{v.denominator}}}"


def _val_json(v: Fraction):
    return int(v) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


class Labelled:
    """Labelled constants such as A = 2, B = -1/2, for the `pair` answer kind."""

    def __init__(self, labels, values):
        self.labels = tuple(labels)
        self.values = tuple(as_fraction(v) for v in values)

    @property
    def answer(self):
        return {lab: _val_json(v) for lab, v in zip(self.labels, self.values)}

    @property
    def text(self):
        return ", ".join(f"{lab} = {_val_text(v)}" for lab, v in zip(self.labels, self.values))

    def fields(self):
        latex = r",\ ".join(f"{lab} = {_val_tex(v)}" for lab, v in zip(self.labels, self.values))
        return dict(answer_kind="pair", answer=self.answer, answer_text=self.text, answer_latex=latex)

    def distractor(self, tag):
        return Distractor(self.text, tag, self.answer)


def _pick_labelled(key: Labelled, wrong, rng, count=3):
    """Different wrong sets of constants from `(values, tag)` pairs; a set within the grader's 0.06 tolerance of the key in
    every component, or repeating another, is dropped."""
    chosen: list[Labelled] = []
    tags: list[str] = []
    for values, tag in rng.sample(list(wrong), len(list(wrong))):
        cand = Labelled(key.labels, values)
        if all(abs(a - b) < Fraction(6, 100) for a, b in zip(cand.values, key.values)):
            continue
        if any(cand.values == other.values for other in chosen):
            continue
        chosen.append(cand)
        tags.append(tag)
        if len(chosen) == count:
            break
    if len(chosen) < 2:
        raise Redraw("fewer than two distinct wrong sets of constants")
    return [c.distractor(t) for c, t in zip(chosen, tags)]


def _ratio(num, den_text: str, den_tex: str):
    """(plain text, LaTeX) of a fraction whose numerator is a polynomial in x; the LaTeX is wrapped for a stem."""
    return f"({shown(num)})/{den_text}", rf"\(\frac{{{tex(num)}}}{{{den_tex}}}\)"


def _numerator(*pairs):
    """Sum of coefficient times factor, expanded."""
    return expand(sum(c * f for c, f in pairs))


def _quad(b):
    return f"x^2 + {b}"


def _build(rng, patterns, slots, plain, key: Labelled, wrong, working):
    """`slots` maps each stem slot to its (plain text, LaTeX) pair."""
    text, latex = say(rng, patterns, {k: v[0] for k, v in slots.items()}, {k: v[1] for k, v in slots.items()}, plain)
    return dict(text=text, latex=latex, working=working, distractors=_pick_labelled(key, wrong, rng), **key.fields())


_WRITE = (
    "Write {F} in the form {G}, where {K} are constants. Find {K}.",
    "Express {F} in the form {G}, where {K} are constants. Find the values of {K}.",
)


def _tex_term(c) -> str:
    return tex(c)


# -- easy ---------------------------------------------------------------------------------------------------------


def _easy(rng):
    a = _nonzero(rng, -5, 5)
    A, B = _nonzero(rng, -6, 6), _nonzero(rng, -9, 9)
    num = _numerator((A, x - a), (B, 1))
    den = (x - a) ** 2
    check(same_function(num / den, A / (x - a) + B / den), "the decomposition is wrong")
    lin = _lin(a)
    key = Labelled("AB", [A, B])
    wrong = [
        ([A, num.subs(x, 0)], "r3b-pf-repeated-constant-unshifted"),
        ([B, A], "r3b-pf-constants-swapped"),
        ([A, B - 2 * a * A], "r3b-pf-repeated-shift-sign"),
        ([-A, B], "r3b-pf-numerator-sign"),
    ]
    working = [
        f"Multiply through by ({lin})^2: {shown(num)} = A({lin}) + B.",
        f"Comparing the x terms gives A = {A}. At x = {a}, the left side is {num.subs(x, a)}, so B = {B}.",
    ]
    slots = dict(
        F=_ratio(num, f"({lin})^2", f"({lin})^2"),
        G=(f"A/({lin}) + B/({lin})^2", rf"\(\frac{{A}}{{{lin}}} + \frac{{B}}{{({lin})^2}}\)"),
    )
    return _build(rng, _WRITE, slots, dict(K="A and B"), key, wrong, working)


# -- medium -------------------------------------------------------------------------------------------------------


def _draw_abc(rng):
    """Constants A, B, C (all nonzero), the shift a, and b of x^2 + b; numerator coefficients n2, n1, n0."""
    a = _nonzero(rng, -4, 4)
    b = rng.randint(1, 6)
    A, B, C = (_nonzero(rng, -5, 5) for _ in range(3))
    num = _numerator((A, x**2 + b), (B * x + C, x - a))
    return a, b, A, B, C, num


def _medium(rng):
    a, b, A, B, C, num = _draw_abc(rng)
    den = (x - a) * (x**2 + b)
    check(same_function(num / den, A / (x - a) + (B * x + C) / (x**2 + b)), "the decomposition is wrong")
    p = Poly(num, x)
    n2, n1, n0 = p.coeff_monomial(x**2), p.coeff_monomial(x), p.coeff_monomial(1)
    check((n2, n1, n0) == (A + B, C - a * B, A * b - a * C), "the coefficient equations are wrong")
    lin, quad = _lin(a), _quad(b)
    key = Labelled("ABC", [A, B, C])
    a_wrong = Rational(num.subs(x, -a), a * a + b)
    b_wrong = n2 - a_wrong
    wrong = [
        ([a_wrong, b_wrong, n1 + a * b_wrong], "r3b-pf-evaluated-at-wrong-sign"),
        ([A, n2, C], "r3b-pf-b-not-reduced"),
        ([A, C, B], "r3b-pf-b-c-swapped"),
        ([A, B, -C], "r3b-pf-c-sign"),
    ]
    working = [
        f"Multiply through by ({lin})({quad}): {shown(num)} = A({quad}) + (Bx + C)({lin}).",
        f"At x = {a}: {num.subs(x, a)} = {a * a + b}A, so A = {A}.",
        f"The x^2 terms give {n2} = A + B, so B = {B}. The x terms give {n1} = C {'-' if a > 0 else '+'} {'' if abs(a) == 1 else abs(a)}B, so C = {C}.",
    ]
    slots = dict(
        F=_ratio(num, f"(({lin})({quad}))", rf"({lin})({quad})"),
        G=(
            f"A/({lin}) + (Bx + C)/({quad})",
            rf"\(\frac{{A}}{{{lin}}} + \frac{{Bx + C}}{{{quad}}}\)",
        ),
    )
    return _build(rng, _WRITE, slots, dict(K="A, B and C"), key, wrong, working)


# -- hard ---------------------------------------------------------------------------------------------------------


def _hard_four(rng):
    """(x - a)^2 (x^2 + b): four constants."""
    a = _nonzero(rng, -3, 3)
    b = rng.randint(1, 5)
    A, B, C, D = (_nonzero(rng, -4, 4) for _ in range(4))
    num = _numerator((A, (x - a) * (x**2 + b)), (B, x**2 + b), (C * x + D, (x - a) ** 2))
    den = (x - a) ** 2 * (x**2 + b)
    check(
        same_function(num / den, A / (x - a) + B / (x - a) ** 2 + (C * x + D) / (x**2 + b)),
        "the decomposition is wrong",
    )
    check(B == Rational(num.subs(x, a), a * a + b), "B does not follow from substituting x = a")
    lin, quad = _lin(a), _quad(b)
    key = Labelled("ABCD", [A, B, C, D])
    wrong = [
        ([B, A, C, D], "r3b-pf-a-b-swapped"),
        ([A, B, D, C], "r3b-pf-c-d-swapped"),
        ([A, B, -C, -D], "r3b-pf-c-sign"),
        ([A, num.subs(x, a), C, D], "r3b-pf-substitution-not-divided"),
    ]
    working = [
        f"Multiply through by ({lin})^2({quad}): {shown(num)} = A({lin})({quad}) + B({quad}) + (Cx + D)({lin})^2.",
        f"At x = {a}: {num.subs(x, a)} = {a * a + b}B, so B = {B}.",
        f"Compare the coefficients of x^3, x^2 and 1 to find A = {A}, C = {C}, D = {D}.",
    ]
    slots = dict(
        F=_ratio(num, f"(({lin})^2({quad}))", rf"({lin})^2({quad})"),
        G=(
            f"A/({lin}) + B/({lin})^2 + (Cx + D)/({quad})",
            rf"\(\frac{{A}}{{{lin}}} + \frac{{B}}{{({lin})^2}} + \frac{{Cx + D}}{{{quad}}}\)",
        ),
    )
    return _build(rng, _WRITE, slots, dict(K="A, B, C and D"), key, wrong, working)


_INTEGRAL = (
    "Given that {I} = P ln|{L}| + Q ln({Q}) + R arctan({T}) + c, find {K}.",
    "It is known that {I} = P ln|{L}| + Q ln({Q}) + R arctan({T}) + c, where c is a constant. Find {K}.",
)


def _hard_integral(rng):
    """The integral of N/((x - a)(x^2 + s^2)) as P ln|x - a| + Q ln(x^2 + s^2) + R arctan(x/s)."""
    a = _nonzero(rng, -4, 4)
    s = rng.choice([2, 3, 4])
    b = s * s
    A, B, C = (_nonzero(rng, -6, 6) for _ in range(3))
    num = _numerator((A, x**2 + b), (B * x + C, x - a))
    den = (x - a) * (x**2 + b)
    P, Q, R = Rational(A), Rational(B, 2), Rational(C, s)
    F = P * log(Abs(x - a)) + Q * log(x**2 + b) + R * atan(x / s)
    check_antiderivative(F, num / den)
    lin, quad = _lin(a), _quad(b)
    key = Labelled("PQR", [P, Q, R])
    wrong = [
        ([P, B, R], "r3b-pf-log-not-halved"),
        ([P, Q, C], "r3b-pf-arctan-not-divided"),
        ([P, Q, Rational(C * s)], "r3b-pf-arctan-multiplied"),
        ([P, B, C], "r3b-pf-quadratic-term-direct"),
        ([-P, Q, R], "r3b-pf-numerator-sign"),
    ]
    working = [
        f"Decompose: {shown(num)} over ({lin})({quad}) = A/({lin}) + (Bx + C)/({quad}) with A = {A}, B = {B}, C = {C}.",
        f"Integrate: A ln|{lin}| + (B/2) ln({quad}) + {'C arctan(x)' if s == 1 else f'(C/{s}) arctan(x/{s})'}, so P = {P}, Q = {Q}, R = {R}.",
    ]
    integrand = _ratio(num, f"(({lin})({quad}))", rf"({lin})({quad})")
    text, latex = say(
        rng,
        _INTEGRAL,
        dict(I=f"∫ {integrand[0]} dx", L=lin, Q=quad, T=f"x/{s}", K="P, Q and R"),
        dict(
            I=rf"\(\int \frac{{{tex(num)}}}{{({lin})({quad})}}\,\mathrm{{d}}x\)",
            L=lin,
            Q=quad,
            T=f"x/{s}",
            K="P, Q and R",
        ),
    )
    return dict(text=text, latex=latex, working=working, distractors=_pick_labelled(key, wrong, rng), **key.fields())


@generator(
    "r3b_partial_fractions",
    level=14,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry
def r3b_partial_fractions(rng, tier="easy"):
    """Partial fractions with a repeated or irreducible quadratic factor; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_four, _hard_integral])(rng)


# -- exam ---------------------------------------------------------------------------------------------------------


@generator("r3b_exam_partial_fractions", level=14, presets={"exam": dict(tier="exam")})
@retry
def r3b_exam_partial_fractions(rng, tier="exam"):
    """Exam chain: A by substitution, B and C from coefficients, then the exact value of the integral over [0, s]."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "exam constants"):
        a = -rng.randint(1, 4)
        s = rng.choice([1, 2, 3])
        b = s * s
        A, B, C = (_nonzero(rng, -5, 5) for _ in range(3))
        if abs(a) != s or s == 1:
            break
    num = _numerator((A, x**2 + b), (B * x + C, x - a))
    den = (x - a) * (x**2 + b)
    p = Poly(num, x)
    n2, n1 = p.coeff_monomial(x**2), p.coeff_monomial(x)
    check(A == Rational(num.subs(x, a), a * a + b) and B == n2 - A and C == n1 + a * B, "the constants are wrong")
    value = A * log(Rational(s - a, -a)) + Rational(B, 2) * log(2) + Rational(C, 4 * s) * pi
    check_definite(value, num / den, 0, s)
    lin, quad = _lin(a), _quad(b)
    stem = (
        f"The expression ({shown(num)})/(({lin})({quad})) can be written as "
        f"A/({lin}) + (Bx + C)/({quad}), where A, B and C are constants."
    )
    stem_latex = (
        rf"The expression \(\frac{{{tex(num)}}}{{({lin})({quad})}}\) can be written as "
        rf"\(\frac{{A}}{{{lin}}} + \frac{{Bx + C}}{{{quad}}}\), where \(A\), \(B\) and \(C\) are constants."
    )
    n_a = num.subs(x, a)
    pa = part(
        "Find the value of A.",
        "Find the value of \\(A\\).",
        1,
        [f"Multiply through by ({lin})({quad}) and put x = {a}: {n_a} = {a * a + b}A."],
        frac_fields(A),
        pick_fractions(
            A,
            [
                (Rational(num.subs(x, -a), a * a + b), "r3b-pf-evaluated-at-wrong-sign"),
                (n_a, "r3b-pf-substitution-not-divided"),
                (Rational(n_a, a * a), "r3b-pf-quadratic-factor-forgotten"),
                (-A, "r3b-pf-numerator-sign"),
            ],
            rng=rng,
        ),
    )
    pb = part(
        "Find the value of B.",
        "Find the value of \\(B\\).",
        1,
        [f"Comparing the x^2 terms: {n2} = A + B, so B = {n2} - ({A}) = {B}."],
        frac_fields(B),
        pick_fractions(
            B,
            [
                (n2, "r3b-pf-b-not-reduced"),
                (n2 + A, "r3b-pf-b-sign"),
                (-B, "r3b-pf-numerator-sign"),
                (A - n2, "r3b-pf-b-sign"),
            ],
            rng=rng,
        ),
        follow_through=f"({n2}) - part_a",
    )
    pc = part(
        "Find the value of C.",
        "Find the value of \\(C\\).",
        1,
        [f"Comparing the x terms: {n1} = C {'-' if a > 0 else '+'} {'' if abs(a) == 1 else abs(a)}B, so C = {C}."],
        frac_fields(C),
        pick_fractions(
            C,
            [
                (n1 - a * B, "r3b-pf-c-sign"),
                (n1, "r3b-pf-c-shift-dropped"),
                (-C, "r3b-pf-numerator-sign"),
                (n1 + a * A, "r3b-pf-c-from-wrong-constant"),
            ],
            rng=rng,
        ),
        follow_through=f"({n1}) + ({a})*part_b",
    )
    pd = part(
        f"Find the exact value of the integral of the expression from x = 0 to x = {s}.",
        rf"Find the exact value of \(\int_0^{{{s}}} \frac{{{tex(num)}}}{{({lin})({quad})}}\,\mathrm{{d}}x\).",
        4,
        [
            f"Integrate: A ln|{lin}| + (B/2) ln({quad}) + {'C arctan(x)' if s == 1 else f'(C/{s}) arctan(x/{s})'}.",
            f"Substitute the limits 0 and {s}: A ln({Rational(s - a, -a)}) + (B/2) ln 2 + C pi/{4 * s}.",
        ],
        exact_fields(value),
        pick_expr(
            value,
            False,
            [
                Cand(A * log(Rational(s - a, -a)) + Rational(B, 2) * log(2), "r3b-pf-arctan-term-dropped", const=False),
                Cand(
                    A * log(Rational(s - a, -a)) + B * log(2) + Rational(C, 4 * s) * pi,
                    "r3b-pf-log-not-halved",
                    const=False,
                ),
                Cand(
                    A * log(Rational(s - a, -a)) + Rational(B, 2) * log(2) + Rational(C, 4) * pi,
                    "r3b-pf-arctan-not-divided",
                    const=False,
                ),
                Cand(
                    A * log(s - a) + Rational(B, 2) * log(2) + Rational(C, 4 * s) * pi,
                    "r3b-pf-lower-limit-ignored",
                    const=False,
                ),
                Cand(
                    A * log(Rational(s - a, -a)) + Rational(B, 2) * log(2) + Rational(C, 2 * s) * pi,
                    "r3b-pf-arctan-limit-slip",
                    const=False,
                ),
            ],
            rng=rng,
        ),
        follow_through=(f"part_a*log({Rational(s - a, -a)}) + part_b*log(2)/2 + part_c*pi/{4 * s}"),
    )
    return dict(text=stem, latex=latex_from_inline(stem_latex), parts=[pa, pb, pc, pd])
