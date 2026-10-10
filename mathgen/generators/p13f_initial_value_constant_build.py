"""Generates `p13f_initial_value_constant` questions (`ucv1-initial-value-constant`): the constant in an ODE
solution fixed by an initial value.

Boundary against the prerequisite ``solve-ode-separable-poly`` (topic
``differential_equation``, ``kind == "separable_poly"``,
``mathgen/generators/calculus.py``): that skill derives the GENERAL
solution ``y = F(x) + C`` of ``dy/dx = p(x)`` from a stated polynomial
``p(x)`` -- the learner does the integration. This skill's kernel is
deliberately narrower and never overlaps: ``F(x)`` is **supplied
already-integrated** (a stated polynomial antiderivative, never derived
from an ODE here), plus one initial condition ``y(x0) = y0``. The ONLY
thing the learner computes is::

    C = y0 - F(x0)

``F`` is always a pure polynomial in ``x`` with no bare constant term (the
antiderivative of a polynomial never has one -- any x^0 term would be
integrated from a nonzero constant and so would itself carry an x^1
factor), and is never an exponential form -- `exponential ODE activation`
is an explicit exclusion, so `solve-ode-exponential`'s territory is never
touched either.

Difficulty progression:
- easy:   x0 = 0 -- every term of F vanishes trivially (0**k == 0 for every
          k >= 1), so C = y0 directly. Deliberately trivial substitution.
- medium: x0 a small positive integer, F has 2-3 terms including at least
          one exponent >= 2 -- genuine substitution and arithmetic.
- hard:   x0 negative, F built so a negative base raised to an EVEN
          exponent must be handled correctly (the sign trap) while an ODD
          exponent keeps the base's sign -- a sign-handling escalation, not
          bigger numbers.
- exam:   a VERBAL condition is translated into y(x0) = y0 first -- either
          "crosses the x-axis at x = x0" (y0 = 0) or "passes through the
          point where x = x0 and y = y0" (a coordinate pair rather than
          function notation).
"""

from __future__ import annotations

from fractions import Fraction

from sympy import Rational, symbols

from ..core import (
    check,
    check_distinct,
    check_pairwise_distinct,
    check_equivalent,
    generator,
    latex_answer,
    paren_negative,
    phrase,
    pick_distractors,
    rational_text,
)

_MAX_ATTEMPTS = 500
_x_sym = symbols("x")


def _term_piece(k: int, exp: int) -> tuple[bool, str]:
    """`(is_negative, magnitude_text)` for the term `Fraction(k, exp) * x**exp`
    -- the coefficient is exactly what antiderivative notation looks like:
    an integer power ``x^exp`` (or bare ``x``) whose own /exp denominator
    survives only when it doesn't cancel (e.g. k=6, exp=2 renders "3x^2",
    not "6x^2/2")."""
    frac = Fraction(k, exp)
    neg = frac < 0
    mag = abs(frac)
    if exp == 1:
        num = mag.numerator  # denominator is always 1 here
        coefpart = "" if num == 1 else str(num)
        return neg, f"{coefpart}x"
    var = f"x^{exp}"
    num = mag.numerator
    coefpart = "" if num == 1 else str(num)
    if mag.denominator == 1:
        return neg, f"{coefpart}{var}"
    return neg, f"{coefpart}{var}/{mag.denominator}"


def _poly_text(terms: list[tuple[int, int]]) -> str:
    parts = []
    for i, (k, exp) in enumerate(terms):
        neg, text = _term_piece(k, exp)
        if i == 0:
            parts.append(f"-{text}" if neg else text)
        else:
            parts.append(f"- {text}" if neg else f"+ {text}")
    return " ".join(parts)


def _eval_F(terms: list[tuple[int, int]], x0: int) -> Fraction:
    """Construction's own direct route: sum Fraction(k, exp) * x0**exp."""
    total = Fraction(0)
    for k, exp in terms:
        total += Fraction(k, exp) * (x0**exp)
    return total


def _independent_eval_F(terms: list[tuple[int, int]], x0: int) -> Fraction:
    """Independent route: build the SYMBOLIC polynomial with sympy and
    substitute -- a structurally different code path from the direct
    Fraction-power loop above."""
    expr = sum(Rational(k, exp) * _x_sym**exp for k, exp in terms)
    value = expr.subs(_x_sym, x0)
    r = Rational(value)
    return Fraction(int(r.p), int(r.q))


def _drop_term_F(terms: list[tuple[int, int]], drop_index: int, x0: int) -> Fraction:
    remaining = [t for i, t in enumerate(terms) if i != drop_index]
    return _eval_F(remaining, x0)


def _offer_candidate(answer: Fraction, value: Fraction, offered: dict, label: str) -> None:
    if value == answer:
        return
    if any(value == existing for existing in offered.values()):
        return
    check_distinct(answer, value, msg=f"{label} collided with the answer")
    check_pairwise_distinct(answer, *offered.values(), value, msg=f"{label} collided with another candidate")
    offered[label] = value


def _fraction_payload(value: Fraction) -> tuple[dict, str]:
    value = Fraction(value)
    return {"p": value.numerator, "q": value.denominator}, rational_text(value)


_EASY_TEMPLATES = (
    "The general solution to a first-order ODE is y = {F} + C. Given that y(0) = {y0}, find C.",
    "A first-order differential equation has general solution y = {F} + C. "
    "The initial condition y(0) = {y0} is given. Find C.",
    "y = {F} + C is the general solution of a first-order ODE. Using y(0) = {y0}, determine C.",
)

_MEDIUM_TEMPLATES = (
    "The general solution to a first-order ODE is y = {F} + C. Given that y({x0}) = {y0}, find C.",
    "A first-order differential equation has general solution y = {F} + C. "
    "The initial condition y({x0}) = {y0} is given. Find C.",
    "y = {F} + C is the general solution of a first-order ODE. Using y({x0}) = {y0}, determine C.",
)

_HARD_TEMPLATES = _MEDIUM_TEMPLATES

_EXAM_ZERO_TEMPLATES = (
    "The general solution to a first-order ODE is y = {F} + C. The solution crosses the x-axis at x = {x0}. Find C.",
    "A first-order differential equation has general solution y = {F} + C, "
    "and its graph crosses the x-axis at x = {x0}. Find C.",
    "y = {F} + C is the general solution of a first-order ODE. Given that "
    "the curve crosses the x-axis when x = {x0}, determine C.",
)

_EXAM_POINT_TEMPLATES = (
    "The general solution to a first-order ODE is y = {F} + C. The solution "
    "passes through the point where x = {x0} and y = {y0}. Find C.",
    "A first-order differential equation has general solution y = {F} + C. "
    "Its graph passes through the point where x = {x0} and y = {y0}. Find C.",
    "y = {F} + C is the general solution of a first-order ODE, and it "
    "passes through the point where x = {x0} and y = {y0}. Determine C.",
)


def _build(rng, *, x0_pool, y0_pool, exp_pool, n_terms, coef_range) -> dict:
    for _attempt in range(_MAX_ATTEMPTS):
        x0 = rng.choice(x0_pool)
        y0 = rng.choice(y0_pool)
        if x0 == y0:
            continue  # keeps substitutes-into-wrong-term exercisable
        exps = (
            rng.sample(exp_pool, n_terms)
            if n_terms <= len(exp_pool)
            else [rng.choice(exp_pool) for _ in range(n_terms)]
        )
        terms = []
        for exp in exps:
            k = rng.choice([v for v in coef_range if v != 0])
            terms.append((k, exp))

        F_x0 = _eval_F(terms, x0)
        indep = _independent_eval_F(terms, x0)
        check_equivalent(indep, F_x0, "sympy substitution disagrees with the direct evaluation of F(x0)")

        answer = y0 - F_x0
        check(F_x0 + answer == y0, "F(x0) + C does not round-trip to y0")

        offered: dict[str, Fraction] = {}
        _offer_candidate(answer, Fraction(y0), offered, "forgets-to-subtract")
        _offer_candidate(answer, y0 + F_x0, offered, "sign-error-on-F")
        wrong_term_val = y0 - _eval_F(terms, y0)
        _offer_candidate(answer, wrong_term_val, offered, "substitutes-into-wrong-term")
        if len(terms) >= 2:
            drop_idx = rng.randrange(len(terms))
            dropped_F = _drop_term_F(terms, drop_idx, x0)
            drop_val = y0 - dropped_F
            _offer_candidate(answer, drop_val, offered, "drops-a-term-of-F")
        _offer_candidate(answer, F_x0, offered, "reports-F-as-C")
        _offer_candidate(answer, y0 - 2 * F_x0, offered, "double-subtracts-F")

        if not offered:
            continue

        F_text = _poly_text(terms)
        answer_text = rational_text(answer)
        distractor_strs = [rational_text(v) for v in offered.values()]
        distractors = pick_distractors(answer_text, distractor_strs)
        if not distractors:
            continue

        return F_text, x0, y0, answer, answer_text, distractors, terms, F_x0

    raise RuntimeError("could not draw a usable initial-value-constant question")


def _build_easy(rng) -> dict:
    result = _build(
        rng,
        x0_pool=(0,),
        y0_pool=tuple(v for v in range(-20, 21) if v != 0),
        exp_pool=(2, 3),
        n_terms=2,
        coef_range=range(-6, 7),
    )
    F_text, x0, y0, answer, answer_text, distractors, terms, F_x0 = result
    text = phrase(rng, _EASY_TEMPLATES, F=F_text, y0=y0)
    working = [
        f"F({x0}) = 0 (every term of F vanishes at x = 0).",
        f"C = y0 - F(x0) = {y0} - {paren_negative(F_x0)} = {answer_text}.",
    ]
    return dict(
        text=text,
        latex=text,
        answer=_fraction_payload(answer)[0],
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )


def _build_medium(rng) -> dict:
    result = _build(
        rng,
        x0_pool=tuple(range(1, 5)),
        y0_pool=tuple(v for v in range(-9, 10) if v != 0),
        exp_pool=(2, 3, 4),
        n_terms=2,
        coef_range=range(-8, 9),
    )
    F_text, x0, y0, answer, answer_text, distractors, terms, F_x0 = result
    text = phrase(rng, _MEDIUM_TEMPLATES, F=F_text, x0=x0, y0=y0)
    working = [
        f"F({x0}) = {rational_text(F_x0)}.",
        f"C = y0 - F(x0) = {y0} - {paren_negative(F_x0)} = {answer_text}.",
    ]
    return dict(
        text=text,
        latex=text,
        answer=_fraction_payload(answer)[0],
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )


def _build_hard(rng) -> dict:
    result = _build(
        rng,
        x0_pool=tuple(range(-5, 0)),
        y0_pool=tuple(v for v in range(-9, 10) if v != 0),
        exp_pool=(2, 4),  # both even -- the sign trap requires an even power
        n_terms=2,
        coef_range=range(-8, 9),
    )
    F_text, x0, y0, answer, answer_text, distractors, terms, F_x0 = result
    text = phrase(rng, _HARD_TEMPLATES, F=F_text, x0=x0, y0=y0)
    working = [
        f"F({x0}) = {rational_text(F_x0)} (note: a negative base raised to an even power is positive).",
        f"C = y0 - F(x0) = {y0} - {paren_negative(F_x0)} = {answer_text}.",
    ]
    return dict(
        text=text,
        latex=text,
        answer=_fraction_payload(answer)[0],
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )


def _build_exam(rng) -> dict:
    variant = rng.choice(("zero", "point"))
    if variant == "zero":
        result = _build(
            rng,
            x0_pool=tuple(v for v in range(-5, 6) if v != 0),
            y0_pool=(0,),
            exp_pool=(2, 3, 4),
            n_terms=2,
            coef_range=range(-8, 9),
        )
        F_text, x0, y0, answer, answer_text, distractors, terms, F_x0 = result
        text = phrase(rng, _EXAM_ZERO_TEMPLATES, F=F_text, x0=x0)
    else:
        result = _build(
            rng,
            x0_pool=tuple(v for v in range(-5, 6) if v != 0),
            y0_pool=tuple(v for v in range(-9, 10) if v != 0),
            exp_pool=(2, 3, 4),
            n_terms=2,
            coef_range=range(-8, 9),
        )
        F_text, x0, y0, answer, answer_text, distractors, terms, F_x0 = result
        text = phrase(rng, _EXAM_POINT_TEMPLATES, F=F_text, x0=x0, y0=y0)
    working = [
        f"y({x0}) = {y0} (translating the verbal condition).",
        f"F({x0}) = {rational_text(F_x0)}.",
        f"C = y0 - F(x0) = {y0} - {paren_negative(F_x0)} = {answer_text}.",
    ]
    return dict(
        text=text,
        latex=text,
        answer=_fraction_payload(answer)[0],
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )


@generator(
    "p13f_initial_value_constant",
    level=17,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def initial_value_constant(rng, tier="easy"):
    if tier == "easy":
        return _build_easy(rng)
    if tier == "medium":
        return _build_medium(rng)
    if tier == "hard":
        return _build_hard(rng)
    if tier == "exam":
        return _build_exam(rng)
    raise ValueError(f"unknown tier {tier!r}")
