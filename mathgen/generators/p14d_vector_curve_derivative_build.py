"""Generates `p14d_vector_curve_derivative` questions for `ucv1-vector-valued-derivative`: differentiate a two-
component polynomial curve r(t) = (x(t), y(t)) componentwise and evaluate the derivative at a stated parameter
t0.

Each component is stored as a list of ``(power, coeff)`` pairs with integer
coefficients, degree <= 3. Construction differentiates term-by-term with a
hand loop over those pairs (``_diff_terms``/``_eval_terms``); the independent
check builds a *fresh* SymPy expression from the same ``(power, coeff)``
pairs via a different route (``sympy.diff`` then ``.subs``), never reusing
the hand-rolled differentiation -- see ``_derive``.
"""

from __future__ import annotations

from fractions import Fraction

import sympy as sp

from ..core import (
    check_equivalent,
    check_pairwise_distinct,
    generator,
    join_terms,
    phrase,
    pick_distractors,
    rational_text,
    symbolically_equal,
)

_T = sp.Symbol("t")
_MAX_DRAWS = 200


def _nz(rng, lo, hi):
    """A nonzero random integer in [lo, hi]."""
    v = 0
    while v == 0:
        v = rng.randint(lo, hi)
    return v


def _diff_terms(terms):
    """Power-rule differentiate [(power, coeff), ...] term by term. A
    constant term (power 0) simply drops out."""
    return [(p - 1, c * p) for p, c in terms if p >= 1]


def _eval_terms(terms, t0: Fraction) -> Fraction:
    total = Fraction(0)
    for p, c in terms:
        total += Fraction(c) * (t0**p)
    return total


def _sympy_poly(terms):
    return sum(sp.Integer(c) * _T**p for p, c in terms)


def _term_var(power: int) -> str:
    if power == 0:
        return ""
    if power == 1:
        return "t"
    return f"t^{power}"


def _poly_text(terms) -> str:
    ordered = sorted(((p, c) for p, c in terms if c != 0), reverse=True)
    return join_terms([(c, _term_var(p)) for p, c in ordered])


def _t0_text(t0: Fraction) -> str:
    return rational_text(t0)


def _pair_text(a: Fraction, b: Fraction) -> str:
    return f"({rational_text(a)}, {rational_text(b)})"


def _derive(terms_x, terms_y, t0: Fraction):
    """The construction route (hand loop over (power, coeff)) and the
    independent check route (a freshly built sympy expression from the same
    coefficients, differentiated with sympy.diff and substituted with
    .subs) -- both computed here and cross-checked with check_equivalent,
    unconditionally, per component."""
    ans_x = _eval_terms(_diff_terms(terms_x), t0)
    ans_y = _eval_terms(_diff_terms(terms_y), t0)
    t0_sym = sp.Rational(t0.numerator, t0.denominator)
    check_x = sp.diff(_sympy_poly(terms_x), _T).subs(_T, t0_sym)
    check_y = sp.diff(_sympy_poly(terms_y), _T).subs(_T, t0_sym)
    check_equivalent(ans_x, check_x, "x'(t0) disagrees with the independent sympy route")
    check_equivalent(ans_y, check_y, "y'(t0) disagrees with the independent sympy route")
    orig_x = _eval_terms(terms_x, t0)
    orig_y = _eval_terms(terms_y, t0)
    return ans_x, ans_y, orig_x, orig_y


def _standard_distractors(ans_x, ans_y, orig_x, orig_y):
    """swapped-components, forgot-to-differentiate (evaluate the original
    function instead), sign-error (negate whichever component is nonzero,
    so the negation is never a no-op)."""
    swap = (ans_y, ans_x)
    eval_orig = (orig_x, orig_y)
    if ans_x != 0:
        sign_err = (-ans_x, ans_y)
    elif ans_y != 0:
        sign_err = (ans_x, -ans_y)
    else:
        sign_err = (Fraction(-1), Fraction(-1))
    return swap, eval_orig, sign_err


def _all_pairwise_distinct(values) -> bool:
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if symbolically_equal(values[i], values[j]):
                return False
    return True


_LEAD_PATTERNS = [
    "A curve is defined by r(t) = ({x_text}, {y_text}). Find r'(t) at t = {t0_text}.",
    "For the vector-valued curve r(t) = ({x_text}, {y_text}), find r'(t) at t = {t0_text}.",
    "Let r(t) = ({x_text}, {y_text}) be a curve. Find r'(t) at t = {t0_text}.",
]


def _finalise(rng, terms_x, terms_y, t0, ans_x, ans_y, swap, eval_orig, sign_err):
    x_text = _poly_text(terms_x)
    y_text = _poly_text(terms_y)
    t0_text = _t0_text(t0)
    answer_text = _pair_text(ans_x, ans_y)
    candidates = [_pair_text(*swap), _pair_text(*eval_orig), _pair_text(*sign_err)]
    text = phrase(rng, _LEAD_PATTERNS, x_text=x_text, y_text=y_text, t0_text=t0_text)
    dx_text = _poly_text(_diff_terms(terms_x)) or "0"
    dy_text = _poly_text(_diff_terms(terms_y)) or "0"
    working = [
        f"x'(t) = {dx_text}, so x'({t0_text}) = {rational_text(ans_x)}.",
        f"y'(t) = {dy_text}, so y'({t0_text}) = {rational_text(ans_y)}.",
    ]
    return dict(
        text=text,
        answer={"x": ans_x, "y": ans_y},
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, candidates),
    )


def _easy_component(rng, degree=None):
    """Monomial or binomial, degree <= 2."""
    if degree is None:
        degree = rng.randint(1, 2)
    terms = [(degree, _nz(rng, -5, 5))]
    if rng.random() < 0.6:
        lower = rng.choice(list(range(0, degree)))
        terms.append((lower, _nz(rng, -5, 5)))
    return terms


def _degree3_component(rng, include_constant=None):
    terms = [(3, _nz(rng, -4, 4)), (2, _nz(rng, -6, 6)), (1, _nz(rng, -6, 6))]
    if include_constant is None:
        include_constant = rng.random() < 0.5
    if include_constant:
        terms.append((0, _nz(rng, -6, 6)))
    return terms


def _linear_component(rng):
    """Purely linear (optionally + constant): the derivative is the SAME
    constant no matter what t0 is."""
    terms = [(1, _nz(rng, -6, 6))]
    if rng.random() < 0.7:
        terms.append((0, _nz(rng, -6, 6)))
    return terms


def _easy_t0(rng) -> Fraction:
    return Fraction(rng.randint(1, 4))


def _medium_t0(rng) -> Fraction:
    return Fraction(rng.randint(1, 6))


def _hard_t0(rng) -> Fraction:
    sign = rng.choice([1, -1])
    if rng.random() < 0.5:
        return Fraction(sign * rng.randint(1, 6))
    n = rng.randint(1, 11)
    if n % 2 == 0:
        n += 1
    return Fraction(sign * n, 2)


def _draw_until_distinct(build_once):
    """Retry the draw (re-consuming rng state, which keeps the seed
    deterministic) until the answer and every distractor candidate are
    pairwise distinct, then re-assert it with check_pairwise_distinct as
    the authoritative, raising check."""
    last = None
    for _ in range(_MAX_DRAWS):
        result = build_once()
        pool = result["pool"]
        if _all_pairwise_distinct(pool):
            check_pairwise_distinct(*pool, msg="a distractor collides with the answer or another distractor")
            return result
        last = result
    raise RuntimeError("p14d_vector_curve_derivative: could not draw a distinct-option question")


def _easy(rng):
    def once():
        # At least one component is forced to degree 2 (rather than both
        # independently drawing degree 1, which would leave the power rule
        # with no substitution step to make on the vanished-constant side --
        # measured at ~25% of draws before this guard).
        if rng.random() < 0.5:
            terms_x = _easy_component(rng, degree=2)
            terms_y = _easy_component(rng)
        else:
            terms_x = _easy_component(rng)
            terms_y = _easy_component(rng, degree=2)
        t0 = _easy_t0(rng)
        ans_x, ans_y, orig_x, orig_y = _derive(terms_x, terms_y, t0)
        swap, eval_orig, sign_err = _standard_distractors(ans_x, ans_y, orig_x, orig_y)
        return dict(
            terms_x=terms_x,
            terms_y=terms_y,
            t0=t0,
            ans_x=ans_x,
            ans_y=ans_y,
            swap=swap,
            eval_orig=eval_orig,
            sign_err=sign_err,
            pool=[(ans_x, ans_y), swap, eval_orig, sign_err],
        )

    r = _draw_until_distinct(once)
    return _finalise(
        rng, r["terms_x"], r["terms_y"], r["t0"], r["ans_x"], r["ans_y"], r["swap"], r["eval_orig"], r["sign_err"]
    )


def _medium(rng):
    def once():
        # Both components degree 3 (more terms than easy, genuinely more
        # arithmetic) -- paired with the tier's real reasoning step: one
        # component (randomly x or y) is FORCED to also carry a constant
        # term, so the learner must decide which terms survive
        # differentiation (the constant vanishes; the degree-1 term becomes
        # a t0-independent constant) rather than just substituting into
        # more monomials. This must stay a genuinely different shape from
        # hard's purely-linear component (degree 1 only, no cubic term) --
        # never collapse the two tiers to the same polynomial structure.
        if rng.random() < 0.5:
            terms_x = _degree3_component(rng, include_constant=True)
            terms_y = _degree3_component(rng)
        else:
            terms_x = _degree3_component(rng)
            terms_y = _degree3_component(rng, include_constant=True)
        t0 = _medium_t0(rng)
        ans_x, ans_y, orig_x, orig_y = _derive(terms_x, terms_y, t0)
        swap, eval_orig, sign_err = _standard_distractors(ans_x, ans_y, orig_x, orig_y)
        return dict(
            terms_x=terms_x,
            terms_y=terms_y,
            t0=t0,
            ans_x=ans_x,
            ans_y=ans_y,
            swap=swap,
            eval_orig=eval_orig,
            sign_err=sign_err,
            pool=[(ans_x, ans_y), swap, eval_orig, sign_err],
        )

    r = _draw_until_distinct(once)
    return _finalise(
        rng, r["terms_x"], r["terms_y"], r["t0"], r["ans_x"], r["ans_y"], r["swap"], r["eval_orig"], r["sign_err"]
    )


def _hard(rng):
    def once():
        if rng.random() < 0.5:
            terms_x = _linear_component(rng)
            terms_y = _degree3_component(rng)
        else:
            terms_x = _degree3_component(rng)
            terms_y = _linear_component(rng)
        t0 = _hard_t0(rng)
        ans_x, ans_y, orig_x, orig_y = _derive(terms_x, terms_y, t0)
        swap, eval_orig, sign_err = _standard_distractors(ans_x, ans_y, orig_x, orig_y)
        return dict(
            terms_x=terms_x,
            terms_y=terms_y,
            t0=t0,
            ans_x=ans_x,
            ans_y=ans_y,
            swap=swap,
            eval_orig=eval_orig,
            sign_err=sign_err,
            pool=[(ans_x, ans_y), swap, eval_orig, sign_err],
        )

    r = _draw_until_distinct(once)
    return _finalise(
        rng, r["terms_x"], r["terms_y"], r["t0"], r["ans_x"], r["ans_y"], r["swap"], r["eval_orig"], r["sign_err"]
    )


_EXAM_PATTERNS = [
    "A curve has x(t) = {x_text} and y(t) = {y_text}. Find r'(t) at t = {t0}.",
    "For the vector-valued curve r(t) = (x(t), y(t)) with x(t) = {x_text} and y(t) = {y_text}, find r'(t) at t = {t0}.",
    "Let r(t) = (x(t), y(t)), where x(t) = {x_text} and y(t) = {y_text}. Find r'(t) at t = {t0}.",
]


def _factor_text(p: int) -> str:
    """`t - p` written the way a student would write it, sign-adjusted for a
    negative `p` (so p=-2 renders 't + 2', never 't - -2')."""
    return join_terms([(1, "t"), (-p, "")])


def _product_text(p1: int, q1: int) -> str:
    return f"({_factor_text(p1)})({_factor_text(q1)})"


def _expand_product_terms(p1: int, q1: int):
    """(t - p1)(t - q1) = t^2 - (p1+q1)t + p1*q1, as (power, coeff) pairs."""
    return [(2, 1), (1, -(p1 + q1)), (0, p1 * q1)]


def _finalise_exam(rng, p1, q1, terms_y, t0, ans_x, ans_y, forgot_one, swap, eval_orig, sign_err):
    x_text = _product_text(p1, q1)
    y_text = _poly_text(terms_y)
    t0_text = _t0_text(t0)
    answer_text = _pair_text(ans_x, ans_y)
    candidates = [_pair_text(*forgot_one), _pair_text(*swap), _pair_text(*eval_orig), _pair_text(*sign_err)]
    text = phrase(rng, _EXAM_PATTERNS, x_text=x_text, y_text=y_text, t0=t0_text)
    dy_text = _poly_text(_diff_terms(terms_y)) or "0"
    working = [
        f"x(t) = ({_factor_text(p1)})({_factor_text(q1)}). Product rule: "
        f"x'(t) = ({_factor_text(q1)}) + ({_factor_text(p1)}) = "
        f"{join_terms([(2, 't'), (-(p1 + q1), '')])}. "
        f"(Equivalently, expand to {_poly_text(_expand_product_terms(p1, q1))} and differentiate "
        f"term by term -- both routes agree.) So x'({t0_text}) = {rational_text(ans_x)}.",
        f"y'(t) = {dy_text}, so y'({t0_text}) = {rational_text(ans_y)}.",
    ]
    return dict(
        text=text,
        answer={"x": ans_x, "y": ans_y},
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, candidates),
    )


def _exam(rng):
    def once():
        p1 = _nz(rng, -6, 6)
        q1 = p1
        while q1 == p1:
            q1 = _nz(rng, -6, 6)
        terms_y = _degree3_component(rng)
        t0 = _hard_t0(rng)

        # Both derivation routes for the product component, cross-checked
        # against each other AND against a fresh sympy expression -- this is
        # the "genuinely different derivation route" the product-form
        # component is meant to force.
        expand_terms = _expand_product_terms(p1, q1)
        ans_x_expand = _eval_terms(_diff_terms(expand_terms), t0)
        ans_x_product_rule = (t0 - q1) + (t0 - p1)
        check_equivalent(
            ans_x_product_rule,
            ans_x_expand,
            "product-rule and expand-then-differentiate routes disagree for x'(t0)",
        )
        _, ans_y, _, orig_y = _derive(expand_terms, terms_y, t0)
        ans_x = ans_x_expand
        orig_x = _eval_terms(expand_terms, t0)

        forgot_one = (t0 - q1, ans_y)  # reports "(t - q1)" alone, drops "(t - p1)"
        swap, eval_orig, sign_err = _standard_distractors(ans_x, ans_y, orig_x, orig_y)
        return dict(
            p1=p1,
            q1=q1,
            terms_y=terms_y,
            t0=t0,
            ans_x=ans_x,
            ans_y=ans_y,
            forgot_one=forgot_one,
            swap=swap,
            eval_orig=eval_orig,
            sign_err=sign_err,
            pool=[(ans_x, ans_y), forgot_one, swap, eval_orig, sign_err],
        )

    r = _draw_until_distinct(once)
    return _finalise_exam(
        rng,
        r["p1"],
        r["q1"],
        r["terms_y"],
        r["t0"],
        r["ans_x"],
        r["ans_y"],
        r["forgot_one"],
        r["swap"],
        r["eval_orig"],
        r["sign_err"],
    )


@generator(
    "p14d_vector_curve_derivative",
    level=18,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p14d_vector_curve_derivative(rng, tier="easy"):
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    if tier == "hard":
        return _hard(rng)
    if tier == "exam":
        return _exam(rng)
    raise ValueError(f"unknown tier {tier!r}")
