"""Generates `factorise_quadratic` questions: factorise a quadratic with any non-zero integer leading coefficient
into two binomials.

Factorise-quadratic.

Distinct from the existing ``quadratic`` generator (algebra.py, behind skill
``solve-quadratics-factorable``): that skill asks for the ROOTS of a quadratic
drawn with leading coefficient always 1. This skill asks for the
FACTORISATION itself (``answer_form="factored"``) and covers a leading
coefficient != 1, which ``quadratic`` never draws at all.

Construction is backwards from two binomial factors (p*x + q)(r*x + s) with
integer p, q, r, s, p >= 1, r >= 1 (the factors' own leading coefficients are
drawn positive; the overall trinomial leading coefficient a = p*r can still
be displayed negative at the hard tier -- see below). Discipline carried over
from ``difference_of_squares`` (p8a_difference_of_squares_build.py): no
common numerical factor within a bracket (gcd(p, q) == 1, gcd(r, s) == 1) or
across the whole trinomial (gcd(a, gcd(b, c)) == 1), otherwise
``sympy.factor`` would pull out a numerical factor the construction did not
express as a bracket, and the independent check below would disagree with
the construction by design, not by accident.
"""

from __future__ import annotations

import math

import sympy
from sympy import Poly, expand, symbols

from ..core import (
    check_distinct,
    check_equivalent,
    fmt_signed,
    generator,
    join_terms,
    pick_distractors,
    symbolically_equal,
)

x = symbols("x")


def _bracket_text(p: int, q: int) -> str:
    """'{p}x + {q}' / '{p}x - {q}' from a signed constant q -- a local, one-off adaptation of
    p8a_difference_of_squares_build.py's ``_factor_text``/``_bracket_pair`` (that pair takes a magnitude and
    an explicit sign; this one takes an already-signed q directly, since q can be any signed integer here, not
    just +/- a shared magnitude)."""
    sign = "+" if q >= 0 else "-"
    return f"{fmt_signed(p, 'x')} {sign} {abs(q)}"


def _bracket_pair(p: int, q: int, r: int, s: int) -> str:
    return f"({_bracket_text(p, q)})({_bracket_text(r, s)})"


def _coeffs(expr) -> tuple[int, int, int]:
    """(a, b, c) for a degree-2 polynomial in x, zero-padded (Poly.all_coeffs
    keeps interior zero coefficients, e.g. Poly(x**2 - 9, x).all_coeffs() ==
    [1, 0, -9]), so this is always exactly 3 long for our inputs."""
    coeffs = Poly(expr, x).all_coeffs()
    while len(coeffs) < 3:
        coeffs.insert(0, 0)
    return int(coeffs[-3]), int(coeffs[-2]), int(coeffs[-1])


def _value(p: int, q: int, r: int, s: int):
    return expand((p * x + q) * (r * x + s))


def _draw_easy(rng):
    root1 = rng.randint(1, 9)
    root2 = rng.randint(1, 9)
    return 1, -root1, 1, -root2


def _draw_medium(rng):
    # At least one of root1, root2 negative -- resample away from the
    # "both positive" easy-tier shape. Deliberately not resampling AWAY from
    # a mixed pair that happens to land on b == 0 (a difference-of-squares
    # -adjacent trinomial): that is a legitimate outcome of this draw, not
    # something to chase or avoid.
    while True:
        root1 = rng.randint(1, 9) * rng.choice([1, -1])
        root2 = rng.randint(1, 9) * rng.choice([1, -1])
        if root1 < 0 or root2 < 0:
            return 1, -root1, 1, -root2


def _draw_hard(rng):
    while True:
        p = rng.choice([1, 2, 3, 4])
        r = rng.choice([2, 3, 4])  # r >= 2 always, so a = p * r != 1 always.
        q = rng.randint(1, 9) * rng.choice([1, -1])
        s = rng.randint(1, 9) * rng.choice([1, -1])
        if math.gcd(p, abs(q)) != 1:
            continue
        if math.gcd(r, abs(s)) != 1:
            continue
        a_coef, b_coef, c_coef = _coeffs(_value(p, q, r, s))
        if math.gcd(abs(a_coef), math.gcd(abs(b_coef), abs(c_coef))) != 1:
            continue
        return p, q, r, s


@generator(
    "factorise_quadratic",
    level=9,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def factorise_quadratic(rng, tier: str = "easy"):
    if tier == "easy":
        p, q, r, s = _draw_easy(rng)
    elif tier == "medium":
        p, q, r, s = _draw_medium(rng)
    else:
        p, q, r, s = _draw_hard(rng)

    trinomial = _value(p, q, r, s)

    # Hard tier only: on roughly half of seeds, display the NEGATED trinomial
    # (leading coefficient effectively negative), carrying the sign into
    # exactly one bracket so the product still equals the negated trinomial:
    # -(p*x + q)*(r*x + s) == (-p*x - q)*(r*x + s).
    overall_sign = 1
    if tier == "hard":
        overall_sign = rng.choice([1, -1])
    if overall_sign == -1:
        p_disp, q_disp, r_disp, s_disp = -p, -q, r, s
        trinomial_disp = expand(-trinomial)
    else:
        p_disp, q_disp, r_disp, s_disp = p, q, r, s
        trinomial_disp = trinomial

    # Independent check, second route: sympy.factor is a general factorisation
    # ALGORITHM, not a re-run of this construction's own p, q, r, s -- it does
    # not know how trinomial_disp was built.
    check_equivalent(
        sympy.factor(trinomial_disp),
        (p_disp * x + q_disp) * (r_disp * x + s_disp),
        "factorise_quadratic construction must agree with sympy.factor's independent route",
    )

    a_coef, b_coef, c_coef = _coeffs(trinomial_disp)
    trinomial_text = join_terms([(a_coef, "x^2"), (b_coef, "x"), (c_coef, "")])
    answer_text = _bracket_pair(p_disp, q_disp, r_disp, s_disp)

    cands: list[str] = []

    # sign-flip-one-factor: flips the sign of q OR s alone (chosen per seed
    # for variety) -- e.g. (p*x - q)(r*x + s) for a true (p*x + q)(r*x + s).
    if rng.choice([True, False]):
        flip_one = _value(p_disp, -q_disp, r_disp, s_disp)
        flip_one_text = _bracket_pair(p_disp, -q_disp, r_disp, s_disp)
    else:
        flip_one = _value(p_disp, q_disp, r_disp, -s_disp)
        flip_one_text = _bracket_pair(p_disp, q_disp, r_disp, -s_disp)
    if not symbolically_equal(trinomial_disp, flip_one):
        check_distinct(trinomial_disp, flip_one, msg="sign-flip-one-factor must be a genuinely different value")
        cands.append(flip_one_text)

    # sign-flip-both-factors: flips the sign of both q and s.
    flip_both = _value(p_disp, -q_disp, r_disp, -s_disp)
    flip_both_text = _bracket_pair(p_disp, -q_disp, r_disp, -s_disp)
    if not symbolically_equal(trinomial_disp, flip_both):
        check_distinct(trinomial_disp, flip_both, msg="sign-flip-both-factors must be a genuinely different value")
        cands.append(flip_both_text)

    # swapped-cross-terms, hard tier only: swaps which bracket carries which leading coefficient --
    # (r*x + q)(p*x + s) instead of (p*x + q)(r*x + s). Checked against the DISPLAYED p_disp/r_disp
    # (post sign-flip), because that is the answer actually offered; when p == r pre-sign-flip and
    # overall_sign == 1, p_disp == r_disp and this collides with the true answer and must be
    # dropped.
    if tier == "hard":
        swapped = _value(r_disp, q_disp, p_disp, s_disp)
        swapped_text = _bracket_pair(r_disp, q_disp, p_disp, s_disp)
        if not symbolically_equal(trinomial_disp, swapped):
            check_distinct(trinomial_disp, swapped, msg="swapped-cross-terms must be a genuinely different value")
            cands.append(swapped_text)

    text = f"Factorise {trinomial_text}."
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=pick_distractors(answer_text, cands),
        answer_form="factored",
    )
