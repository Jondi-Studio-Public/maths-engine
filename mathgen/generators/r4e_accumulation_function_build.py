"""Generates `r4e_accumulation_function` questions (easy, medium, hard) and the exam chain `r4e_accumulation_function_chain`.

Skill `accumulation-function` (answer `fraction`): F(x) is the integral of f(t) from t = a to t = x.

* easy: a value F(c), which needs the lower limit a (F(a) = 0), not f(c) and not the integral from 0;
* medium: F'(c) = f(c), whatever the lower limit, or F''(c) = f'(c);
* hard: either the x-value where F has a local maximum (a sign change of f from positive to negative, not the vertex
  of f and not the lower limit), or F'(c) when the upper limit is a function of x, by the chain rule:
  F(x) = integral of f from a to g(x) has F'(x) = f(g(x)) g'(x);
* exam: for f(t) = -k(t - p)(t - q), a value F(c), where F is increasing, where it has its local maximum and the
  value of F there, the last following through from the learner's own x-value.

Polynomials are coefficient lists in exact arithmetic (`r4e_common.pev`, `pder`, `pint`). Each key is cross-checked
by a second route: composite Simpson's rule (exact for a cubic), a central difference quotient of F, or a direct look
at F on either side of the claimed maximum.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import attempts, check, generator, phrase
from .r4e_common import (
    Redraw,
    definite,
    draw,
    fraction_part,
    inequality_part,
    num,
    pder,
    pev,
    pint,
    poly_code,
    poly_text,
    single,
)


def _poly(rng, degree):
    coeffs = [rng.choice((1, 2, 3, -1, -2))]
    coeffs += [rng.choice((-4, -3, -2, -1, 0, 1, 2, 3, 4)) for _ in range(degree)]
    return coeffs


def _simpson_check(f, lo, hi, value) -> None:
    """Composite Simpson's rule with 2 strips is exact for a cubic: a second route to the integral."""
    mid = Fraction(lo + hi, 2)
    check((Fraction(hi - lo, 6)) * (pev(f, lo) + 4 * pev(f, mid) + pev(f, hi)) == value, "integral disagrees")


# -- easy ---------------------------------------------------------------------------------------------------------

_EASY_STEMS = (
    "Let F(x) be the integral of f(t) from t = {a} to t = x, where f(t) = {f}.",
    "The function F is defined by accumulating f(t) = {f} from t = {a} up to t = x.",
)


def _easy(rng):
    for _ in attempts(500, "a nonzero value distinct from its distractors"):
        f = _poly(rng, rng.choice((1, 2)))
        a = rng.choice((1, 2, -1, 3))
        c = a + rng.choice((1, 2, 3))
        key = definite(f, a, c)
        wrong = [
            (pev(f, c), "accumulation-value-as-rate"),
            (definite(f, 0, c), "accumulation-ignores-lower-limit"),
            (-key, "accumulation-limits-reversed"),
            (pev(f, c) * (c - a), "accumulation-rectangle-for-integral"),
        ]
        if key != 0 and len({key, *(w for w, _ in wrong)}) >= 4:
            break
    _simpson_check(f, a, c, key)
    anti = pint(f)
    stem = phrase(rng, _EASY_STEMS, a=a, f=poly_text(f, "t"))
    working = [
        f"F(x) = [{poly_text(anti, 't')}] from {a} to x.",
        f"F({c}) = {num(pev(anti, c))} - ({num(pev(anti, a))}) = {num(key)}.",
    ]
    return single(stem, fraction_part(f"Find F({c}).", key, 2, working, wrong))


# -- medium -------------------------------------------------------------------------------------------------------


def _medium(rng):
    for _ in attempts(500, "a derivative value distinct from its distractors"):
        f = _poly(rng, rng.choice((2, 3)))
        a = rng.choice((0, 1, 2, -1, 3))
        c = rng.choice((-2, -1, 1, 2, 3))
        second = rng.random() < 0.4
        fc, fa, dfc = pev(f, c), pev(f, a), pev(pder(f), c)
        if second:
            key = dfc
            wrong = [
                (fc, "accumulation-derivative-one-short"),
                (pev(pder(pder(f)), c), "accumulation-derivative-too-many"),
                (definite(f, a, c), "accumulation-value-as-rate"),
            ]
            prompt, label = f"Find F''({c}).", "F''"
        else:
            key = fc
            wrong = [
                (fc - fa, "accumulation-derivative-subtracts-lower-limit"),
                (dfc, "accumulation-differentiates-integrand"),
                (definite(f, a, c), "accumulation-value-as-rate"),
            ]
            prompt, label = f"Find F'({c}).", "F'"
        if len({key, *(w for w, _ in wrong)}) == 4:
            break
    # second route: differentiate the explicit polynomial F(x) = G(x) - G(a)
    anti = pint(f)
    check(pev(pder(anti), c) == fc, "F' disagrees with f")
    check(pev(pder(pder(anti)), c) == dfc, "F'' disagrees with f'")
    stem = phrase(rng, _EASY_STEMS, a=a, f=poly_text(f, "t"))
    working = [
        "By the fundamental theorem of calculus, F'(x) = f(x), whatever the lower limit.",
        f"{label}({c}) = {num(key)}.",
    ]
    return single(stem, fraction_part(prompt, key, 2, working, wrong))


# -- hard ---------------------------------------------------------------------------------------------------------


def _hard_max(rng):
    for _ in attempts(500, "a quadratic integrand whose roots and vertex differ from the lower limit"):
        p, q = rng.sample(range(-3, 6), 2)
        p, q = min(p, q), max(p, q)
        k = rng.choice((1, 2, 3))
        a = rng.choice((0, 1, 2, -1, 4))
        if q - p >= 2 and a not in (p, q) and Fraction(p + q, 2) != a:
            break
    f = [-k, k * (p + q), -k * p * q]
    # second route: F really is larger at x = q than a little to either side, and smaller than at the lower neighbour p
    here = definite(f, a, q)
    check(
        all(definite(f, a, q + d) < here for d in (Fraction(1, 10), Fraction(-1, 10))), "F is not locally greatest at q"
    )
    wrong = [
        (p, "accumulation-min-taken-for-max"),
        (Fraction(p + q, 2), "accumulation-max-of-integrand"),
        (a, "accumulation-lower-limit-as-stationary-point"),
    ]
    stem = phrase(rng, _EASY_STEMS, a=a, f=poly_text(f, "t"))
    working = [
        "F'(x) = f(x), so the stationary points of F are the roots of f.",
        f"f changes from positive to negative at x = {q}, so F has a local maximum there.",
    ]
    return single(stem, fraction_part("Find the x-value at which F has a local maximum.", q, 3, working, wrong))


# (text of g(x), g, g')
_UPPER = (
    ("x^2", lambda v: v * v, lambda v: 2 * v),
    ("2x + 1", lambda v: 2 * v + 1, lambda v: 2),
    ("x^3", lambda v: v**3, lambda v: 3 * v * v),
)


def _hard_chain_rule(rng):
    for _ in attempts(500, "a chain-rule value distinct from its distractors"):
        f = _poly(rng, 2)
        a = rng.choice((0, 1, 2, -1))
        g_text, g, dg = rng.choice(_UPPER)
        c = rng.choice((1, 2, -1))
        key = pev(f, g(c)) * dg(c)
        wrong = [
            (pev(f, g(c)), "accumulation-chain-rule-forgotten"),
            (pev(f, c) * dg(c), "accumulation-upper-limit-not-substituted"),
            (pev(f, g(c)) * g(c), "accumulation-chain-rule-wrong-factor"),
        ]
        if key != 0 and len({key, *(w for w, _ in wrong)}) == 4:
            break
    # second route: a central difference of F(x) = integral of f from a to g(x), whose error shrinks with h^2
    h = Fraction(1, 10_000)
    slope = (definite(f, a, g(c + h)) - definite(f, a, g(c - h))) / (2 * h)
    check(
        abs(slope - key) <= Fraction(1, 100) * (1 + abs(key)), "chain-rule value disagrees with a difference quotient"
    )
    stem = f"Let F(x) be the integral of f(t) from t = {a} to t = {g_text}, where f(t) = {poly_text(f, 't')}."
    working = [
        f"F(x) = integral of f from {a} to u, where u = {g_text}, so F'(x) = f(u) · du/dx.",
        f"F'({c}) = f({g(c)}) × {dg(c)} = {num(key)}.",
    ]
    return single(stem, fraction_part(f"Find F'({c}).", key, 3, working, wrong))


@generator(
    "r4e_accumulation_function",
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4e_accumulation_function(rng, tier: str = "easy"):
    """The accumulation function F(x) = integral of f from a to x; see the module docstring for the ladder."""
    if tier == "easy":
        return draw(_easy, rng)
    if tier == "medium":
        return draw(_medium, rng)
    if tier == "hard":
        return draw(_hard_max if rng.random() < 0.5 else _hard_chain_rule, rng)
    raise ValueError(f"unknown tier {tier!r}")


# -- exam chain ---------------------------------------------------------------------------------------------------


def _chain(rng):
    for _ in attempts(500, "a quadratic integrand with distinct roots and a lower limit away from them"):
        p, q = rng.sample(range(-2, 6), 2)
        p, q = min(p, q), max(p, q)
        k = rng.choice((1, 2, 3))
        a = rng.choice((0, 1, -1, 2))
        c = rng.choice((1, 2, 3, 4, 5, -1))
        if q - p >= 2 and a not in (p, q) and c not in (a, p, q):
            break
    f = [-k, k * (p + q), -k * p * q]
    value_c = definite(f, a, c)
    max_value = definite(f, a, q)
    anti = pint(f)
    check(pev(f, p) == 0 and pev(f, q) == 0 and pev(f, Fraction(p + q, 2)) > 0, "f is not positive between its roots")
    check(
        all(definite(f, a, q + d) < max_value for d in (Fraction(1, 10), Fraction(-1, 10))),
        "F is not locally greatest at q",
    )
    _simpson_check(f, a, c, value_c)
    if value_c == 0 or max_value == 0:
        raise Redraw("a zero value")
    stem = phrase(rng, _EASY_STEMS, a=a, f=poly_text(f, "t"))
    part_a = fraction_part(
        f"Find F({c}).",
        value_c,
        2,
        [f"F(x) = G(x) - G({a}) with G(t) = {poly_text(anti, 't')}.", f"F({c}) = {num(value_c)}."],
        [
            (pev(f, c), "accumulation-value-as-rate"),
            (definite(f, 0, c), "accumulation-ignores-lower-limit"),
            (-value_c, "accumulation-limits-reversed"),
        ],
    )
    part_b = inequality_part(
        "Find the values of x for which F is increasing.",
        [(p, q)],
        2,
        [f"F'(x) = f(x) = {poly_text(f, 'x')}, which is positive when {p} < x < {q}."],
        [
            ([(None, p), (q, None)], "accumulation-increasing-decreasing-swapped"),
            ([(None, Fraction(p + q, 2))], "accumulation-max-of-integrand"),
            ([(p, None)], "accumulation-ignores-second-root"),
        ],
    )
    part_c = fraction_part(
        "Hence find the x-value at which F has its local maximum.",
        q,
        1,
        [f"F is increasing for {p} < x < {q} and decreasing after x = {q}, so the local maximum is at x = {q}."],
        [
            (p, "accumulation-min-taken-for-max"),
            (Fraction(p + q, 2), "accumulation-max-of-integrand"),
            (a, "accumulation-lower-limit-as-stationary-point"),
        ],
    )
    follow = f"{poly_code(anti, 'part_c')} - ({num(pev(anti, a))})".replace("/", "/")
    part_d = fraction_part(
        "Find the local maximum value of F.",
        max_value,
        3,
        [f"F({q}) = G({q}) - G({a}) = {num(max_value)}."],
        [
            (definite(f, a, p), "accumulation-min-taken-for-max"),
            (pev(anti, q), "accumulation-ignores-lower-limit"),
            (pev(f, Fraction(p + q, 2)), "accumulation-max-of-integrand"),
            (pev(f, q), "accumulation-value-as-rate"),
        ],
        follow_through=follow,
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


@generator("r4e_accumulation_function_chain", level=17, presets={"exam": dict(tier="exam")})
def r4e_accumulation_function_chain(rng, tier: str = "exam"):
    """F(x) = integral of -k(t - p)(t - q) from a: F(c), where F increases, its local maximum and the value there."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return draw(_chain, rng)
