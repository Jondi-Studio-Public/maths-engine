"""Generates `exd_x_r2b_second_derivative_chain` questions: a multi-part exam chain on the second derivative test.

Skill: senior-understand-and-use-the-second-derivative-test-for (exam tier).

A cubic `f(x) = k x^3 + A x^2 + B x + C` has an unknown leading coefficient `k` and a stationary point at a stated
`x = p`. It has a second stationary point at `x = q`. The parts run as a senior exam sets them:

(a) find k from f'(p) = 0                                         fraction, 2 marks
(b) find the x-coordinate q of the other stationary point         fraction, 1 mark,  q = -2A/(3k) - p
(c) f''(q) and, from its sign, the nature of that point           ordered_list, 2 marks
(d) the value of f at that point                                  fraction, 1 mark,  f(q) from k and q

## Construction: choose the roots first

`k`, `p` and `q` are drawn, then `f'(x) = 3k (x - p)(x - q)` fixes `A = -3k(p + q)/2` and `B = 3kpq` (so `k (p + q)`
is chosen to make A a whole number), and `C` is free. `f''(q) = 3k (q - p)` is never zero, so the test is always
conclusive, and its sign decides the nature: minimum when `k (q - p) > 0`.

Every key is re-derived with SymPy (`diff`, `solve`) from the finished polynomial, never from the construction.
"""

from __future__ import annotations

import sympy
from sympy import Rational, diff, symbols

from ..core import attempts, check, generator, join_terms, paren_negative, phrase
from ._exd_distinct_parts import distinct_part_answers
from .exd_x_r2b_support import fraction_part, words_part, draw

x = symbols("x")

_K = (-3, -2, -1, 1, 2, 3)
_STEMS = (
    "The function f(x) = {f}, where k is a constant, has a stationary point at x = {p}.",
    "Let f(x) = {f}, where k is a constant. The graph of f has a stationary point at x = {p}.",
    "The graph of y = f(x), where f(x) = {f} and k is a constant, has a stationary point where x = {p}.",
)


def _nature(second: int) -> str:
    return "local minimum" if second > 0 else "local maximum"


@distinct_part_answers
def _build(rng):
    """A cubic with an unknown coefficient: find k, the other stationary point, its nature and its value."""
    for _ in attempts(1000, "a cubic with two stationary points and whole coefficients"):
        k = rng.choice(_K)
        p = rng.choice([v for v in range(-4, 5) if v])
        q = rng.choice([v for v in range(-4, 5) if v])
        if q == p or p + q == 0 or (3 * k * (p + q)) % 2:
            continue
        a_coef, b_coef = -3 * k * (p + q) // 2, 3 * k * p * q
        c_coef = rng.choice([v for v in range(-12, 13) if v])
        break

    # ---- independent derivation from the finished polynomial ---------------------------------------------------
    kk = symbols("k")
    poly = kk * x**3 + a_coef * x**2 + b_coef * x + c_coef
    check(sympy.solve(diff(poly, x).subs(x, p), kk) == [k], "k does not make x = p stationary")
    f = poly.subs(kk, k)
    check(set(sympy.solve(diff(f, x), x)) == {p, q}, "stationary points are not p and q")
    second_q = int(diff(f, x, 2).subs(x, q))
    check(second_q == 3 * k * (q - p) and second_q != 0, "f''(q) disagrees with the construction")
    value_q = int(f.subs(x, q))
    nature = _nature(second_q)

    f_text = join_terms([(1, "kx^3"), (a_coef, "x^2"), (b_coef, "x"), (c_coef, "")])
    stem = phrase(rng, _STEMS, f=f_text, p=p)

    # ---- (a) k --------------------------------------------------------------------------------------------------
    num = 2 * a_coef * p + b_coef
    a_wrong = [
        (Rational(num, 3 * p * p), "sign-error-moving-term"),  # f'(p) = 0 rearranged with the wrong sign
        (Rational(-num, p * p), "forgot-power-multiplier"),  # 3kx^2 differentiated without its 3
        (Rational(-(a_coef * p + b_coef), 3 * p * p), "forgot-power-multiplier"),  # 2Ax lost its 2
        (Rational(-(a_coef * p**2 + b_coef * p + c_coef), p**3), "solved-f-zero-not-f-prime"),  # f(p) = 0 solved
        (Rational(-a_coef, 3 * p), "second-derivative-set-to-zero"),  # f''(p) = 0 solved
    ]
    part_a = fraction_part(
        "Find the value of k.",
        k,
        2,
        [
            f"f'(x) = {join_terms([(3, 'kx^2'), (2 * a_coef, 'x'), (b_coef, '')])}, and f'({p}) = 0.",
            f"{3 * p * p}k {'+' if num >= 0 else '-'} {abs(num)} = 0, so k = {k}.",
        ],
        a_wrong,
    )

    # ---- (b) the other stationary point --------------------------------------------------------------------------
    b_wrong = [
        (Rational(-2 * a_coef, 3 * k), "sum-of-roots-taken-as-root"),  # the sum of the roots, not the other root
        (Rational(-a_coef, 3 * k), "second-derivative-set-to-zero"),  # solved f''(x) = 0
        (-q, "sign-flip-one-root"),
        (Rational(b_coef, 3 * k), "product-of-roots-taken-as-root"),  # the product of the roots
    ]
    part_b = fraction_part(
        "Find the x-coordinate of the other stationary point of the graph.",
        q,
        1,
        [
            f"f'(x) = {3 * k}x^2 {'+' if -3 * k * (p + q) >= 0 else '-'} {abs(3 * k * (p + q))}x "
            f"{'+' if b_coef >= 0 else '-'} {abs(b_coef)} = {3 * k}(x - {paren_negative(p)})(x - {paren_negative(q)}).",
            f"The stationary points are x = {p} and x = {q}.",
        ],
        b_wrong,
        follow_through=f"-2*{paren_negative(a_coef)}/(3*part_a) - {paren_negative(p)}",
    )

    # ---- (c) f''(q) and the nature ------------------------------------------------------------------------------
    other = "local maximum" if nature == "local minimum" else "local minimum"
    at_p = int(diff(f, x, 2).subs(x, p))
    dropped = 6 * q + 2 * a_coef  # f'' with the k left out of 6kx
    c_wrong = [
        ([second_q, other], "second-derivative-sign-reversed"),
        ([at_p, _nature(at_p)], "second-derivative-at-wrong-point"),
        ([dropped, _nature(dropped)] if dropped else None, "forgets-the-multiplier"),
        ([-second_q, other], "sign-reversed-result"),
    ]
    part_c = words_part(
        "Find f''(x) at the stationary point found in (b) and state its nature. Give the value of f''(x), "
        "then local maximum or local minimum, separated by a comma.",
        [second_q, nature],
        2,
        [
            f"f''(x) = {6 * k}x {'+' if 2 * a_coef >= 0 else '-'} {abs(2 * a_coef)}, so f''({q}) = {second_q}.",
            f"f''({q}) is {'positive' if second_q > 0 else 'negative'}, so the point is a {nature}.",
        ],
        [w for w in c_wrong if w[0] is not None],
    )

    # ---- (d) the value at that point ---------------------------------------------------------------------------
    d_wrong = [
        (int(f.subs(x, p)), "evaluated-at-the-wrong-x"),
        (value_q - c_coef, "dropped-constant-term"),
        (int(diff(f, x, 2).subs(x, q)), "evaluated-wrong-function-order"),
        (-value_q, "sign-reversed-result"),
    ]
    part_d = fraction_part(
        "Find the value of f(x) at this stationary point.",
        value_q,
        1,
        [f"f({q}) = {k}({q})^3 + {a_coef}({q})^2 + {b_coef}({q}) + {c_coef} = {value_q}."],
        d_wrong,
        follow_through=f"part_a*part_b**3 + {paren_negative(a_coef)}*part_b**2 + {paren_negative(b_coef)}*part_b + "
        f"{paren_negative(c_coef)}",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_r2b_second_derivative_chain", level=17, presets={"exam": dict(tier="exam")})
def exd_x_r2b_second_derivative_chain(rng, tier: str = "exam"):
    """A cubic with an unknown coefficient: find k, the other stationary point, its nature and its value."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return draw(_build, rng)
