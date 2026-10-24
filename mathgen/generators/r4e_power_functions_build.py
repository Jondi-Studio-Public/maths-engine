"""Generates `r4e_power_functions` questions (easy, medium, hard) and the exam chain `r4e_power_functions_chain`.

Skill `power-functions-rational-index` (answer `fraction`): power functions x^(p/q) with a rational index, where
q is the root and p the power, so x^(p/q) = (q-th root of x)^p.

* easy: evaluate x^(p/q) at a perfect q-th power, positive or negative index (and a negative input when q is odd, where
  the real root exists);
* medium: where a translated power function is defined and what it takes: the least x in the domain, or the least
  value in the range, of f(x) = (x - h)^(p/q) + k. For an even root the domain starts at h, for an even power over an odd
  root the range starts at k;
* hard: solve a x^(p/q) = c for the positive x (raise both sides to q/p), or solve x^(u) = k x^(v) for x > 0 by dividing;
* exam: y = a x^(p/q) through a point: the multiplier a, the domain, a value, and the x that gives a stated y (the last
  two following through from a).

Every key is verified with SymPy's exact powers and by substituting the answer back into the equation.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd

import sympy

from ..core import attempts, check, generator, phrase
from .r4e_common import Redraw, draw, fraction_part, inequality_part, num, single

x = sympy.Symbol("x")


_ROOT_NAMES = {2: "square root", 3: "cube root", 4: "fourth root", 5: "fifth root", 6: "sixth root"}


def _root_name(q: int) -> str:
    """The q-th root as it is said: "cube root", "fourth root" ("3-th root" is not English)."""
    return _ROOT_NAMES.get(q, f"{q}th root")


def fr(v) -> Fraction:
    v = sympy.Rational(v)
    return Fraction(int(v.p), int(v.q))


def index_text(p: int, q: int) -> str:
    return f"{p}/{q}" if q != 1 else str(p)


def _pw(base, e) -> str:
    """base^e written on paper: a power of 1 is just the base."""
    return str(base) if e == 1 else f"{base}^{e}"


def power_text(p: int, q: int, base: str = "x") -> str:
    return f"{base}^({p}/{q})"


def real_power(c, p: int, q: int):
    """c^(p/q) as the real value: the q-th root (real, so negative c needs odd q) raised to the power p."""
    root = sympy.real_root(sympy.Integer(c), q) if c < 0 else sympy.Integer(c) ** sympy.Rational(1, q)
    return root**p


_INDICES = [(p, q) for q in (2, 3, 4) for p in (1, 2, 3, 5, -1, -2, -3) if gcd(abs(p), q) == 1]


# -- easy ---------------------------------------------------------------------------------------------------------

_EASY_STEMS = (
    "The function f is defined by f(x) = {f}.",
    "Let f(x) = {f}.",
)


def _easy(rng):
    for _ in attempts(500, "an evaluation distinct from its distractors"):
        p, q = rng.choice(_INDICES)
        r = rng.choice([v for v in (2, 3, 4, 5, -2, -3) if v > 0 or q % 2 == 1])
        if q == 4 and r > 3:
            continue
        c = r**q
        if abs(c) > 700:
            continue
        key = sympy.Rational(r) ** p
        wrong = [
            (fr(r), "rational-index-ignores-power"),
            (
                fr(sympy.Integer(c) ** p) if abs(p) <= 3 and abs(c) ** abs(p) < 10**6 else None,
                "rational-index-ignores-root",
            ),
            (fr(sympy.Rational(c**p, q)) if p > 0 else None, "rational-index-divides-instead-of-roots"),
            (fr(1 / key), "rational-index-reciprocal-confused"),
        ]
        wrong = [(v, t) for v, t in wrong if v is not None and v != fr(key)]
        if len(wrong) >= 3:
            break
    check(real_power(c, p, q) == key, "the rational power disagrees")
    check(sympy.Rational(c) ** sympy.Rational(p, q) == key if c > 0 else True, "SymPy's power disagrees")
    stem = phrase(rng, _EASY_STEMS, f=power_text(p, q))
    working = [
        f"x^({p}/{q}) means the {_root_name(q)} of x, raised to the power {p}.",
        f"The {_root_name(q)} of {c} is {r}, and {_pw(r, p)} = {num(key)}.",
    ]
    return single(stem, fraction_part(f"Find f({c}).", fr(key), 2, working, wrong))


# -- medium -------------------------------------------------------------------------------------------------------


def _medium(rng):
    for _ in attempts(500, "a domain or range endpoint distinct from its distractors"):
        p, q = rng.choice([(p, q) for (p, q) in _INDICES if p > 0])
        h, k = rng.choice((-4, -3, -2, -1, 1, 2, 3, 5)), rng.choice((-6, -4, -2, -1, 1, 2, 3, 4))
        even_root = q % 2 == 0
        even_power_odd_root = p % 2 == 0 and q % 2 == 1
        if not (even_root or even_power_odd_root):
            continue
        ask = "domain" if (even_root and rng.random() < 0.5) else "range"
        if ask == "domain" and not even_root:
            continue
        key = h if ask == "domain" else k
        wrong = [
            (k if ask == "domain" else h, "rational-index-domain-range-confused"),
            (-h if ask == "domain" else -k, "rational-index-translation-sign"),
            (h + k, "rational-index-translation-combined"),
            (0, "rational-index-ignores-translation"),
        ]
        wrong = [(fr(v), t) for v, t in wrong if v != key]
        if len(wrong) >= 3 and h != k:
            break
    f_text = f"(x {'-' if h > 0 else '+'} {abs(h)})^({p}/{q}) {'+' if k > 0 else '-'} {abs(k)}"
    shifted = sympy.Rational(1, 2)
    if ask == "domain":
        # independent check: just below x = h the even root is of a negative number, so it is not real
        check(sympy.im(sympy.N((-shifted) ** sympy.Rational(p, q))) != 0, "defined to the left of the claimed domain")
        check(sympy.im(sympy.N(shifted ** sympy.Rational(p, q))) == 0, "not defined at the claimed domain")
        prompt = "What is the least value of x for which f(x) is defined as a real number?"
        working = [
            f"The {_root_name(q)} of (x {'-' if h > 0 else '+'} {abs(h)}) is real only when x {'-' if h > 0 else '+'} {abs(h)} ≥ 0.",
            f"So the least x is {h}.",
        ]
    else:
        # the base power is never negative and is 0 only at x = h, on both sides of it
        check(
            real_power(1, p, q) > 0 and real_power(-1, p, q) > 0 if q % 2 == 1 else real_power(1, p, q) > 0,
            "the power is not positive",
        )
        prompt = "What is the least value of f(x)?"
        working = [
            f"(x {'-' if h > 0 else '+'} {abs(h)})^({p}/{q}) is never negative, with least value 0 at x = {h}.",
            f"So the least value of f is 0 + ({k}) = {k}.",
        ]
    stem = phrase(rng, _EASY_STEMS, f=f_text)
    return single(stem, fraction_part(prompt, fr(key), 2, working, wrong))


# -- hard ---------------------------------------------------------------------------------------------------------


def _hard_equation(rng):
    for _ in attempts(500, "an equation with a distinct positive solution"):
        p, q = rng.choice(_INDICES)
        s = rng.choice((2, 3, 4, 5, 6)) if q < 4 else rng.choice((2, 3))
        a = rng.choice((1, 2, 3, 4, 5))
        x0 = s**q  # positive solution
        value = sympy.Rational(s) ** p
        c = a * value
        if x0 > 1300 or abs(c) > 400 or c.q > 400:
            continue
        not_inverted = (c / a) ** sympy.Rational(p, q)
        wrong = [
            (fr(c / a), "rational-index-ignores-the-index"),
            (fr(value), "rational-index-solves-for-the-value"),
            (fr(c), "rational-index-multiplier-not-divided"),
            (fr(1 / value) if p < 0 else None, "rational-index-solves-for-the-value"),
            (fr(not_inverted) if not_inverted.is_Rational else None, "rational-index-not-inverted"),
        ]
        wrong = [(v, t) for v, t in wrong if v is not None and v != Fraction(x0)]
        if len(wrong) >= 3:
            break
    check(a * sympy.Integer(x0) ** sympy.Rational(p, q) == c, "the stated solution does not satisfy the equation")
    which = "positive " if p % 2 == 0 else ""
    lhs = f"{'' if a == 1 else a}x^({p}/{q})"
    stem = f"Solve the equation {lhs} = {num(c)}."
    prompt = f"Find the {which}value of x."
    # The reciprocal index q/p, with the sign on the numerator and no "/1".
    inv_num, inv_den = (q, p) if p > 0 else (-q, -p)
    inv = str(inv_num) if inv_den == 1 else f"{inv_num}/{inv_den}"
    value_text = num(value)
    base = f"({value_text})" if "/" in value_text or value_text.startswith("-") else value_text
    working = [
        f"Divide by {a}: x^({p}/{q}) = {value_text}." if a != 1 else f"x^({p}/{q}) = {value_text}.",
        f"Raise both sides to the power {inv}: x = {base}^{inv if inv.isdigit() else f'({inv})'} = {x0}.",
    ]
    return single(stem, fraction_part(prompt, x0, 3, working, wrong))


def _hard_intersection(rng):
    for _ in attempts(500, "an intersection with a distinct solution"):
        n = rng.choice((2, 3))
        v = rng.choice((1, 2))
        kk = rng.choice((2, 3, 4, 5, 6)) if n == 2 else rng.choice((2, 3, 4))
        x0 = kk**n
        u_num = v * n + 1  # exponent u = v + 1/n = (v n + 1)/n
        wrong = [
            (fr(kk), "rational-index-root-not-undone"),
            (
                fr(sympy.Rational(kk) ** sympy.Rational(1, n))
                if sympy.Rational(kk) ** sympy.Rational(1, n) in sympy.S.Rationals
                else None,
                "rational-index-inverted-the-wrong-way",
            ),
            (fr(kk ** (n + 1)), "rational-index-power-too-high"),
            (fr(kk ** (v * n + 1)) if v * n + 1 != n + 1 else None, "rational-index-power-too-high"),
        ]
        wrong = [(w, t) for w, t in wrong if w is not None and w != Fraction(x0)]
        if len(wrong) >= 2 and x0 < 300:
            break
    xs = sympy.Integer(x0)
    check(xs ** sympy.Rational(u_num, n) == kk * xs**v, "the stated solution does not satisfy the equation")
    stem = f"Solve x^({u_num}/{n}) = {kk}{'x' if v == 1 else f'x^{v}'} for x > 0."
    working = [
        f"Divide both sides by {_pw('x', v)} (x > 0): x^(1/{n}) = {kk}.",
        f"Raise both sides to the power {n}: x = {_pw(kk, n)} = {x0}.",
    ]
    return single(stem, fraction_part("Find the value of x.", x0, 3, working, wrong))


@generator(
    "r4e_power_functions",
    level=17,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4e_power_functions(rng, tier: str = "easy"):
    """Power functions with a rational index: evaluate, domain and range, solve; see the module docstring."""
    if tier == "easy":
        return draw(_easy, rng)
    if tier == "medium":
        return draw(_medium, rng)
    if tier == "hard":
        return draw(_hard_equation if rng.random() < 0.6 else _hard_intersection, rng)
    raise ValueError(f"unknown tier {tier!r}")


# -- exam chain ---------------------------------------------------------------------------------------------------


def _chain(rng):
    for _ in attempts(500, "a multiplier and two follow-through values distinct from their distractors"):
        p, q = rng.choice([(p, q) for (p, q) in _INDICES if p > 0])
        a = rng.choice((2, 3, 4, 5))
        s = rng.choice((2, 3)) if q < 4 else 2
        x0, y0 = s**q, a * s**p
        r = rng.choice([v for v in (1, 2, 3, -1, -2, -3) if v > 0 or q % 2 == 1])
        c = r**q
        u = rng.choice((1, 2, 3))
        t_val = a * u**p
        if q == 4 and (abs(r) > 2 or u > 2):
            continue
        if abs(c) <= 400 and u**q <= 400 and u != s and c != x0 and y0 != t_val and r != 1:
            break
    f_c = a * r**p
    sol = u**q
    check(a * sympy.Integer(x0) ** sympy.Rational(p, q) == y0, "the point is not on the graph")
    check(a * real_power(c, p, q) == f_c, "f(c) disagrees")
    check(a * sympy.Integer(sol) ** sympy.Rational(p, q) == t_val, "the solution does not give t")
    stem = phrase(
        rng,
        (
            "The graph of y = a{f}, where a is a constant, passes through the point ({x0}, {y0}).",
            "A power function has the rule y = a{f}, where a is a constant. The point ({x0}, {y0}) lies on its graph.",
            "The curve y = a{f} goes through the point ({x0}, {y0}), where a is a constant.",
        ),
        f=power_text(p, q),
        x0=x0,
        y0=y0,
    )
    domain = [(0, None, True, False)] if q % 2 == 0 else [(None, None)]
    part_a = fraction_part(
        "Find the value of a.",
        a,
        2,
        [f"{x0}^({p}/{q}) = {_pw(s, p)} = {s**p}, so a = {y0}/{s**p} = {a}."],
        [
            (fr(sympy.Rational(y0, x0)), "rational-index-ignores-the-index"),
            (fr(sympy.Rational(y0, s)), "rational-index-ignores-power"),
            (fr(sympy.Rational(y0) * s**p), "rational-index-multiplier-not-divided"),
        ],
    )
    part_b = inequality_part(
        "State the largest set of real values of x for which the function is defined.",
        domain,
        1,
        [f"The {_root_name(q)} of x is real {'only for x ≥ 0' if q % 2 == 0 else 'for every x'}."],
        [
            ([(0, None, True, False)] if q % 2 else [(None, None)], "rational-index-domain-root-parity"),
            ([(0, None)], "rational-index-domain-endpoint-open-closed"),
            ([(None, 0, False, True)], "rational-index-domain-reversed"),
        ],
    )
    part_c = fraction_part(
        f"Find the value of y when x = {c}.",
        f_c,
        2,
        [f"{c}^({p}/{q}) = {_pw(f'({r})', p)} = {r**p}, so y = {a} × {r**p} = {f_c}."],
        [
            (fr(a * c), "rational-index-ignores-the-index"),
            (fr(a * c**p) if abs(c) ** p < 10**6 else None, "rational-index-ignores-root"),
            (fr(a + r**p), "rational-index-multiplier-added"),
        ],
        follow_through=f"part_a*{r**p}",
    )
    part_d = fraction_part(
        f"Find the positive value of x for which y = {t_val}.",
        sol,
        3,
        [
            f"{a}x^({p}/{q}) = {t_val} gives x^({p}/{q}) = {u**p}, so x = ({u**p})^{f'({q}/{p})' if p != 1 else q} = {sol}."
        ],
        [
            (fr(sympy.Rational(t_val, a)), "rational-index-solves-for-the-value"),
            (fr(t_val), "rational-index-multiplier-not-divided"),
            (fr(u), "rational-index-root-not-undone"),
        ],
        follow_through=f"({t_val}/part_a)**({q}/{p})",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


@generator("r4e_power_functions_chain", level=17, presets={"exam": dict(tier="exam")})
def r4e_power_functions_chain(rng, tier: str = "exam"):
    """y = a x^(p/q) through a point: a, the domain, a value, and the x giving a stated y."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return draw(_chain, rng)
