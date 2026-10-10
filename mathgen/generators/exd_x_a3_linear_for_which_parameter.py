"""Exam source for `senior-identify-and-solve-linear-equations`: make an equation linear, then solve it.

The practice tiers show a fixed equation and ask whether it is linear. Here the equation carries a parameter ``k``
and the learner must (1) expand both sides, (2) find the value of ``k`` that cancels the ``x^2`` terms, and (3) solve
the resulting linear equation. The answer is one option that states both results, e.g. ``k = 4; x = 5``.

Three shapes of equation (all with a single bracket pair on each side that produces the quadratic terms):

  * ``one``   -- ``(k*x + p)(x + q) = c*x^2 + e*x + f``
  * ``plus``  -- ``(x + p)(x + q) + k*x^2 = e*x + f``
  * ``both``  -- ``(k*x + p)(x + q) = (c*x + r)(x + s) + e*x + f``

About a quarter of seeds are a fourth shape, ``(kx + p)(kx + q) = -m*x^2 + e*x + f``, whose x^2 coefficient
``k^2 + m`` is never zero for real k; the key is then "No value of k makes the equation linear".

Built BACKWARDS from the integer solution ``x0`` and the integer ``k``; the key is re-derived with SymPy by expanding,
solving the ``x^2`` coefficient for ``k`` and then solving the remaining linear equation.
"""

from __future__ import annotations

import random

from sympy import Eq, Poly, Symbol, expand, solve

from ..core import check, fmt_signed, generator, join_terms, phrase, rational_text

_X, _K = Symbol("x"), Symbol("k")


def _sgn(v: int) -> str:
    return f"+ {v}" if v > 0 else f"- {-v}"


def _nz(rng: random.Random, lo: int, hi: int, *avoid: int) -> int:
    return rng.choice([v for v in range(lo, hi + 1) if v != 0 and v not in avoid])


def _draw_no_value(rng: random.Random) -> dict | None:
    """``(kx + p)(kx + q) = -m*x^2 + e*x + f``: the x^2 coefficient is k^2 + m = 0, which has no real k."""
    p, q, e, f = _nz(rng, -6, 6), _nz(rng, -6, 6), _nz(rng, -6, 6), _nz(rng, -9, 9)
    m = rng.randint(1, 5)
    full = expand((_K * _X + p) * (_K * _X + q) - (-m * _X**2 + e * _X + f))
    check(not any(v.is_real for v in solve(Poly(full, _X).nth(2), _K)), "a real k would make it linear")
    lhs_t = f"(kx {_sgn(p)})(kx {_sgn(q)})"
    rhs_t = join_terms([(-m, "x^2"), (e, "x"), (f, "")])
    right = "No value of k makes the equation linear"
    cands = []
    for kv in (m, -m, 0):
        c1, c0 = Poly(full, _X).nth(1).subs(_K, kv), Poly(full, _X).nth(0).subs(_K, kv)
        if c1 != 0:
            cands.append(f"k = {kv}; x = {rational_text(-c0 / c1)}")
    out = list(dict.fromkeys(cands))
    if len(out) < 3:
        return None
    text = phrase(
        rng,
        [
            "{eq}. For which value of k is this equation linear in x? Give that value of k and the solution for x.",
            "The equation {eq} contains a constant k. Find the value of k that makes it linear in x, and solve it for that k.",
        ],
        eq=f"{lhs_t} = {rhs_t}",
    )
    return dict(
        text=text,
        latex="plain:" + text,
        answer=right,
        answer_text=right,
        distractors=out[:3],
        working=["Expand both sides: the x^2 coefficient is k^2 + " + str(m) + ", which is never zero for real k."],
    )


def _draw(rng: random.Random) -> dict | None:
    if rng.random() < 0.25:
        return _draw_no_value(rng)
    shape = rng.choice(["one", "plus", "both"])
    x0 = _nz(rng, -7, 7)
    p, q = _nz(rng, -6, 6), _nz(rng, -6, 6)
    e = _nz(rng, -6, 6)
    if shape == "one":
        k = _nz(rng, -5, 6, 1, -1)
        lhs = (_K * _X + p) * (_X + q)
        rhs0 = k * _X**2 + e * _X
        lhs_t = f"(kx {_sgn(p)})(x {_sgn(q)})"
        wrong_k_pool = [-k, k + 1, k - 1]
    elif shape == "plus":
        k = -1
        lhs = (_X + p) * (_X + q) + _K * _X**2
        rhs0 = e * _X
        lhs_t = f"(x {_sgn(p)})(x {_sgn(q)}) + kx^2"
        wrong_k_pool = [1, 0, 2]
    else:
        k = _nz(rng, -5, 6, 1, -1)
        r, s = _nz(rng, -6, 6), _nz(rng, -6, 6)
        lhs = (_K * _X + p) * (_X + q)
        rhs0 = (k * _X + r) * (_X + s) + e * _X
        lhs_t = f"(kx {_sgn(p)})(x {_sgn(q)})"
        wrong_k_pool = [-k, k + 1, k - 1]
    # choose f so that the solution is x0 once k is substituted
    base = expand((lhs - rhs0).subs(_K, k))
    poly = Poly(base, _X)
    check(poly.degree() <= 1)
    b1, b0 = poly.nth(1), poly.nth(0)
    if b1 == 0:
        return None
    # b1*x + b0 = f  -> x0 = (f - b0)/b1  -> f = b1*x0 + b0
    f = int(b1 * x0 + b0)
    if f == 0:
        return None
    rhs = rhs0 + f
    if shape == "one":
        rhs_t = join_terms([(k, "x^2"), (e, "x"), (f, "")])
    elif shape == "plus":
        rhs_t = join_terms([(e, "x"), (f, "")])
    else:
        rhs_t = join_terms([(1, f"({fmt_signed(k, 'x')} {_sgn(r)})(x {_sgn(s)})"), (e, "x"), (f, "")])
    # Independent derivation: x^2 coefficient must vanish for exactly one k.
    full = expand(lhs - rhs)
    ks = solve(Poly(full, _X).nth(2), _K)
    check(ks == [k], f"parameter solve gave {ks}, expected {k}")
    xs = solve(Eq(full.subs(_K, k), 0), _X)
    check(xs == [x0], f"solution {xs} != {x0}")

    def x_for(kv: int):
        """The solution a learner gets with parameter kv if they simply ignore any x^2 terms left."""
        lin = Poly(full, _X)
        c1, c0 = lin.nth(1).subs(_K, kv), lin.nth(0).subs(_K, kv)
        return None if c1 == 0 else -c0 / c1

    right = f"k = {k}; x = {x0}"
    cands = []
    # right k, but b0 is added instead of subtracted when solving
    x_slip = (f + b0) / b1  # adds b0 instead of subtracting it
    cands.append(f"k = {k}; x = {rational_text(x_slip)}")
    for kw in wrong_k_pool:
        xv = x_for(kw)
        if kw != k and xv is not None:
            cands.append(f"k = {kw}; x = {rational_text(xv)}")
            break
    cands.append("No value of k makes the equation linear")  # wrong here: a real k always exists
    cands.append(f"k = {k}; x = {rational_text(-x0)}")
    out, seen = [], {right}
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    if len(out) < 3:
        return None
    text = phrase(
        rng,
        [
            "{eq}. For which value of k is this equation linear in x? Give that value of k and the solution for x.",
            "The equation {eq} contains a constant k. Find the value of k that makes it linear in x, and solve it for that k.",
        ],
        eq=f"{lhs_t} = {rhs_t}",
    )
    return dict(
        text=text,
        latex="plain:" + text,
        answer=right,
        answer_text=right,
        distractors=out[:3],
        working=["Expand both sides and set the x^2 coefficient to zero.", f"k = {k}, then x = {x0}"],
    )


@generator(
    "exd_x_a3_linear_for_which_parameter",
    level=11,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_a3_linear_for_which_parameter(rng: random.Random, tier: str = "exam") -> dict:
    """See the module docstring. Only the exam tier exists."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    while True:
        out = _draw(rng)
        if out is not None:
            return out
