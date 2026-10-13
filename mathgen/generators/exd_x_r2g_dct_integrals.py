"""Exam question: dominated convergence on a three-point measure space (`ucv1-dominated-convergence-check`).

The sample space is {x1, x2, x3} with stated masses mu({xi}) = wi. The functions are f_n(xi) = Li + ci/n for
n = 1, 2, 3, .... The parts chain:

* (a) the integral of the pointwise limit f, the sum of wi Li (2 marks);
* (b) the integral of the smallest dominating function g(xi) = sup over n of |f_n(xi)|, which is finite, so the
  dominated-convergence hypotheses hold (2 marks);
* (c) the integral of f_n0 for a stated n0 (a fraction, 1 mark);
* (d) the smallest n with |integral f_n - integral f| < epsilon, using the gap at n0, which is the sum of wi ci
  divided by n (follow-through from (a) and (c); 2 marks).

The supremum of |Li + ci/n| over n >= 1 is max(|Li|, |Li + ci|) because Li + ci/n is monotone in n.
"""

from __future__ import annotations

import random
from fractions import Fraction
from math import floor

from ..core import check, generator
from .exd_x_r2g_support import build_part, latex_with_sets, redraws


def _term(coef: int, offset: int) -> str:
    return f"{coef} {'+' if offset > 0 else '-'} {abs(offset)}/n"


def _draw(rng: random.Random) -> dict:
    for _ in range(1000):
        w = [rng.randint(1, 5) for _ in range(3)]
        limits = [rng.choice([-1, 1]) * rng.randint(1, 6) for _ in range(3)]
        offsets = [rng.choice([-1, 1]) * rng.randint(1, 6) for _ in range(3)]
        n0 = rng.randint(2, 6)
        eps = Fraction(rng.randint(1, 9), rng.choice([10, 20, 50, 100]))
        total_c = sum(wi * ci for wi, ci in zip(w, offsets))
        if total_c == 0 or len(set(w)) == 1:
            continue
        n = floor(abs(Fraction(total_c)) / eps) + 1
        if not 4 <= n <= 120 or n <= n0:
            continue
        limit_int = sum(wi * li for wi, li in zip(w, limits))
        if limit_int == 0:
            continue
        sup_sum = sum(wi * max(abs(li), abs(li + ci)) for wi, li, ci in zip(w, limits, offsets))
        if sup_sum in (
            sum(wi * abs(li) for wi, li in zip(w, limits)),
            sum(wi * abs(li + ci) for wi, li, ci in zip(w, limits, offsets)),
        ):
            continue  # the supremum must sit at n = 1 for some points and at the limit for others
        return dict(w=w, limits=limits, offsets=offsets, n0=n0, eps=eps, total_c=total_c, n=n)
    check(False, "no acceptable measure-space draw")


_STEMS = [
    "On the three-point space {{x1, x2, x3}} with masses mu({{x1}}) = {w1}, mu({{x2}}) = {w2} and mu({{x3}}) = {w3}, "
    "let f_n(x1) = {t1}, f_n(x2) = {t2} and f_n(x3) = {t3} for n = 1, 2, 3, ....",
    "Let mu be the measure on {{x1, x2, x3}} with mu({{x1}}) = {w1}, mu({{x2}}) = {w2}, mu({{x3}}) = {w3}. A "
    "sequence of functions is given by f_n(x1) = {t1}, f_n(x2) = {t2}, f_n(x3) = {t3}, for n = 1, 2, 3, ....",
    "Three points x1, x2, x3 carry masses {w1}, {w2} and {w3}. For n = 1, 2, 3, ... define f_n(x1) = {t1}, "
    "f_n(x2) = {t2} and f_n(x3) = {t3}.",
]


@generator("exd_x_r2g_dct_integrals", level=17, presets={"exam": dict(tier="exam")})
@redraws
def exd_x_r2g_dct_integrals(rng: random.Random, tier: str = "exam") -> dict:
    """Integral of the limit, of the smallest dominating function, of f_n0, and the n that makes the gap small."""
    check(tier == "exam", "exd_x_r2g_dct_integrals: only the exam tier exists")
    d = _draw(rng)
    w, lim, off, n0, eps, c_total, n = (d[k] for k in ("w", "limits", "offsets", "n0", "eps", "total_c", "n"))
    integral_f = sum(wi * li for wi, li in zip(w, lim))
    integral_g = sum(wi * max(abs(li), abs(li + ci)) for wi, li, ci in zip(w, lim, off))
    integral_fn0 = Fraction(integral_f) + Fraction(c_total, n0)
    gap0 = abs(integral_fn0 - integral_f)
    stem = rng.choice(_STEMS).format(
        w1=w[0], w2=w[1], w3=w[2], t1=_term(lim[0], off[0]), t2=_term(lim[1], off[1]), t3=_term(lim[2], off[2])
    )
    eps_s = f"{float(eps):g}"
    eps_ft = f"({eps.numerator}/{eps.denominator})"
    unweighted_c = sum(off)
    parts = [
        build_part(
            "Find the integral of the pointwise limit f = lim f_n with respect to mu.",
            integral_f,
            kind="fraction",
            marks=2,
            working=[
                f"f(xi) = {lim[0]}, {lim[1]}, {lim[2]} at x1, x2, x3.",
                f"Integral = sum of mass x value = {integral_f}.",
            ],
            wrong=[
                (sum(lim), "ignores-the-measure-weights"),
                (sum(wi * (li + ci) for wi, li, ci in zip(w, lim, off)), "first-term-as-limit"),
                (sum(wi * abs(li) for wi, li in zip(w, lim)), "integrates-absolute-value-of-limit"),
                (integral_f + c_total, "limit-includes-constant-term"),
                (-integral_f, "answer-sign-reversed"),
                (Fraction(integral_f, sum(w)), "divides-by-total-mass"),
            ],
        ),
        build_part(
            "The smallest function g with |f_n| <= g for every n is integrable, so dominated convergence applies. "
            "Find the integral of g with respect to mu.",
            integral_g,
            kind="fraction",
            marks=2,
            working=[
                "g(xi) = max(|Li|, |Li + ci|), since Li + ci/n is monotone in n.",
                f"Integral = {integral_g}.",
            ],
            wrong=[
                (sum(wi * abs(li) for wi, li in zip(w, lim)), "bound-from-limit-only"),
                (sum(wi * abs(li + ci) for wi, li, ci in zip(w, lim, off)), "bound-from-first-term-only"),
                (sum(wi * (abs(li) + abs(ci)) for wi, li, ci in zip(w, lim, off)), "bound-not-the-smallest"),
                (sum(max(abs(li), abs(li + ci)) for li, ci in zip(lim, off)), "ignores-the-measure-weights"),
                (Fraction(integral_g, sum(w)), "divides-by-total-mass"),
            ],
        ),
        build_part(
            f"Find the integral of f_{n0} with respect to mu, as a fraction in lowest terms if it is not a whole "
            "number.",
            integral_fn0,
            kind="fraction",
            marks=1,
            working=[f"Integral of f_{n0} = {integral_f} + ({c_total})/{n0} = {integral_fn0}."],
            wrong=[
                (Fraction(integral_f), "limit-value-for-nth-term"),
                (integral_f + Fraction(unweighted_c, n0), "ignores-the-measure-weights"),
                (integral_f + c_total * n0, "reads-1-over-n-as-n"),
                (integral_f + Fraction(c_total, n0 - 1), "off-by-one-count"),
            ],
        ),
        build_part(
            f"By dominated convergence the integral of f_n tends to the integral of f. Using your answers to (a) and "
            f"(c), find the smallest n for which the integral of f_n differs from the integral of f by less than "
            f"{eps_s}.",
            n,
            kind="number",
            marks=2,
            sensible_range=(0, None),
            follow_through=f"floor({n0}*Abs(part_c - part_a)/{eps_ft}) + 1",
            working=[
                f"The gap at n is {abs(c_total)}/n (it was {gap0} at n = {n0}).",
                f"{abs(c_total)}/n < {eps_s} needs n > {Fraction(abs(c_total)) / eps}, so n = {n}.",
            ],
            wrong=[
                (n - 1, "off-by-one-count"),
                (n + 1, "off-by-one-count"),
                (floor(Fraction(abs(unweighted_c)) / eps) + 1 if unweighted_c else None, "ignores-the-measure-weights"),
                (floor(Fraction(abs(c_total)) / (eps * n0)) + 1, "gap-taken-as-already-divided-by-n"),
            ],
        ),
    ]
    return dict(text=stem, latex=latex_with_sets(stem), parts=parts)
