"""Cumulative distribution functions of polynomial densities: `r3f_pdf_cdf` and its exam chain.

Skill `pdf-cdf-from-polynomial`. A continuous random variable X has a density f that is a polynomial on [lo, hi] and
zero elsewhere. Its cumulative distribution function is F(x) = integral of f from lo to x, and
P(a < X < b) = F(b) - F(a).

Tiers (topic `r3f_pdf_cdf`), each with a different piece of reasoning:

* easy   -- from a stated density to F(x); from a stated F(x) to a probability P(a < X < b); from a stated F(x) back to
            the density by differentiating;
* medium -- the density has an unknown constant k: find k, then build F(x), or find P(X > c) through F, or a
            conditional probability P(X < c | X > a) = (F(c) - F(a)) / (1 - F(a));
* hard   -- a quantile: the median, a stated percentile or the interquartile range, by solving F(m) = p. The equation is
            a polynomial one and the root is found to 2 decimal places.

The exam chain (topic `exd_x_r3f_pdf_cdf`) takes f(x) = k (x - a)^n on [a, b]: (a) k, (b) F(x), (c) P(c < X < d)
through F, (d) the median from F(m) = 1/2. Parts (b) to (d) carry `follow_through`.

Every density is built from a shape polynomial g and k = 1 / (integral of g). F is integrated exactly with SymPy; the
probabilities are re-derived by exact Fractions from the coefficients, and every quantile by bisection on F (a route
that uses no formula for the root), then confirmed by evaluating F at the rounded root.
"""

from __future__ import annotations

import random
from fractions import Fraction

import sympy

from ..core import attempts, check, generator
from ._r3f_kit import (
    X,
    build_until,
    clear_of_boundary,
    expression_part,
    formula_text,
    fraction_part,
    number_part,
    single,
    single_expression,
)

# -- polynomial arithmetic in exact Fractions -----------------------------------------------------------------


def _coeffs(expr) -> list[Fraction]:
    """Coefficients of a polynomial in x, constant term first."""
    poly = sympy.Poly(sympy.expand(expr), X)
    return [Fraction(int(c.p), int(c.q)) for c in reversed(poly.all_coeffs())]


def _eval(coeffs: list[Fraction], x: Fraction) -> Fraction:
    total = Fraction(0)
    for c in reversed(coeffs):
        total = total * x + c
    return total


def _bisect(coeffs: list[Fraction], lo: Fraction, hi: Fraction, target: Fraction) -> Fraction:
    """x in [lo, hi] with F(x) = target for an increasing polynomial F, to within 10^-12."""
    low, high = Fraction(lo), Fraction(hi)
    for _ in range(60):
        mid = (low + high) / 2
        if _eval(coeffs, mid) < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def _integrate(expr, lo=None, hi=None):
    """Integral of a polynomial in x: the antiderivative with no constant, or its change from lo to hi. Exact, and far
    cheaper than a general symbolic integral."""
    anti = sympy.Poly(sympy.expand(expr), X).integrate().as_expr()
    if lo is None:
        return anti
    return anti.subs(X, hi) - anti.subs(X, lo)


class Density:
    """A polynomial density k g(x) on [lo, hi]."""

    def __init__(self, g, lo: int, hi: int, name: str):
        self.g_raw, self.g, self.lo, self.hi, self.name = g, sympy.expand(g), lo, hi, name
        total = _integrate(self.g, lo, hi)
        check(total > 0, "the shape has no area")
        self.k = 1 / total
        self.f = sympy.expand(self.k * self.g)
        self.F = sympy.expand(_integrate(self.f, lo, X))
        check(self.F.subs(X, lo) == 0 and self.F.subs(X, hi) == 1, "F does not run from 0 to 1")
        check(sympy.diff(self.F, X) - self.f == 0, "F' is not f")
        check(all(self.f.subs(X, lo + sympy.Rational(i, 8) * (hi - lo)) >= 0 for i in range(9)), "negative density")
        self.fc, self.Fc = _coeffs(self.f), _coeffs(self.F)

    def cdf(self, x) -> Fraction:
        return _eval(self.Fc, Fraction(x))

    def prob(self, a, b) -> Fraction:
        return self.cdf(b) - self.cdf(a)

    def quantile(self, p: Fraction) -> Fraction:
        m = _bisect(self.Fc, Fraction(self.lo), Fraction(self.hi), Fraction(p))
        check(abs(self.cdf(m) - p) < Fraction(1, 10**9), "quantile does not satisfy F(m) = p")
        return m

    def mean(self) -> Fraction:
        value = _integrate(X * self.f, self.lo, self.hi)
        return Fraction(int(value.p), int(value.q))


_DENSITIES: dict = {}


def _cached_density(g, lo: int, hi: int, name: str) -> Density:
    """A Density is built from exact algebra that costs tens of milliseconds, and the same few shapes recur, so keep them."""
    key = (str(g), lo, hi, name)
    if key not in _DENSITIES:
        _DENSITIES[key] = Density(g, lo, hi, name)
    return _DENSITIES[key]


def _shapes(lo: int, hi: int):
    w = hi - lo
    return {
        "rising": ((X - lo) ** 2, "power"),
        "rising3": ((X - lo) ** 3, "power"),
        "linear": ((X - lo), "power"),
        "falling": ((hi - X) ** 2, "falling"),
        "falling-linear": ((hi - X), "falling"),
        "arch": ((X - lo) * (hi - X), "arch"),
        "ramp": ((X - lo) + w, "ramp"),
    }


_NAMES_EASY = ("rising", "rising3", "linear", "falling", "falling-linear", "arch", "ramp")


def _density(rng: random.Random, names=_NAMES_EASY, lo_choices=(0,), span=(2, 3, 4, 5)) -> Density:
    lo = rng.choice(lo_choices)
    hi = lo + rng.choice(span)
    name = rng.choice(names)
    g, _family = _shapes(lo, hi)[name]
    return _cached_density(g, lo, hi, name)


def _top_level_sum(text: str) -> bool:
    depth = 0
    for i, ch in enumerate(text):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and text[i : i + 3] in (" + ", " - "):
            return True
    return False


def _density_text(d: Density, with_k: bool) -> str:
    body = formula_text(d.f)
    if with_k:
        raw = str(d.g_raw).replace("**", "^").replace("*", "")
        body = f"k({raw})" if _top_level_sum(raw) else f"k{raw}"
    return f"f(x) = {body} for {d.lo} ≤ x ≤ {d.hi}, and f(x) = 0 otherwise"


def _cdf_text(d: Density) -> str:
    return f"F(x) = 0 for x < {d.lo}, F(x) = {formula_text(d.F)} for {d.lo} ≤ x ≤ {d.hi}, and F(x) = 1 for x > {d.hi}"


def _inside_pair(rng: random.Random, d: Density) -> tuple[int, int]:
    for _ in attempts(100, "two interior bounds"):
        a = rng.randint(d.lo, d.hi - 1)
        b = rng.randint(a + 1, d.hi)
        if (a, b) != (d.lo, d.hi) and 0 < d.prob(a, b) < 1:
            return a, b
    raise AssertionError("unreachable")


# -- easy ------------------------------------------------------------------------------------------------------


def _easy_cdf(rng: random.Random) -> dict:
    d = _density(rng, lo_choices=(0, 1))
    stem = f"A continuous random variable X has probability density function {_density_text(d, False)}."
    slips = [
        (d.f, "r3f-cdf-left-as-the-density"),
        (_integrate(d.f), "r3f-cdf-omits-the-lower-limit"),
        (_integrate(d.f) / 2, "r3f-cdf-wrong-integration-constant"),
        (sympy.diff(d.f, X), "r3f-cdf-differentiates-the-density"),
        (d.F + sympy.Rational(1, 2), "r3f-cdf-wrong-integration-constant"),
    ]
    part = expression_part(
        f"Find the cumulative distribution function F(x) for {d.lo} ≤ x ≤ {d.hi}.",
        d.F,
        2,
        [f"F(x) is the integral of f(t) from {d.lo} to x.", f"F(x) = {formula_text(d.F)}."],
        slips,
    )
    return single(f"{stem} {part['prompt']}", part)


def _easy_prob(rng: random.Random) -> dict:
    d = _density(rng, names=("rising", "rising3", "linear", "falling-linear", "arch"), lo_choices=(0, 1, 2))
    a, b = _inside_pair(rng, d)
    p = d.prob(a, b)
    part = fraction_part(
        f"Find P({a} < X < {b}).",
        p,
        2,
        [f"P({a} < X < {b}) = F({b}) - F({a}) = {d.cdf(b)} - {d.cdf(a)}."],
        [
            (d.cdf(b), "r3f-cdf-probability-uses-upper-value-only"),
            (1 - p, "r3f-cdf-gives-complement"),
            (d.cdf(a), "r3f-cdf-probability-uses-lower-value-only"),
            (d.cdf(b) + d.cdf(a), "r3f-cdf-adds-the-values"),
            (Fraction(b - a, d.hi - d.lo), "r3f-cdf-assumes-uniform"),
        ],
        sensible_range=(0, 1),
    )
    stem = f"The cumulative distribution function of a continuous random variable X is {_cdf_text(d)}. {part['prompt']}"
    return single_expression(stem, part)


def _easy_pdf(rng: random.Random) -> dict:
    d = _density(rng, names=("rising", "rising3", "falling", "arch"), lo_choices=(0, 1, 2))
    stem = f"The cumulative distribution function of a continuous random variable X is {_cdf_text(d)}."
    part = expression_part(
        f"Find the probability density function f(x) for {d.lo} < x < {d.hi}.",
        d.f,
        2,
        [
            "The density is the derivative of the cumulative distribution function: f(x) = F'(x).",
            f"f(x) = {formula_text(d.f)}.",
        ],
        [
            (_integrate(d.F), "r3f-pdf-integrates-the-cdf"),
            (d.F, "r3f-pdf-left-as-the-cdf"),
            (d.f * 2, "r3f-pdf-derivative-doubled"),
            (sympy.diff(d.f, X), "r3f-pdf-differentiates-twice"),
        ],
    )
    return single(f"{stem} {part['prompt']}", part)


def _easy(rng: random.Random) -> dict:
    return rng.choice((_easy_cdf, _easy_prob, _easy_pdf))(rng)


# -- medium ----------------------------------------------------------------------------------------------------


def _medium_density(rng: random.Random) -> Density:
    return _density(rng, names=("rising", "rising3", "falling", "arch", "ramp"), lo_choices=(0, 1, 2), span=(2, 3, 4))


def _medium_cdf(rng: random.Random) -> dict:
    d = _medium_density(rng)
    stem = f"A continuous random variable X has probability density function {_density_text(d, True)}, where k is a constant."
    k_less = _integrate(d.g)  # F with k left out
    part = expression_part(
        f"Find the cumulative distribution function F(x) for {d.lo} ≤ x ≤ {d.hi}.",
        d.F,
        3,
        [
            f"The total area is 1, so k = {d.k}.",
            f"F(x) is the integral of f(t) from {d.lo} to x, so F(x) = {formula_text(d.F)}.",
        ],
        [
            (d.k * k_less, "r3f-cdf-omits-the-lower-limit"),
            (k_less, "r3f-cdf-forgets-k"),
            (d.f, "r3f-cdf-left-as-the-density"),
            (_integrate(d.g, d.lo, X), "r3f-cdf-forgets-k"),
        ],
    )
    return single(f"{stem} {part['prompt']}", part)


def _medium_tail(rng: random.Random) -> dict:
    d = _medium_density(rng)
    c = rng.randint(d.lo + 1, d.hi - 1)
    p = 1 - d.cdf(c)
    stem = f"A continuous random variable X has probability density function {_density_text(d, True)}, where k is a constant."
    part = fraction_part(
        f"Find P(X > {c}).",
        p,
        3,
        [f"k = {d.k}, so F(x) = {formula_text(d.F)}.", f"P(X > {c}) = 1 - F({c}) = 1 - {d.cdf(c)}."],
        [
            (d.cdf(c), "r3f-cdf-reports-lower-tail"),
            (Fraction(d.hi - c, d.hi - d.lo), "r3f-cdf-assumes-uniform"),
            (p / 2, "r3f-cdf-halves-the-tail"),
            (sympy_to_fraction(_integrate(d.g, c, d.hi)), "r3f-cdf-forgets-k"),
        ],
        sensible_range=(0, 1),
    )
    return single_expression(f"{stem} {part['prompt']}", part)


def sympy_to_fraction(value) -> Fraction:
    value = sympy.nsimplify(value)
    return Fraction(int(value.p), int(value.q))


def _medium_conditional(rng: random.Random) -> dict:
    for _ in attempts(200, "a conditional probability that is not trivial"):
        d = _medium_density(rng)
        if d.hi - d.lo < 3:
            continue
        a = rng.randint(d.lo + 1, d.hi - 2)
        c = rng.randint(a + 1, d.hi - 1)
        p = (d.cdf(c) - d.cdf(a)) / (1 - d.cdf(a))
        if 0 < p < 1:
            break
    stem = f"A continuous random variable X has probability density function {_density_text(d, True)}, where k is a constant."
    part = fraction_part(
        f"Find P(X < {c} | X > {a}).",
        p,
        3,
        [
            f"k = {d.k}, so F(x) = {formula_text(d.F)}.",
            f"P(X < {c} | X > {a}) = (F({c}) - F({a})) / (1 - F({a})) = {p}.",
        ],
        [
            (d.cdf(c) - d.cdf(a), "r3f-cdf-conditional-not-rescaled"),
            (d.cdf(c), "r3f-cdf-conditional-ignores-the-condition"),
            ((d.cdf(c) - d.cdf(a)) / d.cdf(c), "r3f-cdf-conditional-divides-by-the-wrong-event"),
            (1 - p, "r3f-cdf-gives-complement"),
        ],
        sensible_range=(0, 1),
    )
    return single_expression(f"{stem} {part['prompt']}", part)


def _medium(rng: random.Random) -> dict:
    return rng.choice((_medium_cdf, _medium_tail, _medium_conditional))(rng)


# -- hard ------------------------------------------------------------------------------------------------------


def _hard_density(rng: random.Random) -> Density:
    return _density(
        rng, names=("rising", "rising3", "falling", "arch", "falling-linear"), lo_choices=(0, 1, 2), span=(3, 4, 5, 6)
    )


def _hard_median(rng: random.Random) -> dict:
    for _ in attempts(500, "a median clear of a rounding boundary"):
        d = _hard_density(rng)
        m = d.quantile(Fraction(1, 2))
        if clear_of_boundary(m, 2, Fraction(1, 5)) and abs(m - Fraction(d.lo + d.hi, 2)) > Fraction(1, 5):
            break
    stem = f"A continuous random variable X has probability density function {_density_text(d, False)}."
    part = number_part(
        "Find the median of X, correct to 2 decimal places.",
        m,
        2,
        3,
        [f"F(m) = 1/2 where F(x) = {formula_text(d.F)}.", f"Solving gives m = {float(m):.2f}."],
        [
            (Fraction(d.lo + d.hi, 2), "r3f-median-taken-as-midpoint"),
            (d.mean(), "r3f-median-taken-as-mean"),
            (d.quantile(Fraction(1, 4)), "r3f-median-uses-a-quarter"),
            (d.quantile(Fraction(3, 4)), "r3f-median-uses-three-quarters"),
        ],
        sensible_range=(d.lo, d.hi),
    )
    return single_expression(f"{stem} {part['prompt']}", part)


def _hard_percentile(rng: random.Random) -> dict:
    for _ in attempts(500, "a percentile clear of a rounding boundary"):
        d = _hard_density(rng)
        tail = rng.choice((Fraction(1, 10), Fraction(1, 5), Fraction(1, 4), Fraction(3, 10)))
        c = d.quantile(1 - tail)
        if clear_of_boundary(c, 2, Fraction(1, 5)):
            break
    stem = f"A continuous random variable X has probability density function {_density_text(d, False)}."
    part = number_part(
        f"Find the value c for which P(X > c) = {float(tail):g}, correct to 2 decimal places.",
        c,
        2,
        3,
        [
            f"P(X > c) = 1 - F(c) = {float(tail):g}, so F(c) = {float(1 - tail):g}.",
            f"Solving gives c = {float(c):.2f}.",
        ],
        [
            (d.quantile(tail), "r3f-percentile-uses-the-lower-tail"),
            (d.quantile(Fraction(1, 2)), "r3f-percentile-uses-the-median"),
            (d.lo + (d.hi - d.lo) * (1 - tail), "r3f-percentile-assumes-uniform"),
        ],
        sensible_range=(d.lo, d.hi),
    )
    return single_expression(f"{stem} {part['prompt']}", part)


def _hard_iqr(rng: random.Random) -> dict:
    for _ in attempts(500, "an interquartile range clear of a rounding boundary"):
        d = _hard_density(rng)
        q1, q3 = d.quantile(Fraction(1, 4)), d.quantile(Fraction(3, 4))
        iqr = q3 - q1
        if clear_of_boundary(iqr, 2, Fraction(1, 5)) and clear_of_boundary(q1, 2, Fraction(1, 5)):
            break
    stem = f"A continuous random variable X has probability density function {_density_text(d, False)}."
    part = number_part(
        "Find the interquartile range of X, correct to 2 decimal places.",
        iqr,
        2,
        3,
        [
            f"Solve F(Q1) = 1/4 and F(Q3) = 3/4: Q1 = {float(q1):.2f}, Q3 = {float(q3):.2f}.",
            f"IQR = Q3 - Q1 = {float(iqr):.2f}.",
        ],
        [
            (q3, "r3f-iqr-reports-the-upper-quartile"),
            (Fraction(d.hi - d.lo, 2), "r3f-iqr-assumes-uniform"),
            (d.quantile(Fraction(1, 2)) - q1, "r3f-iqr-uses-the-median-as-an-end"),
            (q3 + q1, "r3f-iqr-adds-the-quartiles"),
        ],
        sensible_range=(0, d.hi - d.lo),
    )
    return single_expression(f"{stem} {part['prompt']}", part)


def _hard(rng: random.Random) -> dict:
    return rng.choice((_hard_median, _hard_percentile, _hard_iqr))(rng)


@generator(
    "r3f_pdf_cdf",
    level=13,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3f_pdf_cdf(rng: random.Random, tier: str = "easy") -> dict:
    """Polynomial pdf and cdf: build F (easy), unknown constant and conditioning (medium), quantiles (hard)."""
    if tier == "easy":
        return build_until(rng, _easy)
    if tier == "medium":
        return build_until(rng, _medium)
    if tier == "hard":
        return build_until(rng, _hard)
    raise ValueError(f"unknown tier {tier!r}")


# -- exam chain ------------------------------------------------------------------------------------------------


def _exam(rng: random.Random) -> dict:
    for _ in attempts(500, "a density with a median clear of a rounding boundary"):
        a = rng.choice((0, 1, 2, 3))
        w = rng.choice((3, 4, 5, 6))
        n = rng.choice((1, 2, 3))
        b = a + w
        k = Fraction(n + 1, w ** (n + 1))
        d = _cached_density((X - a) ** n, a, b, "exam")
        check(sympy_to_fraction(d.k) == k, "k disagrees with the closed form")
        c1 = rng.randint(a, b - 1)
        c2 = rng.randint(c1 + 1, b)
        if (c1, c2) == (a, b):
            continue
        median = d.quantile(Fraction(1, 2))
        if clear_of_boundary(median, 2, Fraction(1, 5)):
            break
    p = d.prob(c1, c2)
    power = f"^{n}" if n > 1 else ""
    shape = f"k(x - {a}){power}" if a else f"kx{power}"
    base = f"(x - {a})" if a else "x"
    stem = (
        f"A continuous random variable X has probability density function f(x) = {shape} for {a} ≤ x ≤ {b}, "
        "and f(x) = 0 otherwise, where k is a constant."
    )
    antider_no_k = _integrate((X - a) ** n, a, X)
    parts = [
        fraction_part(
            "Find the value of k.",
            k,
            2,
            [f"The total area is 1, so k × (integral of the shape from {a} to {b}) = 1.", f"k = {k}."],
            [
                (Fraction(1, b - a), "r3f-cdf-assumes-uniform"),
                (Fraction(n + 1, b ** (n + 1)) if a else Fraction(n, w**n), "r3f-k-power-not-raised"),
                (Fraction(w ** (n + 1), n + 1), "r3f-k-inverted"),
            ],
            sensible_range=(0, None),
        ),
        expression_part(
            f"Hence find the cumulative distribution function F(x) for {a} ≤ x ≤ {b}.",
            d.F,
            2,
            [f"F(x) is the integral of f(t) from {a} to x.", f"F(x) = {formula_text(d.F)}."],
            [
                (antider_no_k, "r3f-cdf-forgets-k"),
                (d.f, "r3f-cdf-left-as-the-density"),
                (
                    sympy.Rational(k.numerator, k.denominator) * ((X - a) ** (n + 1)) / n,
                    "r3f-cdf-divides-by-n-not-n-plus-one",
                ),
                (sympy.Rational(k.numerator, k.denominator) * X ** (n + 1) / (n + 1), "r3f-cdf-omits-the-lower-limit"),
            ],
            follow_through=f"part_a*({base})**({n + 1})/{n + 1}",
        ),
        fraction_part(
            f"Hence find P({c1} < X < {c2}).",
            p,
            2,
            [f"P = F({c2}) - F({c1}) = {d.cdf(c2)} - {d.cdf(c1)}."],
            [
                (d.cdf(c2), "r3f-cdf-probability-uses-upper-value-only"),
                (1 - p, "r3f-cdf-gives-complement"),
                (d.cdf(c1), "r3f-cdf-probability-uses-lower-value-only"),
                (Fraction(c2 - c1, b - a), "r3f-cdf-assumes-uniform"),
            ],
            follow_through=f"part_a*(({c2}-{a})**{n + 1}-({c1}-{a})**{n + 1})/{n + 1}",
            sensible_range=(0, 1),
        ),
        number_part(
            "Find the median of X, correct to 2 decimal places.",
            median,
            2,
            3,
            ["The median m satisfies F(m) = 1/2.", f"m = {float(median):.2f}."],
            [
                (Fraction(a + b, 2), "r3f-median-taken-as-midpoint"),
                (d.mean(), "r3f-median-taken-as-mean"),
                (d.quantile(Fraction(1, 4)), "r3f-median-uses-a-quarter"),
            ],
            follow_through=f"{a}+((({n + 1})/(2*part_a))**(1/{n + 1}))",
            sensible_range=(a, b),
        ),
    ]
    return dict(text=stem, parts=parts)


@generator("exd_x_r3f_pdf_cdf", level=13, presets={"exam": dict(tier="exam")})
def exd_x_r3f_pdf_cdf(rng: random.Random, tier: str = "exam") -> dict:
    """k, the cumulative distribution function, a probability through it, then the median."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return build_until(rng, _exam)
