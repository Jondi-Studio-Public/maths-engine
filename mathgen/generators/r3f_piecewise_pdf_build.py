"""Piecewise, trigonometric and exponential densities: `r3f_piecewise_pdf` and its exam chain.

Skill `piecewise-pdf-quantile`. Every answer is a decimal, stated to a fixed number of places.

Three families of density, each with an unknown constant k fixed by "total area is 1":

* triangle -- rises linearly from 0 at x = 0 to a peak at the mode c, then falls linearly to 0 at x = d (the pieces are
  written out, so the learner reads two formulae); k = 2 / d;
* sine     -- f(x) = k sin(pi x / L) on [0, L]; k = pi / (2 L);
* truncated exponential -- f(x) = k e^(-lambda x) on [0, b]; k = lambda / (1 - e^(-lambda b)).

Tiers (topic `r3f_piecewise_pdf`), each with a different piece of reasoning:

* easy   -- the triangle: k from the total area; P(X <= t) as the area of a triangle; E(X) = (c + d) / 3;
* medium -- the triangle's median (which piece it lies on depends on the mode), P(t1 < X < t2) across the mode; the
            sine density's k and a probability;
* hard   -- a quantile or mean that needs real integration: the truncated exponential's probability, median and mean
            (integration by parts), the triangle's upper percentile on the falling piece, the sine density's percentile.

The exam chain (topic `exd_x_r3f_piecewise_pdf`) is on a triangle: (a) k, (b) P(X <= t) on the rising piece, (c) the
median (it lies on the rising piece), (d) P(t1 < X < t2) across the mode. Parts (b) to (d) carry `follow_through`.

Each key is computed in closed form and re-derived by numerical integration of the density (Simpson's rule on a fine
grid, 1e-7 accuracy) so a wrong closed form cannot survive. Draws whose rounded keys sit near a rounding boundary are
discarded.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import attempts, check, generator, phrase
from ._r3f_kit import ThinOptions, build_until, clear_of_boundary, dec, number_part, single

PI = math.pi


def fx(value: float) -> Fraction:
    return Fraction(repr(float(value)))


def simpson(f, a: float, b: float, n: int = 1500) -> float:
    """Simpson's rule for the integral of f over [a, b] with 2n panels."""
    h = (b - a) / (2 * n)
    total = f(a) + f(b)
    for i in range(1, 2 * n):
        total += (4 if i % 2 else 2) * f(a + i * h)
    return total * h / 3


def bisect_float(g, lo: float, hi: float) -> float:
    """Root of an increasing g on [lo, hi]."""
    for _ in range(100):
        mid = (lo + hi) / 2
        if g(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# -- the triangle ----------------------------------------------------------------------------------------------


class Triangle:
    """Density rising from 0 at x = 0 to k at x = c and falling to 0 at x = d."""

    def __init__(self, c: int, d: int):
        self.c, self.d = Fraction(c), Fraction(d)
        self.k = 2 / self.d
        self.ci, self.di = c, d
        self._c, self._d, self._k = float(c), float(d), float(self.k)  # floats once, so pdf() is cheap inside Simpson

    def pdf(self, x: float) -> float:
        c, d, k = self._c, self._d, self._k
        if 0 <= x <= c:
            return k * x / c
        if c < x <= d:
            return k * (d - x) / (d - c)
        return 0.0

    def cdf(self, t) -> Fraction:
        t = Fraction(t)
        if t <= self.c:
            return t * t / (self.c * self.d)
        return 1 - (self.d - t) ** 2 / (self.d * (self.d - self.c))

    def median(self) -> float:
        if self.c >= self.d / 2:
            return math.sqrt(float(self.c * self.d) / 2)
        return float(self.d) - math.sqrt(float(self.d * (self.d - self.c)) / 2)

    def quantile(self, p: Fraction) -> float:
        p = float(p)
        c, d = float(self.c), float(self.d)
        if p <= c / d:
            return math.sqrt(p * c * d)
        return d - math.sqrt((1 - p) * d * (d - c))

    def mean(self) -> Fraction:
        return (self.c + self.d) / 3

    def _integral(self, f, lo: float, hi: float) -> float:
        """Simpson's rule on each straight piece of the density, so the kink at the mode adds no error."""
        total = 0.0
        for left, right in ((0.0, self._c), (self._c, self._d)):
            a, b = max(lo, left), min(hi, right)
            if b > a:
                total += simpson(f, a, b, 200)
        return total

    def check_numerically(self):
        area = self._integral(self.pdf, 0, self._d)
        check(abs(area - 1) < 1e-9, "triangle does not have area 1")
        check(abs(self._integral(lambda x: x * self.pdf(x), 0, self._d) - float(self.mean())) < 1e-9, "triangle mean")
        m = self.median()
        check(abs(self._integral(self.pdf, 0, m) - 0.5) < 1e-9, "triangle median")


def _scaled(coef: Fraction, shape: str) -> str:
    """`coef` times `shape` written as a learner reads it: `x/10`, `3(10 - x)/40`."""
    lead = "" if coef.numerator == 1 else str(coef.numerator)
    return f"{lead}{shape}" if coef.denominator == 1 else f"{lead}{shape}/{coef.denominator}"


def _tri_text(c: int, d: int, unknown_k: bool = True) -> str:
    if unknown_k:
        left, right = f"kx/{c}", f"k({d} - x)/{d - c}"
    else:
        left = _scaled(Fraction(2, c * d), "x")
        right = _scaled(Fraction(2, d * (d - c)), f"({d} - x)")
    return (
        f"f(x) = {left} for 0 \u2264 x \u2264 {c}, f(x) = {right} for {c} \u2264 x \u2264 {d}, and f(x) = 0 otherwise"
    )


_D_CHOICES = (4, 5, 8, 10, 16, 20, 25)


def _tri(rng: random.Random, c_above_half: bool | None = None) -> Triangle:
    for _ in attempts(500, "a triangle with a mode strictly inside"):
        d = rng.choice(_D_CHOICES)
        c = rng.randint(1, d - 1)
        if c_above_half is True and c * 2 < d:
            continue
        if c_above_half is False and c * 2 > d:
            continue
        if 2 * c == d:
            continue
        tri = Triangle(c, d)
        tri.check_numerically()
        return tri
    raise AssertionError("unreachable")


def _tri_stem(tri: Triangle, unknown_k: bool = True) -> str:
    tail = ", where k is a constant." if unknown_k else "."
    return (
        f"A continuous random variable X has probability density function {_tri_text(tri.ci, tri.di, unknown_k)}{tail}"
    )


# -- easy ------------------------------------------------------------------------------------------------------


def _easy_k(rng):
    tri = _tri(rng)
    part = number_part(
        "Find the value of k, correct to 3 decimal places.",
        tri.k,
        3,
        2,
        [
            "The graph is a triangle of base d and height k, with area (1/2) x d x k = 1.",
            f"k = 2/{tri.di} = {dec(tri.k)}.",
        ],
        [
            (Fraction(1, tri.di), "r3f-triangle-area-without-the-half"),
            (Fraction(1, tri.ci), "r3f-triangle-k-from-the-mode"),
            (2 / (tri.di - tri.ci), "r3f-triangle-k-from-the-falling-piece"),
            (tri.k * 2, "r3f-triangle-area-doubled"),
        ],
        sensible_range=(0, None),
    )
    return single(f"{_tri_stem(tri)} {part['prompt']}", part)


def _easy_prob(rng):
    tri = _tri(rng)
    t = rng.randint(1, tri.ci)
    p = tri.cdf(t)
    part = number_part(
        f"Find P(X ≤ {t}), correct to 4 decimal places.",
        p,
        4,
        2,
        [
            f"For 0 ≤ x ≤ {tri.ci} the graph is a straight line through the origin, so the area up to x = {t} is a triangle.",
            f"P(X ≤ {t}) = (1/2) x {t} x f({t}) = {float(p):.4f}.",
        ],
        [
            (t * tri.k * Fraction(t, tri.ci), "r3f-triangle-area-without-the-half"),
            (Fraction(t, tri.di), "r3f-cdf-assumes-uniform"),
            (1 - p, "r3f-cdf-gives-complement"),
            (tri.k * t / 2, "r3f-triangle-uses-the-height-as-the-area"),
        ],
        sensible_range=(0, 1),
        within=(0, 1),
    )
    return single(f"{_tri_stem(tri, False)} {part['prompt']}", part)


def _easy_mean(rng):
    tri = _tri(rng)
    part = number_part(
        "Find E(X), correct to 2 decimal places.",
        tri.mean(),
        2,
        2,
        [f"For a triangular density on [0, {tri.di}] with peak at {tri.ci}, E(X) = (0 + {tri.ci} + {tri.di})/3."],
        [
            (Fraction(tri.di, 2), "r3f-mean-taken-as-midpoint"),
            (tri.c, "r3f-triangle-mean-taken-as-mode"),
            ((tri.c + tri.d) / 2, "r3f-triangle-mean-averages-mode-and-end"),
            (tri.d / 3, "r3f-triangle-mean-ignores-the-mode"),
        ],
        sensible_range=(0, tri.di),
    )
    stem = (
        f"The density of a continuous random variable X is triangular: it rises in a straight line from 0 at x = 0 to "
        f"its peak at x = {tri.ci}, then falls in a straight line to 0 at x = {tri.di}, and is 0 elsewhere. "
    )
    return single(stem + part["prompt"], part)


def _easy(rng):
    return rng.choice((_easy_k, _easy_prob, _easy_mean))(rng)


# -- medium ----------------------------------------------------------------------------------------------------


def _clear(value, dp):
    return clear_of_boundary(fx(value) if not isinstance(value, Fraction) else value, dp, Fraction(1, 5))


def _medium_median(rng):
    for _ in attempts(500, "a median clear of a rounding boundary"):
        tri = _tri(rng)
        m = tri.median()
        if _clear(m, 2):
            break
    on_rising = tri.c * 2 >= tri.d
    wrong_piece = (
        float(tri.d) - math.sqrt(float(tri.d * (tri.d - tri.c)) / 2)
        if on_rising
        else math.sqrt(float(tri.c * tri.d) / 2)
    )
    piece = (
        f"on the rising piece, where F(x) = x^2/{tri.ci * tri.di}"
        if on_rising
        else (f"on the falling piece, where 1 - F(x) = ({tri.di} - x)^2/{tri.di * (tri.di - tri.ci)}")
    )
    part = number_part(
        "Find the median of X, correct to 2 decimal places.",
        fx(m),
        2,
        3,
        [f"F({tri.ci}) = {tri.cdf(tri.ci)}, so the median lies {piece}.", f"Setting F(m) = 1/2 gives m = {m:.2f}."],
        [
            (tri.d / 2, "r3f-median-taken-as-midpoint"),
            (tri.c, "r3f-median-taken-as-mode"),
            (fx(wrong_piece), "r3f-quantile-uses-the-wrong-piece"),
            (tri.mean(), "r3f-median-taken-as-mean"),
        ],
        sensible_range=(0, tri.di),
    )
    return single(f"{_tri_stem(tri)} {part['prompt']}", part)


def _medium_across(rng):
    for _ in attempts(500, "two bounds either side of the mode"):
        tri = _tri(rng)
        if tri.di < 8:
            continue
        t1 = rng.randint(1, tri.ci - 1) if tri.ci > 1 else None
        t2 = rng.randint(tri.ci + 1, tri.di - 1) if tri.ci < tri.di - 1 else None
        if t1 and t2:
            break
    p = tri.cdf(t2) - tri.cdf(t1)
    part = number_part(
        f"Find P({t1} < X < {t2}), correct to 4 decimal places.",
        p,
        4,
        3,
        [
            f"The mode is {tri.ci}, so the area is split into two triangles-and-trapeziums. Use P = 1 - P(X ≤ {t1}) - P(X ≥ {t2}).",
            f"P(X ≤ {t1}) = {float(tri.cdf(t1)):.4f} and P(X ≥ {t2}) = {float(1 - tri.cdf(t2)):.4f}, so P = {float(p):.4f}.",
        ],
        [
            (1 - tri.cdf(t1), "r3f-triangle-ignores-the-upper-tail"),
            (tri.cdf(t2) + tri.cdf(t1), "r3f-triangle-adds-the-cumulative-values"),
            (Fraction(t2 - t1, tri.di), "r3f-cdf-assumes-uniform"),
            (1 - p, "r3f-cdf-gives-complement"),
        ],
        sensible_range=(0, 1),
        within=(0, 1),
    )
    return single(f"{_tri_stem(tri)} {part['prompt']}", part)


def _sin_setup(rng):
    length = rng.choice((3, 4, 6, 8, 12))
    return length, fx(PI / (2 * length))


def _sin_stem(length: int) -> str:
    return (
        f"A continuous random variable X has probability density function f(x) = k sin(πx/{length}) for "
        f"0 ≤ x ≤ {length}, and f(x) = 0 otherwise, where k is a constant. Angles are in radians."
    )


def _medium_sin(rng):
    length, k = _sin_setup(rng)
    check(abs(simpson(lambda x: float(k) * math.sin(PI * x / length), 0, length) - 1) < 1e-6, "sine density area")
    if rng.random() < 0.45:
        part = number_part(
            "Find the value of k, correct to 4 decimal places.",
            k,
            4,
            2,
            [
                f"The area is {length}-wide: the integral of sin(πx/{length}) from 0 to {length} is {2 * length}/π.",
                f"k = π/{2 * length}.",
            ],
            [
                (fx(1 / (2 * length)), "r3f-trig-k-omits-pi"),
                (fx(PI / length), "r3f-trig-integral-of-sine-taken-as-one-hump"),
                (fx(2 * length / PI), "r3f-trig-k-inverted"),
                (fx(PI / (2 * length)) / 2, "r3f-trig-k-halved"),
            ],
            sensible_range=(0, None),
        )
        return single(f"{_sin_stem(length)} {part['prompt']}", part)
    opts = [length // 3] if length % 3 == 0 else []
    opts += [length // 4] if length % 4 == 0 else []
    opts += [length // 6] if length % 6 == 0 else []
    a = rng.choice(opts)
    p = (1 - math.cos(PI * a / length)) / 2
    part = number_part(
        f"Find P(X < {a}), correct to 4 decimal places.",
        fx(p),
        4,
        3,
        [
            f"P(X < {a}) = k × (the integral of sin(πx/{length}) from 0 to {a}) with k = π/{2 * length}.",
            f"= (1 - cos(π/{length // a}))/2 = {p:.4f}.",
        ],
        [
            (fx(1 - math.cos(PI * a / length)), "r3f-trig-forgets-k"),
            (fx((1 - math.cos(math.radians(PI * a / length))) / 2), "r3f-trig-calculator-in-degrees"),
            (fx(1 - p), "r3f-cdf-gives-complement"),
            (fx(a / length), "r3f-cdf-assumes-uniform"),
        ],
        sensible_range=(0, 1),
        within=(0, 1),
    )
    if not _clear(p, 4):
        raise ThinOptions("boundary")
    return single(f"{_sin_stem(length)} {part['prompt']}", part)


def _medium(rng):
    return rng.choice((_medium_median, _medium_across, _medium_sin))(rng)


# -- hard ------------------------------------------------------------------------------------------------------


def _exp_setup(rng):
    lam = rng.choice((0.5, 0.25, 0.2, 0.4))
    b = rng.choice((3, 4, 5, 6, 8))
    k = lam / (1 - math.exp(-lam * b))
    check(abs(simpson(lambda x: k * math.exp(-lam * x), 0, b) - 1) < 1e-7, "exponential density area")
    return lam, b, k


def _exp_stem(lam, b):
    return (
        f"A continuous random variable X has probability density function f(x) = ke^(-{lam:g}x) for 0 ≤ x ≤ {b}, "
        "and f(x) = 0 otherwise, where k is a constant."
    )


def _boundary_retry(*values_dp):
    for v, dp in values_dp:
        if not _clear(v, dp):
            raise ThinOptions("value near a rounding boundary")


def _hard_exp_prob(rng):
    lam, b, k = _exp_setup(rng)
    a = rng.randint(1, b - 1)
    p = (1 - math.exp(-lam * a)) / (1 - math.exp(-lam * b))
    _boundary_retry((p, 4))
    check(abs(simpson(lambda x: k * math.exp(-lam * x), 0, a) - p) < 1e-7, "exponential probability")
    part = number_part(
        f"Find P(X < {a}), correct to 4 decimal places.",
        fx(p),
        4,
        4,
        [
            f"k = {lam:g}/(1 - e^(-{lam:g}x{b})) = {k:.4f}.",
            f"P(X < {a}) = k × (1 - e^(-{lam:g}x{a}))/{lam:g} = {p:.4f}.",
        ],
        [
            (fx(1 - math.exp(-lam * a)), "r3f-truncated-uses-untruncated-cdf"),
            (fx(k * (1 - math.exp(-lam * a))), "r3f-exp-integral-omits-the-rate"),
            (fx(1 - p), "r3f-cdf-gives-complement"),
            (fx(a / b), "r3f-cdf-assumes-uniform"),
        ],
        sensible_range=(0, 1),
        within=(0, 1),
    )
    return single(f"{_exp_stem(lam, b)} {part['prompt']}", part)


def _hard_exp_median(rng):
    lam, b, k = _exp_setup(rng)
    m = -math.log(1 - (1 - math.exp(-lam * b)) / 2) / lam
    _boundary_retry((m, 2))
    check(abs(simpson(lambda x: k * math.exp(-lam * x), 0, m) - 0.5) < 1e-7, "exponential median")
    part = number_part(
        "Find the median of X, correct to 2 decimal places.",
        fx(m),
        2,
        4,
        [f"F(m) = (1 - e^(-{lam:g}m))/(1 - e^(-{lam:g}x{b})) = 1/2.", f"m = {m:.2f}."],
        [
            (fx(math.log(2) / lam), "r3f-truncated-uses-untruncated-cdf"),
            (fx(b / 2), "r3f-median-taken-as-midpoint"),
            (fx(1 / lam), "r3f-median-taken-as-mean"),
            (fx(-math.log(1 - (1 - math.exp(-lam * b)) / 2) / lam / 2), "r3f-quantile-halved"),
        ],
        sensible_range=(0, b),
    )
    return single(f"{_exp_stem(lam, b)} {part['prompt']}", part)


def _hard_exp_mean(rng):
    lam, b, k = _exp_setup(rng)
    mean = 1 / lam - b * math.exp(-lam * b) / (1 - math.exp(-lam * b))
    _boundary_retry((mean, 2))
    check(abs(simpson(lambda x: x * k * math.exp(-lam * x), 0, b) - mean) < 1e-7, "exponential mean")
    part = number_part(
        "Find E(X), correct to 2 decimal places.",
        fx(mean),
        2,
        4,
        ["E(X) = k × (integral of x e^(-λx) from 0 to b), by parts.", f"E(X) = {mean:.2f}."],
        [
            (fx(1 / lam), "r3f-truncated-uses-untruncated-mean"),
            (fx(b / 2), "r3f-mean-taken-as-midpoint"),
            (fx(k * (1 / lam - b * math.exp(-lam * b) / lam)), "r3f-exp-integral-omits-the-rate"),
            (fx(1 / lam + b * math.exp(-lam * b) / (1 - math.exp(-lam * b))), "r3f-by-parts-sign-error"),
        ],
        sensible_range=(0, b),
    )
    return single(f"{_exp_stem(lam, b)} {part['prompt']}", part)


def _hard_tri_quantile(rng):
    for _ in attempts(500, "a percentile on the falling piece clear of a rounding boundary"):
        tri = _tri(rng)
        p = rng.choice((Fraction(9, 10), Fraction(3, 4), Fraction(4, 5), Fraction(19, 20)))
        if p <= tri.c / tri.d:
            continue
        q = tri.quantile(p)
        if _clear(q, 2):
            break
    part = number_part(
        f"Find the value q such that P(X ≤ q) = {float(p):g}, correct to 2 decimal places.",
        fx(q),
        2,
        4,
        [
            f"F({tri.ci}) = {tri.cdf(tri.ci)} < {float(p):g}, so q lies on the falling piece.",
            f"1 - F(q) = ({tri.di} - q)^2/{tri.di * (tri.di - tri.ci)} = {1 - float(p):g}, so q = {q:.2f}.",
        ],
        [
            (fx(math.sqrt(float(p) * tri.ci * tri.di)), "r3f-quantile-uses-the-wrong-piece"),
            (fx(float(p) * tri.di), "r3f-cdf-assumes-uniform"),
            (fx(tri.di - math.sqrt(float(p) * tri.di * (tri.di - tri.ci))), "r3f-quantile-swaps-p-and-one-minus-p"),
            (tri.c, "r3f-median-taken-as-mode"),
        ],
        sensible_range=(0, tri.di),
    )
    return single(f"{_tri_stem(tri)} {part['prompt']}", part)


def _hard_sin_quantile(rng):
    for _ in attempts(500, "a percentile clear of a rounding boundary"):
        length = rng.choice((4, 6, 8, 10, 12))
        p = rng.choice((Fraction(1, 10), Fraction(1, 5), Fraction(1, 4), Fraction(3, 10), Fraction(9, 10)))
        q = length / PI * math.acos(1 - 2 * float(p))
        if _clear(q, 2):
            break
    part = number_part(
        f"Find the value q such that P(X ≤ q) = {float(p):g}, correct to 2 decimal places.",
        fx(q),
        2,
        4,
        [
            f"F(x) = (1 - cos(πx/{length}))/2 (with k = π/{2 * length}), so cos(πq/{length}) = {1 - 2 * float(p):g}.",
            f"q = {q:.2f}.",
        ],
        [
            (fx(float(p) * length), "r3f-cdf-assumes-uniform"),
            (fx(length / PI * math.acos(1 - float(p))), "r3f-trig-forgets-k"),
            (fx(length - q), "r3f-quantile-swaps-p-and-one-minus-p"),
        ],
        sensible_range=(0, length),
    )
    return single(f"{_sin_stem(length)} {part['prompt']}", part)


def _hard(rng):
    return rng.choice((_hard_exp_prob, _hard_exp_median, _hard_exp_mean, _hard_tri_quantile, _hard_sin_quantile))(rng)


@generator(
    "r3f_piecewise_pdf",
    level=13,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3f_piecewise_pdf(rng: random.Random, tier: str = "easy") -> dict:
    """Triangle, sine and truncated-exponential densities: area (easy), pieces and radians (medium), quantiles (hard)."""
    if tier == "easy":
        return build_until(rng, _easy)
    if tier == "medium":
        return build_until(rng, _medium)
    if tier == "hard":
        return build_until(rng, _hard)
    raise ValueError(f"unknown tier {tier!r}")


# -- exam chain ------------------------------------------------------------------------------------------------


def _exam(rng: random.Random) -> dict:
    for _ in attempts(1000, "a triangle whose answers are clear of rounding boundaries"):
        tri = _tri(rng, c_above_half=True)
        if tri.di < 8:
            continue
        t = rng.randint(1, tri.ci - 1) if tri.ci > 1 else None
        if t is None:
            continue
        t1 = t
        t2 = rng.randint(tri.ci + 1, tri.di - 1) if tri.ci < tri.di - 1 else None
        if t2 is None:
            continue
        m = tri.median()
        pb, pd = tri.cdf(t), tri.cdf(t2) - tri.cdf(t1)
        if _clear(m, 2) and _clear(pb, 4) and _clear(pd, 4) and t1 != tri.ci:
            break
    c, d, k = tri.ci, tri.di, tri.k
    tails = tri.cdf(t1) + (1 - tri.cdf(t2))
    parts = [
        number_part(
            "Find the value of k, correct to 3 decimal places.",
            k,
            3,
            2,
            ["The graph is a triangle of base d and height k, with area (1/2) x d x k = 1.", f"k = 2/{d} = {dec(k)}."],
            [
                (Fraction(1, d), "r3f-triangle-area-without-the-half"),
                (Fraction(1, c), "r3f-triangle-k-from-the-mode"),
                (2 / (d - c), "r3f-triangle-k-from-the-falling-piece"),
            ],
            sensible_range=(0, None),
        ),
        number_part(
            f"Hence find P(X ≤ {t1}), correct to 4 decimal places.",
            pb,
            4,
            2,
            [
                f"On the rising piece the area up to x = {t1} is (1/2) × {t1} × f({t1}) = k × {t1}^2/(2 × {c}).",
                f"P = {float(pb):.4f}.",
            ],
            [
                (k * t1 * Fraction(t1, c), "r3f-triangle-area-without-the-half"),
                (Fraction(t1, d), "r3f-cdf-assumes-uniform"),
                (1 - pb, "r3f-cdf-gives-complement"),
            ],
            follow_through=f"part_a*{t1}**2/(2*{c})",
            sensible_range=(0, 1),
            within=(0, 1),
        ),
        number_part(
            "Find the median of X, correct to 2 decimal places.",
            fx(m),
            2,
            3,
            [
                f"F({c}) = {tri.cdf(c)} is at least 1/2, so the median lies on the rising piece: k m^2/(2 x {c}) = 1/2.",
                f"m = {m:.2f}.",
            ],
            [
                (tri.d / 2, "r3f-median-taken-as-midpoint"),
                (tri.c, "r3f-median-taken-as-mode"),
                (fx(float(tri.d) - math.sqrt(float(tri.d * (tri.d - tri.c)) / 2)), "r3f-quantile-uses-the-wrong-piece"),
                (tri.mean(), "r3f-median-taken-as-mean"),
            ],
            follow_through=f"sqrt({c}/part_a)",
            sensible_range=(0, d),
        ),
        number_part(
            f"Find P({t1} < X < {t2}), correct to 4 decimal places.",
            pd,
            4,
            3,
            [
                f"P = 1 - P(X ≤ {t1}) - P(X ≥ {t2}).",
                f"P(X ≥ {t2}) = k({d} - {t2})^2/(2 x {d - c}), so P = {float(pd):.4f}.",
            ],
            [
                (1 - tri.cdf(t1), "r3f-triangle-ignores-the-upper-tail"),
                (tri.cdf(t2) + tri.cdf(t1), "r3f-triangle-adds-the-cumulative-values"),
                (tails, "r3f-cdf-gives-complement"),
                (Fraction(t2 - t1, d), "r3f-cdf-assumes-uniform"),
            ],
            follow_through=f"1-part_a*{t1}**2/(2*{c})-part_a*({d}-{t2})**2/(2*({d}-{c}))",
            sensible_range=(0, 1),
            within=(0, 1),
        ),
    ]
    stem = phrase(
        rng,
        [
            "A continuous random variable X has probability density function {f}, where k is a constant.",
            "The probability density function of a continuous random variable X is {f}, where k is a constant.",
            "Let X be a continuous random variable with probability density function {f}, where k is a constant.",
        ],
        f=_tri_text(tri.ci, tri.di, True),
    )
    return dict(text=stem, parts=parts)


@generator("exd_x_r3f_piecewise_pdf", level=13, presets={"exam": dict(tier="exam")})
def exd_x_r3f_piecewise_pdf(rng: random.Random, tier: str = "exam") -> dict:
    """k, a probability on the rising piece, the median, then a probability across the mode."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return build_until(rng, _exam)
