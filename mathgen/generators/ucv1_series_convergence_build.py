"""Generates `ucv1_series_convergence` and `exd_x_uc_series_convergence` questions: summing and testing series.

One `university-core-v1` skill, ``ucv1-series-convergence`` (domain ``functions-sequences``). Every answer is a
rational number (a sum, a limit of a ratio, a radius), so the skill is graded as a fraction.

Tiers (the reasoning escalates; the numbers stay small):

- ``easy``: the sum of a geometric series written in sigma notation. The start index is 0, 1 or 2, the ratio may be
  negative, and the exponent may be n - 1, so the first term has to be found before the formula is used.
- ``medium``: the limit L = lim |a(n+1)/a(n)| of the ratio test for a term built from powers, factorials or both,
  or the sum of a telescoping series c/((n + r)(n + r + g)) with a gap g of 1 or 2 (partial fractions).
- ``hard``: the radius of convergence of a power series, the sum of a geometric series whose ratio is an expression
  in x at a given x, or a telescoping sum with a gap of 3 or 4 or with three factors in the denominator.
- ``exam`` (topic ``exd_x_uc_series_convergence``): a four-part chain. Either a telescoping series (the partial
  fraction constant, the partial sum S(n), the sum to infinity, then the least n for which S - S(n) is below a given
  tolerance) or a geometric series in x (the ratio at a given x, the sum there, the sum without the first term, then
  the x that gives a stated sum).

Every answer is re-derived by a second route: exact partial sums for the telescoping series, a long float partial
sum for the geometric ones, and the ratio of consecutive terms evaluated through `math.lgamma` at a large n for the
ratio-test limits and the radii.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from fractions import Fraction

import sympy

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, latex_answer, rational_text
from ._ucv1_calc_markup import M, coef, plain, say, tex_rational, value_string, value_text

TOPIC = "ucv1_series_convergence"
EXAM_TOPIC = "exd_x_uc_series_convergence"

_N = sympy.Symbol("n")


def _lone_shape(answer_text: str, distractors) -> bool:
    """Whether the answer is the only integer or only fraction among its options, so it can be picked without working."""
    shape = "fraction" if "/" in answer_text else "integer"
    return all(("fraction" if "/" in d.text else "integer") != shape for d in distractors)


def _fraction_distractors(answer: Fraction, candidates, n: int = 4) -> list[Distractor]:
    """Tagged distractors, dropping any candidate that equals the answer or an earlier distractor."""
    seen = [answer]
    out: list[Distractor] = []
    for value, misconception in candidates:
        value = Fraction(value)
        if value in seen:
            continue
        seen.append(value)
        out.append(Distractor(rational_text(value), misconception, exact_value(value)))
        if len(out) == n:
            break
    return out


_POINTS = (2, 3, 5, 8, 13, 21)


def _equivalent(a, b) -> bool:
    """Equal as rational functions of n, compared exactly at six points (cheap, unlike `simplify`)."""
    a, b = sympy.sympify(a), sympy.sympify(b)
    return all(a.subs(_N, v) == b.subs(_N, v) for v in _POINTS)


def _expression_distractors(answer, candidates, n: int = 4) -> list[Distractor]:
    seen = [answer]
    out: list[Distractor] = []
    for value, misconception in candidates:
        value = sympy.sympify(value)
        if any(_equivalent(value, s) for s in seen):
            continue
        seen.append(value)
        out.append(Distractor(_expression_text(value), misconception, value_string(value)))
        if len(out) == n:
            break
    return out


def _expression_text(expr) -> str:
    """An expression in n the way it is typed: ``n(2n + 5)/(3(n + 1)(n + 2))``."""
    text = value_text(expr)
    return text.replace("·", "*").replace(")*(", ")(").replace("n*(", "n(").replace("*", "")


def _pick(rng: random.Random, values, exclude=()):
    return rng.choice([v for v in values if v not in exclude])


# -- geometric series -----------------------------------------------------------------------------------------


_RATIOS = [
    Fraction(1, 2),
    Fraction(-1, 2),
    Fraction(1, 3),
    Fraction(-1, 3),
    Fraction(2, 3),
    Fraction(-2, 3),
    Fraction(1, 4),
    Fraction(-1, 4),
    Fraction(3, 4),
    Fraction(-3, 4),
    Fraction(1, 5),
    Fraction(-1, 5),
    Fraction(2, 5),
    Fraction(3, 5),
    Fraction(4, 5),
    Fraction(-3, 5),
]


def _geometric_check(first: Fraction, ratio: Fraction, total: Fraction) -> None:
    """Second route: a long partial sum, by float, against the closed form."""
    check(abs(ratio) < 1, "ratio out of range")
    term = float(first)
    r = float(ratio)
    partial = math.fsum(term * r**i for i in range(300))
    check(abs(partial - float(total)) < 1e-9, f"geometric partial sum {partial} != {total}")


def _ratio_tex(ratio: Fraction) -> str:
    sign = "-" if ratio < 0 else ""
    return rf"\left({sign}\frac{{{abs(ratio.numerator)}}}{{{ratio.denominator}}}\right)"


def _easy_geometric(rng: random.Random):
    form = rng.choice(["power", "power", "shift", "fraction"])
    start = rng.choice([1, 2]) if form == "shift" else rng.choice([0, 1, 1, 2])
    a = rng.randint(2, 12)
    if form == "fraction":
        b = rng.randint(2, 7)
        ratio = Fraction(1, b)
        shift = 0
        term_tex = rf"\frac{{{a}}}{{{b}^{{n}}}}"
    elif form == "shift":
        ratio = rng.choice(_RATIOS)
        shift = 1
        term_tex = rf"{a}{_ratio_tex(ratio)}^{{n-1}}"
    else:
        ratio = rng.choice(_RATIOS)
        shift = 0
        term_tex = rf"{a}{_ratio_tex(ratio)}^{{n}}"
    first = a * ratio ** (start - shift)
    total = first / (1 - ratio)
    _geometric_check(first, ratio, total)
    body = rf"\sum_{{n={start}}}^{{\infty}} {term_tex}"
    a_frac = Fraction(a)
    cands = [
        (a_frac / (1 - ratio), "geometric-sum-formula-start-mismatch"),
        (a_frac * ratio / (1 - ratio), "geometric-sum-formula-start-mismatch"),
        (first / (1 + ratio), "calc-sign-slip"),
        (a_frac * ratio**start / (1 - ratio), "geometric-exponent-shift-missed")
        if shift
        else (first * ratio / (1 - ratio), "geometric-first-term-skipped"),
        (first, "geometric-sum-first-term-only"),
    ]
    first_text = rational_text(first)
    working = [
        f"The first term is {first_text} and the common ratio is {rational_text(ratio)}, which has absolute value below 1.",
        f"The sum is {first_text}/(1 - ({rational_text(ratio)})) = {rational_text(total)}.",
    ]
    return body, total, cands, working


# -- ratio test -------------------------------------------------------------------------------------------------


def _log_term(kind: tuple, n: int) -> float:
    """log |a_n| for the ratio-test families, through lgamma so large n is cheap."""
    name = kind[0]
    if name == "poly":  # n^m (c/d)^n
        _, m, c, d = kind
        return m * math.log(n) + n * math.log(c / d)
    if name == "inv":  # c^n/(n d^n)
        _, c, d = kind
        return n * math.log(c / d) - math.log(n)
    if name == "central":  # (kn)!/((n!)^k c^n)
        _, k, c = kind
        return math.lgamma(k * n + 1) - k * math.lgamma(n + 1) - n * math.log(c)
    if name == "exp_fact":  # c^n/n!
        _, c = kind
        return n * math.log(c) - math.lgamma(n + 1)
    if name == "inv_central":  # (n!)^k c^n/(kn)!
        _, k, c = kind
        return k * math.lgamma(n + 1) + n * math.log(c) - math.lgamma(k * n + 1)
    raise ValueError(kind)


def _numeric_ratio(kind: tuple, n: int = 100_000) -> float:
    return math.exp(_log_term(kind, n + 1) - _log_term(kind, n))


def _check_ratio(kind: tuple, limit: Fraction, what: str) -> None:
    got = _numeric_ratio(kind)
    want = float(limit)
    check(abs(got - want) < 2e-3 * max(1.0, want), f"{what}: numeric ratio {got}, exact {want}")


def _ratio_families(rng: random.Random):
    """``(term_tex, kind, limit, wrong candidates, working)`` for one ratio-test term."""
    name = rng.choice(["poly", "poly", "inv", "central", "central", "exp_fact"])
    if name == "poly":
        m = rng.choice([1, 2, 3])
        c, d = rng.choice(
            [(c, d) for c in range(1, 10) for d in range(1, 10) if c != d and math.gcd(c, d) == 1 and max(c, d) <= 9]
        )
        limit = Fraction(c, d)
        n_pow = "n" if m == 1 else f"n^{{{m}}}"
        term_tex = rf"{n_pow}\left(\frac{{{c}}}{{{d}}}\right)^{{n}}"
        kind = ("poly", m, c, d)
        cands = [
            (Fraction(1), "ratio-test-exponential-factor-ignored"),
            (Fraction(d, c), "calc-ratio-inverted"),
            (Fraction(c), "calc-constant-factor-dropped"),
            (Fraction(0), "ratio-test-limit-of-term") if c < d else (Fraction(d), "calc-constant-factor-dropped"),
        ]
        working = [
            f"The ratio a(n+1)/a(n) is ((n + 1)/n)^{m} × {c}/{d}."
            if m > 1
            else f"The ratio a(n+1)/a(n) is ((n + 1)/n) × {c}/{d}.",
            f"As n → ∞, ((n + 1)/n) → 1, so L = {rational_text(limit)}.",
        ]
        return term_tex, kind, limit, cands, working
    if name == "inv":
        c, d = rng.choice([(c, d) for c in range(1, 10) for d in range(1, 10) if c != d and math.gcd(c, d) == 1])
        limit = Fraction(c, d)
        term_tex = rf"\frac{{{c}^{{n}}}}{{n\,{d}^{{n}}}}"
        kind = ("inv", c, d)
        cands = [
            (Fraction(1), "ratio-test-exponential-factor-ignored"),
            (Fraction(d, c), "calc-ratio-inverted"),
            (Fraction(c), "calc-constant-factor-dropped"),
            (Fraction(0), "ratio-test-limit-of-term") if c < d else (Fraction(d), "calc-constant-factor-dropped"),
        ]
        working = [
            f"The ratio a(n+1)/a(n) is (n/(n + 1)) × {c}/{d}.",
            f"As n → ∞, n/(n + 1) → 1, so L = {rational_text(limit)}.",
        ]
        return term_tex, kind, limit, cands, working
    if name == "central":
        k = rng.choice([2, 3])
        c = rng.choice([1, 2, 3, 4, 5, 6, 8, 9, 10, 12] if k == 2 else [1, 2, 3, 4, 5, 9, 10, 27, 30])
        limit = Fraction(k**k, c)
        term_tex = rf"\frac{{({k}n)!}}{{(n!)^{{{k}}}\,{c}^{{n}}}}"
        kind = ("central", k, c)
        cands = [
            (Fraction(k, c), "factorial-ratio-misjudged"),
            (Fraction(c, k**k), "calc-ratio-inverted"),
            (Fraction(k**k), "calc-constant-factor-dropped"),
            (Fraction(1, c), "factorial-ratio-misjudged"),
        ]
        top = "(2n + 2)(2n + 1)" if k == 2 else "(3n + 3)(3n + 2)(3n + 1)"
        bottom = "(n + 1)²" if k == 2 else "(n + 1)³"
        working = [
            f"The ratio a(n+1)/a(n) is {top}/({bottom} × {c}).",
            f"As n → ∞ the factors in n give {k**k}, so L = {rational_text(limit)}.",
        ]
        return term_tex, kind, limit, cands, working
    c = rng.randint(2, 12)
    limit = Fraction(0)
    term_tex = rf"\frac{{{c}^{{n}}}}{{n!}}"
    kind = ("exp_fact", c)
    cands = [
        (Fraction(c), "factorial-ratio-misjudged"),
        (Fraction(1), "ratio-limit-assumed-one"),
        (Fraction(1, c), "calc-ratio-inverted"),
        (Fraction(c, 2), "factorial-ratio-misjudged"),
    ]
    working = [
        f"The ratio a(n+1)/a(n) is {c}/(n + 1).",
        "As n → ∞ this tends to 0, so L = 0.",
    ]
    return term_tex, kind, limit, cands, working


def _medium_ratio(rng: random.Random):
    term_tex, kind, limit, cands, working = _ratio_families(rng)
    _check_ratio(kind, limit, term_tex)
    return term_tex, limit, cands, working


# -- telescoping ------------------------------------------------------------------------------------------------


def _leftover(r: int, g: int) -> Fraction:
    """The sum of 1/(r + i) for i = 1..g: what survives of a telescoping series with gap g."""
    return sum((Fraction(1, r + i) for i in range(1, g + 1)), Fraction(0))


def _telescoping_check(c: int, r: int, g: int, total: Fraction) -> None:
    """Second route: exact partial sums plus the exact tail must equal the total for every n."""
    partial = Fraction(0)
    for n in range(1, 11):
        partial += Fraction(c, (n + r) * (n + r + g))
        tail = Fraction(c, g) * sum((Fraction(1, n + r + i) for i in range(1, g + 1)), Fraction(0))
        check(partial + tail == total, f"telescoping identity fails at n = {n}")


def _telescoping_tex(c: int, r: int, g: int, index: str = "n") -> str:
    first = index if r == 0 else f"{index} + {r}"
    second = f"{index} + {r + g}"
    left = first if r == 0 else f"({first})"
    return rf"\frac{{{c}}}{{{left}({second})}}"


def _telescoping(rng: random.Random, gaps):
    g = rng.choice(gaps)
    r = rng.choice([0, 1, 2, 3] if g <= 2 else [0, 1, 2])
    c = rng.randint(1, 6) if g < 3 else rng.choice([g, 2 * g, 3 * g, 1, 2, 3])
    total = Fraction(c, g) * _leftover(r, g)
    _telescoping_check(c, r, g, total)
    body = rf"\sum_{{n=1}}^{{\infty}} {_telescoping_tex(c, r, g)}"
    leftover = _leftover(r, g)
    shifted = sum((Fraction(1, r + i) for i in range(2, g + 2)), Fraction(0))
    cands = [
        (Fraction(c, g) * shifted, "telescoping-tail-off-by-one"),
        (c * leftover, "telescoping-gap-factor-dropped"),
        (Fraction(c, g) * Fraction(1, r + 1), "telescoping-leftover-terms-miscounted"),
        (Fraction(c, g) * (Fraction(1, r + 1) + Fraction(1, r + g + 1)), "telescoping-difference-added"),
        (
            Fraction(c, g) * sum((Fraction(1, r + i) for i in range(0, g)), Fraction(0)),
            "geometric-sum-formula-start-mismatch",
        )
        if r > 0
        else (Fraction(c * g), "scaled-the-wrong-way"),
    ]
    terms = " + ".join(f"1/{r + i}" for i in range(1, g + 1))
    first_part = f"1/(n{'' if r == 0 else f' + {r}'})"
    working = [
        f"By partial fractions, {c}/((n{'' if r == 0 else f' + {r}'})(n + {r + g})) = ({rational_text(Fraction(c, g))})({first_part} - 1/(n + {r + g})).",
        f"The terms cancel in steps of {g}, leaving {'only the first term' if g == 1 else f'the first {g} terms'} of "
        f"1/(n{'' if r == 0 else f' + {r}'}): {terms}.",
        f"The sum is ({rational_text(Fraction(c, g))})({rational_text(leftover)}) = {rational_text(total)}.",
    ]
    return body, total, cands, working


def _three_factor(rng: random.Random):
    r = rng.choice([0, 1, 2, 3])
    c = rng.choice([1, 2, 3, 4, 6])
    total = Fraction(c, 2 * (r + 1) * (r + 2))
    # second route: exact partial sums against the exact tail c/(2(n + r + 1)(n + r + 2))
    partial = Fraction(0)
    for n in range(1, 11):
        partial += Fraction(c, (n + r) * (n + r + 1) * (n + r + 2))
        check(partial + Fraction(c, 2 * (n + r + 1) * (n + r + 2)) == total, "three-factor telescoping fails")
    first = "n" if r == 0 else f"(n + {r})"
    body = rf"\sum_{{n=1}}^{{\infty}} \frac{{{c}}}{{{first}(n + {r + 1})(n + {r + 2})}}"
    cands = [
        (Fraction(c, (r + 1) * (r + 2)), "telescoping-gap-factor-dropped"),
        (Fraction(c, 2 * (r + 2) * (r + 3)), "telescoping-tail-off-by-one"),
        (Fraction(c, 4 * (r + 1) * (r + 2)), "telescoping-gap-factor-applied-twice"),
        (Fraction(c, 2 * (r + 1)), "telescoping-leftover-terms-miscounted"),
    ]
    base = f"n{'' if r == 0 else f' + {r}'}"
    working = [
        f"By partial fractions, 1/(({base})(n + {r + 1})(n + {r + 2})) = (1/2)(1/(({base})(n + {r + 1})) - 1/((n + {r + 1})(n + {r + 2}))).",
        f"The sum telescopes to (1/2) × 1/({r + 1} × {r + 2}) times {c}, which is {rational_text(total)}.",
    ]
    return body, total, cands, working


# -- radius of convergence --------------------------------------------------------------------------------------


def _radius(rng: random.Random):
    name = rng.choice(["poly", "inv", "central", "inv_central", "squared", "shifted"])
    if name == "poly":
        m = rng.choice([1, 2, 3])
        d = rng.randint(2, 9)
        radius = Fraction(d)
        num, den = ("n" if m == 1 else f"n^{{{m}}}"), f"{d}^{{n}}"
        power = "n"
        kind = ("poly", m, 1, d)
        cands = [
            (Fraction(1, d), "radius-ratio-not-inverted"),
            (Fraction(1), "radius-assumed-one"),
            (Fraction(d * m), "radius-polynomial-factor-multiplied"),
        ]
        working = [f"The ratio of consecutive coefficients tends to 1/{d}, so the series converges for |x| < {d}."]
    elif name == "inv":
        c, d = rng.choice([(c, d) for c in range(1, 7) for d in range(1, 10) if c != d and math.gcd(c, d) == 1])
        radius = Fraction(d, c)
        num, den = f"{c}^{{n}}", rf"n\,{d}^{{n}}"
        power = "n"
        kind = ("inv", c, d)
        cands = [
            (Fraction(c, d), "radius-ratio-not-inverted"),
            (Fraction(1), "radius-assumed-one"),
            (Fraction(d), "calc-constant-factor-dropped"),
            (Fraction(c * d), "scaled-the-wrong-way"),
        ]
        working = [f"The ratio of consecutive coefficients tends to {c}/{d}, so R = {rational_text(radius)}."]
    elif name == "central":
        k = rng.choice([2, 3])
        radius = Fraction(1, k**k)
        num, den = f"({k}n)!", f"(n!)^{{{k}}}"
        power = "n"
        kind = ("central", k, 1)
        cands = [
            (Fraction(k**k), "radius-ratio-not-inverted"),
            (Fraction(1, k), "factorial-ratio-misjudged"),
            (Fraction(1), "radius-assumed-one"),
            (Fraction(k), "factorial-ratio-misjudged"),
        ]
        working = [f"The ratio of consecutive coefficients tends to {k**k}, so R = 1/{k**k}."]
    elif name == "inv_central":
        k = rng.choice([2, 3])
        c = rng.choice([1, 2, 3, 4, 5] if k == 2 else [1, 2, 3])
        radius = Fraction(k**k, c)
        num, den = rf"(n!)^{{{k}}}\,{c}^{{n}}", f"({k}n)!"
        power = "n"
        kind = ("inv_central", k, c)
        cands = [
            (Fraction(c, k**k), "radius-ratio-not-inverted"),
            (Fraction(k, c), "factorial-ratio-misjudged"),
            (Fraction(1), "radius-assumed-one"),
            (Fraction(k**k), "calc-constant-factor-dropped"),
        ]
        working = [f"The ratio of consecutive coefficients tends to {c}/{k**k}, so R = {rational_text(radius)}."]
    elif name == "squared":
        d = rng.choice([4, 9, 16, 25])
        radius = Fraction(int(math.isqrt(d)))
        num, den = "n", f"{d}^{{n}}"
        power = "2n"
        kind = ("poly", 1, 1, d)
        cands = [
            (Fraction(d), "radius-x-squared-not-rooted"),
            (Fraction(1, int(math.isqrt(d))), "radius-ratio-not-inverted"),
            (Fraction(1), "radius-assumed-one"),
            (Fraction(1, d), "radius-ratio-not-inverted"),
        ]
        working = [f"The ratio of consecutive terms tends to x²/{d}, which is below 1 when |x| < {math.isqrt(d)}."]
    else:
        m = rng.choice([1, 2])
        d = rng.randint(2, 8)
        a = rng.randint(1, 4)
        radius = Fraction(d)
        num, den = ("n" if m == 1 else f"n^{{{m}}}"), f"{d}^{{n}}"
        power = "n"
        kind = ("poly", m, 1, d)
        cands = [
            (Fraction(1, d), "radius-ratio-not-inverted"),
            (Fraction(a + d), "radius-centre-added"),
            (Fraction(1), "radius-assumed-one"),
            (Fraction(a), "radius-centre-confused"),
        ]
        working = [
            f"The ratio of consecutive coefficients tends to 1/{d}, so R = {d}; the centre {a} does not change the radius."
        ]
        series_tex = rf"\sum_{{n=1}}^{{\infty}} \frac{{{num}\,(x - {a})^{{n}}}}{{{den}}}"
        _radius_check(kind, radius, power, series_tex)
        return series_tex, radius, cands, working
    series_tex = rf"\sum_{{n=1}}^{{\infty}} \frac{{{num}\,x^{{{power}}}}}{{{den}}}"
    _radius_check(kind, radius, power, series_tex)
    return series_tex, radius, cands, working


def _radius_check(kind: tuple, radius: Fraction, power: str, what: str) -> None:
    """Second route: the ratio of consecutive coefficients at a large n gives the radius."""
    coefficient_ratio = _numeric_ratio(kind)
    got_radius = (1 / coefficient_ratio) ** (1 / (2 if power == "2n" else 1))
    want = float(radius)
    check(abs(got_radius - want) < 2e-3 * max(1.0, want), f"{what}: numeric radius {got_radius}, exact {want}")


# -- geometric series in x ----------------------------------------------------------------------------------------


def _linear_tex(b: int, c: int) -> str:
    """The TeX of bx + c: x + 2, 2x - 1, 3x."""
    text = f"{coef(b)}x"
    if c:
        text += f" {'+' if c > 0 else '-'} {abs(c)}"
    return text


@dataclass
class _GeometricX:
    b: int
    c: int
    d: int
    x0: int
    x1: int

    def ratio(self, x) -> Fraction:
        return Fraction(self.b * x + self.c, self.d)


def _geometric_x_setup(rng: random.Random) -> _GeometricX:
    for _ in attempts(200, "a geometric-in-x setup"):
        d = rng.randint(3, 9)
        b = rng.choice([1, 2, 3, -1, -2])
        c = rng.randint(-4, 4)
        x0, x1 = rng.sample(range(-3, 6), 2)
        g = _GeometricX(b, c, d, x0, x1)
        r0, r1 = g.ratio(x0), g.ratio(x1)
        if r0 == 0 or r1 == 0 or abs(r0) > Fraction(4, 5) or abs(r1) > Fraction(4, 5) or r0 == r1:
            continue
        if Fraction(b * x0, d) == r0 or Fraction(x0 + c, d) == r0:  # keep the wrong ratios distinct
            continue
        return g


def _hard_geometric_x(rng: random.Random):
    g = _geometric_x_setup(rng)
    start = rng.choice([1, 2])
    a = rng.randint(1, 8)
    r0 = g.ratio(g.x0)
    first = a * r0**start
    total = first / (1 - r0)
    _geometric_check(first, r0, total)
    base_tex = rf"\left(\frac{{{_linear_tex(g.b, g.c)}}}{{{g.d}}}\right)"
    coefficient = "" if a == 1 else str(a)
    body = rf"\sum_{{n={start}}}^{{\infty}} {coefficient}{base_tex}^{{n}}"
    wrong_ratio_constant = Fraction(g.b * g.x0, g.d)
    wrong_ratio_coefficient = Fraction(g.x0 + g.c, g.d)
    cands = [
        (a / (1 - r0), "geometric-sum-formula-start-mismatch"),
        (first / (1 + r0), "calc-sign-slip"),
        (a * r0 / (1 - r0), "geometric-sum-formula-start-mismatch"),
    ]
    for wrong, why in (
        (wrong_ratio_constant, "geometric-ratio-constant-term-dropped"),
        (wrong_ratio_coefficient, "geometric-ratio-coefficient-dropped"),
    ):
        if abs(wrong) < 1:  # a ratio of 1 or more would not give a sum to report
            cands.append((a * wrong**start / (1 - wrong), why))
    working = [
        f"At x = {g.x0} the common ratio is {rational_text(r0)}, which has absolute value below 1.",
        f"The first term (n = {start}) is {rational_text(first)}, so the sum is {rational_text(first)}/(1 - ({rational_text(r0)})) = {rational_text(total)}.",
    ]
    return body, g.x0, total, cands, working


# -- the tiers --------------------------------------------------------------------------------------------------


_LEADS_SUM = ["Find the exact sum of ", "Evaluate ", "Find the value of "]


@dataclass
class _Case:
    text: tuple
    answer: Fraction
    wrongs: list
    working: list
    extras: dict = field(default_factory=dict)


def _easy_case(rng: random.Random) -> _Case:
    body, total, cands, working = _easy_geometric(rng)
    return _Case(say(rng.choice(_LEADS_SUM), M(body), "."), total, cands, working)


def _medium_case(rng: random.Random) -> _Case:
    if rng.random() < 0.55:
        term_tex, limit, cands, working = _medium_ratio(rng)
        text = say(
            "For the series ",
            M(r"\sum_{n=1}^{\infty} a_{n}"),
            " with ",
            M(rf"a_{{n}} = {term_tex}"),
            ", find ",
            M(r"L = \lim_{n\to\infty}\left|\frac{a_{n+1}}{a_{n}}\right|"),
            ".",
        )
        return _Case(text, limit, cands, working)
    body, total, cands, working = _telescoping(rng, [1, 2])
    return _Case(say(rng.choice(_LEADS_SUM), M(body), "."), total, cands, working)


def _hard_case(rng: random.Random) -> _Case:
    pick = rng.choice(["radius", "radius", "geometric", "telescoping", "three"])
    if pick == "radius":
        series, radius, cands, working = _radius(rng)
        return _Case(say("Find the radius of convergence of the power series ", M(series), "."), radius, cands, working)
    if pick == "geometric":
        body, x0, total, cands, working = _hard_geometric_x(rng)
        return _Case(say("Find the sum of the series ", M(body), " when ", M(f"x = {x0}"), "."), total, cands, working)
    if pick == "telescoping":
        body, total, cands, working = _telescoping(rng, [3, 4])
        return _Case(say(rng.choice(_LEADS_SUM), M(body), "."), total, cands, working)
    body, total, cands, working = _three_factor(rng)
    return _Case(say(rng.choice(_LEADS_SUM), M(body), "."), total, cands, working)


_TIERS = {"easy": _easy_case, "medium": _medium_case, "hard": _hard_case}


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def ucv1_series_convergence(rng: random.Random, tier: str = "easy") -> dict:
    """A series to sum, test or bound: easy geometric, medium ratio test or telescoping, hard radius and telescoping."""
    check(tier in _TIERS, f"unknown tier {tier!r}")
    case = _TIERS[tier](rng)
    distractors = _fraction_distractors(case.answer, case.wrongs)
    if _lone_shape(rational_text(case.answer), distractors):
        # no wrong option reads like the answer (the only integer, or the only fraction): add the sign slip
        distractors = _fraction_distractors(case.answer, [(-case.answer, "calc-sign-slip"), *case.wrongs])
    check(len(distractors) >= 2, "fewer than two distractors")
    text, latex = case.text
    return dict(
        text=text,
        latex=latex,
        answer=exact_value(case.answer),
        answer_text=rational_text(case.answer),
        answer_latex=latex_answer(sympy.Rational(case.answer.numerator, case.answer.denominator)),
        answer_kind="fraction",
        working=case.working,
        distractors=distractors,
    )


# -- exam: a four-part chain ---------------------------------------------------------------------------------------


@dataclass
class _Link:
    prompt: tuple
    kind: str  # "fraction" or "expression"
    value: object
    marks: int
    working: list
    wrongs: list
    ft: str | None = None
    sensible: tuple | None = None


def _link_part(link: _Link) -> dict:
    if link.kind == "fraction":
        distractors = _fraction_distractors(link.value, link.wrongs)
        answer, text = exact_value(link.value), rational_text(link.value)
        latex = latex_answer(sympy.Rational(link.value.numerator, link.value.denominator))
    else:
        distractors = _expression_distractors(link.value, link.wrongs)
        answer, text = value_string(link.value), _expression_text(link.value)
        latex = latex_answer(link.value)
    check(len(distractors) >= 2, f"fewer than two distractors for {link.prompt[0]!r}")
    part = dict(
        prompt=link.prompt[0],
        prompt_latex=link.prompt[1],
        answer_kind=link.kind,
        answer=answer,
        answer_text=text,
        answer_latex=latex,
        marks=link.marks,
        working=link.working,
        distractors=distractors,
    )
    if link.ft:
        part["follow_through"] = link.ft
    if link.sensible and link.kind == "fraction":
        part["sensible_range"] = link.sensible
    return part


_EPSILONS = [Fraction(1, 20), Fraction(1, 50), Fraction(1, 100), Fraction(1, 200), Fraction(1, 500), Fraction(1, 1000)]


def _tail(c: int, r: int, g: int, n: int) -> Fraction:
    """S - S(n) for the telescoping series c/((k + r)(k + r + g)), k from 1."""
    return Fraction(c, g) * sum((Fraction(1, n + r + i) for i in range(1, g + 1)), Fraction(0))


def _least_n(tail, epsilon: Fraction, limit: int = 100_000) -> int:
    """The least n >= 1 with tail(n) < epsilon (tail is decreasing), by doubling then bisection."""
    hi = 1
    while tail(hi) >= epsilon:
        hi *= 2
        check(hi <= limit, "no n found")
    lo = hi // 2 + 1 if hi > 1 else 1
    while lo < hi:
        mid = (lo + hi) // 2
        if tail(mid) < epsilon:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _clear_tolerance(c: int, r: int, g: int, eps: Fraction) -> bool:
    """True when the least n is moderate and neither neighbouring tail sits on the tolerance itself."""
    least = _least_n(lambda m: _tail(c, r, g, m), eps)
    if not 4 <= least <= 700:
        return False
    margin = Fraction(1, 10**9)
    return all(abs(_tail(c, r, g, m) - eps) > margin for m in (least - 1, least))


def _exam_telescoping(rng: random.Random):
    for _ in attempts(100, "a telescoping series with a usable tolerance"):
        g = rng.choice([1, 2])
        r = rng.choice([0, 1, 2, 3])
        c = rng.randint(1, 6)
        epsilons = [e for e in _EPSILONS if _clear_tolerance(c, r, g, e)]
        if epsilons:
            break
    amount = Fraction(c, g)
    leftover = _leftover(r, g)
    total = amount * leftover
    _telescoping_check(c, r, g, total)

    n = _N
    tail_terms = [sympy.Rational(1) / (n + r + i) for i in range(1, g + 1)]
    lead_terms = [sympy.Rational(1, r + i) for i in range(1, g + 1)]
    a_sym = sympy.Rational(amount.numerator, amount.denominator)
    # The answer is kept in the form the follow-through expression evaluates to, so the engine can check the two are
    # equal without a slow `simplify` of a rational function.
    s_n = a_sym * (sum(lead_terms) - sum(tail_terms))
    for m in range(1, 9):  # second route for part (b): direct partial sums
        direct = sum((Fraction(c, (k + r) * (k + r + g)) for k in range(1, m + 1)), Fraction(0))
        closed = s_n.subs(n, m)
        check(Fraction(int(closed.p), int(closed.q)) == direct, f"S({m}) disagrees")

    eps = rng.choice(epsilons)
    least = _least_n(lambda m: _tail(c, r, g, m), eps)
    # second route for part (d): float partial sums bracket the tolerance either side of the answer
    terms = [c / ((k + r) * (k + r + g)) for k in range(1, least + 1)]
    for m, expect_below in ((least, True), (least - 1, False)):
        tail = float(total) - math.fsum(terms[:m])
        check((tail < float(eps)) == expect_below and abs(tail - float(eps)) > 1e-12, f"tolerance bracket fails at {m}")

    k_index = "k"
    a_tex = _telescoping_tex(c, r, g, k_index)
    stem = say(
        "Let ",
        M(rf"a_{{k}} = {a_tex}"),
        " for ",
        M(r"k \ge 1"),
        ", and let ",
        M(r"S_{n} = \sum_{k=1}^{n} a_{k}"),
        " and ",
        M(r"S = \sum_{k=1}^{\infty} a_{k}"),
        ".",
    )
    first_k = "k" if r == 0 else f"(k + {r})"
    split = rf"A\left(\frac{{1}}{{{'k' if r == 0 else f'k + {r}'}}} - \frac{{1}}{{k + {r + g}}}\right)"
    prompt_a = say("Find the constant A for which ", M(rf"a_{{k}} = {split}"), " for every k.")
    prompt_b = say("Hence find ", M(r"S_{n}"), " in terms of n.")
    prompt_c = say("Hence find the exact value of ", M("S"), ".")
    prompt_d = say("Find the least integer n for which ", M(rf"S - S_{{n}} < {tex_rational(eps)}"), ".")

    lead_text = " + ".join(f"1/{r + i}" for i in range(1, g + 1))
    ft_lead = "(" + " + ".join(f"1/{r + i}" for i in range(1, g + 1)) + ")"
    ft_tail = " - ".join(f"1/(n + {r + i})" for i in range(1, g + 1))
    amount_text = rational_text(amount)
    a_wrongs = [
        (Fraction(c), "telescoping-gap-factor-dropped"),
        (Fraction(1, g), "calc-constant-factor-dropped"),
        (Fraction(c * g), "scaled-the-wrong-way"),
        (Fraction(g, c), "calc-ratio-inverted"),
    ]
    tail_off = [sympy.Rational(1) / (n + r + i) for i in range(0, g)]
    b_wrongs = [
        (a_sym * (sum(lead_terms) - sum(tail_off)), "telescoping-tail-off-by-one"),
        (a_sym * (sum(lead_terms) + sum(tail_terms)), "telescoping-difference-added"),
        (a_sym * (lead_terms[0] - tail_terms[-1]), "telescoping-leftover-terms-miscounted"),
        (sympy.Integer(c) * (sum(lead_terms) - sum(tail_terms)), "telescoping-gap-factor-dropped"),
        (-s_n, "calc-sign-slip"),
    ]
    c_wrongs = [
        (amount * Fraction(1, r + 1), "telescoping-leftover-terms-miscounted"),
        (c * leftover, "telescoping-gap-factor-dropped"),
        (Fraction(0), "telescoping-everything-cancels"),
        (amount * (leftover + Fraction(1, r + g + 1)), "telescoping-tail-off-by-one"),
        (amount * Fraction(1, r + 1) + amount * Fraction(1, r + g + 1), "telescoping-difference-added"),
    ]
    first_only = _least_n(lambda m: amount * Fraction(1, m + r + 1), eps)
    no_gap = _least_n(lambda m: _tail(c, r, g, m) * g, eps)
    no_shift = _least_n(lambda m: amount * sum((Fraction(1, m + i) for i in range(1, g + 1)), Fraction(0)), eps)
    d_wrongs = [
        (Fraction(least - 1), "least-n-off-by-one"),
        (Fraction(least + 1), "least-n-off-by-one"),
        (Fraction(first_only), "telescoping-leftover-terms-miscounted"),
        (Fraction(no_gap), "telescoping-gap-factor-dropped"),
        (Fraction(no_shift), "series-shift-ignored"),
    ]
    links = [
        _Link(
            prompt_a,
            "fraction",
            amount,
            2,
            [
                f"1/({first_k}(k + {r + g})) = (1/{g})(1/{'k' if r == 0 else f'(k + {r})'} - 1/(k + {r + g})), so A = {c} × (1/{g}) = {amount_text}."
            ],
            a_wrongs,
            sensible=(0, None),
        ),
        _Link(
            prompt_b,
            "expression",
            s_n,
            3,
            [
                f"S(n) = {amount_text} × (({lead_text}) - ({' + '.join(f'1/(n + {r + i})' for i in range(1, g + 1))})), since the other terms cancel in steps of {g}.",
            ],
            b_wrongs,
            ft=f"part_a*({ft_lead} - {ft_tail})",
        ),
        _Link(
            prompt_c,
            "fraction",
            total,
            2,
            [
                f"As n → ∞ the terms 1/(n + {r + 1}), … tend to 0, so S = {amount_text} × ({lead_text}) = {rational_text(total)}."
            ],
            c_wrongs,
            ft=f"part_a*{ft_lead}",
            sensible=(0, None),
        ),
        _Link(
            prompt_d,
            "fraction",
            Fraction(least),
            3,
            [
                f"S - S(n) = {amount_text} × ({' + '.join(f'1/(n + {r + i})' for i in range(1, g + 1))}), which is decreasing in n.",
                f"It is first below {rational_text(eps)} at n = {least}: at n = {least - 1} it equals {rational_text(_tail(c, r, g, least - 1))} or more.",
            ],
            d_wrongs,
            sensible=(0, None),
        ),
    ]
    return stem, links


def _exam_geometric(rng: random.Random):
    g = _geometric_x_setup(rng)
    r0, r1 = g.ratio(g.x0), g.ratio(g.x1)
    s0 = 1 / (1 - r0)
    from_one = s0 - 1
    target = 1 / (1 - r1)
    # second routes: long float partial sums
    _geometric_check(Fraction(1), r0, s0)
    _geometric_check(r0, r0, from_one)
    _geometric_check(Fraction(1), r1, target)
    check(Fraction(g.b * g.x1 + g.c, g.d) == r1, "target ratio mismatch")

    base_tex = rf"\left(\frac{{{_linear_tex(g.b, g.c)}}}{{{g.d}}}\right)"
    stem = say(
        "Let ",
        M(rf"S(x) = \sum_{{n=0}}^{{\infty}} {base_tex}^{{n}}"),
        " for the values of x at which the series converges.",
    )
    prompt_a = say("Find the common ratio of the series when ", M(f"x = {g.x0}"), ".")
    prompt_b = say("Hence find ", M(f"S({g.x0})"), ".")
    prompt_c = say("Hence find ", M(rf"\sum_{{n=1}}^{{\infty}} {base_tex}^{{n}}"), " when ", M(f"x = {g.x0}"), ".")
    prompt_d = say("Find the value of x for which ", M(f"S(x) = {tex_rational(target)}"), ".")

    wrong_constant = Fraction(g.b * g.x0, g.d)
    wrong_coefficient = Fraction(g.x0 + g.c, g.d)
    x_from_inverse = (Fraction(g.d) * (1 - 1 / target) - g.c) / g.b  # = x1 (the key)
    check(x_from_inverse == g.x1, "x1 mismatch")
    x_ratio_is_sum = (Fraction(g.d) * target - g.c) / g.b  # sets the ratio equal to the sum
    x_from_reciprocal = (Fraction(g.d) / target - g.c) / g.b  # r = 1/S
    x_from_sum_minus_one = (Fraction(g.d) * (target - 1) - g.c) / g.b  # r = S - 1
    x_sign = (Fraction(g.d) * (1 - 1 / target) * -1 - g.c) / g.b
    links = [
        _Link(
            prompt_a,
            "fraction",
            r0,
            2,
            [
                f"At x = {g.x0} the numerator {plain(_linear_tex(g.b, g.c))} equals {g.b * g.x0 + g.c}, so the ratio is {rational_text(r0)}."
            ],
            [
                (wrong_constant, "geometric-ratio-constant-term-dropped"),
                (wrong_coefficient, "geometric-ratio-coefficient-dropped"),
                (-r0, "calc-sign-slip"),
                (1 / r0, "calc-ratio-inverted"),
            ],
        ),
        _Link(
            prompt_b,
            "fraction",
            s0,
            2,
            [
                f"The ratio has absolute value below 1, so S({g.x0}) = 1/(1 - ({rational_text(r0)})) = {rational_text(s0)}."
            ],
            [
                (r0 / (1 - r0), "geometric-sum-formula-start-mismatch"),
                (1 / (1 + r0), "calc-sign-slip"),
                (1 - r0, "calc-ratio-inverted"),
                (r0 / (1 + r0), "calc-sign-slip"),
            ],
            ft="1/(1 - part_a)",
        ),
        _Link(
            prompt_c,
            "fraction",
            from_one,
            3,
            [
                f"The sum from n = 1 is the sum from n = 0 less the n = 0 term, which is 1: {rational_text(s0)} - 1 = {rational_text(from_one)}."
            ],
            [
                (s0, "geometric-sum-formula-start-mismatch"),
                (s0 - r0, "geometric-first-term-skipped"),
                (1 - s0, "calc-sign-slip"),
                (s0 - 2, "geometric-first-term-skipped"),
            ],
            ft="part_b - 1",
        ),
        _Link(
            prompt_d,
            "fraction",
            Fraction(g.x1),
            3,
            [
                f"S(x) = 1/(1 - r) = {rational_text(target)} gives r = {rational_text(r1)}.",
                f"So ({plain(_linear_tex(g.b, g.c))})/{g.d} = {rational_text(r1)}, which gives x = {g.x1}.",
            ],
            [
                (x_ratio_is_sum, "geometric-ratio-set-equal-to-sum"),
                (x_from_reciprocal, "calc-ratio-inverted"),
                (x_from_sum_minus_one, "geometric-first-term-skipped"),
                (x_sign, "calc-sign-slip"),
            ],
        ),
    ]
    return stem, links


@generator(EXAM_TOPIC, level=19, presets={"exam": dict(tier="exam")})
def exd_x_uc_series_convergence(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: a telescoping series (A, S(n), S, least n) or a geometric series in x (r, S, tail sum, x)."""
    check(tier == "exam", f"unknown tier {tier!r}")
    stem, links = (_exam_telescoping if rng.random() < 0.55 else _exam_geometric)(rng)
    return dict(text=stem[0], latex=stem[1], parts=[_link_part(link) for link in links])
