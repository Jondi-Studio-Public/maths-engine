"""Generates `fy1so_series_tests` (easy, medium, hard) and the exam chain `fy1so_exd_series_tests`.

Skill ``ucv1-series-comparison-tests`` (domain ``functions-sequences``, answer ``fraction``): deciding whether a series of
positive or alternating terms converges, by the p-series test, comparison and limit comparison, the integral test, and the
absolute and conditional convergence of alternating series. The geometric, ratio-test, telescoping and radius-of-convergence
questions belong to ``ucv1-series-convergence``. Every answer is a whole number or a fraction, so the whole skill (including
the exam chain) is marked by the same grader.

Tiers (the reasoning escalates; the numbers stay small):

- ``easy``: the p-series test with a parameter. Find the least positive integer k for which Σ n^m / (r-th root of n^k)
  converges: the terms are 1/n^p with p = k/r - m, and p has to exceed 1 (not just reach it).
- ``medium``: limit comparison. For a_n a ratio of polynomials (or with a root) compared with b_n = 1/n^p, find
  L = lim a_n / b_n, the number that licenses the comparison. The leading terms decide it; the lower terms drop out.
- ``hard``: absolute versus conditional convergence across a family. For Σ(-1)^n n^m / (r-th root of n^k) with k = 1, 2, ..., N,
  count the values of k for which the series converges absolutely, converges conditionally, or diverges. Each k gives its own
  p = k/r - m: divergent for p <= 0 (the terms do not tend to 0), conditional for 0 < p <= 1, absolute for p > 1.
- ``exam`` (topic ``fy1so_exd_series_tests``): a four-part chain on a_n = c/(n + b)^q (q a whole number of at least 2).
  (a) the improper integral from 1 to infinity of c/(x + b)^q dx, (b) the integral-test upper bound on the sum, a_1 plus (a)
  (follows through from (a)), (c) the partial sum S_3, (d) the sharper upper bound S_3 plus the integral from 3 (follows
  through from (c)).

Each behaviour (the exponent p, the limit L) is checked against the numeric slope of log a_n between n = 10^5 and 10^6, and the
exam bounds are checked against a long partial sum.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import sympy

from ..core import attempts, check, generator, rational_text
from ._fy1so_common import Link, M, build_part, build_single, say

TOPIC = "fy1so_series_tests"
EXAM_TOPIC = "fy1so_exd_series_tests"


def _slope(term, ns=(10**5, 10**6)) -> float:
    """-d log a_n / d log n between two large n: the exponent p a term behaves like."""
    a, b = (math.log(abs(term(n))) for n in ns)
    return -(b - a) / (math.log(ns[1]) - math.log(ns[0]))


def _term_tex(c: int, m: int, r: int, k: int | str, style: str) -> str:
    """c n^m divided by the r-th root of n^k (``style`` "root") or by n^(k/r) (``style`` "power") as TeX."""
    top = ("" if c == 1 else str(c)) + ("" if m == 0 else "n" if m == 1 else f"n^{{{m}}}")
    top = top or "1"
    if style == "root":
        bottom = rf"\sqrt{{n^{{{k}}}}}" if r == 2 else rf"\sqrt[{r}]{{n^{{{k}}}}}"
    else:
        bottom = f"n^{{{k}/{r}}}"
    return rf"\frac{{{top}}}{{{bottom}}}"


def _exponent(m: int, r: int, k: int) -> Fraction:
    """p with n^m / (r-th root of n^k) = 1 / n^p."""
    return Fraction(k, r) - m


def _check_exponent(m: int, r: int, k: int) -> None:
    """Second route: the numeric slope of log a_n reproduces p = k/r - m."""
    p = _exponent(m, r, k)
    check(abs(_slope(lambda n: n**m / n ** (k / r)) - float(p)) < 1e-6, "the exponent disagrees with the numeric slope")


# -- easy: the least k for a p-series -----------------------------------------------------------------------------


def _easy(rng: random.Random):
    r = rng.choice([2, 3, 4])
    m = rng.choice([0, 1, 2])
    c = rng.choice([1, 1, 2, 3, 4, 5])
    style = rng.choice(["root", "power"])
    least = r * (m + 1) + 1
    _check_exponent(m, r, least)
    _check_exponent(m, r, least - 1)
    check(_exponent(m, r, least) > 1 >= _exponent(m, r, least - 1), "the least k is wrong")
    wrongs = [
        (least - 1, "series-p-boundary-included"),
        (least + 1, "series-k-overshoot"),
        (r + 1, "series-numerator-power-ignored"),
        (m + 2, "series-root-index-ignored"),
        (r * m + 1, "series-numerator-power-ignored"),
        (r * (m + 1) + r, "series-k-overshoot"),
    ]
    prompt = say(
        "Find the smallest positive integer ",
        M("k"),
        " for which the series ",
        M(rf"\sum_{{n=1}}^{{\infty}} {_term_tex(c, m, r, 'k', style)}"),
        " converges.",
    )
    working = [
        f"The terms equal 1/n^p with p = k/{r} - {m} = (k - {r * m})/{r}.",
        f"The series converges when p > 1, so k > {r * (m + 1)}. At k = {r * (m + 1)} the series is the harmonic series, which diverges.",
        f"The smallest integer is k = {least}.",
    ]
    return prompt, Link(prompt, "fraction", least, working=working, wrongs=wrongs)


# -- medium: the limit in a limit comparison ---------------------------------------------------------------------


def _medium(rng: random.Random):
    return _medium_rational(rng) if rng.random() < 0.6 else _medium_root(rng)


def _medium_rational(rng: random.Random):
    for _ in attempts(200, "a polynomial ratio"):
        u, v = rng.randint(1, 9), rng.randint(1, 9)
        s, w = rng.randint(2, 7), rng.randint(1, 9)
        deg = rng.choice([1, 2, 3])
        if math.gcd(u, s) == 1 or rng.random() < 0.3:
            break
    numerator_degree, denominator_degree = deg, deg + 2
    p = 2
    limit = Fraction(u, s)

    def a(n):
        return (u * n**numerator_degree + v * n ** (numerator_degree - 1)) / (s * n**denominator_degree + w)

    check(abs(_slope(a) - p) < 1e-4, "the comparison series has the wrong exponent")
    check(abs(a(10**7) * 10**14 - float(limit)) < 1e-5, "the limit disagrees with a_n n^2")
    top = f"{u}n^{{{numerator_degree}}}" if numerator_degree > 1 else f"{u}n"
    top = top.replace("1n", "n")
    lower = (
        f"{v}n^{{{numerator_degree - 1}}}"
        if numerator_degree - 1 > 1
        else f"{v}n"
        if numerator_degree - 1 == 1
        else str(v)
    )
    bottom = f"{s}n^{{{denominator_degree}}} + {w}"
    wrongs = [
        (Fraction(s, u), "series-limit-inverted"),
        (Fraction(u), "series-denominator-coefficient-dropped"),
        (Fraction(u + v, s), "series-lower-terms-kept"),
        (Fraction(1, s), "series-numerator-coefficient-dropped"),
        (Fraction(u * w, s), "series-lower-terms-kept"),
        (Fraction(u, s + w), "series-lower-terms-kept"),
    ]
    prompt = say(
        "Let ",
        M(rf"a_{{n}} = \frac{{{top} + {lower}}}{{{bottom}}}"),
        " and ",
        M(r"b_{n} = \frac{1}{n^{2}}"),
        ". Find ",
        M(r"L = \lim_{n \to \infty} \frac{a_{n}}{b_{n}}"),
        ".",
    )
    working = [
        f"a_n / b_n = n²({top.replace('{', '').replace('}', '')} + {lower.replace('{', '').replace('}', '')}) / ({bottom.replace('{', '').replace('}', '')}).",
        f"Dividing top and bottom by n^{denominator_degree}, the limit is {u}/{s} = {rational_text(limit)}.",
    ]
    return prompt, Link(prompt, "fraction", limit, working=working, wrongs=wrongs)


def _medium_root(rng: random.Random):
    for _ in attempts(200, "a ratio with a root"):
        u, v = rng.randint(1, 9), rng.randint(1, 9)
        s, w = rng.randint(2, 7), rng.randint(1, 9)
        if math.gcd(u, s) == 1:
            break
    limit = Fraction(u, s)

    def a(n):
        return (u * n + v) / (s * n * math.sqrt(n) + w)

    check(abs(_slope(a) - 0.5) < 1e-4, "the comparison series has the wrong exponent")
    check(abs(a(10**10) * 10**5 - float(limit)) < 1e-4, "the limit disagrees with a_n sqrt(n)")
    wrongs = [
        (Fraction(s, u), "series-limit-inverted"),
        (Fraction(u), "series-denominator-coefficient-dropped"),
        (Fraction(u + v, s), "series-lower-terms-kept"),
        (Fraction(1, s), "series-numerator-coefficient-dropped"),
        (Fraction(u, s + w), "series-lower-terms-kept"),
    ]
    prompt = say(
        "Let ",
        M(rf"a_{{n}} = \frac{{{u if u > 1 else ''}n + {v}}}{{{s}n\sqrt{{n}} + {w}}}"),
        " and ",
        M(r"b_{n} = \frac{1}{\sqrt{n}}"),
        ". Find ",
        M(r"L = \lim_{n \to \infty} \frac{a_{n}}{b_{n}}"),
        ".",
    )
    working = [
        f"a_n / b_n = √n({u}n + {v}) / ({s}n√n + {w}).",
        f"Dividing top and bottom by n√n, the limit is {u}/{s} = {rational_text(limit)}.",
    ]
    return prompt, Link(prompt, "fraction", limit, working=working, wrongs=wrongs)


# -- hard: absolute, conditional or divergent across a family --------------------------------------------------------


def _classify(m: int, r: int, k: int) -> str:
    """Absolutely convergent, conditionally convergent or divergent, from the numeric exponent of the terms."""
    p = _slope(lambda n: n**m / n ** (k / r))
    if p < 1e-6:
        return "divergent"
    return "conditional" if p < 1 + 1e-6 else "absolute"


def _hard(rng: random.Random):
    r = rng.choice([2, 3, 4])
    m = rng.choice([1, 2])
    extra = rng.randint(1, 4)
    n_max = r * (m + 1) + extra
    c = rng.choice([1, 2, 3])
    style = rng.choice(["root", "power"])
    ask = rng.choice(["absolute", "conditional", "divergent"])
    counts = {"absolute": 0, "conditional": 0, "divergent": 0}
    for k in range(1, n_max + 1):
        counts[_classify(m, r, k)] += 1
    expected = {"absolute": extra, "conditional": r, "divergent": r * m}
    check(counts == expected, "the counts disagree with the numeric classification")
    answer = counts[ask]
    everything = n_max
    wrongs_for = {
        "absolute": [
            (extra + 1, "series-p-boundary-included"),
            (n_max - r * m, "series-converges-counted-as-absolute"),
            (max(extra - 1, 0), "series-p-boundary-excluded"),
            (extra + r, "series-p-boundary-included"),
            (n_max - r * m - extra + 1, "series-converges-counted-as-absolute"),
        ],
        "conditional": [
            (r - 1, "series-p-boundary-excluded"),
            (r + 1, "series-p-boundary-included"),
            (n_max - r * m, "series-converges-counted-as-absolute"),
            (0, "series-conditional-confused-with-divergent"),
            (r * m, "series-conditional-confused-with-divergent"),
            (extra, "series-absolute-confused-with-conditional"),
        ],
        "divergent": [
            (r * m + r, "series-conditional-confused-with-divergent"),
            (r * m - 1, "series-p-boundary-excluded"),
            (r * m + 1, "series-p-boundary-included"),
            (everything - r * m, "series-converges-counted-as-absolute"),
            (r * (m - 1) if m > 1 else r, "series-numerator-power-ignored"),
        ],
    }
    words = {
        "absolute": "absolutely convergent",
        "conditional": "conditionally convergent (convergent, but not absolutely convergent)",
        "divergent": "divergent",
    }
    prompt = say(
        "For each integer ",
        M("k"),
        f" from 1 to {n_max}, consider the series ",
        M(rf"\sum_{{n=1}}^{{\infty}} (-1)^{{n}}\,{_term_tex(c, m, r, 'k', style)}"),
        f". For how many of these values of k is the series {words[ask]}?",
    )
    working = [
        f"The terms have size 1/n^p with p = k/{r} - {m}.",
        f"Divergent when p <= 0 (the terms do not tend to 0): k <= {r * m}. Conditionally convergent when 0 < p <= 1: {r * m} < k <= {r * (m + 1)}. "
        f"Absolutely convergent when p > 1: k > {r * (m + 1)}.",
        f"Among k = 1, ..., {n_max} that gives {counts['divergent']} divergent, {counts['conditional']} conditional and {counts['absolute']} absolute, so the answer is {answer}.",
    ]
    return prompt, Link(prompt, "fraction", answer, working=working, wrongs=wrongs_for[ask])


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fy1so_series_tests(rng: random.Random, tier: str = "easy") -> dict:
    """A convergence-test quantity: the least k for a p-series, the limit in a limit comparison, a count of conditional cases."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    prompt, link = {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)
    return build_single(prompt, link)


# -- exam: a four-part chain on the integral test -----------------------------------------------------------------------


def _term(c: int, b: int, q: int, n: int) -> Fraction:
    return Fraction(c, (n + b) ** q)


def _tail_integral(c: int, b: int, q: int, start: int) -> Fraction:
    """The integral from `start` to infinity of c/(x + b)^q dx."""
    return Fraction(c, (q - 1) * (start + b) ** (q - 1))


def _exam(rng: random.Random):
    for _ in attempts(100, "an integral-test chain"):
        c = rng.choice([1, 2, 3, 4, 5, 6])
        b = rng.choice([0, 1, 2, 3])
        q = rng.choice([2, 3, 4])
        break
    x = sympy.Symbol("x")
    integral = _tail_integral(c, b, q, 1)
    # second route: the antiderivative differentiates back to the integrand and vanishes at infinity
    antiderivative = -sympy.Rational(c, q - 1) / (x + b) ** (q - 1)
    check(
        sympy.simplify(sympy.diff(antiderivative, x) - sympy.Rational(c) / (x + b) ** q) == 0, "antiderivative is wrong"
    )
    check(sympy.limit(antiderivative, x, sympy.oo) == 0, "the antiderivative does not vanish at infinity")
    check(-antiderivative.subs(x, 1) == sympy.Rational(integral.numerator, integral.denominator), "integral disagrees")
    a1 = _term(c, b, q, 1)
    upper = a1 + integral
    s3 = sum((_term(c, b, q, n) for n in range(1, 4)), Fraction(0))
    sharper = s3 + _tail_integral(c, b, q, 3)
    # second route: a long partial sum with the integral tail brackets the sum, and both bounds hold
    partial = math.fsum(c / (n + b) ** q for n in range(1, 2001))
    tail = float(_tail_integral(c, b, q, 2000))
    estimate = partial + tail
    check(estimate <= float(sharper) <= float(upper) and float(upper) - float(sharper) > 0, "the bounds do not nest")
    check(
        float(s3) < estimate - 0 and estimate - float(s3) <= float(_tail_integral(c, b, q, 3)) + 1e-9,
        "the sum is outside S_3 and the tail bound",
    )

    def a_tex(n: str) -> str:
        base = n if b == 0 else f"({n} + {b})"
        return rf"\frac{{{c}}}{{{base}^{{{q}}}}}"

    stem = say(
        "Let ",
        M(rf"a_{{n}} = {a_tex('n')}"),
        " for ",
        M(r"n \ge 1"),
        ", and let ",
        M(r"S = \sum_{n=1}^{\infty} a_{n}"),
        ". The function ",
        M(rf"f(x) = {a_tex('x')}"),
        " is positive and decreasing for ",
        M(r"x \ge 1"),
        ", with ",
        M("f(n) = a_{n}"),
        ".",
    )
    p_a = say("Find ", M(r"\int_{1}^{\infty} f(x)\,dx"), " exactly.")
    p_b = say(
        "Hence write down the upper bound for ",
        M("S"),
        " that the integral test gives: ",
        M(r"a_{1} + \int_{1}^{\infty} f(x)\,dx"),
        ".",
    )
    p_c = say("Find the partial sum ", M(r"S_{3} = a_{1} + a_{2} + a_{3}"), " exactly.")
    p_d = say(
        "Hence find the sharper upper bound for ",
        M("S"),
        ": ",
        M(r"S_{3} + \int_{3}^{\infty} f(x)\,dx"),
        ".",
    )
    k = q - 1
    a_wrongs = [
        (Fraction(c, (1 + b) ** k), "series-integral-power-rule-slip"),
        (Fraction(1, k * (1 + b) ** k), "series-integral-constant-dropped"),
        (Fraction(c, q * (1 + b) ** q), "series-integral-power-rule-slip"),
        (Fraction(c, k * b**k) if b else None, "series-integral-wrong-lower-limit"),
        (a1, "series-integral-replaced-by-first-term"),
        (-integral, "series-integral-sign-slip"),
    ]
    b_wrongs = [
        (a1, "series-bound-integral-omitted"),
        (integral, "series-bound-first-term-omitted"),
        (a1 + _tail_integral(c, b, q, 0) if b else None, "series-integral-wrong-lower-limit"),
        (integral + _term(c, b, q, 2), "series-bound-wrong-first-term"),
    ]
    s2 = sum((_term(c, b, q, n) for n in range(1, 3)), Fraction(0))
    s4 = s3 + _term(c, b, q, 4)
    c_wrongs = [
        (s2, "series-partial-sum-terms-miscounted"),
        (s4, "series-partial-sum-terms-miscounted"),
        (sum((_term(c, b, q, n) for n in range(2, 5)), Fraction(0)), "series-partial-sum-terms-miscounted"),
        (3 * a1, "series-partial-sum-terms-miscounted"),
    ]
    d_wrongs = [
        (s3 + _tail_integral(c, b, q, 4), "series-tail-integral-start-off-by-one"),
        (s3 + _tail_integral(c, b, q, 2), "series-tail-integral-start-off-by-one"),
        (s3, "series-tail-integral-omitted"),
        (_tail_integral(c, b, q, 3), "series-partial-sum-omitted"),
        (s3 + _term(c, b, q, 4), "series-tail-integral-omitted"),
    ]
    ft_b = f"part_a + {a1.numerator}/{a1.denominator}"
    ft_d = f"part_c + {c}/({k}*{3 + b}**{k})"
    links = [
        Link(
            p_a,
            "fraction",
            integral,
            2,
            [f"The antiderivative is -{c}/({k}(x + {b})^{k}), so the integral from 1 to infinity is {integral}."],
            a_wrongs,
            sensible=(0, None),
        ),
        Link(
            p_b,
            "fraction",
            upper,
            1,
            [f"a_1 = {a1}, so the bound is {a1} + {integral} = {upper}."],
            b_wrongs,
            ft=ft_b,
            sensible=(0, None),
        ),
        Link(
            p_c,
            "fraction",
            s3,
            2,
            [f"S_3 = {a1} + {_term(c, b, q, 2)} + {_term(c, b, q, 3)} = {s3}."],
            c_wrongs,
            sensible=(0, None),
        ),
        Link(
            p_d,
            "fraction",
            sharper,
            2,
            [
                f"The integral from 3 to infinity is {_tail_integral(c, b, q, 3)}, so the bound is {s3} + {_tail_integral(c, b, q, 3)} = {sharper}."
            ],
            d_wrongs,
            ft=ft_d,
            sensible=(0, None),
        ),
    ]
    return stem, links


@generator(EXAM_TOPIC, level=19, presets={"exam": {}})
def fy1so_exd_series_tests(rng: random.Random) -> dict:
    """The integral test on c/(n + b)^q: the improper integral, the bound it gives, S_3 and the sharper bound."""
    stem, links = _exam(rng)
    return dict(text=stem[0], latex=stem[1], parts=[build_part(link) for link in links])
