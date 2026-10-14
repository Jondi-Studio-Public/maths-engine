"""Generates `fy1so_alternating_series` (easy, medium, hard) and the exam chain `fy1so_exd_alternating_series`.

Skill ``ucv1-alternating-series-error`` (domain ``functions-sequences``, answer ``fraction``): the alternating series
estimation theorem. For S = Σ(-1)^(n+1) b_n with b_n positive, decreasing and tending to 0, the partial sum S_N of the first
N terms differs from S by less than the first omitted term: |S - S_N| < b_(N+1), and S lies between S_N and S_(N+1).

Tiers (the reasoning escalates; the numbers stay small):

- ``easy``: the error bound for a stated N, which is b_(N+1) (the first term left out), for b_n such as 1/n², 1/(n² + 1),
  1/(2n - 1), 1/n! or 1/(n 2^n).
- ``medium``: the number of terms N that guarantees an error below a stated tolerance: the least N with b_(N+1) < tolerance,
  an inequality to solve and then an off-by-one to avoid (the bound uses b_(N+1), not b_N).
- ``hard``: the estimate itself. Find the least N that guarantees the tolerance, then add exactly N terms with their
  alternating signs: the answer is S_N as a fraction.
- ``exam`` (topic ``fy1so_exd_alternating_series``): a four-part chain on one series. (a) the error bound for S_N at a stated N,
  (b) S_N exactly, (c) the lower end of the interval S_N - bound, which must contain the sum (follows through from (a) and
  (b)), (d) the least N for a stated tolerance.

Each b_n is exact (a Fraction), checked to decrease over its first 60 terms. The sums are checked against a long partial sum
averaged with the next (which brackets the sum far more tightly than the theorem's bound).
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import attempts, check, generator, rational_text
from ._fy1so_common import Link, M, build_part, build_single, plain, say

TOPIC = "fy1so_alternating_series"
EXAM_TOPIC = "fy1so_exd_alternating_series"

# name -> (TeX of b_n, b_n as an exact Fraction, learner text of b_n)
_FAMILIES = {
    "sq": (r"\frac{1}{n^{2}}", lambda n: Fraction(1, n * n)),
    "cube": (r"\frac{1}{n^{3}}", lambda n: Fraction(1, n**3)),
    "sq1": (r"\frac{1}{n^{2} + 1}", lambda n: Fraction(1, n * n + 1)),
    "odd": (r"\frac{1}{2n - 1}", lambda n: Fraction(1, 2 * n - 1)),
    "lin3": (r"\frac{1}{3n + 1}", lambda n: Fraction(1, 3 * n + 1)),
    "pronic": (r"\frac{1}{n(n + 1)}", lambda n: Fraction(1, n * (n + 1))),
    "fact": (r"\frac{1}{n!}", lambda n: Fraction(1, math.factorial(n))),
    "pow2": (r"\frac{1}{n\,2^{n}}", lambda n: Fraction(1, n * 2**n)),
    "pow3": (r"\frac{1}{n\,3^{n}}", lambda n: Fraction(1, n * 3**n)),
    "sqplus": (r"\frac{1}{(n + 1)^{2}}", lambda n: Fraction(1, (n + 1) ** 2)),
    "cube1": (r"\frac{1}{n^{3} + 1}", lambda n: Fraction(1, n**3 + 1)),
    "oddsq": (r"\frac{1}{(2n - 1)^{2}}", lambda n: Fraction(1, (2 * n - 1) ** 2)),
    "pow4": (r"\frac{1}{n\,4^{n}}", lambda n: Fraction(1, n * 4**n)),
}
_EASY_NAMES = ["sq", "cube", "sq1", "odd", "lin3", "pronic", "fact", "pow2", "pow3", "sqplus", "cube1", "oddsq", "pow4"]
_MEDIUM_NAMES = ["sq", "cube", "sq1", "odd", "lin3", "pronic", "sqplus", "cube1", "oddsq"]
_HARD_NAMES = ["cube", "fact", "pow2", "pow3", "sq1", "cube1", "oddsq", "pow4"]


def _b(name: str):
    return _FAMILIES[name][1]


def _check_decreasing(name: str) -> None:
    b = _b(name)
    check(all(b(n + 1) < b(n) and b(n) > 0 for n in range(1, 60)), f"{name} is not positive and decreasing")


def _series_tex(name: str) -> str:
    return rf"\sum_{{n=1}}^{{\infty}} (-1)^{{n+1}}\,{_FAMILIES[name][0]}"


def _partial(name: str, count: int) -> Fraction:
    b = _b(name)
    return sum(((-1) ** (n + 1) * b(n) for n in range(1, count + 1)), Fraction(0))


def _least_terms(name: str, eps: Fraction, limit: int = 4000) -> int | None:
    """The least N with b_(N+1) < eps."""
    b = _b(name)
    for count in range(1, limit):
        if b(count + 1) < eps:
            return count
    return None


def _long_sum(name: str) -> float:
    """The sum of the series, as the average of two consecutive long partial sums."""
    b = _b(name)
    terms = [(-1) ** (n + 1) * float(b(n)) for n in range(1, 402)]
    return (math.fsum(terms[:400]) + math.fsum(terms)) / 2


def _check_sum_bracketed(name: str, count: int) -> None:
    """Second route: the sum, from a long averaged partial sum, lies within the bound of S_N and between S_N and S_(N+1)."""
    b = _b(name)
    long_sum = _long_sum(name)
    low, high = sorted((float(_partial(name, count)), float(_partial(name, count + 1))))
    check(low - 1e-9 <= long_sum <= high + 1e-9, f"{name}: the sum is not between S_{count} and S_{count + 1}")
    check(abs(long_sum - float(_partial(name, count))) <= float(b(count + 1)) + 1e-9, f"{name}: bound too small")


def _tolerance_text(eps: Fraction) -> str:
    return {
        Fraction(1, 10): "0.1",
        Fraction(1, 100): "0.01",
        Fraction(1, 1000): "0.001",
        Fraction(1, 10000): "0.0001",
    }.get(eps, f"1/{eps.denominator}")


def _tolerance_tex(eps: Fraction) -> str:
    text = _tolerance_text(eps)
    return text if "/" not in text else rf"\frac{{1}}{{{eps.denominator}}}"


# -- easy ------------------------------------------------------------------------------------------------------------


def _easy(rng: random.Random):
    name = rng.choice(_EASY_NAMES)
    _check_decreasing(name)
    b = _b(name)
    limit = 6 if name in ("fact", "pow2", "pow3") else 9
    n_terms = rng.randint(2, limit)
    value = b(n_terms + 1)
    _check_sum_bracketed(name, n_terms)
    wrongs = [
        (b(n_terms), "alt-error-last-term-used"),
        (b(n_terms + 2), "alt-error-term-after-next"),
        (b(n_terms + 1) / 2, "alt-error-halved"),
        (b(n_terms + 1) + b(n_terms + 2), "alt-error-two-terms-added"),
        (-b(n_terms + 1), "alt-error-signed"),
        (b(n_terms - 1) if n_terms > 1 else b(1), "alt-error-last-term-used"),
    ]
    ask = rng.choice(
        [
            "The series satisfies the conditions of the alternating series test. The sum of its first {n} terms, S_{n}, is used to approximate the sum S. Use the alternating series estimation theorem to find the upper bound on |S - S_{n}|.",
            "The sum S of the alternating series is approximated by the sum of its first {n} terms. What upper bound does the alternating series estimation theorem give for the error?",
        ]
    ).format(n=n_terms)
    prompt = say("Consider ", M(_series_tex(name)), ". " + ask)
    working = [
        f"The error is less than the first term left out, b_{n_terms + 1}.",
        f"b_{n_terms + 1} = {rational_text(value)}.",
    ]
    return prompt, Link(prompt, "fraction", value, working=working, wrongs=wrongs)


# -- medium ----------------------------------------------------------------------------------------------------------

_EPS = [
    Fraction(1, 10),
    Fraction(1, 100),
    Fraction(1, 1000),
    Fraction(1, 10000),
    Fraction(1, 20),
    Fraction(1, 50),
    Fraction(1, 200),
]


def _medium(rng: random.Random):
    for _ in attempts(200, "a tolerance with a clear least N"):
        name = rng.choice(_MEDIUM_NAMES)
        eps = rng.choice(_EPS)
        count = _least_terms(name, eps)
        b = _b(name)
        if count is not None and 2 <= count <= 120 and b(count) > eps and b(count + 1) < eps:
            break
    _check_decreasing(name)
    _check_sum_bracketed(name, count)
    wrongs = [
        (count + 1, "alt-terms-uses-last-term"),
        (count - 1, "alt-terms-off-by-one"),
    ]
    tenfold = _least_terms(name, eps * 10)
    if tenfold is not None:
        wrongs.append((tenfold, "alt-tolerance-misread"))
    slack = _least_terms(name, eps / 10)
    if slack is not None and slack < 3000:
        wrongs.append((slack, "alt-tolerance-misread"))
    wrongs.append((count + 2, "alt-terms-uses-last-term"))
    ask = rng.choice(
        [
            "How many terms of the series must be added to guarantee that the partial sum differs from the sum by less than {eps}? Use the alternating series estimation theorem.",
            "Use the alternating series estimation theorem to find the least number of terms that guarantees an error of less than {eps} when the sum is approximated by a partial sum.",
        ]
    ).format(eps=_tolerance_text(eps))
    prompt = say(
        "The series ", M(_series_tex(name)), " satisfies the conditions of the alternating series test. " + ask
    )
    working = [
        f"With N terms the error is less than b_(N+1), so we need b_(N+1) < {_tolerance_text(eps)}.",
        f"b_{count + 1} = {rational_text(b(count + 1))} < {_tolerance_text(eps)}, but b_{count} = {rational_text(b(count))} is not. So N = {count}.",
    ]
    return prompt, Link(prompt, "fraction", Fraction(count), working=working, wrongs=wrongs)


# -- hard ------------------------------------------------------------------------------------------------------------

_HARD_EPS = [Fraction(1, d) for d in (10, 20, 30, 40, 50, 80, 100, 200, 300)]


def _hard(rng: random.Random):
    for _ in attempts(300, "a tolerance with a small clear N"):
        name = rng.choice(_HARD_NAMES)
        eps = rng.choice(_HARD_EPS)
        count = _least_terms(name, eps)
        b = _b(name)
        if count is not None and 3 <= count <= 6 and b(count) > eps and b(count + 1) < eps:
            break
    _check_decreasing(name)
    _check_sum_bracketed(name, count)
    value = _partial(name, count)
    positive = sum((b(n) for n in range(1, count + 1)), Fraction(0))
    wrongs = [
        (_partial(name, count - 1), "alt-terms-off-by-one"),
        (_partial(name, count + 1), "alt-terms-uses-last-term"),
        (positive, "alt-partial-sum-sign-pattern"),
        (-value, "alt-partial-sum-sign-pattern"),
        (value - b(count + 1), "alt-partial-sum-bound-subtracted"),
    ]
    prompt = say(
        "The series ",
        M(_series_tex(name)),
        " satisfies the conditions of the alternating series test. Use the alternating series estimation theorem to find the fewest terms that guarantee an error of less than ",
        M(_tolerance_tex(eps)),
        ", and give the partial sum of that many terms exactly, as a fraction.",
    )
    terms = " ".join(f"{'+' if n % 2 else '-'} {rational_text(b(n))}" for n in range(1, count + 1)).lstrip("+ ")
    working = [
        f"We need b_(N+1) < {_tolerance_text(eps)}: b_{count + 1} = {rational_text(b(count + 1))} works and b_{count} = {rational_text(b(count))} does not, so N = {count}.",
        f"S_{count} = {terms} = {rational_text(value)}.",
    ]
    return prompt, Link(prompt, "fraction", value, working=working, wrongs=wrongs)


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fy1so_alternating_series(rng: random.Random, tier: str = "easy") -> dict:
    """The alternating series estimation theorem: the error bound, the number of terms, and the estimate itself."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    prompt, link = {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)
    return build_single(prompt, link)


# -- exam: a four-part chain -----------------------------------------------------------------------------------------

_EXAM_NAMES = ["sq", "cube", "sq1", "odd", "lin3", "pronic", "fact", "pow2", "pow3", "sqplus"]


def _exam(rng: random.Random):
    for _ in attempts(300, "an exam series"):
        name = rng.choice(_EXAM_NAMES)
        eps = rng.choice(_EPS)
        count = _least_terms(name, eps)
        b = _b(name)
        n0 = rng.randint(3, 6)
        if count is not None and 2 <= count <= 60 and b(count) > eps and b(count + 1) < eps and count != n0:
            break
    _check_decreasing(name)
    _check_sum_bracketed(name, n0)
    bound = b(n0 + 1)
    s_n = _partial(name, n0)
    lower = s_n - bound
    check(
        float(lower) - 1e-9 <= _long_sum(name) <= float(s_n + bound),
        "the interval misses the sum",
    )
    stem = say(
        "Let ",
        M(r"S = \sum_{n=1}^{\infty} (-1)^{n+1}\,b_{n}"),
        " where ",
        M(rf"b_{{n}} = {_FAMILIES[name][0]}"),
        ". Let ",
        M(r"S_{N}"),
        " be the sum of the first N terms.",
    )
    p_a = say(
        "Find the bound that the alternating series estimation theorem gives for ",
        M(rf"\left|S - S_{{{n0}}}\right|"),
        ".",
    )
    p_b = say("Find ", M(rf"S_{{{n0}}}"), " exactly, as a fraction.")
    p_c = say(
        "Hence find the lower end of the interval ",
        M(rf"S_{{{n0}}} - \text{{(bound)}} \le S \le S_{{{n0}}} + \text{{(bound)}}"),
        ", using the bound from part (a).",
    )
    p_d = say(
        "Find the least N for which the theorem guarantees ",
        M(r"\left|S - S_{N}\right| <"),
        " ",
        M(_tolerance_tex(eps)),
        ".",
    )
    a_wrongs = [
        (b(n0), "alt-error-last-term-used"),
        (b(n0 + 2), "alt-error-term-after-next"),
        (b(n0 + 1) / 2, "alt-error-halved"),
        (b(n0 + 1) + b(n0 + 2), "alt-error-two-terms-added"),
    ]
    b_wrongs = [
        (_partial(name, n0 - 1), "alt-partial-sum-wrong-count"),
        (_partial(name, n0 + 1), "alt-partial-sum-wrong-count"),
        (sum((b(n) for n in range(1, n0 + 1)), Fraction(0)), "alt-partial-sum-sign-pattern"),
        (-s_n, "alt-partial-sum-sign-pattern"),
    ]
    c_wrongs = [
        (s_n + bound, "alt-interval-wrong-end"),
        (s_n, "alt-interval-bound-not-applied"),
        (bound - s_n, "alt-interval-bound-subtracted-the-wrong-way"),
        (s_n - b(n0), "alt-error-last-term-used"),
    ]
    d_wrongs = [
        (count + 1, "alt-terms-uses-last-term"),
        (count - 1, "alt-terms-off-by-one"),
    ]
    tenfold = _least_terms(name, eps * 10)
    if tenfold is not None:
        d_wrongs.append((tenfold, "alt-tolerance-misread"))
    d_wrongs.append((count + 2, "alt-terms-uses-last-term"))
    links = [
        Link(
            p_a,
            "fraction",
            bound,
            1,
            [f"The error is less than b_{n0 + 1} = {rational_text(bound)}."],
            a_wrongs,
            sensible=(0, None),
        ),
        Link(
            p_b,
            "fraction",
            s_n,
            2,
            [
                f"S_{n0} = "
                + " ".join(f"{'+' if n % 2 else '-'} {rational_text(b(n))}" for n in range(1, n0 + 1)).lstrip("+ ")
                + f" = {rational_text(s_n)}."
            ],
            b_wrongs,
        ),
        Link(
            p_c,
            "fraction",
            lower,
            1,
            [f"{rational_text(s_n)} - {rational_text(bound)} = {rational_text(lower)}."],
            c_wrongs,
            ft="part_b - part_a",
        ),
        Link(
            p_d,
            "fraction",
            Fraction(count),
            2,
            [
                f"b_{count + 1} = {rational_text(b(count + 1))} < {_tolerance_text(eps)} but b_{count} = {rational_text(b(count))} is not, so N = {count}."
            ],
            d_wrongs,
            sensible=(0, None),
        ),
    ]
    return stem, links


@generator(EXAM_TOPIC, level=19, presets={"exam": {}})
def fy1so_exd_alternating_series(rng: random.Random) -> dict:
    """The alternating series estimation theorem: the error bound, S_N, the interval it fixes, and the least N."""
    stem, links = _exam(rng)
    return dict(text=stem[0], latex=stem[1], parts=[build_part(link) for link in links])
