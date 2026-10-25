"""Exam source for `senior-recognise-the-graphs-of-y-sin-x-y`: count the features of sine, cosine and tangent graphs.

The practice tiers (and the existing multiple-choice exam) ask which of the three functions a drawn graph is. Here
the function is named, or described by two or three of its features, and the learner counts what happens to its graph
on a stated extended domain. Every answer is a typed whole number:

  * sine and cosine, on a domain whose end points are multiples of pi / 2: (a) the number of x-intercepts, counting
    any at an end point; (b) the number of turning points strictly between the end points; (c) how many of those are
    maximum points (follow-through from (b)); (d) how many points have y = 1/2; (e) the sum of the x-coordinates of the
    maximum points, as a multiple of pi / 2.
  * tangent, on a domain whose end points are multiples of pi / 4 that are not asymptotes: (a) the number of
    x-intercepts; (b) the number of vertical asymptotes; (c) the number of separate pieces the asymptotes cut the
    graph into (follow-through from (b)); (d) how many points have y = 1; (e) the sum of the x-coordinates of those
    points, as a multiple of pi / 4.

Every count is made twice: by evaluating the function at the lattice of points where the feature can occur and
counting the ones that meet it, and by residue classes of the multiple of pi / 2 (or pi / 4); the draw is rejected if
they differ.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import VerificationError, attempts, check, generator
from ._exd_x_pc_kit import count_part
from ._spread_kit import uses_rng
from ._r3d_kit import pi_text

_EPS = 1e-9
_FUNCS = {"sin": math.sin, "cos": math.cos, "tan": math.tan}


def _domain_text(lo: Fraction, hi: Fraction) -> str:
    return f"{pi_text(lo)} ≤ x ≤ {pi_text(hi)}"


def _wave_counts(name: str, lo2: int, hi2: int) -> dict:
    """Counts for sin or cos on [lo2 pi/2, hi2 pi/2], by evaluation at multiples of pi/6."""
    f = _FUNCS[name]
    us = range(3 * lo2, 3 * hi2 + 1)  # x = u pi / 6
    zeros = sum(1 for u in us if abs(f(u * math.pi / 6)) < _EPS)
    turning = [u for u in us if 3 * lo2 < u < 3 * hi2 and abs(f(u * math.pi / 6)) > 1 - _EPS]
    maxima = sum(1 for u in turning if f(u * math.pi / 6) > 0)
    half = sum(1 for u in us if abs(f(u * math.pi / 6) - 0.5) < _EPS)
    first_is_max = f(turning[0] * math.pi / 6) > 0 if turning else False
    # The same counts from the residue classes of the multiple of pi/2 (and of pi/6 for the line y = 1/2).
    ks = range(lo2, hi2 + 1)
    zero_class, turn_class = (0, 1) if name == "sin" else (1, 0)
    check(zeros == sum(1 for k in ks if k % 2 == zero_class), "zero count differs between the two routes")
    inner = [k for k in ks if lo2 < k < hi2 and k % 2 == turn_class]
    check(len(turning) == len(inner), "turning point count differs between the two routes")
    check(maxima == sum(1 for k in inner if k % 4 == (1 if name == "sin" else 0)), "maximum count differs")
    residues = {1, 5} if name == "sin" else {2, 10}  # u mod 12 where f = 1/2
    check(half == sum(1 for u in us if u % 12 in residues), "y = 1/2 count differs between the two routes")
    check(all(u % 3 == 0 for u in turning), "a turning point is not a multiple of pi/2")
    max_sum = sum(u // 3 for u in turning if f(u * math.pi / 6) > 0)  # in units of pi/2
    min_sum = sum(u // 3 for u in turning if f(u * math.pi / 6) < 0)
    check(max_sum == sum(k for k in inner if k % 4 == (1 if name == "sin" else 0)), "maximum sum differs")
    return dict(
        max_sum=max_sum,
        min_sum=min_sum,
        zeros=zeros,
        turning=len(turning),
        maxima=maxima,
        half=half,
        first_is_max=first_is_max,
        zeros_inside=sum(1 for k in ks if lo2 < k < hi2 and k % 2 == zero_class),
        turning_ends=sum(1 for k in ks if k % 2 == turn_class),
    )


def _tan_counts(lo4: int, hi4: int) -> dict:
    """Counts for tan on [lo4 pi/4, hi4 pi/4], by evaluation at multiples of pi/4."""
    js = range(lo4, hi4 + 1)
    zeros = sum(1 for j in js if abs(math.tan(j * math.pi / 4)) < _EPS)
    asymptotes = sum(1 for j in js if abs(math.cos(j * math.pi / 4)) < _EPS)
    one = sum(1 for j in js if abs(math.tan(j * math.pi / 4) - 1) < _EPS)
    check(zeros == sum(1 for j in js if j % 4 == 0), "zero count differs between the two routes")
    check(asymptotes == sum(1 for j in js if j % 4 == 2), "asymptote count differs between the two routes")
    check(one == sum(1 for j in js if j % 4 == 1), "tan x = 1 count differs between the two routes")
    one_sum = sum(j for j in js if abs(math.tan(j * math.pi / 4) - 1) < _EPS)  # in units of pi/4
    check(one_sum == sum(j for j in js if j % 4 == 1), "sum of the tan x = 1 points differs between the routes")
    return dict(
        one_sum=one_sum,
        zeros=zeros,
        asymptotes=asymptotes,
        one=one,
        zeros_sum=sum(j for j in js if j % 4 == 0),
        zeros_inside=sum(1 for j in js if lo4 < j < hi4 and j % 4 == 0),
    )


_DESCRIPTION = {
    "sin": "passes through the origin, rises to a greatest value of 1 at x = π/2 and repeats every 2π",
    "cos": "crosses the y-axis at (0, 1), falls to a least value of -1 at x = π and repeats every 2π",
    "tan": "passes through the origin with gradient 1, has a vertical asymptote at x = π/2 and repeats every π",
}


def _intro(rng: random.Random, name: str, domain: str) -> str:
    style = rng.choice(("named", "interval", "described"))
    if style == "named":
        return f"The graph of y = {name} x is drawn for {domain}."
    if style == "interval":
        return f"Consider the function y = {name} x on the domain {domain}."
    return f"A curve {_DESCRIPTION[name]}. It is drawn for {domain}."


def _wave(rng: random.Random, name: str) -> dict:
    for _ in attempts(1000, "a domain with at least nine turning points"):
        lo2 = rng.randint(-16, 4)
        hi2 = lo2 + rng.randint(10, 22)
        if hi2 <= 16:
            c = _wave_counts(name, lo2, hi2)
            if c["turning"] >= 9 and c["zeros"] >= 3:
                break
    lo, hi = Fraction(lo2, 2), Fraction(hi2, 2)
    width_periods = (hi2 - lo2) // 4
    domain = _domain_text(lo, hi)
    zeros, turning, maxima, half = c["zeros"], c["turning"], c["maxima"], c["half"]
    other = _wave_counts("cos" if name == "sin" else "sin", lo2, hi2)
    ft_max = "floor((part_b + 1)/2)" if c["first_is_max"] else "floor(part_b/2)"
    check(
        maxima == (turning + 1) // 2 if c["first_is_max"] else maxima == turning // 2, "maximum count disagrees with b"
    )
    zero_text = "sin x = 0" if name == "sin" else "cos x = 0"
    a = count_part(
        "How many times does the graph meet the x-axis on this domain, counting any point at an end of the domain?",
        zeros,
        1,
        [
            f"{zero_text} at every {'multiple of π' if name == 'sin' else 'odd multiple of π/2'}; those in {domain} number {zeros}."
        ],
        [
            (other["zeros"], "tgx-swaps-sine-and-cosine"),
            (c["zeros_inside"], "tgx-drops-endpoint-zeros"),
            (max(width_periods, 1), "tgx-counts-periods-not-features"),
            (turning, "tgx-counts-turning-points-for-zeros"),
            (zeros + 1, None),
            (zeros - 1, None),
        ],
    )
    b = count_part(
        "How many turning points (maximum or minimum points) does the graph have with x strictly between the end points?",
        turning,
        1,
        [
            f"The graph has a maximum or minimum at every {'odd multiple of π/2' if name == 'sin' else 'multiple of π'}: {turning} strictly inside {domain}."
        ],
        [
            (other["turning"], "tgx-swaps-sine-and-cosine"),
            (c["turning_ends"], "tgx-counts-the-end-points"),
            (zeros, "tgx-counts-zeros-for-turning-points"),
            (turning + 1, None),
            (turning - 1, None),
        ],
    )
    c_part = count_part(
        "Using part (b), how many of these turning points are maximum points, where y = 1?",
        maxima,
        2,
        [
            "Maximum and minimum points alternate along the graph.",
            f"Of the {turning} turning points, {maxima} are maxima.",
        ],
        [
            (turning - maxima, "tgx-counts-minima-not-maxima"),
            (zeros // 2, "tgx-halves-the-wrong-count"),
            (maxima + 1, "tgx-rounds-the-alternation-the-wrong-way"),
            (turning // 2 + (0 if c["first_is_max"] else 1), None),
            (maxima - 1, None),
            (maxima - 2, None),
            (maxima + 2, None),
        ],
        follow=ft_max,
        hi=turning - 1,  # turning points alternate, so some are minima
    )
    d = count_part(
        "How many points on the graph have y = 1/2?",
        half,
        2,
        [f"Each cycle of {name} x reaches the value 1/2 twice. Counting them in {domain} gives {half}."],
        [
            (zeros, "tgx-uses-the-zero-count"),
            (maxima, "tgx-uses-the-maximum-line"),
            (half + 1, "tgx-adds-a-partial-cycle-solution"),
            (max(half - 1, 1), None),
            (max(width_periods, 1), "tgx-counts-periods-not-features"),
            (half + 2, None),
        ],
    )
    ksum = c["max_sum"]
    if ksum == 0:
        raise ValueError("retry")
    e = count_part(
        "Find the sum of the x-coordinates of the maximum points counted in part (c). "
        "Give the sum as a multiple of π/2, so that a sum of 3π is written as 6.",
        ksum,
        2,
        [
            f"The maximum points inside {domain} are at the {maxima} values of x that are {'π/2 + 2nπ' if name == 'sin' else '2nπ'}.",
            f"Written as multiples of π/2, their sum is {ksum}.",
        ],
        [
            (-ksum, "tgx-sign-slip-in-the-sum"),
            (c["min_sum"], "tgx-sums-the-minima"),
            (ksum * 2, "tgx-reports-the-sum-in-wrong-units"),
            (ksum + maxima, None),
            (ksum - maxima, None),
            (ksum + 2, None),
            (ksum - 2, None),
        ],
        signed=True,
    )
    if None in (a, b, c_part, d, e):
        raise ValueError("retry")
    stem = _intro(rng, name, domain)
    return dict(text=stem, parts=[a, b, c_part, d, e])


def _tan(rng: random.Random) -> dict:
    for _ in attempts(1000, "a tangent domain with end points clear of the asymptotes"):
        lo4 = rng.randint(-24, 8)
        hi4 = lo4 + rng.randint(16, 36)
        if lo4 % 4 != 2 and hi4 % 4 != 2 and hi4 <= 24:
            c = _tan_counts(lo4, hi4)
            if c["asymptotes"] >= 5 and c["zeros"] >= 4 and c["one"] >= 4:
                break
    lo, hi = Fraction(lo4, 4), Fraction(hi4, 4)
    domain = _domain_text(lo, hi)
    zeros, asym, one = c["zeros"], c["asymptotes"], c["one"]
    pieces = asym + 1
    periods = max((hi4 - lo4) // 4, 1)
    a = count_part(
        "How many times does the graph meet the x-axis on this domain, counting any point at an end of the domain?",
        zeros,
        1,
        [f"tan x = 0 at every multiple of π; those in {domain} number {zeros}."],
        [
            (asym, "tgx-swaps-zeros-and-asymptotes"),
            (c["zeros_inside"], "tgx-drops-endpoint-zeros"),
            (periods, "tgx-counts-periods-not-features"),
            (zeros + 1, None),
            (zeros - 1, None),
            (zeros + 2, None),
        ],
    )
    b = count_part(
        "How many vertical asymptotes does the graph have on this domain?",
        asym,
        1,
        [f"The asymptotes are at every odd multiple of π/2; those in {domain} number {asym}."],
        [
            (zeros, "tgx-swaps-zeros-and-asymptotes"),
            (pieces, "tgx-counts-pieces-for-asymptotes"),
            (2 * asym, "tgx-uses-the-sine-period"),
            (asym + 2, None),
            (asym - 1, None),
        ],
    )
    c_part = count_part(
        "The asymptotes cut the graph into separate pieces. Using part (b), how many pieces are there?",
        pieces,
        2,
        ["Each asymptote starts a new piece, and the piece before the first asymptote is also drawn."],
        [
            (periods + 1, "tgx-counts-periods-not-features"),
            (asym + 2, "tgx-adds-one-per-end"),
            (zeros, "tgx-uses-the-zero-count"),
            (asym * 2, None),
            (pieces + 2, None),
            (pieces - 4, None),
            (pieces - 5, None),
        ],
        follow="part_b + 1",
        exclude=(asym,),  # pieces = asymptotes + 1, so the asymptote count is never a piece count
    )
    d = count_part(
        "How many points on the graph have y = 1?",
        one,
        2,
        [f"tan x = 1 at x = π/4 plus any multiple of π. Counting them in {domain} gives {one}."],
        [
            (pieces, "tgx-one-solution-per-piece"),
            (zeros, "tgx-uses-the-zero-count"),
            (asym, "tgx-counts-asymptotes-not-solutions"),
            (one + 1, None),
            (one - 1, None),
            (one + 2, None),
        ],
    )
    usum = c["one_sum"]
    if usum == 0:
        raise ValueError("retry")
    e = count_part(
        "Find the sum of the x-coordinates of the points counted in part (d). "
        "Give the sum as a multiple of π/4, so that a sum of 3π is written as 12.",
        usum,
        2,
        [
            f"The points are at x = π/4 + nπ; in multiples of π/4 these are 1 + 4n.",
            f"Adding the {one} values inside {domain} gives {usum}.",
        ],
        [
            (-usum, "tgx-sign-slip-in-the-sum"),
            (usum * 4, "tgx-reports-the-sum-in-wrong-units"),
            (c["zeros_sum"], "tgx-sums-the-intercepts-instead"),
            (usum + one, None),
            (usum - one, None),
            (usum + 4, None),
            (usum - 4, None),
        ],
        signed=True,
    )
    if None in (a, b, c_part, d, e):
        raise ValueError("retry")
    stem = _intro(rng, "tan", domain)
    return dict(text=stem, parts=[a, b, c_part, d, e])


@generator("exd_x_tg1_extended_domain_chain", level=13, presets={"exam": dict(tier="exam")})
@uses_rng
def exd_x_tg1_extended_domain_chain(rng: random.Random, tier: str = "exam") -> dict:
    """See the module docstring. Only the exam tier exists."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    name = rng.choice(("sin", "cos", "tan"))
    for _ in attempts(60, "a draw with three usable wrong options per part"):
        try:
            return _tan(rng) if name == "tan" else _wave(rng, name)
        except ValueError as exc:
            if str(exc) != "retry":
                raise
    raise VerificationError("unreachable")
