"""Generates `w5b_e_limit_table` questions (easy, medium, hard): the number e from tables of (a^h - 1)/h.

``e-as-a-limit-from-a-table`` (``functions-sequences``, answer ``number``). The gradient of y = a^x at (0, 1) is the
limit of (a^h - 1)/h as h tends to 0, and e is the base for which that limit is exactly 1. Every question shows tables
of the quotient at small h, computed as a calculator would show them, and asks for a base or a limit as a typed number.

* easy: four or five bases, each shown at h = 0.01 and 0.001. Which base makes the quotient approach 1?
* medium: the same choice, but the h = 0.1 row is shown as well and is built to mislead: the base whose h = 0.1 value
  is nearest 1 is NOT the base the trend settles on, so the learner must read the trend down each column.
* hard: two reasoning moves. (1) Two bases on either side of e, their limits read from the last row: linear
  interpolation for the base whose limit is 1. (2) A table for a^1 only, and the limit for a base that is a power of
  it: (T^h - 1)/h with T = a^k tends to k times the limit for a, because T^h = a^(kh).

The exam chain `exd_x_w5b_e_limit_table` lives in its own module and reuses `_w5b_e_limit_kit`.

Every key is rebuilt a second way: the limit is ln a, so the nearest base to 1 is found from math.log and must match the
one found from the displayed cells, and the interpolation and scaling answers are checked against the exact ln values
(the displayed and the exact routes must round to the same two places, clear of a rounding boundary).
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import check, draw_until, generator, phrase
from ._w5b_e_limit_kit import (
    ROWS_ALL,
    ROWS_FINE,
    base_text,
    limit,
    away_from_e,
    draw_bracket,
    target_text,
    nearest_to_one,
    places_for,
    quotient,
    row_text,
    table_text,
)
from ._w5b_kit import number_question

_E = math.e


def _frac(x: str) -> Fraction:
    return Fraction(x)


def _grid_draw(rng):
    step = rng.choice([Fraction(1, 10), Fraction(1, 20), Fraction(3, 20), Fraction(1, 10)])
    count = rng.choice([4, 5]) if step <= Fraction(1, 10) else 4
    start = Fraction(rng.randint(240, 276), 100)
    return [start + step * i for i in range(count)]


def _choice_draw(rows, trap):
    def accept_for(rng):
        bases = _grid_draw(rng)
        return bases

    def accept(bases):
        by_ln = sorted(bases, key=lambda b: abs(limit(b) - 1))
        if abs(limit(by_ln[1]) - 1) - abs(limit(by_ln[0]) - 1) < 0.008:
            return False
        if by_ln[0] in (bases[0], bases[-1]):
            return False  # the answer has a neighbour each side, so "the middle" is not a tell
        grid = {b: {h: quotient(b, h) for h in rows} for b in bases}
        best, gap = nearest_to_one(bases, grid, rows[-1])
        if best != by_ln[0] or gap < 0.008:
            return False
        coarse, _ = nearest_to_one(bases, {b: {"0.1": quotient(b, "0.1")} for b in bases}, "0.1")
        return (coarse != best) if trap else True

    return accept_for, accept


def _choice(rng, rows, trap, stems):
    draw, accept = _choice_draw(rows, trap)
    bases = draw_until(rng, draw, accept, tries=400, what="a run of bases with one clear nearest to 1")
    places = places_for(bases)
    table, grid = table_text(bases, rows, places)
    best, _ = nearest_to_one(bases, grid, rows[-1])
    check(best == min(bases, key=lambda b: abs(math.log(float(b)) - 1)), "the table and ln a pick different bases")
    low = [b for b in bases if b < best]
    high = [b for b in bases if b > best]
    coarse, _ = nearest_to_one(bases, {b: {"0.1": quotient(b, "0.1")} for b in bases}, "0.1")
    text = phrase(rng, stems, table=table)
    last = rows[-1]
    working = [
        f"At h = {last} the value for a = {base_text(best, places)} is {float(grid[best][last]):.4f}, "
        f"the nearest to 1 of the bases shown.",
        "The values in each row keep moving towards the limit as h shrinks, so the last row is the best guide.",
        f"The base is {base_text(best, places)}.",
    ]
    candidates = [
        ("picks-base-from-the-coarsest-row", coarse if coarse != best else None),
        ("picks-the-neighbouring-base-above", high[0] if high else None),
        ("picks-the-neighbouring-base-below", low[-1] if low else None),
        ("picks-the-largest-base", bases[-1]),
        ("gives-the-limit-instead-of-the-base", Fraction(1)),
    ]
    return number_question(text, best, places, working, candidates)


_EASY_STEMS = (
    "The gradient of y = a^x at the point (0, 1) is the limit of (a^h - 1)/h as h tends to 0. A calculator gives "
    "this quotient for several bases: {table}. Which of these bases gives a gradient of 1 at (0, 1)?",
    "(a^h - 1)/h was evaluated for each base a below at two small values of h. {table}. "
    "For which of these values of a does (a^h - 1)/h approach 1 as h tends to 0?",
    "Several exponential curves y = a^x are being compared by the gradient at their y-intercept. {table}. "
    "Each line shows (a^h - 1)/h for one base. Which base has gradient 1 at its y-intercept?",
)
_MEDIUM_STEMS = (
    "A student tabulates (a^h - 1)/h for candidate bases, trying to find the base whose gradient at (0, 1) is exactly "
    "1: {table}. Which base does the table support?",
    "The table shows (a^h - 1)/h for each base a as h is made smaller. {table}. "
    "Using the trend as h approaches 0, which of the bases has (a^h - 1)/h tending to 1?",
    "To locate the base of the exponential function that is its own derivative, (a^h - 1)/h was evaluated at three "
    "values of h for each candidate. {table}. Which candidate should be chosen?",
)


def _easy(rng):
    return _choice(rng, ROWS_FINE, False, _EASY_STEMS)


def _medium(rng):
    return _choice(rng, ROWS_ALL, True, _MEDIUM_STEMS)


# -- hard ---------------------------------------------------------------------------------------------------------

_INTERP_STEMS = (
    "The table shows (a^h - 1)/h for two bases. {table}. "
    "Take the h = 0.001 values as the limits and use linear interpolation between the two bases to estimate the base "
    "whose limit is {target}. Give it to 2 decimal places.",
    "For y = a^x the gradient at (0, 1) is the limit of (a^h - 1)/h. {table}. "
    "Reading the limit for each base from its h = 0.001 value, interpolate linearly between the bases to estimate the "
    "value of a that gives a gradient of exactly {target} at (0, 1). Answer to 2 decimal places.",
    "A student brackets the base that gives gradient {target} at (0, 1) between two trial bases. {table}. "
    "Treating the h = 0.001 values as the limits, find by linear interpolation the base for which the limit equals {target}, "
    "to 2 decimal places.",
)
_SCALE_STEMS = (
    "The table shows (a^h - 1)/h for a = {a}. {table}. Use it to estimate the limit of ({t}^h - 1)/h as h tends to "
    "0. Answer to 2 decimal places.",
    "A calculator gives (a^h - 1)/h for a = {a}. {table}. Without evaluating any other base, estimate "
    "the gradient of y = {t}^x at the point (0, 1), to 2 decimal places.",
    "For a = {a} the quotient (a^h - 1)/h behaves as shown. {table}. Estimate the limit as h tends to 0 of "
    "({t}^h - 1)/h, correct to 2 decimal places.",
)
_SCALE_CASES = (
    (2, 2),
    (2, 3),
    (3, 2),
    (3, 3),
    (5, 2),
    (10, 2),
    (2, -1),
    (5, -1),
    (4, -1),
    (2, -2),
    (4, Fraction(1, 2)),
    (9, Fraction(1, 2)),
    (16, Fraction(1, 2)),
    (25, Fraction(1, 2)),
)


def _clear_of_boundary(x: float, places: int = 2, margin: float = 0.12) -> bool:
    scaled = x * 10**places
    return abs(scaled - math.floor(scaled) - 0.5) > margin


def _hard_interpolate(rng):
    def accept(trio):
        target, lo, hi = trio
        a, b = quotient(lo, "0.001"), quotient(hi, "0.001")
        if not (a < target < b):
            return False
        est = float(lo + (hi - lo) * (target - a) / (b - a))
        return _clear_of_boundary(est) and abs(est - 2.72) > 0.15

    target, lo, hi = draw_until(
        rng, draw_bracket, accept, tries=600, what="two bases bracketing e^L with a clean interpolation"
    )
    table, grid = table_text([lo, hi], ROWS_ALL, 2)
    a, b = grid[lo]["0.001"], grid[hi]["0.001"]
    est = lo + (hi - lo) * (target - a) / (b - a)
    text = phrase(rng, _INTERP_STEMS, table=table, target=target_text(target))
    working = [
        f"The limits are about {float(a):.4f} at a = {base_text(lo, 2)} and {float(b):.4f} at a = {base_text(hi, 2)}.",
        f"The limit rises {float(b - a):.4f} over the bases and needs to rise {float(target - a):.4f} from the lower one.",
        f"a = {base_text(lo, 2)} + {float(hi - lo):.2f} x {float(target - a):.4f}/{float(b - a):.4f}, which is about "
        f"{float(est):.2f}.",
    ]
    ratio = (target - a) / (b - a)
    candidates = [
        ("midpoint-of-the-two-bases-used", (lo + hi) / 2),
        ("interpolation-ratio-inverted", lo + (hi - lo) * (b - target) / (b - a)),
        ("nearer-base-taken-without-interpolating", lo if (target - a) < (b - target) else hi),
        ("interpolation-fraction-not-scaled-to-the-bases", ratio),
    ]
    return number_question(text, est, 2, working, away_from_e(candidates))


def _hard_scale(rng):
    def draw(r):
        return r.choice(_SCALE_CASES)

    def accept(case):
        a, k = case
        target = float(k) * limit(Fraction(a))
        last = float(k) * float(quotient(Fraction(a), "0.001"))
        return _clear_of_boundary(target) and round(target, 2) == round(last, 2)

    a, k = draw_until(rng, draw, accept, tries=200, what="a base and exponent with a clean scaled limit")
    base = Fraction(a)
    target_base = base**k if isinstance(k, int) else Fraction(int(round(math.sqrt(a))))
    check(
        float(target_base) ** (1 / float(k)) == float(a) or abs(float(target_base) ** float(k) - a) < 1e-9,
        "T is not a^k",
    )
    t_text = str(target_base) if target_base.denominator == 1 else f"{float(target_base):g}"
    table, cells = row_text(base, ROWS_ALL)
    grid = {base: cells}
    last = grid[base]["0.001"]
    exact = float(k) * limit(base)
    value = Fraction(k) * last
    check(round(float(value), 2) == round(exact, 2), "scaled table value and k ln a disagree")
    text = phrase(rng, _SCALE_STEMS, a=str(a), t=t_text, table=table)
    working = [
        f"({t_text})^h = ({a}^{k if isinstance(k, int) else '(1/2)'})^h = {a}^(kh) with k = "
        f"{k if isinstance(k, int) else '1/2'}.",
        f"So ({t_text}^h - 1)/h = k × ({a}^(kh) - 1)/(kh), and the limit is k times the limit for a = {a}.",
        f"The limit for a = {a} is about {float(last):.4f}, so the answer is {float(value):.2f}.",
    ]
    kf = float(k)
    candidates = [
        ("limit-not-scaled-by-the-exponent", last),
        ("exponent-added-to-the-limit", last + Fraction(k)),
        ("limit-raised-to-the-exponent", last**k if isinstance(k, int) and abs(k) <= 4 else None),
        ("limit-divided-by-the-exponent", last / Fraction(k)),
        ("sign-of-exponent-dropped", abs(value) if kf < 0 else None),
    ]
    return number_question(text, value, 2, working, away_from_e(candidates))


@generator(
    "w5b_e_limit_table",
    level=18,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w5b_e_limit_table(rng, tier="easy"):
    """The base e from tables of (a^h - 1)/h; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_interpolate, _hard_scale, _hard_scale])(rng)
