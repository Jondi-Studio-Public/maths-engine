"""Generates `exam_parts_normal_distribution` questions: a four-part, connected exam question on a normal
distribution, for the skill `senior-calculate-probabilities-and-quantiles-associated-with-a-given`.

## The shape

One Australian scenario (masses of packaged produce, commute times, heights of seedlings) with
`X ~ N(mu, sigma^2)`, `mu` and `sigma` stated. Then:

- (a) the z-score of a stated value `x`, exact to 2 decimal places (1 mark);
- (b) hence `P(X < x)` (or `P(X > x)`, varied by seed), correct to 4 decimal places (2 marks,
  `follow_through` from (a) through `erf`);
- (c) the expected number of items in a sample of `n` (50 to 500) beyond that value, to the nearest whole number
  (1 mark, `follow_through` `n*part_b`);
- (d) the value `k` with `P(X < k) = p` for a stated `p`, to 1 decimal place (2 marks, independent of (a) to (c)).

## Construction: draw the z-score FIRST

`x` is built as `mu + z*sigma` from an exact `z = k/100` (hundredths) and integer `mu` and `sigma`, so the
displayed `x` has at most 2 decimal places and the z-score recovered from the DISPLAYED numbers is exactly
`k/100`. Every probability is computed from that exact z with `math.erf` and rounded once, at display, then
re-derived a second way with `statistics.NormalDist(mu, sigma)`. Draws whose displayed value sits within
1e-6 of a rounding boundary are rejected, as are draws where the expected count computed from the exact
probability and from the 4 decimal place probability a learner carries forward could round differently.

## Misconceptions behind the distractors

- (a) `variance-as-standard-deviation` (divides by sigma^2), `forgets-the-standard-deviation` (reports x - mu),
  `z-sign-reversed` (mu - x over sigma).
- (b) `wrong-tail` (the complementary probability), `variance-as-standard-deviation` (standardises with sigma^2),
  `forgets-the-standard-deviation` (uses x - mu as the z-score).
- (c) `wrong-tail` (the count for the other tail), `variance-as-standard-deviation`,
  `forgets-the-standard-deviation`, then truncating rather than rounding.
- (d) `inverts-the-wrong-tail` (inverse cdf of 1 - p), `reports-z-not-x` (the z-score itself),
  `variance-as-standard-deviation` (mu + sigma^2 z), `forgets-the-standard-deviation` (mu + z).

Every distractor is checked unconditionally against its part's grader and must clear it by construction.
"""

from __future__ import annotations

import math
from fractions import Fraction
from statistics import NormalDist

from ..core import (
    VerificationError,
    check,
    check_equivalent,
    generator,
    latex_answer,
    numeric_answer,
    phrase,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)
from ..grading import typed

_STD = NormalDist(0.0, 1.0)
_SQRT2 = math.sqrt(2.0)
_BOUNDARY = 1e-6  # scaled distance from a rounding boundary below which a draw is rejected

#: Quantile proportions, with `P(X < k) = p`. None is 0.5 (z = 0 would make several distractors coincide).
_PROPORTIONS = (0.90, 0.95, 0.10, 0.75, 0.05, 0.80, 0.25, 0.99, 0.975, 0.20)

#: Scenarios. `lower`/`upper` finish "a ... has/have ___ X" in the expected-count part.
_SCENARIOS = (
    dict(
        unit="g",
        mus=(450, 500, 600, 750, 900, 1000, 1250, 1500),
        sigmas=(8, 10, 12, 15, 20, 25, 30, 40),
        noun="bags",
        quantity="mass",
        intro=(
            "A packing shed in {place} fills bags of {item}. The mass, X grams, of a bag is normally "
            "distributed with mean {mu} g and standard deviation {sigma} g, so X ~ N({mu}, {sigma}²).",
            "The mass, X grams, of a bag of {item} packed on a farm near {place} is normally distributed "
            "with mean {mu} g and standard deviation {sigma} g, so X ~ N({mu}, {sigma}²).",
        ),
        places=("Bundaberg", "Mildura", "Griffith", "Stanthorpe", "Donnybrook", "Gingin", "Cygnet"),
        items=("carrots", "potatoes", "oranges", "apples", "sweet potatoes", "onions", "mandarins"),
        lower="a mass of less than {x} g",
        upper="a mass of more than {x} g",
        value_phrase="a mass of {x} g",
    ),
    dict(
        unit="minutes",
        mus=(25, 30, 35, 40, 45, 50, 55, 60),
        sigmas=(3, 4, 5, 6, 7, 8, 9),
        noun="commuters",
        quantity="time",
        intro=(
            "The time, X minutes, that a commuter takes to travel by train from {place} to the city is "
            "normally distributed with mean {mu} minutes and standard deviation {sigma} minutes, so "
            "X ~ N({mu}, {sigma}²).",
            "On weekday mornings, the time, X minutes, for the {place} bus journey into the city is normally "
            "distributed with mean {mu} minutes and standard deviation {sigma} minutes, so X ~ N({mu}, {sigma}²).",
        ),
        places=("Parramatta", "Frankston", "Ipswich", "Elizabeth", "Joondalup", "Cleveland", "Penrith"),
        items=("",),
        lower="a journey time of less than {x} minutes",
        upper="a journey time of more than {x} minutes",
        value_phrase="a journey time of {x} minutes",
    ),
    dict(
        unit="mm",
        mus=(120, 150, 180, 200, 240, 300, 350, 400),
        sigmas=(8, 10, 12, 15, 18, 20, 25),
        noun="seedlings",
        quantity="height",
        intro=(
            "At a nursery in {place}, the height, X millimetres, of a {item} seedling at eight weeks is "
            "normally distributed with mean {mu} mm and standard deviation {sigma} mm, so X ~ N({mu}, {sigma}²).",
            "The height, X millimetres, of {item} seedlings grown by the {place} landcare group is normally "
            "distributed with mean {mu} mm and standard deviation {sigma} mm, so X ~ N({mu}, {sigma}²).",
        ),
        places=("Toowoomba", "Bendigo", "Launceston", "Armidale", "Albany", "Alice Springs", "Murray Bridge"),
        items=("eucalypt", "wattle", "banksia", "grevillea", "bottlebrush"),
        lower="a height of less than {x} mm",
        upper="a height of more than {x} mm",
        value_phrase="a height of {x} mm",
    ),
)


def _phi(z: float) -> float:
    """The standard normal cumulative probability, from the error function."""
    return 0.5 * (1.0 + math.erf(z / _SQRT2))


def _boundary_ok(scaled: float, offset: float) -> bool:
    """True when `scaled` is farther than `_BOUNDARY` from the rounding boundary `k + offset`."""
    frac = scaled - math.floor(scaled)
    return abs(frac - offset) > _BOUNDARY and abs(frac - offset) < 1 - _BOUNDARY


def _round_float(value: float, places: int) -> Fraction:
    return round_half_up(Fraction(value), places)


def _shown(value: Fraction, places: int) -> str:
    return stated_precision_string(value, places) if places else str(int(value))


def _count_text(value: float) -> str:
    return str(int(_round_float(value, 0)))


def _draw(rng):
    for _ in range(400):
        sc = rng.choice(_SCENARIOS)
        mu = rng.choice(sc["mus"])
        sigma = rng.choice(sc["sigmas"])
        k = rng.randint(30, 250) * rng.choice((-1, 1))
        lower = rng.random() < 0.5
        n = rng.randint(50, 500)
        p = rng.choice(_PROPORTIONS)

        z_exact = Fraction(k, 100)
        x = mu + z_exact * sigma  # exact, at most 2 decimal places
        if x <= 0:
            continue
        z = float(z_exact)
        tail = _phi(z) if lower else _phi(-z)
        # Second route: the distribution object itself.
        nd = NormalDist(mu, sigma)
        tail2 = nd.cdf(float(x)) if lower else 1.0 - nd.cdf(float(x))
        check(abs(tail - tail2) < 1e-9, "probability routes disagree")
        if not _boundary_ok(tail * 1e4, 0.5):
            continue
        tail4 = _round_float(tail, 4)
        # Expected count: from the exact probability and from the 4 dp probability a learner carries forward.
        raw_exact = n * tail
        raw_rounded = n * float(tail4)
        if not _boundary_ok(raw_exact, 0.5) or abs(raw_exact - math.floor(raw_exact) - 0.5) < n * 5e-5 + 1e-6:
            continue
        count = _round_float(raw_exact, 0)
        if count != _round_float(raw_rounded, 0) or count < 3:
            continue
        # Quantile.
        q_exact = mu + sigma * _STD.inv_cdf(p)
        if not _boundary_ok(q_exact * 10.0, 0.5):
            continue
        return sc, mu, sigma, k, z_exact, x, lower, n, p, tail, tail4, count, q_exact
    raise VerificationError("no clean normal-distribution draw found")


def _build_distractors(answer_text, candidates, grader_part, places):
    """Filter `candidates` against the canonical answer and check each survivor is graded wrong."""
    picked = pick_distractors(answer_text, candidates, 3, tolerance=stated_precision_tolerance(places) if places else 0)
    for d in picked:
        check(not typed.grade("number", grader_part, d).correct, "a distractor is accepted by the grader")
    check(len(picked) >= 2, "too few distractors survive")
    return picked


class _G:
    """Minimal graded object for checking a distractor against a part's canonical answer."""

    def __init__(self, answer, answer_text, prompt):
        self.answer = answer
        self.answer_text = answer_text
        self.text = prompt
        self.answer_alt = None
        self.answer_form = None
        self.distractors = []
        self.option_notes = {}


@generator("exam_parts_normal_distribution", level=13, presets={"exam": {}})
def exam_parts_normal_distribution(rng):
    sc, mu, sigma, k, z_exact, x, lower, n, p, tail, tail4, count, q_exact = _draw(rng)
    x_text = stated_precision_string(x, 2)
    z_text = stated_precision_string(z_exact, 2)
    z = float(z_exact)
    sym = "<" if lower else ">"
    p_text = f"{p:.3g}" if p in (0.975,) else f"{p:.2f}"

    # ---- the stem -------------------------------------------------------------------------------------
    template = rng.choice(sc["intro"])
    stem = template.format(place=rng.choice(sc["places"]), item=rng.choice(sc["items"]), mu=mu, sigma=sigma)

    # ---- (a) the z-score ------------------------------------------------------------------------------
    # Independent route: standardise from the DISPLAYED numbers.
    check_equivalent(Fraction(x_text) - mu, z_exact * sigma, "z-score route disagrees")
    check(Fraction(z_text) == (Fraction(x_text) - mu) / sigma, "z-score is not exact to 2 decimal places")
    a_prompt = f"Find the z-score for {sc['value_phrase'].format(x=x_text)}, correct to 2 decimal places."
    a_answer = _G(numeric_answer(z_exact), z_text, a_prompt)
    a_cands = [
        stated_precision_string(_round_float((float(x) - mu) / sigma**2, 2), 2),  # variance as sd
        stated_precision_string(_round_float(float(x) - mu, 2), 2),  # forgets sd
        stated_precision_string(-z_exact, 2),  # sign reversed
    ]
    a_dist = _build_distractors(z_text, a_cands, a_answer, 2)
    a_notes = {
        a_cands[0]: "variance-as-standard-deviation: divided by sigma squared instead of sigma",
        a_cands[1]: "forgets-the-standard-deviation: reported x minus the mean without dividing by sigma",
        a_cands[2]: "z-sign-reversed: subtracted the other way, (mean minus x) over sigma",
    }

    # ---- (b) the probability --------------------------------------------------------------------------
    b_text = _shown(tail4, 4)
    b_prompt = f"Hence find P(X {sym} {x_text}), correct to 4 decimal places."
    b_answer = _G(numeric_answer(tail4), b_text, b_prompt)
    other = _round_float(1.0 - float(tail), 4)
    zv = (float(x) - mu) / sigma**2
    zn = float(x) - mu
    b_cands = [
        _shown(other, 4),  # wrong tail
        _shown(_round_float(_phi(zv) if lower else _phi(-zv), 4), 4),  # variance as sd
        _shown(_round_float(_phi(zn) if lower else _phi(-zn), 4), 4),  # forgets sd
    ]
    b_dist = _build_distractors(b_text, b_cands, b_answer, 4)
    b_notes = {
        b_cands[0]: "wrong-tail: gave the probability of the opposite tail",
        b_cands[1]: "variance-as-standard-deviation: standardised with sigma squared",
        b_cands[2]: "forgets-the-standard-deviation: used x minus the mean as the z-score",
    }
    b_ft = "(1 + erf(part_a/sqrt(2)))/2" if lower else "(1 - erf(part_a/sqrt(2)))/2"

    # ---- (c) the expected count -----------------------------------------------------------------------
    c_text = _shown(count, 0)
    event = sc["lower" if lower else "upper"].format(x=x_text)
    c_prompt = (
        f"In a sample of {n} {sc['noun']}, how many would be expected to have {event}? "
        "Give your answer correct to the nearest whole number."
    )
    c_answer = _G(numeric_answer(count), c_text, c_prompt)
    tail_var = _phi(zv) if lower else _phi(-zv)
    tail_sd = _phi(zn) if lower else _phi(-zn)
    c_cands = [
        _count_text(n * (1.0 - float(tail4))),  # wrong tail
        _count_text(n * tail_var),  # variance as sd
        str(math.floor(n * float(tail4))),  # truncates
        _count_text(n * tail_sd),  # forgets sd
    ]
    c_dist = _build_distractors(c_text, c_cands, c_answer, 0)
    c_notes = {
        c_cands[0]: "wrong-tail: used the probability of the opposite tail",
        c_cands[1]: "variance-as-standard-deviation: probability from a z-score divided by sigma squared",
        c_cands[2]: "truncated the expected count instead of rounding to the nearest whole number",
        c_cands[3]: "forgets-the-standard-deviation: probability from x minus the mean as the z-score",
    }

    # ---- (d) the quantile -----------------------------------------------------------------------------
    d_text = _shown(_round_float(q_exact, 1), 1)
    d_prompt = f"Find the value k such that P(X < k) = {p_text}, correct to 1 decimal place."
    d_answer = _G(numeric_answer(_round_float(q_exact, 1)), d_text, d_prompt)
    zq = _STD.inv_cdf(p)
    d_values = [
        mu + sigma * _STD.inv_cdf(1.0 - p),  # inverts the wrong tail
        zq,  # reports z, not x
        mu + sigma**2 * zq,  # variance as sd
        mu + zq,  # forgets sd
    ]
    for v in d_values:
        check(abs(v - q_exact) > 0.5, "a quantile distractor sits too close to the answer")
    d_cands = [_shown(_round_float(v, 1), 1) for v in d_values]
    d_dist = _build_distractors(d_text, d_cands, d_answer, 1)
    d_notes = {
        d_cands[0]: "inverts-the-wrong-tail: used the inverse cdf of 1 minus p",
        d_cands[1]: "reports-z-not-x: stopped at the z-score without converting back to X",
        d_cands[2]: "variance-as-standard-deviation: used mean plus sigma squared times z",
        d_cands[3]: "forgets-the-standard-deviation: used mean plus z",
    }

    # ---- cross-check (d) a second way: the distribution object reproduces P(X < k) = p ----------------
    check(abs(NormalDist(mu, sigma).cdf(q_exact) - p) < 1e-9, "quantile does not reproduce p")

    def notes(allnotes, dist):
        return {d: allnotes[d] for d in dist if d in allnotes}

    parts = [
        dict(
            prompt=a_prompt,
            answer_kind="number",
            answer=numeric_answer(z_exact),
            answer_text=z_text,
            answer_latex=latex_answer(z_text),
            marks=1,
            working=[f"z = (x - μ) / σ = ({x_text} - {mu}) / {sigma} = {z_text}"],
            distractors=a_dist,
            option_notes=notes(a_notes, a_dist),
        ),
        dict(
            prompt=b_prompt,
            answer_kind="number",
            answer=numeric_answer(tail4),
            answer_text=b_text,
            answer_latex=latex_answer(b_text),
            marks=2,
            follow_through=b_ft,
            sensible_range=(0, 1),
            working=[
                f"P(X {sym} {x_text}) = P(Z {sym} {z_text})",
                f"= {b_text} (using technology, normal cdf with mean 0 and standard deviation 1)",
            ],
            distractors=b_dist,
            option_notes=notes(b_notes, b_dist),
        ),
        dict(
            prompt=c_prompt,
            answer_kind="number",
            answer=numeric_answer(count),
            answer_text=c_text,
            answer_latex=latex_answer(c_text),
            marks=1,
            follow_through=f"{n}*part_b",
            working=[
                f"Expected number = {n} × {b_text} = {n * float(tail4):.2f}, so {c_text} to the nearest whole number"
            ],
            distractors=c_dist,
            option_notes=notes(c_notes, c_dist),
        ),
        dict(
            prompt=d_prompt,
            answer_kind="number",
            answer=numeric_answer(_round_float(q_exact, 1)),
            answer_text=d_text,
            answer_latex=latex_answer(d_text),
            marks=2,
            working=[
                f"z = InvNorm({p_text}) = {zq:.4f}",
                f"k = μ + zσ = {mu} + {zq:.4f} × {sigma} = {d_text}",
            ],
            distractors=d_dist,
            option_notes=notes(d_notes, d_dist),
        ),
    ]
    return dict(text=stem, parts=parts)
