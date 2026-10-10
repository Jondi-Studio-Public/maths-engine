"""Generates questions on finding the correlation coefficient using technology (`senior-use-technology-to-find-
the-correlation-coefficient-an`, ACMEM144).

One senior statistics skill, ``senior-use-technology-to-find-the-correlation-coefficient-an`` (ACMEM144).

## ``p16b_correlation_r``

Kernel: given a small, fully stated bivariate data set (5-8
pairs, printed in the prompt), find Pearson's product-moment correlation
coefficient r between two stated variables, to 3 decimal places. "Use
technology": the platform is the technology -- nothing
here asks for the formula by hand. The parenthetical "(an indicator of the
strength of linear association)" is assessed at the hard tier (strength =
|r|).

### Construction

Every data set is built forward: distinct integer x values, then
``y = base + slope * x + noise`` (noise an integer drawn from a bounded
range), so the printed data is always small integers. r is computed EXACTLY
via the Fraction-arithmetic Pearson formula (`_pearson_r`) up to the final
``sqrt`` (irrational in general), then rounded half-up to 3 decimal places
with `stated_precision_string`. Every draw is rejected and resampled unless
0.2 <= |r| <= 0.98 (never 0 or +/-1, never a near-constant column) and the
3-d.p. rounding is unambiguous (`rounds_unambiguously(r, 3, margin=0.1)`).

### Independent check

`_verify_r` recomputes r via the STANDARD LIBRARY's own
``statistics.correlation`` (a genuinely different algorithm/implementation
from the hand-written Fraction-arithmetic formula construction uses) over
the exact same printed sample, and asserts the two agree once rounded to
3 d.p. -- never reusing `_pearson_r`.

### Tiers (each changes the decision)

- **easy** -- one data set (5-6 pairs), two named printed columns. Decision:
  enter paired data and read r.
- **medium** -- the variable of interest is NOT a printed column: it is
  derived from two printed columns (e.g. start/finish time -> time taken;
  points for/against -> winning margin; price before/after -> discount).
  r is asked between x and the DERIVED variable. Decision: construct the
  variable before correlating. Distractor: r with either raw column.
- **hard** -- two data sets A and B (5-7 pairs each), same variable names,
  different settings. "Which shows the stronger linear association? Give
  its correlation coefficient, with its sign, to 3 decimal places."
  Strength = |r|. Construct-to-target: the STRONGER set's class (negative
  or positive) is drawn first, 50/50, then realised by choosing a context
  whose own NATURAL sign matches. The WEAKER set's sign is drawn independently, 50/50, and may
  disagree with its own context's natural sign (a small/noisy sample can go
  either way at |r| <= 0.5) -- so "pick the larger r" fails on the half
  where the stronger set is negative. Tie gap |r_A| - |r_B| enforced >= 0.10
  (comfortably over the 0.05 floor) by resampling each set from a disjoint
  |r|-magnitude band. Decision: strength is |r|, not r. Distractor: the
  other set's r.
- **exam** -- frequency data: each printed row is (x, y, count), 4-6
  distinct pairs with counts 1-4; r of the FULL data set (each pair counted
  that many times). Decision: repeated observations weight the correlation.
  Distractor: r of the distinct pairs, ignoring counts (must clear the key
  by construction; resampled otherwise).

Every tier's context is drawn from >= 3 distinct printed scenarios so
`mathgen.text_quality.prompt_shape` sees genuine sentence variety, not just a
digit change.
"""

from __future__ import annotations

import math
import statistics
from fractions import Fraction

from ..core import (
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    latex_answer,
    numeric_answer,
    phrase,
    pick_distractors,
    rounds_unambiguously,
    stated_precision_string,
    stated_precision_tolerance,
)

_DP = 3
_TOL = stated_precision_tolerance(_DP)
_MIN_ABS_R = Fraction(2, 10)
_MAX_ABS_R = Fraction(98, 100)


# ============================================================================
# Shared Pearson-r machinery
# ============================================================================


def _pearson_r(xs, ys) -> float | None:
    """Exact Fraction arithmetic up to the final sqrt (construction route)."""
    n = len(xs)
    check(len(ys) == n and n >= 2, "p16b_correlation_r: mismatched/short data")
    xbar = Fraction(sum(xs), n)
    ybar = Fraction(sum(ys), n)
    sxy = sum((Fraction(x) - xbar) * (Fraction(y) - ybar) for x, y in zip(xs, ys))
    sxx = sum((Fraction(x) - xbar) ** 2 for x in xs)
    syy = sum((Fraction(y) - ybar) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return float(sxy) / math.sqrt(float(sxx) * float(syy))


def _round_r(r: float) -> Fraction:
    """r rounded half-up to `_DP` places, landed exactly on the grid (the
    caller has already asserted `rounds_unambiguously`, so ordinary
    round-half-to-even vs round-half-up disagreement never bites here)."""
    scaled = round(r * (10**_DP))
    return Fraction(scaled, 10**_DP)


def _r_text(fr: Fraction) -> str:
    return stated_precision_string(fr, _DP)


def _verify_r(xs, ys, r_rounded: Fraction):
    """Independent route: `statistics.correlation` (stdlib), a different
    implementation from `_pearson_r`'s hand-written Fraction formula, over
    the identical printed sample."""
    r2 = statistics.correlation(xs, ys)
    check(
        rounds_unambiguously(r2, _DP, margin=0.1),
        "p16b_correlation_r: independent recomputation lands on a rounding boundary",
    )
    check_equivalent(
        _round_r(r2),
        r_rounded,
        "p16b_correlation_r: statistics.correlation disagrees with the constructed r",
    )


def _fmt_pairs(xs, ys) -> str:
    return ", ".join(f"({x}, {y})" for x, y in zip(xs, ys))


def _candidate_fraction(r: float | None) -> Fraction | None:
    if r is None:
        return None
    if not rounds_unambiguously(r, _DP, margin=0.05):
        return None
    return _round_r(r)


def _cap(text: str) -> str:
    """Every prompt starts with an uppercase letter or a digit, regardless of which template/context was
    drawn (a context label used at the start of a template can otherwise be lowercase)."""
    return text[0].upper() + text[1:] if text else text


# ============================================================================
# easy: one data set, two printed columns
# ============================================================================

_EASY_CONTEXTS = (
    dict(
        x_label="hours studied",
        y_label="test mark (%)",
        x_bare="hours studied",
        y_bare="test mark",
        units=", with test mark in percent",
        n_note="students",
        x_low=1,
        x_high=12,
        base=32,
        slope=Fraction(4, 1),
        noise=7,
        y_low=0,
        y_high=100,
    ),
    dict(
        x_label="daily maximum temperature (°C)",
        y_label="ice-cream sales ($)",
        x_bare="daily maximum temperature",
        y_bare="ice-cream sales",
        units=", with temperature in °C and sales in dollars",
        n_note="days",
        x_low=14,
        x_high=36,
        base=40,
        slope=Fraction(6, 1),
        noise=25,
        y_low=0,
        y_high=400,
    ),
    dict(
        x_label="age (years)",
        y_label="reaction time (ms)",
        x_bare="age",
        y_bare="reaction time",
        units=", with age in years and reaction time in milliseconds",
        n_note="participants",
        x_low=15,
        x_high=68,
        base=160,
        slope=Fraction(3, 2),
        noise=18,
        y_low=100,
        y_high=520,
    ),
    dict(
        x_label="arm span (cm)",
        y_label="height (cm)",
        x_bare="arm span",
        y_bare="height",
        units=", with arm span and height both in centimetres",
        n_note="people",
        x_low=140,
        x_high=198,
        base=10,
        slope=Fraction(9, 10),
        noise=7,
        y_low=120,
        y_high=210,
    ),
)

# The bare (unit-free) labels name the tuple order; units, when the context has any, are stated once
# in a trailing clause, never nested inside the tuple notation.
_EASY_TEMPLATES = (
    "{n} {n_note} gave these ({x_bare}, {y_bare}) pairs{units}: {pairs}. Using "
    "technology, find the correlation coefficient, r, between {x_bare} and {y_bare}, "
    "to 3 decimal places.",
    "The correlation between {x_label} and {y_label} is wanted for {n} {n_note}. The "
    "({x_bare}, {y_bare}) pairs are: {pairs}. Use technology to find r, to 3 decimal "
    "places.",
    "For {n} {n_note}, these ({x_bare}, {y_bare}) pairs were recorded{units}: {pairs}. "
    "Using technology, calculate the correlation coefficient, r, to 3 decimal places.",
)


def _draw_linear(rng, x_low, x_high, base, slope, noise, y_low, y_high, n):
    xs = rng.sample(range(x_low, x_high + 1), n)
    ys = []
    for x in xs:
        y = round(base + slope * x) + rng.randint(-noise, noise)
        ys.append(y)
    if any(y < y_low or y > y_high for y in ys):
        return None, None
    return xs, ys


def _p16b_build_easy(rng):
    for _attempt in range(600):
        ctx = rng.choice(_EASY_CONTEXTS)
        n = rng.choice([5, 6])
        xs, ys = _draw_linear(
            rng, ctx["x_low"], ctx["x_high"], ctx["base"], ctx["slope"], ctx["noise"], ctx["y_low"], ctx["y_high"], n
        )
        if xs is None:
            continue
        r = _pearson_r(xs, ys)
        if r is None or not (float(_MIN_ABS_R) <= abs(r) <= float(_MAX_ABS_R)):
            continue
        if not rounds_unambiguously(r, _DP, margin=0.1):
            continue
        r_rounded = _round_r(r)
        _verify_r(xs, ys, r_rounded)

        # Distractors: negated (only offered when |r| >= 0.05, so it always
        # clears the grader by a comfortable margin); r-squared (the
        # coefficient of determination); mispaired (the y column read
        # bottom-to-top against the same x order).
        candidates = []
        # Measured over 200 seeds, the negated key was among the distractors in 200/200 draws of
        # every tier -- every question offered exactly one +/- pair, and the key was always one of
        # its two members. Draw it with the rng on only about half of draws (the other genuine
        # misconceptions already in `candidates` cover the rest).
        if abs(r_rounded) >= Fraction(5, 100) and rng.random() < 0.5:
            candidates.append(_candidate_fraction(-r))
        candidates.append(_candidate_fraction(r * r))
        ys_reversed = list(reversed(ys))
        r_mispaired = _pearson_r(xs, ys_reversed)
        candidates.append(_candidate_fraction(r_mispaired))
        candidates = [c for c in candidates if c is not None and c != r_rounded]
        if len(candidates) < 2:
            continue
        try:
            check_pairwise_distinct(r_rounded, *candidates)
        except Exception:
            continue

        answer_text = _r_text(r_rounded)
        cand_texts = [_r_text(c) for c in candidates]
        text = _cap(
            phrase(
                rng,
                _EASY_TEMPLATES,
                n=n,
                n_note=ctx["n_note"],
                x_label=ctx["x_label"],
                y_label=ctx["y_label"],
                x_bare=ctx["x_bare"],
                y_bare=ctx["y_bare"],
                units=ctx["units"],
                pairs=_fmt_pairs(xs, ys),
            )
        )
        return dict(
            text=text,
            latex=text,
            answer=numeric_answer(r_rounded),
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text),
            distractors=pick_distractors(answer_text, cand_texts, tolerance=_TOL),
            working=[
                f"n = {n}: {_fmt_pairs(xs, ys)}.",
                f"r = sum((x - xbar)(y - ybar)) / sqrt(sum((x - xbar)^2) sum((y - ybar)^2)) = {answer_text}.",
            ],
        )
    check(False, "p16b_correlation_r/easy: could not draw a valid data set in 600 attempts")


# ============================================================================
# medium: derived variable
# ============================================================================

_MEDIUM_CONTEXTS = (
    dict(
        x_label="distance run (km)",
        p_label="start time (min)",
        q_label="finish time (min)",
        derived_label="time taken (min)",
        x_bare="distance run",
        p_bare="start time",
        q_bare="finish time",
        derived_bare="time taken",
        units=", with distance in km and both times in minutes",
        mode="q_minus_p",
        x_low=2,
        x_high=15,
        base=6,
        slope=Fraction(3, 1),
        noise=4,
        d_low=1,
        d_high=200,
        other_low=0,
        other_high=60,
    ),
    dict(
        x_label="number of shots taken",
        p_label="points scored",
        q_label="points conceded",
        derived_label="winning margin (points)",
        x_bare="number of shots taken",
        p_bare="points scored",
        q_bare="points conceded",
        derived_bare="winning margin",
        units="",
        mode="p_minus_q",
        x_low=8,
        x_high=40,
        base=-10,
        slope=Fraction(1, 2),
        noise=6,
        d_low=-40,
        d_high=40,
        other_low=10,
        other_high=80,
    ),
    dict(
        x_label="number of items bought",
        p_label="price before discount ($)",
        q_label="price after discount ($)",
        derived_label="discount amount ($)",
        x_bare="number of items bought",
        p_bare="price before discount",
        q_bare="price after discount",
        derived_bare="discount amount",
        units=", with both prices in dollars",
        mode="p_minus_q",
        x_low=1,
        x_high=20,
        base=4,
        slope=Fraction(3, 1),
        noise=5,
        d_low=1,
        d_high=180,
        other_low=8,
        other_high=200,
    ),
)

# The derived variable's own name keeps its unit parenthesis (single level, not nested) but is never
# followed by a second bracketed clause -- "the {derived_label} is {diff_phrase}", not
# "{derived_label} ({diff_phrase})".
_MEDIUM_TEMPLATES = (
    "For {n} cases, these ({x_bare}, {p_bare}, {q_bare}) triples were recorded{units}: "
    "{triples}. The {derived_label} is {diff_phrase}. Using technology, find the "
    "correlation coefficient, r, between {x_bare} and the {derived_bare}, to 3 decimal "
    "places.",
    "{n} rows of ({x_bare}, {p_bare}, {q_bare}) data were recorded{units}: {triples}. "
    "The {derived_label} is {diff_phrase} for each row. Use technology to find the "
    "correlation coefficient, r, between {x_bare} and the {derived_bare}, to 3 decimal "
    "places.",
    "Data for {n} cases{units}: ({x_bare}, {p_bare}, {q_bare}) = {triples}. Take the "
    "{derived_label} to be {diff_phrase} for every row, and find the correlation "
    "coefficient, r, between {x_bare} and the {derived_bare}, to 3 decimal places.",
)


def _p16b_build_medium(rng):
    for _attempt in range(600):
        ctx = rng.choice(_MEDIUM_CONTEXTS)
        n = rng.choice([5, 6])
        xs = rng.sample(range(ctx["x_low"], ctx["x_high"] + 1), n)
        derived = []
        for x in xs:
            d = round(ctx["base"] + ctx["slope"] * x) + rng.randint(-ctx["noise"], ctx["noise"])
            derived.append(d)
        if any(d < ctx["d_low"] or d > ctx["d_high"] for d in derived):
            continue
        others = [rng.randint(ctx["other_low"], ctx["other_high"]) for _ in xs]
        # mode "q_minus_p": derived = q - p, "other" is p (the baseline column).
        # mode "p_minus_q": derived = p - q, "other" is q (the baseline column).
        if ctx["mode"] == "q_minus_p":
            p_vals = others
            q_vals = [p + d for p, d in zip(p_vals, derived)]
        else:
            q_vals = others
            p_vals = [q + d for q, d in zip(q_vals, derived)]
        if any(v < 0 for v in p_vals) or any(v < 0 for v in q_vals):
            continue

        r = _pearson_r(xs, derived)
        if r is None or not (float(_MIN_ABS_R) <= abs(r) <= float(_MAX_ABS_R)):
            continue
        if not rounds_unambiguously(r, _DP, margin=0.1):
            continue
        r_rounded = _round_r(r)
        _verify_r(xs, derived, r_rounded)

        candidates = []
        r_raw_p = _pearson_r(xs, p_vals)
        r_raw_q = _pearson_r(xs, q_vals)
        candidates.append(_candidate_fraction(r_raw_p))
        candidates.append(_candidate_fraction(r_raw_q))
        # Measured over 200 seeds, the negated key was among the distractors in 200/200 draws of
        # every tier -- every question offered exactly one +/- pair, and the key was always one of
        # its two members. Draw it with the rng on only about half of draws (the other genuine
        # misconceptions already in `candidates` cover the rest).
        if abs(r_rounded) >= Fraction(5, 100) and rng.random() < 0.5:
            candidates.append(_candidate_fraction(-r))
        candidates = [c for c in candidates if c is not None and c != r_rounded]
        if len(candidates) < 2:
            continue
        try:
            check_pairwise_distinct(r_rounded, *candidates)
        except Exception:
            continue

        answer_text = _r_text(r_rounded)
        cand_texts = [_r_text(c) for c in candidates]
        triples = ", ".join(f"({x}, {p}, {q})" for x, p, q in zip(xs, p_vals, q_vals))
        diff_phrase = (
            f"{ctx['q_label']} minus {ctx['p_label']}"
            if ctx["mode"] == "q_minus_p"
            else f"{ctx['p_label']} minus {ctx['q_label']}"
        )
        text = _cap(
            phrase(
                rng,
                _MEDIUM_TEMPLATES,
                n=n,
                x_label=ctx["x_label"],
                p_label=ctx["p_label"],
                q_label=ctx["q_label"],
                derived_label=ctx["derived_label"],
                x_bare=ctx["x_bare"],
                p_bare=ctx["p_bare"],
                q_bare=ctx["q_bare"],
                derived_bare=ctx["derived_bare"],
                units=ctx["units"],
                triples=triples,
                diff_phrase=diff_phrase,
            )
        )
        return dict(
            text=text,
            latex=text,
            answer=numeric_answer(r_rounded),
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text),
            distractors=pick_distractors(answer_text, cand_texts, tolerance=_TOL),
            working=[
                f"{ctx['derived_label']} for each row: {', '.join(str(d) for d in derived)}.",
                f"r(x, {ctx['derived_label']}) = {answer_text}.",
            ],
        )
    check(False, "p16b_correlation_r/medium: could not draw a valid data set in 600 attempts")


# ============================================================================
# hard: two data sets, strength = |r|
# ============================================================================

_HARD_CONTEXTS = (
    # Positive-natural (more of x genuinely goes with more of y):
    dict(
        x_label="weekly training hours",
        y_label="fitness score",
        x_low=1,
        x_high=15,
        base=30,
        slope=Fraction(4, 1),
        y_low=0,
        y_high=100,
    ),
    dict(
        x_label="advertising spend ($100s)",
        y_label="monthly sales ($1000s)",
        x_low=1,
        x_high=20,
        base=10,
        slope=Fraction(3, 1),
        y_low=0,
        y_high=120,
    ),
    dict(
        x_label="hours of study per week",
        y_label="exam score (%)",
        x_low=1,
        x_high=15,
        base=35,
        slope=Fraction(3, 1),
        y_low=0,
        y_high=100,
    ),
    # Negative-natural (more of x genuinely goes with less of y):
    dict(
        x_label="hours of screen time per day",
        y_label="sleep duration (minutes)",
        x_low=1,
        x_high=12,
        base=555,
        slope=Fraction(-15, 1),
        y_low=300,
        y_high=600,
    ),
    dict(
        x_label="age of the car (years)",
        y_label="resale value ($1000s)",
        x_low=1,
        x_high=15,
        base=60,
        slope=Fraction(-3, 1),
        y_low=0,
        y_high=100,
    ),
    dict(
        x_label="minutes of daily exercise",
        y_label="resting heart rate (bpm)",
        x_low=5,
        x_high=95,
        base=95,
        slope=Fraction(-1, 2),
        y_low=25,
        y_high=110,
    ),
)

_HARD_TEMPLATES = (
    "Two data sets recorded {x_label} and {y_label} in two different groups. Data set A: "
    "{pairs_a}. Data set B: {pairs_b}. Which data set shows the stronger linear association "
    "between {x_label} and {y_label}? Using technology, give its correlation coefficient, "
    "with its sign, to 3 decimal places.",
    "{x_label} and {y_label} were measured for two separate groups. Group A gave "
    "these pairs: {pairs_a}. Group B gave: {pairs_b}. Which group shows "
    "the stronger linear association between {x_label} and {y_label}? Give its correlation "
    "coefficient, with its sign, to 3 decimal places.",
    "Two independent samples recorded {x_label} against {y_label}. Sample A: {pairs_a}. "
    "Sample B: {pairs_b}. Which sample shows the stronger linear association? Using "
    "technology, state its correlation coefficient, with its sign, to 3 decimal places.",
)

_HARD_STRONG_LOW, _HARD_STRONG_HIGH = Fraction(65, 100), Fraction(95, 100)
_HARD_WEAK_LOW, _HARD_WEAK_HIGH = Fraction(2, 10), Fraction(5, 10)


def _draw_signed_set(rng, ctx, n, want_negative, abs_low, abs_high, noise_lo, noise_hi):
    # `abs(ctx["slope"])` -- the caller controls the printed sign via
    # `want_negative`. For the STRONGER set, the caller
    # only ever passes a `ctx` whose own natural sign already matches
    # `want_negative` (see `_p16b_build_hard`), so this draw's sign is never
    # forced away from a real association. For the WEAKER set, either
    # sign is allowed (a small/noisy sample can go either way at
    # |r| <= 0.5), so `want_negative` there is drawn independently.
    slope_dir = -1 if want_negative else 1
    slope_mag = abs(ctx["slope"])
    for _ in range(400):
        xs = rng.sample(range(ctx["x_low"], ctx["x_high"] + 1), n)
        noise = rng.randint(noise_lo, noise_hi)
        ys = []
        for x in xs:
            y = round(ctx["base"] + slope_dir * slope_mag * x) + rng.randint(-noise, noise)
            ys.append(y)
        if any(y < ctx["y_low"] or y > ctx["y_high"] for y in ys):
            continue
        r = _pearson_r(xs, ys)
        if r is None:
            continue
        if want_negative and r >= 0:
            continue
        if not want_negative and r <= 0:
            continue
        if not (float(abs_low) <= abs(r) <= float(abs_high)):
            continue
        if not rounds_unambiguously(r, _DP, margin=0.1):
            continue
        return xs, ys, r
    return None, None, None


def _p16b_build_hard(rng):
    for _attempt in range(300):
        # Draw the class FIRST: the stronger set is
        # negative or positive, 50/50 -- then pick a context whose own
        # natural sign matches, rather than picking a context and then
        # forcing an arbitrary sign onto it.
        strong_negative = rng.choice([True, False])
        matching = [c for c in _HARD_CONTEXTS if (c["slope"] < 0) == strong_negative]
        check(matching, "p16b_correlation_r/hard: no context matches the drawn natural sign")
        ctx = rng.choice(matching)

        n_a = rng.choice([5, 6, 7])
        n_b = rng.choice([5, 6, 7])

        weak_negative = rng.choice([True, False])
        strong_is_a = rng.choice([True, False])

        strong_n = n_a if strong_is_a else n_b
        weak_n = n_b if strong_is_a else n_a

        strong_xs, strong_ys, strong_r = _draw_signed_set(
            rng,
            ctx,
            strong_n,
            strong_negative,
            _HARD_STRONG_LOW,
            _HARD_STRONG_HIGH,
            2,
            6,
        )
        if strong_xs is None:
            continue
        weak_xs, weak_ys, weak_r = _draw_signed_set(
            rng,
            ctx,
            weak_n,
            weak_negative,
            _HARD_WEAK_LOW,
            _HARD_WEAK_HIGH,
            6,
            14,
        )
        if weak_xs is None:
            continue
        if abs(strong_r) - abs(weak_r) < 0.05:
            continue

        if strong_is_a:
            xs_a, ys_a, r_a = strong_xs, strong_ys, strong_r
            xs_b, ys_b, r_b = weak_xs, weak_ys, weak_r
        else:
            xs_a, ys_a, r_a = weak_xs, weak_ys, weak_r
            xs_b, ys_b, r_b = strong_xs, strong_ys, strong_r

        r_a_rounded = _round_r(r_a)
        r_b_rounded = _round_r(r_b)
        _verify_r(xs_a, ys_a, r_a_rounded)
        _verify_r(xs_b, ys_b, r_b_rounded)
        if abs(r_a_rounded) - abs(r_b_rounded) == 0:
            continue

        answer_r = r_a_rounded if strong_is_a else r_b_rounded
        other_r = r_b_rounded if strong_is_a else r_a_rounded

        candidates = [other_r]
        # Draw the negated key with the rng on only about half of draws.
        if abs(answer_r) >= Fraction(5, 100) and rng.random() < 0.5:
            candidates.append(_candidate_fraction(-float(answer_r)))
        candidates.append(_candidate_fraction(float(answer_r) * float(answer_r)))
        candidates = [c for c in candidates if c is not None and c != answer_r]
        if len(candidates) < 2:
            continue
        try:
            check_pairwise_distinct(answer_r, *candidates)
        except Exception:
            continue

        answer_text = _r_text(answer_r)
        cand_texts = [_r_text(c) for c in candidates]
        text = _cap(
            phrase(
                rng,
                _HARD_TEMPLATES,
                x_label=ctx["x_label"],
                y_label=ctx["y_label"],
                pairs_a=_fmt_pairs(xs_a, ys_a),
                pairs_b=_fmt_pairs(xs_b, ys_b),
            )
        )
        return dict(
            text=text,
            latex=text,
            answer=numeric_answer(answer_r),
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text),
            distractors=pick_distractors(answer_text, cand_texts, tolerance=_TOL),
            working=[
                f"Data set A: {_fmt_pairs(xs_a, ys_a)}; r_A = {_r_text(r_a_rounded)}.",
                f"Data set B: {_fmt_pairs(xs_b, ys_b)}; r_B = {_r_text(r_b_rounded)}.",
                f"|r_A| vs |r_B|: the stronger association is the one with the larger absolute value, {answer_text}.",
            ],
        )
    check(False, "p16b_correlation_r/hard: could not draw a valid pair of data sets in 300 attempts")


# ============================================================================
# exam: frequency-weighted data
# ============================================================================

_EXAM_CONTEXTS = (
    dict(
        x_label="hours of practice",
        y_label="number of typing errors",
        x_bare="hours of practice",
        y_bare="number of typing errors",
        units="",
        x_low=1,
        x_high=10,
        base=22,
        slope=Fraction(-2, 1),
        noise=3,
        y_low=0,
        y_high=40,
    ),
    dict(
        x_label="temperature (°C)",
        y_label="cups of hot chocolate sold",
        x_bare="temperature",
        y_bare="cups of hot chocolate sold",
        units=", with temperature in °C",
        x_low=0,
        x_high=20,
        base=35,
        slope=Fraction(-3, 2),
        noise=4,
        y_low=0,
        y_high=60,
    ),
    dict(
        x_label="age (years)",
        y_label="resting heart rate (bpm)",
        x_bare="age",
        y_bare="resting heart rate",
        units=", with age in years and heart rate in beats per minute",
        x_low=20,
        x_high=70,
        base=50,
        slope=Fraction(1, 3),
        noise=4,
        y_low=40,
        y_high=100,
    ),
)

_EXAM_TEMPLATES = (
    "A survey recorded {m} distinct ({x_bare}, {y_bare}) pairs{units}, each observed a "
    "stated number of times (the frequency): {rows}. Using technology, find the "
    "correlation coefficient, r, of the FULL data set (counting each pair as many times "
    "as its frequency states), to 3 decimal places.",
    "These {m} distinct ({x_bare}, {y_bare}) pairs{units}, were each observed a stated "
    "number of times: {rows}. Treating each pair as occurring its stated number of "
    "times, use technology to find the correlation coefficient, r, of the full data set, "
    "to 3 decimal places.",
    "{m} distinct combinations of {x_bare} and {y_bare} were observed{units}. Each was "
    "observed a stated number of times: {rows}. Find the correlation coefficient, r, of "
    "the full data set (each pair repeated its stated frequency), to 3 decimal places.",
)


def _p16b_build_exam(rng):
    for _attempt in range(600):
        ctx = rng.choice(_EXAM_CONTEXTS)
        m = rng.choice([4, 5, 6])
        xs = rng.sample(range(ctx["x_low"], ctx["x_high"] + 1), m)
        ys = []
        for x in xs:
            y = round(ctx["base"] + ctx["slope"] * x) + rng.randint(-ctx["noise"], ctx["noise"])
            ys.append(y)
        if any(y < ctx["y_low"] or y > ctx["y_high"] for y in ys):
            continue
        counts = [rng.randint(1, 4) for _ in xs]

        expanded_xs, expanded_ys = [], []
        for x, y, c in zip(xs, ys, counts):
            expanded_xs += [x] * c
            expanded_ys += [y] * c

        r = _pearson_r(expanded_xs, expanded_ys)
        if r is None or not (float(_MIN_ABS_R) <= abs(r) <= float(_MAX_ABS_R)):
            continue
        if not rounds_unambiguously(r, _DP, margin=0.1):
            continue
        r_rounded = _round_r(r)
        _verify_r(expanded_xs, expanded_ys, r_rounded)

        r_ignores_counts = _pearson_r(xs, ys)
        ignores_counts_fraction = _candidate_fraction(r_ignores_counts)
        # The ignores-counts distractor must clear the key by
        # construction, or the draw is resampled -- a full resample of the WHOLE
        # draw (not merely dropping this one candidate), since it is the one
        # distractor this tier exists to encode.
        if ignores_counts_fraction is None or ignores_counts_fraction == r_rounded:
            continue
        candidates = [ignores_counts_fraction]
        # Measured over 200 seeds, the negated key was among the distractors in 200/200 draws of
        # every tier -- every question offered exactly one +/- pair, and the key was always one of
        # its two members. Draw it with the rng on only about half of draws (the other genuine
        # misconceptions already in `candidates` cover the rest).
        if abs(r_rounded) >= Fraction(5, 100) and rng.random() < 0.5:
            candidates.append(_candidate_fraction(-r))
        candidates.append(_candidate_fraction(r * r))
        candidates = [c for c in candidates if c is not None and c != r_rounded]
        if len(candidates) < 2:
            continue
        try:
            check_pairwise_distinct(r_rounded, *candidates)
        except Exception:
            continue

        answer_text = _r_text(r_rounded)
        cand_texts = [_r_text(c) for c in candidates]
        rows = ", ".join(f"({x}, {y}, freq {c})" for x, y, c in zip(xs, ys, counts))
        text = _cap(
            phrase(
                rng,
                _EXAM_TEMPLATES,
                m=m,
                x_label=ctx["x_label"],
                y_label=ctx["y_label"],
                x_bare=ctx["x_bare"],
                y_bare=ctx["y_bare"],
                units=ctx["units"],
                rows=rows,
            )
        )
        return dict(
            text=text,
            latex=text,
            answer=numeric_answer(r_rounded),
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text),
            distractors=pick_distractors(answer_text, cand_texts, tolerance=_TOL),
            working=[
                f"Distinct pairs and frequencies: {rows}.",
                f"Full (weighted) data set has {sum(counts)} observations; r = {answer_text}.",
            ],
        )
    check(False, "p16b_correlation_r/exam: could not draw a valid frequency table in 600 attempts")


_P16B_BUILDERS = {
    "easy": _p16b_build_easy,
    "medium": _p16b_build_medium,
    "hard": _p16b_build_hard,
    "exam": _p16b_build_exam,
}


@generator(
    "p16b_correlation_r",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p16b_correlation_r(rng, tier="easy"):
    """Find Pearson's product-moment correlation coefficient r, to 3 decimal
    places, for a small stated bivariate data set (ACMEM144)."""
    check(tier in _P16B_BUILDERS, f"p16b_correlation_r: unknown tier {tier!r}")
    return _P16B_BUILDERS[tier](rng)
