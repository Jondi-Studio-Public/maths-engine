"""Generates questions on using the coefficient of determination to assess a linear fit (`senior-use-the-
coefficient-of-determination-to-assess-the`, ACMGM060).

One senior statistics skill, ``senior-use-the-coefficient-of-determination-to-assess-the`` (ACMGM060, General
Mathematics).

Scope: given a STATED correlation coefficient r for two variables, compute the
coefficient of determination r^2 and report it as a decimal or percentage. r is never
computed from data here -- that operation belongs to the active
``senior-use-technology-to-find-the-correlation-coefficient-an`` (ACMEM144), which
computes r from a printed bivariate data set and never squares it (one of its own
misconceptions, ``r-squared``, is a learner wrongly reporting r^2 there instead of r,
confirming the two operations are already treated as distinct).

### Construction

r is drawn directly as an exact two-decimal-place value, ``k/100`` with
``30 <= k <= 99`` (so ``0.30 <= |r| <= 0.99``, never 0 or +/-1). Because r is stated to
exactly 2 d.p., r^2 = k^2/10000 is an EXACT terminating decimal at 4 d.p. (as a decimal)
or 2 d.p. (as a percentage) -- there is never a rounding boundary to defend, unlike a
correlation coefficient computed from noisy data.

### Independent check

``_verify_r_squared`` recomputes r^2 through ``decimal.Decimal`` arithmetic (a different
module and a different code path from the ``Fraction`` multiplication construction
uses) over the identical printed r, and asserts the two agree exactly.

### Tiers (each changes the decision)

- **easy** -- one printed r, positive, report r^2 as a decimal to 4 d.p. Decision:
  square a stated r.
- **medium** -- one printed r, NEGATIVE, report the PERCENTAGE of the variation in the
  response variable explained by the explanatory variable, to 2 d.p. Decision: square a
  negative (the sign vanishes) and convert to a percentage.
- **hard** -- two explanatory variables' r values are stated against the same response
  variable; report the percentage of variation explained by the one that explains MORE.
  Decision: compare |r|, not signed r -- when a context's natural sign is negative (the sign is the CONTEXT's own natural sign, never drawn independently of
  it, e.g. age of a car / kilometres travelled against resale value, both naturally
  negative), the stronger (larger-|r|) variable's signed r is smaller (more negative)
  than the weaker variable's, so "pick the larger signed r" fails; only "pick the larger
  |r|" always works. A "positive"-sign context (e.g. hours of study / practice tests
  against exam score) carries no such trap. Distractor: the OTHER variable's r^2, as a
  percentage.
- **exam** -- two least-squares models are each described by a sentence naming the sign
  of their line's slope and stating |r|; the question asks how many percentage POINTS
  more of the variation one model explains than the other -- a DIFFERENCE of two r^2
  values (decision: square twice, convert both to percentages, then subtract), a genuine
  escalation over hard's "pick the larger one" single-value read.

Every tier's context is drawn from >= 3 distinct printed scenarios so
`mathgen.text_quality.prompt_shape` sees genuine sentence variety, not just a digit
change.
"""

from __future__ import annotations

import math
from decimal import Decimal
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
    stated_precision_string,
    stated_precision_tolerance,
)

_DEC_DP = 4
_PCT_DP = 2
_DEC_TOL = stated_precision_tolerance(_DEC_DP)
_PCT_TOL = stated_precision_tolerance(_PCT_DP)
_R_MIN = 30  # |r| >= 0.30
_R_MAX = 99  # |r| <= 0.99


def _draw_r(rng, *, negative, low=_R_MIN, high=_R_MAX) -> Fraction:
    """Draw r = k/100, |r| in [low/100, high/100], with the given sign."""
    k = rng.randint(low, high)
    return Fraction(-k if negative else k, 100)


def _r_sq(r: Fraction) -> Fraction:
    return r * r


def _verify_r_squared(r: Fraction, r2: Fraction) -> None:
    """Independent route: `decimal.Decimal` squaring of the same printed r --
    a different module/algorithm from the `Fraction` multiplication
    construction uses."""
    dec_r = Decimal(r.numerator) / Decimal(r.denominator)
    dec_r2 = dec_r * dec_r
    check_equivalent(
        Fraction(dec_r2),
        r2,
        "p19b_coefficient_determination: decimal.Decimal check disagrees with Fraction r^2",
    )


def _dec_text(v: Fraction) -> str:
    return stated_precision_string(v, _DEC_DP)


def _pct_text(v: Fraction) -> str:
    """`v` (a proportion, e.g. r^2) displayed as a percentage to `_PCT_DP`
    places -- the bare number, no '%' suffix (grading.py's typed-number path
    strips a trailing '%' from a learner's answer via its tokenizer, but the
    canonical `answer_text` itself is never given the suffix -- convention
    matches `p8e_percentage_review_build.py`)."""
    return stated_precision_string(v * 100, _PCT_DP)


def _approx_fraction(x: float, dp: int) -> Fraction:
    """Round an irrational float (e.g. a sqrt) to an exact Fraction at `dp`
    decimal places, for a distractor display only (never a graded canonical)."""
    scale = 10**dp
    return Fraction(round(x * scale), scale)


def _far_enough(a: Fraction, b: Fraction, tol: Fraction) -> bool:
    return abs(a - b) > tol


def _cap(text: str) -> str:
    return text[0].upper() + text[1:] if text else text


def _r_display(r: Fraction) -> str:
    """r, exact to 2 d.p. by construction, displayed with its sign."""
    return f"{float(r):.2f}"


# ============================================================================
# easy: one positive r, decimal form
# ============================================================================

_EASY_CONTEXTS = (
    dict(x_label="hours of practice", y_label="typing speed"),
    dict(x_label="advertising spend", y_label="monthly sales"),
    dict(x_label="age", y_label="reaction time"),
    dict(x_label="study hours per week", y_label="exam score"),
)

_EASY_TEMPLATES = (
    "The correlation coefficient between {x_label} and {y_label} for a group of subjects "
    "is r = {r}. Find the coefficient of determination, r^2, as a decimal, to 4 decimal "
    "places.",
    "For {x_label} and {y_label}, the correlation coefficient is found to be r = {r}. "
    "Report the coefficient of determination, r^2, as a decimal, to 4 decimal places.",
    "A researcher reports a correlation coefficient of r = {r} between {x_label} and "
    "{y_label}. Find r^2, as a decimal, to 4 decimal places.",
)


def _p19b_build_easy(rng):
    for _attempt in range(200):
        ctx = rng.choice(_EASY_CONTEXTS)
        r = _draw_r(rng, negative=False)
        r2 = _r_sq(r)
        _verify_r_squared(r, r2)
        answer_text = _dec_text(r2)

        candidates = []
        candidates.append(_dec_text(r))  # r itself, not squared
        candidates.append(_dec_text(2 * r))  # doubling instead of squaring
        sqrt_r = _approx_fraction(math.sqrt(float(r)), _DEC_DP)
        candidates.append(_dec_text(sqrt_r))  # took the square root instead

        seen = set()
        cand_texts = []
        for c in candidates:
            c_val = Fraction(c)
            if c_val == r2 or not _far_enough(c_val, r2, _DEC_TOL):
                continue
            if c in seen:
                continue
            seen.add(c)
            cand_texts.append(c)
        if len(cand_texts) < 2:
            continue
        try:
            check_pairwise_distinct(r2, *(Fraction(c) for c in cand_texts))
        except Exception:
            continue

        text = _cap(phrase(rng, _EASY_TEMPLATES, x_label=ctx["x_label"], y_label=ctx["y_label"], r=_r_display(r)))
        return dict(
            text=text,
            latex=text,
            answer=numeric_answer(r2),
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text),
            distractors=pick_distractors(answer_text, cand_texts, tolerance=_DEC_TOL),
            working=[f"r = {_r_display(r)}.", f"r^2 = {_r_display(r)} x {_r_display(r)} = {answer_text}."],
        )
    check(False, "p19b_coefficient_determination/easy: could not draw a valid r in 200 attempts")


# ============================================================================
# medium: one negative r, percentage form
# ============================================================================

_MEDIUM_CONTEXTS = (
    dict(x_label="hours of screen time per day", y_label="sleep duration"),
    dict(x_label="age of a car", y_label="resale value"),
    dict(x_label="minutes of daily exercise", y_label="resting heart rate"),
    dict(x_label="distance from the city centre", y_label="house price"),
)

_MEDIUM_TEMPLATES = (
    "The correlation coefficient between {x_label} and {y_label} is found to be "
    "r = {r}. Find the percentage of the variation in {y_label} explained by "
    "{x_label}, to 2 decimal places.",
    "A study reports a correlation coefficient of r = {r} between {x_label} and "
    "{y_label}. What percentage of the variation in {y_label} is explained by "
    "{x_label}? Give your answer to 2 decimal places.",
    "For {x_label} and {y_label}, r = {r}. Find, to 2 decimal places, the percentage "
    "of the variation in {y_label} that is explained by {x_label}.",
)


def _p19b_build_medium(rng):
    for _attempt in range(200):
        ctx = rng.choice(_MEDIUM_CONTEXTS)
        r = _draw_r(rng, negative=True)
        r2 = _r_sq(r)
        _verify_r_squared(r, r2)
        answer_text = _pct_text(r2)

        candidates = []
        candidates.append(_pct_text(r))  # r itself, reported as a percentage (sign kept)
        candidates.append(_pct_text(-r2))  # squared but sign kept (r was negative)
        candidates.append(_pct_text(1 - r2))  # the UNexplained share
        two_r_mag = 2 * abs(r)
        candidates.append(_pct_text(two_r_mag))  # doubling instead of squaring

        seen = set()
        cand_texts = []
        for c in candidates:
            c_val = Fraction(c) / 100
            if c_val == r2 or not _far_enough(c_val * 100, r2 * 100, _PCT_TOL):
                continue
            if c in seen:
                continue
            seen.add(c)
            cand_texts.append(c)
        if len(cand_texts) < 2:
            continue
        try:
            check_pairwise_distinct(r2 * 100, *(Fraction(c) for c in cand_texts))
        except Exception:
            continue

        text = _cap(phrase(rng, _MEDIUM_TEMPLATES, x_label=ctx["x_label"], y_label=ctx["y_label"], r=_r_display(r)))
        return dict(
            text=text,
            latex=text,
            answer=numeric_answer(r2 * 100),
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text),
            distractors=pick_distractors(answer_text, cand_texts, tolerance=_PCT_TOL),
            working=[f"r = {_r_display(r)}.", f"r^2 = {_dec_text(r2)}; as a percentage: {answer_text}."],
        )
    check(False, "p19b_coefficient_determination/medium: could not draw a valid r in 200 attempts")


# ============================================================================
# hard: two explanatory variables, same response, compare |r|
# ============================================================================

_HARD_CONTEXTS = (
    # sign="positive": BOTH explanatory variables naturally rise with y (more
    # study/practice tests -> higher score; more rainfall/fertiliser -> higher
    # yield). Naive "pick the larger signed r" agrees with "pick the larger
    # |r|" here, so no strong/weak-sign trap is possible -- correctly so.
    dict(
        y_label="final exam score",
        x1_label="hours of study per week",
        x2_label="number of practice tests completed",
        sign="positive",
    ),
    dict(
        y_label="crop yield",
        x1_label="rainfall during the growing season",
        x2_label="amount of fertiliser applied",
        sign="positive",
    ),
    # sign="negative": BOTH explanatory variables naturally fall as y rises
    # (older car / more kilometres travelled -> lower resale value; more
    # exercise / more vegetables eaten -> lower body fat). With both r
    # negative, the MORE negative (larger |r|) one looks "smaller" under a
    # naive signed comparison, so "pick the larger signed r" always picks the
    # WEAKER variable here -- the trap the hard tier is built to test.
    dict(
        y_label="resale value of a car",
        x1_label="age of the car",
        x2_label="total kilometres travelled",
        sign="negative",
    ),
    dict(
        y_label="body fat percentage",
        x1_label="hours of exercise per week",
        x2_label="servings of vegetables eaten per day",
        sign="negative",
    ),
)

_HARD_TEMPLATES = (
    "Two explanatory variables are correlated with {y_label}: {x1_label} has "
    "correlation coefficient r = {r1}, and {x2_label} has correlation coefficient "
    "r = {r2}. Which explanatory variable explains more of the variation in "
    "{y_label}? Give the percentage of variation it explains, to 2 decimal places.",
    "For {y_label}, the correlation coefficient with {x1_label} is r = {r1}, and the "
    "correlation coefficient with {x2_label} is r = {r2}. Which of the two explains "
    "more of the variation in {y_label}, and what percentage does it explain, to 2 "
    "decimal places?",
    "{x1_label} and {x2_label} are both correlated with {y_label}: r = {r1} for "
    "{x1_label} and r = {r2} for {x2_label}. Find the percentage of the variation in "
    "{y_label} explained by whichever of the two explains more, to 2 decimal places.",
)

_HARD_MIN_GAP = Fraction(5, 100)  # |r| gap between the two variables


def _p19b_build_hard(rng):
    for _attempt in range(200):
        ctx = rng.choice(_HARD_CONTEXTS)

        # Both explanatory variables in a "negative" context are negative; both in a "positive"
        # context are positive. With BOTH r negative, the stronger (larger-|r|) variable is the MORE
        # negative one, so a learner who naively compares SIGNED r picks the weaker, less-negative
        # variable -- "compare |r|, not signed r" is the only decision that gets this right. A
        # "positive" context carries no such trap (signed and |r| comparisons agree), correctly so.
        is_negative = ctx["sign"] == "negative"
        strong_mag = rng.randint(_R_MIN + 20, _R_MAX)
        weak_mag = rng.randint(_R_MIN, strong_mag - int(_HARD_MIN_GAP * 100))
        if weak_mag < _R_MIN:
            continue
        strong_r = Fraction(-strong_mag if is_negative else strong_mag, 100)
        weak_r = Fraction(-weak_mag if is_negative else weak_mag, 100)

        strong_is_x1 = rng.choice([True, False])
        r1 = strong_r if strong_is_x1 else weak_r
        r2 = weak_r if strong_is_x1 else strong_r

        r1_sq = _r_sq(r1)
        r2_sq = _r_sq(r2)
        _verify_r_squared(r1, r1_sq)
        _verify_r_squared(r2, r2_sq)

        stronger_sq = r1_sq if strong_is_x1 else r2_sq
        weaker_sq = r2_sq if strong_is_x1 else r1_sq
        check(
            stronger_sq > weaker_sq,
            "p19b_coefficient_determination/hard: stronger variable does not have the larger r^2",
        )
        answer_text = _pct_text(stronger_sq)
        stronger_r = r1 if strong_is_x1 else r2

        candidates = []
        candidates.append(_pct_text(weaker_sq))  # the OTHER (weaker) variable's r^2
        candidates.append(_pct_text(stronger_r))  # the stronger variable's r itself, not squared
        candidates.append(_pct_text(1 - stronger_sq))  # the unexplained share of the stronger one

        seen = set()
        cand_texts = []
        for c in candidates:
            c_val = Fraction(c) / 100
            if c_val == stronger_sq or not _far_enough(c_val * 100, stronger_sq * 100, _PCT_TOL):
                continue
            if c in seen:
                continue
            seen.add(c)
            cand_texts.append(c)
        if len(cand_texts) < 2:
            continue
        try:
            check_pairwise_distinct(stronger_sq * 100, *(Fraction(c) for c in cand_texts))
        except Exception:
            continue

        text = _cap(
            phrase(
                rng,
                _HARD_TEMPLATES,
                y_label=ctx["y_label"],
                x1_label=ctx["x1_label"],
                x2_label=ctx["x2_label"],
                r1=_r_display(r1),
                r2=_r_display(r2),
            )
        )
        return dict(
            text=text,
            latex=text,
            answer=numeric_answer(stronger_sq * 100),
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text),
            distractors=pick_distractors(answer_text, cand_texts, tolerance=_PCT_TOL),
            working=[
                f"r1 = {_r_display(r1)}, r1^2 = {_dec_text(r1_sq)}.",
                f"r2 = {_r_display(r2)}, r2^2 = {_dec_text(r2_sq)}.",
                f"Larger |r| gives the larger r^2: {answer_text}%.",
            ],
        )
    check(False, "p19b_coefficient_determination/hard: could not draw a valid pair in 200 attempts")


# ============================================================================
# exam: two models, difference of two r^2 values
# ============================================================================

_EXAM_CONTEXTS = (
    dict(
        y_label="sale price of a house",
        a_label="floor area",
        a_slope="positive",
        b_label="distance from the city centre",
        b_slope="negative",
    ),
    dict(
        y_label="marathon finishing time",
        a_label="weekly training distance",
        a_slope="negative",
        b_label="age",
        b_slope="positive",
    ),
    dict(
        y_label="fuel consumption of a car",
        a_label="engine size",
        a_slope="positive",
        b_label="tyre pressure",
        b_slope="negative",
    ),
)

_EXAM_TEMPLATES = (
    "Two least-squares models are fitted to predict {y_label}. Model A, using {a_label}, "
    "has a {a_slope} slope, with correlation stated as |r| = {r1}. Model B, using "
    "{b_label}, has a {b_slope} slope, with correlation stated as |r| = {r2}. How many "
    "more percentage points of the variation in {y_label} does Model A explain than "
    "Model B? Give your answer to 2 decimal places.",
    "For {y_label}, Model A (using {a_label}, a {a_slope} slope) has |r| = {r1}, and "
    "Model B (using {b_label}, a {b_slope} slope) has |r| = {r2}. How many more "
    "percentage points of the variation does Model A explain than Model B, to 2 decimal "
    "places?",
    "Predicting {y_label}: Model A, using {a_label}, has a {a_slope} slope and |r| = "
    "{r1}. Model B, using {b_label}, has a {b_slope} slope and |r| = {r2}. Find how many "
    "more percentage points of the variation in {y_label} Model A explains than Model B, "
    "to 2 decimal places.",
)

_EXAM_MIN_GAP = Fraction(5, 100)


def _p19b_build_exam(rng):
    for _attempt in range(200):
        ctx = rng.choice(_EXAM_CONTEXTS)
        r1_mag = rng.randint(_R_MIN + 20, _R_MAX)
        r2_mag = rng.randint(_R_MIN, r1_mag - int(_EXAM_MIN_GAP * 100))
        if r2_mag < _R_MIN:
            continue
        r1 = Fraction(r1_mag, 100)  # |r| for model A, the one that explains more
        r2 = Fraction(r2_mag, 100)  # |r| for model B

        r1_sq = _r_sq(r1)
        r2_sq = _r_sq(r2)
        _verify_r_squared(r1, r1_sq)
        _verify_r_squared(r2, r2_sq)
        diff = r1_sq - r2_sq
        check(diff > 0, "p19b_coefficient_determination/exam: Model A does not explain more")
        answer_text = _pct_text(diff)

        candidates = []
        candidates.append(_pct_text(r1_sq))  # Model A's own share alone, forgot to subtract
        candidates.append(_pct_text(r2_sq))  # Model B's own share alone
        candidates.append(_pct_text(r1 - r2))  # forgot to square before differencing

        seen = set()
        cand_texts = []
        for c in candidates:
            c_val = Fraction(c) / 100
            if c_val == diff or not _far_enough(c_val * 100, diff * 100, _PCT_TOL):
                continue
            if c in seen:
                continue
            seen.add(c)
            cand_texts.append(c)
        if len(cand_texts) < 2:
            continue
        try:
            check_pairwise_distinct(diff * 100, *(Fraction(c) for c in cand_texts))
        except Exception:
            continue

        text = _cap(
            phrase(
                rng,
                _EXAM_TEMPLATES,
                y_label=ctx["y_label"],
                a_label=ctx["a_label"],
                a_slope=ctx["a_slope"],
                b_label=ctx["b_label"],
                b_slope=ctx["b_slope"],
                r1=_r_display(r1),
                r2=_r_display(r2),
            )
        )
        return dict(
            text=text,
            latex=text,
            answer=numeric_answer(diff * 100),
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text),
            distractors=pick_distractors(answer_text, cand_texts, tolerance=_PCT_TOL),
            working=[
                f"Model A: |r| = {_r_display(r1)}, r^2 = {_dec_text(r1_sq)}.",
                f"Model B: |r| = {_r_display(r2)}, r^2 = {_dec_text(r2_sq)}.",
                f"Difference (percentage points): {answer_text}.",
            ],
        )
    check(False, "p19b_coefficient_determination/exam: could not draw a valid pair in 200 attempts")


_P19B_BUILDERS = {
    "easy": _p19b_build_easy,
    "medium": _p19b_build_medium,
    "hard": _p19b_build_hard,
    "exam": _p19b_build_exam,
}


@generator(
    "p19b_coefficient_determination",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def p19b_coefficient_determination(rng, tier="easy"):
    """Compute the coefficient of determination r^2 from a stated correlation
    coefficient r, reported as a decimal or percentage (ACMGM060)."""
    check(tier in _P19B_BUILDERS, f"p19b_coefficient_determination: unknown tier {tier!r}")
    return _P19B_BUILDERS[tier](rng)
