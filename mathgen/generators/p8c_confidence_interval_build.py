"""Approximate confidence intervals for a mean AND a proportion (co-located pair).

- ``p8c_confidence_interval_mean``
  (``senior-examine-the-approximate-confidence-interval-left-overline-mathrm``,
  ACMSM141, domain statistics-probability, answer: pair)
  -- both bounds of (x-bar - z*s/sqrt(n), x-bar + z*s/sqrt(n)) as an interval
  estimate for the population mean. Context: delivery times, in minutes, of a
  sample of parcels handled by a courier depot.

- ``p8c_confidence_interval_proportion``
  (``senior-use-the-approximate-confidence-interval-left-widehat-p``,
  ACMMM178, domain statistics-probability, answer: pair)
  -- both bounds of (phat - z*sqrt(phat*(1-phat)/n), phat + z*sqrt(phat*(1-phat)/n))
  as an interval estimate for the population proportion. Context: a survey of
  voters. Reported as PERCENTAGES to 1dp (not raw 0-1 proportions to 2-3dp):
  `mathgen/grading/typed.py::grade_pair`'s tolerance (`_PAIR_TOLERANCE = 0.051`) is
  FIXED and does not scale with displayed decimal places the way
  `grade_number` does, so grading the raw proportion to 2-3dp would let a
  genuinely different misconception distractor collide with the correct
  answer under that tolerance.

The two skills are co-located deliberately: they share the Z_VALUES table and the resample-until-unambiguous
idiom, and this module is the place their cross-skill vocabulary boundary is enforced by construction --

- the mean skill's question text ALWAYS says "sample mean", "sample standard
  deviation" (or "sample variance"), "population mean" and "mu", and NEVER
  "proportion"/"phat"/counts of successes;
- the proportion skill's question text ALWAYS says "sample proportion"
  ("phat"), "population proportion" and "p", and NEVER "mean"/"standard
  deviation" or a measured continuous quantity.

`z` is always either stated directly in the question text (a named confidence level with its z value in
parentheses) or looked up from a three-row table given inline in the question text itself (the `hard` tier) --
never assumed memorised, matching `p5d_interval_traffic_build.py`'s own convention.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import (
    GRADER_TOLERANCE,
    check,
    generator,
    latex_answer,
    pick_distractors,
    rounds_unambiguously,
)

Z_VALUES = {"90%": 1.645, "95%": 1.96, "99%": 2.576}


def _verify_interval(center: float, lower: float, upper: float, margin_independent: float, eps: float = 1e-6):
    """Independent check, two routes, mirroring
    `coordinate_geometry_build.py`'s `_verify_midpoint`: (1) the midpoint
    identity -- averaging the constructed bounds must return the centre
    (sample mean or sample proportion) they were built around; (2) the margin
    identity -- half the constructed bounds' spread must equal a margin
    computed a genuinely different way (through squares and a single sqrt,
    from raw inputs, never by re-using construction's own `margin`
    expression). Operates on RAW, unrounded values so the identity is exact
    up to floating-point noise, never confounded by display rounding; the
    generator rounds separately, only for `answer`/`answer_text`. Module-level
    so a test can call it directly with a flipped sign or a perturbed bound."""
    check(
        abs((lower + upper) / 2 - center) < eps,
        f"midpoint identity failed: (lower+upper)/2={(lower + upper) / 2} != center={center}",
    )
    check(
        abs((upper - lower) / 2 - margin_independent) < eps,
        f"margin identity failed: (upper-lower)/2={(upper - lower) / 2} "
        f"!= independently computed margin={margin_independent}",
    )


# ============================================================================
# Skill A: p8c_confidence_interval_mean
# ============================================================================

_MEAN_XBAR_RANGE = range(60, 301)  # minutes -- kept comfortably above any
# plausible margin so a positive lower
# bound is the common case, not a fluke.
_MEAN_S_RANGE = range(5, 41)
_MEAN_N_RANGE = range(16, 101)


@generator(
    "p8c_confidence_interval_mean",
    level=12,
    presets={
        "easy": dict(give_variance=False, z_lookup=False),
        "medium": dict(give_variance=True, z_lookup=False),
        "hard": dict(give_variance=True, z_lookup=True),
    },
)
def p8c_confidence_interval_mean(rng, give_variance=False, z_lookup=False):
    """Examine the approximate confidence interval (x-bar - z*s/sqrt(n),
    x-bar + z*s/sqrt(n)) for the population mean, given bounded, fully
    specified inputs (ACMSM141)."""
    for _attempt in range(500):
        x_bar = rng.choice(_MEAN_XBAR_RANGE)
        s = rng.choice(_MEAN_S_RANGE)
        n = rng.choice(_MEAN_N_RANGE)
        confidence = rng.choice(list(Z_VALUES))
        z = Z_VALUES[confidence]
        variance = s * s

        margin = z * s / math.sqrt(n)
        lower = x_bar - margin
        upper = x_bar + margin
        if not (rounds_unambiguously(lower, 1) and rounds_unambiguously(upper, 1)):
            continue
        lower_r = round(lower, 1)
        upper_r = round(upper, 1)
        if lower_r <= 0:
            continue  # a negative delivery time is not a sensible bound

        answer_text = f"lower = {lower_r:.1f}, upper = {upper_r:.1f}"

        # Misconception distractors, formatted exactly like the answer.
        m_no_sqrt_n = z * s  # omits-sqrt-n
        m_no_z = s / math.sqrt(n)  # omits-z
        candidate_pairs = [
            (round(x_bar - m_no_sqrt_n, 1), round(x_bar + m_no_sqrt_n, 1)),
            (round(x_bar - m_no_z, 1), round(x_bar + m_no_z, 1)),
            (round(x_bar, 1), round(x_bar + 2 * margin, 1)),  # one-sided-margin
        ]
        candidate_texts = [f"lower = {lo:.1f}, upper = {hi:.1f}" for lo, hi in candidate_pairs]

        # Every distractor pair must clear grade_pair's fixed 0.051 tolerance
        # in at least one component (Fraction comparison on the *displayed*
        # 1dp values, never a float ==/subtraction), or a resample is forced --
        # do not rely on the parameter ranges alone.
        gap_ok = all(
            abs(Fraction(str(lo)) - Fraction(str(lower_r))) > GRADER_TOLERANCE
            or abs(Fraction(str(hi)) - Fraction(str(upper_r))) > GRADER_TOLERANCE
            for lo, hi in candidate_pairs
        )
        if not gap_ok:
            continue
        break
    else:
        raise RuntimeError(
            "p8c_confidence_interval_mean: no draw satisfied unambiguous rounding, "
            "a positive lower bound and distractor separation in 500 attempts"
        )

    # Independent check: a genuinely different route from construction's
    # z*s/sqrt(n) -- through squares and a single sqrt, sqrt(z**2*s**2/n) --
    # never reusing the `margin` variable's own expression.
    margin_independent = math.sqrt((z**2) * (s**2) / n)
    _verify_interval(x_bar, lower, upper, margin_independent)

    distractors = pick_distractors(answer_text, candidate_texts, tolerance=GRADER_TOLERANCE)
    check(len(distractors) == 3, "a distractor collided with the answer within grader tolerance")

    stat_clause = (
        f"a sample standard deviation of {s} minutes"
        if not give_variance
        else f"a sample variance of {variance} square minutes"
    )
    base = (
        f"A sample of {n} parcels processed by a courier depot has a sample mean "
        f"delivery time of {x_bar} minutes and {stat_clause}. "
    )
    if not z_lookup:
        instruction = (
            f"Using a {confidence} confidence level (z = {z}), calculate the approximate "
            f"confidence interval for the population mean delivery time, mu, giving both "
            f"bounds to 1 decimal place, in minutes."
        )
    else:
        instruction = (
            f"Given that a 90% confidence level uses z = 1.645, a 95% confidence level "
            f"uses z = 1.96, and a 99% confidence level uses z = 2.576, calculate the "
            f"approximate confidence interval for the population mean delivery time, mu, "
            f"at the {confidence} confidence level, giving both bounds to 1 decimal place, "
            f"in minutes."
        )
    question_text = base + instruction

    working = []
    if give_variance:
        working.append(f"s = the square root of {variance} = {s} minutes.")
    working.append(
        f"Margin of error = z times s divided by the square root of n "
        f"= {z} times {s} divided by the square root of {n} = {round(margin, 3)}."
    )
    working.append(
        f"Lower bound = {x_bar} minus {round(margin, 3)} = {lower_r:.1f}. "
        f"Upper bound = {x_bar} plus {round(margin, 3)} = {upper_r:.1f}."
    )

    return {
        "text": question_text,
        "latex": f"plain:{question_text}",
        "answer": {"lower": lower_r, "upper": upper_r},
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer_text, plain=True),
        "distractors": distractors,
        "working": working,
    }


# ============================================================================
# Skill B: p8c_confidence_interval_proportion
# ============================================================================

_PROP_PHAT_HUNDREDTHS = range(10, 91)  # 0.10 to 0.90 in increments of 0.01
_PROP_N_RANGE = range(50, 301)


@generator(
    "p8c_confidence_interval_proportion",
    level=12,
    presets={
        "easy": dict(give_counts=False, z_lookup=False),
        "medium": dict(give_counts=True, z_lookup=False),
        "hard": dict(give_counts=True, z_lookup=True),
    },
)
def p8c_confidence_interval_proportion(rng, give_counts=False, z_lookup=False):
    """Use the approximate confidence interval (phat - z*sqrt(phat*(1-phat)/n),
    phat + z*sqrt(phat*(1-phat)/n)) for the population proportion, given
    bounded, fully specified inputs (ACMMM178)."""
    for _attempt in range(500):
        phat_hundredths = rng.choice(_PROP_PHAT_HUNDREDTHS)
        step = 100 // math.gcd(100, phat_hundredths)
        n = rng.choice([m for m in _PROP_N_RANGE if m % step == 0])
        x = phat_hundredths * n // 100
        phat = x / n  # exact, by construction of n as a multiple of step

        confidence = rng.choice(list(Z_VALUES))
        z = Z_VALUES[confidence]

        margin_p = z * math.sqrt(phat * (1 - phat) / n)
        lower_p = phat - margin_p
        upper_p = phat + margin_p
        if not (0 < lower_p and upper_p < 1):
            continue  # a proportion interval outside [0,1] is not sensible
        # Unambiguous rounding is asserted on the DISPLAYED percentage value,
        # never on the raw proportion -- rounding phat+/-margin to 1dp in
        # proportion space is a different, wrong boundary from rounding the
        # x100 percentage to 1dp.
        if not (rounds_unambiguously(lower_p * 100, 1) and rounds_unambiguously(upper_p * 100, 1)):
            continue
        lower_pct = round(lower_p * 100, 1)
        upper_pct = round(upper_p * 100, 1)

        answer_text = f"lower = {lower_pct:.1f}, upper = {upper_pct:.1f}"

        # Misconception distractors, in the same percentage-point units as
        # the answer (never proportion-space units).
        m_no_sqrt_pct = z * phat * (1 - phat) / n * 100  # omits-sqrt
        m_no_complement_pct = z * math.sqrt(phat / n) * 100  # omits-complement
        margin_pct = margin_p * 100
        candidate_pairs = [
            (round(phat * 100 - m_no_sqrt_pct, 1), round(phat * 100 + m_no_sqrt_pct, 1)),
            (round(phat * 100 - m_no_complement_pct, 1), round(phat * 100 + m_no_complement_pct, 1)),
            (round(phat * 100, 1), round(phat * 100 + 2 * margin_pct, 1)),  # one-sided-margin
        ]
        candidate_texts = [f"lower = {lo:.1f}, upper = {hi:.1f}" for lo, hi in candidate_pairs]

        gap_ok = all(
            abs(Fraction(str(lo)) - Fraction(str(lower_pct))) > GRADER_TOLERANCE
            or abs(Fraction(str(hi)) - Fraction(str(upper_pct))) > GRADER_TOLERANCE
            for lo, hi in candidate_pairs
        )
        if not gap_ok:
            continue
        break
    else:
        raise RuntimeError(
            "p8c_confidence_interval_proportion: no draw satisfied unambiguous "
            "rounding, a valid [0,1] proportion interval and distractor "
            "separation in 500 attempts"
        )

    # Independent check: a genuinely different route from construction's
    # z*sqrt(phat*(1-phat)/n) -- through z**2 and a single sqrt, computed
    # from phat expressed as an exact Fraction (x/n) rather than reusing
    # `margin_p`'s own expression.
    phat_frac = Fraction(x, n)
    variance_term = float(phat_frac * (1 - phat_frac) / n)
    margin_independent_pct = math.sqrt((z**2) * variance_term) * 100
    _verify_interval(phat * 100, lower_p * 100, upper_p * 100, margin_independent_pct, eps=1e-4)

    distractors = pick_distractors(answer_text, candidate_texts, tolerance=GRADER_TOLERANCE)
    check(len(distractors) == 3, "a distractor collided with the answer within grader tolerance")

    if not give_counts:
        sample_clause = f"the sample proportion who intend to vote yes, phat, is {phat:.2f}."
    else:
        sample_clause = f"{x} said they intend to vote yes, giving a sample proportion, phat."
    base = f"In a survey of {n} voters, {sample_clause} "
    if not z_lookup:
        instruction = (
            f"Using a {confidence} confidence level (z = {z}), calculate the approximate "
            f"confidence interval for the population proportion, p, giving both bounds as "
            f"percentages to 1 decimal place."
        )
    else:
        instruction = (
            f"Given that a 90% confidence level uses z = 1.645, a 95% confidence level "
            f"uses z = 1.96, and a 99% confidence level uses z = 2.576, calculate the "
            f"approximate confidence interval for the population proportion, p, at the "
            f"{confidence} confidence level, giving both bounds as percentages to 1 "
            f"decimal place."
        )
    question_text = base + instruction

    working = []
    if give_counts:
        working.append(f"phat = {x} divided by {n} = {phat:.2f}.")
    working.append(
        f"Margin of error = z times the square root of (phat times (1 - phat) divided "
        f"by n) = {z} times the square root of ({phat:.2f} times {1 - phat:.2f} divided "
        f"by {n}) = {round(margin_p, 4)}, or {round(margin_pct, 2)} percentage points."
    )
    working.append(
        f"Lower bound = {phat * 100:.1f} minus {round(margin_pct, 2)} = {lower_pct:.1f}. "
        f"Upper bound = {phat * 100:.1f} plus {round(margin_pct, 2)} = {upper_pct:.1f}."
    )

    return {
        "text": question_text,
        "latex": f"plain:{question_text}",
        "answer": {"lower": lower_pct, "upper": upper_pct},
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer_text, plain=True),
        "distractors": distractors,
        "working": working,
    }
