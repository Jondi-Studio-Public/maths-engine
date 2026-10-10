"""Exam question: re-scale a confidence interval for a proportion to a different confidence level.

The sample size and the sample proportion are NOT given. The learner is told one interval, at one
confidence level, and must see that its centre is the sample proportion and its half-width is
`z * standard_error`, so the half-width at another level is the old half-width times `z_new / z_old`.
That is a change in kind from substituting into the interval formula: the method has to be recognised.

Built backwards from the interval that is shown. The centre and half-width are chosen as exact
tenths of a percentage point, so the stated interval is exact; the new half-width is then an exact
rational (the z values are quoted to three decimals), and both new bounds are rounded half-up to one
decimal place only at the very end. Draws whose new bounds sit within 0.01 of a rounding tie, or
outside (0, 100), are rejected.

Answer kind: `pair` (`lower = ..., upper = ...`), graded with the fixed 0.051 pair tolerance, so every
distractor is required to differ from the key by more than that in at least one bound.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import GRADER_TOLERANCE, check, generator, latex_answer, phrase, pick_distractors, round_half_up

_Z = {90: Fraction(1645, 1000), 95: Fraction(196, 100), 99: Fraction(2576, 1000)}
_Z_TABLE = "z = 1.645 for 90%, z = 1.96 for 95% and z = 2.576 for 99%"

_TEMPLATES = (
    "A random sample is used to estimate a population proportion. The approximate {old}% confidence interval "
    "for the proportion is ({lo}%, {hi}%). The sample itself is not available. Using {table}, find the "
    "approximate {new}% confidence interval from the same sample, giving both bounds as percentages to 1 "
    "decimal place.",
    "A polling company reports that an approximate {old}% confidence interval for the proportion of voters "
    "supporting a policy is {lo}% to {hi}%. Use {table} to work out the approximate {new}% confidence "
    "interval that the same survey gives. Give both bounds as percentages to 1 decimal place.",
    "From one sample, the approximate {old}% confidence interval for a population proportion p is "
    "{lo}% < p < {hi}%. Neither the sample size nor the sample proportion is stated. Given {table}, "
    "determine the approximate {new}% confidence interval for p, with both bounds as percentages to 1 "
    "decimal place.",
    "The approximate {old}% confidence interval for the proportion of faulty items from a sample is "
    "({lo}%, {hi}%). Given {table}, state the approximate {new}% confidence interval for the same sample, "
    "writing both bounds as percentages to 1 decimal place.",
)


def _tenths(x: Fraction) -> str:
    check((x * 10).denominator == 1, "ci_rescale: value is not on the tenths grid")
    return f"{float(x):.1f}"


def _near_tie(x: Fraction) -> bool:
    """True when `x` is within 0.01 of a 1 d.p. rounding tie (a multiple of 0.1 plus 0.05)."""
    r = (x * 100) % 10  # position inside its tenth, in hundredths
    return abs(r - 5) < Fraction(1)


def _build(rng: random.Random) -> dict:
    for _ in range(500):
        old, new = rng.sample(sorted(_Z), 2)
        centre = Fraction(rng.randint(150, 850), 10)
        half = Fraction(rng.randint(15, 70), 10)
        lo, hi = centre - half, centre + half
        new_half = half * _Z[new] / _Z[old]
        new_lo, new_hi = centre - new_half, centre + new_half
        if not (0 < new_lo and new_hi < 100):
            continue
        if _near_tie(new_lo) or _near_tie(new_hi):
            continue
        key_lo, key_hi = round_half_up(new_lo, 1), round_half_up(new_hi, 1)

        # Named misconceptions, each a pair of bounds around the same centre.
        wrong_halves = [
            half * _Z[new],  # multiplies by the new z instead of the ratio of z values
            half * _Z[old] / _Z[new],  # ratio upside down
            half + (_Z[new] - _Z[old]),  # adds the difference in z instead of scaling
        ]
        candidates = []
        for h in wrong_halves:
            c_lo, c_hi = round_half_up(centre - h, 1), round_half_up(centre + h, 1)
            candidates.append(f"lower = {float(c_lo):.1f}, upper = {float(c_hi):.1f}")
        # Keeps the old lower bound and only moves the upper bound (treats the margin as one-sided).
        candidates.append(f"lower = {float(lo):.1f}, upper = {float(round_half_up(hi + 2 * (new_half - half), 1)):.1f}")

        answer_text = f"lower = {float(key_lo):.1f}, upper = {float(key_hi):.1f}"
        distractors = pick_distractors(answer_text, candidates, tolerance=GRADER_TOLERANCE)
        if len(distractors) < 3:
            continue

        # Independent re-derivation: the ratio of half-widths must equal the ratio of z values.
        check(
            (new_hi - new_lo) / (hi - lo) == _Z[new] / _Z[old],
            "ci_rescale: widths are not in the ratio of the z values",
        )
        check(
            (new_lo + new_hi) / 2 == (lo + hi) / 2,
            "ci_rescale: the centre moved",
        )
        text = phrase(rng, _TEMPLATES, old=old, new=new, lo=_tenths(lo), hi=_tenths(hi), table=_Z_TABLE)
        working = [
            f"Centre (sample proportion) = ({_tenths(lo)} + {_tenths(hi)}) / 2 = {float(centre):.1f}%.",
            f"Half-width at {old}% = ({_tenths(hi)} - {_tenths(lo)}) / 2 = {float(half):.1f} percentage points.",
            f"Half-width at {new}% = {float(half):.1f} x {float(_Z[new])} / {float(_Z[old])} = "
            f"{float(new_half):.3f} percentage points.",
            f"Interval = {float(centre):.1f} -/+ {float(new_half):.3f} = {answer_text}.",
        ]
        return dict(
            text=text,
            latex=f"plain:{text}",
            answer={"lower": float(key_lo), "upper": float(key_hi)},
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text, plain=True),
            distractors=distractors[:3],
            working=working,
        )
    raise RuntimeError("exd_x_s5_ci_rescale: no draw satisfied the constraints in 500 attempts")


@generator("exd_x_s5_ci_rescale", level=12, presets={"exam": dict(tier="exam")})
def exd_x_s5_ci_rescale(rng: random.Random, tier: str = "exam") -> dict:
    """Re-scale a stated confidence interval for a proportion to another confidence level."""
    check(tier == "exam", "exd_x_s5_ci_rescale: only the exam tier exists")
    return _build(rng)
