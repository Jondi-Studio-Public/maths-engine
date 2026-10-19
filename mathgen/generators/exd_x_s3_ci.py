"""Exam-tier source for `p8c_confidence_interval_mean`: an interval is GIVEN at one confidence level and the learner
must produce the interval for the same sample at another level.

The sample standard deviation and the sample size are not what the answer depends on. The learner has to see that
the interval is centred on the sample mean, that its half-width is ``z * s / sqrt(n)``, and so the half-width at the
new level is the old half-width multiplied by ``z_new / z_old``. The three z values come from an inline table.

Two framings, chosen per seed: the whole interval is quoted, or only the sample mean and one bound.

Built backwards from a chosen centre and half-width (both exact tenths), so the quoted bounds are exact and the key
is computed with exact fractions then rounded half-up to 1 decimal place. A draw is skipped when either rounded
bound sits near a rounding boundary or when any distractor lands within the grader's tolerance of the key.

Distractors, each a named slip:

* the quoted interval unchanged (never rescales);
* the half-width scaled by the inverse ratio ``z_old / z_new``;
* both quoted bounds multiplied by the ratio (treats the interval, not its margin, as proportional to z);
* the half-width multiplied by ``z_new`` itself (not by the ratio).
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import GRADER_TOLERANCE, check, generator, latex_answer, pick_distractors, round_half_up

Z = {"90%": Fraction(1645, 1000), "95%": Fraction(196, 100), "99%": Fraction(2576, 1000)}
_TABLE = "a 90% confidence level uses z = 1.645, a 95% level uses z = 1.96 and a 99% level uses z = 2.576"

_CONTEXTS = (
    ("the time taken by a courier depot to deliver a parcel", "minutes"),
    ("the mass of a loaf from a bakery", "grams"),
    ("the time a customer waits in a queue", "seconds"),
    ("the length of a bolt from a factory line", "millimetres"),
)


def _tenths(value: Fraction) -> Fraction:
    return round_half_up(value * 10) / 10


def _pair(lo: Fraction, hi: Fraction) -> str:
    return f"lower = {float(lo):.1f}, upper = {float(hi):.1f}"


def _near_boundary(value: Fraction) -> bool:
    """True when rounding `value` to 1 d.p. is within 0.01 of a coin flip."""
    frac = abs(value * 10) % 1
    return abs(frac - Fraction(1, 2)) < Fraction(1, 100)


@generator(
    "exd_x_s3_ci",
    level=12,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_s3_ci(rng: random.Random, tier: str = "exam"):
    """Rescale a quoted confidence interval for a mean to a different confidence level."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in range(2000):
        old, new = rng.sample(list(Z), 2)
        centre = Fraction(rng.randint(600, 3000), 10)
        half = Fraction(rng.randint(25, 150), 10)
        n = rng.randint(16, 120)
        lo_old, hi_old = centre - half, centre + half
        ratio = Z[new] / Z[old]
        new_half = half * ratio
        lo_new, hi_new = _tenths(centre - new_half), _tenths(centre + new_half)
        if _near_boundary(centre - new_half) or _near_boundary(centre + new_half):
            continue
        answer_text = _pair(lo_new, hi_new)

        inv_half = half / ratio
        cands = [
            (lo_old, hi_old),
            (_tenths(centre - inv_half), _tenths(centre + inv_half)),
            (_tenths(lo_old * ratio), _tenths(hi_old * ratio)),
            (_tenths(centre - half * Z[new]), _tenths(centre + half * Z[new])),
        ]
        cand_texts = [_pair(lo, hi) for lo, hi in cands]
        distractors = pick_distractors(answer_text, cand_texts, tolerance=GRADER_TOLERANCE)
        gaps_ok = all(
            abs(lo - lo_new) > GRADER_TOLERANCE or abs(hi - hi_new) > GRADER_TOLERANCE
            for (lo, hi), t in zip(cands, cand_texts)
            if t in distractors
        )
        if len(distractors) == 3 and gaps_ok:
            break
    else:
        raise RuntimeError("no draw gave an unambiguous key with three separated distractors")

    # Independent route: recover the standard error from the quoted bounds, then rebuild at the new level.
    se = (hi_old - lo_old) / 2 / Z[old]
    check((lo_old + hi_old) / 2 == centre, "quoted bounds must be centred on the sample mean")
    check(_tenths(centre - Z[new] * se) == lo_new and _tenths(centre + Z[new] * se) == hi_new, "key does not match")

    thing, unit = rng.choice(_CONTEXTS)
    if rng.random() < 0.5:
        given = f"A {old} confidence interval for the population mean of {thing} is ({float(lo_old):.1f}, {float(hi_old):.1f}) {unit}, based on a sample of {n}."
    else:
        given = (
            f"A sample of {n} measurements of {thing} has a sample mean of {float(centre):.1f} {unit}. "
            f"The lower bound of the {old} confidence interval for the population mean is {float(lo_old):.1f} {unit}."
        )
    text = (
        f"{given} Given that {_TABLE}, find the {new} confidence interval for the population mean from the same "
        f"sample. Use the z values exactly as given and do not round any intermediate value; round only the two final "
        f"bounds to 1 decimal place, in {unit}."
    )
    working = [
        f"The interval is centred on the sample mean {float(centre):.1f}, and its half-width {float(half):.1f} is z times s/sqrt(n).",
        f"At the {new} level the half-width is {float(half):.1f} times {float(Z[new])} divided by {float(Z[old])}.",
        answer_text + ".",
    ]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer={"lower": float(lo_new), "upper": float(hi_new)},
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        working=working,
        distractors=distractors,
    )
