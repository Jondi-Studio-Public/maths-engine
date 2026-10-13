"""Exam question: pool two stages of an experiment, then use the pooled relative frequency to predict.

Two stages of the same process are reported as counts out of trials. Part (a) pools them: the relative frequency of
the whole experiment is total successes over total trials, not the average of the two stage rates (the stages have
different sizes, and the question is built so the two disagree). Part (b) uses it to predict a count in a further
stated number of trials, and part (c) uses it for the chance of at least one success in a few further trials.

Built from the counts. Part (a) is rounded to 3 places; (b) and (c) are worked from that rounded value, as the prompt
says, and a draw is kept only when the exact route (the unrounded relative frequency) gives the same displayed (b)
and (c), so the learner's route does not matter.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import attempts, check, generator
from .exd_x_r2k_support import retry_build, PROBABILITY, POSITIVE, clear_of_boundary, number_part, rd, tx

# stem, unit, plural, success noun phrase, "will ..." phrase, "is ..." phrase (singular event)
_CONTEXTS = (
    (
        "A machine's output was tested in two stages. In the first stage, {c1} of {n1} items were faulty. In the "
        "second stage, {c2} of {n2} items were faulty.",
        "item",
        "items",
        "faulty items",
        "will be faulty",
        "is faulty",
    ),
    (
        "A netball player took {n1} shots at goal in the first half of a season and scored {c1}. In the second half "
        "she took {n2} shots and scored {c2}.",
        "shot",
        "shots",
        "successful shots",
        "will be successful",
        "is successful",
    ),
    (
        "A bus company timed {n1} trips in Term 1, and {c1} arrived late. In Term 2 it timed {n2} trips, and {c2} "
        "arrived late.",
        "trip",
        "trips",
        "late trips",
        "will arrive late",
        "arrives late",
    ),
    (
        "A nursery planted {n1} seeds in a first tray, and {c1} germinated. It then planted {n2} seeds in a second "
        "tray, and {c2} germinated.",
        "seed",
        "seeds",
        "germinating seeds",
        "will germinate",
        "germinates",
    ),
)


def _build_k_relfreq(rng: random.Random) -> dict | None:
    """Pooled relative frequency, the expected count in more trials, and the chance of at least one success."""
    for _ in attempts(3000, "counts whose pooled relative frequency differs from the average of the two"):
        n1 = rng.randint(20, 150)
        n2 = rng.randint(20, 150)
        if abs(n1 - n2) < 15:
            continue
        r1 = Fraction(rng.randint(4, 55), 100)
        r2 = Fraction(rng.randint(4, 55), 100)
        c1, c2 = round(r1 * n1), round(r2 * n2)
        if abs(r1 - r2) > Fraction(22, 100):
            continue
        if not (1 < c1 < n1 and 1 < c2 < n2):
            continue
        pooled = Fraction(c1 + c2, n1 + n2)
        average = (Fraction(c1, n1) + Fraction(c2, n2)) / 2
        if abs(pooled - average) < Fraction(6, 1000) or not clear_of_boundary(pooled, 3, Fraction(1, 5)):
            continue
        a_val = rd(pooled, 3)
        M = rng.choice((200, 250, 300, 400, 500, 600, 800, 1000, 1200, 1500))
        k = rng.choice((2, 3, 4))
        exp_val = M * a_val
        exp_exact = M * pooled
        if rd(exp_val, 0) != rd(exp_exact, 0) or not clear_of_boundary(exp_val, 0, Fraction(1, 5)):
            continue
        at_least = 1 - (1 - a_val) ** k
        at_least_exact = 1 - (1 - pooled) ** k
        if rd(at_least, 3) != rd(at_least_exact, 3) or not clear_of_boundary(at_least, 3, Fraction(1, 5)):
            continue
        break
    stem_t, unit, plural, noun, will, is_ = rng.choice(_CONTEXTS)
    stem = stem_t.format(c1=c1, n1=n1, c2=c2, n2=n2)
    b_val = rd(exp_val, 0)
    c_val = rd(at_least, 3)

    pa = number_part(
        f"Find the relative frequency of {noun} over both stages combined, correct to 3 decimal places.",
        pooled,
        3,
        2,
        [f"Combined: ({c1} + {c2}) / ({n1} + {n2}) = {c1 + c2}/{n1 + n2} = {tx(pooled, 3)}."],
        [
            (average, "averaged-the-two-rates"),
            (1 - pooled, "gave-complement-for-event"),
            (Fraction(c1 + c2, (n1 + n2) - (c1 + c2)) if 2 * (c1 + c2) < n1 + n2 else None, "divided-by-failures"),
            (Fraction(c1 + c2, max(n1, n2)) if c1 + c2 < max(n1, n2) else None, "used-one-stage-only"),
        ],
        sensible_range=PROBABILITY,
    )
    pb = number_part(
        f"Use your value from (a) to estimate how many of the next {M} {plural} {will}, to the nearest whole number.",
        exp_val,
        0,
        1,
        [f"{M} x {tx(a_val, 3)} = {float(exp_val):.1f}, so about {b_val}."],
        [
            (M * (1 - a_val), "gave-complement-for-event"),
            (M * average, "averaged-the-two-rates"),
            (M * Fraction(c1, n1), "used-one-stage-only"),
            (M * Fraction(c2, n2), "used-one-stage-only"),
        ],
        follow_through=f"{M}*part_a",
        sensible_range=POSITIVE,
    )
    pc = number_part(
        f"Use your value from (a) to estimate the probability that at least one of the next {k} {plural} {is_}, "
        f"assuming the {plural} are independent. Give your answer correct to 3 decimal places.",
        at_least,
        3,
        2,
        [f"P(at least one) = 1 - (1 - {tx(a_val, 3)})^{k} = {tx(at_least, 3)}."],
        [
            (k * a_val if k * a_val < 1 else None, "added-probabilities-for-at-least-one"),
            (a_val**k, "all-instead-of-at-least-one"),
            ((1 - a_val) ** k, "none-instead-of-at-least-one"),
            (a_val, "single-trial-for-several"),
        ],
        follow_through=f"1-(1-part_a)**{k}",
        sensible_range=PROBABILITY,
    )
    if pa is None or pb is None or pc is None:
        return None
    return dict(text=stem, parts=[pa, pb, pc])


@generator("exd_x_r2k_relfreq", level=11, presets={"exam": dict(tier="exam")})
def exd_x_r2k_relfreq(rng: random.Random, tier: str = "exam") -> dict:
    """Pooled relative frequency, the expected count in more trials, and the chance of at least one success."""
    check(tier == "exam", "exd_x_r2k_relfreq: only the exam tier exists")
    return retry_build(rng, _build_k_relfreq, "exd_x_r2k_relfreq")
