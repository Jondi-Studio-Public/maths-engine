"""Exam question: relative frequencies from a frequency table with one count lost, used as probability estimates.

The stem gives the number of trials and four category counts, one of which is not shown. Part (a) recovers the
missing count from the total. Part (b) uses relative frequency as the estimate of the probability of an event made
of two categories, one of which is the category found in (a). Part (c) projects that estimate onto a stated number
of future trials. Part (d) uses it again for "at least one of the next two trials". Parts (b) to (d) carry
`follow_through`, so a slip in (a) is not paid for three times.

Built backwards from the table. The probability in (d) is re-derived independently by listing the four outcomes of
two independent trials, and the projection in (c) is kept clear of a rounding boundary so the key does not depend
on how a learner rounds.
"""

from __future__ import annotations

import random
from fractions import Fraction
from itertools import product

from ..answers import Distractor, exact_value
from ..core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    generator,
    latex_answer,
    pick_distractors,
    rational_text,
    round_half_up,
)

# (stem with {T}, categories, singular unit of a trial)
_CONTEXTS = (
    (
        "{T} train arrivals at a suburban station were each recorded as one of four outcomes.",
        ("early", "on time", "late", "very late"),
        "train arrival",
    ),
    (
        "{T} calls to a library help line were each recorded as one of four outcomes.",
        ("resolved", "transferred", "abandoned", "call-back arranged"),
        "call",
    ),
    (
        "{T} parcels sent by a courier were each recorded by how quickly they were delivered.",
        ("same day", "next day", "two to three days", "more than three days"),
        "parcel",
    ),
    (
        "{T} seeds from a packet were sown, and the result for each seed was recorded.",
        ("sprouted early", "sprouted late", "failed to sprout", "eaten by birds"),
        "seed",
    ),
    (
        "{T} shoppers at a market stall were each recorded by what they bought.",
        ("fruit only", "vegetables only", "fruit and vegetables", "no purchase"),
        "shopper",
    ),
)
_PROJECT_TO = (40, 50, 60, 80, 100, 120, 150, 200, 250)
_MARGIN = Fraction(1, 10)  # keep the projected count this far from a rounding boundary (.5)


def _within_unit_interval(candidates: list[Distractor]) -> list[Distractor]:
    """The candidates that are possible probabilities: strictly between 0 and 1, never ruled out on sight."""
    return [c for c in candidates if 0 < c.value < 1]


def _distractor_sets(counts, missing, partner, total, horizon, p, prob_d, answer_c):
    """Three tagged distractors for each of the four parts, or None when a part cannot get three distinct ones."""
    m, o = counts[missing], counts[partner]
    known_sum = total - m
    a_text, b_text = str(m), rational_text(p)
    c_text, d_text = str(int(answer_c)), rational_text(prob_d)
    others = [k for k in range(4) if k != missing]
    a_cands = [
        Distractor(str(known_sum), "reports-known-sum-not-remainder", known_sum),
        Distractor(str(total), "reports-total-as-missing-value", total),
        Distractor(str(m + counts[others[0]]), "left-a-class-out-of-the-sum", m + counts[others[0]]),
        Distractor(str(m + counts[others[1]] + 1), "left-a-class-out-of-the-sum", m + counts[others[1]] + 1),
    ]
    partner_only = Fraction(o, total)
    wrong_den = Fraction(m + o, known_sum)
    missing_only = Fraction(m, total)
    b_cands = [
        Distractor(rational_text(partner_only), "omits-the-missing-category", partner_only),
        Distractor(rational_text(wrong_den), "wrong-denominator-for-proportion", wrong_den),
        Distractor(rational_text(1 - p), "gave-complement-for-event", 1 - p),
        Distractor(rational_text(missing_only), "counts-only-one-named-category", missing_only),
        Distractor(
            rational_text(Fraction(m + o, total - o)),
            "wrong-denominator-for-proportion",
            Fraction(m + o, total - o),
        ),
    ]
    c_cands = [
        Distractor(
            str(int(round_half_up(horizon * (1 - p)))), "projects-the-complement", round_half_up(horizon * (1 - p))
        ),
        Distractor(str(m + o), "reports-raw-count-not-projection", m + o),
        Distractor(
            str(int(round_half_up(horizon * partner_only))),
            "omits-the-missing-category",
            round_half_up(horizon * partner_only),
        ),
        Distractor(
            str(int(round_half_up(horizon * missing_only))),
            "counts-only-one-named-category",
            round_half_up(horizon * missing_only),
        ),
        Distractor(
            str(int(round_half_up(horizon * wrong_den))),
            "wrong-denominator-for-proportion",
            round_half_up(horizon * wrong_den),
        ),
    ]
    d_cands = [
        Distractor(rational_text(p * p), "all-instead-of-at-least-one", p * p),
        Distractor(rational_text(2 * p), "added-probabilities-for-at-least-one", 2 * p),
        Distractor(rational_text((1 - p) ** 2), "none-instead-of-at-least-one", (1 - p) ** 2),
        Distractor(rational_text(1 - p * p), "gave-complement-of-both", 1 - p * p),
    ]
    a_d = pick_distractors(a_text, a_cands, 3, tolerance=GRADER_TOLERANCE)
    b_d = pick_distractors(b_text, _within_unit_interval(b_cands), 3)
    c_d = pick_distractors(c_text, c_cands, 3, tolerance=GRADER_TOLERANCE)
    d_d = pick_distractors(d_text, _within_unit_interval(d_cands), 3)
    if not len(a_d) == len(b_d) == len(c_d) == len(d_d) == 3:
        return None
    return a_d, b_d, c_d, d_d


def _build(rng: random.Random) -> dict:
    stem_fmt, cats, unit = rng.choice(_CONTEXTS)
    for _ in attempts(2000, "a table whose projection avoids a rounding boundary"):
        missing = rng.randrange(4)
        partner = rng.choice([k for k in range(4) if k != missing])
        counts = [rng.randint(3, 30) for _ in range(4)]
        total = sum(counts)
        if not 30 <= total <= 110:
            continue
        horizon = rng.choice(_PROJECT_TO)
        p = Fraction(counts[missing] + counts[partner], total)
        projected = horizon * p
        if abs((projected % 1) - Fraction(1, 2)) <= _MARGIN or projected < 5 or p == Fraction(1, 2):
            continue
        sets = _distractor_sets(counts, missing, partner, total, horizon, p, 1 - (1 - p) ** 2, round_half_up(projected))
        if sets is not None:
            break
    m, o = counts[missing], counts[partner]
    known_sum = total - m
    for d in sum(sets, []):
        check(isinstance(d, Distractor), "a distractor lost its tag")

    # Independent routes: two trials by enumeration, and the projection by direct rounding.
    outcomes = sum((p if x else 1 - p) * (p if y else 1 - p) for x, y in product((True, False), repeat=2) if x or y)
    prob_d = 1 - (1 - p) ** 2
    check(outcomes == prob_d, "at-least-one probability disagrees with the enumeration")
    answer_c = round_half_up(projected)
    check(abs(answer_c - projected) < Fraction(1, 2) - _MARGIN, "projection too close to a rounding boundary")

    shown = ", ".join(f"{cat}: {'not shown' if k == missing else counts[k]}" for k, cat in enumerate(cats))
    stem = f"{stem_fmt.format(T=total)} The counts were {shown}."
    event = f"{cats[missing]} or {cats[partner]}"
    event_phrase = f"the next {unit} is recorded as {event}"

    a_text, b_text = str(m), rational_text(p)
    c_text, d_text = str(int(answer_c)), rational_text(prob_d)

    a_d, b_d, c_d, d_d = sets

    parts = [
        dict(
            prompt=f"Find the number of {unit}s recorded as {cats[missing]}.",
            answer_kind="number",
            answer=m,
            answer_text=a_text,
            answer_latex=latex_answer(a_text),
            marks=1,
            sensible_range=(0, None),
            working=[f"{total} - {known_sum} = {m}."],
            distractors=a_d,
        ),
        dict(
            prompt=(
                f"Use the relative frequency to estimate the probability that {event_phrase}. "
                "Give your answer as a fraction in simplest form."
            ),
            answer_kind="fraction",
            answer=exact_value(p),
            answer_text=b_text,
            answer_latex=latex_answer(b_text),
            marks=2,
            sensible_range=(0, 1),
            follow_through=f"(part_a + {o})/{total}",
            working=[f"Relative frequency = ({m} + {o})/{total} = {b_text}."],
            distractors=b_d,
        ),
        dict(
            prompt=(
                f"Using your estimate from (b), predict how many of the next {horizon} {unit}s will be recorded "
                f"as {event}. Round to the nearest whole number."
            ),
            answer_kind="number",
            answer=int(answer_c),
            answer_text=c_text,
            answer_latex=latex_answer(c_text),
            marks=1,
            sensible_range=(0, None),
            follow_through=f"{horizon}*part_b",
            working=[
                f"{horizon} × {b_text} = {c_text}."
                if projected == int(projected)
                else f"{horizon} × {b_text} = {float(projected):.2f}, so about {c_text}."
            ],
            distractors=c_d,
        ),
        dict(
            prompt=(
                f"Using your estimate from (b), find the probability that at least one of the next two {unit}s "
                f"is recorded as {event}. Give your answer as a fraction in simplest form."
            ),
            answer_kind="fraction",
            answer=exact_value(prob_d),
            answer_text=d_text,
            answer_latex=latex_answer(d_text),
            marks=2,
            sensible_range=(0, 1),
            follow_through="1 - (1 - part_b)**2",
            working=[
                f"P(neither) = (1 - {b_text})^2 = {rational_text((1 - p) ** 2)}.",
                f"P(at least one) = 1 - {rational_text((1 - p) ** 2)} = {d_text}.",
            ],
            distractors=d_d,
        ),
    ]
    return dict(text=stem, parts=parts)


@generator("exd_x_r2l_relfreq_chain", level=12, presets={"exam": dict(tier="exam")})
def exd_x_r2l_relfreq_chain(rng: random.Random, tier: str = "exam") -> dict:
    """A frequency table with one count lost: recover it, estimate a probability, project it, then use it twice."""
    check(tier == "exam", "exd_x_r2l_relfreq_chain: only the exam tier exists")
    return _build(rng)
