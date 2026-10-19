"""Probability generator for finite sample space (ACMEM155): calculate an event
probability from an explicitly enumerated finite sample space.

The observable contract: enumerate a small supplied finite sample space and
calculate one event probability. The sample space is stated in the question text
(the learner enumerates and counts it, not guesses an implied model). The answer
is an exact Rational probability in lowest terms, graded by grade_fraction.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, generator, pick_distractors, rational_text


def _verify_probability(outcomes, favourable_outcomes, prob):
    """Independent check: verify the probability by a route different from
    construction. The construction counts favourable outcomes and divides by
    the total; this check counts the complement and verifies that complement
    probability + event probability == 1, using the actual listed outcomes."""
    check(set(favourable_outcomes) <= set(outcomes), "favourable outcomes must be a subset of the outcomes list")
    complement_count = len(outcomes) - len(favourable_outcomes)
    complement_prob = Fraction(complement_count, len(outcomes))
    check(prob + complement_prob == 1, "probability plus complement probability must equal 1")
    check(0 < prob < 1, "probability must be strictly between 0 and 1 (not certain or impossible)")


@generator(
    "finite_sample_space_probability",
    level=8,
    presets={
        "easy": dict(space_size=6, pool_hi=10, event_type="single"),
        "medium": dict(space_size=8, pool_hi=14, event_type="multiple"),
        "hard": dict(space_size=10, pool_hi=20, event_type="intersection"),
    },
)
def finite_sample_space_probability(rng, space_size=8, pool_hi=14, event_type="multiple"):
    """Calculate the probability of an event from an explicitly enumerated
    finite sample space."""
    if space_size < 2:
        raise ValueError("space_size must be at least 2")
    if pool_hi < space_size:
        raise ValueError("pool_hi must be >= space_size")

    # Generate distinct outcomes as a random sample from 1 to pool_hi,
    # then sort them to present as an enumerated list
    outcomes = sorted(rng.sample(range(1, pool_hi + 1), space_size))

    # Define the event based on type and difficulty
    if event_type == "single":
        # Easy: single property (e.g., "even")
        if rng.random() < 0.5:
            # Event: even numbers
            event_name = "even"
            favourable = [x for x in outcomes if x % 2 == 0]
        else:
            # Event: odd numbers
            event_name = "odd"
            favourable = [x for x in outcomes if x % 2 == 1]
    elif event_type == "multiple":
        # Medium: union of properties (e.g., "even or greater than 5")
        if rng.random() < 0.5:
            # Event: even or greater than threshold
            threshold = rng.randint(space_size // 2, space_size - 1)
            event_name = f"even or greater than {threshold}"
            favourable = [x for x in outcomes if x % 2 == 0 or x > threshold]
        else:
            # Event: divisible by k where 2 <= k <= space_size/2
            k = rng.randint(2, max(2, space_size // 2))
            event_name = f"divisible by {k}"
            favourable = [x for x in outcomes if x % k == 0]
    else:  # event_type == "intersection"
        # Hard: intersection (e.g., "even and greater than 5")
        threshold = rng.randint(2, space_size - 2)
        event_name = f"even and greater than {threshold}"
        favourable = [x for x in outcomes if x % 2 == 0 and x > threshold]

    # Guard against degenerate cases: probability must be strictly between 0 and 1
    # Reroll the event if it's certain or impossible.
    #
    # Bounded, not `while True`: for some outcome sets no redraw can succeed --
    # e.g. an all-odd sample under the "even and greater than N" intersection
    # event has an empty favourable set for every threshold. That is reachable
    # (10 odds exist in 1..20, so the hard preset can draw them, at roughly
    # 1 in 185k) and an unbounded loop would hang the caller rather than fail.
    # Generation runs during catalogue validation, so a hang would freeze the
    # test suite; raising instead makes it loud and diagnosable.
    attempts = 0
    while not (0 < len(favourable) < len(outcomes)):
        attempts += 1
        if attempts > 50:
            raise ValueError(
                f"no non-degenerate {event_type!r} event exists for outcomes {outcomes} after {attempts - 1} redraws"
            )
        if event_type == "single":
            if rng.random() < 0.5:
                event_name = "even"
                favourable = [x for x in outcomes if x % 2 == 0]
            else:
                event_name = "odd"
                favourable = [x for x in outcomes if x % 2 == 1]
        elif event_type == "multiple":
            if rng.random() < 0.5:
                threshold = rng.randint(space_size // 2, space_size - 1)
                event_name = f"even or greater than {threshold}"
                favourable = [x for x in outcomes if x % 2 == 0 or x > threshold]
            else:
                k = rng.randint(2, max(2, space_size // 2))
                event_name = f"divisible by {k}"
                favourable = [x for x in outcomes if x % k == 0]
        else:
            threshold = rng.randint(2, space_size - 2)
            event_name = f"even and greater than {threshold}"
            favourable = [x for x in outcomes if x % 2 == 0 and x > threshold]

    # Calculate the probability
    prob = Fraction(len(favourable), len(outcomes))
    _verify_probability(outcomes, favourable, prob)

    answer_text = rational_text(prob)
    outcomes_str = ", ".join(map(str, outcomes))

    # Misconception distractors, all genuine probabilities (strictly between 0 and 1):
    #  - complement: reports (total - favourable)/total instead of favourable/total
    #  - favourable-over-other: favourable/(total - favourable), the other way up when that would exceed 1
    #  - off-by-one counts and a count that includes one extra outcome in the total
    fav, total = len(favourable), len(outcomes)
    other_count = total - fav
    ratio = Fraction(fav, other_count)
    candidates = [
        Fraction(other_count, total),
        ratio if ratio < 1 else Fraction(other_count, fav),
        Fraction(fav + 1, total),
        Fraction(fav - 1, total) if fav > 1 else Fraction(fav, total + 1),
        Fraction(fav, total + 1),
        Fraction(fav, total - 1),
    ]
    candidates = [rational_text(c) for c in candidates if 0 < c < 1]

    return dict(
        text=(
            f"A bag contains cards numbered {outcomes_str}. "
            f"One card is drawn at random. "
            f"What is the probability that the card is {event_name}?"
        ),
        latex=rf"\text{{P(card is {event_name})}}",
        answer={"p": prob.numerator, "q": prob.denominator},
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"Outcomes: {outcomes_str}. Total = {len(outcomes)}.",
            f"Favourable outcomes ({event_name}): {', '.join(map(str, favourable))}. Count = {len(favourable)}.",
            f"P({event_name}) = {len(favourable)}/{len(outcomes)} = {answer_text}.",
        ],
        distractors=pick_distractors(answer_text, candidates),
    )
