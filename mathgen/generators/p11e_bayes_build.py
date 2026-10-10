"""Bayes update from finite population counts.

Kernel: the REVERSAL, not a lookup --. The learner is given a PRIOR population split across two groups as raw
counts, and a per-group RATE ("a out of every b") for an evidence property. They must first construct the
joint counts themselves (population x rate) before forming the posterior ratio. This is deliberately NOT the
shape of the shelf-mate `conditional_probability_counts` (`university_foundations.py`), which hands the
learner an already-built 2x2 count table for a single cell-ratio lookup -- there is no update to perform
there, only a read.

Scenario: two machines each produce a stated number of items and each has a
stated "faulty" rate. easy conditions on the evidence (faulty); medium
conditions on its complement (not faulty), forcing an extra subtraction per
group; hard states the two machines' populations only as a ratio plus a
total, so the learner must derive the populations before doing either of
the above.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    phrase,
    pick_distractors,
    VerificationError,
)

_MAX_ATTEMPTS = 200

_BAYES_DIRECT_TEMPLATES = (
    "Machine 1 produces {n1} items and Machine 2 produces {n2} items. {a1} out of every {b1} "
    "items from Machine 1 are faulty, and {a2} out of every {b2} items from Machine 2 are "
    "faulty. An item is picked at random from the combined output and found to {phrase}. "
    "What is the probability it came from Machine 1?",
    "Two machines feed a combined output line. Machine 1 produces {n1} items and Machine 2 "
    "produces {n2} items. {a1} out of every {b1} items from Machine 1 are faulty, and {a2} out "
    "of every {b2} items from Machine 2 are faulty. An item is picked at random from the "
    "combined output and found to {phrase}. Find the probability that it came from Machine 1.",
    "A factory runs two machines. Machine 1 produces {n1} items and Machine 2 produces {n2} "
    "items. {a1} out of every {b1} items from Machine 1 are faulty, and {a2} out of every {b2} "
    "items from Machine 2 are faulty. An item is picked at random from the combined output and "
    "found to {phrase}. Given this, what is the probability the item came from Machine 1?",
)

_BAYES_RATIO_TEMPLATES = (
    "Machine 1 and Machine 2 produce items in the ratio {k1} : {k2}, with {total} items "
    "produced in total. {a1} out of every {b1} items from Machine 1 are faulty, and {a2} out "
    "of every {b2} items from Machine 2 are faulty. An item is picked at random from the "
    "combined output and found to {phrase}. What is the probability it came from Machine 1?",
    "Two machines share production. Machine 1 and Machine 2 produce items in the ratio {k1} : "
    "{k2}, with {total} items produced in total. {a1} out of every {b1} items from Machine 1 "
    "are faulty, and {a2} out of every {b2} items from Machine 2 are faulty. An item is picked "
    "at random from the combined output and found to {phrase}. Find the probability that it "
    "came from Machine 1.",
    "A factory splits its output between two machines. Machine 1 and Machine 2 produce items "
    "in the ratio {k1} : {k2}, with {total} items produced in total. {a1} out of every {b1} "
    "items from Machine 1 are faulty, and {a2} out of every {b2} items from Machine 2 are "
    "faulty. An item is picked at random from the combined output and found to {phrase}. Given "
    "this, what is the probability the item came from Machine 1?",
)


# Candidate shared helper: fraction_answer_payload(value: Fraction) -> tuple[dict, str] -- "{p, q} answer
# dict + p/q display text for a Fraction-answer skill" -- already hand-rolled, verbatim, as
# `_fraction_payload` in `mathgen/generators/university_foundations.py` (`discrete_expected_value`,
# `conditional_probability_counts`) and now a third time here. It would belong in `mathgen/core.py`;
# inlined locally below, don't hand-roll a third copy" rule.
def _fraction_payload(value: Fraction) -> tuple[dict, str]:
    value = Fraction(value)
    text = str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    return {"p": value.numerator, "q": value.denominator}, text


def _draw_direct(rng, denom_lo: int, denom_hi: int, pop_lo: int, pop_hi: int):
    """n1, n2 stated directly as small population counts (easy/medium)."""
    b1 = rng.randint(denom_lo, denom_hi)
    b2 = rng.randint(denom_lo, denom_hi)
    a1 = rng.randint(1, b1 - 1)
    a2 = rng.randint(1, b2 - 1)
    n1 = rng.choice([m for m in range(b1, pop_hi + 1, b1) if m >= pop_lo])
    n2 = rng.choice([m for m in range(b2, pop_hi + 1, b2) if m >= pop_lo])
    return n1, n2, a1, b1, a2, b2


def _draw_ratio(rng, denom_lo: int, denom_hi: int, ratio_lo: int, ratio_hi: int):
    """n1, n2 stated only as a ratio k1:k2 plus a total (hard) -- the total
    is built as a multiple of `(k1+k2)*b1*b2` so the derived populations are
    guaranteed to divide exactly by each machine's own rate denominator,
    never a rounded quantity."""
    b1 = rng.randint(denom_lo, denom_hi)
    b2 = rng.randint(denom_lo, denom_hi)
    a1 = rng.randint(1, b1 - 1)
    a2 = rng.randint(1, b2 - 1)
    k1 = rng.randint(ratio_lo, ratio_hi)
    k2 = rng.randint(ratio_lo, ratio_hi)
    while k2 == k1:
        k2 = rng.randint(ratio_lo, ratio_hi)
    total = (k1 + k2) * b1 * b2

    # Independent-of-construction check: derive n1, n2 from the ratio and
    # total exactly as the learner must, via Fraction division, never via
    # the `k * b1 * b2` shortcut used to build `total` above.
    derived_n1 = Fraction(total * k1, k1 + k2)
    derived_n2 = Fraction(total * k2, k1 + k2)
    check(
        derived_n1.denominator == 1 and derived_n2.denominator == 1,
        f"ratio-derived population not exact: total={total}, k1={k1}, k2={k2}",
    )
    n1, n2 = int(derived_n1), int(derived_n2)
    check(
        n1 == k1 * b1 * b2 and n2 == k2 * b1 * b2,
        f"ratio-derived population mismatch: n1={n1}, n2={n2}, k1={k1}, k2={k2}, b1={b1}, b2={b2}",
    )
    return n1, n2, a1, b1, a2, b2, k1, k2, total


def _independent_check(n1: int, f1: int, n2: int, f2: int, complement: bool, answer: Fraction):
    """Simulate the population EXPLICITLY as two labelled lists, filter by
    the conditioning event directly on the explicit list, and count -- a
    genuinely different route from the `f1/(f1+f2)`-shaped formula
    construction above (verifies the artefact the learner receives, not the
    formula the generator happened to build it with)."""
    group1 = [("group1", "E")] * f1 + [("group1", "not_E")] * (n1 - f1)
    group2 = [("group2", "E")] * f2 + [("group2", "not_E")] * (n2 - f2)
    population = group1 + group2
    target = "not_E" if complement else "E"
    filtered = [item for item in population if item[1] == target]
    count_group1 = sum(1 for item in filtered if item[0] == "group1")
    enumerated = Fraction(count_group1, len(filtered))
    check_equivalent(enumerated, answer, "bayes posterior enumeration mismatch")


def _bayes_finite_counts(rng, tier: str):
    for _attempt in range(_MAX_ATTEMPTS):
        if tier == "hard":
            n1, n2, a1, b1, a2, b2, k1, k2, total = _draw_ratio(rng, 2, 5, 2, 6)
            asked_complement = rng.choice([True, False])
        else:
            n1, n2, a1, b1, a2, b2 = _draw_direct(rng, 2, 9, 12, 60)
            k1 = k2 = total = None
            asked_complement = tier == "medium"

        f1 = n1 * a1 // b1
        f2 = n2 * a2 // b2
        check(f1 * b1 == n1 * a1, f"f1 not exact: n1={n1}, a1={a1}, b1={b1}")
        check(f2 * b2 == n2 * a2, f"f2 not exact: n2={n2}, a2={a2}, b2={b2}")
        check(0 < f1 < n1, f"f1 out of range: n1={n1}, a1={a1}, b1={b1}, f1={f1}")
        check(0 < f2 < n2, f"f2 out of range: n2={n2}, a2={a2}, b2={b2}, f2={f2}")

        not_f1, not_f2 = n1 - f1, n2 - f2
        counts = (not_f1, not_f2) if asked_complement else (f1, f2)
        answer = Fraction(counts[0], counts[0] + counts[1])

        likelihood = Fraction(b1 - a1, b1) if asked_complement else Fraction(a1, b1)
        prior = Fraction(n1, n1 + n2)
        wrong_group = Fraction(counts[1], counts[0] + counts[1])
        candidates = [likelihood, prior, wrong_group]
        if tier == "medium":
            candidates.append(Fraction(f1, f1 + f2))  # forgot-to-complement

        try:
            check_pairwise_distinct(answer, *candidates)
        except VerificationError:
            continue

        _independent_check(n1, f1, n2, f2, asked_complement, answer)
        break
    else:
        raise RuntimeError(f"bayes_finite_counts: no collision-free draw found for tier={tier}")

    answer_payload, answer_text = _fraction_payload(answer)
    distractors = pick_distractors(
        answer_text,
        [_fraction_payload(c)[1] for c in candidates],
        n=4 if tier == "medium" else 3,
    )

    phrase_word = "not be faulty" if asked_complement else "be faulty"
    condition_label = "not faulty" if asked_complement else "faulty"

    if tier == "hard":
        derive_step = (
            f"Machine 1's share: {total} x {k1}/({k1}+{k2}) = {n1} items; "
            f"Machine 2's share: {total} x {k2}/({k1}+{k2}) = {n2} items. "
        )
        text = phrase(
            rng,
            _BAYES_RATIO_TEMPLATES,
            k1=k1,
            k2=k2,
            total=total,
            a1=a1,
            b1=b1,
            a2=a2,
            b2=b2,
            phrase=phrase_word,
        )
    else:
        derive_step = ""
        text = phrase(
            rng,
            _BAYES_DIRECT_TEMPLATES,
            n1=n1,
            n2=n2,
            a1=a1,
            b1=b1,
            a2=a2,
            b2=b2,
            phrase=phrase_word,
        )

    if asked_complement:
        count_step = (
            f"Machine 1 has {a1}/{b1} x {n1} = {f1} faulty, so {n1} - {f1} = {not_f1} are {condition_label}; "
            f"Machine 2 has {a2}/{b2} x {n2} = {f2} faulty, so {n2} - {f2} = {not_f2} are {condition_label}."
        )
    else:
        count_step = (
            f"Machine 1 has {a1}/{b1} x {n1} = {f1} faulty items; Machine 2 has {a2}/{b2} x {n2} = {f2} faulty items."
        )
    ratio_step = f"P(Machine 1 | {condition_label}) = {counts[0]}/({counts[0]} + {counts[1]}) = {answer_text}."
    working = [derive_step + count_step, ratio_step] if derive_step else [count_step, ratio_step]

    return dict(
        text=text,
        answer=answer_payload,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


@generator(
    "bayes_finite_counts",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def bayes_finite_counts(rng, tier="easy"):
    """Reverse a prior population split through a per-group rate to a
    posterior: the learner builds the joint counts (population x rate) then
    forms the ratio -- the Bayes update this skill exercises, never a
    pre-built table lookup (that shape is `conditional_probability_counts`).
    """
    return _bayes_finite_counts(rng, tier)
