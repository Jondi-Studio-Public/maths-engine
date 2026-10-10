"""Generates `p11d_compound_outcomes_replacement` questions: the probability of a two-draw compound outcome when
the first ball is returned to the bag.

One skill: ``f10-compound-outcomes-replacement`` (AC9M9P01, WITH REPLACEMENT
ONLY -- the without-replacement half of the descriptor is out of scope and
is exercised only as the mandatory ``no-replacement-values`` distractor
below).

Scenario: a bag holds ``R`` balls of colour A and ``B`` balls of colour B,
``T = R + B`` in total. A ball is drawn, its colour noted, then RETURNED
before a second ball is drawn -- stated explicitly in every question. The
four two-stage outcomes are AA, AB, BA, BB with probabilities ``p*p``,
``p*q``, ``q*p``, ``q*q`` where ``p = R/T``, ``q = B/T``. The question states
the four outcomes in a stated order (canonical at easy/medium, a genuinely
shuffled order at hard) and asks for the matching probability list,
positionally, as an ``ordered_list``.

Difficulty progression (genuine reasoning escalation, not span):
- easy: small T, single-draw probabilities already simple; canonical order
  AA, AB, BA, BB -- one reasoning step (multiply the two stated draws).
- medium: larger T, forced so the single-draw fraction R/T is NOT already in
  lowest terms -- an extra reasoning step (simplify before multiplying).
  Canonical order.
- hard: the four outcomes are stated in a genuinely shuffled, non-canonical
  order -- the learner must match each NAMED outcome to its own probability
  rather than pattern-filling a memorised canonical sequence, a different
  kind of reasoning step again (identification, not computation).

Both colour counts are always >= 2 at every tier (not just >= 1): with R == 1, the mandatory
``no-replacement-values`` distractor's P(AA) term is ``1/T * 0/(T-1) == 0``, which renders as the bare integer
"0" among three proper fractions -- exactly the "answer identifiable by shape" defect that
must be avoided.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd

from ..core import check, check_pairwise_distinct, generator, latex_answer, rational_text

_RESAMPLE_ATTEMPTS = 400

#: (colour A, colour B) name pairs drawn for variety across seeds.
_COLOUR_PAIRS = [
    ("red", "blue"),
    ("green", "yellow"),
    ("black", "white"),
    ("orange", "purple"),
]

#: Canonical outcome order as (first-draw code, second-draw code) pairs,
#: "A"/"B" meaning colour A / colour B respectively.
_CANONICAL_CODES = [("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")]


def _label(colour_of: dict, code_pair: tuple[str, str]) -> str:
    return f"{colour_of[code_pair[0]]}-then-{colour_of[code_pair[1]]}"


def _verify_with_replacement(R: int, B: int, T: int, order: list[tuple[str, str]], probs: list[Fraction]) -> None:
    """Independent check: recompute all 4 with-replacement outcome
    probabilities from the RAW R/B/T integers via a flat dict LITERAL keyed
    by outcome code string (built with no loop at all) -- a structurally
    different route from the main construction's `prob_of[c1] * prob_of[c2]`
    list comprehension over `order`. Also enforces the figure-plausibility
    rule: every probability sits in [0, 1]."""
    check(R >= 2 and B >= 2, f"degenerate colour count: R={R}, B={B}")
    table = {
        "AA": Fraction(R, T) * Fraction(R, T),
        "AB": Fraction(R, T) * Fraction(B, T),
        "BA": Fraction(B, T) * Fraction(R, T),
        "BB": Fraction(B, T) * Fraction(B, T),
    }
    expected = [table["".join(pair)] for pair in order]
    check(expected == probs, f"with-replacement probabilities mismatch: expected {expected}, got {probs}")
    for value in probs:
        check(0 <= value <= 1, f"probability out of [0, 1]: {value}")


def _no_replacement_probs(R: int, B: int, T: int, order: list[tuple[str, str]]) -> list[Fraction]:
    """The `no-replacement-values` misconception: every outcome recomputed as
    if the second draw did not replace the first (T >= 2 always holds since
    T >= 5 at every tier)."""
    table = {
        "AA": Fraction(R, T) * Fraction(R - 1, T - 1),
        "AB": Fraction(R, T) * Fraction(B, T - 1),
        "BA": Fraction(B, T) * Fraction(R, T - 1),
        "BB": Fraction(B, T) * Fraction(B - 1, T - 1),
    }
    return [table["".join(pair)] for pair in order]


def _swapped_colour_probs(R: int, B: int, T: int, order: list[tuple[str, str]]) -> list[Fraction]:
    """The `swapped-colour-probabilities` misconception: p and q swapped
    throughout, as if the learner mixed up which colour has which
    single-draw probability. Only ever called when R != B (guaranteed by
    construction below), since R == B would collapse this to the answer."""
    swapped = {"A": Fraction(B, T), "B": Fraction(R, T)}
    return [swapped[c1] * swapped[c2] for c1, c2 in order]


@generator(
    "p11d_compound_outcomes_replacement",
    level=9,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p11d_compound_outcomes_replacement(rng, tier="easy"):
    """Two-stage WITH-REPLACEMENT compound outcomes: list the probability of
    each of the four (colour, colour) outcomes, in a stated order.

    Tiers:
    - easy: T in {5, 6, 7}; single-draw fractions may or may not already be
      reduced (not forced either way); canonical outcome order.
    - medium: T in {10, 12}; R is forced so gcd(R, T) > 1 -- R/T (and,
      automatically, B/T = (T-R)/T, since gcd(B,T) = gcd(T-R,T) = gcd(R,T))
      is NOT already in lowest terms, so the learner must simplify the
      single-draw probability before multiplying. Canonical order.
    - hard: T in {8, 9, 10, 12}; the four outcomes are stated in a genuinely
      shuffled, non-canonical order every seed.

    Every tier draws R, B as discrete integer counts with 2 <= R, B (never a
    percentage -- the mandatory no-replacement distractor needs a real total
    T and a real count to subtract 1 from) and R != B (required so the
    `swapped-colour-probabilities` misconception is always exercisable, per
    the alternative of "always draw R != B").
    """
    for _ in range(_RESAMPLE_ATTEMPTS):
        color_a, color_b = rng.choice(_COLOUR_PAIRS)
        colour_of = {"A": color_a, "B": color_b}

        if tier == "easy":
            T = rng.choice([5, 6, 7])
        elif tier == "medium":
            T = rng.choice([10, 12])
        else:
            T = rng.choice([8, 9, 10, 12])

        candidates = [r for r in range(2, T - 1) if r != T - r]
        if tier == "medium":
            candidates = [r for r in candidates if gcd(r, T) > 1]
        if not candidates:
            continue
        R = rng.choice(candidates)
        B = T - R
        check(R >= 2 and B >= 2, f"degenerate colour count: R={R}, B={B}")
        check(R != B, "R must differ from B for swapped-colour-probabilities to be exercisable")
        if tier == "medium":
            check(gcd(R, T) > 1, f"medium tier must force an unreduced single-draw probability: R={R}, T={T}")

        if tier == "hard":
            order = list(_CANONICAL_CODES)
            rng.shuffle(order)
            if order == _CANONICAL_CODES:
                continue  # must be genuinely non-canonical every seed
        else:
            order = list(_CANONICAL_CODES)

        p = Fraction(R, T)
        q = Fraction(B, T)
        prob_of = {"A": p, "B": q}
        answer_values = [prob_of[c1] * prob_of[c2] for c1, c2 in order]
        _verify_with_replacement(R, B, T, order, answer_values)

        no_replace_values = _no_replacement_probs(R, B, T, order)
        swapped_values = _swapped_colour_probs(R, B, T, order)

        check_pairwise_distinct(
            answer_values,
            no_replace_values,
            swapped_values,
            msg="a probability-list distractor collides with the answer or another distractor",
        )
        for label_, values in (
            ("answer", answer_values),
            ("no-replace", no_replace_values),
            ("swapped", swapped_values),
        ):
            for value in values:
                check(0 <= value <= 1, f"{label_} probability out of [0, 1]: {value}")

        order_labels = [_label(colour_of, pair) for pair in order]
        answer_text = ", ".join(rational_text(v) for v in answer_values)
        no_replace_text = ", ".join(rational_text(v) for v in no_replace_values)
        swapped_text = ", ".join(rational_text(v) for v in swapped_values)

        question_text = (
            f"A bag contains {R} {color_a} balls and {B} {color_b} balls, {T} in total. "
            f"A ball is drawn at random, its colour is noted, then it is returned to the bag "
            f"before a second ball is drawn. List the probability of each of these four "
            f"outcomes, in this order: {', '.join(order_labels)}."
        )

        working = [
            f"P({color_a}) = {rational_text(p)}, P({color_b}) = {rational_text(q)} -- unchanged on "
            f"the second draw because the ball is returned.",
            f"Each outcome's probability is the product of its two draws, e.g. "
            f"P({order_labels[0]}) = {rational_text(answer_values[0])}.",
            f"In stated order: {answer_text}.",
        ]

        return dict(
            text=question_text,
            latex=f"plain:{question_text}",
            answer=answer_values,
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text),
            working=working,
            distractors=[no_replace_text, swapped_text],
        )

    raise RuntimeError(
        f"could not build a valid p11d_compound_outcomes_replacement question for tier={tier!r} "
        f"in {_RESAMPLE_ATTEMPTS} attempts"
    )
