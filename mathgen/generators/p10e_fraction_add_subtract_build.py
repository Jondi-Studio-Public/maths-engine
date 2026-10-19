"""Builds the `f10-fraction-add-subtract-equivalent` skill (AC9M5N05, AC9M6N05): adding and subtracting fractions
using equivalent fractions.

``p10e_fraction_add_subtract_equivalent`` -- add/subtract two or three proper
fractions whose denominators are always RELATED (one a multiple of the
other, or -- at the hard tier -- three denominators that share a common
multiple smaller than their plain product), never equal at input and never
fully coprime as a pair, so a genuine common-denominator equivalence step is
always required and the raw exclusion "unrelated-denominator proof" (a full
coprime-pair LCM derivation) is never exercised. This is a deliberately
narrower skill than the readiness skill ``fraction-arithmetic`` (topic
``fractions``, ``mathgen/generators/arithmetic.py``), which already performs
all four operations on ARBITRARY-denominator fractions: that generator draws
``d1, d2`` independently from ``range(2, max_den)`` with no relatedness
constraint at all, so it is not a substitute for a targeted assessment of the
equivalent-fraction method itself.

``answer: fraction`` -- so no "does not simplify" distractor is ever built; every misconception candidate is
checked to be a genuinely different VALUE, every seed (``check_distinct``).

Every combined value is held as an exact ``fractions.Fraction`` throughout
(never a float); the independent check recomputes the same total directly
from the ORIGINAL, un-scaled ``Fraction(n, d)`` terms via Python's own
``Fraction`` arithmetic (letting it auto-reduce), a route that never touches
this module's own LCD-finding/scaling code, and compares with
``mathgen.core.check_equivalent``.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import check, check_distinct, check_equivalent, generator, latex_answer, pick_distractors, rational_text

_RESAMPLE_ATTEMPTS = 400

# easy: (d1, d2) with d2 an exact multiple of d1, so the LCD is d2 itself and
# only the d1-denominator term needs converting.
_EASY_PAIRS = [
    (2, 4),
    (2, 6),
    (2, 8),
    (2, 10),
    (2, 12),
    (3, 6),
    (3, 9),
    (3, 12),
    (4, 8),
    (4, 12),
    (5, 10),
    (6, 12),
]

# medium: (d1, d2) sharing a common factor > 1 but with neither dividing the
# other, so BOTH terms must convert to reach the LCD.
_MEDIUM_PAIRS = [(4, 6), (4, 10), (6, 8), (6, 9), (8, 12)]

# hard: three denominators whose combined LCM is smaller than their plain
# product (so the set shares real structure, even where one individual pair
# is coprime -- e.g. (2, 3, 4): 2&3 are coprime but 2&4 are not, and the
# LCM (12) is still well below the naive product (24)). Matches the
# worked example (1/2 + 1/3 - 1/4, denominators 2/3/4) directly.
_HARD_TRIPLES = [
    (2, 3, 4),
    (2, 3, 6),
    (2, 3, 8),
    (2, 3, 12),
    (3, 4, 6),
    (2, 4, 6),
    (3, 4, 12),
    (2, 6, 8),
    (4, 6, 8),
]


def _random_proper_numerator(rng, d: int) -> int:
    """A numerator in [1, d-1] coprime to d, so the drawn fraction is itself
    already in lowest terms (an authentic question term, not "2/4")."""
    candidates = [n for n in range(1, d) if math.gcd(n, d) == 1]
    return rng.choice(candidates)


def _combine(terms: list[tuple[int, Fraction]]) -> Fraction:
    """``terms``: [(sign, Fraction), ...], sign is +1/-1. The straightforward
    signed sum -- used only by the independent check, on the ORIGINAL
    (unconverted) Fraction objects, never on the scaled numerators the
    construction below builds."""
    total = Fraction(0)
    for sign, f in terms:
        total += sign * f
    return total


def _denominators_added_candidate(terms: list[tuple[int, Fraction]]) -> Fraction:
    """Misconception: treats fraction addition/subtraction like whole-number
    addition of a pair -- (n1 +/- n2 +/- ...) over the SUM of the original
    denominators, never the LCD."""
    num = sum(sign * f.numerator for sign, f in terms)
    den = sum(f.denominator for _, f in terms)
    return Fraction(num, den)


def _numerator_not_scaled_candidate(rng, terms: list[tuple[int, Fraction]], lcd: int) -> Fraction:
    """Misconception: converts every term's denominator to the LCD but
    forgets to scale ONE term's numerator to match -- applied to a randomly
    chosen term that actually needs scaling (its own denominator != LCD;
    at medium/hard, where more than one term needs scaling, only one of
    them is left unscaled here)."""
    eligible = [i for i, (_, f) in enumerate(terms) if f.denominator != lcd]
    check(eligible, "no term needs scaling to reach the LCD -- construction bug")
    skip = rng.choice(eligible)
    num = 0
    for i, (sign, f) in enumerate(terms):
        scale = 1 if i == skip else lcd // f.denominator
        num += sign * f.numerator * scale
    return Fraction(num, lcd)


def _wrong_operation_sign_candidates(terms: list[tuple[int, Fraction]]) -> list[Fraction]:
    """Misconception, subtraction-only: computes the sum instead of the difference (flips the FIRST '-' to '+'),
    and -- for exactly two terms -- reverses the term order (b - a instead of a - b), guarded against a == b
    so the reversal is never a silent no-op. Returns [] when the expression is addition-only."""
    cands: list[Fraction] = []
    minus_indices = [i for i, (sign, _) in enumerate(terms) if sign == -1]
    if not minus_indices:
        return cands
    flipped = list(terms)
    i0 = minus_indices[0]
    flipped[i0] = (1, flipped[i0][1])
    cands.append(_combine(flipped))
    if len(terms) == 2:
        a, b = terms[0][1], terms[1][1]
        if a != b:
            cands.append(b - a)
    return cands


def _join_expr(numerators: list[int], denoms: list[int], ops: list[int]) -> str:
    parts = [f"{numerators[0]}/{denoms[0]}"]
    for sign, n, d in zip(ops[1:], numerators[1:], denoms[1:]):
        sym = "+" if sign == 1 else "-"
        parts.append(f"{sym} {n}/{d}")
    return " ".join(parts)


def _build(rng, tier: str) -> dict:
    for _ in range(_RESAMPLE_ATTEMPTS):
        if tier == "easy":
            denoms = list(rng.choice(_EASY_PAIRS))
            ops = [1, 1]
        elif tier == "medium":
            denoms = list(rng.choice(_MEDIUM_PAIRS))
            ops = [1, rng.choice([1, -1])]
        else:
            denoms = list(rng.choice(_HARD_TRIPLES))
            rng.shuffle(denoms)
            ops = [1, rng.choice([1, -1]), rng.choice([1, -1])]

        numerators = [_random_proper_numerator(rng, d) for d in denoms]
        terms = list(zip(ops, (Fraction(n, d) for n, d in zip(numerators, denoms))))

        lcd = denoms[0]
        for d in denoms[1:]:
            lcd = math.lcm(lcd, d)

        combined_num = sum(sign * n * (lcd // d) for sign, n, d in zip(ops, numerators, denoms))
        answer = Fraction(combined_num, lcd)

        if answer <= 0 or answer.denominator == 1:
            continue  # keep results a genuine, positive, non-integer fraction

        # Independent check: bypass the LCD-finding/scaling above entirely,
        # recomputing straight from the original (n, d) pairs.
        check_equivalent(answer, _combine(terms), "constructed total disagrees with independent Fraction recompute")

        candidates = [
            _denominators_added_candidate(terms),
            _numerator_not_scaled_candidate(rng, terms, lcd),
            *_wrong_operation_sign_candidates(terms),
        ]
        if any(c == answer for c in candidates):
            continue  # rare collision; resample the whole draw
        check_distinct(answer, *candidates, msg="a misconception candidate equals the constructed answer")

        answer_text = rational_text(answer)
        distractors = pick_distractors(answer_text, [rational_text(c) for c in candidates], n=3)
        if len(distractors) < 2:
            continue

        expr_text = _join_expr(numerators, denoms, ops)
        working = [
            f"LCD of {', '.join(str(d) for d in denoms)} is {lcd}.",
            "Convert each fraction to that denominator: "
            + "; ".join(
                f"{n}/{d} = {n * (lcd // d)}/{lcd}" if d != lcd else f"{n}/{d} already has denominator {lcd}"
                for n, d in zip(numerators, denoms)
            )
            + ".",
            f"Combine the numerators: {answer_text}.",
        ]
        return dict(
            text=f"Calculate {expr_text}.",
            latex=r"\text{Calculate }" + expr_text,
            answer={"p": answer.numerator, "q": answer.denominator},
            answer_text=answer_text,
            answer_latex=latex_answer(answer_text),
            working=working,
            distractors=distractors,
        )

    raise RuntimeError(
        f"could not build a valid fraction_add_subtract question for tier={tier!r} in {_RESAMPLE_ATTEMPTS} attempts"
    )


@generator(
    "p10e_fraction_add_subtract_equivalent",
    level=5,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def p10e_fraction_add_subtract_equivalent(rng, tier="easy"):
    """Add/subtract fractions via common-denominator equivalence.

    easy: two fractions, addition only, one denominator an exact multiple of
    the other (only one term converts). medium: two fractions, addition or
    subtraction, denominators share a factor but neither divides the other
    (both terms convert -- a genuine escalation over easy's single
    conversion). hard: three-term chain mixing addition and subtraction,
    denominators all related via a common LCM, sometimes reducing to a
    smaller denominator than the LCD.
    """
    return _build(rng, tier)
