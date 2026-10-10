"""Exam source for `ucv1-bayes-finite-counts`: a Bayes update on two independent pieces of evidence.

The practice tiers update once, on a single flag. Here each item is screened
twice by independent checks, so the joint counts for each group are the group
size times the per-check rate *twice* -- ``n * r * r`` for two flags, or
``n * r * (1 - r)`` for a flag followed by a clear. The learner has to see that
the checks multiply and then invert the table as before.

Built backwards from the drawn group sizes and rates; the answer is the exact
fraction of flagged-twice (or flag-then-clear) items that come from group 1.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, generator, phrase, pick_distractors

_PATTERNS = {
    "both": [
        "A factory has two machines. Machine 1 makes {n1} of every {tot} items and Machine 2 makes the rest. "
        "Each item is checked twice by independent inspectors. An inspector flags {a1} in {b1} items from "
        "Machine 1 and {a2} in {b2} items from Machine 2. An item is flagged by both inspectors. Find the "
        "probability that it came from Machine 1.",
        "Of every {tot} items, {n1} come from Machine 1 and the others from Machine 2. Two independent checks "
        "are run on each item. A check flags {a1} in {b1} of Machine 1's items and {a2} in {b2} of Machine 2's. "
        "Given that an item is flagged by both checks, find the probability it is from Machine 1.",
    ],
    "mixed": [
        "A factory has two machines. Machine 1 makes {n1} of every {tot} items and Machine 2 makes the rest. "
        "Each item is checked twice by independent inspectors. An inspector flags {a1} in {b1} items from "
        "Machine 1 and {a2} in {b2} items from Machine 2. An item is flagged by the first inspector but not "
        "the second. Find the probability that it came from Machine 1.",
        "Of every {tot} items, {n1} come from Machine 1 and the others from Machine 2. Two independent checks "
        "are run on each item. A check flags {a1} in {b1} of Machine 1's items and {a2} in {b2} of Machine 2's. "
        "Given that the first check flags an item and the second does not, find the probability it is from "
        "Machine 1.",
    ],
}


def _text(fr: Fraction) -> str:
    return str(fr.numerator) if fr.denominator == 1 else f"{fr.numerator}/{fr.denominator}"


@generator(
    "exd_x_s0_bayes2",
    level=12,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_s0_bayes2(rng: random.Random, tier: str = "exam") -> dict:
    """Posterior group probability after two independent checks. See module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in range(300):
        tot = rng.choice([10, 12, 15, 20, 25])
        n1 = rng.randint(2, tot - 2)
        n2 = tot - n1
        b1, b2 = rng.choice([4, 5, 8, 10]), rng.choice([4, 5, 8, 10])
        a1, a2 = rng.randint(1, b1 - 1), rng.randint(1, b2 - 1)
        r1, r2 = Fraction(a1, b1), Fraction(a2, b2)
        kind = rng.choice(["both", "mixed"])
        w1, w2 = Fraction(n1, tot), Fraction(n2, tot)
        if kind == "both":
            j1, j2 = w1 * r1 * r1, w2 * r2 * r2
            once1, once2 = w1 * r1, w2 * r2
            cands = [
                once1 / (once1 + once2),  # uses a single check only
                w1 * r1 * r1 / (w1 * r1 * r1 + w2 * r2),  # squares the rate for one group only
                r1 * r1,  # reports the likelihood
                w1,  # reports the prior
                j2 / (j1 + j2),  # the other group
            ]
        else:
            j1, j2 = w1 * r1 * (1 - r1), w2 * r2 * (1 - r2)
            cands = [
                w1 * r1 / (w1 * r1 + w2 * r2),  # ignores the second check
                w1 * r1 * r1 / (w1 * r1 * r1 + w2 * r2 * r2),  # treats "not flagged" as flagged
                r1 * (1 - r1),  # reports the likelihood
                w1,  # reports the prior
                j2 / (j1 + j2),  # the other group
            ]
        answer = j1 / (j1 + j2)
        # independent enumeration over a whole number of items
        scale = tot * b1 * b1 * b2 * b2
        c1 = n1 * scale // tot
        c2 = n2 * scale // tot
        if kind == "both":
            k1, k2 = c1 * a1 * a1 // (b1 * b1), c2 * a2 * a2 // (b2 * b2)
        else:
            k1, k2 = c1 * a1 * (b1 - a1) // (b1 * b1), c2 * a2 * (b2 - a2) // (b2 * b2)
        check(Fraction(k1, k1 + k2) == answer, "enumerated posterior mismatch")
        if len({answer, *cands}) == len(cands) + 1 and answer.denominator > 1:
            break
    else:
        raise RuntimeError("exd_x_s0_bayes2: no collision-free draw")
    text = phrase(rng, _PATTERNS[kind], tot=tot, n1=n1, a1=a1, b1=b1, a2=a2, b2=b2)
    at = _text(answer)
    return dict(
        text=text,
        answer={"p": answer.numerator, "q": answer.denominator},
        answer_text=at,
        answer_latex=at,
        working=[
            f"Weights: Machine 1 {_text(j1)}, Machine 2 {_text(j2)}; posterior = {_text(j1)}/({_text(j1 + j2)}) = {at}."
        ],
        distractors=pick_distractors(at, [_text(c) for c in cands]),
    )
