"""Exam-tier questions for finite sample space probability: a CONDITIONAL
probability over the ordered outcomes of two dice.

The sample space is not listed; the learner must build it (``n * m`` ordered
pairs) and then restrict it to the outcomes where the given condition ``B``
holds before counting those where ``A`` also holds::

    P(A | B) = |A and B| / |B|

Each condition is a named predicate on the pair ``(a, b)``. The answer is
computed by enumerating the pairs, so it is exact for every seed. A draw is
rejected unless the conditional probability is strictly between 0 and 1 and
differs from the unconditional ``P(A)`` (otherwise the conditioning would be
cosmetic) and from ``P(B | A)`` (otherwise the reversed conditional would be an
unprincipled distractor that the grader accepts).
"""

from __future__ import annotations

import random
from fractions import Fraction
from itertools import product

from ..core import check, generator, pick_distractors, rational_text

# phrase -> predicate on (a, b). Phrases read after "the sum is..." style leads, so each is a full clause.
_CLAUSES = (
    ("the sum is even", lambda a, b: (a + b) % 2 == 0),
    ("the sum is a multiple of 3", lambda a, b: (a + b) % 3 == 0),
    ("the product is even", lambda a, b: (a * b) % 2 == 0),
    ("the first die shows a smaller number than the second", lambda a, b: a < b),
    ("the two numbers differ by exactly 1", lambda a, b: abs(a - b) == 1),
    ("the two numbers are different", lambda a, b: a != b),
    ("the first number divides the second", lambda a, b: b % a == 0),
)


def _threshold_clause(rng: random.Random, n: int, m: int):
    k = rng.randint(max(3, (n + m) // 2 - 1), (n + m) // 2 + 2)
    if rng.random() < 0.5:
        return f"the sum is at least {k}", lambda a, b, k=k: a + b >= k
    return f"the larger number is at most {k - n // 2}", lambda a, b, k=k: max(a, b) <= k - n // 2


@generator(
    "exd_x_s2_sample_space",
    level=10,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_s2_sample_space(rng: random.Random, tier: str = "exam") -> dict:
    """P(A | B) over the ordered outcomes of a pair of dice."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    attempts = 0
    while True:
        attempts += 1
        if attempts > 500:
            raise ValueError("no non-degenerate conditional event found")
        n, m = rng.choice(((6, 6), (4, 6), (6, 8), (5, 6), (4, 8), (6, 10)))
        clauses = list(_CLAUSES) + [_threshold_clause(rng, n, m)]
        (name_a, pred_a), (name_b, pred_b) = rng.sample(clauses, 2)
        space = list(product(range(1, n + 1), range(1, m + 1)))
        a_set = {s for s in space if pred_a(*s)}
        b_set = {s for s in space if pred_b(*s)}
        both = a_set & b_set
        if not both or len(both) == len(b_set):
            continue
        answer = Fraction(len(both), len(b_set))
        p_a = Fraction(len(a_set), len(space))
        reversed_cond = Fraction(len(both), len(a_set))
        if answer in (p_a, reversed_cond):
            continue
        break
    check(0 < answer < 1, "conditional probability must be strictly between 0 and 1")
    check(answer == Fraction(len(both), len(b_set)), "conditional probability mismatch")
    shapes = (
        "A fair {n}-sided die and a fair {m}-sided die are rolled. Given that {b}, what is the probability that {a}?",
        "Two dice, one with {n} faces and one with {m} faces, are rolled together. "
        "It is known that {b}. Find the probability that {a}.",
        "Roll a fair {n}-sided die and then a fair {m}-sided die. Find P({a} | {b}).",
    )
    text = shapes[rng.randrange(len(shapes))].format(n=n, m=m, a=name_a, b=name_b)
    answer_text = rational_text(answer)
    candidates = [
        Fraction(len(both), len(space)),  # divides by the whole sample space, not the restricted one
        reversed_cond,  # conditions on A instead of B
        p_a,  # ignores the condition
        1 - answer,  # complement
        Fraction(len(both), len(b_set) + len(a_set)),
    ]
    return dict(
        text=text,
        latex="plain:" + text,
        answer={"p": answer.numerator, "q": answer.denominator},
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"The sample space has {n}*{m} = {len(space)} ordered outcomes.",
            f"Outcomes where {name_b}: {len(b_set)}. Of those, outcomes where {name_a}: {len(both)}.",
            f"P = {len(both)}/{len(b_set)} = {answer_text}.",
        ],
        distractors=pick_distractors(answer_text, [rational_text(c) for c in candidates]),
    )
