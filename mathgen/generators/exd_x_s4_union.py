"""Exam-tier questions for the complement and union rules of probability.

The easier tiers apply one rule to numbers that are handed over directly. Here the learner must combine
the complement rule with inclusion-exclusion, and work BACKWARDS from a union or a "neither" probability to a
quantity that is not stated:

  exactly_one -- given P(A), P(B) and P(A or B), find P(exactly one of A, B occurs).
                 P(A and B) = P(A) + P(B) - P(A or B) must be found first, then
                 P(exactly one) = P(A) + P(B) - 2 P(A and B).
  intersection -- given P(not A), P(B) and P(A or B), find P(A and B).
                 The complement rule gives P(A); inclusion-exclusion is then solved for the overlap.
  neither -- given P(A), P(B) and P(neither A nor B), find P(A and B).
                 The complement of the union is P(A or B) = 1 - P(neither).

Every question is built from the numerators ``a``, ``b`` (single events) and ``c`` (the overlap) over one
denominator ``d``, so the stated values and the key are exact. The key is re-derived with ``Fraction`` from
the stated values alone before it is returned.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, check_distinct, generator, phrase, pick_distractors, rational_text

_DENOMS = (12, 15, 16, 18, 20, 24, 25, 30, 36, 40)


def _draw(rng):
    """Numerators with a genuine, proper overlap: 1 <= c < min(a, b) and a + b - c < d."""
    d = rng.choice(_DENOMS)
    for _ in range(500):
        a = rng.randint(3, d - 3)
        b = rng.randint(3, d - 3)
        c = rng.randint(1, min(a, b) - 1) if min(a, b) > 1 else 0
        if c >= 1 and a + b - c < d - 1:
            return d, a, b, c
    raise RuntimeError("exd_x_s4_union: no valid draw")


def _frac(n, d):
    return f"{n}/{d}"


def _ok(value, d):
    """A distractor must be a probability: 0 < value < 1."""
    return 0 < Fraction(value, d) < 1


def _build(rng, variant):
    d, a, b, c = _draw(rng)
    u = a + b - c
    if variant == "exactly_one":
        answer = Fraction(a + b - 2 * c, d)
        text = phrase(
            rng,
            (
                "For two events A and B, P(A) = {a}/{d}, P(B) = {b}/{d} and P(A or B) = {u}/{d}. "
                "What is the probability that exactly one of the two events occurs?",
                "In an experiment P(A) = {a}/{d} and P(B) = {b}/{d}. The probability that at least one of "
                "A and B occurs is {u}/{d}. Find the probability that A occurs or B occurs, but not both.",
                "Events A and B satisfy P(A) = {a}/{d}, P(B) = {b}/{d} and P(A or B) = {u}/{d}. "
                "Find the probability that one and only one of them occurs.",
            ),
            a=a,
            b=b,
            u=u,
            d=d,
        )
        # Misconceptions: reports the union (never removes both overlaps), forgets the overlap altogether,
        # reports the overlap itself, or reports the complement of the answer.
        cands = [u, a + b, c, d - (a + b - 2 * c)]
        working = [
            f"P(A and B) = P(A) + P(B) - P(A or B) = {_frac(a, d)} + {_frac(b, d)} - {_frac(u, d)} = {_frac(c, d)}.",
            f"Exactly one: P(A) + P(B) - 2 P(A and B) = {rational_text(answer)}.",
        ]
    elif variant == "intersection":
        answer = Fraction(c, d)
        text = phrase(
            rng,
            (
                "P(not A) = {x}/{d}, P(B) = {b}/{d} and P(A or B) = {u}/{d}. Find P(A and B).",
                "For events A and B, the probability that A does not occur is {x}/{d}. Also P(B) = {b}/{d} "
                "and P(A or B) = {u}/{d}. What is the probability that both events occur?",
                "Given P(not A) = {x}/{d}, P(B) = {b}/{d} and P(A or B) = {u}/{d}, "
                "find the probability that A and B both occur.",
            ),
            x=d - a,
            b=b,
            u=u,
            d=d,
        )
        # Misconceptions: uses P(not A) as if it were P(A); finds P(A only); finds P(B only); forgets to
        # subtract the union from the sum of the singles; reports the union itself.
        cands = [d - a + b - u, a - c, b - c, a + b, u]
        working = [
            f"P(A) = 1 - P(not A) = {_frac(a, d)}.",
            f"P(A and B) = P(A) + P(B) - P(A or B) = {_frac(a, d)} + {_frac(b, d)} - {_frac(u, d)} = "
            f"{rational_text(answer)}.",
        ]
    else:  # neither
        answer = Fraction(c, d)
        n = d - u
        text = phrase(
            rng,
            (
                "P(A) = {a}/{d} and P(B) = {b}/{d}. The probability that neither event occurs is {n}/{d}. "
                "Find P(A and B).",
                "For events A and B, P(A) = {a}/{d} and P(B) = {b}/{d}, and P(neither A nor B) = {n}/{d}. "
                "What is the probability that both events occur?",
                "Events A and B are such that P(A) = {a}/{d}, P(B) = {b}/{d} and the chance that neither "
                "happens is {n}/{d}. Find the probability that they both happen.",
            ),
            a=a,
            b=b,
            n=n,
            d=d,
        )
        # Misconceptions: treats P(neither) as P(A or B); reports the union; reports P(exactly one); reports
        # P(neither) itself.
        cands = [a + b - n, u, a + b - 2 * c, n]
        working = [
            f"P(A or B) = 1 - P(neither) = {_frac(u, d)}.",
            f"P(A and B) = P(A) + P(B) - P(A or B) = {rational_text(answer)}.",
        ]
    return d, a, b, c, u, answer, text, cands, working


@generator(
    "exd_x_s4_union",
    level=11,
    presets={"exam": dict(tier="exam")},
)
def exd_x_s4_union(rng, tier="exam"):
    variant = rng.choice(("exactly_one", "intersection", "neither"))
    d, a, b, c, u, answer, text, cands, working = _build(rng, variant)

    # Independent re-derivation from the inclusion-exclusion identities, using sets of equally likely points.
    only_a, only_b, both = a - c, b - c, c
    outside = d - (only_a + only_b + both)
    check(outside >= 1, "the sample space must have points outside A and B")
    expected = {
        "exactly_one": Fraction(only_a + only_b, d),
        "intersection": Fraction(both, d),
        "neither": Fraction(both, d),
    }[variant]
    check(answer == expected, "key disagrees with the point-count derivation")
    check(0 < answer < 1, "the key must be a proper probability")

    answer_text = rational_text(answer)
    good = [Fraction(x, d) for x in cands if _ok(x, d) and Fraction(x, d) != answer]
    distractors = pick_distractors(answer_text, [rational_text(g) for g in good], n=3)
    check_distinct(answer, *[Fraction(x) for x in distractors])
    check(len(distractors) >= 2, "need at least two principled distractors")

    return dict(
        text=text,
        latex=text,
        answer={"p": answer.numerator, "q": answer.denominator},
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )
