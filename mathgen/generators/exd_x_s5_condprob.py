"""Exam question: a conditional probability that must first be assembled from other probabilities.

The learner is never given the intersection. They are given P(A), P(B) and one more fact that pins
down P(A and B) -- the union, the probability of neither event, or the probability of A without B --
and must choose the route (inclusion-exclusion, complement, or subtraction) before dividing by P(B).

Built backwards from exact hundredths: P(A and B) is drawn first, then P(A) and P(B) strictly larger,
so the stated facts are consistent by construction. Draws where A and B happen to be independent are
rejected (the "assume independence" slip would otherwise coincide with the key). The key is an exact
fraction in lowest terms; the grader compares fractions by value.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, generator, phrase, pick_distractors, rational_text

_UNION = (
    "Events A and B satisfy P(A) = {a}, P(B) = {b} and P(A ∪ B) = {u}. Find P(A | B), giving your answer "
    "as a fraction in simplest form.",
    "For two events A and B, P(A) = {a}, P(B) = {b}, and the probability that at least one of them occurs is "
    "{u}. What is the probability of A given B? Give a fraction in simplest form.",
)
_NEITHER = (
    "Two events A and B have P(A) = {a} and P(B) = {b}. The probability that neither event occurs is {n}. "
    "Calculate P(A | B) as a fraction in simplest form.",
    "In an experiment, P(A) = {a} and P(B) = {b}, and P(neither A nor B) = {n}. Find the probability that A "
    "occurs given that B has occurred, as a fraction in simplest form.",
)
_ONLY_A = (
    "For events A and B, P(A) = {a}, P(B) = {b} and the probability that A occurs but B does not is {o}. "
    "Find P(A | B), giving a fraction in simplest form.",
    "P(A) = {a} and P(B) = {b}. The probability of A without B is {o}. What is the probability of A given "
    "B? Write your answer as a fraction in simplest form.",
)


def _h(x: Fraction) -> str:
    check((x * 100).denominator == 1, "condprob: value off the hundredths grid")
    s = f"{float(x):.2f}"
    return s


def _build(rng: random.Random) -> dict:
    for _ in range(1000):
        i = Fraction(rng.randint(5, 40), 100)
        a = i + Fraction(rng.randint(5, 40), 100)
        b = i + Fraction(rng.randint(5, 40), 100)
        u = a + b - i
        if a == b or i == a * b or u > Fraction(95, 100) or a >= 1 or b >= 1:
            continue
        key = i / b
        reversed_cond = i / a
        candidates = [i, reversed_cond, a, i / u]  # joint, P(B|A), assumes independence, divides by the union
        answer_text = rational_text(key)
        cand_texts = [rational_text(c) for c in candidates]
        distractors = pick_distractors(answer_text, cand_texts)
        if len(distractors) < 3:
            continue
        form = rng.choice(("union", "neither", "only"))
        if form == "union":
            text = phrase(rng, _UNION, a=_h(a), b=_h(b), u=_h(u))
            given = f"P(A and B) = {_h(a)} + {_h(b)} - {_h(u)} = {_h(i)}."
        elif form == "neither":
            text = phrase(rng, _NEITHER, a=_h(a), b=_h(b), n=_h(1 - u))
            given = f"P(A or B) = 1 - {_h(1 - u)} = {_h(u)}, so P(A and B) = {_h(a)} + {_h(b)} - {_h(u)} = {_h(i)}."
        else:
            text = phrase(rng, _ONLY_A, a=_h(a), b=_h(b), o=_h(a - i))
            given = f"P(A and B) = P(A) - P(A without B) = {_h(a)} - {_h(a - i)} = {_h(i)}."
        # Independent check: counts out of 100 equally likely outcomes.
        in_a = int(a * 100)
        in_b = int(b * 100)
        both = int(i * 100)
        check(in_a + in_b - both == int(u * 100), "condprob: union does not match the counts")
        check(Fraction(both, in_b) == key, "condprob: cell-count ratio disagrees with the key")
        return dict(
            text=text,
            latex=text,
            answer={"p": key.numerator, "q": key.denominator},
            answer_text=answer_text,
            answer_latex=answer_text,
            distractors=distractors[:3],
            working=[given, f"P(A | B) = P(A and B) / P(B) = {_h(i)} / {_h(b)} = {answer_text}."],
        )
    raise RuntimeError("exd_x_s5_condprob: no draw satisfied the constraints in 1000 attempts")


@generator("exd_x_s5_condprob", level=12, presets={"exam": dict(tier="exam")})
def exd_x_s5_condprob(rng: random.Random, tier: str = "exam") -> dict:
    """P(A | B) where the intersection must be deduced from a union, a complement or a difference."""
    check(tier == "exam", "exd_x_s5_condprob: only the exam tier exists")
    return _build(rng)
