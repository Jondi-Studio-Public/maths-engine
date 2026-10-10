"""Generates `exd_x_s1_covariance` questions: covariance of linearly transformed variables built from a
joint description that is not a plain probability table.

Two forms, each answered as an exact fraction:

* a 2-by-2 table of counts for X and Y, and the wanted quantity is ``Cov(aX + b, cY + d)`` -- the learner
  must compute ``Cov(X, Y)`` and know that it scales by ``a*c`` and ignores the shifts;
* two events A and B are described by P(A), P(B) and P(A and B); X and Y are their indicator variables
  and the wanted quantity is ``Cov(aX + b, cY + d)``, so the joint distribution has to be recovered first.

Both forms are cross-checked by building the full joint table of the transformed variables and applying
the centred definition of covariance.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, check_equivalent, generator, pick_distractors, rational_text


def _payload(value: Fraction) -> dict:
    return {"p": value.numerator, "q": value.denominator}


def _direct_cov(cells: list[tuple[Fraction, Fraction, Fraction]]) -> Fraction:
    """Centred covariance from (u, v, probability) triples."""
    eu = sum(u * w for u, v, w in cells)
    ev = sum(v * w for u, v, w in cells)
    return sum((u - eu) * (v - ev) * w for u, v, w in cells)


def _transform_text(coef: int, shift: int, name: str) -> str:
    sign = "+" if shift > 0 else "-"
    return f"{coef}{name} {sign} {abs(shift)}"


def _pick_transform(rng: random.Random) -> tuple[int, int, int, int]:
    a = rng.choice([-3, -2, 2, 3, 4])
    c = rng.choice([-3, -2, 2, 3, 5])
    b = rng.choice([x for x in range(-4, 5) if x != 0])
    d = rng.choice([x for x in range(-4, 5) if x != 0])
    return a, b, c, d


def _table_form(rng: random.Random) -> dict:
    while True:
        xs = sorted(rng.sample(range(-3, 4), 2))
        ys = sorted(rng.sample(range(-3, 4), 2))
        counts = [[rng.randint(1, 6) for _ in ys] for _ in xs]
        total = sum(map(sum, counts))
        cells = [
            (Fraction(x), Fraction(y), Fraction(counts[i][j], total))
            for i, x in enumerate(xs)
            for j, y in enumerate(ys)
        ]
        cov = _direct_cov(cells)
        if cov != 0:
            break
    a, b, c, d = _pick_transform(rng)
    answer = a * c * cov
    moved = [(a * x + b, c * y + d, w) for x, y, w in cells]
    check_equivalent(answer, _direct_cov(moved), "direct covariance of the transformed variables disagrees")
    exy = sum(x * y * w for x, y, w in cells)
    cands = [
        cov,  # ignores the transform
        (abs(a) + abs(c)) * cov,  # adds the coefficients instead of multiplying
        a * c * exy,  # forgets to subtract E[X]E[Y]
        a * c * cov + b * d,  # lets the shifts contribute
    ]
    rows = "; ".join(f"X={x}, Y={y}: {counts[i][j]}" for i, x in enumerate(xs) for j, y in enumerate(ys))
    text = (
        f"One of {total} equally likely cases is chosen. The number of cases with each pair of values is: {rows}. "
        f"Find Cov({_transform_text(a, b, 'X')}, {_transform_text(c, d, 'Y')})."
    )
    return dict(answer=answer, text=text, cands=cands)


def _indicator_form(rng: random.Random) -> dict:
    while True:
        den = rng.choice([6, 8, 10, 12, 15, 20])
        pa = rng.randint(2, den - 2)
        pb = rng.randint(2, den - 2)
        pab = rng.randint(1, min(pa, pb) - 1)
        # P(A or B) must not exceed 1: pa + pb - pab <= den ; keep counts in a common denominator
        if pa + pb - pab > den:
            continue
        # make the joint "probability of both" a multiple of 1/(den*den)-free form: pab/den is stated directly
        p_a, p_b, p_ab = Fraction(pa, den), Fraction(pb, den), Fraction(pab, den)
        cov = p_ab - p_a * p_b
        if cov != 0:
            break
    a, b, c, d = _pick_transform(rng)
    answer = a * c * cov
    cells = [
        (Fraction(1), Fraction(1), p_ab),
        (Fraction(1), Fraction(0), p_a - p_ab),
        (Fraction(0), Fraction(1), p_b - p_ab),
        (Fraction(0), Fraction(0), 1 - p_a - p_b + p_ab),
    ]
    check(all(w >= 0 for _, _, w in cells), "negative cell probability")
    check_equivalent(
        answer,
        _direct_cov([(a * x + b, c * y + d, w) for x, y, w in cells]),
        "direct covariance of the transformed indicators disagrees",
    )
    cands = [
        cov,  # ignores the transform
        a * c * p_ab,  # forgets to subtract P(A)P(B)
        a * c * p_a * p_b,  # only the product of the marginals
        a * c * cov + b * d,  # lets the shifts contribute
    ]
    text = (
        f"For events A and B, P(A) = {rational_text(p_a)}, P(B) = {rational_text(p_b)} and "
        f"P(A and B) = {rational_text(p_ab)}. Let X = 1 if A occurs and 0 otherwise, and let Y = 1 if B occurs "
        f"and 0 otherwise. Find Cov({_transform_text(a, b, 'X')}, {_transform_text(c, d, 'Y')})."
    )
    return dict(answer=answer, text=text, cands=cands)


@generator(
    "exd_x_s1_covariance",
    level=17,
    presets={"exam": dict(tier="exam")},
)
def exd_x_s1_covariance(rng: random.Random, tier: str = "exam") -> dict:
    """Covariance of linearly transformed variables from a table or from event probabilities."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in range(50):
        built = _table_form(rng) if rng.random() < 0.5 else _indicator_form(rng)
        answer: Fraction = built["answer"]
        check(answer != 0, "zero covariance is ambiguous")
        answer_text = rational_text(answer)
        distractors = pick_distractors(answer_text, [rational_text(c) for c in built["cands"]])
        if len(distractors) >= 2:
            break
    else:
        raise RuntimeError("could not build enough distractors")
    return dict(
        text=built["text"],
        latex=f"plain:{built['text']}",
        answer=_payload(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"Cov = {answer_text}."],
        distractors=distractors,
    )
