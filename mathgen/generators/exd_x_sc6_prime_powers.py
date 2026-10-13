"""Exam-tier source for `prime-factorisation-exponent`: read a prime factorisation off a product of composite bases.

The stem gives a number N as a product of powers whose bases are themselves composite, such as ``12³ × 45² × 7``.
Three connected parts follow:

* (a) the exponent of the smallest prime in N written as a product of prime powers,
* (b) the exponent of the middle prime,
* (c) the number of positive factors of N, which uses the whole prime factorisation (the exponent of the largest
  prime is read straight off the stem's factors, so (c) follows through from (a) and (b)).

N is never shown as a single number, so the learner has to split every base into primes and distribute the outer
exponent before collecting like primes: the collected exponent of a prime is the sum, over the factors, of
(the prime's multiplicity in the base) x (the outer exponent).
"""

from __future__ import annotations

import random
from math import prod

from sympy import factorint

from mathgen.answers import Distractor
from mathgen.core import check, generator, latex_answer, pick_distractors

_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
_COMPOSITE_BASES = (6, 10, 12, 14, 15, 18, 20, 24, 28, 30, 36, 40, 45, 50)
_PRIME_BASES = (7, 11)


def _power_text(base: int, exponent: int) -> str:
    return str(base) if exponent == 1 else f"{base}{str(exponent).translate(_SUPERSCRIPT)}"


def _draw(rng: random.Random):
    """Factors ``[(base, exponent), ...]`` whose product has exactly three distinct primes."""
    for _ in range(1000):
        bases = rng.sample(_COMPOSITE_BASES, 2)
        factors = [(bases[0], rng.randint(2, 4)), (bases[1], rng.randint(1, 3))]
        if rng.random() < 0.4:
            factors.append((rng.choice(_PRIME_BASES), rng.randint(1, 2)))
        rng.shuffle(factors)
        primes = _collected(factors)
        if len(primes) == 3 and max(primes.values()) <= 12 and len(set(primes.values())) >= 2:
            return factors
    raise AssertionError("no factor set with three distinct primes in 1000 draws")


def _collected(factors) -> dict[int, int]:
    total: dict[int, int] = {}
    for base, exponent in factors:
        for p, multiplicity in factorint(base).items():
            total[p] = total.get(p, 0) + multiplicity * exponent
    return total


def _exponent_candidates(factors, p: int, true: int) -> list[Distractor]:
    """Wrong exponents for the prime `p`, each a principled slip."""
    containing = [(base, e, factorint(base)[p]) for base, e in factors if p in factorint(base)]
    # Counts each base once for each outer power, ignoring how many times p divides the base.
    ignores_inner = sum(e for _, e, _ in containing)
    # Uses the multiplicity inside the base but forgets the outer exponent.
    ignores_outer = sum(m for _, _, m in containing)
    # Stops after the first base that contains p.
    first_only = containing[0][1] * containing[0][2]
    # Multiplies the contributions of different bases instead of adding them.
    multiplied = prod(e * m for _, e, m in containing)
    out = [
        Distractor(str(ignores_inner), "ignores-inner-multiplicity", ignores_inner),
        Distractor(str(ignores_outer), "ignores-outer-exponent", ignores_outer),
    ]
    if len(containing) > 1:
        out.append(Distractor(str(first_only), "uses-one-base-only", first_only))
        out.append(Distractor(str(multiplied), "multiplies-exponents-not-adds", multiplied))
    out.append(Distractor(str(true + 1), "off-by-one-exponent", true + 1))
    if true > 1:
        out.append(Distractor(str(true - 1), "off-by-one-exponent", true - 1))
    out.append(Distractor(str(true + 2), None, true + 2))
    return out


@generator("exd_x_sc6_prime_powers", level=8, presets={"exam": dict(tier="exam")})
def exd_x_sc6_prime_powers(rng: random.Random, tier: str = "exam") -> dict:
    check(tier == "exam", f"unknown tier {tier!r}")
    factors = _draw(rng)
    collected = _collected(factors)
    primes = sorted(collected)
    p1, p2, p3 = primes
    e1, e2, e3 = collected[p1], collected[p2], collected[p3]
    # Independent route: factorise the actual product.
    check(factorint(prod(b**e for b, e in factors)) == collected, "prime exponents disagree with factorint")
    count = (e1 + 1) * (e2 + 1) * (e3 + 1)

    stem = "The number N is given by N = " + " × ".join(_power_text(b, e) for b, e in factors) + "."

    def tagged(candidates, true):
        picked = pick_distractors(str(true), candidates, 3)
        check(len(picked) >= 2, f"too few distractors for key {true}")
        check(all(d.value != true for d in picked), "distractor equals key")
        return picked

    count_candidates = [
        Distractor(str(e1 * e2 * e3), "multiplied-exponents-without-adding-one", e1 * e2 * e3),
        Distractor(str(e1 + e2 + e3), "counted-primes-not-factors", e1 + e2 + e3),
        Distractor(str((e1 + 1) * (e2 + 1) * e3), None, (e1 + 1) * (e2 + 1) * e3),
        Distractor(str((e1 + 1) + (e2 + 1) + (e3 + 1)), "counted-primes-not-factors", (e1 + 1) + (e2 + 1) + (e3 + 1)),
    ]

    def part(label, prompt, value, marks, **extra):
        return dict(
            prompt=prompt,
            answer_kind="number",
            answer=value,
            answer_text=str(value),
            answer_latex=latex_answer(str(value)),
            marks=marks,
            sensible_range=(0, None),
            **extra,
        )

    working_primes = " × ".join(f"{p}{str(collected[p]).translate(_SUPERSCRIPT)}" for p in primes)
    return dict(
        text=stem,
        parts=[
            part(
                "a",
                f"Write N as a product of prime powers. What is the exponent of {p1}?",
                e1,
                2,
                working=[f"Split each base into primes and multiply the exponents out: N = {working_primes}."],
                distractors=tagged(_exponent_candidates(factors, p1, e1), e1),
            ),
            part(
                "b",
                f"What is the exponent of {p2} in the prime factorisation of N?",
                e2,
                1,
                working=[f"The exponent of {p2} is {e2}."],
                distractors=tagged(_exponent_candidates(factors, p2, e2), e2),
            ),
            part(
                "c",
                "Use the prime factorisation to find the number of positive factors of N.",
                count,
                2,
                follow_through=f"(part_a + 1) * (part_b + 1) * {e3 + 1}",
                working=[
                    "The number of factors is the product of (exponent + 1) over the primes.",
                    f"({e1} + 1)({e2} + 1)({e3} + 1) = {count}.",
                ],
                distractors=tagged(count_candidates, count),
            ),
        ],
    )
