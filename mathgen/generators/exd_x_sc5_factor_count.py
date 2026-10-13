"""Exam-tier source for `prime-factorisation`: use the prime factorisation to count factors.

A stated number of items is packed into equal groups. Every group size must divide the number, so the learner has
to factorise it into primes and then count, which is a use of the factorisation rather than a recall of it:

* (a) how many different group sizes are possible (2 marks): ``(e1 + 1)(e2 + 1)...`` for ``N = p1^e1 p2^e2 ...``;
* (b) how many of those sizes are odd (2 marks): the same product with the factor for 2 left out;
* (c) how many are even (1 mark, follow-through: ``part_a - part_b``);
* (d) the largest odd group size (1 mark): the number with every factor of 2 taken out.

Counts are brute-forced by trial division, an independent route from the exponent product.
"""

from __future__ import annotations

import math
import random

from ..answers import Distractor
from ..core import GRADER_TOLERANCE, check, draw_until, generator, latex_answer, pick_distractors

_PRIMES = [2, 3, 5, 7, 11]

_ITEMS = [
    ("A florist has {n} roses to pack into bunches.", "bunch"),
    ("A school canteen has {n} muffins to put into trays.", "tray"),
    ("A camp store has {n} bottles of water to pack into crates.", "crate"),
    ("A nursery has {n} seedlings to plant in rows.", "row"),
    ("A sports store has {n} tennis balls to pack into tubes.", "tube"),
]
_RULES = [
    "Every {unit} must hold the same number, and a {unit} may hold just one or all of them.",
    "Each {unit} must have the same number in it, and all of them may go in a single {unit}.",
]


def _factors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _factorise(n: int) -> dict[int, int]:
    out: dict[int, int] = {}
    for prime in _PRIMES:
        while n % prime == 0:
            out[prime] = out.get(prime, 0) + 1
            n //= prime
    check(n == 1, "number is not 11-smooth")
    return out


@generator("exd_x_sc5_factor_count", level=9, presets={"exam": dict(tier="exam")})
def exd_x_sc5_factor_count(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: count the factors of a number from its prime factorisation, then split them by parity."""
    check(tier == "exam", f"unknown tier {tier!r}")

    def draw(r: random.Random):
        n = 1
        for prime, cap in zip(_PRIMES, (7, 4, 3, 2, 1)):
            n *= prime ** r.randint(0, cap)
        return n

    def acceptable(n: int) -> bool:
        exps = _factorise(n)
        return n % 2 == 0 and 240 <= n <= 6000 and len(exps) >= 2 and exps.get(2, 0) >= 1 and any(p > 2 for p in exps)

    n = draw_until(rng, draw, acceptable, what="a number with a rich factorisation")
    exps = _factorise(n)
    divisors = _factors(n)
    total = len(divisors)
    odd = sum(1 for d in divisors if d % 2)
    even = total - odd
    check(total == math.prod(e + 1 for e in exps.values()), "factor count disagrees with the exponent formula")
    check(odd == math.prod(e + 1 for p, e in exps.items() if p != 2), "odd count disagrees")
    check(even > 0 and odd > 0, "degenerate split")

    big_omega = sum(exps.values())
    a_cands = [
        Distractor(str(big_omega), "counted-primes-not-factors", big_omega),
        Distractor(str(math.prod(exps.values())), "multiplied-exponents-without-adding-one", math.prod(exps.values())),
        Distractor(str(total - 2), "left-out-one-and-the-number", total - 2),
        Distractor(str(len(exps)), "counted-primes-not-factors", len(exps)),
        Distractor(
            str(sum(e + 1 for e in exps.values())), "added-instead-of-multiplied", sum(e + 1 for e in exps.values())
        ),
    ]
    odd_exps = [e for p, e in exps.items() if p != 2]
    b_cands = [
        Distractor(str(even), "counted-the-wrong-class", even),
        Distractor(str(math.prod(odd_exps)), "multiplied-exponents-without-adding-one", math.prod(odd_exps)),
        Distractor(str(total - exps[2]), "counted-the-wrong-class", total - exps[2]),
        Distractor(str(len(exps) - 1), "counted-primes-not-factors", len(exps) - 1),
        Distractor(str(odd + 1), "off-by-one-endpoint", odd + 1),
    ]
    c_cands = [
        Distractor(str(odd), "counted-the-wrong-class", odd),
        Distractor(str(exps[2]), "counted-the-wrong-class", exps[2]),
        Distractor(str(total), "counted-the-wrong-class", total),
        Distractor(str(even - 1), "left-out-one-and-the-number", even - 1),
    ]
    odd_part = n // 2 ** exps[2]
    check(odd_part == max(d for d in divisors if d % 2), "largest odd factor disagrees")
    d_cands = [
        Distractor(str(n), "left-in-factors-of-two", n),
        Distractor(str(n // 2), "left-in-factors-of-two", n // 2),
        Distractor(str(max(p for p in exps if p > 2)), "gave-largest-prime-factor", max(p for p in exps if p > 2)),
        Distractor(str(n // 2 ** (exps[2] - 1)), "left-in-factors-of-two", n // 2 ** (exps[2] - 1)),
    ]
    a_d = pick_distractors(str(total), a_cands, 3, tolerance=GRADER_TOLERANCE)
    b_d = pick_distractors(str(odd), b_cands, 3, tolerance=GRADER_TOLERANCE)
    c_d = pick_distractors(str(even), c_cands, 3, tolerance=GRADER_TOLERANCE)
    d_d = pick_distractors(str(odd_part), d_cands, 3, tolerance=GRADER_TOLERANCE)
    check(min(len(a_d), len(b_d), len(c_d), len(d_d)) >= 2, "too few distractors survive")

    sentence, unit = rng.choice(_ITEMS)
    factorisation = " x ".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(exps.items()))
    return dict(
        text=f"{sentence.format(n=n)} {rng.choice(_RULES).format(unit=unit)}",
        parts=[
            dict(
                prompt=f"How many different numbers of items could go in one {unit}?",
                answer_kind="number",
                answer=total,
                answer_text=str(total),
                answer_latex=latex_answer(str(total)),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"{n} = {factorisation}.",
                    "The number of factors is the product of (exponent + 1) over each prime: "
                    + " x ".join(f"({e} + 1)" for e in exps.values())
                    + f" = {total}.",
                ],
                distractors=a_d,
            ),
            dict(
                prompt=f"How many of those numbers are odd?",
                answer_kind="number",
                answer=odd,
                answer_text=str(odd),
                answer_latex=latex_answer(str(odd)),
                marks=2,
                sensible_range=(0, None),
                working=[
                    "An odd factor uses no 2s, so only the odd primes' powers can vary: "
                    + " x ".join(f"({e} + 1)" for p, e in exps.items() if p != 2)
                    + f" = {odd}."
                ],
                distractors=b_d,
            ),
            dict(
                prompt="How many of those numbers are even?",
                answer_kind="number",
                answer=even,
                answer_text=str(even),
                answer_latex=latex_answer(str(even)),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"part_a - part_b",
                working=[f"{total} - {odd} = {even}."],
                distractors=c_d,
            ),
            dict(
                prompt=f"What is the greatest odd number of items that could go in one {unit}?",
                answer_kind="number",
                answer=odd_part,
                answer_text=str(odd_part),
                answer_latex=latex_answer(str(odd_part)),
                marks=1,
                sensible_range=(0, None),
                working=[f"Take every factor of 2 out of {n} = {factorisation}: {odd_part}."],
                distractors=d_d,
            ),
        ],
    )
