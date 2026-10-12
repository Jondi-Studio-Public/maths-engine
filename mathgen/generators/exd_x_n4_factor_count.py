"""Exam-tier source for `list-factors`: counting factors from a prime factorisation instead of listing them.

A number is given as a product of prime powers, too large to list the factors of by hand, so the learner has to
see the structure of the factors:

    (a) the value of N, (b) how many positive factors N has, (c) how many of those are not perfect squares (or
    not multiples of one of the primes).

Part (b) is the product of each exponent plus one. Part (c) takes a count of the unwanted factors away from part
(b), so it follows through from the learner's own (b). Built backwards from the primes and exponents.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, latex_answer, pick_distractors

_PRIMES = (2, 3, 5, 7)
_CONTEXTS = (
    "A school in Darwin orders N exercise books",
    "A nursery in Tasmania plants N seedlings",
    "A surf club in Cronulla stocks N water bottles",
    "A farm near Toowoomba packs N eggs",
    "A library in Ballarat shelves N novels",
)
_COUNT_PHRASES = (
    "How many positive factors does N have?",
    "How many different positive whole numbers divide N exactly?",
)


def _factor_string(primes, exponents) -> str:
    return " × ".join(str(p) if e == 1 else f"{p}^{e}" for p, e in zip(primes, exponents))


def _divisors(n: int) -> list[int]:
    """Every positive divisor of n, found by trial division up to sqrt(n) (independent of the exponent formula)."""
    small = [d for d in range(1, math.isqrt(n) + 1) if n % d == 0]
    return sorted({*small, *(n // d for d in small)})


def _number_tokens(*texts: str) -> set[int]:
    return {int(t) for text in texts for t in re.findall(r"\d+", text)}


@generator("exd_x_n4_factor_count", level=9, presets={"exam": {"tier": "exam"}})
def exd_x_n4_factor_count(rng, tier: str = "exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a factorisation whose counts are all distinct from the numbers shown"):
        primes = sorted(rng.sample(_PRIMES, 3))
        exponents = [rng.randint(1, 4) for _ in primes]
        value = math.prod(p**e for p, e in zip(primes, exponents))
        if value > 200_000:
            continue
        total = math.prod(e + 1 for e in exponents)
        kind = rng.choice(("squares", "multiples"))
        if kind == "squares":
            unwanted = math.prod(e // 2 + 1 for e in exponents)
            wanted_phrase = "are not perfect squares"
            wrong_unwanted = math.prod(e // 2 for e in exponents)
        else:
            index = rng.randrange(3)
            p = primes[index]
            others = math.prod(e + 1 for i, e in enumerate(exponents) if i != index)
            unwanted = exponents[index] * others
            wanted_phrase = f"are not multiples of {p}"
            wrong_unwanted = None
        answer_c = total - unwanted
        context = rng.choice(_CONTEXTS)
        stem = f"{context}, where N = {_factor_string(primes, exponents)}."
        prompts = (
            "Find the value of N.",
            rng.choice(_COUNT_PHRASES),
            f"Hence find how many of the positive factors of N {wanted_phrase}.",
        )
        shown = _number_tokens(stem, *prompts)
        if {value, total, answer_c} & shown or answer_c <= 1:
            continue

        a_text, b_text, c_text = str(value), str(total), str(answer_c)
        cands_a = [
            Distractor(str(math.prod(primes)), "index-ignored", math.prod(primes)),
            Distractor(
                str(math.prod(p * e for p, e in zip(primes, exponents))),
                "multiplied-base-exponent",
                math.prod(p * e for p, e in zip(primes, exponents)),
            ),
        ]
        cands_b = [
            Distractor(str(math.prod(exponents)), "forgot-zero-exponent-choice", math.prod(exponents)),
            Distractor(
                str(sum(e + 1 for e in exponents)), "factor-count-sum-of-exponents", sum(e + 1 for e in exponents)
            ),
            Distractor(str(sum(exponents)), "factor-count-sum-of-exponents", sum(exponents)),
        ]
        cands_c = [Distractor(str(unwanted), "answered-complement", unwanted)]
        if wrong_unwanted:  # a zero here would just repeat part (b)
            cands_c.append(
                Distractor(str(total - wrong_unwanted), "forgot-zero-exponent-choice", total - wrong_unwanted)
            )
        if total % 2 == 0:
            cands_c.append(Distractor(str(total // 2), "assumed-half-of-factors", total // 2))
        d_a = pick_distractors(a_text, cands_a, 3)
        d_b = pick_distractors(b_text, cands_b, 3)
        d_c = pick_distractors(c_text, cands_c, 3)
        if min(len(d_a), len(d_b), len(d_c)) >= 2:
            break

    # Independent routes: brute-force counting from the value itself.
    divisors = _divisors(value)
    check(len(divisors) == total, "factor count disagrees with brute force")
    if kind == "squares":
        check(sum(math.isqrt(d) ** 2 == d for d in divisors) == unwanted, "square-factor count disagrees")
    else:
        check(sum(d % p == 0 for d in divisors) == unwanted, "multiples count disagrees with brute force")
    check(Fraction(value) == math.prod(Fraction(p) ** e for p, e in zip(primes, exponents)), "value is off")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=prompts[0],
                answer_kind="number",
                answer=value,
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=1,
                sensible_range=(0, None),
                working=[f"N = {_factor_string(primes, exponents)} = {a_text}."],
                distractors=d_a,
            ),
            dict(
                prompt=prompts[1],
                answer_kind="number",
                answer=total,
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    "A factor uses each prime's power from 0 up to its exponent, so there are "
                    + " × ".join(f"({e}+1)" for e in exponents)
                    + f" = {b_text} choices."
                ],
                distractors=d_b,
            ),
            dict(
                prompt=prompts[2],
                answer_kind="number",
                answer=answer_c,
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_b-{unwanted}",
                working=[
                    f"There are {unwanted} factors that "
                    + ("are perfect squares" if kind == "squares" else f"are multiples of {p}")
                    + f", so {b_text} − {unwanted} = {c_text}."
                ],
                distractors=d_c,
            ),
        ],
    )
