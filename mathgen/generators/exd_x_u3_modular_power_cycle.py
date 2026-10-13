"""Exam-tier source for `modular-arithmetic-congruence`: a huge power reduced through its cycle.

The practice tiers reduce an integer, or a sum of two integers, modulo m. Here the integer is a^N with N in the
hundreds or thousands, which no one can write down, so the learner must see that the powers of a repeat. Part (a)
finds the length of the cycle (the least k with a^k = 1 mod m); part (b) reduces the exponent by it and finishes
the arithmetic. Part (b) carries `follow_through`: a wrong cycle length from (a) used correctly still earns (b).

Built backwards from m, a unit a and the exponent, and checked against Python's three-argument `pow`.
"""

from __future__ import annotations

import math

from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, latex_answer, pick_distractors


_STEMS = [
    "The powers of {a} are reduced modulo {m}: each power {a}^1, {a}^2, {a}^3, ... is replaced by its remainder on "
    "division by {m}.",
    "A counter in a game cycles through the remainders of {a}^1, {a}^2, {a}^3, ... on division by {m}.",
    "A checksum program in a hardware shop in Dubbo works with remainders on division by {m}, and needs to "
    "understand the sequence of powers of {a}.",
]
_A_PROMPTS = [
    "Find the smallest positive whole number k such that {a}^k leaves remainder 1 when divided by {m}.",
    "After how many steps does the sequence of remainders of the powers of {a} first return to 1? Give the "
    "number of steps k, where {a}^k leaves remainder 1 on division by {m}.",
    "Find the length k of the repeating cycle: the least positive k with {a}^k ≡ 1 (mod {m}).",
]
_B_PROMPTS = [
    "Using your value of k from (a), find the remainder when {a}^{n} is divided by {m}.",
    "Using your cycle length from (a), work out {a}^{n} (mod {m}) as a number from 0 to {m} - 1.",
    "Hence, using your value of k, find the remainder on dividing {a}^{n} by {m}.",
]


def _order(a: int, m: int) -> int:
    value, k = a % m, 1
    while value != 1:
        value = value * a % m
        k += 1
        check(k <= m, "base is not a unit")
    return k


@generator("exd_x_u3_modular_power_cycle", level=14, presets={"exam": dict(tier="exam")})
def exd_x_u3_modular_power_cycle(rng, tier: str = "exam") -> dict:
    """Exam tier: the cycle length of the powers of a modulo m, then a^N mod m."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a base, modulus and exponent with a long enough cycle"):
        m = rng.randint(11, 97)
        a = rng.randint(2, m - 2)
        if math.gcd(a, m) != 1:
            continue
        k = _order(a, m)
        n = rng.randint(120, 4999)
        r = n % k
        if k < 5 or r in (0, 1) or n % m == r or k == m - 1 or k + 1 == m:
            continue
        key = pow(a, n, m)
        if key in (0, 1, a, m) or key == n % m or k in (a, m, n):
            continue
        break
    check(pow(a, n, m) == pow(a, r, m), "reducing the exponent by the cycle length changed the residue")
    check(all(pow(a, j, m) != 1 for j in range(1, k)) and pow(a, k, m) == 1, "k is not the least cycle length")

    stem = rng.choice(_STEMS).format(a=a, m=m)
    a_prompt = rng.choice(_A_PROMPTS).format(a=a, m=m)
    b_prompt = rng.choice(_B_PROMPTS).format(a=a, m=m, n=n)
    k_cand = [
        Distractor(str(k + 1), "cycle-length-off-by-one", k + 1),
        Distractor(str(k - 1), "cycle-length-off-by-one", k - 1),
        Distractor(str(m), "cycle-length-is-modulus", m),
        Distractor(str(m - 1), None, m - 1),
        Distractor(str(2 * k), None, 2 * k),
    ]
    key_cand = [
        Distractor(str(pow(a, n % m, m)), "exponent-not-reduced-by-cycle", pow(a, n % m, m)),
        Distractor(str(pow(a, n // k, m)), None, pow(a, n // k, m)),
        Distractor(str(n % m), None, n % m),
        Distractor(str(a * n % m), None, a * n % m),
        Distractor(str(pow(a, (r + 1) % k, m)), None, pow(a, (r + 1) % k, m)),
    ]
    k_d = pick_distractors(str(k), k_cand, 3)
    key_d = pick_distractors(str(key), key_cand, 3)
    check(len(k_d) >= 2 and len(key_d) >= 2, "too few distractors")

    # Exponent reduced by the learner's own cycle length: n mod part_a, written with floor (no "%" in expressions).
    r_expr = f"{n} - part_a*floor({n}/part_a)"
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=a_prompt,
                answer_kind="number",
                answer=k,
                answer_text=str(k),
                answer_latex=latex_answer(str(k)),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"Reduce successive powers of {a} modulo {m} until the remainder is 1.",
                    f"{a}^{k} ≡ 1 (mod {m}), and no smaller positive power is, so k = {k}.",
                ],
                distractors=k_d,
            ),
            dict(
                prompt=b_prompt,
                answer_kind="number",
                answer=key,
                answer_text=str(key),
                answer_latex=latex_answer(str(key)),
                marks=3,
                follow_through=f"{a}**({r_expr}) - {m}*floor({a}**({r_expr})/{m})",
                working=[
                    f"Since {a}^{k} ≡ 1, only the remainder of the exponent on division by {k} matters: {n} = {n // k} × {k} + {r}.",
                    f"{a}^{n} ≡ {a}^{r} ≡ {key} (mod {m}).",
                ],
                distractors=key_d,
            ),
        ],
    )
