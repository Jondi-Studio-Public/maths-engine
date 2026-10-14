"""Generates `exd_x_r3j_euler_power_residue` questions: a large power modulo a product of two primes.

Skill: `ucv1-number-theory-computation` (exam tier). The practice tiers are handed the factorisation and compute
one power residue; an exam gives only n and a large exponent, so the learner must find the structure first:

(a) the larger prime factor of n                                    number, 1 mark
(b) phi(n)                                                          number, 2 marks, (q - 1)(n/q - 1)
(c) the remainder r when e is divided by phi(n)                     number, 1 mark,  e - phi floor(e / phi)
(d) the remainder when m^e is divided by n                          number, 2 marks, m^r - n floor(m^r / n)

## Construction

n = p q for two distinct primes, m is coprime to n (so m^e = m^r (mod n) with r = e mod phi(n), which is the
shortcut the chain leads to), and e = k phi + r with k from 1 to 4 and r from 2 to 40, so the reduced exponent is
small enough to finish by hand with repeated squaring. Part (c) has a `sensible_range` of (0, n) so that a wild
reply to (c) never reaches the power in (d)'s follow-through.
"""

from __future__ import annotations

import math

from sympy import primerange

from ..core import attempts, check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._exd_x_r3j_kit import number_part

_PRIMES = list(primerange(5, 24))
_STEMS = [
    "The number n = {n} is the product of two distinct primes. Let m = {m} and e = {e}.",
    "Let n = {n}, which is a product of two distinct primes, and let m = {m} and e = {e}.",
    "A number n = {n} is the product of two different primes, and m = {m} and e = {e} are given.",
]


@generator("exd_x_r3j_euler_power_residue", level=17, presets={"exam": {}})
@distinct_part_answers
def exd_x_r3j_euler_power_residue(rng):
    for _ in attempts(2000, "a modulus, base and exponent with a small reduced exponent"):
        p, q = sorted(rng.sample(_PRIMES, 2))
        n, phi = p * q, (p - 1) * (q - 1)
        m = rng.randint(2, 40)
        r = rng.randint(2, 40)
        k = rng.randint(1, 4)
        e = k * phi + r
        if not (math.gcd(m, n) == 1 and r < phi and m < n and e > 100 and pow(m, r, n) not in (0, 1, m)):
            continue
        # Parts (c) and (d) each need two distractors that differ from the key.
        wrong_c = {k, e % n, e % (p - 1), e % (q - 1), *({e - phi} if k > 1 else set())} - {r}
        wrong_d = {pow(m, e % n, n), pow(m, e % (p - 1), n), pow(m, r, q), (m * r) % n, pow(m, k, n)} - {pow(m, e, n)}
        if len(wrong_c) >= 2 and len(wrong_d) >= 2:
            break
    check(pow(m, e, n) == pow(m, r, n), "Euler's reduction disagrees with direct exponentiation")
    answer = pow(m, e, n)

    stem = rng.choice(_STEMS).format(n=n, m=m, e=e)
    parts = [
        number_part(
            "Find the larger prime factor of n.",
            q,
            0,
            1,
            [f"{n} = {p} × {q}."],
            [
                (p, "gave-other-root"),
                (p + q, "sum-or-difference-of-factors-for-factor"),
                (q - p, "sum-or-difference-of-factors-for-factor"),
            ],
            sensible_range=(1, n),
        ),
        number_part(
            "Find φ(n).",
            phi,
            0,
            2,
            [f"φ({n}) = ({p} − 1)({q} − 1) = {phi}."],
            [
                (n - 1, "totient-as-n-minus-one"),
                ((p - 1) + (q - 1), "added-instead-of-multiplied"),
                (n - p - q, "totient-dropped-the-plus-one"),
                (p * (q - 1), "totient-used-one-prime-unreduced"),
                (p * q - p, "totient-used-one-prime-unreduced"),
            ],
            follow_through=f"(part_a - 1)*({n}/part_a - 1)",
            sensible_range=(0, n),
        ),
        number_part(
            "Find the remainder when e is divided by φ(n).",
            r,
            0,
            1,
            [f"{e} = {k}×{phi} + {r}, so the remainder is {r}."],
            [
                (k, "quotient-for-remainder"),
                (e % n, "reduced-exponent-modulo-n"),
                *([(e - phi, "stopped-reducing-the-exponent")] if k > 1 else []),
                (e % (p - 1), "reduced-exponent-modulo-one-prime"),
                (e % (q - 1), "reduced-exponent-modulo-one-prime"),
            ],
            follow_through=f"{e} - part_b*floor({e}/part_b)",
            sensible_range=(0, n),
        ),
        number_part(
            "Find the remainder when m^e is divided by n.",
            answer,
            0,
            2,
            [
                f"Since gcd({m}, {n}) = 1, {m}^{e} ≡ {m}^{r} (mod {n}).",
                f"{m}^{r} ≡ {answer} (mod {n}).",
            ],
            [
                (pow(m, e % n, n), "reduced-exponent-modulo-n"),
                (pow(m, e % (p - 1), n), "reduced-exponent-modulo-one-prime"),
                (pow(m, r, q), "residue-modulo-one-factor-only"),
                ((m * r) % n, "multiplied-instead-of-exponentiated"),
                (pow(m, k, n), "quotient-for-remainder"),
            ],
            follow_through=f"{m}**part_c - {n}*floor({m}**part_c/{n})",
        ),
    ]
    return dict(text=stem, parts=parts)
