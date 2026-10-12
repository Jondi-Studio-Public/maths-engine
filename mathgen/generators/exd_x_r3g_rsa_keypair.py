"""Exam tier for "compute one RSA modular step": build a key pair, then decrypt.

The practice tiers hand the learner the modulus and the exponent. Here only the two primes, the public exponent
and a ciphertext are given, and the learner works the whole decryption through four connected parts:

(a) the modulus n = p q;
(b) Euler's totient phi(n) = (p - 1)(q - 1), which follows from the primes (and from n, via p + q);
(c) the private exponent d, the inverse of e modulo phi(n);
(d) the plaintext m = c^d mod n.

Built backwards from the message: draw the primes, an exponent e (an odd prime coprime to phi), and a message m
with gcd(m, n) = 1, and encrypt it. Each key is re-derived with Python's modular `pow`.

## Follow-through

SymPy has no modular inverse that the follow-through parser may call, so (c) is written with Fermat's little
theorem: for a prime e that does not divide phi, phi^(e-2) is the inverse of phi modulo e, so
d = (1 + k * phi) / e with k = e - (phi^(e-2) mod e) is the inverse of e modulo phi lying in (0, phi). (d) is
`c^d - n * floor(c^d / n)`. Part (c) has a sensible range of (0, phi), which also keeps the power in (d) small
whatever a learner types for d.
"""

from __future__ import annotations

import math
import random

from ..core import attempts, generator
from ._exd_x_r3_distinct import distinct_answers
from ._exd_x_r3g_kit import clashes, number_part

_PRIMES = (7, 11, 13, 17, 19, 23)
_EXPONENTS = (3, 5, 7, 11, 13)
_NAMES = (("Alice", "Bob"), ("Maya", "Jin"), ("Zara", "Ethan"), ("Nadia", "Oscar"))
_STEMS = (
    "{a} generates an RSA key pair from the primes p = {p} and q = {q}, with public exponent e = {e}. "
    "{b} encrypts a message m (an integer with 1 < m < n) as c = m^e mod n and sends {a} the ciphertext c = {c}.",
    "An RSA system is set up with primes p = {p} and q = {q} and public exponent e = {e}. The ciphertext "
    "c = {c} was produced from a message m with 1 < m < n using c = m^e mod n.",
    "For a small RSA example, {a} chooses the primes p = {p} and q = {q} and the public exponent e = {e}. "
    "{b} sends {a} the encrypted message c = {c}, where c = m^e mod n for a message m with 1 < m < n.",
)


def _draw(rng: random.Random):
    p, q = rng.sample(_PRIMES, 2)
    e = rng.choice(_EXPONENTS)
    n, phi = p * q, (p - 1) * (q - 1)
    if e in (p, q) or math.gcd(e, phi) != 1:
        return None
    m = rng.randint(2, n - 2)
    if math.gcd(m, n) != 1:
        return None
    c = pow(m, e, n)
    d = pow(e, -1, phi)
    if c in (m, 0, 1) or d == e or d < 3:
        return None
    return p, q, e, n, phi, d, m, c


@generator("exd_x_r3g_rsa_keypair", level=12, presets={"exam": dict(tier="exam")})
@distinct_answers
def exd_x_r3g_rsa_keypair(rng: random.Random, tier: str = "exam") -> dict:
    """From two primes, e and a ciphertext: n, phi(n), d, then the plaintext."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(2000, "an RSA example with a clean decryption"):
        drawn = _draw(rng)
        if drawn is None:
            continue
        p, q, e, n, phi, d, m, c = drawn
        a, b = rng.choice(_NAMES)
        stem = rng.choice(_STEMS).format(a=a, b=b, p=p, q=q, e=e, c=c)
        if clashes((n, phi, d, m), stem):
            continue
        if len({pow(c, e, n), pow(c, d, phi), c * d % n, c} - {m}) < 2:
            continue  # the last part needs two distractors that differ from the key
        break

    inverse_mod_n = pow(e, -1, n) if math.gcd(e, n) == 1 else None
    w = f"part_b**{e - 2}"
    d_expr = f"(1 + ({e} - ({w} - {e}*floor(({w})/{e})))*part_b)/{e}"
    parts = [
        number_part(
            "Find the modulus n.",
            n,
            0,
            1,
            [f"n = p q = {p} x {q} = {n}."],
            [(p + q, "added-instead-of-multiplied"), (phi, "modulus-and-totient-swapped")],
        ),
        number_part(
            "Find the value of Euler's totient φ(n).",
            phi,
            0,
            1,
            [f"φ(n) = (p - 1)(q - 1) = {p - 1} x {q - 1} = {phi}."],
            [
                (n - 1, "totient-as-n-minus-one"),
                (n - p - q, "totient-dropped-the-plus-one"),
                (n, "modulus-and-totient-swapped"),
                ((p - 1) + (q - 1), "added-instead-of-multiplied"),
            ],
            follow_through=f"part_a - {p + q} + 1",
            sensible_range=(0, None),
        ),
        number_part(
            "Find the private exponent d: the integer with 0 < d < φ(n) such that e × d leaves remainder 1 "
            "when divided by φ(n).",
            d,
            0,
            2,
            [f"Solve {e}d ≡ 1 (mod {phi}).", f"d = {d}, since {e} x {d} = {e * d} = {(e * d - 1) // phi} x {phi} + 1."],
            [
                (inverse_mod_n, "rsa-inverse-taken-modulo-n"),
                (phi - d, "sign-of-result-reversed"),
                (e, "rsa-private-exponent-equals-public"),
                (1 + (phi + 1) // e, "rsa-private-exponent-from-phi-over-e"),
            ],
            follow_through=d_expr,
            sensible_range=(0, phi),
        ),
        number_part(
            "Decrypt the ciphertext: find the message m.",
            m,
            0,
            2,
            [f"m = c^d mod n = {c}^{d} mod {n}.", f"m = {m}."],
            [
                (pow(c, e, n), "rsa-decrypted-with-public-exponent"),
                (pow(c, d, phi), "rsa-power-reduced-modulo-totient"),
                (c * d % n, "multiplied-instead-of-exponentiated"),
                (c, "rsa-ciphertext-reported-as-message"),
            ],
            follow_through=f"{c}**part_c - part_a*floor({c}**part_c/part_a)",
            sensible_range=(1, None),
        ),
    ]
    return dict(text=stem, parts=parts)
