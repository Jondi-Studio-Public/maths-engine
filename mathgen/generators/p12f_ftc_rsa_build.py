"""Generates `p12f_rsa_modular_step` questions: one RSA modular-exponentiation step with small stated numbers.

One RSA modular-exponentiation step.

``ucv1-ftc-polynomial`` was found to duplicate the already-shipping ``definite-integral`` skill
(``skills/integration/definite-integral.yaml``, same kernel: evaluate a definite polynomial integral by
finding an antiderivative and applying endpoint subtraction, `answer: number`, misconceptions
`forgot-subtract-lower`/`swapped-limits`). Only ``ucv1-rsa-modular-step`` is built here.

Kernel: compute one RSA modular-exponentiation step, c = m^e mod n
(encrypt) or m = c^d mod n (decrypt), for small stated parameters. This is
structurally different from every shipping number-theory neighbour:
`modular-arithmetic-congruence` reduces a given integer/sum mod n (no
exponentiation at all); `modular-multiplicative-inverse` finds x with
ax = 1 (mod n) (an inverse, not a power); `euclidean-algorithm-gcd`
computes a gcd. None of the three ever raises a base to a stated exponent
mod n.
"""

from __future__ import annotations

from math import gcd

from ..core import check, generator, phrase, pick_distractors


def _mod_pow_manual(base: int, exp: int, modulus: int) -> int:
    """Independent route: binary-expansion repeated squaring, computed by
    hand-written loop code. Never calls Python's built-in three-argument
    `pow`, so this is a genuinely different code path from the one
    construction uses to compute the RSA step itself."""
    result = 1
    b = base % modulus
    e = exp
    while e > 0:
        if e & 1:
            result = (result * b) % modulus
        b = (b * b) % modulus
        e >>= 1
    return result


def _rsa_params(rng, primes, exponents):
    """Draw a coprime (p, q, e) triple and derive n, phi and d = e^-1 mod phi."""
    while True:
        p, q = rng.sample(list(primes), 2)
        n = p * q
        phi = (p - 1) * (q - 1)
        e = rng.choice(list(exponents))
        if gcd(e, phi) == 1:
            break
    d = pow(e, -1, phi)
    return p, q, n, phi, e, d


def _rsa_message(rng, n, e):
    """Draw plaintext m in [2, n-2], resampling away from any RSA fixed
    point (m^e mod n == m) -- a well-known trivial-case artefact of small
    RSA parameters that would let a learner who does nothing at all get the
    right answer."""
    while True:
        m = rng.randint(2, n - 2)
        if pow(m, e, n) != m:
            return m


def _misconception_candidates(base, exp, other_exp, n, phi, actual):
    """The four misconceptions, computed the same way for both the
    encrypt direction (base=m, exp=e, other_exp=d) and the decrypt
    direction (base=c, exp=d, other_exp=e).

    `forgot_mod_reduction` (the unreduced power) grows exponentially with
    `exp` while every other option and the answer stay in n's own range
    (`answer < n`). Past a small exponent that unreduced value is tens of
    digits long next to a 2-3 digit answer -- identifiable by SHAPE alone
    (length), not by doing the arithmetic, which the build
    conventions rule out ("no option identifiable by its shape",
    `midpoint_coordinate_scalar`). Gate it out once its digit count would
    give it away; the other three misconceptions stay in n's own range
    regardless of exp, so they never need this gate.
    """
    forgot_mod_reduction = pow(base, exp)
    used_phi_instead_of_n = pow(base, exp, phi)
    multiplied_instead_of_exponentiated = (base * exp) % n
    used_wrong_exponent = pow(base, other_exp, n)
    candidates = [str(used_phi_instead_of_n), str(multiplied_instead_of_exponentiated), str(used_wrong_exponent)]
    if len(str(forgot_mod_reduction)) <= len(str(actual)) + 2:
        candidates.insert(0, str(forgot_mod_reduction))
    return pick_distractors(str(actual), candidates)


_ENCRYPT_TEMPLATES_TEXT = (
    "An RSA system uses primes p = {p}, q = {q}, so n = p*q = {n}, with "
    "public exponent e = {e}. Encrypt the message m = {m}: find c = m^e mod n.",
    "Using RSA with primes p = {p} and q = {q} (n = {n}) and public exponent "
    "e = {e}, compute c = m^e mod n for the plaintext m = {m}.",
    "RSA parameters: p = {p}, q = {q}, n = {n}, e = {e}. What is c = m^e mod n when the plaintext is m = {m}?",
)
_ENCRYPT_TEMPLATES_LATEX = (
    r"p={p},\,q={q},\,n={n},\,e={e},\,m={m};\quad \text{{find }} c = m^e \bmod n.",
    r"p={p},\,q={q},\,n={n},\,e={e},\,m={m};\quad \text{{compute }} c = m^e \bmod n.",
    r"p={p},\,q={q},\,n={n},\,e={e},\,m={m};\quad \text{{find }} c = m^e \bmod n.",
)

_DECRYPT_TEMPLATES_TEXT = (
    "An RSA system has modulus n = {n} and private exponent d = {d}. Decrypt "
    "the ciphertext c = {c}: find m = c^d mod n.",
    "Using RSA with n = {n} and private exponent d = {d}, decrypt the ciphertext c = {c} by computing m = c^d mod n.",
    "RSA decryption: n = {n}, d = {d}, ciphertext c = {c}. What is the plaintext m = c^d mod n?",
)
_DECRYPT_TEMPLATES_LATEX = (
    r"n={n},\,d={d},\,c={c};\quad \text{{find }} m = c^d \bmod n.",
    r"n={n},\,d={d},\,c={c};\quad \text{{compute }} m = c^d \bmod n.",
    r"n={n},\,d={d},\,c={c};\quad \text{{find }} m = c^d \bmod n.",
)

_CHAIN_TEMPLATES_TEXT = (
    "An RSA system uses primes p = {p} and q = {q}, with public exponent "
    "e = {e}. First find n = p*q, then encrypt the message m = {m}: find "
    "c = m^e mod n.",
    "RSA setup: p = {p}, q = {q}, e = {e}. Determine n = p*q, and then "
    "compute c = m^e mod n for the plaintext m = {m}.",
    "Given RSA primes p = {p}, q = {q} and public exponent e = {e}: work "
    "out n = p*q, and use it to find c = m^e mod n for m = {m}.",
)
_CHAIN_TEMPLATES_LATEX = (
    r"p={p},\,q={q},\,e={e},\,m={m};\quad n=pq;\quad \text{{find }} c = m^e \bmod n.",
    r"p={p},\,q={q},\,e={e},\,m={m};\quad n=pq;\quad \text{{compute }} c = m^e \bmod n.",
    r"p={p},\,q={q},\,e={e},\,m={m};\quad n=pq;\quad \text{{find }} c = m^e \bmod n.",
)


@generator(
    "p12f_rsa_modular_step",
    level=16,
    presets={
        "easy": dict(mode="encrypt", primes=(3, 5, 7, 11, 13), exponents=(3,)),
        "medium": dict(mode="encrypt", primes=(11, 13, 17, 19, 23), exponents=(5, 7)),
        "hard": dict(mode="decrypt", primes=(13, 17, 19, 23, 29, 31), exponents=(3, 5, 7)),
        "exam": dict(mode="chain", primes=(3, 5, 7, 11, 13), exponents=(3,)),
    },
)
def rsa_modular_step(rng, mode="encrypt", primes=(3, 5, 7, 11, 13), exponents=(3,)):
    p, q, n, phi, e, d = _rsa_params(rng, primes, exponents)
    m = _rsa_message(rng, n, e)
    c = pow(m, e, n)

    # Independent route: hand-written repeated squaring, never the builtin
    # three-argument pow.
    check(c == _mod_pow_manual(m, e, n), "encrypt: independent repeated-squaring route disagrees")
    # Parameter-validity assertion, every seed: d really is e's inverse
    # mod phi, a real round-trip invariant.
    check(pow(c, d, n) == m, "round trip: pow(c, d, n) != m -- d is not e's inverse mod phi")

    if mode == "encrypt":
        cands = _misconception_candidates(m, e, d, n, phi, c)
        text = phrase(rng, _ENCRYPT_TEMPLATES_TEXT, p=p, q=q, n=n, e=e, m=m)
        latex = phrase(rng, _ENCRYPT_TEMPLATES_LATEX, p=p, q=q, n=n, e=e, m=m)
        return dict(
            text=text,
            latex=latex,
            answer=c,
            answer_text=str(c),
            answer_latex=str(c),
            working=[f"n = {p} × {q} = {n}", f"c = {m}^{e} mod {n} = {c}"],
            distractors=cands,
        )

    if mode == "decrypt":
        cands = _misconception_candidates(c, d, e, n, phi, m)
        text = phrase(rng, _DECRYPT_TEMPLATES_TEXT, n=n, d=d, c=c)
        latex = phrase(rng, _DECRYPT_TEMPLATES_LATEX, n=n, d=d, c=c)
        return dict(
            text=text,
            latex=latex,
            answer=m,
            answer_text=str(m),
            answer_latex=str(m),
            working=[f"m = {c}^{d} mod {n} = {m}"],
            distractors=cands,
        )

    # mode == "chain": exam tier -- determine n first, then encrypt.
    cands = _misconception_candidates(m, e, d, n, phi, c)
    text = phrase(rng, _CHAIN_TEMPLATES_TEXT, p=p, q=q, e=e, m=m)
    latex = phrase(rng, _CHAIN_TEMPLATES_LATEX, p=p, q=q, e=e, m=m)
    return dict(
        text=text,
        latex=latex,
        answer=c,
        answer_text=str(c),
        answer_latex=str(c),
        working=[f"n = {p} × {q} = {n}", f"c = {m}^{e} mod {n} = {c}"],
        distractors=cands,
    )
