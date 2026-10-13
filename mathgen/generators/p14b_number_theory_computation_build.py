"""Generates `p14b_number_theory_computation` questions: m^e mod n computed via Euler's theorem from a supplied
prime factorisation of n.

Ucv1-number-theory-computation.

Kernel: given a modulus n together with its SUPPLIED prime factorisation, an
exponent e and a base m, compute m^e mod n by a method that genuinely
REQUIRES the factorisation -- Euler's theorem, reducing e modulo phi(n)
(computed from the supplied factors) before the final, much smaller modular
exponentiation. e is always constructed several times larger than phi(n),
so reducing via the factorisation is genuinely load-bearing, not decorative:
computing m^e mod n directly on the unreduced e would do far more
multiplications.

**Boundary against `ucv1-rsa-modular-step`** (already shipping,
`mathgen/generators/p12f_ftc_rsa_build.py`): that skill "NEVER uses a
supplied factorisation of the exponent's reduction modulus; it is a bare
repeated-squaring computation". Every question
here states the factorisation as a given input and every construction
literally reduces the exponent through phi(n) computed from it -- a bare
`m^e mod n` is never offered as this skill's own method, only as a
misconception ("forgot to reduce the exponent").

**Boundary against `modular-arithmetic-congruence`/
`modular-multiplicative-inverse`**: those are single-step congruence/inverse
primitives with no exponentiation at all.

**Boundary against `ucv1-chinese-remainder-two-congruences`**: that skill
solves two SUPPLIED congruences directly. This skill's exam tier also uses
CRT, but only after DERIVING the two congruences itself (m^e mod p and
m^e mod q, from the supplied factorisation of n = p*q) -- a genuinely
different, two-stage process from combining two already-stated congruences.

Tiers:
- easy: n = p*q (two supplied primes), a large stated e, base m coprime to
  n; reduce e mod phi(n) = (p-1)(q-1), then compute the much smaller power.
- medium: same n = p*q shape, but e is given as an unevaluated EXPRESSION
  (2^a + b) the learner must first evaluate.
- hard: n = p^2*q (a repeated prime factor), requiring the generalised
  Euler totient product formula phi(p^2*q) = p(p-1)(q-1) rather than the
  two-factor (p-1)(q-1).
- exam: the base m is NOT coprime to one of n's prime factors (m is a
  multiple of p), so the Euler shortcut (which requires coprimality) is
  invalid; the correct route computes m^e mod p and m^e mod q separately
  (mod p trivially 0, mod q by Fermat since m IS coprime to q) and combines
  them with the Chinese Remainder Theorem.
"""

from __future__ import annotations

from ..core import check, generator, phrase, pick_distractors

_EASY_PRIMES = (5, 7, 11, 13, 17, 19)
_HARD_P_PRIMES = (5, 7, 11)
_HARD_Q_PRIMES = (13, 17, 19, 23, 29)
_EXAM_PRIMES = (5, 7, 11, 13, 17, 19)


def _two_factor_phi(p: int, q: int) -> int:
    return (p - 1) * (q - 1)


def _prime_power_phi(p: int, a: int) -> int:
    """phi(p**a) = p**(a-1) * (p - 1), the general prime-power Euler formula."""
    return p ** (a - 1) * (p - 1)


def _crt2(r1: int, m1: int, r2: int, m2: int) -> int:
    """The unique x in [0, m1*m2) with x = r1 (mod m1) and x = r2 (mod m2),
    for coprime m1, m2 -- the standard two-modulus CRT combination formula."""
    inv = pow(m1, -1, m2)
    k = ((r2 - r1) * inv) % m2
    return (r1 + m1 * k) % (m1 * m2)


def _coprime_m(rng, n: int, *bad_factors: int) -> int:
    while True:
        m = rng.randint(2, n - 2)
        if all(m % f != 0 for f in bad_factors):
            return m


def _truncated_exponent_distractor(rng, m: int, e: int, n: int, answer: int) -> str:
    """Shared misconception: forgets to reduce the exponent through the
    factorisation at all, and instead just truncates e to a smaller,
    plausible-looking value (drops the last digit) -- a genuine slip, not
    "the right answer computed the slow way"."""
    e0 = e // 10 if e >= 20 else max(e - 1, 1)
    val = pow(m, e0, n)
    if val == answer:
        e0 = max(e0 - 1, 1)
        val = pow(m, e0, n)
    return str(val)


def _wrong_modulus_distractor(m: int, e: int, n: int, answer: int) -> str:
    """Shared misconception: confuses the modulus n with the totient
    phi(n) as the exponent-reduction modulus."""
    val = pow(m, e % n, n)
    return str(val)


_EASY_TEXT = (
    "n = {p} × {q} = {n}, using this factorisation of n. Compute m^e mod n for m = {m}, e = {e}.",
    "A modulus n = {n} is supplied together with its factorisation n = {p} × {q}. Find m^e mod n for m = {m} and e = {e}.",
    "The modulus n = {n} factors as {p} × {q}. Using this factorisation, compute m^e mod n where m = {m}, e = {e}.",
)

_MEDIUM_TEXT = (
    "n = {p} × {q} = {n}, using this factorisation of n. The exponent is e = 2^{a} + {b}. Compute m^e mod n for m = {m}.",
    "A modulus n = {n} is supplied together with its factorisation n = {p} × {q}. The exponent e = 2^{a} + {b}. Find m^e mod n for m = {m}.",
    "The modulus n = {n} factors as {p} × {q}. First evaluate e = 2^{a} + {b}, then compute m^e mod n for m = {m}, using the factorisation of n.",
)

_HARD_TEXT = (
    "n = {p}^2 × {q} = {n}, using this prime factorisation of n. Compute m^e mod n for m = {m}, e = {e}.",
    "A modulus n = {n} is supplied together with its factorisation n = {p}^2 × {q}. Find m^e mod n for m = {m} and e = {e}.",
    "The modulus n = {n} factors as {p}^2 × {q}. Using this factorisation, compute m^e mod n where m = {m}, e = {e}.",
)

_EXAM_TEXT = (
    "n = {p} × {q} = {n}. Compute m^e mod n for m = {m}, e = {e}.",
    "Given n = {n}, with factorisation n = {p} × {q}: find m^e mod n for m = {m} and e = {e}.",
    "The modulus n = {n} factors as {p} × {q}. Find m^e mod n for m = {m}, e = {e}.",
)


@generator(
    "p14b_number_theory_computation",
    level=16,
    presets={
        "easy": dict(variant="euler_two_factor"),
        "medium": dict(variant="euler_expression_exponent"),
        "hard": dict(variant="euler_prime_power"),
        "exam": dict(variant="crt_noncoprime"),
    },
)
def number_theory_computation(rng, variant="euler_two_factor"):
    if variant == "euler_two_factor":
        return _euler_two_factor(rng)
    if variant == "euler_expression_exponent":
        return _euler_expression_exponent(rng)
    if variant == "euler_prime_power":
        return _euler_prime_power(rng)
    if variant == "crt_noncoprime":
        return _crt_noncoprime(rng)
    raise ValueError(f"unknown variant {variant!r}")


def _euler_two_factor(rng):
    p, q = rng.sample(_EASY_PRIMES, 2)
    n = p * q
    phi = _two_factor_phi(p, q)

    while True:
        # m is redrawn on every retry too: a base whose multiplicative order
        # happens to divide both phi(n) and n makes the "uses n instead of
        # phi" distractor collide with the true answer for EVERY exponent,
        # not just this one draw, so only re-sampling e cannot escape it.
        m = _coprime_m(rng, n, p, q)
        factor = rng.randint(3, 6)
        remainder = rng.randint(1, phi - 1)
        e = phi * factor + remainder
        e_reduced = e % phi
        answer = pow(m, e_reduced, n)
        # The independent oracle-grade check: agree with Python's own
        # three-argument pow on the ORIGINAL, unreduced e and n.
        check(answer == pow(m, e, n), "euler_two_factor: factorisation-based route disagrees with pow(m, e, n)")
        trunc = _truncated_exponent_distractor(rng, m, e, n, answer)
        wrong_mod = _wrong_modulus_distractor(m, e, n, answer)
        if trunc != str(answer) and wrong_mod != str(answer):
            break
    cands = pick_distractors(str(answer), [trunc, wrong_mod])

    text = phrase(rng, _EASY_TEXT, p=p, q=q, n=n, m=m, e=e)
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[
            f"phi(n) = ({p} - 1)({q} - 1) = {phi}",
            f"e mod phi(n) = {e} mod {phi} = {e_reduced}",
            f"m^e mod n = {m}^{e_reduced} mod {n} = {answer}",
        ],
        distractors=cands,
    )


def _euler_expression_exponent(rng):
    p, q = rng.sample(_EASY_PRIMES, 2)
    n = p * q
    phi = _two_factor_phi(p, q)

    while True:
        # m is redrawn every retry -- see _euler_two_factor's comment.
        m = _coprime_m(rng, n, p, q)
        a = rng.randint(9, 13)
        b = rng.randint(1, 50)
        e = 2**a + b
        e_reduced = e % phi
        answer = pow(m, e_reduced, n)
        check(
            answer == pow(m, e, n), "euler_expression_exponent: factorisation-based route disagrees with pow(m, e, n)"
        )
        trunc = _truncated_exponent_distractor(rng, m, e, n, answer)
        wrong_mod = _wrong_modulus_distractor(m, e, n, answer)
        if trunc != str(answer) and wrong_mod != str(answer):
            break
    cands = pick_distractors(str(answer), [trunc, wrong_mod])

    text = phrase(rng, _MEDIUM_TEXT, p=p, q=q, n=n, m=m, a=a, b=b)
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[
            f"e = 2^{a} + {b} = {e}",
            f"phi(n) = ({p} - 1)({q} - 1) = {phi}",
            f"e mod phi(n) = {e} mod {phi} = {e_reduced}",
            f"m^e mod n = {m}^{e_reduced} mod {n} = {answer}",
        ],
        distractors=cands,
    )


def _euler_prime_power(rng):
    # Resample q away from q = 1 (mod p): when p | (q - 1), the Carmichael
    # lambda(n) already divides the WRONG two-factor totient (p-1)(q-1),
    # which makes the "drops the power correction term" misconception
    # produce the mathematically correct residue for every m and every e --
    # never a genuine misconception for that (p, q) pair.
    p = rng.choice(_HARD_P_PRIMES)
    q = rng.choice(_HARD_Q_PRIMES)
    while (q - 1) % p == 0:
        q = rng.choice(_HARD_Q_PRIMES)
    n = p * p * q
    phi = _prime_power_phi(p, 2) * (q - 1)
    # hard-specific misconception: uses the two-factor formula (p-1)(q-1),
    # dropping the power-based correction term p that phi(p^2 * q) needs.
    naive_phi = _two_factor_phi(p, q)

    while True:
        # m is redrawn every retry -- see _euler_two_factor's comment.
        m = _coprime_m(rng, n, p, q)
        factor = rng.randint(3, 6)
        remainder = rng.randint(1, phi - 1)
        e = phi * factor + remainder
        e_reduced = e % phi
        answer = pow(m, e_reduced, n)
        check(answer == pow(m, e, n), "euler_prime_power: factorisation-based route disagrees with pow(m, e, n)")
        trunc = _truncated_exponent_distractor(rng, m, e, n, answer)
        wrong_mod = _wrong_modulus_distractor(m, e, n, answer)
        dropped_power_term = pow(m, e % naive_phi, n)
        if trunc != str(answer) and wrong_mod != str(answer) and str(dropped_power_term) != str(answer):
            break
    cands = pick_distractors(str(answer), [trunc, wrong_mod, str(dropped_power_term)])

    text = phrase(rng, _HARD_TEXT, p=p, q=q, n=n, m=m, e=e)
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[
            f"phi(n) = phi({p}^2)({q} - 1) = {p}({p} - 1)({q} - 1) = {phi}",
            f"e mod phi(n) = {e} mod {phi} = {e_reduced}",
            f"m^e mod n = {m}^{e_reduced} mod {n} = {answer}",
        ],
        distractors=cands,
    )


def _crt_noncoprime(rng):
    """m is a multiple of p (gcd(m, n) = p != 1). Mod p, m^k = 0 for EVERY
    k >= 1 -- reduction cannot change that, only losing it entirely if the
    reduced exponent collapses to 0. Mod q, m IS coprime to q, and
    naive_phi = (p-1)(q-1) is always a multiple of phi(q) = q-1, so Fermat
    reduction through naive_phi alone still gives the right residue mod q
    for ANY nonzero reduced exponent -- meaning a naive Euler-shortcut
    reduction "fails silently" in exactly ONE place: when e happens to be
    an exact multiple of naive_phi, so the reduced exponent is 0 and
    pow(m, 0, n) = 1 is offered in place of the true (nonzero mod p)
    residue. e is constructed as exactly that multiple, so the shortcut's
    failure is genuine and deterministic, not a rare resampled coincidence.
    """
    p, q = rng.sample(_EXAM_PRIMES, 2)
    n = p * q
    t = rng.randint(1, q - 1)
    m = p * t  # gcd(m, n) = p != 1: m is a multiple of p, coprime to q

    q_phi = q - 1  # phi(q), the Fermat modulus for the coprime factor q used
    # by the real CRT route
    naive_phi = _two_factor_phi(p, q)  # (p-1)(q-1) -- the wrong modulus the
    # exam-specific misconception uses
    factor = rng.randint(3, 6)
    e = naive_phi * factor  # an EXACT multiple: the shortcut's reduced
    # exponent is 0, the one case it truly fails.

    m_mod_q = m % q
    e_reduced = e % q_phi  # == 0, since naive_phi is a multiple of q_phi
    residue_mod_q = pow(m_mod_q, e_reduced, q)  # correctly 1, by Fermat
    # mod p: m = p*t is a multiple of p, so m^e = 0 (mod p) for e >= 1
    # (the ORIGINAL, unreduced e -- never affected by any exponent
    # reduction, since any positive power of a multiple of p is 0 mod p).
    answer = _crt2(0, p, residue_mod_q, q)
    # The independent oracle-grade check: agree with Python's own
    # three-argument pow on the ORIGINAL, unreduced e and n.
    check(answer == pow(m, e, n), "crt_noncoprime: CRT-combined route disagrees with pow(m, e, n)")

    trunc = _truncated_exponent_distractor(rng, m, e, n, answer)

    # exam-specific misconception: applies the Euler shortcut anyway despite
    # gcd(m, n) != 1 -- reduces the WHOLE exponent through naive_phi and
    # computes a single pow(m, e mod naive_phi, n) = pow(m, 0, n) = 1,
    # never splitting into the p and q components CRT requires.
    applies_euler_shortcut = pow(m, e % naive_phi, n)
    check(applies_euler_shortcut == 1, "crt_noncoprime: the naive shortcut was expected to collapse to 1")

    while trunc == str(answer) or str(applies_euler_shortcut) == str(answer):
        t = rng.randint(1, q - 1)
        m = p * t
        e_reduced = e % q_phi
        m_mod_q = m % q
        residue_mod_q = pow(m_mod_q, e_reduced, q)
        answer = _crt2(0, p, residue_mod_q, q)
        check(answer == pow(m, e, n), "crt_noncoprime: CRT-combined route disagrees with pow(m, e, n) after resample")
        trunc = _truncated_exponent_distractor(rng, m, e, n, answer)
        applies_euler_shortcut = pow(m, e % naive_phi, n)

    check(
        trunc != str(answer) and str(applies_euler_shortcut) != str(answer),
        "crt_noncoprime: a distractor equals the answer",
    )
    cands = pick_distractors(str(answer), [trunc, str(applies_euler_shortcut)])

    text = phrase(rng, _EXAM_TEXT, p=p, q=q, n=n, m=m, e=e)
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[
            f"gcd(m, n) = {p} != 1, so Euler's theorem does not apply directly.",
            f"m^e mod {p} = 0, since {p} | m.",
            f"m^e mod {q} = {m_mod_q}^{e_reduced} mod {q} = {residue_mod_q} (Fermat, e mod (q-1) = {e_reduced}).",
            f"Combine by CRT: m^e mod n = {answer}",
        ],
        distractors=cands,
    )
