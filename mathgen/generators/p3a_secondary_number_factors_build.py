"""- ``factors_multiples_divisibility`` (AC9M5N02, prerequisites ``list-factors``,
  ``prime-factorisation``) -- given two natural numbers, determine whether one
  divides the other exactly (with no remainder). The question is
  binary classification (yes/no): does ``a`` divide
  ``b`` exactly? The independent check uses ``b % a == 0`` to verify the
  prediction is correct, but construction takes a different route (for "yes",
  construct ``b = a * k``; for "no", construct ``b`` then adjust if needed).

- ``prime_composite_square_properties`` (AC9M6N02, prerequisites ``list-factors``,
  ``prime-factorisation``) -- given a stated natural number and one of the
  properties prime, composite or perfect square, determine whether the number
  has that property. The independent check uses a different primality test
  (e.g. trial division against a different bound or precomputed table) for
  primes; perfect squares use ``math.isqrt`` to verify the "no" case.
"""

from __future__ import annotations

import math
from ..core import check, generator, latex_answer, pick_distractors

# -- factors_multiples_divisibility ------------------------------------------

_OTHER_LABEL = {"yes": "no", "no": "yes"}

_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _factored_text(factors: dict[int, int]) -> str:
    """Render {prime: exponent} as ``2³ × 3 × 5²`` (primes ascending)."""
    parts = []
    for prime in sorted(factors):
        exp = factors[prime]
        parts.append(str(prime) if exp == 1 else f"{prime}{str(exp).translate(_SUPERSCRIPT)}")
    return " × ".join(parts)


def _factored_value(factors: dict[int, int]) -> int:
    value = 1
    for prime, exp in factors.items():
        value *= prime**exp
    return value


def _prime_factors(n: int) -> dict[int, int]:
    out: dict[int, int] = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            out[d] = out.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        out[n] = out.get(n, 0) + 1
    return out


def _factored_divisibility(rng):
    """Hard-tier kind: the dividend is given in prime-factorised form, so the
    learner must compare prime powers (a divides b iff every prime power of a
    is covered by b) instead of carrying out a division.
    Returns (a, b_factors, predicted)."""
    predicted = rng.choice(["yes", "no"])
    b_primes = sorted(rng.sample([2, 3, 5, 7], 3))
    b_fac = {p: rng.randint(1, 3) for p in b_primes}
    a_fac = {p: rng.randint(0, b_fac[p]) for p in b_primes}
    if sum(1 for e in a_fac.values() if e > 0) < 2:
        a_fac[b_primes[0]] = b_fac[b_primes[0]]
        a_fac[b_primes[1]] = 1
    if predicted == "no":
        outsider = [p for p in (2, 3, 5, 7) if p not in b_fac]
        culprit = rng.choice(b_primes + outsider)
        a_fac[culprit] = b_fac.get(culprit, 0) + 1
    a_fac = {p: e for p, e in a_fac.items() if e > 0}
    a = _factored_value(a_fac)
    # Independent check: integer remainder, not prime-power comparison.
    _verify_divisibility(a, _factored_value(b_fac), predicted)
    return a, b_fac, predicted


def _verify_divisibility(a: int, b: int, predicted: str) -> None:
    """Independent check: verify b % a == 0 matches the predicted label."""
    is_divisible = b % a == 0
    predicted_divisible = predicted == "yes"
    check(
        is_divisible == predicted_divisible,
        f"{b} % {a} = {b % a}, expected divisible={predicted_divisible}",
    )


def _sample_divisibility_pair(
    rng, predicted: str, a_range: tuple[int, int], b_range: tuple[int, int], force_near_miss: bool = False
) -> tuple[int, int]:
    """Sample (a, b) pair to match the predicted divisibility outcome.

    For "yes": construct b = a * k for random k.
    For "no": construct b to not be divisible by a, optionally as a near-miss.
    """
    a_lo, a_hi = a_range
    b_lo, b_hi = b_range

    a = rng.randint(a_lo, a_hi)

    if predicted == "yes":
        # b = a * k for random k, ensuring b is in range
        k_lo = max(1, (b_lo + a - 1) // a)  # ceil(b_lo / a)
        k_hi = b_hi // a
        if k_lo > k_hi:
            raise ValueError(f"Cannot construct divisible pair in range for a={a}")
        k = rng.randint(k_lo, k_hi)
        b = a * k
    else:
        # b not divisible by a
        if force_near_miss:
            # Make b be ±1 from a multiple of a (a near-miss)
            # Try several quotients to find a valid near-miss
            quotient_min = b_lo // a
            quotient_max = b_hi // a
            candidates = []

            # Try all quotients and collect valid near-miss values
            for quotient in range(quotient_min, quotient_max + 1):
                base_multiple = quotient * a
                # Try ±1 from this multiple
                for offset in [-1, 1]:
                    b_candidate = base_multiple + offset
                    if b_lo <= b_candidate <= b_hi and b_candidate % a != 0:
                        candidates.append(b_candidate)

            if not candidates:
                raise ValueError(f"Cannot find near-miss for a={a} in range [{b_lo}, {b_hi}]")
            b = rng.choice(candidates)
        else:
            # Sample b randomly and reject if divisible
            attempts = 0
            b = rng.randint(b_lo, b_hi)
            while b % a == 0 and attempts < 500:
                b = rng.randint(b_lo, b_hi)
                attempts += 1
            if attempts == 500:
                raise ValueError(f"Cannot find non-divisible pair for a={a} after 500 attempts")

    _verify_divisibility(a, b, predicted)
    return a, b


@generator(
    "factors_multiples_divisibility",
    level=5,
    presets={
        "easy": dict(a_range=(2, 12), b_range=(2, 100), force_near_miss=False),
        "medium": dict(a_range=(2, 20), b_range=(2, 300), force_near_miss=None),  # 1 in 3
        "hard": dict(a_range=(7, 30), b_range=(100, 999), force_near_miss=True),
    },
)
def factors_multiples_divisibility(
    rng, a_range=(2, 12), b_range=(2, 100), force_near_miss=False, form="number", choose=False
):
    """Given two natural numbers a and b, determine whether a divides b exactly.
    Two-option classification: yes/no.

    ``form="factored"`` (hard tier): b is shown in prime-factorised form and the
    learner decides by comparing prime powers rather than by dividing.

    ``choose=True`` (what the skill file sets) asks the four-option form instead: which of four numbers
    does ``a`` divide exactly (hard: which of four divides the factorised number). See
    ``w2_pick_the_one``; a blind guess is then right a quarter of the time, not half."""
    if choose:
        from .w2_pick_the_one import divides_which

        return divides_which(rng, a_range, b_range, form)
    if form == "factored":
        a, b_fac, predicted = _factored_divisibility(rng)
        b_txt = _factored_text(b_fac)
        a_txt = _factored_text(_prime_factors(a))
        text = f"Does {a} divide {b_txt} exactly (with no remainder)?"
        if predicted == "yes":
            working = [f"{a} = {a_txt}, and every prime power in it is also in {b_txt}, so yes."]
        else:
            working = [f"{a} = {a_txt}, but {b_txt} does not contain all of those prime powers, so no."]
        return dict(
            text=text,
            latex=f"plain:{text}",
            answer=predicted,
            answer_text=predicted,
            answer_latex=latex_answer(predicted, plain=True),
            working=working,
            distractors=pick_distractors(predicted, [_OTHER_LABEL[predicted]], n=1),
        )
    # Choose predicted answer first
    predicted = rng.choice(["yes", "no"])

    # For medium difficulty, force near-miss on 1/3 of seeds
    if force_near_miss is None:
        force_near_miss = rng.random() < 1 / 3

    a, b = _sample_divisibility_pair(rng, predicted, a_range, b_range, force_near_miss=force_near_miss)

    if predicted == "yes":
        question_text = f"Does {a} divide {b} exactly (with no remainder)?"
        working = [f"{b} ÷ {a} = {b // a} with no remainder, so yes."]
    else:
        remainder = b % a
        question_text = f"Does {a} divide {b} exactly (with no remainder)?"
        working = [f"{b} ÷ {a} = {b // a} with remainder {remainder}, so no."]

    other_label = _OTHER_LABEL[predicted]

    return dict(
        text=question_text,
        latex=f"plain:{question_text}",
        answer=predicted,
        answer_text=predicted,
        answer_latex=latex_answer(predicted, plain=True),
        working=working,
        distractors=pick_distractors(predicted, [other_label], n=1),
    )


# -- prime_composite_square_properties ----------------------------------------


def _is_prime_trial_division_forward(n: int) -> bool:
    """Check if n is prime via trial division up to sqrt(n), checking from 2 upward."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def _is_prime_trial_division_reverse(n: int) -> bool:
    """Check if n is prime via trial division up to sqrt(n), checking from sqrt(n) downward.

    This is a genuinely different route from forward checking, so it serves as an
    independent verification: checks from larger divisors toward smaller ones.
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    limit = int(math.sqrt(n)) + 1
    for i in range(limit - 1 if (limit - 1) % 2 == 1 else limit - 2, 2, -2):
        if n % i == 0:
            return False
    return True


# Alias for use in construction
_is_prime_trial_division = _is_prime_trial_division_forward


def _is_perfect_square(n: int) -> bool:
    """Check if n is a perfect square."""
    sqrt_n = math.isqrt(n)
    return sqrt_n * sqrt_n == n


def _is_composite(n: int) -> bool:
    """Check if n is composite (not prime, and n >= 4)."""
    if n < 4:
        return False
    return not _is_prime_trial_division(n)


def _verify_property(n: int, prop: str, predicted: str) -> None:
    """Independent check using a different route than construction."""
    if prop == "prime":
        # Construction used forward trial division; verify with reverse trial division
        is_property = _is_prime_trial_division_reverse(n)
    elif prop == "composite":
        is_property = _is_composite(n)
    elif prop == "perfect square":
        is_property = _is_perfect_square(n)
    else:
        raise ValueError(f"Unknown property: {prop}")

    predicted_true = predicted == "yes"
    check(
        is_property == predicted_true,
        f"n={n}, property={prop}, is_property={is_property}, predicted={predicted}",
    )


def _sample_property_pair(
    rng, predicted: str, prop: str, n_range: tuple[int, int], force_near_miss: bool = False
) -> int:
    """Sample a number n matching the predicted property.

    For predicted="yes": construct n to have the property.
    For predicted="no": sample n and adjust if needed to lack the property.
    For near-miss construction, pick n just off a boundary (e.g., one less than
    a perfect square, or one more than a prime).
    """
    n_lo, n_hi = n_range

    if predicted == "yes":
        if prop == "prime":
            # Find all primes in range via trial division
            candidates = []
            for candidate in range(n_lo, n_hi + 1):
                if _is_prime_trial_division(candidate):
                    candidates.append(candidate)
            if not candidates:
                raise ValueError(f"No primes in range [{n_lo}, {n_hi}]")
            return rng.choice(candidates)
        elif prop == "composite":
            # Sample a composite number
            candidates = [n for n in range(n_lo, n_hi + 1) if _is_composite(n)]
            if not candidates:
                raise ValueError(f"No composites in range [{n_lo}, {n_hi}]")
            return rng.choice(candidates)
        elif prop == "perfect square":
            # Construct a perfect square
            sqrt_lo = max(2, int(math.sqrt(n_lo)))  # Exclude 1
            sqrt_hi = int(math.sqrt(n_hi))
            if sqrt_lo > sqrt_hi:
                raise ValueError(f"No perfect squares in range [{n_lo}, {n_hi}]")
            sqrt_n = rng.randint(sqrt_lo, sqrt_hi)
            return sqrt_n * sqrt_n
    else:  # predicted == "no"
        if prop == "prime":
            # Sample a non-prime
            n = rng.randint(n_lo, n_hi)
            attempts = 0
            while _is_prime_trial_division(n) and attempts < 500:
                n = rng.randint(n_lo, n_hi)
                attempts += 1
            if attempts == 500:
                raise ValueError(f"Cannot find non-prime in range after 500 attempts")

            # For near-miss, make it just off a prime
            if force_near_miss:
                # Find a prime and nudge ±1
                candidates = []
                for test_n in range(n_lo, n_hi + 1):
                    if _is_prime_trial_division(test_n):
                        if test_n + 1 <= n_hi:
                            candidates.append(test_n + 1)
                        if test_n - 1 >= n_lo:
                            candidates.append(test_n - 1)
                if candidates:
                    n = rng.choice(candidates)
            return n
        elif prop == "composite":
            # Sample a non-composite (prime or 1, but 1 is excluded)
            candidates = [n for n in range(max(2, n_lo), n_hi + 1) if _is_prime_trial_division(n)]
            if not candidates:
                raise ValueError(f"No non-composites in range")
            return rng.choice(candidates)
        elif prop == "perfect square":
            # Sample a non-perfect-square
            n = rng.randint(n_lo, n_hi)
            attempts = 0
            while _is_perfect_square(n) and attempts < 500:
                n = rng.randint(n_lo, n_hi)
                attempts += 1
            if attempts == 500:
                raise ValueError(f"Cannot find non-perfect-square after 500 attempts")

            # For near-miss, make it just off a perfect square
            if force_near_miss:
                sqrt_n = math.isqrt(n)
                candidates = []
                for sq in [sqrt_n * sqrt_n, (sqrt_n + 1) * (sqrt_n + 1)]:
                    if sq + 1 <= n_hi:
                        candidates.append(sq + 1)
                    if sq - 1 >= n_lo:
                        candidates.append(sq - 1)
                if candidates:
                    n = rng.choice(candidates)
            return n

    raise ValueError("Unreachable")


@generator(
    "prime_composite_square_properties",
    level=6,
    presets={
        "easy": dict(n_range=(2, 30), prop="prime", force_near_miss=False),
        "medium": dict(n_range=(2, 100), prop=None, force_near_miss=False),  # randomised
        "hard": dict(n_range=(50, 300), prop=None, force_near_miss=None),  # randomised, 1 in 3 near-miss
    },
)
def prime_composite_square_properties(
    rng, n_range=(2, 30), prop=None, force_near_miss=False, form="number", choose=False
):
    """Given a natural number and a property (prime, composite, or perfect square),
    determine whether the number has that property. Two-option classification: yes/no.

    Note: 1 is excluded from the sampled pool (neither prime nor composite by
    standard convention, and also a perfect square -- an edge case this skill
    was not designed to adjudicate).

    ``form="factored"`` (hard tier): the number is shown in prime-factorised form
    and the learner decides whether it is a perfect square from the parity of
    the exponents, with no multiplying out.

    ``choose=True`` (what the skill file sets) asks which of four numbers is prime, composite or a perfect
    square (hard: which of four factorised numbers is a perfect square); see ``w2_pick_the_one``.
    """
    if choose:
        from .w2_pick_the_one import property_pick

        return property_pick(rng, n_range[0], n_range[1], prop, form)
    if form == "factored":
        predicted = rng.choice(["yes", "no"])
        primes = sorted(rng.sample([2, 3, 5, 7, 11], rng.choice([2, 3])))
        fac = {p: rng.choice([2, 4]) for p in primes}
        if predicted == "no":
            for p in rng.sample(primes, rng.choice([1, 2])):
                fac[p] = rng.choice([1, 3])
        n = _factored_value(fac)
        _verify_property(n, "perfect square", predicted)
        text = f"Is {_factored_text(fac)} a perfect square?"
        if predicted == "yes":
            working = ["Every exponent is even, so the number is a perfect square."]
        else:
            bad = ", ".join(str(p) for p in primes if fac[p] % 2)
            working = [f"The exponent of {bad} is odd, so the number is not a perfect square."]
        return dict(
            text=text,
            latex=f"plain:{text}",
            answer=predicted,
            answer_text=predicted,
            answer_latex=latex_answer(predicted, plain=True),
            working=working,
            distractors=pick_distractors(predicted, [_OTHER_LABEL[predicted]], n=1),
        )
    # Choose predicted answer first
    predicted = rng.choice(["yes", "no"])

    # For medium/hard, randomise the property
    if prop is None:
        prop = rng.choice(["prime", "composite", "perfect square"])

    # For hard difficulty, force near-miss on 1/3 of seeds
    if force_near_miss is None:
        force_near_miss = rng.random() < 1 / 3

    # Exclude 1 from the range
    n_lo, n_hi = n_range
    if n_lo == 1:
        n_lo = 2

    n = _sample_property_pair(rng, predicted, prop, (n_lo, n_hi), force_near_miss=force_near_miss)
    _verify_property(n, prop, predicted)

    # Format question based on property
    if prop == "perfect square":
        question_text = f"Is {n} a perfect square?"
    else:
        question_text = f"Is {n} a {prop} number?"

    # Working explanation
    if predicted == "yes":
        if prop == "prime":
            working = [f"{n} can only be divided by 1 and itself, so it is prime."]
        elif prop == "composite":
            working = [f"{n} has factors other than 1 and itself, so it is composite."]
        elif prop == "perfect square":
            sqrt_n = math.isqrt(n)
            working = [f"{sqrt_n} × {sqrt_n} = {n}, so it is a perfect square."]
    else:
        if prop == "prime":
            # Find a factor
            for i in range(2, int(math.sqrt(n)) + 1):
                if n % i == 0:
                    working = [f"{n} is divisible by {i}, so it is not prime."]
                    break
        elif prop == "composite":
            working = [f"{n} can only be divided by 1 and itself, so it is not composite (it is prime)."]
        elif prop == "perfect square":
            sqrt_n = math.isqrt(n)
            working = [
                f"The closest perfect squares are {sqrt_n * sqrt_n} and {(sqrt_n + 1) * (sqrt_n + 1)}, so {n} is not a perfect square."
            ]

    other_label = _OTHER_LABEL[predicted]

    return dict(
        text=question_text,
        latex=f"plain:{question_text}",
        answer=predicted,
        answer_text=predicted,
        answer_latex=latex_answer(predicted, plain=True),
        working=working,
        distractors=pick_distractors(predicted, [other_label], n=1),
    )
