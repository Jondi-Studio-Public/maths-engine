"""Generates `w7b_number_systems_primes` questions (easy, medium, hard) and the exam chain
`exd_x_w7b_number_systems_primes`.

``number-bases-primes-and-divisors`` (``number-theory``, answer ``number``): numbers written in base 2, 8 and 16, binary
addition, primality testing by trial division up to the square root, and what a prime factorisation says about divisors.
Every answer is a whole number: either a decimal value, a count, or a numeral that uses only the digits 0 and 1 (or 0 to
7) and is read as a number, so no answer is a letter and none begins with a zero.

* easy: a binary numeral to decimal, a decimal number to binary, an octal or hexadecimal numeral to decimal.
* medium: a binary numeral to octal (grouping in threes), a hexadecimal numeral to binary, the sum of two binary numerals,
  the smallest prime factor of a composite number, or how many primes (or which largest prime) trial division needs.
* hard: the number of divisors, the number of square divisors or the sum of the divisors from a number or its prime
  factorisation, a greatest common divisor or lowest common multiple from two numbers, the number of trial divisions
  that expose a composite number whose smallest factor is large, and a hexadecimal numeral that must be converted first.
* exam (``exd_x_w7b_number_systems_primes``): a hexadecimal numeral: its decimal value, how many primes trial division
  needs, its number of divisors and its greatest common divisor with a second number.

Every key is built from a chosen value or exponent vector and re-derived a second way: numerals by converting back with
``int(numeral, base)``, divisor counts and sums by listing every divisor, primality by SymPy, gcd and lcm by ``math``.
"""

from __future__ import annotations

import math
import sympy

from ..core import attempts, check, generator, phrase
from ._w6a_kit import enough, num_dist, num_fields, part, retry_draw

TOPIC = "w7b_number_systems_primes"
EXAM_TOPIC = "exd_x_w7b_number_systems_primes"

AREA = "number-bases-primes"
MISCONCEPTIONS = [
    (
        "nsp-reads-bits-reversed",
        "Reads the digits of the numeral in the wrong order, so the place values run backwards.",
        AREA,
    ),
    ("nsp-counts-ones", "Counts the 1s in the binary numeral instead of adding the place values they stand for.", AREA),
    (
        "nsp-reads-as-decimal",
        "Reads the digits of a binary, octal or hexadecimal numeral as an ordinary decimal number.",
        AREA,
    ),
    (
        "nsp-powers-start-at-one",
        "Starts the place values at 2 (or 8, or 16) instead of 1, so every value is a base too large.",
        AREA,
    ),
    ("nsp-hex-letter-off-by-one", "Gives a hexadecimal letter a value one too large (A as 11 up to F as 16).", AREA),
    (
        "nsp-remainders-read-top-down",
        "Writes the remainders of repeated division in the order they were found instead of from the last to the first.",
        AREA,
    ),
    ("nsp-stops-division-early", "Stops the repeated division one step early and leaves off the last digit.", AREA),
    (
        "nsp-appends-a-digit",
        "Divides once too often or multiplies in an extra place, giving a numeral one digit too long.",
        AREA,
    ),
    (
        "nsp-no-conversion",
        "Writes the decimal digits of the number as if they were already the numeral in the new base.",
        AREA,
    ),
    (
        "nsp-groups-from-the-left",
        "Groups the binary digits in threes from the left instead of from the right, so the groups do not line up with the place values.",
        AREA,
    ),
    ("nsp-nibbles-swapped", "Writes the four-digit groups for the hexadecimal digits in the wrong order.", AREA),
    (
        "nsp-digit-values-not-nibbles",
        "Writes each hexadecimal digit's value in binary without padding it to four digits.",
        AREA,
    ),
    ("nsp-binary-no-carry", "Adds binary digits without carrying, treating 1 + 1 as 0 in every column.", AREA),
    ("nsp-adds-as-decimal", "Adds the two binary numerals as if they were decimal numbers.", AREA),
    ("nsp-drops-final-carry", "Loses the carry out of the leftmost column, so the sum has too few digits.", AREA),
    ("nsp-gives-cofactor", "Gives the larger factor found when the smallest factor was asked.", AREA),
    (
        "nsp-gives-square-root",
        "Gives the whole-number part of the square root of the number instead of its smallest prime factor.",
        AREA,
    ),
    ("nsp-overshoots-first-factor", "Gives the next prime after the smallest prime factor.", AREA),
    ("nsp-tests-every-integer", "Counts every whole number up to the square root instead of only the primes.", AREA),
    ("nsp-counts-one-as-prime", "Counts 1 as a prime when listing the primes up to the square root.", AREA),
    ("nsp-omits-two", "Leaves 2 out of the primes to be tried.", AREA),
    ("nsp-uses-root-itself", "Gives the whole-number part of the square root itself, although it is not prime.", AREA),
    ("nsp-tests-up-to-half", "Tests every prime up to half the number instead of stopping at its square root.", AREA),
    (
        "nsp-rounds-root-up",
        "Rounds the square root up to the next whole number before listing the primes, so one prime too many is tested.",
        AREA,
    ),
    (
        "nsp-adds-exponents",
        "Adds the exponents in the prime factorisation instead of adding one to each and multiplying.",
        AREA,
    ),
    ("nsp-multiplies-exponents", "Multiplies the exponents without adding one to each first.", AREA),
    (
        "nsp-counts-prime-factors",
        "Gives the number of prime factors, counted with repeats, instead of the number of divisors.",
        AREA,
    ),
    ("nsp-counts-distinct-primes-only", "Counts only the distinct prime factors as divisors.", AREA),
    ("nsp-sum-one-and-n-only", "Adds only 1 and the number itself when summing the divisors.", AREA),
    ("nsp-gives-count-for-sum", "Gives the number of divisors where their sum was asked.", AREA),
    (
        "nsp-sum-adds-one-to-powers",
        "Multiplies (1 + p^a) for each prime instead of summing the whole geometric series of powers.",
        AREA,
    ),
    (
        "nsp-gives-count-for-square",
        "Gives the number of all divisors where the number of square divisors was asked.",
        AREA,
    ),
    (
        "nsp-rounds-halves-up",
        "Rounds each half-exponent up instead of down when counting the even exponents a square divisor may use.",
        AREA,
    ),
    (
        "nsp-gcd-lcm-swapped",
        "Gives the lowest common multiple where the greatest common divisor was asked, or the reverse.",
        AREA,
    ),
    ("nsp-gcd-is-difference", "Gives the difference of the two numbers as their greatest common divisor.", AREA),
    ("nsp-lcm-is-product", "Gives the product of the two numbers as their lowest common multiple.", AREA),
    (
        "nsp-gcd-ignores-exponents",
        "Multiplies the common primes each to the first power, ignoring the smaller exponent in the numbers.",
        AREA,
    ),
    (
        "nsp-lcm-drops-unshared",
        "Multiplies only the primes the two numbers share when finding the lowest common multiple.",
        AREA,
    ),
    (
        "nsp-counts-up-to-factor-only",
        "Counts the primes up to the smallest factor but forgets to count the factor itself.",
        AREA,
    ),
    (
        "nsp-counts-up-to-root",
        "Counts the primes tried up to the square root, where the primes only up to the smallest factor were needed.",
        AREA,
    ),
]

_DIGITS = "0123456789ABCDEF"


# -- numerals ---------------------------------------------------------------------------------------------------------


def _numeral(n: int, base: int) -> str:
    out = ""
    for _ in range(n.bit_length()):  # a base-b digit takes at least one bit, so this many steps always suffice
        if n == 0:
            break
        out = _DIGITS[n % base] + out
        n //= base
    return out or "0"


def _value(numeral: str, base: int) -> int:
    return int(numeral, base)


def _as_number(numeral: str):
    """The numeral read as a whole number (digits 0-9 only), or None when it has a letter or a leading zero."""
    return int(numeral) if numeral.isdigit() and not numeral.startswith("0") else None


def _numeral_candidates(candidates):
    """[(numeral string, id)] -> [(whole number, id)]: a numeral with a letter or a leading zero is dropped."""
    return [(_as_number(s), tag) for s, tag in candidates if _as_number(s) is not None]


def _answer(value: int) -> dict:
    return num_fields(value)


_BASE_WORD = {2: "binary", 8: "octal", 16: "hexadecimal"}


def _random_bits(rng, length: int, min_zeros: int = 1, min_ones: int = 2) -> str:
    for _ in attempts(500, "a binary numeral with a mixture of digits"):
        bits = "1" + "".join(rng.choice("01") for _ in range(length - 1))
        if bits.count("0") >= min_zeros and bits.count("1") >= min_ones and not bits.endswith("0" * 3):
            return bits
    raise AssertionError  # pragma: no cover


# -- easy -------------------------------------------------------------------------------------------------------------


def _easy_bin_to_dec(rng):
    bits = _random_bits(rng, rng.randint(6, 9))
    value = _value(bits, 2)
    check(sum(int(b) * 2**i for i, b in enumerate(reversed(bits))) == value, "binary value differs")
    cands = [
        (int(bits[::-1], 2), "nsp-reads-bits-reversed"),
        (int(bits), "nsp-reads-as-decimal"),
        (2 * value, "nsp-powers-start-at-one"),
        (bits.count("1"), "nsp-counts-ones"),
    ]
    dist = num_dist(value, cands)
    enough(dist)
    stem = phrase(
        rng,
        (
            "What is the value, written in decimal, of the binary number {b}?",
            "Convert the binary number {b} to decimal.",
            "The binary numeral {b} stands for which whole number in ordinary decimal notation?",
        ),
        b=bits,
    )
    places = " + ".join(f"{2 ** (len(bits) - 1 - i)}" for i, d in enumerate(bits) if d == "1")
    working = [f"Add the place values of the 1s: {places}.", f"The total is {value}."]
    return dict(text=stem, working=working, distractors=dist, **_answer(value), answer_kind="number")


def _easy_dec_to_bin(rng):
    n = rng.randint(37, 250)
    bits = _numeral(n, 2)
    check(int(bits, 2) == n, "binary numeral differs")
    shifted = bits + "0"
    cands = _numeral_candidates(
        [
            (bits[::-1], "nsp-remainders-read-top-down"),
            (bits[:-1], "nsp-stops-division-early"),
            (shifted, "nsp-appends-a-digit"),
            (str(n), "nsp-no-conversion"),
        ]
    )
    dist = num_dist(int(bits), cands)
    enough(dist)
    stem = phrase(
        rng,
        (
            "Write the decimal number {n} in binary. Give the answer using only the digits 0 and 1.",
            "Convert {n} to binary, giving your answer as a string of 0s and 1s.",
            "What is {n} in base 2? Write the numeral with the digits 0 and 1 only.",
        ),
        n=n,
    )
    steps = [f"{n >> i} = 2 x {n >> (i + 1)} remainder {(n >> i) & 1}" for i in range(n.bit_length())]
    working = [
        "Divide by 2 repeatedly: " + "; ".join(steps) + ".",
        f"Read the remainders from the last to the first: {bits}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_answer(int(bits)), answer_kind="number")


def _easy_other_to_dec(rng):
    base = rng.choice((8, 16))
    if base == 8:
        n = rng.randint(70, 500)
    else:
        n = rng.randint(40, 4000)
    numeral = _numeral(n, base)
    if base == 16 and not any(c.isalpha() for c in numeral):
        n = n + 10 if (n % 16) < 6 else n
        numeral = _numeral(n, base)
    check(_value(numeral, base) == n, "numeral value differs")
    check(sum(_DIGITS.index(c) * base**i for i, c in enumerate(reversed(numeral))) == n, "place values differ")
    cands = [
        (_value(numeral[::-1], base), "nsp-reads-bits-reversed"),
        (base * n, "nsp-powers-start-at-one"),
    ]
    if numeral.isdigit():
        cands.append((int(numeral), "nsp-reads-as-decimal"))
    if base == 16:
        cands.append(
            (
                sum((_DIGITS.index(c) + (1 if c.isalpha() else 0)) * 16**i for i, c in enumerate(reversed(numeral))),
                "nsp-hex-letter-off-by-one",
            )
        )
    dist = num_dist(n, cands)
    enough(dist)
    stem = phrase(
        rng,
        (
            "What is the value, in decimal, of the {w} number {x}?",
            "Convert the {w} number {x} to decimal.",
        ),
        w=_BASE_WORD[base],
        x=numeral,
    )
    terms = " + ".join(f"{_DIGITS.index(c)} x {base}^{len(numeral) - 1 - i}" for i, c in enumerate(numeral))
    working = [f"Multiply each digit by its place value: {terms}.", f"The total is {n}."]
    return dict(text=stem, working=working, distractors=dist, **_answer(n), answer_kind="number")


# -- medium -----------------------------------------------------------------------------------------------------------


def _medium_bin_to_octal(rng):
    length = rng.choice((7, 8, 10, 11))
    bits = _random_bits(rng, length)
    octal = _numeral(int(bits, 2), 8)
    check(int(octal, 8) == int(bits, 2), "octal numeral differs")
    pad = (-len(bits)) % 3
    check(pad != 0, "the binary numeral already groups into threes")
    left = [bits[i : i + 3] for i in range(0, len(bits), 3)]
    left_digits = "".join(str(int(g, 2)) for g in left)
    cands = _numeral_candidates(
        [
            (left_digits, "nsp-groups-from-the-left"),
            (octal[::-1], "nsp-reads-bits-reversed"),
            (str(int(bits, 2)), "nsp-no-conversion"),
            (octal[:-1] if len(octal) > 2 else "", "nsp-stops-division-early"),
        ]
    )
    dist = num_dist(int(octal), cands)
    enough(dist)
    stem = phrase(
        rng,
        (
            "Convert the binary number {b} to octal. Give the octal digits as your answer.",
            "Write the binary number {b} in base 8.",
        ),
        b=bits,
    )
    groups = [("0" * pad + bits)[i : i + 3] for i in range(0, len(bits) + pad, 3)]
    working = [
        f"Group the binary digits in threes from the right: {' '.join(groups)}.",
        f"Each group is one octal digit: {octal}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_answer(int(octal)), answer_kind="number")


def _medium_hex_to_bin(rng):
    n = rng.randint(0x40, 0xFFF)
    h = _numeral(n, 16)
    check(len(h) >= 2, "short hex numeral")
    if not any(c.isalpha() for c in h):
        h = h[:-1] + rng.choice("ABCDEF")
    n = _value(h, 16)
    bits = _numeral(n, 2)
    check(int(bits, 2) == n, "binary numeral differs")
    check("".join(f"{_DIGITS.index(c):04b}" for c in h).lstrip("0") == bits, "nibble expansion differs")
    swapped = "".join(f"{_DIGITS.index(c):04b}" for c in reversed(h)).lstrip("0")
    unpadded = "".join(f"{_DIGITS.index(c):b}" for c in h)
    cands = _numeral_candidates(
        [
            (swapped, "nsp-nibbles-swapped"),
            (unpadded, "nsp-digit-values-not-nibbles"),
            (bits[::-1], "nsp-reads-bits-reversed"),
            (_numeral(int(h.translate(str.maketrans("ABCDEF", "123456")), 16), 2), "nsp-hex-letter-off-by-one"),
        ]
    )
    dist = num_dist(int(bits), cands)
    enough(dist)
    stem = phrase(
        rng,
        (
            "Convert the hexadecimal number {h} to binary. Give the answer as a string of 0s and 1s with no leading zeros.",
            "Write the hexadecimal number {h} in binary, using only the digits 0 and 1 and no leading zeros.",
        ),
        h=h,
    )
    working = [
        "Write each hexadecimal digit as four binary digits: " + " ".join(f"{_DIGITS.index(c):04b}" for c in h) + ".",
        f"Join them and drop the leading zeros: {bits}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_answer(int(bits)), answer_kind="number")


def _medium_binary_sum(rng):
    for _ in attempts(500, "two binary numerals that need carries"):
        la, lb = rng.randint(5, 8), rng.randint(5, 8)
        a, b = _random_bits(rng, la), _random_bits(rng, lb)
        total = int(a, 2) + int(b, 2)
        if (
            bin(int(a, 2) ^ int(b, 2)) != bin(total)
            and (int(a, 2) & int(b, 2)).bit_count() >= 2
            and total.bit_length() > max(la, lb)
        ):
            break
    sum_bits = _numeral(total, 2)
    check(int(sum_bits, 2) == int(a, 2) + int(b, 2), "binary sum differs")
    longest = max(la, lb)
    cands = _numeral_candidates(
        [
            (_numeral(int(a, 2) ^ int(b, 2), 2), "nsp-binary-no-carry"),
            (str(int(a) + int(b)), "nsp-adds-as-decimal"),
            (_numeral(total % (1 << longest), 2), "nsp-drops-final-carry"),
            (_numeral(total, 2)[::-1], "nsp-reads-bits-reversed"),
        ]
    )
    dist = num_dist(int(sum_bits), cands)
    enough(dist)
    stem = phrase(
        rng,
        (
            "Add the binary numbers {a} and {b}. Give the sum in binary, using only the digits 0 and 1.",
            "Find {a} + {b} in binary arithmetic. Write the answer as a binary numeral.",
        ),
        a=a,
        b=b,
    )
    working = [
        f"In decimal the numbers are {int(a, 2)} and {int(b, 2)}, so a check value for the sum is {total}.",
        f"Adding column by column with carries gives {sum_bits}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_answer(int(sum_bits)), answer_kind="number")


def _next_prime(p: int) -> int:
    return int(sympy.nextprime(p))


def _prev_prime(p: int) -> int:
    return int(sympy.prevprime(p))


def _smallest_factor(n: int) -> int:
    for d in range(2, math.isqrt(n) + 1):  # trial division to the square root
        if n % d == 0:
            return d
    return n


def _medium_smallest_factor(rng):
    p = rng.choice(list(sympy.primerange(7, 48)))
    q = rng.choice(list(sympy.primerange(p, 90)))
    n = p * q
    check(_smallest_factor(n) == p, "smallest factor differs")
    check(min(sympy.factorint(n)) == p, "smallest prime factor differs (SymPy)")
    root = math.isqrt(n)
    cands = [
        (q, "nsp-gives-cofactor") if q != p else None,
        (root, "nsp-gives-square-root"),
        (_next_prime(p), "nsp-overshoots-first-factor"),
    ]
    dist = num_dist(p, [c for c in cands if c])
    enough(dist)
    stem = phrase(
        rng,
        (
            "{n} is not prime and it has no factor of 2, 3 or 5. Using trial division by primes, what is its smallest prime factor?",
            "By trial division with the primes in order, find the smallest prime factor of {n}. ({n} is odd and not divisible by 3 or 5.)",
        ),
        n=n,
    )
    primes = list(sympy.primerange(2, p + 1))
    working = [
        f"Try the primes in order: {', '.join(map(str, primes))}.",
        f"{p} is the first that divides {n} ({n} = {p} x {q}).",
    ]
    return dict(text=stem, working=working, distractors=dist, **_answer(p), answer_kind="number")


def _medium_primes_needed(rng):
    n = rng.randint(120, 2500)
    root = math.isqrt(n)
    if root * root == n:
        n += 1
        root = math.isqrt(n)
    primes = list(sympy.primerange(2, root + 1))
    count = len(primes)
    check(
        count == sum(1 for k in range(2, root + 1) if all(k % d for d in range(2, math.isqrt(k) + 1))),
        "prime count differs",
    )
    ask_largest = rng.random() < 0.4
    if ask_largest:
        value = primes[-1]
        nxt = _next_prime(root)
        cands = [
            (root if root != value else None, "nsp-uses-root-itself"),
            (nxt, "nsp-rounds-root-up"),
            (_prev_prime(n // 2 + 1) if n // 2 > 3 else None, "nsp-tests-up-to-half"),
        ]
        ask = "what is the largest prime that must be tried"
        working = [
            f"The square root of {n} is between {root} and {root + 1}.",
            f"The largest prime up to {root} is {value}.",
        ]
    else:
        value = count
        cands = [
            (root - 1, "nsp-tests-every-integer"),
            (count + 1, "nsp-counts-one-as-prime"),
            (count - 1 if count > 3 else None, "nsp-omits-two"),
            (
                len(list(sympy.primerange(2, root + 3)))
                if len(list(sympy.primerange(2, root + 3))) != count
                else count + 2,
                "nsp-rounds-root-up",
            ),
            (len(list(sympy.primerange(2, n // 2 + 1))), "nsp-tests-up-to-half"),
        ]
        ask = "how many different primes must be tried"
        working = [
            f"The square root of {n} is between {root} and {root + 1}, so test the primes up to {root}.",
            f"These are {', '.join(map(str, primes))}: {count} primes.",
        ]
    dist = num_dist(value, [c for c in cands if c and c[0] is not None and c[0] > 0])
    enough(dist)
    stem = (
        f"To decide whether {n} is prime by trial division, you divide it by primes up to its square root. For {n}, {ask} "
        "(when the number turns out to be prime)?"
    )
    return dict(text=stem, working=working, distractors=dist, **_answer(value), answer_kind="number")


# -- hard -------------------------------------------------------------------------------------------------------------

_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _factor_text(primes, exps) -> str:
    return " × ".join(f"{p}{str(e).translate(_SUP)}" if e > 1 else f"{p}" for p, e in zip(primes, exps))


def _exponents(rng, k: int, lo=1, hi=4):
    return [rng.randint(lo, hi) for _ in range(k)]


def _divisors_of(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _hard_divisors(rng):
    for _ in attempts(500, "a number with a modest size"):
        k = rng.choice((2, 3))
        primes = sorted(rng.sample([2, 3, 5, 7, 11], k))
        exps = _exponents(rng, k, 1, 4)
        n = math.prod(p**e for p, e in zip(primes, exps))
        if 60 <= n <= 6000:
            break
    kind = rng.choice(("count", "sigma", "sigma", "squares"))
    given_as_number = rng.random() < 0.5
    divs = _divisors_of(n)
    check(len(divs) == math.prod(e + 1 for e in exps) == sympy.divisor_count(n), "divisor count differs")
    if kind == "count":
        value = len(divs)
        cands = [
            (sum(exps), "nsp-counts-prime-factors"),
            (sum(e + 1 for e in exps), "nsp-adds-exponents"),
            (math.prod(exps), "nsp-multiplies-exponents"),
            (k, "nsp-counts-distinct-primes-only"),
        ]
        ask = "How many positive divisors does it have?"
        working = [
            f"Add 1 to each exponent: {', '.join(str(e + 1) for e in exps)}.",
            f"Multiply: {' x '.join(str(e + 1) for e in exps)} = {value}.",
        ]
    elif kind == "squares":
        value = sum(1 for d in divs if math.isqrt(d) ** 2 == d)
        check(value == math.prod(e // 2 + 1 for e in exps), "square divisor count differs")
        cands = [
            (len(divs), "nsp-gives-count-for-square"),
            (math.prod(e // 2 for e in exps) if all(e >= 2 for e in exps) else None, "nsp-multiplies-exponents"),
            (sum(e // 2 for e in exps) + 1, "nsp-adds-exponents"),
            (math.prod((e + 1) // 2 + 1 for e in exps), "nsp-rounds-halves-up"),
        ]
        ask = "How many of its positive divisors are perfect squares?"
        working = [
            f"A square divisor uses an even exponent of each prime, from 0 up to the exponent in {n}.",
            f"Choices per prime: {', '.join(str(e // 2 + 1) for e in exps)}; the product is {value}.",
        ]
    else:
        value = sum(divs)
        check(value == sympy.divisor_sigma(n), "divisor sum differs")
        cands = [
            (n + 1, "nsp-sum-one-and-n-only"),
            (len(divs), "nsp-gives-count-for-sum"),
            (math.prod(1 + p**e for p, e in zip(primes, exps)), "nsp-sum-adds-one-to-powers"),
            (sum(primes) + 1 + n, "nsp-sum-one-and-n-only"),
        ]
        ask = "What is the sum of all its positive divisors?"
        working = [
            "For each prime power add the powers from p^0 up to p^e: "
            + ", ".join("(" + " + ".join(str(p**i) for i in range(e + 1)) + ")" for p, e in zip(primes, exps))
            + ".",
            f"Multiply the sums: {value}.",
        ]
    dist = num_dist(value, [c for c in cands if c and c[0] is not None])
    enough(dist)
    if given_as_number:
        intro = f"Let N = {n}."
        working.insert(0, f"Factorise: {n} = {_factor_text(primes, exps)}.")
    else:
        intro = f"Let N = {_factor_text(primes, exps)}."
    return dict(text=f"{intro} {ask}", working=working, distractors=dist, **_answer(value), answer_kind="number")


def _hard_gcd_lcm(rng):
    for _ in attempts(500, "two numbers with several shared primes"):
        primes = [2, 3, 5, 7]
        e1 = [rng.randint(0, 4) for _ in primes]
        e2 = [rng.randint(0, 4) for _ in primes]
        a = math.prod(p**e for p, e in zip(primes, e1))
        b = math.prod(p**e for p, e in zip(primes, e2))
        shared = [i for i in range(4) if e1[i] and e2[i]]
        if 100 <= a <= 5000 and 100 <= b <= 5000 and a != b and len(shared) >= 2 and math.gcd(a, b) not in (1, a, b):
            break
    g, l = math.gcd(a, b), math.lcm(a, b)
    check(g * l == a * b, "gcd x lcm differs from the product")
    check(g == math.prod(p ** min(x, y) for p, x, y in zip(primes, e1, e2)), "gcd from exponents differs")
    check(l == math.prod(p ** max(x, y) for p, x, y in zip(primes, e1, e2)), "lcm from exponents differs")
    ask_gcd = rng.random() < 0.5
    if ask_gcd:
        value = g
        cands = [
            (l, "nsp-gcd-lcm-swapped"),
            (math.prod(p for p, x, y in zip(primes, e1, e2) if x and y), "nsp-gcd-ignores-exponents"),
            (math.prod(p ** max(x, y) for p, x, y in zip(primes, e1, e2) if x and y), "nsp-lcm-drops-unshared"),
            (abs(a - b), "nsp-gcd-is-difference"),
        ]
        word, abbreviation = "greatest common divisor", "gcd"
    else:
        value = l
        cands = [
            (g, "nsp-gcd-lcm-swapped"),
            (a * b, "nsp-lcm-is-product"),
            (math.prod(p ** max(x, y) for p, x, y in zip(primes, e1, e2) if x and y), "nsp-lcm-drops-unshared"),
        ]
        word, abbreviation = "lowest common multiple", "lcm"
    dist = num_dist(value, cands)
    enough(dist)
    stem = phrase(
        rng,
        (
            "Find the {w} of {a} and {b}.",
            "Using prime factorisations, work out the {w} of {a} and {b}.",
        ),
        w=word,
        a=a,
        b=b,
    )
    working = [
        f"{a} = {_factor_text([p for p, e in zip(primes, e1) if e], [e for e in e1 if e])} and {b} = {_factor_text([p for p, e in zip(primes, e2) if e], [e for e in e2 if e])}.",
        f"Take the {'smaller' if ask_gcd else 'larger'} exponent of each prime: {abbreviation} = {value}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_answer(value), answer_kind="number")


def _hard_trial_count(rng):
    ps = list(sympy.primerange(31, 100))
    p = rng.choice(ps[:-1])
    q = rng.choice([x for x in ps if x > p])
    n = p * q
    check(_smallest_factor(n) == p, "smallest factor differs")
    tried = list(sympy.primerange(2, p + 1))
    value = len(tried)
    to_root = len(list(sympy.primerange(2, math.isqrt(n) + 1)))
    cands = [
        (value - 1, "nsp-counts-up-to-factor-only"),
        (value + 1, "nsp-counts-one-as-prime"),
        (to_root if to_root != value else None, "nsp-counts-up-to-root"),
        (p - 1, "nsp-tests-every-integer"),
    ]
    dist = num_dist(value, [c for c in cands if c[0] is not None])
    enough(dist)
    stem = (
        f"{n} is the product of two primes. A student tests {n} for factors by dividing by each prime in order (2, 3, 5, 7, ...), "
        "and stops at the first prime that divides it exactly. How many primes does the student divide by, counting the one that works?"
    )
    working = [
        f"The smallest prime factor of {n} is {p} ({n} = {p} x {q}).",
        f"The primes up to {p} are {', '.join(map(str, tried))}: {value} divisions.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_answer(value), answer_kind="number")


def _hard_hex_divisors(rng):
    for _ in attempts(500, "a hexadecimal numeral that is a composite with letters"):
        k = rng.choice((2, 3))
        primes = sorted(rng.sample([2, 3, 5, 7, 11, 13], k))
        exps = _exponents(rng, k, 1, 3)
        n = math.prod(p**e for p, e in zip(primes, exps))
        h = _numeral(n, 16)
        if 256 <= n <= 4095 and any(c.isalpha() for c in h) and h.isalnum():
            break
    count = len(_divisors_of(n))
    check(count == sympy.divisor_count(n) == math.prod(e + 1 for e in exps), "divisor count differs")
    cands = [
        (sum(exps), "nsp-counts-prime-factors"),
        (sum(e + 1 for e in exps), "nsp-adds-exponents"),
        (math.prod(exps), "nsp-multiplies-exponents"),
        (k, "nsp-counts-distinct-primes-only"),
    ]
    dist = num_dist(count, cands)
    enough(dist)
    stem = (
        f"The hexadecimal number {h} is converted to decimal. How many positive divisors does the decimal number have?"
    )
    working = [
        f"{h} in hexadecimal is {n} in decimal.",
        f"{n} = {_factor_text(primes, exps)}, so the number of divisors is {' x '.join(str(e + 1) for e in exps)} = {count}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **_answer(count), answer_kind="number")


_EASY = (_easy_bin_to_dec, _easy_dec_to_bin, _easy_other_to_dec)
_MEDIUM = (_medium_bin_to_octal, _medium_hex_to_bin, _medium_binary_sum, _medium_smallest_factor, _medium_primes_needed)
_HARD = (_hard_divisors, _hard_divisors, _hard_gcd_lcm, _hard_gcd_lcm, _hard_trial_count, _hard_hex_divisors)


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def w7b_number_systems_primes(rng, tier="easy"):
    """Number bases, binary arithmetic, primality and divisors; the tiers are described in the module docstring."""
    pool = {"easy": _EASY, "medium": _MEDIUM, "hard": _HARD}.get(tier)
    if not (pool is not None):
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


# -- exam -------------------------------------------------------------------------------------------------------------


@generator(EXAM_TOPIC, level=19, presets={"exam": dict(tier="exam")})
def exd_x_w7b_number_systems_primes(rng, tier="exam"):
    """Exam chain on a hexadecimal numeral: decimal value, primes to try, number of divisors, gcd with a second number."""
    if not (tier == "exam"):
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _exam_chain)


def _exam_chain(rng):
    for _ in attempts(20000, "a hexadecimal numeral with a rich factorisation"):
        k = rng.choice((2, 3))
        primes = sorted(rng.sample([2, 3, 5, 7, 11, 13], k))
        exps = _exponents(rng, k, 1, 3)
        n = math.prod(p**e for p, e in zip(primes, exps))
        h = _numeral(n, 16)
        if not (300 <= n <= 4095 and any(c.isalpha() for c in h)):
            continue
        m_primes = [2, 3, 5, 7, 11, 13]
        m_exps = [rng.randint(0, 3) for _ in m_primes]
        m = math.prod(p**e for p, e in zip(m_primes, m_exps))
        g = math.gcd(n, m)
        if 100 <= m <= 3000 and g not in (1, m, n) and g >= 6:
            break
    else:
        raise AssertionError  # pragma: no cover
    root = math.isqrt(n)
    tries = len(list(sympy.primerange(2, root + 1)))
    count = len(_divisors_of(n))
    check(int(h, 16) == n, "hex value differs")
    check(count == math.prod(e + 1 for e in exps), "divisor count differs")
    check(
        math.gcd(n, m) == math.prod(p ** min(e, dict(zip(primes, exps)).get(p, 0)) for p, e in zip(m_primes, m_exps)),
        "gcd differs",
    )

    a_dist = num_dist(
        n,
        [
            (int(h[::-1], 16), "nsp-reads-bits-reversed"),
            (
                sum((_DIGITS.index(c) + (1 if c.isalpha() else 0)) * 16**i for i, c in enumerate(reversed(h))),
                "nsp-hex-letter-off-by-one",
            ),
            (n + 16, "nsp-powers-start-at-one"),
        ],
    )
    b_dist = num_dist(
        tries,
        [
            (root - 1, "nsp-tests-every-integer"),
            (tries + 1, "nsp-counts-one-as-prime"),
            (len(list(sympy.primerange(2, n // 2 + 1))), "nsp-tests-up-to-half"),
            (
                len(list(sympy.primerange(2, root + 3)))
                if len(list(sympy.primerange(2, root + 3))) != tries
                else tries + 2,
                "nsp-rounds-root-up",
            ),
        ],
    )
    c_dist = num_dist(
        count,
        [
            (sum(exps), "nsp-counts-prime-factors"),
            (sum(e + 1 for e in exps), "nsp-adds-exponents"),
            (math.prod(exps), "nsp-multiplies-exponents"),
            (k, "nsp-counts-distinct-primes-only"),
        ],
    )
    d_dist = num_dist(
        g,
        [
            (math.lcm(n, m), "nsp-gcd-lcm-swapped"),
            (math.prod(p for p in primes if m % p == 0), "nsp-gcd-ignores-exponents"),
            (
                math.prod(
                    p ** max(e, dict(zip(primes, exps))[p]) for p, e in zip(m_primes, m_exps) if p in primes and e
                ),
                "nsp-lcm-drops-unshared",
            ),
            (abs(n - m), "nsp-gcd-is-difference"),
        ],
    )
    for d in (a_dist, b_dist, c_dist, d_dist):
        enough(d)
    return dict(
        text=f"The hexadecimal number {h} is converted to decimal; call the decimal number N.",
        parts=[
            part(
                "Find N.", "number", num_fields(n), 1, [f"{h} in hexadecimal is {n}."], a_dist, sensible_range=(0, None)
            ),
            part(
                "To test N for primality by trial division, how many primes up to the square root of N would have to be tried?",
                "number",
                num_fields(tries),
                2,
                [
                    f"The square root of {n} is between {root} and {root + 1}.",
                    f"There are {tries} primes up to {root}.",
                ],
                b_dist,
                sensible_range=(0, None),
            ),
            part(
                "How many positive divisors does N have?",
                "number",
                num_fields(count),
                2,
                [
                    f"{n} = {_factor_text(primes, exps)}, so there are {' x '.join(str(e + 1) for e in exps)} = {count} divisors."
                ],
                c_dist,
                sensible_range=(0, None),
            ),
            part(
                f"Find the greatest common divisor of N and {m}.",
                "number",
                num_fields(g),
                2,
                [
                    f"{m} = {_factor_text([p for p, e in zip(m_primes, m_exps) if e], [e for e in m_exps if e])}; the common prime powers multiply to {g}."
                ],
                d_dist,
                sensible_range=(0, None),
            ),
        ],
    )
