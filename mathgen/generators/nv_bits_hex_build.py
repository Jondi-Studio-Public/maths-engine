"""Generates `nv_bits_hex_counts` questions (easy, medium, hard) and the exam chain `exd_x_nv_bits_hex_counts`.

``bits-hex-and-digit-counts`` (``number-theory``): how many bits a number needs, the largest value and the number of
different values a fixed number of bits can hold, and hexadecimal numerals typed as text. The conversions themselves
(binary to decimal, decimal to binary, hexadecimal to decimal or binary, binary addition) belong to
``number-bases-primes-and-divisors``, whose answers are all digit-only numbers; this skill adds what that one leaves
out: sizes and counts, and hexadecimal answers that contain letters (graded as a one-item ordered list, which compares a
text item case-insensitively and keeps a leading zero significant).

* easy: the bits needed for a number (not at a power-of-two boundary), the largest value n bits can hold, or how many
  different values n bits can hold.
* medium: a decimal number written in hexadecimal (typed), a binary numeral grouped into fours and written in
  hexadecimal (typed), the hexadecimal digits that n bits need, or how many different hexadecimal codes of n digits exist.
* hard: the bits needed at a boundary (256 needs 9 bits, 255 needs 8), the bits that give M different items each their own
  code, a number written in hexadecimal to a stated width (leading zeros kept) or as a six-digit colour code, and the
  largest n-bit number written in hexadecimal (3FF, not FFF), or the largest n-digit hexadecimal value as a number.
* exam (``exd_x_nv_bits_hex_counts``): a sensor reading up to N: (a) the bits needed, (b) how many values those bits can
  hold (from (a)), (c) how many of them go unused (from (a)), and (d) the hexadecimal digits a full reading needs (from
  (a)) or N written in hexadecimal (typed).

Every key is re-derived a second way: sizes by comparing powers of two, hexadecimal by ``int(text, 16)`` and ``format``.
"""

from __future__ import annotations

import math

from ..answers import Distractor
from ..core import attempts, check, generator, latex_answer, phrase
from ..grading import GRADERS
from ._exd_distinct_parts import distinct_part_answers
from ._nv_kit import spread_order
from ._spread_kit import _RNG, NEAR, uses_rng
from ._w6a_kit import TooFewDistractors, enough, num_dist, num_fields, part, retry_draw
from types import SimpleNamespace

TOPIC = "nv_bits_hex_counts"
EXAM_TOPIC = "exd_x_nv_bits_hex_counts"
AREA = "bits-hex-and-digit-counts"

MISCONCEPTIONS = [
    (
        "nv-bits-log-rounded-down",
        "Takes the bits needed for a number as the whole part of its base-2 logarithm, one too few.",
        AREA,
    ),
    ("nv-bits-counts-decimal-digits", "Gives the number of decimal digits of the number as the number of bits.", AREA),
    ("nv-bits-one-extra", "Adds a bit for the number zero or for a sign, giving one bit too many.", AREA),
    (
        "nv-bits-power-boundary",
        "Gives the exponent when the number is a power of two (256 as 8 bits), forgetting that 256 itself needs a ninth bit.",
        AREA,
    ),
    ("nv-bits-round-to-nearest", "Rounds the base-2 logarithm to the nearest whole number instead of up.", AREA),
    ("nv-bits-divides-by-eight", "Divides the number by 8, treating the number of bits as the number of bytes.", AREA),
    (
        "nv-largest-is-power",
        "Gives 2 to the power n as the largest value n bits can hold, which is the number of values (the largest is one less).",
        AREA,
    ),
    (
        "nv-largest-half",
        "Gives 2 to the power (n - 1) as the largest value n bits can hold, the value of the top bit alone.",
        AREA,
    ),
    (
        "nv-largest-linear",
        "Gives 2n - 1 as the largest value, doubling the number of bits instead of using powers of two.",
        AREA,
    ),
    ("nv-values-squares", "Gives n squared as the number of different values n bits can hold.", AREA),
    ("nv-values-doubles-bits", "Gives 2n as the number of different values n bits can hold.", AREA),
    (
        "nv-values-forgets-zero",
        "Gives 2 to the power n, less one, as the number of values, leaving out the value zero.",
        AREA,
    ),
    (
        "nv-hex-remainders-reversed",
        "Writes the remainders of repeated division by 16 in the order found, not from the last to the first.",
        AREA,
    ),
    (
        "nv-hex-letter-misvalue",
        "Gives a hexadecimal letter the wrong value (for example B as 12), so a digit is one out.",
        AREA,
    ),
    (
        "nv-hex-decimal-digits",
        "Writes the decimal digits of the number as if they were already hexadecimal digits.",
        AREA,
    ),
    (
        "nv-hex-groups-from-left",
        "Groups the binary digits in fours from the left, so the groups do not line up with the place values.",
        AREA,
    ),
    (
        "nv-hex-groups-of-three",
        "Groups the binary digits in threes (octal digits) when writing a hexadecimal numeral.",
        AREA,
    ),
    (
        "nv-hex-no-padding",
        "Leaves a short group of binary digits unpadded, reading it with the wrong place values.",
        AREA,
    ),
    (
        "nv-hex-digits-rounded-down",
        "Divides the number of bits by 4 and rounds down, one hexadecimal digit too few.",
        AREA,
    ),
    (
        "nv-hex-digits-bits-over-three",
        "Divides the number of bits by 3 (octal grouping) to count hexadecimal digits.",
        AREA,
    ),
    ("nv-hex-codes-multiply", "Multiplies the number of digits by 16 instead of using 16 to that power.", AREA),
    ("nv-hex-codes-base-ten", "Uses 10 to the power of the digit count, as if the digits were decimal.", AREA),
    ("nv-hex-padding-dropped", "Drops the leading zeros of a numeral that was to be written to a stated width.", AREA),
    ("nv-hex-padding-wrong-side", "Pads a short numeral with zeros on the right, changing its value.", AREA),
    (
        "nv-hex-top-digit-f",
        "Writes the largest n-bit number with every hexadecimal digit as F, ignoring that the top digit holds fewer bits.",
        AREA,
    ),
    ("nv-hex-top-digit-power", "Uses 2 to the power of the leftover bits as the top digit instead of one less.", AREA),
    (
        "nv-unused-off-by-one",
        "Counts the unused values of a bit pattern as 2 to the power n minus the largest reading, forgetting that zero is a value.",
        AREA,
    ),
    ("nv-unused-total-values", "Gives the total number of values rather than the number that are unused.", AREA),
    ("nv-colour-channels-swapped", "Writes the colour channels in the wrong order in the hexadecimal code.", AREA),
    ("nv-largest-two-less", "Takes two away from 2 to the power n, not one, to get the largest n-bit value.", AREA),
    (
        "nv-values-half",
        "Gives 2 to the power (n - 1) as the number of values, leaving out the two choices of one bit.",
        AREA,
    ),
    (
        "nv-hex-last-digit-dropped",
        "Stops the repeated division by 16 one step early, leaving off the last digit.",
        AREA,
    ),
    ("nv-hex-digits-bits-over-two", "Divides the number of bits by 2 to count hexadecimal digits.", AREA),
    ("nv-hex-digits-one-extra", "Adds one hexadecimal digit to the count that is needed.", AREA),
    (
        "nv-hex-codes-swapped",
        "Swaps the base and the exponent, working out the digit count to the power 16 instead of 16 to the power of the digit count.",
        AREA,
    ),
    (
        "nv-largest-top-place",
        "Gives the place value of the leading hexadecimal place (16 to the power k - 1) as the largest number.",
        AREA,
    ),
    (
        "nv-hex-top-digit-count",
        "Uses the number of leftover bits as the top hexadecimal digit instead of the largest value they hold.",
        AREA,
    ),
    ("nv-unused-gives-reading", "Gives the largest reading as the number of unused values.", AREA),
]

_FORMAT = " Type the digits only: no 0x, no #, no spaces."
_DIGITS = "0123456789ABCDEF"


# -- typed hexadecimal answers ----------------------------------------------------------------------------------------


def _reads_as_number(text: str) -> bool:
    """True for a numeral such as 3E0 or 12E4, which a number parser reads as scientific notation."""
    try:
        float(text)
    except ValueError:
        return False
    return not text.isdigit()


def hex_fields(text: str) -> dict:
    """A hexadecimal numeral as a typed answer: a one-item ordered list, compared case-insensitively as text.
    A numeral that reads as a number in scientific notation (1E5) is refused, so the draw is made again."""
    if _reads_as_number(text):
        raise TooFewDistractors(f"{text} reads as a number in scientific notation")
    return dict(
        answer_kind="ordered_list", answer=[text], answer_text=text, answer_latex=latex_answer(text, plain=True)
    )


def _hex_accepts(key: str, reply: str) -> bool:
    probe = SimpleNamespace(
        answer=[key], answer_text=key, text="", answer_alt=None, answer_form=None, answer_kind="ordered_list"
    )
    return bool(GRADERS["ordered_list"](probe, reply).correct)


def _same_number(key: str, text: str) -> bool:
    """Two all-digit numerals that differ only in leading zeros: equal as numbers, so the answer contract rejects them."""
    return key.isdigit() and text.isdigit() and int(key) == int(text)


def hex_dist(key: str, candidates, n: int = 4) -> list[Distractor]:
    """Tagged wrong hexadecimal numerals: a candidate that repeats the key, or is accepted by the real grader, is dropped."""
    out: list[Distractor] = []
    seen = {key.upper()}
    for text, tag in candidates:
        if (
            not text
            or text.upper() in seen
            or _reads_as_number(text)
            or _same_number(key, text)
            or _hex_accepts(key, text)
        ):
            continue
        seen.add(text.upper())
        out.append(Distractor(text, tag, [text]))
        if len(out) == n:
            break
    return out


def _test_value(text: str) -> int:
    """How an option is ordered: an all-digit numeral as a decimal number, one with a letter as hexadecimal."""
    return int(text) if text.isdigit() else int(text, 16)


def hex_spread(key: str, candidates, count: int = 3) -> list[Distractor]:
    """`hex_dist` over candidates ordered so that a random number of the chosen wrong numerals are below the key in value.

    Plain near misses (the number a little more or less, same width) make up a side the named slips leave short. A slip
    equal in value to the key (the decimal digits written as hexadecimal) can be neither below nor above it; it takes
    one of the places now and then."""
    rng = _RNG.get()
    usable = [c for c in candidates if c[0] and not _reads_as_number(c[0])]
    if rng is None:
        return hex_dist(key, usable, count)
    kv = _test_value(key)
    width = len(key) if key.startswith("0") else 0
    n = int(key, 16)
    near = [(_hex(n + d, width), NEAR) for d in (-256, -16, -2, -1, 1, 2, 16, 256) if n + d > 0]
    near = [c for c in near if not _reads_as_number(c[0])]
    named = [(_test_value(c[0]), c) for c in usable if _test_value(c[0]) != kv]
    ties = [c for c in usable if _test_value(c[0]) == kv and c[0].upper() != key.upper()]
    fillers = [(_test_value(c[0]), c) for c in near]
    ordered = [c for _, c in spread_order(kv, named, fillers, count)]
    if ties and rng.random() < 0.3:
        ordered[rng.randrange(min(count, len(ordered)))] = rng.choice(ties)
    return hex_dist(key, ordered + usable, count)


def _hex(n: int, width: int = 0) -> str:
    text = format(n, "X")
    return text.rjust(width, "0")


def _from_hex(text: str) -> int:
    return int(text, 16)


def _dist(key, cands, lo=1, hi=None):
    """Three wrong whole numbers with a random number of them below the key, so its rank among the options varies."""
    return _spread(key, cands, lo, hi, 3)


def _spread(key, cands, lo=1, hi=None, count=3):
    fillers = [(key + d, NEAR) for d in (-4, -3, -2, -1, 1, 2, 3, 4)]
    ordered = spread_order(key, cands, fillers, count, lo=lo, hi=hi)
    dist = num_dist(key, ordered, n=count)
    enough(dist)
    return dist


def _num(rng, patterns, **slots):
    return phrase(rng, patterns, **slots)


# -- easy -------------------------------------------------------------------------------------------------------------


def _easy_bits_needed(rng):
    for _ in attempts(200, "a number away from a power-of-two boundary"):
        n = rng.randint(20, 4000)
        if n & (n - 1) != 0 and (n + 1) & n != 0 and (n + 2) & (n + 1) != 0:
            break
    bits = n.bit_length()
    check(2 ** (bits - 1) <= n < 2**bits, "bit length differs")
    check(bits == next(k for k in range(1, 40) if 2**k > n), "bits needed differs")
    cands = [
        (bits - 1, "nv-bits-log-rounded-down"),
        (len(str(n)), "nv-bits-counts-decimal-digits"),
        (bits + 1, "nv-bits-one-extra"),
        (round(math.log2(n)) if round(math.log2(n)) != bits else None, "nv-bits-round-to-nearest"),
        (n // 8, "nv-bits-divides-by-eight"),
    ]
    dist = _dist(bits, cands)
    stem = _num(
        rng,
        (
            "A scoreboard counts whole numbers from 0 up to {n}, stored in binary. What is the smallest number of bits that can hold {n}?",
            "What is the least number of binary digits (bits) needed to write the whole number {n}?",
            "A counter stores a whole number in binary. How many bits does it need to be able to store {n}?",
        ),
        n=n,
    )
    working = [
        f"2^{bits - 1} = {2 ** (bits - 1)} is not more than {n}, and 2^{bits} = {2**bits} is more than {n}.",
        f"So {n} fits in {bits} bits, but not in {bits - 1}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(bits), answer_kind="number")


def _easy_largest(rng):
    n = rng.randint(4, 12)
    value = 2**n - 1
    check(value == int("1" * n, 2), "all ones differs")
    cands = [
        (2**n, "nv-largest-is-power"),
        (2 ** (n - 1), "nv-largest-half"),
        (2 * n - 1, "nv-largest-linear"),
        (2**n - 2, "nv-largest-two-less"),
    ]
    dist = _dist(value, cands)
    stem = _num(
        rng,
        (
            "What is the largest whole number that can be stored in {n} bits (as an ordinary binary number, counting from 0)?",
            "A memory cell has {n} bits. What is the largest whole number it can hold?",
            "A sensor sends its reading as a binary number with {n} digits. What is the largest reading it can send?",
        ),
        n=n,
    )
    working = [f"All {n} bits set to 1 gives {'1' * n} in binary.", f"That is 2^{n} - 1 = {2**n} - 1 = {value}."]
    return dict(text=stem, working=working, distractors=dist, **num_fields(value), answer_kind="number")


def _easy_values(rng):
    n = rng.randint(3, 11)
    value = 2**n
    check(len({format(i, f"0{n}b") for i in range(value)}) == value, "pattern count differs")
    cands = [
        (n * n, "nv-values-squares"),
        (2 * n, "nv-values-doubles-bits"),
        (2**n - 1, "nv-values-forgets-zero"),
        (2 ** (n - 1), "nv-values-half"),
    ]
    dist = _dist(value, cands)
    stem = _num(
        rng,
        (
            "How many different values (counting zero) can be stored using {n} bits?",
            "A code uses {n} binary digits. How many different codes are possible?",
            "A switch panel has {n} switches, each on or off. How many different settings can the panel show?",
        ),
        n=n,
    )
    working = [f"Each bit has 2 choices, so {n} bits give 2^{n} = {value} patterns."]
    return dict(text=stem, working=working, distractors=dist, **num_fields(value), answer_kind="number")


# -- medium -----------------------------------------------------------------------------------------------------------


def _hex_with_letter(rng, lo, hi, digits=None):
    for _ in attempts(500, "a hexadecimal numeral with a letter"):
        n = rng.randint(lo, hi)
        h = _hex(n)
        if any(c.isalpha() for c in h) and (digits is None or len(h) == digits):
            return n, h
    raise AssertionError  # pragma: no cover


def _medium_dec_to_hex(rng):
    n, h = _hex_with_letter(rng, 60, 4000)
    check(_from_hex(h) == n, "hex value differs")
    steps, q = [], n
    remainders = []
    for _ in range(n.bit_length()):
        if q == 0:
            break
        remainders.append(q % 16)
        q //= 16
    wrong_letter = "".join(_DIGITS[(_DIGITS.index(c) + 1) % 16] if c.isalpha() else c for c in h)
    cands = [
        (h[::-1], "nv-hex-remainders-reversed"),
        (wrong_letter, "nv-hex-letter-misvalue"),
        (_hex(n + 16 if n % 16 != 15 else n - 16), NEAR),
        (h[:-1] if len(h) > 2 else "", "nv-hex-last-digit-dropped"),
        (str(n), "nv-hex-decimal-digits"),
    ]
    dist = hex_spread(h, cands)
    enough(dist)
    stem = _num(
        rng,
        (
            "Write the decimal number {n} in hexadecimal. Type the hexadecimal digits (letters in either case).",
            "Convert {n} to base 16. Use the digits 0 to 9 and the letters A to F.",
            "What is {n} as a hexadecimal numeral?",
        ),
        n=n,
    )
    divs = "; ".join(
        f"{r_in} = 16 x {r_in // 16} remainder {r_in % 16}" for r_in in [n // 16**i for i in range(len(h))]
    )
    working = [
        f"Divide by 16 repeatedly: {divs}.",
        f"Read the remainders from the last to the first, with 10 = A up to 15 = F: {h}.",
    ]
    return dict(text=stem + _FORMAT, working=working, distractors=dist, **hex_fields(h))


def _medium_bin_to_hex(rng):
    for _ in attempts(500, "a binary numeral that does not group into fours"):
        length = rng.choice((9, 10, 11, 13, 14))
        bits = "1" + "".join(rng.choice("01") for _ in range(length - 1))
        h = _hex(int(bits, 2))
        if any(c.isalpha() for c in h):
            break
    pad = (-len(bits)) % 4
    padded = "0" * pad + bits
    groups = [padded[i : i + 4] for i in range(0, len(padded), 4)]
    check("".join(_DIGITS[int(g, 2)] for g in groups) == h, "grouped digits differ")
    left = [bits[i : i + 4] for i in range(0, len(bits), 4)]
    left_text = "".join(_DIGITS[int(g, 2)] for g in left)
    threes_pad = "0" * ((-len(bits)) % 3) + bits
    threes = "".join(str(int(threes_pad[i : i + 3], 2)) for i in range(0, len(threes_pad), 3))
    unpadded = "".join(_DIGITS[int(g, 2)] for g in [bits[: len(bits) - 4 * (len(groups) - 1)]] + groups[1:])
    cands = [
        (left_text, "nv-hex-groups-from-left"),
        (threes, "nv-hex-groups-of-three"),
        (h[::-1], "nv-hex-remainders-reversed"),
        (unpadded, "nv-hex-no-padding"),
        (_hex(int(bits, 2) + 1), NEAR),
        (_hex(max(int(bits, 2) - 1, 1)), NEAR),
    ]
    dist = hex_spread(h, cands)
    enough(dist)
    stem = _num(
        rng,
        (
            "Write the binary number {b} in hexadecimal by grouping its digits. Type the hexadecimal digits.",
            "Convert the binary numeral {b} to base 16 (use letters A to F where needed).",
            "A device shows the bit pattern {b}. Write this pattern as a hexadecimal numeral.",
        ),
        b=bits,
    )
    working = [
        f"Group the bits in fours from the right, adding {pad} zero{'s' if pad != 1 else ''} on the left: {' '.join(groups)}.",
        "Each group is one hexadecimal digit: "
        + " ".join(f"{g} = {_DIGITS[int(g, 2)]}" for g in groups)
        + f", giving {h}.",
    ]
    return dict(text=stem + _FORMAT, working=working, distractors=dist, **hex_fields(h))


def _medium_hex_digit_count(rng):
    n = rng.choice([b for b in range(9, 31) if b % 4])
    value = -(-n // 4)
    check(16 ** (value - 1) <= 2**n - 1 < 16**value or value == 1, "digit count differs")
    check(len(_hex(2**n - 1)) == value, "hex length differs")
    cands = [
        (n // 4, "nv-hex-digits-rounded-down"),
        (-(-n // 3), "nv-hex-digits-bits-over-three"),
        (n // 2, "nv-hex-digits-bits-over-two"),
        (value + 1, "nv-hex-digits-one-extra"),
    ]
    dist = _dist(value, cands)
    stem = _num(
        rng,
        (
            "How many hexadecimal digits are needed to write the largest {n}-bit number?",
            "A value of {n} bits is shown in hexadecimal. What is the greatest number of hexadecimal digits it can need?",
            "Each hexadecimal digit stands for 4 bits. How many hexadecimal digits are needed to show {n} bits?",
        ),
        n=n,
    )
    working = [
        f"Each hexadecimal digit covers 4 bits, and {n} / 4 = {n / 4:g}.",
        f"Round up, because a partly used digit is still a digit: {value}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(value), answer_kind="number")


def _medium_hex_codes(rng):
    k = rng.randint(2, 5)
    value = 16**k
    check(len({format(i, f"0{k}X") for i in range(value)}) == value, "code count differs")
    cands = [
        (16 * k, "nv-hex-codes-multiply"),
        (10**k, "nv-hex-codes-base-ten"),
        (16**k - 1, "nv-values-forgets-zero"),
        (k**16 if k**16 < 10**7 else None, "nv-hex-codes-swapped"),
    ]
    dist = _dist(value, cands)
    stem = _num(
        rng,
        (
            "A locker code is {k} hexadecimal digits long (0 to 9 and A to F), and leading zeros are allowed. How many different codes are possible?",
            "How many different {k}-digit hexadecimal numerals are there, counting those that begin with 0?",
            "A device identifier is a string of {k} hexadecimal characters. How many identifiers can exist?",
        ),
        k=k,
    )
    working = [f"Each of the {k} places has 16 choices, so there are 16^{k} = {value} codes."]
    return dict(text=stem, working=working, distractors=dist, **num_fields(value), answer_kind="number")


# -- hard -------------------------------------------------------------------------------------------------------------


def _hard_bits_boundary(rng):
    k = rng.randint(6, 12)
    n = rng.choice([2**k, 2**k, 2**k - 1, 2**k + 1])
    bits = n.bit_length()
    check(bits == (k + 1 if n >= 2**k else k), "boundary bits differ")
    check(2 ** (bits - 1) <= n < 2**bits, "bit length differs")
    cands = [
        (bits - 1, "nv-bits-power-boundary" if n == 2**k else "nv-bits-log-rounded-down"),
        (bits + 1, "nv-bits-one-extra"),
        (len(str(n)), "nv-bits-counts-decimal-digits"),
        (round(math.log2(n)) if round(math.log2(n)) != bits else None, "nv-bits-round-to-nearest"),
    ]
    dist = _dist(bits, cands)
    stem = _num(
        rng,
        (
            "A ticket system numbers its tickets in binary starting from 0, and the last ticket is number {n}. How many bits must each ticket number have?",
            "What is the fewest bits that can store every whole number from 0 up to {n}?",
            "An age field must store whole numbers up to {n} in binary. How many bits does it need at least?",
        ),
        n=n,
    )
    working = [
        f"2^{bits - 1} = {2 ** (bits - 1)} and 2^{bits} = {2**bits}.",
        f"{n} is at least {2 ** (bits - 1)} and less than {2**bits}, so {bits} bits are needed.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(bits), answer_kind="number")


def _hard_codes_for_items(rng):
    k = rng.randint(5, 12)
    m = rng.choice([2**k, 2**k + 1, 2**k - 1, rng.randint(2**k + 2, 2 ** (k + 1) - 2)])
    bits = (m - 1).bit_length()
    check(2**bits >= m > 2 ** (bits - 1), "codes bits differ")
    cands = [
        (bits - 1, "nv-bits-log-rounded-down"),
        (bits + 1, "nv-bits-one-extra"),
        (m.bit_length() if m.bit_length() != bits else None, "nv-bits-power-boundary"),
        (m // 8, "nv-bits-divides-by-eight"),
    ]
    dist = _dist(bits, cands)
    stem = _num(
        rng,
        (
            "A library wants to give each of its {m} members a different binary code, all of the same length. What is the smallest number of bits it can use?",
            "{m} different products each need their own binary barcode of equal length. How many bits per barcode are needed at least?",
            "What is the smallest number of bits for which there are at least {m} different bit patterns?",
        ),
        m=m,
    )
    working = [
        f"n bits give 2^n patterns. 2^{bits - 1} = {2 ** (bits - 1)} is less than {m}, but 2^{bits} = {2**bits} is at least {m}.",
        f"So {bits} bits are needed.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(bits), answer_kind="number")


def _hard_padded_hex(rng):
    width = rng.choice([3, 4])
    n = rng.randint(16 ** (width - 3) + 1, 16 ** (width - 1) - 1)
    h = _hex(n, width)
    check(len(h) == width and _from_hex(h) == n and h.startswith("0"), "padded numeral differs")
    short = _hex(n)
    cands = [
        (short, "nv-hex-padding-dropped"),
        (short.ljust(width, "0"), "nv-hex-padding-wrong-side"),
        (_hex(n + 1, width), NEAR),
        (str(n).rjust(width, "0"), "nv-hex-decimal-digits"),
        (h[::-1], "nv-hex-remainders-reversed"),
    ]
    dist = hex_spread(h, cands)
    enough(dist)
    stem = _num(
        rng,
        (
            "A memory address is shown as exactly {w} hexadecimal digits, with leading zeros added. Write the decimal address {n} in that form.",
            "Write {n} in hexadecimal using exactly {w} digits (put zeros in front if needed).",
            "A serial number is stored as a {w}-digit hexadecimal numeral. What is the serial number whose decimal value is {n}?",
        ),
        w=width,
        n=n,
    )
    working = [
        f"{n} in hexadecimal is {short}.",
        f"The numeral must have {width} digits, so add {width - len(short)} zero{'s' if width - len(short) != 1 else ''} in front: {h}.",
    ]
    return dict(text=stem + _FORMAT, working=working, distractors=dist, **hex_fields(h))


def _hard_colour(rng):
    for _ in attempts(200, "a colour with a zero-padded channel"):
        r, g, b = (rng.choice([rng.randint(0, 15), rng.randint(16, 255)]) for _ in range(3))
        h = f"{r:02X}{g:02X}{b:02X}"
        if any(c.isalpha() for c in h) and len({r, g, b}) == 3 and min(r, g, b) < 16:
            break
    check(h == format((r << 16) | (g << 8) | b, "06X"), "colour code differs")
    unpadded = "".join(f"{v:X}" for v in (r, g, b))
    cands = [
        (unpadded, "nv-hex-padding-dropped"),
        (f"{b:02X}{g:02X}{r:02X}", "nv-colour-channels-swapped"),
        (f"{r:02X}{b:02X}{g:02X}", "nv-colour-channels-swapped"),
        (f"{r:02X}{g:02X}{min(b + 1, 255):02X}", NEAR),
    ]
    dist = hex_spread(h, cands)
    enough(dist)
    stem = _num(
        rng,
        (
            "A web colour is written as three pairs of hexadecimal digits for the red, green and blue amounts, each from 0 to 255. Write red {r}, green {g}, blue {b} as a six-digit code, with no # sign.",
            "An LED strip takes its colour as six hexadecimal digits: two each for red, then green, then blue (each amount is 0 to 255). Give the code for red {r}, green {g}, blue {b}.",
        ),
        r=r,
        g=g,
        b=b,
    )
    working = [
        f"Each amount becomes exactly two hexadecimal digits: {r} = {r:02X}, {g} = {g:02X}, {b} = {b:02X}.",
        f"Join them in the order red, green, blue: {h}.",
    ]
    return dict(text=stem + _FORMAT, working=working, distractors=dist, **hex_fields(h))


def _hard_largest_hex(rng):
    n = rng.choice([b for b in range(5, 23) if b % 4])
    top = n % 4
    h = format(2**n - 1, "X")
    check(h == _DIGITS[2**top - 1] + "F" * (n // 4) and int(h, 16) == 2**n - 1, "largest hexadecimal differs")
    cands = [
        ("F" * len(h), "nv-hex-top-digit-f"),
        (_DIGITS[2**top] + "F" * (n // 4), "nv-hex-top-digit-power"),
        (_DIGITS[top] + "F" * (n // 4), "nv-hex-top-digit-count"),
        ("F" * (n // 4), "nv-hex-digits-rounded-down"),
    ]
    dist = hex_spread(h, cands)
    enough(dist)
    stem = _num(
        rng,
        (
            "Write the largest whole number that can be stored in {n} bits as a hexadecimal numeral.",
            "A register of {n} bits is filled with 1s. What hexadecimal numeral does it show?",
            "All {n} bits of a field are set to 1. Give the value of the field in hexadecimal.",
        ),
        n=n,
    )
    groups = [("1" * top)] + ["1111"] * (n // 4)
    working = [
        f"Group the {n} ones in fours from the right: {' '.join(g for g in groups if g)}.",
        f"The leftover {top} bit{'s' if top != 1 else ''} give {_DIGITS[2**top - 1]}, each full group gives F: {h}.",
    ]
    return dict(text=stem + _FORMAT, working=working, distractors=dist, **hex_fields(h))


def _hard_largest_n_digits(rng):
    k = rng.randint(2, 4)
    value = 16**k - 1
    check(value == int("F" * k, 16), "largest hexadecimal value differs")
    cands = [
        (16**k, "nv-largest-is-power"),
        (16 * k - 1, "nv-hex-codes-multiply"),
        (10**k - 1, "nv-hex-codes-base-ten"),
        (16 ** (k - 1), "nv-largest-top-place"),
    ]
    dist = _dist(value, cands)
    stem = _num(
        rng,
        (
            "What is the largest whole number (in decimal) that can be written with {k} hexadecimal digits?",
            "A counter displays {k} hexadecimal digits. What is the largest value, in decimal, that it can display?",
        ),
        k=k,
    )
    working = [
        f"The largest {k}-digit hexadecimal numeral is {'F' * k}.",
        f"That is 16^{k} - 1 = {16**k} - 1 = {value}.",
    ]
    return dict(text=stem, working=working, distractors=dist, **num_fields(value), answer_kind="number")


_EASY = (_easy_bits_needed, _easy_largest, _easy_values)
_MEDIUM = (_medium_dec_to_hex, _medium_bin_to_hex, _medium_hex_digit_count, _medium_hex_codes)
_HARD = (
    _hard_bits_boundary,
    _hard_codes_for_items,
    _hard_padded_hex,
    _hard_colour,
    _hard_largest_hex,
    _hard_largest_n_digits,
)


@generator(
    TOPIC,
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@uses_rng
def nv_bits_hex_counts(rng, tier="easy"):
    """Bits needed, largest and number of values, hexadecimal numerals typed as text; tiers as in the module docstring."""
    pool = {"easy": _EASY, "medium": _MEDIUM, "hard": _HARD}.get(tier)
    if pool is None:
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, rng.choice(pool))


# -- exam -------------------------------------------------------------------------------------------------------------


@generator(EXAM_TOPIC, level=19, presets={"exam": dict(tier="exam")})
@distinct_part_answers
@uses_rng
def exd_x_nv_bits_hex_counts(rng, tier="exam"):
    """Exam chain: bits for a largest reading, values those bits hold, unused values, hexadecimal digits or numeral."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry_draw(rng, _exam_chain)


_SENSORS = (
    "a rainfall gauge that reports whole millimetres",
    "a tank-level sensor that reports whole litres",
    "a light sensor that reports whole units of brightness",
    "a pedometer that reports a whole number of steps",
)


def _exam_chain(rng):
    for _ in attempts(500, "a reading limit away from a power of two"):
        top = rng.randint(120, 60000)
        bits = top.bit_length()
        room = 2**bits - 1 - top
        if bits % 4 and room >= 20 and (top + 1) & top:
            break
    values = 2**bits
    check(values == 1 << bits and values > top and values // 2 <= top, "values differ")
    unused = values - 1 - top
    digits = -(-bits // 4)
    check(digits == len(format(values - 1, "X")), "digit count differs")
    sensor = rng.choice(_SENSORS)
    ask_hex = rng.random() < 0.5
    a_dist = _spread(
        bits,
        [
            (bits - 1, "nv-bits-log-rounded-down"),
            (len(str(top)), "nv-bits-counts-decimal-digits"),
            (bits + 1, "nv-bits-one-extra"),
            (top // 8, "nv-bits-divides-by-eight"),
        ],
    )
    b_dist = _spread(
        values,
        [
            (bits * bits, "nv-values-squares"),
            (2 * bits, "nv-values-doubles-bits"),
            (values - 1, "nv-values-forgets-zero"),
            (values // 2, "nv-values-half"),
        ],
    )
    c_dist = _spread(
        unused,
        [
            (unused + 1, "nv-unused-off-by-one"),
            (unused - 1, "nv-unused-off-by-one"),
            (values, "nv-unused-total-values"),
            (values - top - 2, "nv-unused-off-by-one"),
            (top, "nv-unused-gives-reading"),
        ],
    )
    for d in (a_dist, b_dist, c_dist):
        enough(d)
    parts = [
        part(
            "How many bits are needed to store every reading from 0 up to the largest reading?",
            "number",
            num_fields(bits),
            2,
            [
                f"2^{bits - 1} = {2 ** (bits - 1)} is not more than {top}, and 2^{bits} = {values} is more, so {bits} bits are needed."
            ],
            a_dist,
            sensible_range=(1, None),
        ),
        part(
            "How many different values can that many bits hold, counting zero?",
            "number",
            num_fields(values),
            1,
            [f"{bits} bits give 2^{bits} = {values} different patterns."],
            b_dist,
            sensible_range=(2, None),
            follow_through="2**part_a",
        ),
        part(
            "How many of those values are not used by readings from 0 up to the largest reading?",
            "number",
            num_fields(unused),
            2,
            [
                f"The readings 0 to {top} use {top + 1} values.",
                f"{values} - {top + 1} = {unused} values are left over.",
            ],
            c_dist,
            sensible_range=(0, None),
            follow_through=f"2**part_a - {top} - 1",
        ),
    ]
    if ask_hex:
        h = format(top, "X")
        check(int(h, 16) == top, "hex differs")
        cands = [
            (h[::-1], "nv-hex-remainders-reversed"),
            (str(top), "nv-hex-decimal-digits"),
            (format(top + 16, "X") if top % 16 < 15 else format(top - 16, "X"), NEAR),
            (h[:-1], "nv-hex-last-digit-dropped"),
        ]
        d_dist = hex_spread(h, cands)
        enough(d_dist)
        parts.append(
            part(
                "Write the largest reading in hexadecimal." + _FORMAT,
                "ordered_list",
                {k: v for k, v in hex_fields(h).items() if k != "answer_kind"},
                2,
                [f"Divide {top} by 16 repeatedly and read the remainders from the last to the first: {h}."],
                d_dist,
            )
        )
        tail = ""
    else:
        d_dist = _spread(
            digits,
            [
                (bits // 4, "nv-hex-digits-rounded-down"),
                (-(-bits // 3), "nv-hex-digits-bits-over-three"),
                (digits + 1, "nv-hex-digits-one-extra"),
                (bits // 2, "nv-hex-digits-bits-over-two"),
            ],
        )
        enough(d_dist)
        parts.append(
            part(
                "Each hexadecimal digit stands for 4 bits. How many hexadecimal digits are needed to show a full reading of that many bits?",
                "number",
                num_fields(digits),
                1,
                [f"{bits} / 4 = {bits / 4:g}, rounded up to {digits}."],
                d_dist,
                sensible_range=(1, None),
                follow_through="ceiling(part_a/4)",
            )
        )
    text = f"The logger in {sensor} stores each reading as a binary number. The largest reading it ever has to record is {top}."
    return dict(text=text, parts=parts)
