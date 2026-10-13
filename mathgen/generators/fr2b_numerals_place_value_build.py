"""Generates `fr2b_numerals_place_value` questions for `f10-numerals-words-and-place-value` (AC9M3N01). `answer: number`.

Scope: numbers above ten thousand, said, read and written. The learner turns number words into a numeral, says what
a named digit is worth, or moves a number up or down by a power of ten. Ordering is the neighbouring skill
`f10-large-number-naming`; nothing here orders numbers.

Ladder (reasoning escalation, not size):
  easy:   number words to a numeral where at least one place is empty, so the zero has to be written: "twenty-three
          thousand and six" is 23 006, not 236.
  medium: the value of one named digit in a six-digit numeral (all digits different, so the digit names its own
          place). The learner has to read the place, not just the digit.
  hard:   a number given in WORDS is moved up or down by 10, 100, 1 000 or 10 000 where the digit in that place is
          0 (taking away) or 9 (adding), so the change has to regroup across the neighbouring place.

The key is built from the parts of the number, then re-derived a second way: from the words' own numeral, from
string positions and from plain arithmetic. Every distractor is the answer of one named error.
"""

from __future__ import annotations

from ..answers import Distractor
from ..core import attempts, check, generator, phrase
from ._fr2b_kit import build, grouped, int_dist, number_words, tagged

_TIERS = ("easy", "medium", "hard")

_SINGULAR = {
    0: "one",
    1: "ten",
    2: "hundred",
    3: "thousand",
    4: "ten thousand",
    5: "hundred thousand",
}

_PLACE_NAMES = {
    0: "ones",
    1: "tens",
    2: "hundreds",
    3: "thousands",
    4: "ten thousands",
    5: "hundred thousands",
}


def _words_to_numeral_distractors(n: int) -> list[Distractor]:
    thousands, rest = divmod(n, 1000)
    hundreds, tens_ones = divmod(rest, 100)
    pieces = [str(thousands)] + ([str(hundreds)] if hundreds else []) + ([str(tens_ones)] if tens_ones else [])
    digits = str(n)
    inner_zero = digits.replace("0", "", 1) if "0" in digits[1:] else digits + "0"
    # Reverse the last two digits, or the hundreds and tens digits when the last two match.
    swapped = list(digits)
    i = len(swapped) - 2 if swapped[-1] != swapped[-2] else len(swapped) - 3
    swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
    return [
        int_dist(int("".join(pieces)), "fr2b-numeral-written-without-placeholder-zero"),
        int_dist(int(inner_zero), "fr2b-numeral-has-wrong-number-of-zeros"),
        int_dist(int("".join(swapped)), "fr2b-numeral-digits-transposed"),
        int_dist(n * 10, "fr2b-numeral-has-wrong-number-of-zeros"),
    ]


def _easy_number(rng) -> int:
    """A five-digit number with an empty place: no hundreds, no tens, or a round number of hundreds."""
    thousands = rng.randint(10, 99)
    shape = rng.choice(("no-hundreds", "no-tens", "no-tens", "round"))
    if shape == "no-hundreds":
        rest = rng.randint(11, 99)
        rest = rest if rest % 10 else rest + 1
    elif shape == "no-tens":
        rest = rng.randint(1, 9) * 100 + rng.randint(1, 9)
    else:
        rest = rng.randint(1, 9) * 100
    return thousands * 1000 + rest


def _easy(rng):
    n = _easy_number(rng)
    words = number_words(n)
    check(len(str(n)) == 5 and "0" in str(n)[1:], "the number needs an empty place")
    thousands, rest = divmod(n, 1000)
    check(thousands * 1000 + rest == n, "the parts do not rebuild the number")
    text = phrase(
        rng,
        [
            "Write {words} in numerals.",
            "A sign says {words} people visited the museum this year. Write that number in numerals.",
            "Write the number {words} using digits.",
        ],
        words=words,
    )
    distractors = tagged(str(n), _words_to_numeral_distractors(n))
    working = [
        f"{thousands} thousand is {grouped(thousands * 1000)}, and the rest is {rest}, so the number is {grouped(n)}.",
        "A place with nothing in it still needs a 0.",
    ]
    return _result(text, n, distractors, working)


def _medium(rng):
    for _ in attempts(500, "six different digits with a non-zero digit in the chosen place"):
        digits = rng.sample(range(10), 6)
        place = rng.randint(1, 5)
        if digits[0] != 0 and digits[5 - place] != 0:
            break
    n = int("".join(str(d) for d in digits))
    digit = digits[5 - place]
    check(str(n).count(str(digit)) == 1, "the digit must name its own place")
    value = digit * 10**place
    check(int(str(n)[::-1][place]) == digit, "the digit is not in the place")
    text = phrase(
        rng,
        [
            "What is the value of the digit {d} in {n}?",
            "In the number {n}, what is the value of the digit {d}?",
            "The digit {d} is in the number {n}. What is its value?",
        ],
        d=digit,
        n=grouped(n),
    )
    cands = [
        int_dist(digit, "fr2b-digit-value-is-the-digit-itself"),
        int_dist(digit * 10 ** (place + 1), "fr2b-digit-value-from-wrong-place"),
        int_dist(10**place, "fr2b-place-name-given-for-digit-value"),
    ]
    if place >= 1:
        cands.append(int_dist(digit * 10 ** (place - 1), "fr2b-digit-value-from-wrong-place"))
    cands.append(
        int_dist(
            digit * 10 ** (place + 2) if place <= 3 else digit * 10 ** (place - 2), "fr2b-digit-value-from-wrong-place"
        )
    )
    working = [
        f"Counting from the right, the {digit} is in the {_PLACE_NAMES[place]} place.",
        f"{digit} {_SINGULAR[place] if digit == 1 else _PLACE_NAMES[place]} is {grouped(value)}.",
    ]
    return _result(text, value, tagged(str(value), cands), working)


def _regroup_distractors(n: int, place: int, step_sign: int) -> list[Distractor]:
    step = 10**place
    digits = [int(c) for c in str(n)]
    i = len(digits) - 1 - place
    # No regrouping: the digit in the place changes on its own and the neighbouring place is left alone.
    digits[i] = (digits[i] + step_sign) % 10
    no_borrow = int("".join(str(d) for d in digits))
    return [
        int_dist(no_borrow, "fr2b-no-regrouping-across-a-zero-or-nine"),
        int_dist(n + step_sign * step * 10, "fr2b-changes-the-wrong-place"),
        int_dist(n - step_sign * step, "fr2b-more-and-less-reversed"),
        int_dist(n + step_sign * step // 10, "fr2b-changes-the-wrong-place"),
    ]


def _hard(rng):
    for _ in attempts(1000, "a number whose digit in the changed place is 0 or 9"):
        place = rng.randint(1, 4)
        step_sign = rng.choice((1, -1))
        edge = 9 if step_sign == 1 else 0
        n = rng.randint(10_000, 99_999)
        if (n // 10**place) % 10 == edge and n % 10 != 0 and 0 < n + step_sign * 10**place < 100_000:
            break
    step = 10**place
    answer = n + step_sign * step
    check(answer > 0 and (answer // step) % 10 != (n // step) % 10, "the changed place must change")
    check(len(str(answer)) >= 4, "the answer needs at least four digits")
    more = step_sign == 1
    amount = {1: "ten", 2: "one hundred", 3: "one thousand", 4: "ten thousand"}[place]
    word = "more" if more else "less"
    text = phrase(
        rng,
        [
            "What number is {amount} {word} than {words}? Write your answer in numerals.",
            "Start with {words}. What number is {amount} {word}? Write it in numerals.",
        ],
        amount=amount,
        word=word,
        words=number_words(n),
    )
    working = [
        f"{number_words(n).capitalize()} is {grouped(n)}.",
        f"{'Adding' if more else 'Taking away'} {grouped(step)} changes the {_PLACE_NAMES[place]} place, and the "
        f"{'9 becomes 0 with 1 more in the next place' if more else '0 needs 1 regrouped from the next place'}.",
        f"{grouped(n)} {'+' if more else '-'} {grouped(step)} = {grouped(answer)}.",
    ]
    return _result(text, answer, tagged(str(answer), _regroup_distractors(n, place, step_sign)), working)


def _result(text, answer, distractors, working):
    return dict(
        text=text,
        latex="",
        answer_kind="number",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=working,
        distractors=distractors,
    )


_BUILDERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator("fr2b_numerals_place_value", level=3, presets={tier: dict(tier=tier) for tier in _TIERS})
def fr2b_numerals_place_value(rng, tier="easy"):
    """Number words, the value of a digit, and moving a number by a power of ten; tiers in the module docstring."""
    check(tier in _BUILDERS, f"unknown tier {tier!r}")
    return build(_BUILDERS[tier], rng, tier)
