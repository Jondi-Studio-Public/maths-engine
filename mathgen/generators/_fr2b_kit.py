"""Small helpers shared by the `fr2b_*` number and operations generators.

Everything here is plain arithmetic and formatting: number words, thousands grouping, mixed numerals, a
tagged-distractor picker that also re-checks every option against the exact answer, and a Fraction-safe signed
integer formatter. Nothing here draws from a random generator.
"""

from __future__ import annotations

from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import VerificationError, attempts, check, pick_distractors

_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen"
).split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()

MINUS = "-"


def _below_hundred(n: int) -> str:
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] if ones == 0 else f"{_TENS[tens]}-{_ONES[ones]}"


def _below_thousand(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    if hundreds == 0:
        return _below_hundred(rest)
    head = f"{_ONES[hundreds]} hundred"
    return head if rest == 0 else f"{head} and {_below_hundred(rest)}"


def number_words(n: int) -> str:
    """Australian number words for 1 to 999 999: 23 406 is "twenty-three thousand, four hundred and six"."""
    check(1 <= n <= 999_999, "number_words covers 1 to 999 999")
    thousands, rest = divmod(n, 1000)
    if thousands == 0:
        return _below_thousand(rest)
    head = f"{_below_thousand(thousands)} thousand"
    if rest == 0:
        return head
    if rest < 100:
        return f"{head} and {_below_hundred(rest)}"
    return f"{head}, {_below_thousand(rest)}"


def grouped(n: int) -> str:
    """A whole number with a space between thousands, as Australian schools write it: 23406 -> "23 406"."""
    text = f"{abs(n):,}".replace(",", " ")
    return f"-{text}" if n < 0 else text


def mixed_text(value: Fraction) -> str:
    """A positive Fraction as a mixed numeral in lowest terms: 11/4 -> "2 3/4", 5/2 -> "2 1/2", 3 -> "3"."""
    check(value >= 0, "mixed_text is for non-negative values")
    whole, rest = divmod(value.numerator, value.denominator)
    if rest == 0:
        return str(whole)
    frac = f"{rest}/{value.denominator}"
    return frac if whole == 0 else f"{whole} {frac}"


def frac_value(value: Fraction):
    """`Question.answer` for an exact fraction (a whole number stays an int)."""
    return exact_value(value)


def int_dist(value: int, misconception: str) -> Distractor:
    return Distractor(str(value), misconception, value)


def mixed_dist(value: Fraction, misconception: str, *, text: str | None = None) -> Distractor:
    return Distractor(text if text is not None else mixed_text(value), misconception, exact_value(value))


def dec_dist(text: str, value: float, misconception: str) -> Distractor:
    return Distractor(text, misconception, value)


def list_text(items) -> str:
    return ", ".join(str(i) for i in items)


class TooFewDistractors(VerificationError):
    """A draw whose error answers collapse onto the key or each other; `build` draws again."""


def build(builder, rng, tier: str):
    """`builder(rng)`, drawn again when a draw leaves too few distractors. Any other failure is raised as it is."""
    for _ in attempts(40, f"a {tier} question with enough distractors"):
        try:
            return builder(rng)
        except TooFewDistractors:
            continue


def tagged(answer_text: str, candidates, need: int = 3, *, n: int = 3, unordered: bool = False) -> list[Distractor]:
    """The first `n` candidates that differ from the answer, deduplicated, with at least `need` of them.

    A candidate is a `Distractor`. Candidates are listed most-likely error first, so the likeliest errors survive
    when more than `n` are on offer."""
    picked = pick_distractors(answer_text, list(candidates), n, unordered=unordered)
    if len(picked) < need:
        raise TooFewDistractors(f"only {len(picked)} distractors for {answer_text!r}, need {need}")
    check(all(isinstance(d, Distractor) for d in picked), "every distractor must carry its misconception")
    return picked


def differs(value, *others) -> bool:
    """True when `value` is a different number from every one of `others` (Fractions, so 126 2/6 equals 126 1/3)."""
    return all(Fraction(value) != Fraction(o) for o in others)
