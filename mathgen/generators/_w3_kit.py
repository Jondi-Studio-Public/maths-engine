"""Small shared helpers for the `w3_*` consumer-arithmetic builds: money and rounded-number formats with tagged
distractors that the typed grader cannot accept, and a few Fraction helpers.

Nothing here draws random numbers or registers a topic. Amounts are exact `Fraction` dollars; a answer is rounded
half-up to its stated number of places and every distractor is kept at least two units of the last place away from the
answer and from each other, so the grader's half-unit tolerance cannot accept one.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ..answers import Distractor
from ..core import check, round_half_up

__all__ = [
    "Fmt",
    "MONEY",
    "WHOLE",
    "DP1",
    "DP2",
    "dollars",
    "dollars_plain",
    "whole_dollars",
    "clear_of_tie",
    "frac",
    "pct_text",
    "pow_frac",
    "art",
]


def frac(value) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def clear_of_tie(value, places: int, margin: Fraction = Fraction(1, 20)) -> bool:
    """Whether rounding `value` to `places` places is not a near-tie (so the stated answer is unambiguous)."""
    scaled = abs(frac(value)) * 10**places
    rest = scaled - int(scaled)
    return abs(rest - Fraction(1, 2)) > margin


def dollars(value) -> str:
    """A dollar amount as a prompt prints it: $12,345 or $12,345.50."""
    value = frac(value)
    check((value * 100).denominator == 1, f"{value} is not a whole number of cents")
    cents = int(value * 100)
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}${cents // 100:,}" + ("" if cents % 100 == 0 else f".{cents % 100:02d}")


def dollars_plain(value) -> str:
    """Always two decimal places, no symbol: the form of a money answer."""
    value = frac(value)
    check((value * 100).denominator == 1, f"{value} is not a whole number of cents")
    cents = int(value * 100)
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}{cents // 100}.{cents % 100:02d}"


def whole_dollars(value) -> str:
    """$12,345 for a whole number of dollars (checked)."""
    value = frac(value)
    check(value.denominator == 1, f"{value} is not a whole number of dollars")
    return dollars(value)


def pct_text(value) -> str:
    """A percentage as a prompt prints it: 12, 12.5 (no trailing zeros)."""
    value = frac(value)
    check((value * 10).denominator == 1, f"{value} needs more than one decimal place")
    if value.denominator == 1:
        return str(int(value))
    return f"{float(value):.1f}"


def pow_frac(base, n: int) -> Fraction:
    return frac(base) ** n


def art(noun: str) -> str:
    """'a' or 'an' for a noun phrase."""
    return "an" if noun[:1].lower() in "aeiou" else "a"


@dataclass(frozen=True)
class Fmt:
    """How an answer is written: `places` decimal places (0 = a whole number), rounded half-up."""

    places: int

    def value(self, x) -> Fraction:
        return round_half_up(frac(x), self.places)

    def text(self, x) -> str:
        v = self.value(x)
        if self.places == 0:
            return str(int(v))
        scaled = int(abs(v) * 10**self.places)
        whole, rest = divmod(scaled, 10**self.places)
        sign = "-" if v < 0 else ""
        return f"{sign}{whole}.{rest:0{self.places}d}"

    def fields(self, x) -> dict:
        """The answer fields for the exact value `x` rounded to this format; refuses a near-tie."""
        x = frac(x)
        check(clear_of_tie(x, self.places) or (x * 10**self.places).denominator == 1, f"{x} is a rounding near-tie")
        text = self.text(x)
        if self.places == 0:
            return dict(answer=int(text), answer_text=text, answer_latex=text)
        return dict(answer=float(text), answer_text=text, answer_latex=text)

    def dist(
        self, correct, candidates, n: int = 4, allow_negative: bool = False, extremes: int = 1
    ) -> list[Distractor]:
        """Tagged distractors from ``[(value, misconception id), ...]`` in priority order. A value that rounds within
        two units of the last place of the answer, or of an earlier option, is skipped (so is None, and a negative
        value unless allowed). At most `extremes` options may be more than thirty times larger than the answer, and none may be thirty times smaller,
        so the key is not the only value of a believable size."""
        unit = Fraction(1, 10**self.places)
        gap = unit * (1 if self.places == 0 else 2)
        seen = [self.value(correct)]
        out: list[Distractor] = []
        size = abs(frac(correct))
        odd = 0
        for item in candidates:
            if item is None or item[0] is None:
                continue
            value, tag = item
            v = self.value(value)
            if (v < 0 and not allow_negative) or any(abs(v - s) < gap for s in seen):
                continue
            if size and abs(v) * 30 < size:
                continue  # an option far smaller than the key is junk, never a believable slip
            if size and abs(v) > 30 * size:
                if odd >= extremes:
                    continue
                odd += 1
            seen.append(v)
            text = self.text(v)
            out.append(Distractor(text, tag, int(text) if self.places == 0 else float(text)))
            if len(out) == n:
                break
        return out


MONEY = Fmt(2)
WHOLE = Fmt(0)
DP1 = Fmt(1)
DP2 = Fmt(2)
