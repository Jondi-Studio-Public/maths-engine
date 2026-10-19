"""Shared builders for the `r3a_*` binomial and discrete-random-variable skills.

A part is a dict for `mathgen.core.Part`. `number_part` states its answer at a fixed number of decimal places and
keeps every distractor at that same number of places; `fraction_part` states an exact rational. Both take `slips`:
a list of `(value, misconception id)` pairs in order of likelihood, from which up to three distinct wrong options
are kept. A slip is dropped when it repeats the answer, repeats an earlier slip, or is accepted by the real grader
for the part, so every option a learner sees is marked wrong. `single` turns a part into a one-part question.

The module also holds the exact binomial arithmetic the skills share (`pmf`, `cdf`, `span`), the binomial
contexts, and small helpers for choosing draws whose rounded answers are stable.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from types import SimpleNamespace

from ..answers import Distractor, exact_value
from ..core import (
    VerificationError,
    check,
    latex_answer,
    numeric_answer,
    rational_text,
    round_half_up,
    stated_precision_string,
)
from ..grading import GRADERS

_NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:/\d+)?(?![\w.])")


def number_text(value: Fraction, places: int) -> str:
    """`value` rounded half up to `places` places and written at exactly that many places."""
    rounded = round_half_up(Fraction(value), places)
    return stated_precision_string(rounded, places) if places else str(int(rounded))


def _accepted(kind: str, prompt: str, answer, answer_text: str, reply: str) -> bool:
    probe = SimpleNamespace(
        answer=answer, answer_text=answer_text, text=prompt, answer_alt=None, answer_form=None, answer_kind=kind
    )
    return bool(GRADERS[kind](probe, reply).correct)


class ThinOptions(VerificationError):
    """A draw whose slips left fewer than two wrong options; `build_until` redraws it."""


def build_until(rng, build, tries: int = 200):
    """Call `build(rng)` until it returns without `ThinOptions`; the draw that failed is simply discarded."""
    for _ in range(tries):
        try:
            return build(rng)
        except ThinOptions:
            continue
    raise VerificationError(f"no draw with two wrong options in {tries} tries")


def _finish(kind, prompt, answer, text, marks, working, options, follow_through, sensible_range):
    if len(options) < 2:
        raise ThinOptions(f"fewer than two distractors for {prompt!r}")
    part = dict(
        prompt=prompt,
        answer_kind=kind,
        answer=answer,
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=options,
    )
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part


def number_part(
    prompt, value, places, marks, working, slips, *, follow_through=None, sensible_range=None, keep=3, within=None
):
    value = Fraction(value)
    text = number_text(value, places)
    answer = numeric_answer(round_half_up(value, places))
    options: list[Distractor] = []
    seen = {text}
    for item in slips:
        if item is None or item[0] is None:
            continue
        slip, tag = item
        shown = number_text(Fraction(slip), places)
        if within is not None and not within[0] < Fraction(shown) < within[1]:
            continue  # an option outside the possible range (a probability above 1) is not a believable slip
        if shown in seen or _accepted("number", prompt, answer, text, shown):
            continue
        seen.add(shown)
        options.append(Distractor(shown, tag, exact_value(Fraction(shown))))
        if len(options) == keep:
            break
    return _finish("number", prompt, answer, text, marks, working, options, follow_through, sensible_range)


def fraction_part(prompt, value, marks, working, slips, *, follow_through=None, sensible_range=None, keep=3):
    value = Fraction(value)
    text = rational_text(value)
    answer = exact_value(value)
    options: list[Distractor] = []
    seen = {text}
    for item in slips:
        if item is None or item[0] is None:
            continue
        slip, tag = item
        slip = Fraction(slip)
        shown = rational_text(slip)
        if shown in seen or slip == value or _accepted("fraction", prompt, answer, text, shown):
            continue
        seen.add(shown)
        options.append(Distractor(shown, tag, exact_value(slip)))
        if len(options) == keep:
            break
    return _finish("fraction", prompt, answer, text, marks, working, options, follow_through, sensible_range)


def clashes(keys, *texts: str) -> bool:
    """True when any key of two or more characters appears as a whole number in the given texts."""
    shown = {m for t in texts for m in _NUMBER_RE.findall(t)}
    return any(len(str(k)) >= 2 and str(k) in shown for k in keys)


def single(text: str, part: dict) -> dict:
    """A one-part question: `text` is the stem and `part` (from `number_part` / `fraction_part`) supplies the key."""
    return dict(
        text=text,
        answer=part["answer"],
        answer_text=part["answer_text"],
        answer_latex=part["answer_latex"],
        answer_kind=part["answer_kind"],
        working=part["working"],
        distractors=part["distractors"],
    )


def sqrt_fraction(value: Fraction) -> Fraction:
    """The square root of a non-negative `Fraction`, to about 15 significant figures."""
    return Fraction(repr(math.sqrt(value)))


def clear_of_boundary(value: Fraction, places: int, margin: Fraction = Fraction(1, 10)) -> bool:
    """True when `value` is not within `margin` of a rounding boundary at `places` places."""
    scaled = abs(Fraction(value)) * 10**places
    frac = scaled - math.floor(scaled)
    return abs(frac - Fraction(1, 2)) > margin


# -- binomial arithmetic, exact --------------------------------------------------------------------------------


def pmf(n: int, p: Fraction, k: int) -> Fraction:
    """P(X = k) for X ~ Bin(n, p), exact; zero outside 0..n."""
    if k < 0 or k > n:
        return Fraction(0)
    return Fraction(math.comb(n, k)) * p**k * (1 - p) ** (n - k)


def span(n: int, p: Fraction, lo: int, hi: int) -> Fraction:
    """P(lo <= X <= hi), exact."""
    return sum((pmf(n, p, k) for k in range(max(lo, 0), min(hi, n) + 1)), Fraction(0))


def cdf(n: int, p: Fraction, k: int) -> Fraction:
    """P(X <= k), exact."""
    return span(n, p, 0, k)


def dec(p: Fraction) -> str:
    """A probability or proportion as a short decimal, `0.35`."""
    return f"{float(p):g}"


#: (setting with {n} and {p}, what X counts, what the complementary count Y counts, plural noun, plausible p values)
BINOMIAL_CONTEXTS = (
    (
        "A netballer takes {n} penalty shots, each one independently scoring with probability {p}.",
        "the number of shots that score",
        "the number of shots that miss",
        "shots",
        tuple(Fraction(k, 20) for k in range(8, 19)),
    ),
    (
        "An inspector tests {n} phone screens chosen at random, and each is independently faulty with probability {p}.",
        "the number of faulty screens",
        "the number of screens that are not faulty",
        "screens",
        tuple(Fraction(k, 20) for k in range(1, 7)),
    ),
    (
        "A gardener sows {n} seeds, each of which independently germinates with probability {p}.",
        "the number of seeds that germinate",
        "the number of seeds that fail to germinate",
        "seeds",
        tuple(Fraction(k, 20) for k in range(10, 20)),
    ),
    (
        "A call centre rings {n} households, and each household independently answers with probability {p}.",
        "the number of households that answer",
        "the number of households that do not answer",
        "calls",
        tuple(Fraction(k, 20) for k in range(4, 13)),
    ),
    (
        "A council surveys {n} commuters, each of whom independently travels by train with probability {p}.",
        "the number of surveyed commuters who travel by train",
        "the number of surveyed commuters who do not travel by train",
        "commuters",
        tuple(Fraction(k, 20) for k in range(3, 11)),
    ),
    (
        "A lab tests {n} batteries and each one independently passes with probability {p}.",
        "the number of batteries that pass",
        "the number of batteries that fail",
        "batteries",
        tuple(Fraction(k, 20) for k in range(15, 20)),
    ),
)

#: Probabilities that give a mean np with at most two decimal places for any n.
P_CHOICES = tuple(Fraction(k, 20) for k in range(2, 19) if k != 10) + (Fraction(1, 2),)
