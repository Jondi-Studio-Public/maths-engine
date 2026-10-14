"""Small helpers shared by the senior Methods and Specialist builds in the `sr2b_*` generators (rational function
features, reciprocal trigonometric graphs, exact trigonometric values, position-time graphs, signed area, composite
function domains, restricted counting and scale-drawing costs).

Not a generator module (the leading underscore keeps the package loader from importing it): it holds superscripts
for the stem, tagged distractors for each answer kind these builds use, and a labelled-pair answer whose components
may be exact fractions.
"""

from __future__ import annotations

import functools
from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import VerificationError, check, pick_distractors, rational_text

_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def sup(n: int) -> str:
    """`n` as superscript digits: 4 -> ⁴."""
    return str(n).translate(_SUPERSCRIPT)


def frac(value) -> Fraction:
    return Fraction(value)


def int_dist(value, tag: str) -> Distractor:
    """A wrong whole number as a tagged distractor."""
    check(Fraction(value).denominator == 1, f"{value} is not a whole number")
    return Distractor(str(int(value)), tag, int(value))


def rat_dist(value, tag: str) -> Distractor:
    """A wrong exact fraction (or whole number) as a tagged distractor."""
    value = Fraction(value)
    return Distractor(rational_text(value), tag, exact_value(value))


def text_dist(text: str, tag: str) -> Distractor:
    """A wrong typed expression (spelled as an answer is) as a tagged distractor whose value is its own text."""
    return Distractor(text, tag, text)


def pi_text(coeff) -> str:
    """A multiple of pi as it is typed: Fraction(5, 6) -> 5pi/6, 1 -> pi, 0 -> 0."""
    from ..core import format_pi_multiple

    return format_pi_multiple(Fraction(coeff))


def pi_latex(coeff) -> str:
    from ..core import latex_pi_multiple

    return latex_pi_multiple(Fraction(coeff))


def pi_shown(coeff) -> str:
    """A multiple of pi as the stem and working print it: 5π/6."""
    return pi_text(coeff).replace("pi", "π")


class TooFewDistractors(VerificationError):
    """A draw left fewer than two distinct wrong values; `retry_short` draws again."""


def retry_short(build, tries: int = 40):
    """Wrap a generator body so a draw with too few distractors is redrawn (at most `tries` times, deterministically)."""

    @functools.wraps(build)
    def run(rng, *args, **kwargs):
        for _ in range(tries):
            try:
                return build(rng, *args, **kwargs)
            except TooFewDistractors:
                continue
        raise TooFewDistractors(f"{build.__name__}: no draw had enough distractors in {tries} tries")

    return run


def pick(answer_text: str, candidates: list, count: int = 3, *, minimum: int = 2) -> list:
    """The first `count` candidates that are not the answer and not each other, at least `minimum` of them."""
    candidates = [c for c in candidates if not _is_zero_text(getattr(c, "text", str(c))) or _is_zero_text(answer_text)]
    out = pick_distractors(answer_text, candidates, count)
    if len(out) < minimum:
        raise TooFewDistractors(f"too few distractors for {answer_text}: {[str(c) for c in candidates]}")
    return out


def _is_zero_text(text: str) -> bool:
    """Whether a number's text is a plain zero ("0", "0.0", "-0.00")."""
    stripped = text.strip().lstrip("-").replace(".", "")
    return bool(stripped) and set(stripped) == {"0"}


def value_text(v) -> str:
    """A pair component as the answer shows it: a whole number or `p/q`."""
    return rational_text(Fraction(v))


def value_latex(v) -> str:
    v = Fraction(v)
    if v.denominator == 1:
        return str(v.numerator)
    sign = "-" if v < 0 else ""
    return rf"{sign}\frac{{{abs(v.numerator)}}}{{{v.denominator}}}"


def pair_value(v):
    """The stored value of one pair component: an exact `Fraction`."""
    return Fraction(v)


class Pair:
    """A labelled pair such as ``x = -3, y = 2/5`` for the `pair` answer kind."""

    def __init__(self, label1: str, v1, label2: str, v2):
        self.labels = (label1, label2)
        self.values = (Fraction(v1), Fraction(v2))

    @property
    def answer(self) -> dict:
        return {lab: pair_value(v) for lab, v in zip(self.labels, self.values)}

    @property
    def text(self) -> str:
        return ", ".join(f"{lab} = {value_text(v)}" for lab, v in zip(self.labels, self.values))

    @property
    def latex(self) -> str:
        return r",\ ".join(f"{lab} = {value_latex(v)}" for lab, v in zip(self.labels, self.values))

    def fields(self) -> dict:
        return dict(answer_kind="pair", answer=self.answer, answer_text=self.text, answer_latex=self.latex)

    def distractor(self, tag: str) -> Distractor:
        return Distractor(self.text, tag, self.answer)


def pair_candidates(answer: Pair, wrong: list[tuple[str, Fraction, Fraction]]) -> list[Distractor]:
    """Tagged pair distractors `(tag, v1, v2)`; any that differ from the answer by under 0.06 in both
    components (the grader's fallback tolerance) are dropped."""
    out = []
    for tag, v1, v2 in wrong:
        v1, v2 = Fraction(v1), Fraction(v2)
        if abs(v1 - answer.values[0]) < Fraction(6, 100) and abs(v2 - answer.values[1]) < Fraction(6, 100):
            continue
        out.append(Pair(answer.labels[0], v1, answer.labels[1], v2).distractor(tag))
    return out


# -- decimal number answers ---------------------------------------------------------------------------------------


def places_needed(value, max_places: int = 2) -> int:
    """The fewest decimal places that show `value` exactly; the build fails if it needs more than `max_places`."""
    value = Fraction(value)
    for places in range(max_places + 1):
        if (value * 10**places).denominator == 1:
            return places
    raise ValueError(f"{value} needs more than {max_places} decimal places")


def number_fields(value, places: int) -> dict:
    """answer, answer_text and answer_latex for a number shown at exactly `places` decimal places."""
    from ..core import numeric_answer, stated_precision_string

    value = Fraction(value)
    text = stated_precision_string(value, places)
    return dict(answer_kind="number", answer=numeric_answer(value), answer_text=text, answer_latex=text)


def decimal_dist(value, places: int, tag: str):
    """A wrong number as a tagged distractor shown at the answer's number of places (rounded half up), or None when
    it is negative zero."""
    from ..core import numeric_answer, round_half_up, stated_precision_string

    shown = round_half_up(Fraction(value), places)
    return Distractor(stated_precision_string(shown, places), tag, numeric_answer(shown))
