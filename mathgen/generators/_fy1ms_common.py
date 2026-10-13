"""Shared helpers for the first-year multivariable, 3D geometry and statistics builds (modules `fy1ms_*_build`).

* `Fm` and `say`: a sentence written once from prose and formulas, each formula carrying both its plain Unicode
  spelling (what the learner reads) and its TeX, so the two never drift apart.
* `plain_expr` and `tex_expr`: a SymPy expression as plain text and as TeX.
* Number helpers: rounding to a stated number of decimal places, a check that a value is not near a rounding tie,
  tagged number distractors that clear the typed-number grader by a full unit in the last place, and builders for
  number parts, fraction parts, expression parts and a one-mark "choose the correct statement" part.

Nothing here is specific to one skill. Keep it that way: a skill's own errors and wording live in its module.
"""

from __future__ import annotations

import math
import random
import re
from fractions import Fraction

import sympy

from ..answers import Distractor
from ..core import (
    VerificationError,
    _escape_prose,
    attempts,
    check,
    latex_answer,
    pick_distractors,
    rational_text,
    round_half_up,
    stated_precision_string,
)

# ---------------------------------------------------------------------------
# Sentences with maths: prose plus formulas, each formula as (plain, tex)


class Fm(str):
    """A formula inside a sentence. The string is its plain text; `.tex` is its TeX."""

    tex: str

    def __new__(cls, plain: str, tex: str | None = None):
        obj = super().__new__(cls, plain)
        obj.tex = plain if tex is None else tex
        return obj


def say(*parts) -> tuple[str, str]:
    """``(text, latex)`` for a mix of prose strings and `Fm` formulas."""
    text: list[str] = []
    latex: list[str] = []
    for part in parts:
        if isinstance(part, Fm):
            text.append(str(part))
            latex.append("{" + part.tex + "}")
        elif part:
            text.append(part)
            latex.append(r"\text{" + _escape_prose(part) + "}")
    return "".join(text), "".join(latex)


_SUPERSCRIPT = {"2": "²", "3": "³"}


def plain_expr(expr) -> str:
    """A SymPy expression as the learner reads it: ``3*x**2*y - 2*y`` becomes ``3x²y - 2y``."""
    s = sympy.sstr(expr)
    s = re.sub(r"\*\*\(?(-?\d+)\)?", lambda m: _SUPERSCRIPT.get(m.group(1), "^" + m.group(1)), s)
    s = s.replace("sqrt(", "√(")
    return s.replace("*", "")


def tex_expr(expr) -> str:
    return sympy.latex(expr)


def fm_expr(expr) -> Fm:
    return Fm(plain_expr(expr), tex_expr(expr))


def typed_expr(expr) -> str:
    """An expression as a learner types it (and as the expression grader reads it): ``3*x**2*y`` becomes ``3x^2y``."""
    s = sympy.sstr(expr).replace("**", "^")
    return re.sub(r"\*", "", s) if "/" not in s else s.replace("*", "")


# ---------------------------------------------------------------------------
# Numbers


def to_fraction(value) -> Fraction:
    """An exact Fraction for an int, Fraction, SymPy Rational or float (a float as the decimal it prints as)."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if getattr(value, "is_Rational", False):
        return Fraction(int(value.p), int(value.q))
    return Fraction(repr(float(value)))


def dp_text(value, dp: int) -> str:
    """`value` rounded half up to `dp` decimal places, written with exactly `dp` places (an integer when dp is 0)."""
    rounded = round_half_up(to_fraction(value), dp)
    if dp == 0:
        return str(int(rounded))
    return stated_precision_string(rounded, dp)


def places_phrase(dp: int) -> str:
    return f"{dp} decimal place{'s' if dp != 1 else ''}"


def clear_of_tie(value, dp: int, margin: Fraction = Fraction(1, 10)) -> bool:
    """True when `value` is not within `margin` (in units of the last place) of a rounding tie at `dp` places, so a
    learner who rounds an intermediate value still lands on the same last digit."""
    scaled = abs(to_fraction(value)) * 10**dp
    frac = scaled - math.floor(scaled)
    return abs(frac - Fraction(1, 2)) > margin


class Redraw(VerificationError):
    """This draw is unusable (see `build_until`); the generator should try another."""


class TooFewDistractors(Redraw):
    """A draw whose wrong answers collapse onto the right one or onto each other."""


def build_until(rng: random.Random, build, what: str, tries: int = 300):
    """`build(rng)`, redrawn while it raises `Redraw`. Any other error is a real failure and propagates."""
    for _ in attempts(tries, what):
        try:
            return build(rng)
        except Redraw:
            continue
    raise AssertionError("unreachable")  # pragma: no cover


def ft_close(follow_value, answer_text: str, dp: int) -> None:
    """Redraw unless a learner's follow-through value (worked from the earlier parts' ROUNDED answers) rounds to
    within one unit of the last place of the part's own answer: `make()` demands exactly that."""
    ours = Fraction(answer_text)
    theirs = Fraction(dp_text(follow_value, dp))
    if abs(ours - theirs) > (Fraction(1, 10**dp) if dp else 0):
        raise Redraw(f"follow-through {theirs} is too far from {ours}")


def num_dist(value, dp: int, tag: str) -> Distractor:
    text = dp_text(value, dp)
    return Distractor(text, tag, int(text) if dp == 0 else float(text))


def frac_dist(value, tag: str) -> Distractor:
    """A wrong answer to a whole-number question that may itself be a fraction (what a learner types: 7/2)."""
    f = to_fraction(value)
    if f.denominator == 1:
        return Distractor(str(f.numerator), tag, f.numerator)
    return Distractor(f"{f.numerator}/{f.denominator}", tag, float(f))


def pick_nums(answer_text: str, cands: list[Distractor], dp: int, n: int = 4) -> list[Distractor]:
    """Distractors that clear the typed number grader: at least a full unit in the last place from the answer."""
    tolerance = Fraction(1, 10**dp) if dp else Fraction(0)
    return pick_distractors(answer_text, cands, n, tolerance=tolerance)


def number_part(
    prompt,
    value,
    dp: int,
    cands,
    *,
    marks: int = 1,
    working=(),
    follow_through: str | None = None,
    sensible_range=None,
    prompt_latex: str | None = None,
    need: int = 2,
    n: int = 4,
) -> dict:
    """A `number` part (or the fields of a single-answer number question, without the prompt) for `value`."""
    answer_text = dp_text(value, dp)
    distractors = pick_nums(answer_text, cands, dp, n)
    if len(distractors) < need:
        raise TooFewDistractors(f"too few distractors for {prompt!r}: {[str(d) for d in distractors]}")
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=int(answer_text) if dp == 0 else float(answer_text),
        answer_text=answer_text,
        marks=marks,
        working=list(working),
        distractors=distractors,
    )
    if prompt_latex:
        part["prompt_latex"] = prompt_latex
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part


def number_question(text, latex, value, dp, cands, *, working=(), n: int = 4) -> dict:
    """A single-answer number question."""
    part = number_part("", value, dp, cands, working=working, n=n)
    return dict(
        text=text,
        latex=latex,
        answer=part["answer"],
        answer_text=part["answer_text"],
        answer_kind="number",
        working=part["working"],
        distractors=part["distractors"],
    )


def fraction_text_value(value) -> tuple[dict | int, str]:
    f = to_fraction(value)
    exact = f.numerator if f.denominator == 1 else {"p": f.numerator, "q": f.denominator}
    return exact, rational_text(sympy.Rational(f.numerator, f.denominator))


def choice_part(
    rng: random.Random,
    intro: str,
    right: str,
    wrongs: list[tuple[str, str]],
    *,
    marks: int = 1,
    working=(),
) -> dict:
    """A one-mark conclusion as a numbered list of statements; the reply is the number of the correct statement.

    It is a typed `number` part like any other (so graded by the number grader); the position of the correct
    statement is shuffled with `rng`. `wrongs` is a list of (statement, misconception id)."""
    order = [right] + [text for text, _tag in wrongs]
    check(len(set(order)) == len(order), "two statements are the same")
    rng.shuffle(order)
    position = {text: i + 1 for i, text in enumerate(order)}
    listed = " ".join(f"{i + 1}. {text}" for i, text in enumerate(order))
    return dict(
        prompt=f"{intro} Enter the number of the correct statement. {listed}",
        answer_kind="number",
        answer=position[right],
        answer_text=str(position[right]),
        marks=marks,
        working=[f"The correct statement is number {position[right]}: {right}", *working],
        distractors=[Distractor(str(position[text]), tag, position[text]) for text, tag in wrongs],
    )


def exact_rational(value) -> sympy.Rational:
    f = to_fraction(value)
    return sympy.Rational(f.numerator, f.denominator)


def latex_plain(text: str) -> str:
    """`latex_answer` for a typed answer string."""
    return latex_answer(text)
