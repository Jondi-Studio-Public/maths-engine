"""Shared builders for the `r3f_*` continuous-random-variable, linear-combination and inference skills.

A part is a dict for `mathgen.core.Part`. `number_part` states its answer at a fixed number of decimal places and keeps
every distractor at that same number of places; `fraction_part` states an exact rational; `expression_part` states a
formula in x; `mcq_part` offers a stated key among stated wrong options. Each takes `slips`: a list of
`(value, misconception id)` pairs in order of likelihood, from which up to three distinct wrong options are kept. A
slip is dropped when it repeats the key, repeats an earlier slip, or is accepted by the real grader for the part, so
every option a learner sees is marked wrong. `single` turns a part into a one-part question.

The module also holds the normal-distribution arithmetic the skills share (`phi`, `normal_prob`) and small helpers for
choosing draws whose rounded answers are stable.
"""

from __future__ import annotations

import functools
import math
import random
import re
from fractions import Fraction
from types import SimpleNamespace

import sympy

from ..answers import Distractor, exact_value
from ..core import (
    VerificationError,
    attempts,
    check,
    latex_answer,
    numeric_answer,
    rational_text,
    round_half_up,
    stated_precision_string,
)
from ..grading import GRADERS

X = sympy.Symbol("x")

_NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:/\d+)?(?![\w.])")


def number_text(value, places: int) -> str:
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


def build_until(rng: random.Random, build, tries: int = 300):
    """Call `build(rng)` until it returns without `ThinOptions`; the draw that failed is simply discarded."""
    for _ in range(tries):
        try:
            return build(rng)
        except ThinOptions:
            continue
    raise VerificationError(f"no draw with two wrong options in {tries} tries")


def _finish(kind, prompt, answer, text, marks, working, options, follow_through, sensible_range, latex=None):
    if len(options) < 2:
        raise ThinOptions(f"fewer than two distractors for {prompt!r}")
    part = dict(
        prompt=prompt,
        answer_kind=kind,
        answer=answer,
        answer_text=text,
        answer_latex=latex if latex is not None else latex_answer(text),
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


@functools.lru_cache(maxsize=4096)
def _formula_text(expr) -> str:
    return str(sympy.factor(sympy.nsimplify(expr))).replace("**", "^").replace("*", "")


def formula_text(expr) -> str:
    """A SymPy expression in x written the way a learner would type it, factored where it factors: `x^3/27`,
    `(x - 1)^2(x + 3)/16`."""
    return _formula_text(sympy.sympify(expr))


def expression_part(prompt, expr, marks, working, slips, *, follow_through=None, keep=3):
    """A part answered by a formula in x. `expr` is the key; each slip is `(expression, misconception id)`."""
    expr = sympy.nsimplify(expr)
    text = formula_text(expr)
    answer = str(expr)
    options: list[Distractor] = []
    seen = {text}
    for item in slips:
        if item is None or item[0] is None:
            continue
        slip, tag = item
        slip = sympy.nsimplify(slip)
        shown = formula_text(slip)
        if shown in seen:
            continue
        if expr.is_polynomial(X) and slip.is_polynomial(X):
            # Two polynomials in x with rational coefficients are the same function exactly when their expanded
            # coefficients agree, and the real grader accepts nothing else, so no grader call is needed here; the
            # tests run the real grader on every option.
            if sympy.Poly(sympy.expand(slip - expr), X).is_zero:
                continue
        elif sympy.simplify(slip - expr) == 0 or _accepted("expression", prompt, answer, text, shown):
            continue
        seen.add(shown)
        options.append(Distractor(shown, tag))
        if len(options) == keep:
            break
    return _finish(
        "expression", prompt, answer, text, marks, working, options, follow_through, None, latex=sympy.latex(expr)
    )


def mcq_part(prompt, key, marks, working, wrongs, *, keep=3):
    """A multiple-choice part. `wrongs` is a list of `(text, misconception id)`; at least two distinct survive."""
    options: list[Distractor] = []
    seen = {key}
    for shown, tag in wrongs:
        if shown in seen:
            continue
        seen.add(shown)
        options.append(Distractor(shown, tag))
        if len(options) == keep:
            break
    return _finish("mcq", prompt, key, key, marks, working, options, None, None, latex="plain:" + key)


def clashes(keys, *texts: str) -> bool:
    """True when any key of two or more characters appears as a whole number in the given texts."""
    shown = {m for t in texts for m in _NUMBER_RE.findall(t)}
    return any(len(str(k)) >= 2 and str(k) in shown for k in keys)


def single(text: str, part: dict) -> dict:
    """A one-part question: `text` is the stem and `part` supplies the key and the options."""
    return dict(
        text=text,
        answer=part["answer"],
        answer_text=part["answer_text"],
        answer_latex=part["answer_latex"],
        answer_kind=part["answer_kind"],
        working=part["working"],
        distractors=part["distractors"],
    )


def single_expression(text: str, part: dict) -> dict:
    """A one-part question for a skill graded as `expression`: a rational or rounded decimal key is then written as the
    text a learner types, and its options carry no structured value."""
    return dict(
        text=text,
        answer=part["answer_text"],
        answer_text=part["answer_text"],
        answer_latex=part["answer_latex"],
        answer_kind="expression",
        working=part["working"],
        distractors=[Distractor(d.text, d.misconception) for d in part["distractors"]],
    )


def clear_of_boundary(value, places: int, margin: Fraction = Fraction(1, 10)) -> bool:
    """True when `value` is not within `margin` of a rounding boundary at `places` places."""
    scaled = abs(Fraction(value)) * 10**places
    frac = scaled - math.floor(scaled)
    return abs(frac - Fraction(1, 2)) > margin


def dec(value) -> str:
    """A number as a short decimal, `0.35`, `12`, `7.5`."""
    return f"{float(value):g}"


# -- the normal distribution -----------------------------------------------------------------------------------


def phi(z) -> Fraction:
    """P(Z <= z) for a standard normal Z, as a Fraction of the nearest double."""
    return Fraction(repr(0.5 * math.erfc(-float(z) / math.sqrt(2))))


def normal_prob(lo, hi, mean, sd) -> Fraction:
    """P(lo < X < hi) for X ~ N(mean, sd^2); `None` for an unbounded end."""
    upper = Fraction(1) if hi is None else phi((Fraction(hi) - Fraction(mean)) / Fraction(sd))
    lower = Fraction(0) if lo is None else phi((Fraction(lo) - Fraction(mean)) / Fraction(sd))
    return upper - lower


def sqrt_fraction(value) -> Fraction:
    """The square root of a non-negative number, to about 15 significant figures."""
    return Fraction(repr(math.sqrt(Fraction(value))))


#: Pythagorean triples (a, b, c): a, b and c integers with a^2 + b^2 = c^2. Standard deviations drawn from one keep
#: the standard deviation of a sum or a difference exact, so a learner's rounded part answers cannot drift.
TRIPLES = ((3, 4, 5), (6, 8, 10), (5, 12, 13), (9, 12, 15), (8, 15, 17), (12, 16, 20), (15, 20, 25))


def pick_one(rng: random.Random, options):
    return options[rng.randrange(len(options))]


__all__ = [
    "X",
    "ThinOptions",
    "attempts",
    "build_until",
    "check",
    "clashes",
    "clear_of_boundary",
    "dec",
    "expression_part",
    "formula_text",
    "fraction_part",
    "mcq_part",
    "normal_prob",
    "number_part",
    "number_text",
    "phi",
    "pick_one",
    "single",
    "single_expression",
    "sqrt_fraction",
    "TRIPLES",
]
