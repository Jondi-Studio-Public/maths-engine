"""Shared plumbing for the multi-step modelling and algorithm skills (generator prefix `fr2e_`).

Not a generator (the leading underscore keeps the generators package from registering it). It re-exports the
part builders the exam chains already share and adds the two things those skills need that they do not have: a
multiple-choice part whose options are tagged `Distractor`s, and a way to turn a finished part into a
single-answer question so every tier is built with the same checked code.
"""

from __future__ import annotations

from fractions import Fraction

from ..answers import Distractor
from ..core import check, latex_answer
from . import exd_x_r2b_support as _support
from .exd_x_r2b_support import Redraw, as_fraction, draw, expression_part, fraction_part, number_part, words_part

__all__ = [
    "Redraw",
    "as_fraction",
    "draw",
    "exact_number_part",
    "expression_part",
    "fraction_part",
    "number_part",
    "words_part",
    "mcq_part",
    "single",
    "dollars",
    "letter_order",
]


def exact_number_part(prompt, value, marks, working, wrong, **kw) -> dict:
    """`exd_x_r2b_support.exact_number_part` with a clean-up of the wrong answers: a `None` (an error that does not
    apply to these numbers) and, unless `signed`, a negative value (an implausible slip for money) are dropped."""
    signed = kw.pop("signed", False)
    kept = [(v, tag) for v, tag in wrong if v is not None and (signed or v >= 0)]
    kw.setdefault("max_places", 2)
    part = _support.exact_number_part(prompt, value, marks, working, kept, **kw)
    if len(part["distractors"]) < 3:
        raise Redraw("fewer than three distinct wrong answers")
    return part


def dollars(value, cents: bool = False) -> str:
    """Whole dollars as text, `$12`; cents as `$12.50`. With `cents`, always two places (`$12.00`)."""
    value = Fraction(value)
    if value.denominator == 1 and not cents:
        return f"${value.numerator}"
    check((value * 100).denominator == 1, "a price must be a whole number of cents")
    return f"${float(value):.2f}"


def mcq_part(prompt, key, wrong, marks, working, *, wanted=3, answer_latex=None) -> dict:
    """A multiple-choice part. `key` is the one true option text and `wrong` is a list of `(text, misconception id)`.

    Repeats of the key or of an earlier wrong option are dropped; fewer than `wanted` distinct wrong options
    redraws, so every question offers at least four options."""
    options, seen = [], {key}
    for text, tag in wrong:
        if text in seen:
            continue
        seen.add(text)
        options.append(Distractor(text, tag))
        if len(options) == wanted + 1:
            break
    if len(options) < wanted:
        raise Redraw("fewer than three distinct wrong options for a multiple-choice part")
    return dict(
        prompt=prompt,
        answer_kind="mcq",
        answer=key,
        answer_text=key,
        answer_latex=answer_latex or f"plain:{key}",
        marks=marks,
        working=working,
        distractors=options,
    )


def single(stem: str, part: dict) -> dict:
    """The single-answer question for a finished part: the stem followed by the part's prompt."""
    out = {
        "text": f"{stem} {part['prompt']}".strip(),
        "answer": part["answer"],
        "answer_text": part["answer_text"],
        "answer_kind": part["answer_kind"],
        "answer_latex": part.get("answer_latex") or latex_answer(part["answer_text"]),
        "working": part["working"],
        "distractors": part["distractors"],
    }
    if part["answer_kind"] == "mcq":
        out["latex"] = f"plain:{out['text']}"
    return out


def letter_order(rng, steps: list[str]) -> tuple[list[str], list[str]]:
    """Shuffle `steps` (given in their correct order) under the letters A, B, C, ...

    Returns `(lines, key)`: the lettered lines in shuffled order, and the letters that put the steps back in order.
    The shuffle is redrawn until it is not already the correct order."""
    n = len(steps)
    letters = [chr(ord("A") + i) for i in range(n)]
    for _ in range(50):
        order = list(range(n))
        rng.shuffle(order)
        if order != list(range(n)):
            break
    else:
        order = list(reversed(range(n)))
    # order[i] is the index (in the correct order) of the step shown under letter i
    lines = [f"{letters[i]}. {steps[order[i]]}" for i in range(n)]
    key = [letters[order.index(j)] for j in range(n)]
    return lines, key


def rule_text(rule) -> str:
    """A rule in x as plain text with no `*` or `**`: polynomials as 2x^2 + 3x + 1, exponentials as 3 × 2^x."""
    import sympy

    x = sympy.Symbol("x")
    if rule.is_polynomial(x):
        return sympy.sstr(rule, order="lex").replace("**", "^").replace("*x", "x")
    return sympy.sstr(rule).replace("**", "^").replace("*", " × ")
