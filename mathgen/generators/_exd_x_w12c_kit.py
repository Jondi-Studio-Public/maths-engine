"""Small shared builders for the multi-part exam generators in the `exd_x_w12c_*` modules.

Each builder returns a part dict for ``parts=[...]`` (see ``docs/design/multi-part-questions.md``). Candidates are
``(wrong value or text, misconception id)`` pairs in priority order; a candidate is kept only when the real grader
would reject it, so every distractor is a different answer from the key by construction.
"""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from ..answers import Distractor, exact_value
from ..core import VerificationError, attempts, check, latex_answer, pick_distractors, rational_text
from ..grading import GRADERS
from ._exd_x_sc4_kit import num_part

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_X = sympy.Symbol("x")


def int_part(prompt, value, marks, working, cands, *, lo, hi, follow_through=None):
    """A whole-number ``number`` part. Wrong values are written as whole numbers too, so the format never gives the
    key away. ``(lo, hi)`` is the sensible range of a reply."""
    part = num_part(
        prompt, value, 0, marks, working, cands, follow_through=follow_through, signed=True, key_places=True
    )
    part["sensible_range"] = (lo, hi)
    return part


def fraction_part(prompt, value, marks, working, cands, *, follow_through=None):
    """A ``fraction`` part (a whole number is a fraction with denominator 1); fractions are compared by value."""
    value = Fraction(value)
    text = rational_text(value)
    options = [Distractor(rational_text(Fraction(w)), tag, exact_value(Fraction(w))) for w, tag in cands]
    picked = pick_distractors(text, options)
    check(len(picked) >= 2, "fewer than two usable distractors")
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer=exact_value(value),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=picked,
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


def parse_text(text: str):
    """A student-style expression such as ``-(x - 2)^2 + 5`` or ``2x/3 - 1`` as a SymPy expression in x."""
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMS, local_dict={"x": _X})


def expression_part(prompt, text, marks, working, cands, *, follow_through=None):
    """An ``expression`` part whose key is `text`. A candidate equal to the key as an expression is dropped."""
    key = parse_text(text)
    options, seen = [], {sympy.expand(key)}
    for wrong, tag in cands:
        if not wrong:
            continue
        value = sympy.expand(parse_text(wrong))
        if value in seen:
            continue
        seen.add(value)
        options.append(Distractor(wrong, tag, None))
    picked = options[:3]
    check(len(picked) >= 2, "fewer than two usable distractors")
    part = dict(
        prompt=prompt,
        answer_kind="expression",
        answer=sympy.sstr(key),
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=picked,
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


def _accepts(kind: str, answer_text: str, answer, reply: str) -> bool:
    question = SimpleNamespace(answer=answer, answer_text=answer_text, answer_form=None, text="", parts=None)
    return bool(GRADERS[kind](question, reply).correct)


def inequality_part(prompt, text, marks, working, cands):
    """An ``inequality`` part whose key is `text` (for example ``-3 ≤ x < 9``). A candidate the real inequality
    grader accepts is dropped."""
    options, seen = [], {text}
    for wrong, tag in cands:
        if wrong in seen or _accepts("inequality", text, text, wrong):
            continue
        seen.add(wrong)
        options.append(Distractor(wrong, tag, wrong))
    picked = options[:3]
    check(len(picked) >= 2, "fewer than two usable distractors")
    return dict(
        prompt=prompt,
        answer_kind="inequality",
        answer=text,
        answer_text=text,
        answer_latex=latex_answer(text, plain=True),
        marks=marks,
        working=working,
        distractors=picked,
    )


def set_part(prompt, values, marks, working, cands):
    """A ``set`` part of numbers, written ``{a, b}``. A candidate holding the same numbers is dropped."""
    key = sorted(values)
    text = "{" + ", ".join(str(v) for v in key) + "}"
    options, seen = [], {tuple(key)}
    for wrong, tag in cands:
        w = tuple(sorted(wrong))
        if w in seen:
            continue
        seen.add(w)
        options.append(Distractor("{" + ", ".join(str(v) for v in w) + "}", tag, list(w)))
    picked = options[:3]
    check(len(picked) >= 2, "fewer than two usable distractors")
    return dict(
        prompt=prompt,
        answer_kind="set",
        answer=list(key),
        answer_text=text,
        answer_latex=latex_answer(text, plain=True),
        marks=marks,
        working=working,
        distractors=picked,
    )


def redraw(rng, build, tries: int = 80):
    """``build(rng)`` again, with the rng where the last attempt left it, whenever a draw left a part with fewer than
    two usable distractors. Any other failed check (a disagreement between two routes to a key) still raises."""
    for _ in attempts(tries, "a draw whose parts each have two usable distractors"):
        try:
            return build(rng)
        except VerificationError as err:
            if "usable distractors" not in str(err):
                raise
