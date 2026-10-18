"""Shared building blocks for the second-year `w6c_*` generators (singular values, cofactors, Hermitian matrices,
multivariable Taylor polynomials, quadrics, dynamic programming, inventory and coding theory).

Not a generator module (the leading underscore keeps the package loader from importing it as one). Every answer in
these skills is exact: an integer (`number`), a lowest-terms fraction (`fraction`), a vector (`matrix`) or a choice
(`mcq`), and every wrong option is built from a named error.

* ``value_fields`` / ``value_part``: the answer attributes of a single value, and the same as an exam part, with
  distractors filtered to the key's own form (a whole-number key is offered whole-number wrong values only, a fraction
  key fractions only), so the key is never the only option of its shape;
* ``vector_fields`` / ``vector_part``: a vector answer graded as any non-zero multiple (``scalar_multiple``) or exactly;
* ``choice_part``: a multiple-choice exam part with at least four options;
* ``redrawn``: a decorator that redraws a rare draw that cannot be verified or given enough distractors.
"""

from __future__ import annotations

import functools
import re as _re
from fractions import Fraction

import sympy as _sympy

from ..answers import Distractor, exact_value
from ..core import VerificationError, rational_text
from ._ucv1_linalg import column_value, primitive, vec_str, vector_distractors

__all__ = [
    "Math",
    "Redraw",
    "compose",
    "math_of",
    "pieces_of",
    "plain_expr",
    "choice_part",
    "mat_text",
    "redrawn",
    "value_fields",
    "value_part",
    "value_text",
    "vector_fields",
    "vector_part",
]


class Redraw(Exception):
    """A draw that cannot be given enough distinct wrong options; the decorator draws again."""


def redrawn(build):
    """Decorator for a generator function: a draw that raises ``Redraw`` or ``VerificationError`` is redrawn from the
    same random stream (at most eight times), so a seed still gives one deterministic question."""

    @functools.wraps(build)
    def inner(rng, *args, **kwargs):
        failure = None
        for _ in range(8):
            try:
                return build(rng, *args, **kwargs)
            except (Redraw, VerificationError) as exc:
                failure = exc
        raise VerificationError(f"no acceptable draw: {failure}")

    return inner


def value_text(value) -> str:
    """An int or `Fraction` as the learner writes it: ``3``, ``-3/4``."""
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else rational_text(value)


def _kind(value) -> str:
    return "number" if Fraction(value).denominator == 1 else "fraction"


def _wrong_values(correct, wrong, n, minimum):
    """Tagged wrong values, in priority order, that are in the key's own form, distinct, and not the key."""
    correct = Fraction(correct)
    kind = _kind(correct)
    seen = {correct}
    out = []
    for tag, value in wrong:
        if value is None:
            continue
        value = Fraction(value)
        if value in seen or _kind(value) != kind:
            continue
        seen.add(value)
        out.append(Distractor(value_text(value), tag, exact_value(value)))
        if len(out) == n:
            break
    if len(out) < minimum:
        raise Redraw(f"only {len(out)} wrong options for {value_text(correct)}")
    return out


def value_fields(value, wrong, *, n: int = 4, minimum: int = 3) -> dict:
    """Answer attributes for a single exact value. ``wrong`` is a list of ``(misconception id, wrong value)``."""
    value = Fraction(value)
    text = value_text(value)
    return dict(
        answer_kind=_kind(value),
        answer=exact_value(value),
        answer_text=text,
        answer_latex=text,
        distractors=_wrong_values(value, wrong, n, minimum),
    )


def value_part(prompt, value, marks, working, wrong, *, follow_through=None, sensible_range=None, n=3, minimum=2):
    """An exam part whose key is the exact value ``value``."""
    part = dict(prompt=prompt, marks=marks, working=working, **value_fields(value, wrong, n=n, minimum=minimum))
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part


def mat_text(rows) -> str:
    """A matrix of ints, `Fraction`s or strings as the stem spells it: ``[[1, 2], [3, 4]]``."""
    return (
        "["
        + ", ".join("[" + ", ".join(value_text(v) if not isinstance(v, str) else v for v in row) + "]" for row in rows)
        + "]"
    )


def vector_fields(vec, wrong, *, scalar_multiple: bool = True, n: int = 4, minimum: int = 3) -> dict:
    """Answer attributes for a vector key. ``wrong`` is a list of ``(vector, misconception id)``.

    With ``scalar_multiple`` the key is stored in primitive form and any non-zero multiple is right, so every wrong
    vector is compared in primitive form too (a multiple of a wrong vector is the same wrong answer). Otherwise the
    vector is graded entry for entry."""
    if scalar_multiple:
        vec = primitive(vec)
        distractors = vector_distractors(wrong, vec)[:n]
    else:
        distractors = []
        seen = {tuple(Fraction(v) for v in vec)}
        for cand, tag in wrong:
            if cand is None:
                continue
            key = tuple(Fraction(v) for v in cand)
            if key in seen:
                continue
            seen.add(key)
            distractors.append(Distractor(vec_str(cand), tag, column_value(cand)))
            if len(distractors) == n:
                break
    if len(distractors) < minimum:
        raise Redraw("too few wrong vectors")
    text = vec_str(vec)
    fields = dict(
        answer_kind="matrix",
        answer=column_value(vec),
        answer_text=text,
        answer_latex="plain:" + text,
        distractors=distractors,
    )
    if scalar_multiple:
        fields["answer_form"] = "scalar_multiple"
    return fields


def vector_part(prompt, vec, marks, working, wrong, *, scalar_multiple=True, n=3, minimum=2):
    return dict(
        prompt=prompt,
        marks=marks,
        working=working,
        **vector_fields(vec, wrong, scalar_multiple=scalar_multiple, n=n, minimum=minimum),
    )


def choice_part(prompt, correct, marks, working, wrong, *, minimum: int = 3):
    """A multiple-choice exam part: ``wrong`` is a list of ``(misconception id, option text)``; at least ``minimum``
    different wrong options are needed so there are four or more options."""
    kept = []
    for tag, text in wrong:
        if text is None or text == correct or any(text == t for _m, t in kept):
            continue
        kept.append((tag, text))
    if len(kept) < minimum:
        raise Redraw("too few wrong options")
    return dict(
        prompt=prompt,
        marks=marks,
        working=working,
        answer_kind="mcq",
        answer=correct,
        answer_text=correct,
        distractors=[Distractor(t, m, t) for m, t in kept],
    )


# -- prompts that mix prose and mathematics ------------------------------------------------------------------------

_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


class Math(str):
    """A piece of mathematics in a prompt: the plain text is the string itself and ``tex`` is its LaTeX."""

    tex: str

    def __new__(cls, plain, tex=None):
        self = super().__new__(cls, plain)
        self.tex = tex if tex is not None else str(plain)
        return self


def plain_expr(expr) -> str:
    """A SymPy expression as plain text a learner reads: ``x³y − 2xy²``, ``x e^(xy − 2)``, ``√(x² + y²)``."""
    text = _sympy.sstr(expr, order="grlex")
    text = text.replace("*exp(", " e^(").replace("exp(", "e^(")
    for fn, shown in (("sqrt", "√"), ("log", "ln"), ("sin", "sin"), ("cos", "cos")):
        text = text.replace(f"*{fn}(", f" {shown}(").replace(f"{fn}(", f"{shown}(")
    text = text.replace("**", "^").replace("*", "")
    text = _re.sub(r"\^(\d)(?!\d)", lambda m: m.group(1).translate(_SUPERSCRIPT), text)
    return text.replace(" - ", " − ")


def math_of(expr) -> Math:
    return Math(plain_expr(expr), _sympy.latex(expr))


def pieces_of(template, **values):
    """The pieces of a prompt template: its ``{name}`` slots are filled from ``values``, and a `Math` value stays a
    `Math` piece so `compose` can give it LaTeX. Anything else is spliced into the prose."""
    out = []
    for chunk in _re.split(r"(\{\w+\})", template):
        if chunk.startswith("{") and chunk.endswith("}"):
            value = values[chunk[1:-1]]
            out.append(value if isinstance(value, Math) else str(value))
        elif chunk:
            out.append(chunk)
    return out


def compose(*pieces):
    """``(text, latex)`` of a prompt made of prose (plain strings) and `Math` pieces."""
    text, latex = [], []
    for piece in pieces:
        if isinstance(piece, Math):
            text.append(str(piece))
            latex.append(piece.tex)
        else:
            text.append(piece)
            latex.append(r"\text{" + piece.replace("\\", "").replace("%", r"\%").replace("&", r"\&") + "}")
    return "".join(text), "".join(latex)
