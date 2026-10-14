"""Small helpers shared by the `u2_*` first-year university builds: tagged distractors and answer dictionaries."""

from __future__ import annotations

import functools
import re
from fractions import Fraction

import sympy

from ..answers import Distractor, exact_value
from ..core import VerificationError, check, pick_distractors, rational_text


class Redraw(VerificationError):
    """The question just built cannot be offered (too few usable distractors, or the key looks unlike every
    distractor): `redraws` builds another from the same random stream."""


def redraws(build, tries=40):
    """Decorate a generator so that a `Redraw` from deep inside it simply draws a fresh question."""

    @functools.wraps(build)
    def wrapper(rng, *args, **kw):
        last = None
        for _ in range(tries):
            try:
                return build(rng, *args, **kw)
            except Redraw as error:
                last = error
        raise last

    return wrapper


def int_d(value, tag):
    """A whole-number distractor."""
    return Distractor(str(value), tag, value)


def frac_d(value, tag):
    """A rational distractor, spelled `a/b` when it is not whole."""
    value = Fraction(value)
    return Distractor(rational_text(value), tag, exact_value(value))


def expr_d(text, expr, tag):
    """An expression distractor: `text` is what the learner sees, `expr` the SymPy value it stands for."""
    return Distractor(text, tag, sympy.sstr(sympy.sympify(expr)))


_SHAPES = (
    ("integer", re.compile(r"^-?\d+$")),
    ("decimal", re.compile(r"^-?\d+\.\d+$")),
    ("fraction", re.compile(r"^-?\d+/\d+$")),
)


def _shape(text):
    for name, pattern in _SHAPES:
        if pattern.match(str(text).strip()):
            return name
    return None


def _text_of(option):
    return getattr(option, "text", option)


def _with_a_matching_shape(answer_text, picked, everything):
    """Make sure some option looks like the key (integer, decimal or fraction), so the key cannot be picked by its look."""
    key_shape = _shape(answer_text)
    if key_shape is None or len(picked) < 2 or any(_shape(_text_of(d)) == key_shape for d in picked):
        return picked
    if any(_shape(_text_of(d)) is None for d in picked):
        return picked
    for candidate in everything:
        if _shape(_text_of(candidate)) == key_shape and all(_text_of(candidate) != _text_of(d) for d in picked):
            return [*picked[:-1], candidate]
    if len(picked) >= 2:
        raise Redraw(f"no distractor looks like the key {answer_text!r}")
    return picked


def keep(answer_text, candidates, n=3, minimum=2, **kw):
    """`pick_distractors`, preferring options that are not negative and insisting that at least `minimum` survive.

    When the key is a whole number, decimal or fraction, at least one option of the same kind is kept if any exists.
    """
    answer_text = str(answer_text)
    everything = pick_distractors(answer_text, candidates, max(len(candidates), n), **kw)
    if not answer_text.startswith("-"):
        positive = [c for c in candidates if not str(c).startswith("-")]
        picked = pick_distractors(answer_text, positive, n, **kw)
        if len(picked) >= minimum:
            return _with_a_matching_shape(answer_text, picked, everything)
    picked = pick_distractors(answer_text, candidates, n, **kw)
    if len(picked) < minimum:
        raise Redraw(f"too few distractors for {answer_text!r}")
    return _with_a_matching_shape(answer_text, picked, everything)


def number_q(text, value, working, distractors, places=0):
    """A question dict whose answer is a number (an int unless `places` > 0)."""
    shown = str(value) if not places else f"{value:.{places}f}"
    return dict(
        text=text,
        answer_kind="number",
        answer=value,
        answer_text=shown,
        answer_latex=shown,
        working=working,
        distractors=distractors,
    )


def number_part(prompt, value, marks, working, distractors, follow_through=None, **extra):
    """A part whose answer is a whole number."""
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=value,
        answer_text=str(value),
        marks=marks,
        working=working,
        distractors=distractors,
    )
    if follow_through:
        part["follow_through"] = follow_through
    part.update(extra)
    return part


def lin(coef, const, var="k") -> str:
    """coef*var + const written with proper signs: 2k + 1, -k - 3, 5k, 4."""
    coef, const = Fraction(coef), Fraction(const)
    parts = ""
    if coef != 0:
        mag = abs(coef)
        body = var if mag == 1 else f"{rational_text(mag)}{var}"
        parts = ("-" if coef < 0 else "") + body
    if const != 0:
        if not parts:
            return rational_text(const)
        parts += (" - " if const < 0 else " + ") + rational_text(abs(const))
    return parts or "0"
