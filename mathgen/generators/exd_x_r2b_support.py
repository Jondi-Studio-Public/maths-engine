"""Part builders shared by the `exd_x_r2b_*` multi-part exam generators.

Not a generator itself. Each builder returns the dict for one part of a multi-part question
(`docs/design/multi-part-questions.md`), with its distractors tagged by `Distractor` objects whose tag is an id in
`mathgen.misconceptions`. A wrong answer is given as `(value, tag)`: the builder drops any that equals the key (or
repeats another) so that every option offered is a different, wrong answer, and keeps the first three.
"""

from __future__ import annotations

import re
from fractions import Fraction

import sympy

from ..answers import Distractor, exact_value
from ..core import (
    GRADER_TOLERANCE,
    VerificationError,
    check,
    latex_answer,
    rational_text,
    round_half_up,
    stated_precision_string,
)

_WANTED = 3


class Redraw(VerificationError):
    """A draw that cannot give a sound question (too few distinct wrong answers, a degenerate value): draw again."""


def draw(build, rng, tries: int = 200):
    """Call `build(rng)` until it does not raise `Redraw`; raise `VerificationError` after `tries` attempts."""
    for _ in range(tries):
        try:
            return build(rng)
        except Redraw:
            continue
    raise VerificationError(f"no acceptable draw in {tries} tries")


_NUMBER_TIMES = re.compile(r"(?<=\d)\*(?=[A-Za-z(])|(?<=\))\*(?=[A-Za-z(])")


def plain(expr) -> str:
    """Learner-facing text of a SymPy expression: `^` for powers and no `*` between a number and a symbol."""
    return typed(expr)


def typed(expr) -> str:
    """Text a learner would type for an expression: highest power first, `^` for powers, and no `*` between a
    number and a letter or bracket (`-2x + 11`, `3(x - 1)`)."""
    text = sympy.sstr(expr, order="lex").replace("**", "^")
    return _NUMBER_TIMES.sub("", text)


def _first_wrong(key_text: str, candidates, same, make_option) -> list:
    options, seen = [], {key_text}
    for value, tag in candidates:
        if value is None:
            continue
        option = make_option(value, tag)
        if option.text in seen or same(value):
            continue
        seen.add(option.text)
        options.append(option)
        if len(options) == _WANTED:
            break
    if len(options) < 2:
        raise Redraw("fewer than two distinct wrong answers for a part")
    return options


def fraction_part(
    prompt, value, marks, working, wrong, *, follow_through=None, sensible_range=None, tolerance=Fraction(0)
) -> dict:
    """A part whose answer is the exact rational `value`, graded as a `fraction`."""
    value = Fraction(sympy.Rational(value).p, sympy.Rational(value).q)
    key_text = rational_text(value)

    def build(v, tag):
        v = Fraction(sympy.Rational(v).p, sympy.Rational(v).q)
        return Distractor(rational_text(v), tag, exact_value(v))

    def same(v):
        v = Fraction(sympy.Rational(v).p, sympy.Rational(v).q)
        return abs(v - value) <= tolerance

    options = _first_wrong(key_text, wrong, same, build)
    return _finish(
        dict(
            prompt=prompt,
            answer_kind="fraction",
            answer=exact_value(value),
            answer_text=key_text,
            answer_latex=latex_answer(key_text),
            marks=marks,
            working=working,
            distractors=options,
        ),
        follow_through,
        sensible_range,
    )


def number_part(
    prompt, value, places, marks, working, wrong, *, follow_through=None, sensible_range=None, key=None
) -> dict:
    """A part whose answer is `value` rounded to `places` decimal places, graded as a `number`.

    `value` is an exact `Fraction` (or anything `Fraction` reads); `key` overrides the rounded key when the caller
    has already rounded it. The prompt gets ", correct to N decimal place(s)" appended unless it already says so."""
    value = Fraction(value)
    rounded = round_half_up(value, places) if key is None else Fraction(key)
    key_text = stated_precision_string(rounded, places) if places else str(int(rounded))
    unit = "decimal place" if places == 1 else "decimal places"
    if places and "decimal place" not in prompt:
        prompt = f"{prompt.rstrip('.?')}, correct to {places} {unit}."
    elif not places and "whole number" not in prompt and "nearest" not in prompt:
        prompt = f"{prompt.rstrip('.')}, correct to the nearest whole number."

    def rounded_of(v):
        return round_half_up(Fraction(v), places)

    def build(v, tag):
        r = rounded_of(v)
        return Distractor(stated_precision_string(r, places) if places else str(int(r)), tag, exact_value(r))

    def same(v):
        return abs(rounded_of(v) - rounded) <= max(GRADER_TOLERANCE, Fraction(1, 10**places))

    options = _first_wrong(key_text, wrong, same, build)
    return _finish(
        dict(
            prompt=prompt,
            answer_kind="number",
            answer=_number_value(rounded),
            answer_text=key_text,
            answer_latex=latex_answer(key_text),
            marks=marks,
            working=working,
            distractors=options,
        ),
        follow_through,
        sensible_range,
    )


def decimal_places(value: Fraction, limit: int = 9) -> int | None:
    """The fewest decimal places that write `value` exactly, or None when it needs more than `limit`."""
    for places in range(limit + 1):
        if (value * 10**places).denominator == 1:
            return places
    return None


def exact_number_part(
    prompt,
    value,
    marks,
    working,
    wrong,
    *,
    follow_through=None,
    sensible_range=None,
    max_places=8,
    min_places=0,
    suffix=None,
) -> dict:
    """A part whose answer is the exact terminating decimal `value`, graded as a `number` to the places it needs.

    Each wrong answer is written as its own exact decimal (one that does not terminate within `max_places` is not
    offered), and is dropped when it lies within the grader's tolerance of the key. The prompt gets "Give your answer as an exact
    decimal." appended unless it already says "decimal"."""
    value = Fraction(value)
    places = decimal_places(value, max_places)
    if places is None:
        raise Redraw("the key is not a short terminating decimal")
    places = max(places, min_places)
    key_text = stated_precision_string(value, places) if places else str(int(value))
    if suffix is None and "decimal" not in prompt:
        prompt = f"{prompt.rstrip('.')}. Give your answer as an exact decimal."
    elif suffix:
        prompt = f"{prompt.rstrip('.')}. {suffix}"
    tolerance = Fraction(1, 2 * 10 ** max(places, 1))

    def text_of(v):
        v = Fraction(sympy.Rational(v).p, sympy.Rational(v).q)
        n = decimal_places(v, max_places)
        if n is None:
            return None
        n = max(n, min_places)
        return stated_precision_string(v, n) if n else str(int(v))

    candidates = [(v, tag) for v, tag in wrong if text_of(v) is not None]

    def build(v, tag):
        v = Fraction(sympy.Rational(v).p, sympy.Rational(v).q)
        return Distractor(text_of(v), tag, exact_value(v))

    def same(v):
        return abs(Fraction(sympy.Rational(v).p, sympy.Rational(v).q) - value) <= tolerance

    options = _first_wrong(key_text, candidates, same, build)
    return _finish(
        dict(
            prompt=prompt,
            answer_kind="number",
            answer=_number_value(value),
            answer_text=key_text,
            answer_latex=latex_answer(key_text),
            marks=marks,
            working=working,
            distractors=options,
        ),
        follow_through,
        sensible_range,
    )


def _number_value(rounded: Fraction):
    return int(rounded) if rounded.denominator == 1 else float(rounded)


_PROBES = (
    sympy.Rational(3, 2),
    sympy.Rational(-7, 3),
    sympy.Rational(5, 4),
    sympy.Rational(11, 7),
    sympy.Rational(-2, 5),
    sympy.Rational(13, 3),
)


def same_function(left, right) -> bool:
    """Whether two expressions are the same function, compared in exact arithmetic at several rational points (one
    value for each symbol, in name order). Cheap where `simplify` is not; a point where exactly one side is
    undefined counts as a difference. Wrong only by coincidence at every point, which the probe values make
    impossible for the low-degree expressions these generators write."""
    left, right = sympy.sympify(left), sympy.sympify(right)
    names = sorted(left.free_symbols | right.free_symbols, key=str)
    for shift in range(len(_PROBES)):
        point = {s: _PROBES[(i * 2 + shift) % len(_PROBES)] for i, s in enumerate(names)}
        a, b = left.xreplace(point), right.xreplace(point)
        if a != b and sympy.simplify(a - b) != 0:
            return False
    return True


def expression_part(prompt, expr, marks, working, wrong, *, follow_through=None, prefix="") -> dict:
    """A part whose answer is the SymPy expression `expr`, graded by equivalence. `wrong` holds `(expr, tag)`.

    `prefix` (for example ``"y = "``) is written before the expression in the stated answer and every option."""
    key_text = prefix + typed(expr)

    def same(e):
        return same_function(e, expr)

    def build(e, tag):
        text = typed(e)
        return Distractor(prefix + text, tag, text)

    options = _first_wrong(key_text, wrong, same, build)
    return _finish(
        dict(
            prompt=prompt,
            answer_kind="expression",
            answer=str(expr),
            answer_text=key_text,
            answer_latex=latex_answer(expr),
            marks=marks,
            working=working,
            distractors=options,
        ),
        follow_through,
        None,
    )


def words_part(prompt, items, marks, working, wrong, *, follow_through=None) -> dict:
    """A part answered by an ordered list of items (numbers and words), graded as an `ordered_list`.
    `items` is a list of ints/strings; `wrong` holds `(items, tag)`."""
    key_text = ", ".join(str(i) for i in items)

    def build(v, tag):
        return Distractor(", ".join(str(i) for i in v), tag, list(v))

    def same(v):
        return list(v) == list(items)

    options = _first_wrong(key_text, wrong, same, build)
    return _finish(
        dict(
            prompt=prompt,
            answer_kind="ordered_list",
            answer=list(items),
            answer_text=key_text,
            answer_latex=f"plain:{key_text}",
            marks=marks,
            working=working,
            distractors=options,
        ),
        follow_through,
        None,
    )


def _finish(part: dict, follow_through, sensible_range) -> dict:
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part


def real(expr, digits: int = 40) -> Fraction:
    """A SymPy expression's value as an exact `Fraction` of its first `digits` significant digits."""
    return Fraction(str(sympy.N(expr, digits)))


def clear_of_boundary(value: Fraction, places: int, margin: Fraction = Fraction(1, 50)) -> bool:
    """True when `value` is not within `margin` of a rounding unit (at `places` decimal places) from a rounding
    tie, so that a learner who works a little differently still reads the same last digit."""
    scaled = abs(value) * 10**places
    return abs((scaled % 1) - Fraction(1, 2)) > margin


def as_fraction(value) -> Fraction:
    value = sympy.Rational(value)
    return Fraction(int(value.p), int(value.q))


def nonzero(*values) -> bool:
    return all(v != 0 for v in values)


__all__ = [
    "Redraw",
    "VerificationError",
    "draw",
    "as_fraction",
    "clear_of_boundary",
    "real",
    "decimal_places",
    "exact_number_part",
    "expression_part",
    "fraction_part",
    "nonzero",
    "number_part",
    "plain",
    "same_function",
    "typed",
    "words_part",
]
