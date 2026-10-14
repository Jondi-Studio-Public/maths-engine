"""Shared building blocks for the measurement, pattern and graph-parameter skills of the `fr2c_*` generators.

Not a generator module (the leading underscore keeps the package loader from importing it as one). It holds:

* the field builders for a single (non-exam) question: ``number_fields``, ``mcq_fields`` and ``pair_fields``, each
  taking the answer and a list of ``(misconception id, wrong value)`` pairs and returning the answer attributes
  with tagged distractors that cannot be graded correct;
* the same three as exam parts (``num_part``, ``mcq_part``, ``pair_part``), and ``expr_part`` re-exported from the
  shared exam kit;
* ``leaks``: whether a figure's visible text spells an answer.

A builder raises ``Redraw`` when it cannot find two distinct wrong values, and ``build_with_redraw`` runs a builder
again with fresh draws.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen import figures
from mathgen.answers import Distractor
from mathgen.core import GRADER_TOLERANCE, latex_answer, numeric_answer

from .exd_x_r2a_common import Redraw, build_with_redraw, pretty, typed  # noqa: F401  (re-exported)

__all__ = [
    "Redraw",
    "build_with_redraw",
    "expr_part",
    "expr_fields",
    "typed",
    "pretty",
    "number_fields",
    "mcq_fields",
    "pair_fields",
    "num_part",
    "mcq_part",
    "pair_part",
    "leaks",
]


def leaks(fig: dict, *answer_texts: str) -> bool:
    """Does any answer appear in the figure's visible text? (Generators redraw when it does.)"""
    try:
        figures.assert_no_leak(fig, list(answer_texts))
    except AssertionError:
        return True
    return False


def _shown(value, places: int) -> str:
    value = Fraction(value)
    if places == 0:
        return str(int(value))
    return f"{float(value):.{places}f}"


def _number_choices(value, wrong, places: int, n: int, minimum: int):
    answer = Fraction(value)
    kept: list[tuple[str, Fraction, str]] = []
    for mid, w in wrong:
        if w is None:
            continue
        w = Fraction(w)
        w = Fraction(round(w * 10**places), 10**places)  # shown value
        text = _shown(w, places)
        if abs(w - answer) <= GRADER_TOLERANCE:
            continue
        if any(text == t or abs(w - v) <= GRADER_TOLERANCE for _m, v, t in kept):
            continue
        kept.append((mid, w, text))
        if len(kept) == n:
            break
    if len(kept) < minimum:
        raise Redraw
    return kept


def number_fields(value, wrong, *, places: int = 0, n: int = 3, minimum: int = 3) -> dict:
    """Answer attributes for a `number` question whose key is ``value`` (shown to ``places`` decimals).

    ``wrong`` is a list of ``(misconception id, value)``; each is shown to the same number of decimals, and any within
    the grader's tolerance of the key (or of an earlier one) is dropped."""
    shown = _shown(value, places)
    kept = _number_choices(value, wrong, places, n, minimum)
    return dict(
        answer=numeric_answer(Fraction(shown)),
        answer_text=shown,
        answer_latex=latex_answer(shown),
        answer_kind="number",
        distractors=[Distractor(t, m, numeric_answer(w)) for m, w, t in kept],
    )


def num_part(
    prompt,
    value,
    marks,
    working,
    wrong,
    *,
    places: int = 0,
    follow_through=None,
    sensible_range=None,
    unit="",
    positive_only=False,
):
    """A `number` exam part; see ``number_fields``."""
    wrong = [(m, w) for m, w in wrong if w is not None and (w > 0 or not positive_only)]
    fields = number_fields(value, wrong, places=places, n=3, minimum=2)
    part = dict(prompt=prompt, marks=marks, working=working, **fields)
    if unit:
        part["working"] = working + [f"So the answer is {fields['answer_text']} {unit}."]
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part


def mcq_fields(correct: str, wrong, *, n: int = 3, minimum: int = 3) -> dict:
    """Answer attributes for an `mcq` question: ``correct`` is the option text, ``wrong`` a list of
    ``(misconception id, option text)``. At least ``minimum`` (default three) different wrong options are needed, so the
    question always has four or more options."""
    kept: list[tuple[str, str]] = []
    for mid, text in wrong:
        if text is None or text == correct or any(text == t for _m, t in kept):
            continue
        kept.append((mid, text))
        if len(kept) == n:
            break
    if len(kept) < minimum:
        raise Redraw
    return dict(
        answer=correct,
        answer_text=correct,
        answer_kind="mcq",
        distractors=[Distractor(t, m, t) for m, t in kept],
    )


def mcq_part(prompt, correct, marks, working, wrong, *, minimum: int = 3, **extra):
    """An `mcq` exam part; see ``mcq_fields``."""
    return dict(prompt=prompt, marks=marks, working=working, **mcq_fields(correct, wrong, minimum=minimum), **extra)


def _pair_text(x, y) -> str:
    return f"({_plain(x)}, {_plain(y)})"


def _plain(v) -> str:
    v = Fraction(v)
    if v.denominator == 1:
        return str(v.numerator)
    return f"{float(v):g}"


def pair_fields(point, wrong, *, n: int = 3, minimum: int = 3) -> dict:
    """Answer attributes for a `pair` question whose key is the point ``point`` (two numbers); ``wrong`` is a list of
    ``(misconception id, (x, y))``."""
    x, y = Fraction(point[0]), Fraction(point[1])
    key = _pair_text(x, y)
    kept: list[tuple[str, tuple, str]] = []
    for mid, w in wrong:
        if w is None:
            continue
        wx, wy = Fraction(w[0]), Fraction(w[1])
        text = _pair_text(wx, wy)
        if text == key or (abs(wx - x) <= GRADER_TOLERANCE and abs(wy - y) <= GRADER_TOLERANCE):
            continue
        if any(text == t for _m, _v, t in kept):
            continue
        kept.append((mid, (wx, wy), text))
        if len(kept) == n:
            break
    if len(kept) < minimum:
        raise Redraw
    return dict(
        answer={"x": numeric_answer(x), "y": numeric_answer(y)},
        answer_text=key,
        answer_latex=key,
        answer_kind="pair",
        distractors=[Distractor(t, m, {"x": numeric_answer(v[0]), "y": numeric_answer(v[1])}) for m, v, t in kept],
    )


def pair_part(prompt, point, marks, working, wrong, *, minimum: int = 2, **extra):
    """A `pair` exam part; see ``pair_fields``."""
    return dict(prompt=prompt, marks=marks, working=working, **pair_fields(point, wrong, minimum=minimum), **extra)


def expr_fields(answer, wrong, *, var, prefix: str = "", text: str | None = None, n: int = 3, minimum: int = 3) -> dict:
    """Answer attributes for an `expression` question whose key is the SymPy expression ``answer`` in ``var``.

    ``wrong`` is a list of ``(misconception id, SymPy expression)`` or ``(misconception id, expression, text)``. Each is
    spelled like the key (``pretty``: no ``*``, ``^`` for powers, ``prefix`` in front, unless a ``text`` is given) and
    any that is the same function as the key, or as an earlier wrong one, is dropped. ``text`` spells the key itself."""
    from .exd_x_r2a_common import same_function

    shown = prefix + (text if text is not None else pretty(answer))
    kept: list[tuple[str, object, str]] = []
    for item in wrong:
        mid, e = item[0], item[1]
        if e is None:
            continue
        t = prefix + (item[2] if len(item) > 2 else pretty(e))
        if t == shown or same_function(e, answer, var) or any(t == k[2] or same_function(e, k[1], var) for k in kept):
            continue
        kept.append((mid, e, t))
        if len(kept) == n:
            break
    if len(kept) < minimum:
        raise Redraw
    return dict(
        answer=typed(answer),
        answer_text=shown,
        answer_latex=latex_answer(shown),
        answer_kind="expression",
        distractors=[Distractor(t, m) for m, _e, t in kept],
    )


def expr_part(
    prompt,
    answer,
    marks,
    working,
    wrong,
    *,
    var,
    prefix: str = "",
    text: str | None = None,
    follow_through=None,
    minimum: int = 2,
):
    """An `expression` exam part; see ``expr_fields``."""
    part = dict(
        prompt=prompt,
        marks=marks,
        working=working,
        **expr_fields(answer, wrong, var=var, prefix=prefix, text=text, minimum=minimum),
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part
