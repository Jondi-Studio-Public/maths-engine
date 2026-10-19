"""Small helpers shared by the senior series, complex-number, vector, conic and counting builds in the `r3i_*`
generators.

Not a generator module (the leading underscore keeps the package loader from importing it): it holds tagged
distractors for each answer kind these builds use, answer dictionaries for whole numbers and exact fractions,
vectors and inequalities, and a filter that asks the real grader whether a distractor would be accepted as the
answer.
"""

from __future__ import annotations

import functools
import re
from fractions import Fraction
from types import SimpleNamespace

import sympy

from ..answers import Distractor, exact_value
from ..core import VerificationError, check, pick_distractors, rational_text


class TooFewDistractors(VerificationError):
    """A draw left fewer than two distinct wrong values; `retry_short` draws again."""


def retry_short(build, tries: int = 60):
    """Wrap a generator body so a draw with too few distractors (or no usable numbers) is redrawn, at most `tries`
    times and deterministically."""

    @functools.wraps(build)
    def run(rng, *args, **kwargs):
        for _ in range(tries):
            try:
                return build(rng, *args, **kwargs)
            except TooFewDistractors:
                continue
        raise TooFewDistractors(f"{build.__name__}: no draw had enough distractors in {tries} tries")

    return run


# -- asking the real grader --------------------------------------------------------------------------------------


def accepted_by_grader(fields: dict, reply: str) -> bool:
    """Whether the grader for `fields["answer_kind"]` marks `reply` correct for the answer in `fields`."""
    from ..grading import GRADERS

    question = SimpleNamespace(
        answer=fields["answer"],
        answer_text=fields["answer_text"],
        answer_kind=fields["answer_kind"],
        answer_form=fields.get("answer_form"),
        answer_alt=None,
        text="",
        parts=None,
    )
    return bool(GRADERS[fields["answer_kind"]](question, reply).correct)


def pick(fields: dict, candidates: list, count: int = 3, *, minimum: int = 2, unordered: bool = False) -> list:
    """The first `count` candidates that are not the answer, not each other and not accepted by the real grader as the
    answer, at least `minimum` of them. `None` candidates are skipped."""
    candidates = [c for c in candidates if c is not None]
    candidates = [c for c in candidates if not accepted_by_grader(fields, getattr(c, "text", str(c)))]
    out = pick_distractors(fields["answer_text"], candidates, count, unordered=unordered)
    if len(out) < minimum:
        raise TooFewDistractors(f"too few distractors for {fields['answer_text']}: {[str(c) for c in candidates]}")
    return out


def pick_text(fields: dict, candidates: list, count: int = 3, *, minimum: int = 2) -> list:
    """Like `pick` but without asking the grader: for answers whose candidates are built from structured values (so a
    different text is always a different answer) and whose grader is slow to start. Tests grade every distractor."""
    candidates = [c for c in candidates if c is not None]
    out = pick_distractors(fields["answer_text"], candidates, count)
    if len(out) < minimum:
        raise TooFewDistractors(f"too few distractors for {fields['answer_text']}: {[str(c) for c in candidates]}")
    return out


# -- numbers -----------------------------------------------------------------------------------------------------


def frac(value) -> Fraction:
    return Fraction(value)


def num_fields(value) -> dict:
    """The answer dictionary for an exact value: a whole number is a `number`, anything else a `fraction`."""
    value = Fraction(value)
    if value.denominator == 1:
        return dict(answer_kind="number", answer=int(value), answer_text=str(int(value)), answer_latex=str(int(value)))
    text = rational_text(value)
    sign = "-" if value < 0 else ""
    latex = rf"{sign}\frac{{{abs(value.numerator)}}}{{{value.denominator}}}"
    return dict(answer_kind="fraction", answer=exact_value(value), answer_text=text, answer_latex=latex)


def num_dist(value, tag: str) -> Distractor:
    """A wrong whole number or exact fraction as a tagged distractor."""
    value = Fraction(value)
    if value.denominator == 1:
        return Distractor(str(int(value)), tag, int(value))
    return Distractor(rational_text(value), tag, exact_value(value))


def num_cands(pairs, answer) -> list:
    """Tagged distractors from `(value, tag)` pairs. A value equal to the answer is left out, and so is a fraction when
    the answer is a whole number (it would give the key away)."""
    whole = Fraction(answer).denominator == 1
    return [
        num_dist(v, tag)
        for v, tag in pairs
        if v is not None and Fraction(v) != Fraction(answer) and (not whole or Fraction(v).denominator == 1)
    ]


def expr_num_fields(value) -> dict:
    """An exact number answered through the `expression` grader (for a skill whose answer kind is `expression`, where a
    whole number or fraction is spelled as text): `5`, `-3/4`."""
    text = rational_text(Fraction(value))
    return dict(answer_kind="expression", answer=text, answer_text=text, answer_latex="plain:" + text)


def expr_num_cands(pairs, answer) -> list:
    """Tagged text distractors for `expr_num_fields` from `(value, tag)` pairs, with the same filtering as `num_cands`."""
    whole = Fraction(answer).denominator == 1
    return [
        Distractor(rational_text(Fraction(v)), tag, rational_text(Fraction(v)))
        for v, tag in pairs
        if v is not None and Fraction(v) != Fraction(answer) and (not whole or Fraction(v).denominator == 1)
    ]


# -- expressions -------------------------------------------------------------------------------------------------


def nice(expr) -> str:
    """A SymPy expression as a learner types it: `^` for powers, `ln`, `π`, no multiplication stars before a symbol."""
    s = sympy.sstr(expr)
    s = s.replace("log(", "ln(").replace("**", "^")
    s = re.sub(r"(\d)\*(?=[A-Za-z(])", r"\1", s)
    s = re.sub(r"(?<=[A-Za-z)])\*(?=[A-Za-z(])", "", s)
    return s.replace("I", "i")


def expr_fields(expr, text: str | None = None) -> dict:
    """The answer dictionary for an expression answer. `text` overrides the spelling."""
    shown = text if text is not None else nice(expr)
    return dict(answer_kind="expression", answer=shown, answer_text=shown, answer_latex="plain:" + shown)


def expr_dist(text: str, tag: str) -> Distractor:
    """A wrong typed expression, spelled as an answer is, as a tagged distractor whose value is its own text."""
    return Distractor(text, tag, text)


# -- inequalities ------------------------------------------------------------------------------------------------


def ineq_fields(text: str) -> dict:
    return dict(answer_kind="inequality", answer=text, answer_text=text, answer_latex="plain:" + text)


def ineq_dist(text: str, tag: str) -> Distractor:
    return Distractor(text, tag, text)


# -- vectors -----------------------------------------------------------------------------------------------------


def vec_text(entries) -> str:
    """A vector as an answer is spelled: (1, -2, 3/2)."""
    return "(" + ", ".join(rational_text(Fraction(e)) for e in entries) + ")"


def vec_fields(entries) -> dict:
    from ..grading.matrix import matrix_latex

    entries = [Fraction(e) for e in entries]
    value = [[int(e) if e.denominator == 1 else exact_value(e) for e in entries]]
    return dict(
        answer_kind="matrix",
        answer=value,
        answer_text=vec_text(entries),
        answer_latex=matrix_latex([[sympy.Rational(e.numerator, e.denominator) for e in entries]]),
    )


def vec_dist(entries, tag: str) -> Distractor:
    entries = [Fraction(e) for e in entries]
    value = [[int(e) if e.denominator == 1 else exact_value(e) for e in entries]]
    return Distractor(vec_text(entries), tag, value)


def vlen_sq(v) -> int:
    return sum(Fraction(x) ** 2 for x in v)


def dot(u, v):
    return sum(Fraction(a) * Fraction(b) for a, b in zip(u, v))


def cross(u, v):
    return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])


def vtext(v) -> str:
    """A column vector for a stem, written as a row: (2, -1, 3)."""
    return "(" + ", ".join(str(int(x)) for x in v) + ")"


def signed_text(n: int) -> str:
    """` + 3` or ` - 3` for joining a constant to an expression."""
    return f" + {n}" if n >= 0 else f" - {-n}"


def check_whole(value, what: str) -> int:
    value = Fraction(value)
    check(value.denominator == 1, f"{what} is not a whole number: {value}")
    return int(value)


# -- matrices and vectors of exact (possibly surd) numbers ------------------------------------------------------


def _entry_json(value):
    """A SymPy number as the JSON value a matrix answer stores: an int, an exact fraction, or text for a surd."""
    value = sympy.nsimplify(value) if not isinstance(value, sympy.Basic) else value
    if value.is_Integer:
        return int(value)
    if value.is_Rational:
        return exact_value(value)
    return sympy.sstr(value)


def matrix_fields(rows) -> dict:
    """The answer dictionary for a matrix (or a vector, as one row) of exact SymPy numbers."""
    from ..grading.matrix import matrix_text

    rows = [[sympy.sympify(v) for v in row] for row in rows]
    return dict(
        answer_kind="matrix",
        answer=[[_entry_json(v) for v in row] for row in rows],
        answer_text=matrix_text(rows),
        answer_latex="plain:" + matrix_text(rows),
    )


def matrix_dist(rows, tag: str) -> Distractor:
    fields = matrix_fields(rows)
    return Distractor(fields["answer_text"], tag, fields["answer"])
