"""Part builders shared by the multi-part exam sources whose parts are exact fractions or algebraic expressions.

Each builder returns one entry for the `parts` list of a multi-part question (see `docs/design/multi-part-questions.md`)
with tagged distractors: a candidate is a pair `(misconception id, value)`; the builder drops a candidate that is not a
finite value, that equals the answer (or one of the `avoid` values, such as the follow-through route's answer), or whose
text repeats an earlier candidate's, and keeps the first three.
"""

from __future__ import annotations

import sympy
from sympy import Rational

from ..answers import Distractor, exact_value
from ..core import check, latex_answer, rational_text


_PROBES = ((Rational(3, 7), Rational(-5, 11)), (Rational(-8, 3), Rational(2, 9)), (Rational(13, 5), Rational(7, 4)))


def same_expression(a, b) -> bool:
    """Whether two expressions (polynomials, rational functions, exponentials, constants such as pi) are equal.

    They are compared to 20 digits at a few fixed rational points (once, when they have no free symbols). A gap above
    1e-12 proves they differ; agreement is taken as equality, the safe direction for a generator that only uses this to
    drop a distractor that might be the answer. `symbolically_equal` proves the same thing but costs tens of
    milliseconds on an exponential, which a generator running ~1000 times per test sweep cannot afford."""
    a, b = sympy.sympify(a), sympy.sympify(b)
    if a == b:
        return True
    diff = a - b
    symbols = sorted(diff.free_symbols, key=str)
    points = [{}] if not symbols else [dict(zip(symbols, probe)) for probe in _PROBES]
    for point in points:
        try:
            gap = sympy.N(diff.subs(point), 20)
        except (TypeError, ValueError, ZeroDivisionError):
            return False
        if not gap.is_number or abs(gap) > sympy.Float("1e-12"):
            return False
    return True


def typed(expr) -> str:
    """An expression as a learner would type it: SymPy's text with `^` for powers."""
    return str(expr).replace("**", "^")


def _rational(value):
    try:
        r = Rational(value)
    except (TypeError, ValueError):
        return None
    return r if r.is_finite else None


def frac_part(prompt, value, marks, working, candidates, follow_through=None, sensible_range=None, avoid=()):
    """A part whose answer is the exact rational `value`, graded as a `fraction`."""
    value = Rational(value)
    bad = [value, *(Rational(a) for a in avoid)]
    tagged, seen = [], {rational_text(value)}
    for tag, cand in candidates:
        r = _rational(cand) if cand is not None else None
        if r is None or any(r == b for b in bad) or rational_text(r) in seen:
            continue
        seen.add(rational_text(r))
        tagged.append(Distractor(rational_text(r), tag, exact_value(r)))
    check(len(tagged) >= 2, "too few distractors")
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer=exact_value(value),
        answer_text=rational_text(value),
        answer_latex=latex_answer(rational_text(value)),
        marks=marks,
        working=working,
        distractors=tagged[:3],
    )
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part


def expr_part(prompt, expr, marks, working, candidates, follow_through=None, avoid=(), answer_text=None):
    """A part whose answer is the SymPy expression `expr`, graded as an `expression` (any equivalent form earns it)."""
    text = answer_text or typed(expr)
    bad = [expr, *avoid]
    tagged, seen = [], {text}
    for tag, cand in candidates:
        if cand is None:
            continue
        cand_text = typed(cand)
        if cand_text in seen or any(same_expression(cand, b) for b in bad):
            continue
        seen.add(cand_text)
        tagged.append(Distractor(cand_text, tag, str(cand)))
    check(len(tagged) >= 2, "too few distractors")
    part = dict(
        prompt=prompt,
        answer_kind="expression",
        answer=str(expr),
        answer_text=text,
        answer_latex=latex_answer(expr),
        marks=marks,
        working=working,
        distractors=tagged[:3],
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part
