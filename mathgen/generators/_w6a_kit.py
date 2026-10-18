"""Small shared helpers for the `w6a_*` senior builds: answer fields and tagged distractors for exact numbers,
rounded decimals, exact expressions and sets, plus a part builder for exam chains.

Nothing here draws random numbers or registers a topic; each build module keeps its own mathematics.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..answers import Distractor, exact_value
from ..core import VerificationError, _escape_prose, attempts, rational_text, stated_precision_tolerance
from ._ucv1_calc_markup import M, plain, say, tex_rational, tex_value, value_string, value_text

__all__ = [
    "M",
    "say",
    "plain",
    "tex_say",
    "rat",
    "num_fields",
    "num_dist",
    "dec_fields",
    "dec_dist",
    "expr_fields",
    "expr_dist",
    "set_fields",
    "set_dist",
    "part",
    "enough",
    "retry_draw",
    "clear_of_rounding",
]


def tex_say(*parts) -> tuple[str, str]:
    """`say`, but a formula may use ``\\mathrm{d}`` for the differential; the plain text reads it as ``d``."""
    converted = [M(str(p).replace(r"\mathrm{d}", "d")) if isinstance(p, M) else p for p in parts]
    text, _ = say(*converted)
    tex = "".join(str(p) if isinstance(p, M) else r"\text{" + _escape_prose(p) + "}" for p in parts if p)
    return text, tex


def rat(value) -> Fraction:
    """`value` (int, Fraction or SymPy Rational) as a Fraction."""
    if isinstance(value, Fraction):
        return value
    if getattr(value, "is_Rational", False):
        return Fraction(int(value.p), int(value.q))
    return Fraction(value)


# -- exact numbers (a whole number or a fraction) ----------------------------------------------------------------------


def num_fields(value) -> dict:
    value = rat(value)
    return dict(
        answer=exact_value(value),
        answer_text=rational_text(sympy.Rational(value.numerator, value.denominator)),
        answer_latex=tex_rational(value),
    )


def num_dist(correct, candidates, n: int = 4) -> list[Distractor]:
    """Tagged distractors for an exact number: `candidates` is ``[(value, misconception id), ...]`` in priority order;
    a value equal to the answer, repeated, or None is skipped."""
    seen = {rat(correct)}
    out: list[Distractor] = []
    for value, tag in candidates:
        if value is None:
            continue
        value = rat(value)
        if value in seen:
            continue
        seen.add(value)
        out.append(
            Distractor(rational_text(sympy.Rational(value.numerator, value.denominator)), tag, exact_value(value))
        )
        if len(out) == n:
            break
    _same_shape_as_answer(correct, out)
    return out


def _is_whole(value) -> bool:
    return rat(value).denominator == 1


def _same_shape_as_answer(correct, out) -> None:
    """The answer must not be the only whole number (or the only fraction) among the options."""
    if len(out) < 2:
        return
    shapes = [
        _is_whole(d.value if not isinstance(d.value, dict) else Fraction(d.value["p"], d.value["q"])) for d in out
    ]
    if shapes.count(_is_whole(correct)) == 0:
        raise TooFewDistractors("every distractor has a different shape from the answer")


# -- decimals graded to their own stated precision ---------------------------------------------------------------------


def clear_of_rounding(value: float, places: int, margin: float = 0.12) -> bool:
    """Whether `value` sits clear of a rounding tie at `places` decimal places, so float noise cannot change the digits."""
    scaled = abs(value) * 10**places
    return abs((scaled % 1) - 0.5) > margin * 0.5


def dec_fields(value: float, places: int) -> dict:
    text = f"{value:.{places}f}"
    return dict(answer=float(text), answer_text=text, answer_latex=text)


def dec_dist(correct: float, candidates, places: int, n: int = 4) -> list[Distractor]:
    tol = float(stated_precision_tolerance(places)) * 2
    correct_text = f"{correct:.{places}f}"
    seen = {correct_text}
    out: list[Distractor] = []
    for value, tag in candidates:
        if value is None:
            continue
        text = f"{value:.{places}f}"
        if text in seen or abs(float(text) - float(correct_text)) <= tol:
            continue
        seen.add(text)
        out.append(Distractor(text, tag, float(text)))
        if len(out) == n:
            break
    return out


# -- exact expressions (surds, multiples of pi, logarithms) ---------------------------------------------------------------


def expr_fields(value) -> dict:
    value = sympy.sympify(value)
    return dict(answer=value_string(value), answer_text=value_text(value), answer_latex=tex_value(value))


def _same(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-9
    except TypeError:
        return False


def expr_dist(correct, candidates, n: int = 4) -> list[Distractor]:
    seen = [sympy.sympify(correct)]
    out: list[Distractor] = []
    for value, tag in candidates:
        if value is None:
            continue
        value = sympy.sympify(value)
        if not value.is_real or any(_same(value, s) for s in seen):
            continue
        seen.append(value)
        out.append(Distractor(value_text(value), tag, value_string(value)))
        if len(out) == n:
            break
    return out


# -- sets of whole numbers ---------------------------------------------------------------------------------------------


def _set_text(items) -> str:
    return "{" + ", ".join(str(i) for i in sorted(items)) + "}"


def set_fields(items) -> dict:
    items = sorted(items)
    return dict(answer=list(items), answer_text=_set_text(items), answer_latex="plain:" + _set_text(items))


def set_dist(correct, candidates, n: int = 3) -> list[Distractor]:
    seen = {frozenset(correct)}
    out: list[Distractor] = []
    for items, tag in candidates:
        key = frozenset(items)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(Distractor(_set_text(items), tag, sorted(items)))
        if len(out) == n:
            break
    return out


# -- parts --------------------------------------------------------------------------------------------------------------


def part(prompt: str, kind: str, fields: dict, marks: int, working, distractors, **extra) -> dict:
    return dict(
        prompt=prompt, answer_kind=kind, marks=marks, working=list(working), distractors=distractors, **fields, **extra
    )


class TooFewDistractors(VerificationError):
    """A draw whose error candidates collapsed onto the answer or each other; `retry_draw` redraws it."""


def enough(distractors, n: int = 2, what: str = "distractors") -> None:
    if len(distractors) < n:
        raise TooFewDistractors(f"too few {what}: {len(distractors)}")


def retry_draw(rng, build, what: str = "a question with enough distractors"):
    """`build(rng)`, redrawn when it ends up with too few distractors (the draw is a degenerate one)."""
    for _ in attempts(60, what):
        try:
            return build(rng)
        except TooFewDistractors:
            continue
    raise AssertionError  # pragma: no cover
