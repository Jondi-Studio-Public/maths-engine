"""Exam source for `add-complex-numbers`: a four-part chain that starts from a conjugate and ends at polar data.

The stem gives ``q`` and the equation ``p +/- k times the conjugate of q = s``.

- (a) ``k`` times the conjugate of ``q``, in the form ``a + bi`` (1 mark);
- (b) ``p`` itself (2 marks, ``follow_through`` from (a));
- (c) the modulus of ``p``, to 2 decimal places (2 marks, ``follow_through`` from (b));
- (d) the principal argument of ``p`` in radians, to 2 decimal places (2 marks, ``follow_through`` from (b)).

## Construction

``p`` and ``q`` are drawn as integer pairs and ``s`` is built from them, so ``p`` is exact. ``p`` has two
non-zero components and a modulus that is not a whole number, so (c) is a genuine square root and (d) a genuine
arctangent in a chosen quadrant. Both decimal answers are kept clear of a rounding boundary. A complex answer is
an expression in the symbol ``i``; the modulus and argument follow-through expressions turn that symbol into the
imaginary unit with ``Subs`` before reading its real and imaginary parts.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import sympy

from ..answers import Distractor
from ..core import (
    attempts,
    check,
    check_equivalent,
    generator,
    latex_answer,
    numeric_answer,
    paren_negative,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

_MARGIN = 2e-4


def _cx(re: int, im: int) -> str:
    """``a + bi``, with a bare ``i`` for a unit imaginary part and a part left out when it is zero."""
    mag = "" if abs(im) == 1 else str(abs(im))
    if im == 0:
        return str(re)
    if re == 0:
        return f"{'-' if im < 0 else ''}{mag}i"
    return f"{re} {'+' if im > 0 else '-'} {mag}i"


def _sym(re: int, im: int) -> str:
    """The same number as a sympy string in the symbol ``i`` (the form a graded complex answer takes)."""
    return f"{re} + ({im})*i"


def _tag(re: int, im: int, tag: str) -> Distractor:
    return Distractor(_cx(re, im), tag, _sym(re, im))


def _decimal(x: float) -> Fraction:
    return round_half_up(Fraction(x), 2)


def _clear(x: float) -> bool:
    scaled = x * 100
    return abs((scaled - math.floor(scaled)) - 0.5) > _MARGIN * 100


@generator("exd_x_r2e_complex_add", level=12, presets={"exam": dict(tier="exam")})
def complex_add(rng: random.Random, tier: str = "exam") -> dict:
    """Solve p +/- k conj(q) = s, then read the modulus and argument of p."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    nonzero = [v for v in range(-7, 8) if v]
    for _ in attempts(4000, "a complex pair with a non-square modulus"):
        px, py = rng.choice(nonzero), rng.choice(nonzero)
        qx, qy = rng.choice([v for v in nonzero if abs(v) <= 6]), rng.choice([v for v in nonzero if abs(v) <= 6])
        k = rng.choice((2, 3))
        sign = rng.choice((1, -1))
        sx, sy = px + sign * k * qx, py + sign * k * (-qy)
        if sx == 0 or sy == 0:
            continue
        sq = px * px + py * py
        if math.isqrt(sq) ** 2 == sq:
            continue
        modulus, arg = math.sqrt(sq), math.atan2(py, px)
        if not (_clear(modulus) and _clear(arg)):
            continue
        break

    # Independent route in sympy for p, its modulus and its argument.
    q = qx + qy * sympy.I
    p = px + py * sympy.I
    check_equivalent(sympy.expand(p + sign * k * sympy.conjugate(q)), sx + sy * sympy.I, "p must satisfy the equation")
    check(math.isclose(float(sympy.Abs(p)), modulus, abs_tol=1e-9), "modulus disagrees")
    check(math.isclose(float(sympy.arg(p)), arg, abs_tol=1e-9), "argument disagrees")

    ax, ay = k * qx, -k * qy  # k times the conjugate of q
    mod_text = stated_precision_string(_decimal(modulus), 2)
    arg_text = stated_precision_string(_decimal(arg), 2)
    op = "+" if sign == 1 else "-"
    stem = f"Let q = {_cx(qx, qy)}. The complex number p satisfies p {op} {k} times the conjugate of q = {_cx(sx, sy)}."

    a_cands = [
        _tag(k * qx, k * qy, "forgets-to-conjugate"),
        _tag(qx, -qy, "forgets-the-multiplier"),
        _tag(-k * qx, k * qy, "conjugates-the-wrong-part"),
        _tag(k * qx, -qy, "forgets-the-multiplier"),
    ]
    b_cands = [
        _tag(sx + sign * ax, sy + sign * ay, "sign-error-moving-term"),
        _tag(sx - sign * k * qx, sy - sign * k * qy, "forgets-to-conjugate"),
        _tag(sx - sign * qx, sy + sign * qy, "forgets-the-multiplier"),
        _tag(sx + sign * k * qx, sy + sign * k * qy, "sign-error-moving-term"),
    ]
    abs_p, abs_s = abs(px) + abs(py), math.hypot(sx, sy)
    c_cands = [
        Distractor(stated_precision_string(_decimal(abs_p), 2), "adds-components-for-modulus", _decimal(abs_p)),
        Distractor(stated_precision_string(_decimal(sq), 2), "forgets-square-root", _decimal(sq)),
        Distractor(stated_precision_string(_decimal(abs_s), 2), "modulus-of-the-wrong-number", _decimal(abs_s)),
    ]
    if px * px != py * py:
        subtracted = math.sqrt(abs(px * px - py * py))
        c_cands.append(
            Distractor(
                stated_precision_string(_decimal(subtracted), 2), "subtracts-squares-in-modulus", _decimal(subtracted)
            )
        )
    atan_ratio = math.atan(py / px)
    swapped = math.atan(px / py)
    d_cands = [
        Distractor(stated_precision_string(_decimal(atan_ratio), 2), "ignores-the-quadrant", _decimal(atan_ratio)),
        Distractor(stated_precision_string(_decimal(swapped), 2), "swaps-real-and-imaginary-parts", _decimal(swapped)),
        Distractor(
            stated_precision_string(_decimal(math.degrees(arg)), 2),
            "gives-degrees-not-radians",
            _decimal(math.degrees(arg)),
        ),
        Distractor(stated_precision_string(_decimal(-arg), 2), "gives-the-argument-of-the-conjugate", _decimal(-arg)),
        Distractor(
            stated_precision_string(_decimal(arg + 2 * math.pi if arg < 0 else arg - 2 * math.pi), 2),
            "gives-argument-outside-principal-range",
            _decimal(arg + 2 * math.pi if arg < 0 else arg - 2 * math.pi),
        ),
    ]
    picks = [
        pick_distractors(_cx(ax, ay), a_cands, 3),
        pick_distractors(_cx(px, py), b_cands, 3),
        pick_distractors(mod_text, c_cands, 3, tolerance=Fraction(5, 1000)),
        pick_distractors(arg_text, d_cands, 3, tolerance=Fraction(5, 1000)),
    ]
    check(all(len(x) >= 2 for x in picks), f"a part has too few distractors: {[len(x) for x in picks]}")

    quadrant = 1 if px > 0 < py else 2 if px < 0 < py else 3 if px < 0 else 4
    ft_p = f"({sx} + ({sy})*i) - part_a" if sign == 1 else f"({sx} + ({sy})*i) + part_a"
    re_p, im_p = "re(Subs(part_b,(i,),(I,)))", "im(Subs(part_b,(i,),(I,)))"
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"Write down {k} times the conjugate of q, in the form a + bi.",
                answer_kind="expression",
                answer=_sym(ax, ay),
                answer_text=_cx(ax, ay),
                answer_latex=latex_answer(_cx(ax, ay), plain=True),
                marks=1,
                working=[f"The conjugate of q is {_cx(qx, -qy)}, so {k} times it is {_cx(ax, ay)}."],
                distractors=picks[0],
            ),
            dict(
                prompt="Hence find p, in the form a + bi.",
                answer_kind="expression",
                answer=_sym(px, py),
                answer_text=_cx(px, py),
                answer_latex=latex_answer(_cx(px, py), plain=True),
                marks=2,
                follow_through=ft_p,
                working=[f"p = {_cx(sx, sy)} {'-' if sign == 1 else '+'} ({_cx(ax, ay)}) = {_cx(px, py)}."],
                distractors=picks[1],
            ),
            dict(
                prompt="Find the modulus of p, correct to 2 decimal places.",
                answer_kind="number",
                answer=numeric_answer(_decimal(modulus)),
                answer_text=mod_text,
                answer_latex=latex_answer(mod_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"sqrt({re_p}**2 + {im_p}**2)",
                working=[f"|p| = sqrt({paren_negative(px)}^2 + {paren_negative(py)}^2) = sqrt({sq}) = {mod_text}."],
                distractors=picks[2],
            ),
            dict(
                prompt="Find the principal argument of p, in radians correct to 2 decimal places.",
                answer_kind="number",
                answer=numeric_answer(_decimal(arg)),
                answer_text=arg_text,
                answer_latex=latex_answer(arg_text),
                marks=2,
                follow_through="arg(Subs(part_b,(i,),(I,)))",
                working=[f"p = {_cx(px, py)} lies in quadrant {quadrant}, so Arg(p) = {arg_text} radians."],
                distractors=picks[3],
            ),
        ],
    )
