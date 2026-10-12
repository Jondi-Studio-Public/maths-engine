"""Shared building blocks for the `exd_x_r2a_*` multi-part calculus exam chains.

Nothing here registers a topic. The helpers build the three kinds of part the chains use (an expression, an exact
fraction, a number to a stated number of decimal places), tag every distractor with an id from
`mathgen.misconceptions`, and write the `follow_through` expressions that carry a learner's own earlier answer into
a later part:

* `gradient_ft` and friends read a LINE or a CURVE out of an earlier expression part by evaluating it at a few
  points (`Subs(part_a, (x,), (2,))`), so a learner who typed a different but equivalent form is treated the same;
* `quadratic_coefficients_ft` recovers the coefficients A, B, C of a quadratic from a learner's derivative the same
  way, so "the larger root of your f'(x)" can be followed through.

Every distractor is checked against its answer with `same_function` / exact equality, so a distractor can never be
accepted by the grader.
"""

from __future__ import annotations

import re
from fractions import Fraction
from functools import lru_cache

import mpmath
import sympy
from sympy import Rational, Symbol

from ..answers import Distractor
from ..core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    latex_answer,
    numeric_answer,
    rational_text,
    round_half_up,
    stated_precision_string,
)

X = Symbol("x")
T = Symbol("t")

#: Rational points at which two functions of one variable are compared. Chosen to dodge the poles and zeros that
#: small integer coefficients put at whole numbers and simple fractions.
_PROBES = tuple(
    Rational(a, b)
    for a, b in (
        (3, 7),
        (11, 5),
        (-5, 3),
        (7, 2),
        (13, 9),
        (-9, 4),
        (17, 6),
        (2, 11),
        (25, 1),
        (-25, 1),
        (40, 3),
        (-37, 2),
    )
)


def typed(expr) -> str:
    """The learner-facing typed spelling of an expression: `^` for powers."""
    return str(expr).replace("**", "^")


def pretty(expr) -> str:
    """Worked-solution spelling of an expression: `^` for powers and no multiplication signs (`6x^2 - 36x`)."""
    return typed(expr).replace("*", "")


@lru_cache(maxsize=16384)
def _value_at(expr, var: Symbol, point):
    """`expr` at `point`, exact when rational and to 30 digits otherwise; None where it is undefined or not real."""
    try:
        v = expr.subs(var, point)
        v = sympy.N(v, 30) if not v.is_Rational else v
    except (ZeroDivisionError, TypeError, ValueError):
        return None
    if v.has(sympy.zoo, sympy.nan, sympy.oo, -sympy.oo) or not v.is_real:
        return None
    return v


def same_function(left, right, var: Symbol = X) -> bool:
    """Whether two expressions in `var` take the same value at the probe points (at least three must be finite).

    Cheap: exact substitution and a 30-digit evaluation, where `simplify` would take tens of milliseconds. The value of
    each expression at each point is cached, so comparing one answer with its several distractors evaluates the
    answer once."""
    left, right = sympy.sympify(left), sympy.sympify(right)
    agreed = 0
    for point in _PROBES:
        a, b = _value_at(left, var, point), _value_at(right, var, point)
        if a is None or b is None:
            continue
        if abs(a - b) > Rational(1, 10**9) * (1 + abs(a)):
            return False
        agreed += 1
        if agreed == 3:
            return True
    check(agreed >= 3, "two functions could not be compared at three points")
    return True


def derivative_agrees(f, fprime, var: Symbol = X) -> bool:
    """Whether `fprime` is the derivative of `f`: mpmath's numerical derivative of `f` against `fprime`, at four
    rational points and to 20 digits. Used where `sympy.diff` of a root or reciprocal power would spend 20 ms in
    assumption queries; at least three points must be usable (real and finite) or the draw is rejected."""
    with mpmath.workdps(40):
        fn, gn = sympy.lambdify(var, f, "mpmath"), sympy.lambdify(var, fprime, "mpmath")
        agreed = 0
        for point in _PROBES:
            at_point = mpmath.mpf(point.p) / point.q
            try:
                numeric, claimed = mpmath.diff(fn, at_point), gn(at_point)
            except (ZeroDivisionError, TypeError, ValueError):
                continue
            if isinstance(numeric, mpmath.mpc) or isinstance(claimed, mpmath.mpc):
                continue
            if abs(numeric - claimed) > mpmath.mpf(10) ** -20 * (1 + abs(claimed)):
                return False
            agreed += 1
            if agreed == 4:
                return True
        check(agreed >= 3, "the derivative could not be compared at three points")
        return True


_SLOT_RE = re.compile(r"(\{\w+\})")


def lx(template: str, **math: str) -> str:
    r"""LaTeX for a prompt written as words with `{name}` slots: the words go in `\text{}`, each slot is the LaTeX
    given for it in `math`. The twin of `template.format(...)` with the slots' plain spellings."""
    out = []
    for piece in _SLOT_RE.split(template):
        if not piece:
            continue
        if piece.startswith("{") and piece.endswith("}"):
            out.append(math[piece[1:-1]])
        else:
            out.append(r"\text{" + piece + "}")
    return "".join(out)


def prose(rng, patterns, **slots: tuple[str, str]) -> tuple[str, str]:
    """Pick one wording from `patterns`; return its plain text and its LaTeX. Each slot is `(plain, latex)`."""
    pattern = rng.choice(list(patterns))
    plain = pattern.format(**{k: v[0] for k, v in slots.items()})
    return plain, lx(pattern, **{k: v[1] for k, v in slots.items()})


YF = ("y = f(x)", "y = f(x)")


def x_equals(x0) -> tuple[str, str]:
    """The slot for "x = x0"."""
    return f"x = {rational_text(x0)}", f"x = {_tex_rational(x0)}"


def part_prompt(template: str, **slots: tuple[str, str]) -> dict:
    """`prompt` and `prompt_latex` for a part from a template with `{name}` slots, each `(plain, latex)`."""
    return dict(
        prompt=template.format(**{k: v[0] for k, v in slots.items()}),
        prompt_latex=lx(template, **{k: v[1] for k, v in slots.items()}),
    )


class Redraw(Exception):
    """Raised by a builder to ask for a fresh random draw (a degenerate case, or too few distractors)."""


def build_with_redraw(rng, build, *, tries: int = 300, what: str = "an exam chain"):
    """Call `build(rng)` until it does not raise `Redraw`; raise `VerificationError` after `tries` attempts."""
    for _ in attempts(tries, what):
        try:
            return build(rng)
        except Redraw:
            continue


# -- parts ------------------------------------------------------------------------------------------------------


def _prompt_fields(prompt) -> dict:
    return dict(prompt) if isinstance(prompt, dict) else dict(prompt=prompt)


def _pick(cands, answer_text: str, answer_value, equal, n: int = 3, minimum: int = 2):
    """Up to `n` tagged candidates that differ from the answer and from each other. `equal(a, b)` compares two
    candidate values; each candidate is `(misconception_id, value, text)`."""
    kept: list[tuple[str, object, str]] = []
    for mid, value, text in cands:
        if value is None or text == answer_text or equal(value, answer_value):
            continue
        if any(text == t or equal(value, v) for _m, v, t in kept):
            continue
        kept.append((mid, value, text))
        if len(kept) == n:
            break
    if len(kept) < minimum:
        raise Redraw
    return kept


def expr_part(prompt, answer, marks, working, wrong, follow_through=None, *, var: Symbol = X, text=None, prefix=""):
    # `prompt` is a string, or a dict from `part_prompt` carrying `prompt` and `prompt_latex`.
    """An `expression` part whose answer is the SymPy expression `answer`.

    `wrong` lists `(misconception_id, expression)`; the answer is typed with `prefix` ("y = " for a line)."""
    answer_text = prefix + (text if text is not None else typed(answer))
    cands = [(m, e, prefix + typed(e)) for m, e in wrong if e is not None]
    kept = _pick(cands, answer_text, answer, lambda a, b: same_function(a, b, var))
    part = dict(
        **_prompt_fields(prompt),
        answer_kind="expression",
        answer=typed(answer),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        marks=marks,
        working=working,
        distractors=[Distractor(t, m) for m, _e, t in kept],
    )
    if follow_through:
        part["follow_through"] = follow_through
    return part


def fraction_part(prompt, value, marks, working, wrong, follow_through=None, sensible_range=None):
    """A `fraction` part whose answer is the exact rational `value`. `wrong` lists `(misconception_id, rational)`."""
    value = Rational(value)
    answer_text = rational_text(value)
    cands = []
    for m, v in wrong:
        if v is None:
            continue
        v = sympy.nsimplify(v) if not isinstance(v, sympy.Basic) else v
        if v.is_Rational and v.is_finite:
            cands.append((m, v, rational_text(v)))
    kept = _pick(cands, answer_text, value, lambda a, b: a == b)
    part = dict(
        **_prompt_fields(prompt),
        answer_kind="fraction",
        answer={"p": int(value.p), "q": int(value.q)} if value.q != 1 else int(value.p),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        marks=marks,
        working=working,
        distractors=[Distractor(t, m, Fraction(int(v.p), int(v.q))) for m, v, t in kept],
    )
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part


def number_part(prompt, value, places, marks, working, wrong, follow_through=None, sensible_range=None, unit=""):  # noqa: PLR0913
    """A `number` part: the real number `value` (a SymPy expression or Fraction) correct to `places` decimals.

    `wrong` lists `(misconception_id, real_value)`; each is rounded the same way and must clear the grader."""
    answer_value = round_half_up(_exact(value), places)
    answer_text = stated_precision_string(answer_value, places)
    cands = []
    for m, v in wrong:
        if v is None:
            continue
        r = round_half_up(_exact(v), places)
        cands.append((m, r, stated_precision_string(r, places)))
    kept = _pick(cands, answer_text, answer_value, lambda a, b: abs(a - b) <= GRADER_TOLERANCE)
    shown = f"{answer_text} {unit}".strip()
    part = dict(
        **_prompt_fields(prompt),
        answer_kind="number",
        answer=numeric_answer(answer_value),
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text),
        marks=marks,
        working=working + [f"So the answer is {shown} (to {places} decimal place{'s' if places != 1 else ''})."],
        distractors=[Distractor(t, m, numeric_answer(r)) for m, r, t in kept],
    )
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part


def _exact(value) -> Fraction:
    """A Fraction for a SymPy number or Fraction, accurate to 30 digits (for rounding to a few places)."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    return Fraction(str(sympy.N(sympy.sympify(value), 40)))


def rounding_is_safe(value, places: int, margin: Fraction = Fraction(1, 1000)) -> bool:
    """True when `value` lies at least `margin` (in units of the last place) from a rounding boundary."""
    scaled = _exact(value) * 10**places
    offset = scaled % 1
    return abs(offset - Fraction(1, 2)) > margin


# -- follow-through expressions -----------------------------------------------------------------------------------


def at(label: str, point, var: str = "x") -> str:
    """The follow-through text for part `label`'s expression evaluated at `point`."""
    return f"Subs(part_{label}, ({var},), ({_lit(point)},))"


def _lit(v) -> str:
    v = Rational(v)
    return str(v.p) if v.q == 1 else f"({v.p}/{v.q})"


def shield(text: str, var: str = "x") -> str:
    """`text` wrapped so the grader's parser treats it as one opaque term until the earlier parts are filled in.

    A square root of a sum whose terms are sums of `Subs(...)` applications makes SymPy run a long chain of
    assumption queries (10-15 ms) while the follow-through is built. Wrapping each term in a `Subs` over a variable
    the term does not use gives the same value without them."""
    return f"Subs({text}, ({var},), (0,))"


def line_slope_ft(label: str, var: str = "x") -> str:
    """Slope of the line a learner gave in part `label` (any form `mx + c`)."""
    return f"({at(label, 1, var)} - {at(label, 0, var)})"


def line_intercept_ft(label: str, var: str = "x") -> str:
    return at(label, 0, var)


def quadratic_coefficients_ft(label: str, var: str = "x") -> tuple[str, str, str]:
    """Follow-through texts for A, B, C of the quadratic `A var^2 + B var + C` a learner gave in part `label`."""
    p1, m1, p0 = at(label, 1, var), at(label, -1, var), at(label, 0, var)
    return f"(({p1} + {m1} - 2*{p0})/2)", f"(({p1} - {m1})/2)", f"{p0}"


def paren(value) -> str:
    """A number as it follows an operator in a follow-through expression: negatives and fractions bracketed."""
    v = Rational(value)
    return str(v) if v >= 0 and v.q == 1 else f"({v.p}/{v.q})" if v.q != 1 else f"({v.p})"


# -- the tangent / normal tail shared by several chains -----------------------------------------------------------


def _tex_rational(v) -> str:
    v = Rational(v)
    return str(v.p) if v.q == 1 else rf"{'-' if v < 0 else ''}\frac{{{abs(v.p)}}}{{{v.q}}}"


def tangent_tail(f0, m, x0, *, variant: str, m_at_neg_x0=None):
    """Parts (b), (c) and (d) of a chain whose part (a) found f'(x): the gradient at `x = x0`, a line through the
    point of contact, and one number about that line.

    `variant`:
      * "tangent_x"    - (c) the tangent, (d) where it meets the x-axis;
      * "tangent_area" - (c) the tangent, (d) the area of the triangle it makes with the two axes;
      * "normal_y"     - (c) the normal, (d) where it meets the y-axis.

    `f0` is f(x0) and `m` is f'(x0), both exact rationals; `m_at_neg_x0` is f'(-x0) for the sign-slip distractor.
    Every follow-through is written from the learner's own (b) and (c).
    """
    f0, m, x0 = Rational(f0), Rational(m), Rational(x0)
    if m == 0 or f0 == 0:
        raise Redraw
    normal = variant == "normal_y"
    tangent = m * X + (f0 - m * x0)
    nslope = -1 / m
    normal_line = nslope * X + (f0 - nslope * x0)
    line = normal_line if normal else tangent
    c0 = line.subs(X, 0)  # y-intercept of the line asked for
    if c0 == 0:
        raise Redraw
    slope = nslope if normal else m

    # (b) the gradient at x = x0 -------------------------------------------------------------------------------
    xv, yf = x_equals(x0), YF
    part_b = fraction_part(
        part_prompt("Hence find the gradient of the graph of {yf} at the point where {xv}.", yf=yf, xv=xv),
        m,
        1,
        [f"Substitute x = {rational_text(x0)} into f'(x): f'({rational_text(x0)}) = {rational_text(m)}."],
        [
            ("gave-y-value-for-gradient", f0),
            ("substitution-sign-error", m_at_neg_x0),
            ("sign-reversed-result", -m),
            ("inverted-gradient", 1 / m),
        ],
        follow_through=at("a", x0),
    )

    # (c) the line -----------------------------------------------------------------------------------------------
    kind = "normal" if normal else "tangent"
    if normal:
        wrong_c = [
            ("tangent-instead-of-normal", m * X + (f0 - m * x0)),
            ("normal-slope-reciprocal-only", (1 / m) * X + (f0 - x0 / m)),
            ("normal-slope-negated-only", -m * X + (f0 + m * x0)),
            ("y-value-as-intercept", nslope * X + f0),
        ]
        ft_c = f"-1/part_b*(x - {paren(x0)}) + {paren(f0)}"
        work_c = [
            f"The normal is perpendicular to the tangent, so its gradient is -1/{rational_text(m)} = {rational_text(nslope)}.",
            f"y - {paren(f0)} = {rational_text(nslope)}(x - {paren(x0)}), so y = {pretty(normal_line)}.",
        ]
    else:
        wrong_c = [
            ("used-perpendicular-gradient", nslope * X + (f0 - nslope * x0)),
            ("forgot-point-y-value", m * (X - x0)),
            ("sign-slip-point-slope", m * (X + x0) + f0),
            ("y-value-as-intercept", m * X + f0),
        ]
        ft_c = f"part_b*(x - {paren(x0)}) + {paren(f0)}"
        work_c = [
            f"The tangent passes through ({rational_text(x0)}, {rational_text(f0)}) with gradient {rational_text(m)}.",
            f"y - {paren(f0)} = {rational_text(m)}(x - {paren(x0)}), so y = {pretty(tangent)}.",
        ]
    part_c = expr_part(
        part_prompt(
            "Hence find the equation of the "
            + kind
            + " to the graph of {yf} at the point where {xv}, in the form {form}.",
            yf=yf,
            xv=xv,
            form=("y = mx + c", "y = mx + c"),
        ),
        line,
        2,
        work_c,
        wrong_c,
        follow_through=ft_c,
        prefix="y = ",
    )

    # (d) one number about the line ---------------------------------------------------------------------------
    c_ft, m_ft = line_intercept_ft("c"), line_slope_ft("c")
    if variant == "tangent_x":
        x_int = -c0 / slope
        part_d = fraction_part(
            "Find the x-coordinate of the point where this tangent meets the x-axis.",
            x_int,
            1,
            [f"Put y = 0: 0 = {rational_text(slope)}x + {paren(c0)}, so x = {rational_text(x_int)}."],
            [
                ("sign-error-moving-term", c0 / slope),
                ("forgot-to-divide-by-coefficient", -c0),
                ("inverted-ratio", -slope / c0),
                ("intercepts-confused", c0),
            ],
            follow_through=f"-{c_ft}/{m_ft}",
        )
    elif variant == "tangent_area":
        x_int = -c0 / slope
        area = abs(c0 * x_int) / 2
        part_d = fraction_part(
            "Find the area of the triangle enclosed by this tangent and the two coordinate axes.",
            area,
            2,
            [
                f"The tangent meets the axes at (0, {rational_text(c0)}) and ({rational_text(x_int)}, 0).",
                f"Area = 1/2 × {rational_text(abs(x_int))} × {rational_text(abs(c0))} = {rational_text(area)}.",
            ],
            [
                ("forgot-half-in-triangle-area", abs(c0 * x_int)),
                ("intercept-and-gradient-confused", abs(c0 * slope) / 2),
                ("adds-instead-of-multiplies", (abs(c0) + abs(x_int)) / 2),
            ],
            follow_through=f"{c_ft}**2/(2*{m_ft}*sign({m_ft}))",
            sensible_range=(0, None),
        )
    else:  # normal_y
        part_d = fraction_part(
            "Find the y-coordinate of the point where this normal meets the y-axis.",
            c0,
            1,
            [f"Put x = 0 in the normal: y = {rational_text(c0)}."],
            [
                ("tangent-instead-of-normal", tangent.subs(X, 0)),
                ("y-value-as-intercept", f0),
                ("intercepts-confused", -c0 / nslope),
                ("sign-error-moving-term", -c0),
            ],
            follow_through=line_intercept_ft("c"),
        )
    return [part_b, part_c, part_d]
