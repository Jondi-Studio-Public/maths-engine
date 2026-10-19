"""Shared helpers for the senior mechanics-by-calculus builds (`r3d_*_build.py`).

Every quantity is exact (a `Fraction`, or a SymPy value turned into a `Fraction` with forty digits) until the moment it
is shown, so a displayed answer is one rounding of an exact value. A number is shown at a stated number of decimal
places, never fewer, because the number grader reads its tolerance from the places the canonical answer displays
(see `core.stated_precision_string`).

`G` is the acceleration due to gravity the whole batch uses, 9.8 m/s². Every prompt that needs it says so.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..answers import Distractor, exact_value
from ..core import (
    check,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    rounded_decimal_string,
    stated_precision_string,
    stated_precision_tolerance,
)

G = Fraction(98, 10)
G_NOTE = "Use g = 9.8 m/s²."

#: The symbols the expression answers are written in.
T, X = sympy.symbols("t x")


def exact(expr) -> Fraction:
    """A SymPy number (or plain number) as a `Fraction` good to forty significant digits."""
    if isinstance(expr, Fraction):
        return expr
    return Fraction(str(sympy.N(sympy.sympify(expr), 40)))


def shown(value, dp: int) -> Fraction:
    """`value` rounded half-up to `dp` places."""
    return round_half_up(exact(value), dp)


def text_of(value, dp: int) -> str:
    """`value` shown at exactly `dp` decimal places."""
    return stated_precision_string(shown(value, dp), dp)


def is_exact(value, dp: int) -> bool:
    """Whether `value` already sits on the `dp`-place grid, so showing it rounds nothing."""
    return (exact(value) * 10**dp).denominator == 1


def fit_dp(value, lo: int = 1, hi: int = 3) -> int | None:
    """The fewest places in `lo..hi` at which the exact `value` is shown with no rounding, else None."""
    for dp in range(lo, hi + 1):
        if is_exact(value, dp):
            return dp
    return None


def sqrt_of(value) -> Fraction:
    """The square root of a non-negative rational, as a `Fraction` good to forty digits. A negative argument gives NaN,
    which carries through arithmetic and is dropped when a wrong option is built from it (a slip that leaves no real
    number is not an option)."""
    f = Fraction(value)
    if f < 0:
        return float("nan")
    return exact(sympy.sqrt(sympy.Rational(f.numerator, f.denominator)))


def clear_of_boundary(value, dp: int, margin=Fraction(1, 20)) -> bool:
    """Whether `value` is at least `margin` of a unit in the last place away from a rounding boundary."""
    scaled = abs(exact(value)) * 10**dp
    return abs((scaled % 1) - Fraction(1, 2)) >= margin


def dec(value, places: int = 4) -> str:
    """`value` as the shortest decimal of at most `places` places, for use inside a prompt, a worked step or a wrong
    option. A value that does not terminate in that many places is rounded, which only a wrong option may rely on."""
    return rounded_decimal_string(Fraction(value), places)


def pi_text(multiple) -> str:
    """The angle `multiple` × π written for a stem: "π/6", "5π/12", "2π", "π"."""
    f = Fraction(multiple)
    if f == 0:
        return "0"
    sign = "-" if f < 0 else ""
    f = abs(f)
    num = "" if f.numerator == 1 else str(f.numerator)
    den = "" if f.denominator == 1 else f"/{f.denominator}"
    return f"{sign}{num}π{den}"


def phase_text(multiple) -> str:
    """The phase `multiple` × π as it follows 'nt' in an argument: ' + π/6' or ' - π/6'."""
    f = Fraction(multiple)
    return f" + {pi_text(f)}" if f >= 0 else f" - {pi_text(-f)}"


def tagged(value, dp: int, misconception: str) -> Distractor:
    """A wrong numeric option: `value` shown at `dp` places and tagged with the error that gives it."""
    return Distractor(text_of(value, dp), misconception, exact_value(shown(value, dp)))


def wrong_options(answer_text: str, wrong, dp: int, *, want: int = 2, nonneg: bool = False) -> list[Distractor]:
    """The tagged options for `wrong`, a list of `(value, misconception)`, with every one that the grader would
    accept as the answer (within its stated precision) or that repeats another dropped."""
    tolerance = stated_precision_tolerance(dp) if dp > 0 else 0
    real_answer = Fraction(answer_text) != 0
    # An option that rounds to zero beside a non-zero answer reads as an obvious slip, not as a wrong method.
    kept = []
    for v, tag in wrong:
        try:
            if real_answer and shown(v, dp) == 0:
                continue
            if nonneg and shown(v, dp) < 0:  # a speed or a length is never negative
                continue
        except (ValueError, TypeError, ZeroDivisionError):  # a slip that leaves no real number (a root of a negative)
            continue
        kept.append((v, tag))
    picked = pick_distractors(answer_text, [tagged(v, dp, tag) for v, tag in kept], tolerance=tolerance)
    check(
        len(picked) >= want,
        f"only {len(picked)} usable distractors for {answer_text} from {[(tag, str(v)[:12]) for v, tag in kept]}",
    )
    return picked


def number_fields(value, dp: int, wrong, *, nonneg: bool = False) -> dict:
    """The answer fields of a single-answer `number` question."""
    text = text_of(value, dp)
    return dict(
        answer=numeric_answer(shown(value, dp)) if dp == 0 else float(shown(value, dp)),
        answer_text=text,
        answer_latex=latex_answer(text),
        answer_kind="number",
        distractors=wrong_options(text, wrong, dp, nonneg=nonneg),
    )


def number_part(prompt, value, dp, marks, working, wrong, *, follow=None, sensible=None, nonneg: bool = False) -> dict:
    """One `number` part of an exam chain."""
    part = dict(prompt=prompt, marks=marks, working=working, **number_fields(value, dp, wrong, nonneg=nonneg))
    if follow:
        part["follow_through"] = follow
    if sensible is not None:
        part["sensible_range"] = sensible
    return part


# ------------------------------------------------------------------------------------------------ expressions


def machine(expr) -> str:
    """A SymPy expression as the parseable string kept in `answer` (the display form is `answer_text`)."""
    return str(sympy.sympify(expr)).replace("**", "^")


def differs(left, right, symbol, points=(Fraction(3, 7), Fraction(11, 5), Fraction(17, 4))) -> bool:
    """Whether two expressions in `symbol` give a different value at any of a few awkward sample points."""
    for p in points:
        a = sympy.N(sympy.sympify(left).subs(symbol, sympy.Rational(p.numerator, p.denominator)), 30)
        b = sympy.N(sympy.sympify(right).subs(symbol, sympy.Rational(p.numerator, p.denominator)), 30)
        if abs(a - b) > sympy.Float("1e-9"):
            return True
    return False


def expression_fields(answer, answer_text: str, wrong, symbol) -> dict:
    """The answer fields of an `expression` question. `wrong` is a list of `(sympy expression, text, misconception)`;
    one that has the answer's value (or the answer's text) is dropped."""
    candidates = [Distractor(text, tag, text) for expr, text, tag in wrong if differs(answer, expr, symbol)]
    picked = pick_distractors(answer_text, candidates, 4)
    check(len(picked) >= 2, f"only {len(picked)} usable distractors for {answer_text}")
    return dict(
        answer=machine(answer),
        answer_text=answer_text,
        answer_kind="expression",
        distractors=picked,
    )


def expression_part(prompt, answer, answer_text, marks, working, wrong, symbol, *, follow=None) -> dict:
    """One `expression` part of an exam chain."""
    part = dict(prompt=prompt, marks=marks, working=working, **expression_fields(answer, answer_text, wrong, symbol))
    if follow:
        part["follow_through"] = follow
    return part


# ------------------------------------------------------------------------------------------------ cheap numeric checks


def residual_zero(expr, symbol, points=(Fraction(3, 7), Fraction(11, 5), Fraction(5, 2)), tol: float = 1e-12) -> bool:
    """Whether `expr` (which should be identically zero in `symbol`) is zero at a few awkward points. Much cheaper than
    `sympy.simplify` for exponentials and logarithms, and sound for the smooth expressions used here."""
    return all(
        abs(float(sympy.N(sympy.sympify(expr).subs(symbol, sympy.Rational(p.numerator, p.denominator)), 30))) < tol
        for p in points
    )


def simpson(expr, symbol, a, b, steps: int = 2000) -> float:
    """Composite Simpson's rule for the integral of the SymPy `expr` in `symbol` from `a` to `b`, in floats."""
    f = sympy.lambdify(symbol, expr, "math")
    a, b = float(a), float(b)
    h = (b - a) / steps
    total = f(a) + f(b)
    for i in range(1, steps):
        total += (4 if i % 2 else 2) * f(a + i * h)
    return total * h / 3


def secs(n) -> str:
    """'1 second' or '4 seconds'."""
    n = int(n)
    return f"{n} second" if n == 1 else f"{n} seconds"
