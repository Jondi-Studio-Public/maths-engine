"""Generates `p14e_residual_diagnostic` questions (`ucv1-regression-residual-diagnostic`): a diagnostic read from
a stated regression's residuals.

``ucv1-regression-residual-diagnostic``.

Kernel: given a fitted line ``y = mx + b`` (stated directly, or -- at the
``exam`` tier -- exactly determined by two points that lie ON it) and one
OBSERVED point ``(x0, y0)`` (never on the fitted line's own data set), compute
the residual ``e = y0 - (m*x0 + b)`` and classify the point as
``"above the line"`` / ``"below the line"`` / ``"on the line"`` by the sign of
``e``. The line is always given or exactly determined by two points on it,
never fit from noisy data -- this is the boundary against ``least-squares-line``
(which FITS m, b from a small raw data set and grades the fitted line itself)
and against the coefficient-fitting kernel of ``ucv1-multivariate-regression``:
neither of those ever asks about a *different* point's residual
relative to an already-determined line, and this generator never fits
coefficients from data at all.

Answer space by construction: exactly three fixed
labels, always all three offered as options. The key is drawn FIRST with
``rng.choice`` and the observed point's y-value is constructed to realise
that exact residual sign -- never the reverse (draw numbers, hope the sign
comes out how we want).

This generator instead draws EACH of the two points' classes independently and uniformly from all three labels
at the hard tier too, so "on the line" remains reachable (and offered as the key whenever it is drawn) at
every tier, and the class-balance measurement stays a genuine three-way split throughout.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..core import (
    check,
    check_equivalent,
    fmt_signed,
    generator,
    latex_answer,
    paren_negative,
    phrase,
    pick_distractors,
)

LABELS = ("above the line", "below the line", "on the line")


def _classify(residual) -> str:
    if residual > 0:
        return "above the line"
    if residual < 0:
        return "below the line"
    return "on the line"


def _other_labels(label: str) -> list[str]:
    return [l for l in LABELS if l != label]


def _construct(y_pred: Fraction, label: str, rng, lo: int = 1, hi: int = 6) -> Fraction:
    """Build an observed y-value realising `label` against `y_pred`."""
    if label == "above the line":
        return y_pred + rng.randint(lo, hi)
    if label == "below the line":
        return y_pred - rng.randint(lo, hi)
    return y_pred


def _fmt_coef(m) -> str:
    """The slope as a bare coefficient value: "-4", "1/3", "-7/4" -- a proper
    fraction is left unparenthesised here (parens are added by the callers
    that attach it to a variable)."""
    return str(int(m)) if isinstance(m, Fraction) and m.denominator == 1 else str(m)


def _fmt_slope_term(m) -> str:
    """Unambiguous "mx" text: an integer coefficient renders the way a
    student would write it ("x", "-x", "4x" -- via `fmt_signed`), a proper
    fraction is parenthesised ("(-1/3)x", never the ambiguous "-1/3x")."""
    if isinstance(m, Fraction) and m.denominator != 1:
        return f"({_fmt_coef(m)})x"
    value = int(m) if isinstance(m, Fraction) else m
    return fmt_signed(value, "x")


def _fmt_line(m, b: int) -> str:
    if b == 0:
        return f"y = {_fmt_slope_term(m)}"
    sign = "+" if b >= 0 else "-"
    return f"y = {_fmt_slope_term(m)} {sign} {abs(b)}"


def _fmt_num(v) -> str:
    """Fraction/int -> student-facing text ("5", "17/3", "-2")."""
    return str(v)


# ---------------------------------------------------------------------------
# easy: m, b, x0 small integers; a single observed point.
# ---------------------------------------------------------------------------

_EASY_TEMPLATES = (
    "A line is given by {line}. An observed data point is ({x0}, {y0}). "
    "Is the point above the line, below the line, or on the line?",
    "The fitted line is {line}. It predicts values of y for each x. For the "
    "observed point ({x0}, {y0}), classify its residual: above the line, "
    "below the line, or on the line?",
)


def _easy(rng):
    m = rng.choice([v for v in range(-6, 7) if v != 0])
    b = rng.randint(-6, 6)
    x0 = rng.randint(-6, 6)
    label = rng.choice(LABELS)
    y_pred = Fraction(m) * x0 + b
    y0 = _construct(y_pred, label, rng)

    # Independent recheck: recompute the residual from the rendered numbers
    # alone (no branch on which label construction picked) and confirm its
    # sign matches the intended key.
    e = y0 - (Fraction(m) * x0 + b)
    check(_classify(e) == label, "easy residual classification mismatch")

    line = _fmt_line(m, b)
    text = phrase(rng, _EASY_TEMPLATES, line=line, x0=x0, y0=_fmt_num(y0))
    others = _other_labels(label)
    return dict(
        text=text,
        answer=label,
        answer_text=label,
        answer_latex=latex_answer(label, plain=True),
        working=[
            f"Predicted value at x0={x0}: {_fmt_coef(m)} × {paren_negative(x0)} "
            f"{'+' if b >= 0 else '-'} {abs(b)} = {y_pred}.",
            f"Residual e = y0 - predicted = {y0} - {paren_negative(y_pred)} = {e}, which is "
            f"{'zero' if e == 0 else ('positive' if e > 0 else 'negative')}, so the point is {label}.",
        ],
        distractors=pick_distractors(label, others),
    )


# ---------------------------------------------------------------------------
# medium: m a simple fraction, x0 larger -- fraction arithmetic throughout.
# ---------------------------------------------------------------------------

_MEDIUM_TEMPLATES = (
    "A line is given by {line}. An observed data point is ({x0}, {y0}). "
    "Is the point above the line, below the line, or on the line?",
    "A regression line is {line}. For the observed point ({x0}, {y0}), "
    "classify its residual: above the line, below the line, or on the line?",
)


def _medium(rng):
    d = rng.choice([2, 3, 4])
    a = rng.choice([v for v in range(-3 * d, 3 * d + 1) if v != 0 and v % d != 0])
    m = Fraction(a, d)
    b = rng.randint(-6, 6)
    x0 = rng.choice([v for v in range(-15, 16) if v != 0])
    label = rng.choice(LABELS)
    y_pred = m * x0 + b
    y0 = _construct(y_pred, label, rng)

    e = y0 - (m * x0 + b)
    check(_classify(e) == label, "medium residual classification mismatch")

    line = _fmt_line(m, b)
    text = phrase(rng, _MEDIUM_TEMPLATES, line=line, x0=x0, y0=_fmt_num(y0))
    others = _other_labels(label)
    return dict(
        text=text,
        answer=label,
        answer_text=label,
        answer_latex=latex_answer(label, plain=True),
        working=[
            f"Predicted value at x0={x0}: {_fmt_coef(m)} × {paren_negative(x0)} "
            f"{'+' if b >= 0 else '-'} {abs(b)} = {y_pred}.",
            f"Residual e = y0 - predicted = {y0} - {paren_negative(y_pred)} = {e}, so the point is {label}.",
        ],
        distractors=pick_distractors(label, others),
    )


# ---------------------------------------------------------------------------
# hard: two points listed, classify a NAMED one (label match, not size).
# ---------------------------------------------------------------------------

_HARD_TEMPLATES = (
    "A fitted line is given by {line}. Two observed points are A = ({xa}, {ya}) "
    "and B = ({xb}, {yb}). Classify point {target}'s residual relative to the "
    "line: above the line, below the line, or on the line?",
    "The line {line} is fitted to some data. Two further observed points are "
    "A = ({xa}, {ya}) and B = ({xb}, {yb}). Is point {target} above the line, "
    "below the line, or on the line?",
)


def _hard(rng):
    m = rng.choice([v for v in range(-8, 9) if v != 0])
    b = rng.randint(-8, 8)
    xa = rng.randint(-9, 9)
    xb = rng.choice([v for v in range(-9, 10) if v != xa])
    label_a = rng.choice(LABELS)
    label_b = rng.choice(LABELS)
    pred_a = Fraction(m) * xa + b
    pred_b = Fraction(m) * xb + b
    ya = _construct(pred_a, label_a, rng)
    yb = _construct(pred_b, label_b, rng)

    ea = ya - (Fraction(m) * xa + b)
    eb = yb - (Fraction(m) * xb + b)
    check(_classify(ea) == label_a and _classify(eb) == label_b, "hard residual classification mismatch")

    target = rng.choice(("A", "B"))
    target_label = label_a if target == "A" else label_b

    line = _fmt_line(m, b)
    text = phrase(rng, _HARD_TEMPLATES, line=line, xa=xa, ya=_fmt_num(ya), xb=xb, yb=_fmt_num(yb), target=target)
    others = _other_labels(target_label)
    return dict(
        text=text,
        answer=target_label,
        answer_text=target_label,
        answer_latex=latex_answer(target_label, plain=True),
        working=[
            f"Point {target} = ({xa if target == 'A' else xb}, {ya if target == 'A' else yb}); "
            f"predicted value there is {pred_a if target == 'A' else pred_b}.",
            f"Residual for point {target} is {paren_negative(ea if target == 'A' else eb)}, "
            f"so point {target} is {target_label}.",
        ],
        distractors=pick_distractors(target_label, others),
    )


# ---------------------------------------------------------------------------
# exam: line given via two points ON it -- derive m, b first, then classify.
# ---------------------------------------------------------------------------

_EXAM_TEMPLATES = (
    "A line passes exactly through the points ({x1}, {y1}) and ({x2}, {y2}). "
    "An observed data point is ({x0}, {y0}). First find the line's equation, "
    "then classify the point: is it above the line, below the line, or on "
    "the line?",
    "A fitted line is exactly determined by the two points ({x1}, {y1}) and "
    "({x2}, {y2}). For the observed point ({x0}, {y0}), determine the line's "
    "equation and classify the point's residual: above the line, below the "
    "line, or on the line?",
)


def _exam(rng):
    d = rng.choice([1, 2, 3])
    a = rng.choice([v for v in range(-4 * d, 4 * d + 1) if v != 0 and (d == 1 or v % d != 0)])
    m = Fraction(a, d)
    b = rng.randint(-6, 6)
    k1 = rng.choice([v for v in range(-4, 5) if v != 0])
    k2 = rng.choice([v for v in range(-4, 5) if v != 0 and v != k1])
    x1, x2 = d * k1, d * k2
    y1 = m * x1 + b
    y2 = m * x2 + b
    check(y1.denominator == 1 and y2.denominator == 1, "exam two-point y-values not exact integers")

    # Independent check: re-derive m, b from the two RENDERED points alone,
    # via a linear solve of the two-point system -- not however construction
    # built them.
    m_sym, b_sym = sympy.symbols("m_sym b_sym")
    system = sympy.Matrix([[x1, 1], [x2, 1]])
    rhs = sympy.Matrix([int(y1), int(y2)])
    solution = system.solve(rhs)
    check_equivalent(solution[0], sympy.Rational(a, d), "exam derived slope mismatch")
    check_equivalent(solution[1], sympy.Integer(b), "exam derived intercept mismatch")

    x0 = rng.choice([v for v in range(-15, 16) if v not in (x1, x2)])
    label = rng.choice(LABELS)
    y_pred = m * x0 + b
    y0 = _construct(y_pred, label, rng)

    e = y0 - (m * x0 + b)
    check(_classify(e) == label, "exam residual classification mismatch")

    text = phrase(rng, _EXAM_TEMPLATES, x1=x1, y1=_fmt_num(y1), x2=x2, y2=_fmt_num(y2), x0=x0, y0=_fmt_num(y0))
    others = _other_labels(label)
    return dict(
        text=text,
        answer=label,
        answer_text=label,
        answer_latex=latex_answer(label, plain=True),
        working=[
            f"Slope m = (y2 - y1)/(x2 - x1) = ({paren_negative(y2)} - {paren_negative(y1)})/"
            f"({paren_negative(x2)} - {paren_negative(x1)}) = {m}; "
            f"intercept b = {paren_negative(y1)} - {paren_negative(m)} × {paren_negative(x1)} = {b}.",
            f"Predicted value at x0={x0} is {y_pred}; residual e = {y0} - {paren_negative(y_pred)} = {e}, "
            f"so the point is {label}.",
        ],
        distractors=pick_distractors(label, others),
    )


_TIER_BUILDERS = {
    "direct": _easy,
    "fraction": _medium,
    "named_point": _hard,
    "two_point_line": _exam,
}


@generator(
    "p14e_residual_diagnostic",
    level=16,
    presets={
        "easy": dict(kind="direct"),
        "medium": dict(kind="fraction"),
        "hard": dict(kind="named_point"),
        "exam": dict(kind="two_point_line"),
    },
)
def p14e_residual_diagnostic(rng, kind: str = "direct"):
    """Diagnose a small regression residual: classify an observed point's
    residual relative to a fitted (or exactly two-point-determined) line as
    above / below / on the line. See module docstring for the tier design
    and the boundary against ``least-squares-line`` / ``ucv1-multivariate-
    regression``."""
    builder = _TIER_BUILDERS[kind]
    return builder(rng)
