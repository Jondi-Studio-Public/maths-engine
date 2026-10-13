"""Shared builders for the `exd_x_w12b_*` exam chains: multi-part questions read off a straight-line graph.

Every chain draws one line (or three plotted points) on a coordinate plane whose gridlines are all labelled, so
each value a part needs is read from the figure and nothing the key needs is printed. The builders return part
dicts for ``parts=[...]`` (see ``docs/design/multi-part-questions.md``): a ``fraction`` part for an exact value, a
``pair`` part for a point, an ``expression`` part for an equation ``y = mx + c``. Every distractor is a tagged
``Distractor`` built from one named error and is kept only when it is a different answer from the key.

Line arithmetic is done in real axis units with ``fractions.Fraction``; the figure geometry is the one the
`fg1_*` line-reading generators use.
"""

from __future__ import annotations

from fractions import Fraction as F

from sympy import Rational, Symbol

from .. import figures
from ..answers import Distractor, exact_value
from ..core import VerificationError, attempts, check, latex_answer, pick_distractors, rational_text
from .fg1_line_reading_build import draw_line, eq_text, pick_pair, real_mc, real_pt
from .pv1a_line_figure_build import Window, line_figure

X = Symbol("x")


#: ``follow_through`` text for the slope and the intercept of an earlier expression part ``part_<label>``.
def slope_of(label: str) -> str:
    return f"(Subs(part_{label},(x,),(1,))-Subs(part_{label},(x,),(0,)))"


def icpt_of(label: str) -> str:
    return f"Subs(part_{label},(x,),(0,))"


def rat(v) -> Rational:
    v = F(v)
    return Rational(v.numerator, v.denominator)


def retry(rng, build, what: str, limit: int = 600):
    """The first ``build(rng)`` that does not raise `VerificationError`; the error from running out is the budget's."""
    for _ in attempts(limit, what):
        try:
            return build(rng)
        except VerificationError:
            continue


def line_expr(m, c):
    """The ``answer`` of an equation part: ``m*x + c`` as a SymPy expression string."""
    return str(rat(m) * X + rat(c))


def frac_text(v) -> str:
    return rational_text(rat(v))


def frac_part(prompt, value, marks, working, wrong, *, follow=None, sensible=None, keep=3):
    """A ``fraction`` part. `wrong` is ``[(value, misconception id), ...]`` in priority order."""
    value = F(value)
    key = frac_text(value)
    cands = [Distractor(frac_text(v), tag, exact_value(F(v))) for v, tag in wrong if F(v) != value]
    picked = pick_distractors(key, cands, keep)
    check(len(picked) >= 2, "fewer than two usable distractors")
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer=exact_value(value),
        answer_text=key,
        answer_latex=latex_answer(key),
        marks=marks,
        working=working,
        distractors=picked,
    )
    if sensible is not None:
        part["sensible_range"] = sensible
    if follow:
        part["follow_through"] = follow
    return part


def point_text(x, y) -> str:
    return f"({frac_text(x)}, {frac_text(y)})"


def pair_part(prompt, point, marks, working, wrong, *, keep=3):
    """A ``pair`` part whose key is the integer point `point`. `wrong` is ``[((x, y), id), ...]``."""
    x, y = point
    key = point_text(x, y)
    cands = [Distractor(point_text(a, b), tag, {"x": float(a), "y": float(b)}) for (a, b), tag in wrong]
    picked = pick_distractors(key, cands, keep)
    check(len(picked) >= 2, "fewer than two usable distractors")
    return dict(
        prompt=prompt,
        answer_kind="pair",
        answer={"x": int(x), "y": int(y)},
        answer_text=key,
        answer_latex=key,
        marks=marks,
        working=working,
        distractors=picked,
    )


def expr_part(prompt, m, c, marks, working, wrong, *, follow=None, keep=3):
    """An ``expression`` part whose key is ``y = mx + c``. `wrong` is ``[((m, c), id), ...]``."""
    key = eq_text(m, c)
    seen = {(F(m), F(c))}
    cands = []
    for (wm, wc), tag in wrong:
        if (F(wm), F(wc)) in seen:
            continue
        seen.add((F(wm), F(wc)))
        cands.append(Distractor(eq_text(wm, wc), tag, None))
    picked = pick_distractors(key, cands, keep)
    check(len(picked) >= 2, "fewer than two usable distractors")
    part = dict(
        prompt=prompt,
        answer_kind="expression",
        answer=line_expr(m, c),
        answer_text=key,
        answer_latex=latex_answer(key),
        marks=marks,
        working=working,
        distractors=picked,
    )
    if follow:
        part["follow_through"] = follow
    return part


def beyond_window(rng, win: Window, den: int, *, lo: int = 6, hi: int = 36, sides=(-1, 1)) -> int:
    """An integer x (a multiple of `den`, so a line of gradient a/den has a whole-number y there) outside the window."""
    x_lo, x_hi = win.x_range
    options = []
    for side in sides:
        for k in range(1, hi + 1):
            x = side * k * den
            if lo <= abs(x) <= hi and (x > x_hi or x < x_lo):
                options.append(x)
    check(bool(options), "no x outside the window")
    return rng.choice(options)


def no_leak(fig: dict, parts: list[dict]) -> None:
    """Raise `VerificationError` when a figure label or its alt text shows any part's answer."""
    try:
        figures.assert_no_leak(fig, [p["answer_text"] for p in parts])
    except AssertionError as exc:
        raise VerificationError(str(exc)) from exc


def marked(win: Window, pts, names=None) -> list[dict]:
    """Figure ``points`` entries for grid points `pts`, labelled with `names` when given."""
    out = []
    for i, p in enumerate(pts):
        entry = {"at": list(real_pt(win, p))}
        if names:
            entry["label"] = names[i]
        out.append(entry)
    return out


__all__ = [
    "F",
    "VerificationError",
    "X",
    "beyond_window",
    "check",
    "draw_line",
    "eq_text",
    "expr_part",
    "frac_part",
    "frac_text",
    "icpt_of",
    "line_figure",
    "marked",
    "no_leak",
    "pair_part",
    "pick_pair",
    "point_text",
    "rat",
    "real_mc",
    "real_pt",
    "retry",
    "slope_of",
]
