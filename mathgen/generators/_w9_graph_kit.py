"""Shared pieces of the `w9_*` interactive-graph builds: canvases, equation text, tagged distractors that are checked
against the real grader, and parts for exam chains.

Nothing here draws random numbers or registers a topic. The point of the module is the last rule of the interactive
kinds: a distractor is offered only when the grader marks it wrong, and when it is read back against the value it
stands for it is marked right (so diagnosis finds it).
"""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

from .. import figures
from ..answers import Distractor
from ..core import VerificationError, fmt_signed
from ..figure_kinds.canvas import canvas as _canvas
from ..grading import GRADERS

__all__ = [
    "HALF",
    "TooFewDistractors",
    "plane",
    "x_term",
    "line_text",
    "ineq_text",
    "frac_text",
    "point_text",
    "points_text",
    "checked",
    "dist",
    "enough",
    "retry_draw",
    "part",
]

#: The default canvas runs from -HALF to HALF on both axes.
HALF = 8


class TooFewDistractors(VerificationError):
    """A draw whose error candidates collapsed onto the answer or each other; `retry_draw` redraws it."""


def enough(distractors, n: int = 2) -> None:
    if len(distractors) < n:
        raise TooFewDistractors(f"too few distractors: {len(distractors)}")


def retry_draw(rng, build, what: str = "a question with enough distractors"):
    from ..core import attempts

    for _ in attempts(80, what):
        try:
            return build(rng)
        except TooFewDistractors:
            continue
    raise AssertionError  # pragma: no cover


# ---------------------------------------------------------------------------------------------- text


def frac_text(value) -> str:
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def point_text(p) -> str:
    return f"({frac_text(p[0])}, {frac_text(p[1])})"


def points_text(points) -> str:
    return " ".join(point_text(p) for p in points)


def x_term(m) -> str:
    """The x term of ``y = mx + c``, as the first term: ``2x``, ``-x``, ``(2/3)x``, or ``""`` for m = 0."""
    m = Fraction(m)
    if m == 0:
        return ""
    if m.denominator == 1:
        return fmt_signed(int(m), "x")
    sign = "-" if m < 0 else ""
    return f"{sign}({abs(m.numerator)}/{m.denominator})x"


def _tail(c) -> str:
    c = Fraction(c)
    if c == 0:
        return ""
    return f" {'+' if c > 0 else '-'} {frac_text(abs(c))}"


def line_text(m, c, lhs: str = "y") -> str:
    """``y = 2x - 1``, ``y = -x``, ``y = 3``."""
    m, c = Fraction(m), Fraction(c)
    body = x_term(m) + _tail(c) if m != 0 else frac_text(c)
    return f"{lhs} = {body}"


def ineq_text(m, c, op: str, lhs: str = "y") -> str:
    """``y > 2x - 1`` with `op` one of ``< > ≤ ≥``."""
    return line_text(m, c, lhs).replace("=", op, 1)


# ---------------------------------------------------------------------------------------------- figures


def plane(
    *,
    reply: str,
    tools,
    alt: str,
    half: int = HALF,
    x_range=None,
    y_range=None,
    snap=1,
    points=(),
    segments=(),
    lines=(),
    vertex_labels=None,
    max_points=None,
    grid=1,
    title: str = "",
) -> dict:
    """A coordinate plane with a ``canvas``: what is given is drawn, the learner adds the answer."""
    return figures.cartesian(
        x_range=x_range or (-half, half),
        y_range=y_range or (-half, half),
        alt=alt,
        grid=grid,
        points=points,
        segments=segments,
        lines=lines,
        title=title,
        canvas=_canvas(reply, tools, snap, vertex_labels=vertex_labels, max_points=max_points),
    )


# ---------------------------------------------------------------------------------------------- distractors


def checked(kind: str, answer, answer_text: str, reply: str) -> bool:
    """Whether the real grader of `kind` marks `reply` correct against `answer`."""
    stub = SimpleNamespace(answer=answer, answer_text=answer_text, answer_kind=kind, parts=None)
    return bool(GRADERS[kind](stub, reply).correct)


def dist(kind: str, answer, answer_text: str, candidates, n: int = 4) -> list[Distractor]:
    """Tagged distractors for a drawn answer. `candidates` is ``[(text, value, misconception id), ...]`` in priority
    order, `text` the typed form of a wrong reply and `value` the answer it would be right for. Kept only when the grader
    marks `text` wrong for the real answer and right for `value`, and the text is new."""
    seen = {answer_text.strip()}
    out: list[Distractor] = []
    for text, value, tag in candidates:
        if text is None or value is None or text.strip() in seen:
            continue
        if checked(kind, answer, answer_text, text) or not checked(kind, value, text, text):
            continue
        seen.add(text.strip())
        out.append(Distractor(text, tag, value))
        if len(out) == n:
            break
    return out


def part(prompt: str, kind: str, fields: dict, marks: int, working, distractors, **extra) -> dict:
    fields = {k: v for k, v in fields.items() if k != "answer_kind"}
    return dict(
        prompt=prompt, answer_kind=kind, marks=marks, working=list(working), distractors=distractors, **fields, **extra
    )
