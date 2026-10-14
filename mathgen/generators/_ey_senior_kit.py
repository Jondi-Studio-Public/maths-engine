"""Small helpers shared by the senior figure builds in the ``ey_senior_*`` generators.

Not a generator module (the leading underscore keeps the package loader from importing it). It holds:

* sampling a function of integer-valued shapes onto a ``cartesian`` grid so that every integer ``x`` is an exact
  sample, and reading those samples back (``curve_map``), which is what the figure readers use;
* equation and signed-number text (``arg``, ``signed``);
* a numeric exam part with its misconception distractors (``num_part``), so every exam chain is built the same way.

No figure kind is added here: the pictures are ``cartesian``, ``dot_plot`` and ``histogram`` figures.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Callable, Iterable

from .. import figures
from ..core import GRADER_TOLERANCE, pick_distractors, rational_text, round_half_up, stated_precision_string
from ._sr2b_kit import TooFewDistractors, number_fields, pick, retry_short  # noqa: F401  (re-exported)

LO, HI = -8, 8  # the window for the transformation graphs: unit gridlines, labelled every second one
STEP = 12  # samples per unit of x, so every integer x (and every half and third) is an exact sample


def sample(fn: Callable[[float], float], lo: int = LO, hi: int = HI, ylo: int = LO, yhi: int = HI, label: str = ""):
    """A curve for a ``cartesian`` figure: ``fn`` sampled every 1/12 on [lo, hi], clipped to [ylo, yhi]."""
    c = {"pieces": figures.sample_curve(fn, lo, hi, (ylo, yhi), n=(hi - lo) * STEP + 1)}
    if label:
        c["label"] = label
    return c


def curve_map(curve: dict) -> dict:
    """``{Fraction(x): y}`` over every sample of a curve. Samples are multiples of 1/12, so the keys are exact."""
    return {Fraction(round(x * STEP), STEP): y for piece in curve["pieces"] for x, y in piece}


def by_label(fig: dict) -> dict:
    """The curves of a figure by their label."""
    return {c["label"]: c for c in fig["curves"] if c.get("label")}


def near(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(a - b) < tol


def signed(n) -> str:
    """A whole number with its sign: ``+ 3`` or ``- 3``."""
    n = int(n)
    return f"+ {n}" if n >= 0 else f"- {-n}"


def arg(var: str, n) -> str:
    """``x + 2`` / ``x - 2`` (and just ``x`` for 0)."""
    n = int(n)
    return var if n == 0 else f"{var} {signed(n)}"


def frac_text(v) -> str:
    """A rational as the options show it: ``3``, ``-2``, ``1/2``."""
    return rational_text(Fraction(v))


def num_part(
    prompt: str,
    value,
    wrong: Iterable,
    *,
    working: list[str],
    marks: int = 1,
    follow: str | None = None,
    places: int = 0,
) -> dict:
    """One numeric part of an exam chain. ``wrong`` are the numbers a misconception gives (order = priority)."""
    fields = number_fields(Fraction(value), places)
    texts = [stated_precision_string(round_half_up(Fraction(w), places), places) for w in wrong]
    part = dict(
        prompt=prompt,
        marks=marks,
        working=working,
        distractors=pick(fields["answer_text"], texts),
        **fields,
    )
    if follow:
        part["follow_through"] = follow
    return part


def mcq(key: str, wrong: Iterable[str], *, minimum: int = 2, count: int = 3) -> list[str]:
    """Up to ``count`` distinct wrong options (never the key), in priority order; fewer than ``minimum`` redraws."""
    out = pick_distractors(key, [str(w) for w in wrong], count)
    if len(out) < minimum:
        raise TooFewDistractors(f"too few distractors for {key}: {list(wrong)}")
    return out


__all__ = [
    "GRADER_TOLERANCE",
    "HI",
    "LO",
    "STEP",
    "arg",
    "by_label",
    "curve_map",
    "frac_text",
    "mcq",
    "near",
    "num_part",
    "retry_short",
    "sample",
    "signed",
]
