"""Small shared helpers for the `exd_x_c3_*` exam-tier calculus generators.

Not a generator itself. Each exam question is built backwards from a chosen
answer; when a random draw makes a wrong-answer candidate coincide with the
key (for example a dropped term that happens to vanish), the draw is repeated
rather than a weaker option being offered.
"""

from __future__ import annotations

import re

from ..core import VerificationError, symbolically_equal

_INVERSE_NAMES = {"atan": "arctan", "asin": "arcsin", "acos": "arccos"}
_INVERSE_RE = re.compile(r"\b(atan|asin|acos)\b")


class Redraw(Exception):
    """Raised inside a builder to ask for a fresh random draw."""


def typed(expr) -> str:
    """Answer text: `**` becomes `^`, sympy's inverse-trig names become arcsin/arccos/arctan."""
    return _INVERSE_RE.sub(lambda m: _INVERSE_NAMES[m.group(1)], str(expr).replace("**", "^"))


def prose(expr) -> str:
    """Prompt text: as `typed`, with the multiplication sign shown as a middle dot."""
    return typed(expr).replace("*", "·")


def draw(build, rng, attempts: int = 200) -> dict:
    """Call `build(rng)` until it does not raise `Redraw`."""
    for _ in range(attempts):
        try:
            return build(rng)
        except Redraw:
            continue
    raise VerificationError("no acceptable draw found")


def expr_distractors(answer, candidates, n: int = 3) -> list[str]:
    """The first `n` candidate expressions that differ from `answer` and from each other, as typed text.

    Raises `Redraw` when fewer than `n` qualify, so a builder never ships a distractor equal to the key.
    """
    out: list[str] = []
    kept = []
    for cand in candidates:
        if symbolically_equal(cand, answer) or any(symbolically_equal(cand, k) for k in kept):
            continue
        kept.append(cand)
        out.append(typed(cand))
        if len(out) == n:
            return out
    raise Redraw
