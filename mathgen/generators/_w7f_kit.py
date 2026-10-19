"""Small helpers shared by the ``w7f`` figure exam chains (scale drawings, plans, solids and views).

Not a generator module (the leading underscore keeps ``mathgen.generators`` from importing it as one). Wraps the
``exd_x_w12a`` part builders so a chain can offer candidate wrong answers that a draw could not make, finishes a
chain dict only when no part came up short and no answer is shown by the figure, and splits a composed multi-part
text for the figure readers.
"""

from __future__ import annotations

import re

from mathgen.core import VerificationError, attempts

from . import _space_kit as kit
from ._exd_x_w12a_kit import mcq_part as _mcq_part
from ._exd_x_w12a_kit import number_part

__all__ = ["mcq_part", "number_part", "split_parts", "finish_parts", "retry", "pick"]

_LABEL = re.compile(r"\(([a-z])\) ")


def pick(rng, options):
    """One of ``options``, chosen with ``rng`` (a stem has several honest wordings)."""
    return options[rng.randrange(len(options))]


def split_parts(text: str) -> dict[str, str]:
    """The prompt of each part of a composed multi-part text, by label ("(a) ... (b) ...", up to part z)."""
    pieces = _LABEL.split(text)
    return {pieces[i]: pieces[i + 1].strip() for i in range(1, len(pieces) - 1, 2)}


def mcq_part(prompt, key, marks, working, wrong, **kw):
    """``mcq_part`` that skips options a draw could not make (given as ``(None, tag)``)."""
    return _mcq_part(prompt, key, marks, working, [(t, tag) for t, tag in wrong if t is not None], **kw)


def finish_parts(fig, stem, parts, *, lower: bool = False):
    """The chain dict, or None when any part found too few distractors or the figure shows an answer.

    ``lower`` also rejects an answer shown in lower case (a landmark or room name in a label).
    """
    if any(p is None for p in parts):
        return None
    for p in parts:
        text = p["answer_text"]
        if kit.leaks(fig, text) or (lower and kit.leaks(fig, text.lower())):
            return None
    return dict(text=stem, parts=parts, figure=fig)


def retry(rng, build, what: str, tries: int = 300):
    """``build(rng)`` until it returns a chain (a draw with no usable arrangement raises or returns None)."""
    for _ in attempts(tries, what):
        try:
            out = build(rng)
        except VerificationError:
            continue
        if out is not None:
            return out
    raise AssertionError("unreachable")
