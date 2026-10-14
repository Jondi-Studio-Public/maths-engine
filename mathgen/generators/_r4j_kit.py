"""Shared helpers for the second-year discrete mathematics builds (`r4j_*` generators): generating functions, Prüfer
codes, divisibility posets, Dijkstra's algorithm and forced recurrences.

* `say`: a sentence written once as plain text and LaTeX together, so a fraction or a power is set as maths;
* `comb`, `neg_binomial`: binomial coefficients that are zero outside their range, the way a series coefficient is;
* `keep_distinct`: tagged distractors from `(value, misconception)` pairs, dropping the answer and repeats.
"""

from __future__ import annotations

from math import comb as _comb

from ..answers import Distractor
from ..core import _escape_prose


def say(*parts) -> tuple[str, str]:
    """``(text, latex)`` for a sentence. A part is prose (a string), or ``(plain, tex)`` for maths: the learner reads the
    plain Unicode and the page sets the TeX."""
    text, latex = [], []
    for part in parts:
        if isinstance(part, tuple):
            plain, tex = part
            text.append(plain)
            latex.append(tex + "{}")
        elif part:
            text.append(part)
            latex.append(r"\text{" + _escape_prose(part) + "}")
    return "".join(text), "".join(latex)


def comb(n: int, r: int) -> int:
    """C(n, r), and 0 when r is outside 0..n (a coefficient that does not exist)."""
    return _comb(n, r) if 0 <= r <= n else 0


def neg_binomial(k: int, r: int) -> int:
    """The coefficient of x^k in 1/(1 - x)^r: C(k + r - 1, r - 1), and 0 for k < 0."""
    return comb(k + r - 1, r - 1) if k >= 0 else 0


def keep_distinct(answer, candidates, *, key=lambda v: v):
    """Tagged distractors from ``(value, misconception, text)`` triples, in priority order, without the answer's value
    or a value already offered. ``key`` maps a value to something comparable (a tuple for a list)."""
    seen = {key(answer)}
    out = []
    for value, tag, text in candidates:
        if value is None or key(value) in seen:
            continue
        seen.add(key(value))
        out.append(Distractor(text, tag, value))
    return out
