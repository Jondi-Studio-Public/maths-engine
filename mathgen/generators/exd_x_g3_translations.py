"""Exam-tier source for the skill "Translations and their representation as column vectors".

The practice tiers add a stated column vector to a point.  Here the translation
is never stated: it has to be recovered from a point and its image, then used in
a composite.  Three forms, chosen by the seed:

* ``repeat``  -- ``T`` maps ``A`` to ``A'``; apply ``T`` several times to ``B``.
* ``inverse`` -- apply the inverse of ``T`` to ``B`` and then a stated column vector.
* ``compose`` -- ``T`` maps ``A`` to ``A'`` and ``S`` maps ``A'`` to ``A''``; apply
  ``T`` then ``S`` to ``B`` (only the net displacement ``A'' - A`` matters).

Built backwards from chosen translation vectors, so every coordinate is an
integer and the key is exact.
"""

from __future__ import annotations

import random

from sympy import Matrix

from ..core import check_equivalent, generator, pick_distractors

_NONZERO = [v for v in range(-7, 8) if v != 0]


def _pt(name: str, p) -> str:
    return f"{name}({p[0]}, {p[1]})"


def _answer_text(p) -> str:
    return f"x = {p[0]}, y = {p[1]}"


def _build(rng: random.Random) -> dict:
    form = rng.choice(["repeat", "inverse", "compose"])
    for _ in range(300):
        a = (rng.randint(-8, 8), rng.randint(-8, 8))
        b = (rng.randint(-8, 8), rng.randint(-8, 8))
        t = (rng.choice(_NONZERO), rng.choice(_NONZERO))
        a1 = (a[0] + t[0], a[1] + t[1])
        tm = Matrix(t)
        if form == "repeat":
            n = rng.randint(2, 4)
            final = (b[0] + n * t[0], b[1] + n * t[1])
            check_equivalent(list(Matrix(b) + n * tm), list(final))
            text = (
                f"The translation T maps {_pt('A', a)} to {_pt("A'", a1)}. "
                f"Point {_pt('B', b)} is translated by T {n} times in succession. "
                f"Find the coordinates of the final image of B."
            )
            cands = [
                (b[0] + t[0], b[1] + t[1]),  # applies T once
                (b[0] + (n - 1) * t[0], b[1] + (n - 1) * t[1]),  # counts the given pair as one application
                (b[0] - n * t[0], b[1] - n * t[1]),  # subtracts
                (a1[0] + n * t[0], a1[1] + n * t[1]),  # starts from the image of A, not B
            ]
        elif form == "inverse":
            u = (rng.choice(_NONZERO), rng.choice(_NONZERO))
            final = (b[0] - t[0] + u[0], b[1] - t[1] + u[1])
            check_equivalent(list(Matrix(b) - tm + Matrix(u)), list(final))
            text = (
                f"The translation T maps {_pt('A', a)} to {_pt("A'", a1)}. "
                f"Point {_pt('B', b)} is first translated by the inverse of T and then by "
                f"the column vector ({u[0]}, {u[1]}). Find the coordinates of the final image of B."
            )
            cands = [
                (b[0] + t[0] + u[0], b[1] + t[1] + u[1]),  # uses T instead of its inverse
                (b[0] - t[0] - u[0], b[1] - t[1] - u[1]),  # inverts the second vector as well
                (b[0] + u[0], b[1] + u[1]),  # ignores T
                (b[0] - t[0], b[1] - t[1]),  # ignores the column vector
            ]
        else:
            s = (rng.choice(_NONZERO), rng.choice(_NONZERO))
            a2 = (a1[0] + s[0], a1[1] + s[1])
            final = (b[0] + t[0] + s[0], b[1] + t[1] + s[1])
            check_equivalent(list(Matrix(b) + (Matrix(a2) - Matrix(a))), list(final))
            text = (
                f"The translation T maps {_pt('A', a)} to {_pt("A'", a1)}. "
                f"The translation S maps A' to {_pt("A''", a2)}. "
                f"Point {_pt('B', b)} is translated by T and then by S. "
                f"Find the coordinates of the final image of B."
            )
            cands = [
                (b[0] + t[0], b[1] + t[1]),  # only T
                (b[0] + s[0], b[1] + s[1]),  # only S
                (b[0] + t[0] - s[0], b[1] + t[1] - s[1]),  # subtracts S
                (a2[0], a2[1]),  # reports A'' itself
            ]
        distractors = pick_distractors(_answer_text(final), [_answer_text(c) for c in cands])
        if len(distractors) >= 3 and max(abs(final[0]), abs(final[1])) <= 40:
            break
    else:  # pragma: no cover
        raise ValueError("could not draw a question")
    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer={"x": float(final[0]), "y": float(final[1])},
        answer_text=_answer_text(final),
        answer_latex=_answer_text(final),
        working=[f"B' = ({final[0]}, {final[1]})"],
        distractors=distractors[:3],
    )


@generator(
    "exd_x_g3_translations",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_g3_translations(rng: random.Random, tier: str = "exam") -> dict:
    """Recover a translation from a point and its image, then use it in a composite."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
