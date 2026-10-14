"""Shared helpers for the early-years picture skills (counting, ten frames, part-part-whole, comparing, sharing).

Each skill draws a picture with an existing early-years figure kind (``counters`` or ``fraction_model``) and asks
for a single number that can only be read from the picture. This kit holds the pieces they share: a scatter
layout that fits up to 20 counters, tagged distractors for a whole-number answer, and the question dict.
"""

from __future__ import annotations

from ..answers import Distractor
from ..core import check

__all__ = ["SHAPE_WORD", "scatter20", "ten_frame_boxes", "number_question", "tagged_numbers"]

# The counter shapes a picture can be drawn with, and the plural a stem uses for each.
SHAPE_WORD = {"circle": "counters", "square": "squares", "star": "stars", "diamond": "diamonds"}

# Cells of a 5 by 4 grid; a scatter of up to 20 counters picks some of them and nudges each a little.
_COLS = 5
_ROWS = 4
_X0, _DX = 0.04, 0.23
_Y0, _DY = 0.04, 0.307
_JITTER = 0.02  # less than (0.23 - 0.18) / 2, so no two counters ever come closer than the figure allows


def scatter20(rng, n: int, colours: list[int] | None = None) -> list[dict]:
    """``n`` (1-20) counters at jittered cells of a 5 by 4 grid, chosen at random so the picture has no fixed pattern.

    ``colours`` gives each counter its colour (``1`` plain, ``2`` ringed); by default all are plain.
    """
    check(1 <= n <= _COLS * _ROWS, "scatter20 holds 1 to 20 counters")
    cells = rng.sample(range(_COLS * _ROWS), n)
    cols = list(colours) if colours is not None else [1] * n
    check(len(cols) == n, "one colour per counter")
    dots = []
    for cell, colour in zip(cells, cols):
        r, c = divmod(cell, _COLS)
        x = _X0 + _DX * c + rng.uniform(-_JITTER, _JITTER)
        y = _Y0 + _DY * r + rng.uniform(-_JITTER, _JITTER)
        dots.append({"at": [round(x, 3), round(y, 3)], "colour": colour})
    return dots


def ten_frame_boxes(count: int, *, rng=None, colours: list[int] | None = None, by_columns: bool = False) -> list[int]:
    """One ten frame as 10 boxes. With no ``rng`` the counters fill from the top left, row by row (or two at a time
    down the columns when ``by_columns``); with an ``rng`` they sit in random boxes. ``colours`` (length ``count``)
    sets each counter's colour, default plain."""
    check(0 <= count <= 10, "a ten frame holds 0 to 10 counters")
    cols = list(colours) if colours is not None else [1] * count
    check(len(cols) == count, "one colour per counter")
    boxes = [0] * 10
    order = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9] if by_columns else list(range(10))
    spots = order[:count] if rng is None else rng.sample(range(10), count)
    for spot, colour in zip(spots, cols):
        boxes[spot] = colour
    return boxes


def tagged_numbers(
    answer: int, candidates: list[tuple[int, str | None]], *, low: int = 0, high: int = 40
) -> list[Distractor]:
    """Wrong whole-number options, each with the error it stands for.

    ``candidates`` is ``[(value, misconception id), ...]`` in priority order. Values equal to the answer, repeats,
    and values outside ``low..high`` are dropped. If fewer than three survive, plain near misses (``answer + 1``,
    ``answer + 2``, ... with no error attached) top them up, so a small answer still has options.
    """
    seen = {answer}
    out: list[Distractor] = []
    fillers = [(answer + k, None) for k in range(1, 7)] + [(answer - k, None) for k in range(1, 7)]
    for value, tag in [*candidates, *fillers]:
        if value in seen or not low <= value <= high:
            continue
        if tag is None and len(out) >= 3:
            break
        seen.add(value)
        out.append(Distractor(str(value), tag, value))
    check(len(out) >= 3, f"only {len(out)} distractors survive for the answer {answer}")
    return out[:5]


def number_question(*, text: str, answer: int, distractors: list[Distractor], working: list[str], figure: dict) -> dict:
    """The question dict every skill here returns: a plain-text stem, a whole-number answer and a picture."""
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=working,
        distractors=distractors,
        figure=figure,
    )
