"""Shared helpers for the early-years data skills that draw a ``tally`` figure or a one-to-one picture graph.

The four skills (sort and count objects, one-to-one displays, tally tables, comparing a tally with a picture graph)
ask for a number, or a row's name, that can only be read from the picture. This kit holds what they share: the words a
category can be, a scatter layout for up to 20 objects, readers for the three displays, and the question dicts.
"""

from __future__ import annotations

from collections import Counter

from ..answers import Distractor
from ..core import check
from ..figure_kinds.tally import ICONS
from ._ey2a_kit import number_question, tagged_numbers

__all__ = [
    "PLURAL",
    "ICONS",
    "scatter_objects",
    "item_counts",
    "tally_counts",
    "display_rows",
    "strokes_only",
    "gates_as_ones",
    "number_question",
    "tagged_numbers",
    "label_question",
    "tagged_labels",
    "row_option",
]

# What a category is called in a stem and on a tally row. Fish stays fish.
PLURAL = {
    "apple": "apples",
    "ball": "balls",
    "car": "cars",
    "cat": "cats",
    "dog": "dogs",
    "fish": "fish",
    "flower": "flowers",
    "star": "stars",
}
check(set(PLURAL) == set(ICONS), "every icon has a plural")

# Cells of a 6 by 4 grid; each object takes one cell and is nudged a little, so the scatter has no fixed pattern.
# The spacing (0.2 across, 0.333 down) is wider than the figure's overlap limit (0.17, 0.23).
_COLS, _ROWS = 6, 4
_JITTER = 0.012


def scatter_objects(rng, counts: dict[str, int]) -> list[tuple[str, float, float]]:
    """``counts`` maps an icon name to how many of it to draw (at most 20 in all); returns ``(name, x, y)`` triples
    at random cells of a 6 by 4 grid, in a random order."""
    names = [name for name, n in counts.items() for _ in range(n)]
    check(1 <= len(names) <= _COLS * _ROWS - 4, "a scatter holds 1 to 20 objects")
    rng.shuffle(names)
    cells = rng.sample(range(_COLS * _ROWS), len(names))
    out = []
    for name, cell in zip(names, cells):
        r, c = divmod(cell, _COLS)
        x = c / (_COLS - 1) + rng.uniform(-_JITTER, _JITTER)
        y = r / (_ROWS - 1) + rng.uniform(-_JITTER, _JITTER)
        out.append((name, min(1.0, max(0.0, x)), min(1.0, max(0.0, y))))
    return out


def item_counts(fig: dict) -> dict[str, int]:
    """How many of each kind of object the picture draws."""
    return dict(Counter(it["label"] for it in fig["items"]))


def tally_counts(fig: dict) -> dict[str, int]:
    """Each row of a ``tally`` figure: its label and the count its marks show."""
    return {r["label"]: r["count"] for r in fig["rows"]}


def display_rows(fig: dict) -> dict[str, int]:
    """Row label to count, read from either a ``tally`` table or a one-to-one ``picture_graph`` (each = 1)."""
    if fig["kind"] == "tally":
        return tally_counts(fig)
    check(fig["kind"] == "picture_graph" and fig["key"]["each"] == 1, "a one-to-one display")
    return {r["label"]: int(r["symbols"]) for r in fig["rows"]}


def strokes_only(count: int) -> int:
    """What a learner reads who counts the upright strokes of a tally and skips the slash across each gate."""
    return 4 * (count // 5) + count % 5


def gates_as_ones(count: int) -> int:
    """What a learner reads who counts each gate of five as a single mark."""
    return count // 5 + count % 5


def label_question(*, text: str, answer: str, distractors: list[Distractor], working: list[str], figure: dict) -> dict:
    """The question dict for a multiple-choice answer that is a word (a row's name) or a number written as text."""
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer,
        answer_latex=f"plain:{answer}",
        working=working,
        distractors=distractors,
        figure=figure,
    )


def row_option(label: str) -> str:
    """How a multiple-choice option names a row. It is never the bare label, which the figure itself prints."""
    return f"The {label} row"


def tagged_labels(answer: str, candidates: list[tuple[str, str | None]], *, n: int = 3) -> list[Distractor]:
    """Wrong options that are words, each with the error it stands for; repeats and the answer are dropped."""
    seen = {answer}
    out: list[Distractor] = []
    for text, tag in candidates:
        if text in seen:
            continue
        seen.add(text)
        out.append(Distractor(text, tag, text))
    check(len(out) >= 2, "too few label distractors")
    return out[:n]
