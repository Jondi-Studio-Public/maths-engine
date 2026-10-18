"""The ``tally`` figure kind: a tally table, a scatter of objects to sort, or both together.

A tally table has up to 6 rows. Each row is a short category word and a count from 0 to 30, drawn as tally
marks in gates of five (four upright strokes and one diagonal across them). A count that is not a multiple of
five ends in a part gate of one to four strokes. A row with a count of 0 shows no marks.

``objects`` is the "collect and sort" picture: up to 20 small pictures scattered over the page, the category
word printed under each. Only the eight pictures in :data:`ICONS` can be drawn, and the word under a picture is
its name. The picture shows no total and no count.

Both can be in one figure (the objects above, the table below), and the table can carry a ``pictures`` column,
a pictograph of the same rows, so a learner can compare two displays of the same data.
"""

from __future__ import annotations

from typing import Iterable

from mathgen.figures import _is_num, check, register_kind
from mathgen.figure_kinds.data import SYMBOLS

ICONS = ("apple", "ball", "car", "cat", "dog", "fish", "flower", "star")
MAX_ROWS = 6
MAX_COUNT = 30
MAX_ITEMS = 20
MAX_ROW_LABEL = 12
# With a pictures column beside the tally the page is narrower, so the counts it can show are smaller.
MAX_COUNT_WITH_PICTURES = 20
MAX_PICTURES = 10
# Two objects never overlap when they differ by this much (of the scatter area) across or down.
MIN_ITEM_DX = 0.17
MIN_ITEM_DY = 0.23


def tally(
    *,
    alt: str,
    rows: Iterable[tuple[str, int]] = (),
    items: Iterable[tuple[str, float, float]] = (),
    show_counts: bool = False,
    pictures: dict | None = None,
    title: str = "",
) -> dict:
    """A tally table and/or a scatter of objects.

    - ``rows``: ``(label, count)`` pairs in drawn order, the first row at the TOP. ``label`` is a word of at most
      12 characters, ``count`` a whole number from 0 to 30 (0 to 20 when ``pictures`` is given).
    - ``show_counts``: ``True`` prints each count as a numeral in a Count column beside the marks; the default
      hides it, so the learner works it out from the marks.
    - ``items``: ``(name, x, y)`` triples, the centre of each object in the unit square (x right, y down). ``name``
      is one of :data:`ICONS`. At most 20, and no two overlapping.
    - ``pictures``: ``{"symbol": str, "each": 1 | 2, "counts": [n, ...]}`` adds a pictograph column, one entry per
      row (0 to 10 whole symbols). ``each`` is how many each picture stands for: 1 is a one-to-one display.
    """
    spec: dict = {
        "kind": "tally",
        "alt": alt,
        "rows": [{"label": lb, "count": n} for lb, n in rows],
        "items": [{"label": lb, "at": [round(x, 3), round(y, 3)]} for lb, x, y in items],
        "show_counts": bool(show_counts),
    }
    if pictures is not None:
        spec["pictures"] = {
            "symbol": pictures["symbol"],
            "each": pictures["each"],
            "counts": list(pictures["counts"]),
        }
    if title:
        spec["title"] = title
    return check(spec)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def picture_key_text(spec: dict) -> str:
    """The key drawn under a pictures column."""
    return f"Each picture = {spec['pictures']['each']}"


def _validate_tally(spec: dict) -> list[str]:
    problems: list[str] = []
    rows, items = spec.get("rows"), spec.get("items")
    if not isinstance(rows, list) or not isinstance(items, list):
        return ["rows and items must be lists"]
    if len(rows) > MAX_ROWS:
        problems.append(f"at most {MAX_ROWS} rows")
    if len(items) > MAX_ITEMS:
        problems.append(f"at most {MAX_ITEMS} objects")
    if not rows and not items:
        problems.append("a tally figure needs rows, objects or both")
    if not isinstance(spec.get("show_counts"), bool):
        problems.append("show_counts must be a bool")
    elif spec["show_counts"] and not rows:
        problems.append("show_counts needs rows")
    pictures = spec.get("pictures")
    cap = MAX_COUNT_WITH_PICTURES if pictures is not None else MAX_COUNT
    names = []
    for i, r in enumerate(rows):
        label, count = (r.get("label"), r.get("count")) if isinstance(r, dict) else (None, None)
        if not (isinstance(label, str) and label.strip() and len(label) <= MAX_ROW_LABEL):
            problems.append(f"rows[{i}].label must be a short word")
        names.append(label)
        if not (_is_int(count) and 0 <= count <= cap):
            problems.append(f"rows[{i}].count must be a whole number from 0 to {cap}")
    if len(set(names)) != len(names):
        problems.append("row labels must be distinct")
    placed: list[list[float]] = []
    for i, it in enumerate(items):
        at = it.get("at") if isinstance(it, dict) else None
        if not (it.get("label") in ICONS if isinstance(it, dict) else False):
            problems.append(f"items[{i}].label must be one of {', '.join(ICONS)}")
        if not (isinstance(at, (list, tuple)) and len(at) == 2 and all(_is_num(v) and 0 <= v <= 1 for v in at)):
            problems.append(f"items[{i}].at must be [x, y] within the unit square")
            continue
        for q in placed:
            if abs(at[0] - q[0]) < MIN_ITEM_DX and abs(at[1] - q[1]) < MIN_ITEM_DY:
                problems.append("two objects overlap")
                return problems
        placed.append(list(at))
    if pictures is not None:
        if not isinstance(pictures, dict):
            return problems + ["pictures must be a dict"]
        if not rows:
            problems.append("pictures needs rows")
        if pictures.get("symbol") not in SYMBOLS:
            problems.append(f"pictures.symbol must be one of {', '.join(SYMBOLS)}")
        if pictures.get("each") not in (1, 2) or isinstance(pictures.get("each"), bool):
            problems.append("pictures.each must be 1 or 2")
        counts = pictures.get("counts")
        if not (isinstance(counts, list) and len(counts) == len(rows)):
            problems.append("pictures.counts needs one entry per row")
        elif not all(_is_int(n) and 0 <= n <= MAX_PICTURES for n in counts):
            problems.append(f"pictures.counts must be whole numbers from 0 to {MAX_PICTURES}")
    return problems


def _tally_texts(spec: dict) -> list[str]:
    out = [r["label"] for r in spec.get("rows", [])] + [it["label"] for it in spec.get("items", [])]
    if spec.get("show_counts"):
        out += [str(r["count"]) for r in spec.get("rows", [])]
    if spec.get("pictures"):
        out.append(picture_key_text(spec))
    return out


register_kind("tally", _validate_tally, texts=_tally_texts)
