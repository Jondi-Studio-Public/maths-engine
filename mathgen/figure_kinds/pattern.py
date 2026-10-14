"""Pattern strips for early-years pattern questions.

Kind: ``pattern_strip``, one to three rows of cells. Each cell holds one token:

- a **shape** (``circle``, ``square``, ``triangle``, ``star``, ``heart``, ``diamond``) with a **fill** (``solid``,
  ``open`` or ``striped``), told apart by more than colour: solid is inked in, open is blank inside, striped is hatched;
- a **text** token, a short numeral (0 to 999) or a single letter;
- an **action** picture (``clap``, ``stamp``, ``jump``): a small icon with its word written under it;
- a **stack** of 1 to 10 counters, for a growing pattern drawn as columns;
- a **blank** (a dashed empty box) or a **mark** (a box holding a "?"), the cell the learner works out.

A row has at most 16 cells, a spec at most 3 rows, and a row at most 2 unknown cells. A row may carry a ``label``
("A", "B"), and a ``bracket`` ``[first, last]`` that marks the cells of the repeating unit. With ``numbered`` set,
each cell's place (1, 2, 3, ...) is written under it.

Every spec carries everything that is drawn. The pure functions here (:func:`row_keys`, :func:`smallest_period`,
:func:`structure`, :func:`describe_cell`) re-derive what a learner sees from the spec alone, which a generator's
plausibility reader calls.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from mathgen.figures import check, register_kind

SHAPES = ("circle", "square", "triangle", "star", "heart", "diamond")
FILLS = ("solid", "open", "striped")
ACTIONS = ("clap", "stamp", "jump")
CELL_TYPES = ("shape", "text", "action", "stack", "blank", "mark")

MAX_CELLS = 16
MAX_ROWS = 3
MAX_UNKNOWN_PER_ROW = 2
MAX_STACK = 10
MAX_LABEL = 8

_TEXT = re.compile(r"^([0-9]{1,3}|[A-Za-z])$")


# --------------------------------------------------------------- cells


def shape_cell(shape: str, fill: str = "solid") -> dict:
    return {"type": "shape", "shape": shape, "fill": fill}


def text_cell(text: str | int) -> dict:
    return {"type": "text", "text": str(text)}


def action_cell(action: str) -> dict:
    return {"type": "action", "action": action}


def stack_cell(count: int) -> dict:
    return {"type": "stack", "count": count}


def blank_cell() -> dict:
    return {"type": "blank"}


def mark_cell() -> dict:
    return {"type": "mark"}


def is_unknown(cell: dict) -> bool:
    return cell.get("type") in ("blank", "mark")


def cell_key(cell: dict) -> str | None:
    """A string that is equal for two cells exactly when they look the same; ``None`` for a blank or a mark."""
    t = cell.get("type")
    if t == "shape":
        return f"shape:{cell['shape']}:{cell['fill']}"
    if t == "text":
        return f"text:{cell['text']}"
    if t == "action":
        return f"action:{cell['action']}"
    if t == "stack":
        return f"stack:{cell['count']}"
    return None


def cell_from_key(key: str) -> dict:
    kind, _, rest = key.partition(":")
    if kind == "shape":
        shape, _, fill = rest.partition(":")
        return shape_cell(shape, fill)
    if kind == "text":
        return text_cell(rest)
    if kind == "action":
        return action_cell(rest)
    if kind == "stack":
        return stack_cell(int(rest))
    raise ValueError(f"not a cell key: {key!r}")


def _a(word: str) -> str:
    return ("an " if word[0] in "aeiou" else "a ") + word


def describe_key(key: str) -> str:
    """What a child hears for a cell: "an open circle", "the number 5", "the letter M", "a clap"."""
    kind, _, rest = key.partition(":")
    if kind == "shape":
        shape, _, fill = rest.partition(":")
        return _a(f"{fill} {shape}")
    if kind == "text":
        return f"the number {rest}" if rest.isdigit() else f"the letter {rest}"
    if kind == "action":
        return _a(rest)
    if kind == "stack":
        n = int(rest)
        return f"{n} counter" if n == 1 else f"{n} counters"
    raise ValueError(f"not a cell key: {key!r}")


def describe_cell(cell: dict) -> str:
    key = cell_key(cell)
    if key is None:
        raise ValueError("a blank or a mark has nothing to describe")
    return describe_key(key)


# --------------------------------------------------------------- the spec


def pattern_strip(*, rows: Iterable[dict], alt: str, title: str = "", numbered: bool = False) -> dict:
    """One to three rows of pattern cells.

    - ``rows``: each ``{"cells": [cell, ...], "label"?: str, "bracket"?: [first, last]}``; build cells with
      :func:`shape_cell`, :func:`text_cell`, :func:`action_cell`, :func:`stack_cell`, :func:`blank_cell`,
      :func:`mark_cell`. With more than one row every row needs a distinct label.
    - ``numbered``: write each cell's place under it (1, 2, 3, ...).
    """
    spec: dict[str, Any] = {
        "kind": "pattern_strip",
        "alt": alt,
        "rows": [{**r, "cells": [dict(c) for c in r["cells"]]} for r in rows],
        "numbered": bool(numbered),
    }
    if title:
        spec["title"] = title
    return check(spec)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _validate_cell(cell: Any) -> list[str]:
    if not isinstance(cell, dict):
        return ["a cell must be a dict"]
    t = cell.get("type")
    if t not in CELL_TYPES:
        return [f"cell type must be one of {CELL_TYPES}"]
    if t == "shape":
        out = []
        if cell.get("shape") not in SHAPES:
            out.append(f"shape must be one of {SHAPES}")
        if cell.get("fill") not in FILLS:
            out.append(f"fill must be one of {FILLS}")
        return out
    if t == "text":
        text = cell.get("text")
        return [] if isinstance(text, str) and _TEXT.match(text) else ["text must be a numeral to 999 or one letter"]
    if t == "action":
        return [] if cell.get("action") in ACTIONS else [f"action must be one of {ACTIONS}"]
    if t == "stack":
        n = cell.get("count")
        return [] if _is_int(n) and 1 <= n <= MAX_STACK else [f"a stack holds 1 to {MAX_STACK} counters"]
    return []


def _validate_pattern_strip(spec: dict) -> list[str]:
    rows = spec.get("rows")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ROWS:
        return [f"rows must be a list of 1 to {MAX_ROWS} rows"]
    if not isinstance(spec.get("numbered"), bool):
        return ["numbered must be a bool"]
    problems = []
    labels = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("cells"), list):
            problems.append(f"rows[{i}] needs a list of cells")
            continue
        cells = row["cells"]
        if not 1 <= len(cells) <= MAX_CELLS:
            problems.append(f"rows[{i}] needs 1 to {MAX_CELLS} cells")
            continue
        bad = [p for c in cells for p in _validate_cell(c)]
        problems.extend(f"rows[{i}]: {p}" for p in bad)
        if bad:
            continue
        unknown = sum(is_unknown(c) for c in cells)
        if unknown > MAX_UNKNOWN_PER_ROW:
            problems.append(f"rows[{i}] has more than {MAX_UNKNOWN_PER_ROW} blanks or marks")
        if unknown == len(cells):
            problems.append(f"rows[{i}] shows nothing but blanks or marks")
        label = row.get("label")
        if label is not None and not (isinstance(label, str) and 0 < len(label) <= MAX_LABEL):
            problems.append(f"rows[{i}].label must be a short non-empty string")
        labels.append(label)
        bracket = row.get("bracket")
        if bracket is not None:
            ok = (
                isinstance(bracket, (list, tuple))
                and len(bracket) == 2
                and all(_is_int(v) for v in bracket)
                and 0 <= bracket[0] <= bracket[1] < len(cells)
            )
            if not ok:
                problems.append(f"rows[{i}].bracket must be [first, last] cell indexes in the row")
    if len(rows) > 1 and not problems:
        if any(lbl is None for lbl in labels) or len(set(labels)) != len(labels):
            problems.append("a figure with several rows needs a distinct label on each")
    return problems


def _pattern_strip_texts(spec: dict) -> list[str]:
    out = []
    for row in spec.get("rows", ()):
        if row.get("label"):
            out.append(row["label"])
        for c in row.get("cells", ()):
            if c.get("type") == "text":
                out.append(c["text"])
            elif c.get("type") == "action":
                out.append(c["action"])
            elif c.get("type") == "mark":
                out.append("?")
        if spec.get("numbered"):
            out.extend(str(i + 1) for i in range(len(row.get("cells", ()))))
    return out


register_kind("pattern_strip", _validate_pattern_strip, texts=_pattern_strip_texts)


# --------------------------------------------------------------- reading a strip


def row_keys(spec: dict, row: int = 0) -> list[str | None]:
    """The key of every cell in a row, ``None`` where the cell is a blank or a mark."""
    return [cell_key(c) for c in spec["rows"][row]["cells"]]


def unknown_indexes(keys: list[str | None]) -> list[int]:
    return [i for i, k in enumerate(keys) if k is None]


def fits_period(keys: list[str | None], p: int) -> bool:
    """True when every known cell equals the known cells a whole number of ``p`` away."""
    seen: dict[int, str] = {}
    for i, k in enumerate(keys):
        if k is None:
            continue
        if seen.setdefault(i % p, k) != k:
            return False
    return True


def smallest_period(keys: list[str | None]) -> int:
    """The shortest repeating unit the known cells allow (the whole row if none shorter fits)."""
    for p in range(1, len(keys) + 1):
        if fits_period(keys, p):
            return p
    return len(keys)


def unit_of(keys: list[str | None], p: int) -> list[str | None]:
    """The ``p`` cells of the repeating unit, ``None`` where no known cell shows that place."""
    out: list[str | None] = [None] * p
    for i, k in enumerate(keys):
        if k is not None:
            out[i % p] = k
    return out


def structure(keys: list[str | None]) -> str:
    """The shape of the pattern, ignoring what it is made of: "AB", "AAB", "ABC". Needs a full unit with no gaps."""
    p = smallest_period(keys)
    unit = unit_of(keys, p)
    if None in unit:
        raise ValueError("the strip does not show a whole unit")
    names: dict[str, str] = {}
    return "".join(names.setdefault(k, "ABCDEFGH"[len(names)]) for k in unit)  # type: ignore[arg-type]


def read_gap(spec: dict, row: int = 0) -> str:
    """The key of the cell the one blank or mark must hold, re-derived from the other cells alone."""
    keys = row_keys(spec, row)
    gaps = unknown_indexes(keys)
    if len(gaps) != 1:
        raise ValueError("the row must have exactly one blank or mark")
    p = smallest_period(keys)
    key = unit_of(keys, p)[gaps[0] % p]
    if key is None:
        raise ValueError("no cell shows what belongs in the gap")
    return key


def read_nth(spec: dict, n: int, row: int = 0) -> str:
    """The key of the ``n``th cell (counting from 1) of the pattern, even past the end of the strip."""
    keys = row_keys(spec, row)
    p = smallest_period(keys)
    key = unit_of(keys, p)[(n - 1) % p]
    if key is None:
        raise ValueError("no cell shows that place in the pattern")
    return key


def read_mistake(spec: dict, row: int = 0) -> int:
    """The place (counting from 1) of the one cell that breaks an otherwise repeating pattern.

    A unit length is tried only when every place in it is shown by three or more cells, so the odd one out is
    found by majority. The answer must be the same for every unit length that explains the strip by one
    wrong cell, otherwise the strip is ambiguous and this raises.
    """
    keys = row_keys(spec, row)
    if None in keys:
        raise ValueError("a strip with a mistake has no gaps")
    found: set[int] = set()
    for p in range(1, len(keys) // 3 + 1):
        wrong = []
        for r in range(p):
            members = [(i, keys[i]) for i in range(r, len(keys), p)]
            counts: dict[str | None, int] = {}
            for _, k in members:
                counts[k] = counts.get(k, 0) + 1
            top = max(counts.values())
            if top == len(members):
                continue
            if top != len(members) - 1 or list(counts.values()).count(top) != 1:
                wrong = None
                break
            major = next(k for k, c in counts.items() if c == top)
            wrong += [i for i, k in members if k != major]
        if wrong is not None and len(wrong) == 1:
            found.add(wrong[0] + 1)
    if len(found) != 1:
        raise ValueError("the strip does not show exactly one mistake")
    return found.pop()
