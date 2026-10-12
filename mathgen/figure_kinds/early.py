"""Early-years figure kinds: money, the hundred chart, fraction models, counters and base-ten blocks.

Every kind here is a picture the front end draws and never evaluates. The spec carries every
number, count and position that is drawn; the renderer decides only layout (sizes, colours). Two
colours (``1`` and ``2``) are always distinguishable by more than hue (for example a filled and a
ringed counter), because learners may be colour blind.

- :func:`money`: a collection of Australian coins and notes, drawn as generic discs and
  rectangles with their value printed (no real currency artwork).
- :func:`hundred_chart`: a 10 by 10 chart of 1-100 (or 0-99) with blanked, shaded and ringed cells.
- :func:`fraction_model`: one or two wholes (bar, rectangle grid or circle) cut into equal parts.
- :func:`counters`: ten frames, arrays, groups, dice faces and scattered dots.
- :func:`base_ten`: hundreds flats, tens rods and ones cubes.
"""

from __future__ import annotations

import math
from typing import Iterable

from mathgen.figures import _clean, _is_num, check, register_kind

__all__ = [
    "base_ten",
    "base_ten_value",
    "chart_number_at",
    "counters",
    "counters_total",
    "fraction_model",
    "fraction_shape",
    "hundred_chart",
    "money",
    "money_cents",
    "scatter_dots",
    "DENOMINATIONS",
]


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


# ---------------------------------------------------------------- money

# denomination text -> (type, value in cents)
DENOMINATIONS: dict[str, tuple[str, int]] = {
    "5c": ("coin", 5),
    "10c": ("coin", 10),
    "20c": ("coin", 20),
    "50c": ("coin", 50),
    "$1": ("coin", 100),
    "$2": ("coin", 200),
    "$5": ("note", 500),
    "$10": ("note", 1000),
    "$20": ("note", 2000),
    "$50": ("note", 5000),
    "$100": ("note", 10000),
}

MAX_MONEY_ITEMS = 12
MAX_MONEY_GROUPS = 3


def money(*, collections: Iterable[Iterable[str]], alt: str, labels: Iterable[str] | None = None) -> dict:
    """A collection (or up to three labelled collections) of Australian coins and notes.

    ``collections`` is a list of collections, each a list of denomination strings from
    :data:`DENOMINATIONS` (``"5c" "10c" "20c" "50c" "$1" "$2" "$5" "$10" "$20" "$50" "$100"``).
    ``labels`` names the collections (``"A"``, ``"B"``) when there is more than one; a single
    collection takes no label.

    Spec::

        {"kind": "money", "alt": str,
         "groups": [{"label": str,            # "" for a lone collection, else 1-2 characters
                     "items": [{"denom": str,  # the value printed on it, e.g. "20c", "$5"
                                "type": "coin" | "note",
                                "cents": int}]}]}  # its value in cents (not drawn)

    Items are drawn left to right, wrapping, in the order given. A coin is a round disc and a
    note a wider rectangle, both generic shapes with ``denom`` printed large in the middle (the
    renderer never reproduces real currency artwork). Coins are drawn the same size whatever
    their value, so size never hints at value. A group's ``label`` is printed above it. At most
    12 items per group and 3 groups.
    """
    cols = [list(c) for c in collections]
    if labels is None:
        labels = [""] * len(cols) if len(cols) == 1 else [chr(ord("A") + i) for i in range(len(cols))]
    labels = list(labels)
    groups = []
    for label, col in zip(labels, cols):
        items = []
        for d in col:
            typ, cents = DENOMINATIONS.get(d, ("", 0))
            items.append({"denom": d, "type": typ, "cents": cents})
        groups.append({"label": label, "items": items})
    return check({"kind": "money", "alt": alt, "groups": groups})


def money_cents(spec: dict, group: int = 0) -> int:
    """Total value in cents of one group, summed from the drawn ``cents`` fields."""
    return sum(i["cents"] for i in spec["groups"][group]["items"])


def _validate_money(spec: dict) -> list[str]:
    groups = spec.get("groups")
    if not isinstance(groups, list) or not 1 <= len(groups) <= MAX_MONEY_GROUPS:
        return [f"groups must hold 1 to {MAX_MONEY_GROUPS} collections"]
    problems = []
    seen = set()
    for gi, g in enumerate(groups):
        label = g.get("label") if isinstance(g, dict) else None
        if not isinstance(label, str) or len(label) > 2:
            problems.append(f"groups[{gi}].label must be a string of at most 2 characters")
        elif len(groups) > 1 and not label:
            problems.append(f"groups[{gi}] needs a label when there are several collections")
        elif label in seen:
            problems.append(f"groups[{gi}].label {label!r} repeats")
        seen.add(label)
        items = g.get("items") if isinstance(g, dict) else None
        if not isinstance(items, list) or not 1 <= len(items) <= MAX_MONEY_ITEMS:
            problems.append(f"groups[{gi}].items must hold 1 to {MAX_MONEY_ITEMS} items")
            continue
        for ii, it in enumerate(items):
            known = DENOMINATIONS.get(it.get("denom")) if isinstance(it, dict) else None
            if known is None:
                problems.append(f"groups[{gi}].items[{ii}] has an unknown denomination")
            elif (it.get("type"), it.get("cents")) != known:
                problems.append(f"groups[{gi}].items[{ii}] type/cents do not match {it['denom']!r}")
    return problems


def _money_texts(spec: dict) -> list[str]:
    out = []
    for g in spec["groups"]:
        out.append(g["label"])
        out.extend(i["denom"] for i in g["items"])
    return out


register_kind("money", _validate_money, texts=_money_texts)


# ---------------------------------------------------------------- hundred chart


def hundred_chart(
    *,
    alt: str,
    start: int = 1,
    hidden: Iterable[int] = (),
    shaded: Iterable[int] = (),
    marked: Iterable[int] = (),
) -> dict:
    """A 10 by 10 number chart.

    Cells run left to right, then top to bottom: with ``start`` 1 the first row is 1-10 and the
    last 91-100; with ``start`` 0 it is 0-9 down to 90-99. Each cell is a numeral in a square.

    Spec::

        {"kind": "hundred_chart", "alt": str,
         "start": 0 | 1,        # the number in the top-left cell
         "hidden": [int],       # cells drawn EMPTY (the numeral is not printed)
         "shaded": [int],       # cells filled with the highlight colour (hidden cells may be shaded)
         "marked": [int]}       # cells with a heavy ring around them (hidden cells may be ringed)

    A hidden cell is still in its place in the grid, so a learner finds its number by position.
    Shaded and ringed cells are marked by more than colour (a fill versus a ring). At most 30
    cells are hidden and no row or column is entirely hidden.
    """
    spec = {
        "kind": "hundred_chart",
        "alt": alt,
        "start": start,
        "hidden": sorted(hidden),
        "shaded": sorted(shaded),
        "marked": sorted(marked),
    }
    return check(spec)


def chart_number_at(spec: dict, row: int, col: int) -> int:
    """The number in cell (row, col), both counted from 0 at the top left."""
    return spec["start"] + row * 10 + col


def _validate_hundred_chart(spec: dict) -> list[str]:
    start = spec.get("start")
    if start not in (0, 1) or isinstance(start, bool):
        return ["start must be 0 or 1"]
    lo, hi = start, start + 99
    problems = []
    sets = {}
    for key in ("hidden", "shaded", "marked"):
        vals = spec.get(key)
        if not isinstance(vals, list) or not all(_is_int(v) for v in vals):
            problems.append(f"{key} must be a list of integers")
            continue
        if len(set(vals)) != len(vals):
            problems.append(f"{key} repeats a number")
        if any(not lo <= v <= hi for v in vals):
            problems.append(f"{key} has a number outside {lo}-{hi}")
        sets[key] = set(vals)
    if problems:
        return problems
    hidden = sets["hidden"]
    if len(hidden) > 30:
        problems.append("more than 30 hidden cells")
    for r in range(10):
        if all(lo + r * 10 + c in hidden for c in range(10)):
            problems.append(f"row {r} is entirely hidden")
    for c in range(10):
        if all(lo + r * 10 + c in hidden for r in range(10)):
            problems.append(f"column {c} is entirely hidden")
    return problems


def _chart_texts(spec: dict) -> list[str]:
    hidden = set(spec["hidden"])
    return [str(spec["start"] + i) for i in range(100) if spec["start"] + i not in hidden]


register_kind("hundred_chart", _validate_hundred_chart, texts=_chart_texts)


# ---------------------------------------------------------------- fraction models

FRACTION_SHAPES = ("bar", "rectangle", "circle")
_CIRCLE_PARTS = (2, 3, 4, 5, 6, 8, 10, 12)


def fraction_shape(
    shape: str, parts: int, shaded: Iterable[int], *, rows: int | None = None, cols: int | None = None, label: str = ""
) -> dict:
    """One whole cut into equal parts, for :func:`fraction_model`.

    - ``shape`` ``"bar"``: a horizontal bar cut by vertical lines into ``parts`` equal parts,
      numbered 0.. from the left.
    - ``"rectangle"``: a rectangle grid of ``rows`` by ``cols`` equal cells (``parts == rows *
      cols``), numbered row by row from the top left.
    - ``"circle"``: a circle cut into ``parts`` equal sectors by radii, sector 0 starting at
      12 o'clock and numbering clockwise.
    - ``shaded``: the part numbers drawn filled; the rest are drawn empty. Shaded parts are
      marked by more than colour (a hatch fill).
    - ``label``: a name printed beside the whole (``"A"``), empty for none.
    """
    d = {"shape": shape, "parts": parts, "shaded": sorted(shaded), "label": label}
    if shape == "rectangle":
        d["rows"], d["cols"] = rows, cols
    return d


def fraction_model(*, models: Iterable[dict], alt: str) -> dict:
    """One or two wholes, each divided into equal parts with some shaded.

    Spec::

        {"kind": "fraction_model", "alt": str, "models": [fraction_shape(...), ...]}

    The renderer draws every whole the same overall size (a bar the same length, a circle the
    same diameter, a rectangle the same outline), so two models can be compared fairly, and
    always cuts it into exactly equal parts. When there are two models they must be the same
    ``shape`` and carry labels ``"A"`` and ``"B"``. No fraction numeral is ever drawn.
    """
    return check({"kind": "fraction_model", "alt": alt, "models": [dict(m) for m in models]})


def _validate_fraction_model(spec: dict) -> list[str]:
    models = spec.get("models")
    if not isinstance(models, list) or not 1 <= len(models) <= 2:
        return ["models must hold one or two models"]
    problems = []
    for i, m in enumerate(models):
        shape, parts, shaded = m.get("shape"), m.get("parts"), m.get("shaded")
        if shape not in FRACTION_SHAPES:
            problems.append(f"models[{i}].shape must be one of {FRACTION_SHAPES}")
            continue
        if not _is_int(parts) or not 2 <= parts <= 12:
            problems.append(f"models[{i}].parts must be an integer from 2 to 12")
            continue
        if shape == "circle" and parts not in _CIRCLE_PARTS:
            problems.append(f"models[{i}] circle parts must be one of {_CIRCLE_PARTS} (equal sectors read honestly)")
        if shape == "rectangle":
            rows, cols = m.get("rows"), m.get("cols")
            if not (_is_int(rows) and _is_int(cols) and rows >= 1 and cols >= 1 and rows * cols == parts):
                problems.append(f"models[{i}] rectangle needs rows * cols == parts")
        if not isinstance(shaded, list) or not all(_is_int(s) and 0 <= s < parts for s in shaded):
            problems.append(f"models[{i}].shaded must list part numbers from 0 to parts-1")
        elif len(set(shaded)) != len(shaded):
            problems.append(f"models[{i}].shaded repeats a part")
        elif not 1 <= len(shaded) <= parts - 1:
            problems.append(f"models[{i}] must shade at least one part and leave at least one unshaded")
        if not isinstance(m.get("label"), str) or len(m["label"]) > 2:
            problems.append(f"models[{i}].label must be a string of at most 2 characters")
    if problems:
        return problems
    if len(models) == 2:
        if models[0]["shape"] != models[1]["shape"]:
            problems.append("two models must be the same shape so the wholes are the same size")
        if [m["label"] for m in models] != ["A", "B"]:
            problems.append("two models must be labelled A and B")
    return problems


def _fraction_texts(spec: dict) -> list[str]:
    return [m["label"] for m in spec["models"]]


register_kind("fraction_model", _validate_fraction_model, texts=_fraction_texts)


# ---------------------------------------------------------------- counters

COUNTER_SHAPES = ("circle", "square", "star", "diamond")
COUNTER_LAYOUTS = ("ten_frame", "array", "groups", "dice", "scatter")
MIN_SCATTER_GAP = 0.18


def counters(*, layout: str, alt: str, shape: str = "circle", **kw) -> dict:
    """Counters, dots and dice. ``layout`` picks one of five drawings; ``shape`` is the counter
    shape (``"circle"``, ``"square"``, ``"star"`` or ``"diamond"``) in every layout except
    ``dice`` (whose dots are always round). Colour ``1`` is the main counter and ``2`` the
    second counter (drawn as a ringed, not merely recoloured, counter).

    - ``layout="ten_frame"``, ``frames=[[10 ints], ...]``: one or two ten frames, each 2 rows of
      5 boxes. A frame lists its 10 boxes row by row from the top left, each ``0`` (empty),
      ``1`` or ``2`` (a counter of that colour). Two frames are drawn side by side.
    - ``layout="array"``, ``rows``, ``cols``, optional ``colours``: a full ``rows`` by ``cols``
      grid of counters, 1-10 each way. ``colours`` is a ``rows`` by ``cols`` list of ``1``/``2``
      (default all ``1``) for part-part-whole.
    - ``layout="groups"``, ``groups=[int, ...]``: that many rings (1-9), the i-th holding
      ``groups[i]`` counters (0-9; ``0`` is an empty ring). Rings are drawn left to right,
      wrapping. Counters in a ring are laid out in a tidy block.
    - ``layout="dice"``, ``faces=[{"n": 1-6, "colour": 1|2}, ...]``: 1-3 dice faces, each a
      rounded square with the standard dot pattern for ``n``: 1 centre; 2 opposite corners
      (top left, bottom right); 3 the diagonal; 4 the four corners; 5 the four corners and the
      centre; 6 two columns of three.
    - ``layout="scatter"``, ``dots=[{"at": [x, y], "colour": 1|2}, ...]``: 1-20 counters at
      positions in the unit square (x right, y down, both 0 to 1), no two closer than 0.18.

    The picture draws no text.
    """
    spec = {"kind": "counters", "alt": alt, "layout": layout, "shape": shape}
    for k, v in kw.items():
        spec[k] = v
    return check(spec)


def scatter_dots(rng, n: int, *, colours: Iterable[int] | None = None, arrangement: str = "tidy") -> list[dict]:
    """Positions for a ``scatter`` counters layout, drawn from ``rng`` (deterministic).

    - ``"tidy"``: ``n`` (1-10) of the cells of a 5 by 3 grid, jittered a little, for counting.
    - ``"free"``: ``n`` (1-6) at random positions with no two closer than 0.3, so no recognisable
      pattern forms.
    - ``"clusters"``: ``n`` (2-8) split into a left group and a right group (each 1-4 dots in a
      2 by 2 block) with a clear gap between them. Returns the dots only; the split is not
      recorded, so a reader must count them all.
    """
    cols = list(colours) if colours is not None else [1] * n
    if arrangement == "tidy":
        cells = [(c, r) for r in range(3) for c in range(5)]
        pts = []
        for c, r in rng.sample(cells, n):
            pts.append(
                [
                    _clean(0.08 + 0.21 * c + rng.uniform(-0.012, 0.012)),
                    _clean(0.15 + 0.35 * r + rng.uniform(-0.08, 0.08)),
                ]
            )
    elif arrangement == "free":
        for _ in range(200):
            pts = []
            for _ in range(n):
                for _ in range(60):
                    p = [_clean(rng.uniform(0.08, 0.92)), _clean(rng.uniform(0.12, 0.88))]
                    if all(math.dist(p, q) >= 0.3 for q in pts):
                        pts.append(p)
                        break
            if len(pts) == n:
                break
        else:
            raise ValueError("could not place the dots")
    elif arrangement == "clusters":
        a = rng.randint(max(1, n - 4), min(4, n - 1))
        block = [(0, 0), (1, 0), (0, 1), (1, 1)]
        pts = []
        for side, k in ((0.08, a), (0.70, n - a)):
            for bx, by in rng.sample(block, k):
                pts.append([_clean(side + 0.22 * bx), _clean(0.28 + 0.44 * by)])
    else:
        raise ValueError(f"unknown arrangement {arrangement!r}")
    return [{"at": p, "colour": cols[i]} for i, p in enumerate(pts)]


def counters_total(spec: dict) -> int:
    """How many counters (or dots) the picture shows in all."""
    layout = spec["layout"]
    if layout == "ten_frame":
        return sum(1 for f in spec["frames"] for c in f if c)
    if layout == "array":
        return spec["rows"] * spec["cols"]
    if layout == "groups":
        return sum(spec["groups"])
    if layout == "dice":
        return sum(f["n"] for f in spec["faces"])
    return len(spec["dots"])


def _validate_counters(spec: dict) -> list[str]:
    layout = spec.get("layout")
    if layout not in COUNTER_LAYOUTS:
        return [f"layout must be one of {COUNTER_LAYOUTS}"]
    problems = []
    if spec.get("shape") not in COUNTER_SHAPES:
        problems.append(f"shape must be one of {COUNTER_SHAPES}")
    if layout == "ten_frame":
        frames = spec.get("frames")
        if not isinstance(frames, list) or not 1 <= len(frames) <= 2:
            problems.append("frames must hold one or two ten frames")
        else:
            for i, f in enumerate(frames):
                if not (isinstance(f, list) and len(f) == 10 and all(c in (0, 1, 2) and _is_int(c) for c in f)):
                    problems.append(f"frames[{i}] must be 10 boxes, each 0, 1 or 2")
    elif layout == "array":
        rows, cols = spec.get("rows"), spec.get("cols")
        if not (_is_int(rows) and _is_int(cols) and 1 <= rows <= 10 and 1 <= cols <= 10):
            problems.append("rows and cols must be integers from 1 to 10")
        elif "colours" in spec:
            c = spec["colours"]
            if not (
                isinstance(c, list)
                and len(c) == rows
                and all(isinstance(r, list) and len(r) == cols and all(v in (1, 2) for v in r) for r in c)
            ):
                problems.append("colours must be a rows by cols grid of 1 or 2")
    elif layout == "groups":
        g = spec.get("groups")
        if not (isinstance(g, list) and 1 <= len(g) <= 9 and all(_is_int(n) and 0 <= n <= 9 for n in g)):
            problems.append("groups must list 1-9 rings holding 0-9 counters each")
    elif layout == "dice":
        faces = spec.get("faces")
        if not (isinstance(faces, list) and 1 <= len(faces) <= 3):
            problems.append("faces must hold one to three dice")
        else:
            for i, f in enumerate(faces):
                if not (isinstance(f, dict) and f.get("n") in range(1, 7) and f.get("colour") in (1, 2)):
                    problems.append(f"faces[{i}] needs n from 1 to 6 and colour 1 or 2")
    else:
        dots = spec.get("dots")
        if not (isinstance(dots, list) and 1 <= len(dots) <= 20):
            return problems + ["dots must hold 1 to 20 counters"]
        pts = []
        for i, d in enumerate(dots):
            at = d.get("at") if isinstance(d, dict) else None
            if not (
                isinstance(at, (list, tuple))
                and len(at) == 2
                and all(_is_num(v) and 0 <= v <= 1 for v in at)
                and d.get("colour") in (1, 2)
            ):
                problems.append(f"dots[{i}] needs at [x, y] within the unit square and colour 1 or 2")
            else:
                pts.append(at)
        if not problems:
            for i, p in enumerate(pts):
                for q in pts[i + 1 :]:
                    if math.dist(p, q) < MIN_SCATTER_GAP:
                        problems.append("two dots are too close to tell apart")
                        return problems
    return problems


register_kind("counters", _validate_counters)


# ---------------------------------------------------------------- base ten


def base_ten(*, hundreds: int = 0, tens: int = 0, ones: int = 0, alt: str) -> dict:
    """Base-ten blocks: hundreds flats, tens rods and ones cubes.

    Spec::

        {"kind": "base_ten", "alt": str, "hundreds": int, "tens": int, "ones": int}

    The renderer draws, left to right, the flats (each a 10 by 10 square of small squares), then
    the rods (each a column of 10 small squares), then the loose cubes (single small squares),
    each type in its own labelled-by-position area with no text. The counts are exactly what is
    drawn, so ``ones`` may be 10 or more (an unregrouped collection) and ``tens`` may exceed 9.
    Limits: hundreds 0-9, tens 0-20, ones 0-20, and at least one block.
    """
    return check({"kind": "base_ten", "alt": alt, "hundreds": hundreds, "tens": tens, "ones": ones})


def base_ten_value(spec: dict) -> int:
    """The number the blocks show: 100 * flats + 10 * rods + cubes."""
    return 100 * spec["hundreds"] + 10 * spec["tens"] + spec["ones"]


def _validate_base_ten(spec: dict) -> list[str]:
    limits = {"hundreds": 9, "tens": 20, "ones": 20}
    problems = []
    for key, top in limits.items():
        v = spec.get(key)
        if not _is_int(v) or not 0 <= v <= top:
            problems.append(f"{key} must be an integer from 0 to {top}")
    if not problems and base_ten_value(spec) == 0:
        problems.append("show at least one block")
    return problems


register_kind("base_ten", _validate_base_ten)
