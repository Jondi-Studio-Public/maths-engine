"""Generates `dg_grid_map` questions (easy, medium, hard) and the exam chain `dg_grid_map_chain`.

``place-features-on-a-grid-map`` (``primary-geometry-time``, answer ``grid_cells``): a map is drawn on a square grid with
column letters, row numbers (counted from the bottom, as on the coordinate plane), a north arrow and landmark symbols
with a legend. The learner shades squares on the map's own grid to place a feature by its grid reference, by a move from
a landmark, or as a region. The answer is the set of shaded squares, matched exactly by the `grid_cells` grader; a square
is typed as ``(column, row)`` counting from 0 at the bottom left.

* easy: shade the square with a stated grid reference such as ``C4``.
* medium: shade the square a stated number of squares east or west and north or south of a named landmark.
* hard: shade a rectangular region from its corner references, or the squares that lie strictly between two landmarks in the
  same row or column.
* exam (``dg_grid_map_chain``): (a) shade the square of a grid reference, (b) how many squares east (or west) one landmark
  is from another, (c) how many squares north (or south), (d) the number of squares in the walk along the row then up the
  column, following through from (b) and (c), (e) shade the square a move away from a third landmark.

Every key is built backwards from a chosen cell; the reader recomputes it from the figure's labels and symbols and the
move or reference in the question.
"""

from __future__ import annotations

import re

from .. import figures
from ..core import attempts, check, generator
from ._spread_kit import uses_rng
from ..figure_kinds import grid
from ..grading.grid import cell_text, cells_answer
from ._dg_kit import numd, p_number
from ._w9_graph_kit import dist

AREA = "place-features-on-a-grid-map"
LETTERS = "ABCDEFGHIJ"

MISCONCEPTIONS = [
    (
        "dg-gm-reference-reversed",
        "Reads the grid reference the wrong way round, taking the number as the column and the letter as the row.",
        AREA,
    ),
    ("dg-gm-row-from-top", "Counts the rows from the top of the map instead of from the bottom.", AREA),
    (
        "dg-gm-counts-start-square",
        "Counts the square the move starts from as the first square of the move, ending one square short.",
        AREA,
    ),
    ("dg-gm-one-square-too-far", "Moves one square too far along the row or column.", AREA),
    ("dg-gm-direction-reversed", "Moves west when east was asked, or south when north was asked.", AREA),
    (
        "dg-gm-moves-swapped",
        "Swaps the two moves, going the east-west number of squares up and the north-south number across.",
        AREA,
    ),
    (
        "dg-gm-region-corners-only",
        "Shades only the corner squares of the region instead of every square inside the rectangle.",
        AREA,
    ),
    (
        "dg-gm-region-edge-left-out",
        "Shades a rectangle one row or column short, leaving out the squares on the far edge.",
        AREA,
    ),
    (
        "dg-gm-between-includes-ends",
        "Shades the squares of the two landmarks as well as the squares between them.",
        AREA,
    ),
    ("dg-gm-between-one-short", "Shades one square fewer than lie between the landmarks.", AREA),
    (
        "dg-gm-count-includes-start",
        "Counts the starting square as well as the squares moved, giving one too many.",
        AREA,
    ),
    (
        "dg-gm-count-reference-difference",
        "Subtracts the letters' places and the numbers the wrong way round, mixing up which is the east-west distance.",
        AREA,
    ),
    (
        "dg-gm-walk-as-diagonal",
        "Gives the larger of the two moves as the length of the walk instead of adding them.",
        AREA,
    ),
    ("dg-gm-walk-product", "Multiplies the east-west and north-south moves instead of adding them.", AREA),
]

_NAMEABLE = (
    "house",
    "school",
    "church",
    "hospital",
    "bridge",
    "campground",
    "lookout",
    "post_office",
    "railway_station",
    "lighthouse",
    "picnic_area",
)
_FEATURES = (
    "kiosk",
    "bus stop",
    "bike rack",
    "drinking fountain",
    "first-aid post",
    "information sign",
    "car park",
    "playground",
)


def _a(word):
    """The word with its indefinite article: "an information sign", "a kiosk"."""
    return ("an " if word[:1] in "aeiou" else "a ") + word


def _ref(c, r):
    return f"{LETTERS[c]}{r + 1}"


def _cell(ref):
    return LETTERS.index(ref[0]), int(ref[1:]) - 1


def _name(symbol):
    return grid.SYMBOLS[symbol]


def _the(symbol):
    return f"the {_name(symbol)}"


def _size(rng):
    return rng.choice((7, 8)), rng.choice((6, 7))


def _figure(rng, cells, w, h):
    symbols = [{"at": [c + 0.5, r + 0.5], "symbol": s} for (c, r), s in cells.items()]
    rng.shuffle(symbols)
    return grid.grid_drawing(
        alt="A map drawn on a square grid with column letters, row numbers, landmark symbols with a key and a north arrow. The squares are to be shaded to answer.",
        grid={"type": "square", "width": w, "height": h},
        symbols=symbols,
        legend=True,
        column_labels=list(LETTERS[:w]),
        row_labels=[str(i) for i in range(1, h + 1)],
        north=True,
        canvas="grid",
    )


def _landmarks(rng, w, h, n, avoid=()):
    names = rng.sample(_NAMEABLE, n)
    cells = {}
    for s in names:
        free = [(c, r) for c in range(w) for r in range(h) if (c, r) not in cells and (c, r) not in avoid]
        cells[rng.choice(free)] = s
    return cells


def _inside(cell, w, h):
    return 0 <= cell[0] < w and 0 <= cell[1] < h


def _cells_part(cells, w, h, wrong):
    """``(answer fields, distractors)`` for a grid_cells answer from ``wrong = [(set of cells, tag), ...]``."""
    key = {tuple(c) for c in cells}
    ans = cells_answer(sorted(key), (w, h))
    text = cell_text(key)
    cands = []
    for got, tag in wrong:
        got = {tuple(c) for c in got}
        if got and got != key and all(_inside(c, w, h) for c in got):
            cands.append((cell_text(got), cells_answer(sorted(got), (w, h)), tag))
    ds = dist("grid_cells", ans, text, cands, n=3)
    return dict(answer=ans, answer_text=text, answer_latex="plain:" + text), ds


def _question(text, fig, fields, ds, working):
    if len(ds) < 3:
        return None
    return dict(text=text, figure=fig, answer_kind="grid_cells", distractors=ds, working=working, **fields)


_WORD = {(1, 0): "east", (-1, 0): "west", (0, 1): "north", (0, -1): "south"}


def _move_text(dx, dy):
    parts = []
    if dx:
        parts.append(f"{abs(dx)} square{'s' if abs(dx) != 1 else ''} {'east' if dx > 0 else 'west'}")
    if dy:
        parts.append(f"{abs(dy)} square{'s' if abs(dy) != 1 else ''} {'north' if dy > 0 else 'south'}")
    return " and ".join(parts)


# ---------------------------------------------------------------------------------------------- easy

_EASY = (
    "Shade the square with grid reference {ref} for the new {feature}.",
    "The council will put {a_feature} in square {ref}. Shade that square.",
    "Shade square {ref}, where the {feature} will go.",
)


def _easy(rng):
    w, h = _size(rng)
    c, r = rng.randrange(1, w - 1), rng.randrange(1, h - 1)
    if rng.random() < 0.3:
        c, r = rng.randrange(w), rng.randrange(h)
    cells = _landmarks(rng, w, h, rng.randint(4, 5), avoid={(c, r)})
    fig = _figure(rng, cells, w, h)
    feature = rng.choice(_FEATURES)
    text = rng.choice(_EASY).format(ref=_ref(c, r), feature=feature, a_feature=_a(feature))
    wrong = [
        ({(r, c)} if _inside((r, c), w, h) else set(), "dg-gm-reference-reversed"),
        ({(c, h - 1 - r)}, "dg-gm-row-from-top"),
        ({(c + 1, r)}, "dg-gm-one-square-too-far"),
        ({(c - 1, r)}, "dg-gm-one-square-too-far"),
        ({(c, r + 1)}, "dg-gm-one-square-too-far"),
        ({(c, r - 1)}, "dg-gm-one-square-too-far"),
    ]
    fields, ds = _cells_part({(c, r)}, w, h, wrong)
    working = [
        f"The letter {LETTERS[c]} is column {c + 1} from the left; the number {r + 1} is row {r + 1} up from the bottom.",
        f"Shade the square where column {LETTERS[c]} meets row {r + 1}.",
    ]
    return _question(text, fig, fields, ds, working)


# ---------------------------------------------------------------------------------------------- medium

_MEDIUM = (
    "A new {feature} will go {move} of {the}. Shade the square for the {feature}.",
    "Start at {the}. The {feature} is {move} of it. Shade the square where the {feature} is.",
)


def _medium(rng):
    w, h = _size(rng)
    for _ in attempts(200, "a medium grid map"):
        cells = _landmarks(rng, w, h, rng.randint(4, 5))
        (sc, sr), sym = rng.choice(list(cells.items()))
        dx = rng.choice((-3, -2, -1, 1, 2, 3, 4)) if rng.random() < 0.85 else 0
        dy = rng.choice((-3, -2, -1, 1, 2, 3)) if dx or rng.random() < 1 else 0
        if dx == 0 and dy == 0:
            continue
        target = (sc + dx, sr + dy)
        if not _inside(target, w, h) or target in cells:
            continue
        wrong = [
            ({(sc - dx, sr + dy)} if dx else {(sc, sr + dy)}, "dg-gm-direction-reversed"),
            ({(sc + dx, sr - dy)} if dy else {(sc + dx, sr)}, "dg-gm-direction-reversed"),
            ({(sc + dy, sr + dx)}, "dg-gm-moves-swapped"),
            (
                {(sc + dx - (1 if dx > 0 else -1 if dx < 0 else 0), sr + dy - (1 if dy > 0 else -1 if dy < 0 else 0))},
                "dg-gm-counts-start-square",
            ),
            ({(sc + dx + (1 if dx > 0 else -1 if dx < 0 else 0), sr + dy)}, "dg-gm-one-square-too-far"),
        ]
        fields, ds = _cells_part({target}, w, h, wrong)
        fig = _figure(rng, cells, w, h)
        feature = rng.choice(_FEATURES)
        text = rng.choice(_MEDIUM).format(feature=feature, move=_move_text(dx, dy), the=_the(sym))
        working = [
            f"The {_name(sym)} is in square {_ref(sc, sr)}.",
            f"{_move_text(dx, dy).capitalize()} of it is square {_ref(*target)}.",
        ]
        q = _question(text, fig, fields, ds, working)
        if q is not None:
            return q
    raise AssertionError  # pragma: no cover


# ---------------------------------------------------------------------------------------------- hard


def _rectangle(rng):
    w, h = _size(rng)
    for _ in attempts(100, "a map rectangle"):
        c0, c1 = sorted(rng.sample(range(w), 2))
        r0, r1 = sorted(rng.sample(range(h), 2))
        if c1 - c0 > 3 or r1 - r0 > 3 or (c1 - c0 + 1) * (r1 - r0 + 1) < 4:
            continue
        cells = _landmarks(rng, w, h, rng.randint(3, 4))
        fig = _figure(rng, cells, w, h)
        region = {(c, r) for c in range(c0, c1 + 1) for r in range(r0, r1 + 1)}
        corners = {(c0, r0), (c1, r0), (c0, r1), (c1, r1)}
        wrong = [
            (corners, "dg-gm-region-corners-only"),
            ({(c, r) for c in range(c0, c1) for r in range(r0, r1 + 1)}, "dg-gm-region-edge-left-out"),
            ({(c, r) for c in range(c0, c1 + 1) for r in range(r0, r1)}, "dg-gm-region-edge-left-out"),
            ({(r, c) for c, r in region}, "dg-gm-reference-reversed"),
        ]
        fields, ds = _cells_part(region, w, h, wrong)
        name = rng.choice(("picnic area", "bush garden", "sports field", "market"))
        text = f"A {name} fills the rectangle of squares with opposite corners {_ref(c0, r0)} and {_ref(c1, r1)}. Shade every square of the {name}."
        working = [
            f"Columns {LETTERS[c0]} to {LETTERS[c1]} and rows {r0 + 1} to {r1 + 1}.",
            f"That is {c1 - c0 + 1} columns of {r1 - r0 + 1} squares: {len(region)} squares.",
        ]
        q = _question(text, fig, fields, ds, working)
        if q is not None:
            return q
    raise AssertionError  # pragma: no cover


def _between(rng):
    w, h = _size(rng)
    for _ in attempts(200, "a between question"):
        a, b = rng.sample(_NAMEABLE, 2)
        horizontal = rng.random() < 0.5
        if horizontal:
            r = rng.randrange(h)
            c0, c1 = sorted(rng.sample(range(w), 2))
            if c1 - c0 < 3 or c1 - c0 > 5:
                continue
            ca, cb = (c0, c1) if rng.random() < 0.5 else (c1, c0)
            pa, pb = (ca, r), (cb, r)
            between = {(c, r) for c in range(c0 + 1, c1)}
            ends = {pa, pb}
            short = {(c, r) for c in range(c0 + 1, c1 - 1)}
        else:
            c = rng.randrange(w)
            r0, r1 = sorted(rng.sample(range(h), 2))
            if r1 - r0 < 3 or r1 - r0 > 5:
                continue
            ra, rb = (r0, r1) if rng.random() < 0.5 else (r1, r0)
            pa, pb = (c, ra), (c, rb)
            between = {(c, r) for r in range(r0 + 1, r1)}
            ends = {pa, pb}
            short = {(c, r) for r in range(r0 + 1, r1 - 1)}
        cells = {pa: a, pb: b}
        extra = _landmarks(rng, w, h, 2, avoid=set(between) | ends)
        for k, v in extra.items():
            if v not in cells.values():
                cells[k] = v
        if between & set(cells):
            continue
        fig = _figure(rng, cells, w, h)
        wrong = [
            (between | ends, "dg-gm-between-includes-ends"),
            (short, "dg-gm-between-one-short"),
            (between | {pa}, "dg-gm-between-includes-ends"),
            (between | {pb}, "dg-gm-between-includes-ends"),
        ]
        fields, ds = _cells_part(between, w, h, wrong)
        way = "row" if horizontal else "column"
        text = f"{_the(a).capitalize()} and {_the(b)} are in the same {way}. A path joins them along that {way}. Shade every square of the path that is between the two landmarks, but not the squares of the landmarks."
        working = [
            f"The {_name(a)} is in {_ref(*pa)} and the {_name(b)} is in {_ref(*pb)}.",
            f"{len(between)} squares lie strictly between them in the {way}.",
        ]
        q = _question(text, fig, fields, ds, working)
        if q is not None:
            return q
    raise AssertionError  # pragma: no cover


def _hard(rng):
    return _rectangle(rng) if rng.random() < 0.5 else _between(rng)


@generator("dg_grid_map", level=8, presets={t: {"tier": t} for t in ("easy", "medium", "hard")})
@uses_rng
def dg_grid_map(rng, tier="easy"):
    if tier not in ("easy", "medium", "hard"):
        raise ValueError(f"unknown tier {tier!r}")
    build = {"easy": _easy, "medium": _medium, "hard": _hard}[tier]
    for _ in attempts(500, f"a {tier} grid-map question"):
        d = build(rng)
        if d is not None:
            return d
    raise AssertionError  # pragma: no cover


# ---------------------------------------------------------------------------------------------- exam chain


@generator("dg_grid_map_chain", level=8, presets={"exam": {"tier": "exam"}})
@uses_rng
def dg_grid_map_chain(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(600, "a grid-map chain"):
        w, h = _size(rng)
        cells = _landmarks(rng, w, h, 4)
        items = list(cells.items())
        (xc, xr), x = items[0]
        (yc, yr), y = items[1]
        (zc, zr), z = items[2]
        dx, dy = yc - xc, yr - xr
        if dx == 0 or dy == 0 or abs(dx) > 5 or abs(dy) > 5:
            continue
        if (abs(dx) == 1 or abs(dy) == 1) and rng.random() < 0.8:
            continue  # a gap of one square leaves nothing below it to offer
        ref_c, ref_r = rng.randrange(w), rng.randrange(h)
        if (ref_c, ref_r) in cells:
            continue
        mx, my = rng.choice((-2, -1, 1, 2)), rng.choice((-2, -1, 1, 2))
        target = (zc + mx, zr + my)
        if not _inside(target, w, h) or target in cells or target == (ref_c, ref_r):
            continue
        fig = _figure(rng, cells, w, h)
        feature = rng.choice(_FEATURES)
        fa, da = _cells_part(
            {(ref_c, ref_r)},
            w,
            h,
            [
                ({(ref_r, ref_c)}, "dg-gm-reference-reversed"),
                ({(ref_c, h - 1 - ref_r)}, "dg-gm-row-from-top"),
                ({(ref_c + 1, ref_r)}, "dg-gm-one-square-too-far"),
                ({(ref_c, ref_r + 1)}, "dg-gm-one-square-too-far"),
                ({(ref_c - 1, ref_r)}, "dg-gm-one-square-too-far"),
            ],
        )
        fe, de = _cells_part(
            {target},
            w,
            h,
            [
                ({(zc - mx, zr + my)}, "dg-gm-direction-reversed"),
                ({(zc + mx, zr - my)}, "dg-gm-direction-reversed"),
                ({(zc + my, zr + mx)}, "dg-gm-moves-swapped"),
                ({(zc + mx - (1 if mx > 0 else -1), zr + my - (1 if my > 0 else -1))}, "dg-gm-counts-start-square"),
            ],
        )
        ax, ay = abs(dx), abs(dy)
        db = numd(
            ax,
            [
                (ax + 1, "dg-gm-count-includes-start"),
                (ay, "dg-gm-count-reference-difference"),
                (ax - 1, "dg-gm-counts-start-square"),
                (ax + ay, "dg-gm-count-reference-difference"),
            ],
            lo=min(ax, 1),  # a count of squares moved: 0 is no answer here
            hi=w - 1,
            step=1,
        )
        dc = numd(
            ay,
            [
                (ay + 1, "dg-gm-count-includes-start"),
                (ax, "dg-gm-count-reference-difference"),
                (ay - 1, "dg-gm-counts-start-square"),
                (ax + ay, "dg-gm-count-reference-difference"),
            ],
            lo=min(ay, 1),
            hi=h - 1,
            step=1,
        )
        walk = ax + ay
        dd = numd(
            walk,
            [
                (max(ax, ay), "dg-gm-walk-as-diagonal"),
                (ax * ay, "dg-gm-walk-product"),
                (walk + 1, "dg-gm-count-includes-start"),
                (walk - 1, "dg-gm-counts-start-square"),
            ],
            lo=min(walk, 1),
            hi=w + h - 2,
            step=1,
        )
        if len(da) < 3 or len(de) < 3 or min(len(v) for v in (db, dc, dd)) < 2:
            continue
        ew = "east" if dx > 0 else "west"
        ns = "north" if dy > 0 else "south"
        text = f"A map of a park is drawn on a grid. The {_name(x)} is in square {_ref(xc, xr)}, and the {_name(y)} is in square {_ref(yc, yr)}."
        parts = [
            dict(
                prompt=f"Shade the square {_ref(ref_c, ref_r)} for a new {feature}.",
                answer_kind="grid_cells",
                marks=2,
                working=[f"Column {LETTERS[ref_c]}, row {ref_r + 1}."],
                distractors=da,
                **fa,
            ),
            p_number(
                f"How many squares {ew} of the {_name(x)} is the {_name(y)}?",
                ax,
                1,
                [f"Columns {LETTERS[xc]} and {LETTERS[yc]} are {ax} apart."],
                db,
                sensible_range=(0, w),
            ),
            p_number(
                f"How many squares {ns} of the {_name(x)} is the {_name(y)}?",
                ay,
                1,
                [f"Rows {xr + 1} and {yr + 1} are {ay} apart."],
                dc,
                sensible_range=(0, h),
            ),
            p_number(
                f"A walker goes from the {_name(x)} to the {_name(y)} along the row and then along the column. How many squares does the walker move?",
                walk,
                2,
                [f"{ax} + {ay} = {walk}."],
                dd,
                follow_through="part_b + part_c",
                sensible_range=(0, w + h),
            ),
            dict(
                prompt=f"A bike rack is {_move_text(mx, my)} of the {_name(z)}. Shade the square for the bike rack.",
                answer_kind="grid_cells",
                marks=2,
                working=[
                    f"The {_name(z)} is in square {_ref(zc, zr)}.",
                    f"{_move_text(mx, my).capitalize()} of it is square {_ref(*target)}.",
                ],
                distractors=de,
                **fe,
            ),
        ]
        return dict(text=text, figure=fig, parts=parts)
    raise AssertionError  # pragma: no cover


# ---------------------------------------------------------------------------------------------- the reader


def _symbol_cells(fig):
    return {(int(s["at"][0]), int(s["at"][1])): s["symbol"] for s in fig["symbols"]}


def _cell_of(ref):
    m = re.fullmatch(r"([A-Z])(\d+)", ref)
    return LETTERS.index(m.group(1)), int(m.group(2)) - 1


def _named(fig, name):
    found = [c for c, s in _symbol_cells(fig).items() if grid.SYMBOLS[s] == name]
    check(len(found) == 1, f"the {name} must be on the map once")
    return found[0]


_DIRS = {"east": (1, 0), "west": (-1, 0), "north": (0, 1), "south": (0, -1)}


def _read_map(question) -> str:
    """The shaded squares recomputed from the map's symbols and the reference, move or corners in the question."""
    fig, text = question.figure, question.text
    m = re.search(r"grid reference ([A-J]\d+)|in square ([A-J]\d+)\. Shade|Shade square ([A-J]\d+),", text)
    if m is not None:
        return cell_text({_cell_of(next(g for g in m.groups() if g))})
    if (m := re.search(r"corners ([A-J]\d+) and ([A-J]\d+)", text)) is not None:
        (c0, r0), (c1, r1) = _cell_of(m.group(1)), _cell_of(m.group(2))
        return cell_text(
            {(c, r) for c in range(min(c0, c1), max(c0, c1) + 1) for r in range(min(r0, r1), max(r0, r1) + 1)}
        )
    if "between the two landmarks" in text:
        m = re.match(r"The (.+?) and the (.+?) are in the same (row|column)", text)
        a, b = _named(fig, m.group(1)), _named(fig, m.group(2))
        if m.group(3) == "row":
            lo, hi = sorted((a[0], b[0]))
            return cell_text({(c, a[1]) for c in range(lo + 1, hi)})
        lo, hi = sorted((a[1], b[1]))
        return cell_text({(a[0], r) for r in range(lo + 1, hi)})
    m = re.search(
        r"will go ((?:\d+ squares? (?:east|west|north|south))(?: and \d+ squares? (?:east|west|north|south))?) of the (.+?)\. Shade",
        text,
    ) or re.search(
        r"Start at the (.+?)\. The [\w -]+ is ((?:\d+ squares? (?:east|west|north|south))(?: and \d+ squares? (?:east|west|north|south))?) of it",
        text,
    )
    check(m is not None, f"cannot read the question {text!r}")
    if text.startswith("Start at"):
        name, move = m.group(1), m.group(2)
    else:
        move, name = m.group(1), m.group(2)
    x, y = _named(fig, name)
    for count, way in re.findall(r"(\d+) squares? (east|west|north|south)", move):
        dx, dy = _DIRS[way]
        x, y = x + dx * int(count), y + dy * int(count)
    return cell_text({(x, y)})


figures.register_reader("dg_grid_map", _read_map)
