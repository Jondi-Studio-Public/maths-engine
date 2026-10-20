"""Exam chains for the plan, site-plan and map-reading skills (the ``w7f`` figure work).

Three chains, each read off one drawn plan or map on a square grid (``grid_drawing``).

* ``exd_x_w7f_plan_symbols_chain`` (``senior-interpret-commonly-used-symbols-and-abbreviations-in-scale``): a floor plan
  of eight rooms named by abbreviations. (a) the room with the toilets symbol and no window (``mcq``), (b) the room
  with the hot water system (``mcq``), (c) how many rooms have a window symbol, (d) how many rooms a plumber must
  visit (toilets symbol, hot water system or dishwasher), (e) the plumber's charge at a stated price per room (follows
  (d)).
* ``exd_x_w7f_plan_conventions_chain`` (``senior-use-symbols-and-conventions-for-the-representation-of``): a site plan
  with solid (existing) and dashed (proposed) outlines, dimension labels in metres and a north arrow. (a) the proposed
  feature, named in full from its abbreviation (``mcq``), (b) its compass direction from an existing building
  (``mcq``), (c) its floor area, (d) the area covered by every building once it is built (follows (c)).
* ``exd_x_w7f_map_landmarks_chain`` (``f10-map-feature-and-landmark-representation``): a map with columns A, B, C...
  and rows 1, 2, 3... counted from the bottom. (a) the landmark in a named square, (b) the grid reference of a named
  landmark, (c) the square reached after two moves, (d) the landmark in the square reached (all ``mcq``).

Each chain is built from the drawn arrangement first and its prompts derived from it; the reader registered for each
topic re-derives every part from the figure and the prompts alone.
"""

from __future__ import annotations

import re

from mathgen import figures
from mathgen.core import check, generator
from mathgen.figure_kinds import grid

from . import _space_kit as kit
from ._w7f_kit import finish_parts, mcq_part, number_part, pick, retry, split_parts
from .grid_map_symbols_build import (
    _BUILT,
    _PLANNED,
    COMPASS,
    FEATURES,
    FIXTURES,
    KEYED,
    LETTERS,
    NAMEABLE,
    SITE_H,
    SITE_W,
    WET,
    _BY_VECTOR,
    _box,
    _cell_of,
    _dim,
    _features_of,
    _fill,
    _inside,
    _map_figure,
    _name,
    _plan_figure,
    _plan_rooms,
    _plan_rooms_of,
    _ref,
    _ref_of,
    _rooms_with,
    _sign,
    _site_figure,
    _symbol_cells,
    _symbol_named,
    room_full,
)


# ------------------------------------------------------------------------------------------- 1. floor-plan symbols

_TOILET_NO_WINDOW = "Which room has the toilets symbol but no window symbol?"
_HWS_ASK = "In which room is the hot water system?"
_WINDOWS_ASK = "How many rooms on this plan have a window symbol?"
_PLUMBER_ASK = (
    "A plumber must visit every room that has a toilets symbol, a hot water system (HWS) or a dishwasher (DW). "
    "How many rooms is that?"
)
_CHARGE_ASK = "The plumber charges ${price} for each room visited. Find the total charge, in dollars."
_CHARGE_RE = re.compile(r"charges \$(\d+) for each room")


def _neighbours_of(room, rooms):
    x0, y0, x1, y1 = room["box"]
    return [
        o
        for o in rooms
        if o is not room and o["box"][0] <= x1 and x0 <= o["box"][2] and o["box"][1] <= y1 and y0 <= o["box"][3]
    ]


def _symbols_chain(rng):
    wet = rng.choice(("ENS", "WC"))
    rooms = _plan_rooms(rng, wet)
    by = {r["abbr"]: r for r in rooms}
    toilets = {wet, "BATH"}
    windowless_toilet = rng.choice(sorted(toilets))
    others_dark = [a for a in ("L'DRY", "WIR", "PAN", "STU", "GAR") if a in by and rng.random() < 0.45]
    dark = {windowless_toilet, *others_dark}
    hws_home = rng.choice([a for a in ("L'DRY", "GAR", "BATH") if a in by])
    bedroom = rng.choice([a for a in by if a.startswith("BR")])
    fixtures = {hws_home: "HWS", "KIT": "DW", bedroom: "BIR"}
    fig = _plan_figure(rooms, dark=dark, toilets=toilets, fixtures=fixtures, legend=True, answer="Kitchen")
    if fig is None:
        return None
    visits = toilets | {hws_home, "KIT"}
    windows = len(by) - len(dark)
    price = rng.choice((75, 80, 85, 90, 95, 105, 110, 115, 120, 125, 135, 140, 150, 160, 175, 185))

    key_a = room_full(windowless_toilet)
    other_toilet = next(t for t in toilets if t != windowless_toilet)
    part_a = mcq_part(
        _TOILET_NO_WINDOW,
        key_a,
        1,
        [
            f"The toilets symbol is in {wet} ({room_full(wet)}) and in BATH (Bathroom).",
            f"{windowless_toilet} is the one of them with no window symbol: {key_a}.",
        ],
        [
            (room_full(other_toilet), "plan-ignores-the-window-condition"),
            *[(room_full(a), "plan-window-condition-without-toilets") for a in sorted(dark) if a != windowless_toilet],
            *[(room_full(o["abbr"]), "plan-neighbouring-room") for o in _neighbours_of(by[windowless_toilet], rooms)],
            *[(room_full(a), "plan-other-wet-room") for a in WET if a in by and a not in toilets],
        ],
    )
    key_b = room_full(hws_home)
    part_b = mcq_part(
        _HWS_ASK,
        key_b,
        1,
        [f"HWS stands for hot water system. The label HWS is in the room marked {hws_home}: {key_b}."],
        [
            (room_full("KIT"), "plan-other-fixture-room"),
            (room_full(bedroom), "plan-other-fixture-room"),
            *[(room_full(o["abbr"]), "plan-neighbouring-room") for o in _neighbours_of(by[hws_home], rooms)],
            *[(room_full(a), "plan-other-wet-room") for a in WET if a in by and a != hws_home],
        ],
    )
    part_c = number_part(
        _WINDOWS_ASK,
        windows,
        2,
        [f"Rooms without a window symbol: {', '.join(sorted(dark))}.", f"{len(by)} - {len(dark)} = {windows} rooms."],
        [
            (len(dark), "plan-counts-rooms-without-window"),
            (len(by), "plan-counts-all-rooms"),
            (windows + 1, "plan-count-off-by-one"),
            (windows - 1, "plan-count-off-by-one"),
        ],
    )
    count = len(visits)
    part_d = number_part(
        _PLUMBER_ASK,
        count,
        2,
        [
            f"Toilets symbol: {', '.join(sorted(toilets))}. Hot water system: {hws_home}. Dishwasher: KIT.",
            f"Together that is {count} different rooms.",
        ],
        [
            (len(toilets | {hws_home}), "plan-omits-the-dishwasher"),
            (len(toilets), "plan-counts-toilets-only"),
            (len(toilets) + 2, "plan-counts-a-shared-room-twice"),
            (len(visits | {bedroom}), "plan-counts-the-robe-room"),
            (count + 1, "plan-count-off-by-one"),
            (count - 1, "plan-count-off-by-one"),
            (count + 2, "plan-count-off-by-one"),
        ],
    )
    part_e = number_part(
        _CHARGE_ASK.format(price=price),
        count * price,
        1,
        [f"{count} × ${price} = ${count * price}."],
        [
            ((count - 1) * price, "plan-omits-the-dishwasher"),
            ((count + 1) * price, "plan-count-off-by-one"),
            (count + price, "scd-cost-adds-price"),
            (len(toilets) * price, "plan-counts-toilets-only"),
            (windows * price, "plan-charges-per-window-room"),
        ],
        follow=f"{price}*part_d",
    )
    stem = pick(
        rng,
        (
            "The floor plan of a house is drawn on a square grid. The rooms are named by the abbreviations a plan uses.",
            "An architect has drawn the plan of a house on a square grid, and has labelled each room with the usual "
            "plan abbreviation.",
            "A house plan is drawn to a square grid. Rooms are written as short plan abbreviations, and the symbols "
            "on it are the standard ones.",
        ),
    )
    return finish_parts(fig, stem, [part_a, part_b, part_c, part_d, part_e], lower=True)


@generator("exd_x_w7f_plan_symbols_chain", level=11, presets={"exam": {"tier": "exam"}})
def exd_x_w7f_plan_symbols_chain(rng, tier: str = "exam"):
    """Toilets and windows, the hot water system, a count of windows and a plumber's rooms and charge."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry(rng, _symbols_chain, "a floor plan with enough distractors for every part")


def _label_room(fig, rooms, code):
    spots = [lb["at"] for lb in fig["labels"] if lb["text"] == code]
    check(len(spots) == 1, f"the label {code} must appear once")
    homes = [r["abbr"] for r in rooms if _inside(r["box"], spots[0])]
    check(len(homes) == 1, f"the label {code} must sit in one room")
    return homes[0]


def _read_symbols_chain(question) -> str:
    fig, parts = question.figure, split_parts(question.text)
    check(parts["a"] == _TOILET_NO_WINDOW and parts["b"] == _HWS_ASK and parts["c"] == _WINDOWS_ASK, "unknown prompts")
    rooms = _plan_rooms_of(fig)
    windowed = set(_rooms_with(fig, rooms, "window"))
    toilets = set(_rooms_with(fig, rooms, "toilets"))
    lone = [t for t in toilets if t not in windowed]
    check(len(lone) == 1, "exactly one toilets room lacks a window")
    hws, dw = _label_room(fig, rooms, "HWS"), _label_room(fig, rooms, "DW")
    visits = len(toilets | {hws, dw})
    price = int(_CHARGE_RE.search(parts["e"]).group(1))
    return f"(a) {room_full(lone[0])}; (b) {room_full(hws)}; (c) {len(windowed)}; (d) {visits}; (e) {visits * price}"


figures.register_reader("exd_x_w7f_plan_symbols_chain", _read_symbols_chain)

# ------------------------------------------------------------------------------------------- 2. site-plan conventions

_PROPOSED_ASK = "Which feature on this plan is proposed, and not yet built?"
_DIRECTION_ASK = "From the {a}, in which direction is the proposed feature?"
_DIRECTION_RE = re.compile(r"From the (.+?), in which direction is the proposed feature\?")
_AREA_ASK = (
    "The dimensions on this plan are in metres. What is the floor area, in square metres, of the proposed feature?"
)
_TOTAL_ASK = (
    "Once the proposed feature is built, what is the total floor area, in square metres, of all the buildings and "
    "features on the plan?"
)


def _site_boxes(rng):
    """Four boxes ``(x0, y0, x1, y1)``: existing A, proposed B on a compass line from A, and two more existing."""
    direction = rng.choice(sorted(COMPASS))
    dx, dy = COMPASS[direction]
    gap = rng.randint(4, 7)
    parity_w, parity_h = rng.choice((2, 3)), rng.choice((2, 3))
    sizes = []
    for _ in range(2):
        sizes.append((rng.choice((parity_w, parity_w + 2)), rng.choice((parity_h, parity_h + 2))))
    (wa, ha), (wb, hb) = sizes
    ax, ay = rng.randint(1, SITE_W - wa - 2), rng.randint(1, SITE_H - ha - 2)
    # centre of B is `gap` squares from the centre of A along the compass line (doubled so half squares stay exact)
    bx2 = 2 * ax + wa + 2 * gap * dx - wb
    by2 = 2 * ay + ha + 2 * gap * dy - hb
    if bx2 % 2 or by2 % 2:
        return None
    bx, by = bx2 // 2, by2 // 2
    if not (1 <= bx <= SITE_W - wb - 2 and 1 <= by <= SITE_H - hb - 2):
        return None
    boxes = [(ax, ay, ax + wa, ay + ha), (bx, by, bx + wb, by + hb)]
    for _ in range(2):
        w, h = rng.randint(2, 4), rng.randint(2, 4)
        for _try in range(80):
            x, y = rng.randint(1, SITE_W - w - 2), rng.randint(1, SITE_H - h - 2)
            new = (x, y, x + w, y + h)
            if all(new[2] + 3 <= b[0] or b[2] + 3 <= new[0] or new[3] + 3 <= b[1] or b[3] + 3 <= new[1] for b in boxes):
                boxes.append(new)
                break
        else:
            return None
    # the two main boxes, too, must clear the extra labels
    a, b = boxes[0], boxes[1]
    if not (a[2] + 3 <= b[0] or b[2] + 3 <= a[0] or a[3] + 3 <= b[1] or b[3] + 3 <= a[1]):
        return None
    return direction, boxes


def _conventions_chain(rng):
    drawn = _site_boxes(rng)
    if drawn is None:
        return None
    direction, boxes = drawn
    scale_m = rng.choice((2, 5))
    existing_names = rng.sample(_BUILT, 3)
    proposed = rng.choice(_PLANNED)
    names = [existing_names[0], proposed, existing_names[1], existing_names[2]]
    dims = []
    for x0, y0, x1, y1 in boxes:
        dims.append({"at": [(x0 + x1) / 2, y1 + 0.5], "text": f"{(x1 - x0) * scale_m} m"})
        dims.append({"at": [x1 + 1, (y0 + y1) / 2], "text": f"{(y1 - y0) * scale_m} m"})
    key_a = FEATURES[proposed]
    fig = _site_figure(boxes, names, dashed={1}, dim_labels=dims, scale=f"1 : {100 * scale_m}", answer=key_a)
    if fig is None:
        return None
    areas = [(b[2] - b[0]) * scale_m * (b[3] - b[1]) * scale_m for b in boxes]
    total = sum(areas)
    dx, dy = COMPASS[direction]
    ann = FEATURES[existing_names[0]].lower()
    part_a = mcq_part(
        _PROPOSED_ASK,
        key_a,
        1,
        [f"A dashed outline shows a feature that is proposed. Only {proposed} is dashed: {key_a}."],
        [(FEATURES[n], "plan-names-an-existing-building") for n in existing_names],
    )
    part_b = mcq_part(
        _DIRECTION_ASK.format(a=ann),
        direction,
        1,
        [
            "The north arrow points up the page, so up is north, down is south, right is east and left is west.",
            f"The proposed feature is {direction} of the {ann}.",
        ],
        [
            (_BY_VECTOR[(-dx, -dy)], "plan-reverses-the-direction"),
            (_BY_VECTOR[(-dx, dy)], "plan-swaps-east-and-west"),
            (_BY_VECTOR[(dx, -dy)], "plan-swaps-north-and-south"),
            (_BY_VECTOR[(_sign(dx + dy), _sign(dy - dx))], "plan-turns-the-direction"),
            (_BY_VECTOR[(_sign(dx - dy), _sign(dx + dy))], "plan-turns-the-direction"),
        ],
    )
    pw, ph = (boxes[1][2] - boxes[1][0]) * scale_m, (boxes[1][3] - boxes[1][1]) * scale_m
    part_c = number_part(
        _AREA_ASK,
        areas[1],
        2,
        [f"The proposed feature is the dashed outline, {pw} m by {ph} m.", f"{pw} x {ph} = {areas[1]} square metres."],
        [
            (2 * (pw + ph), "plan-perimeter-for-area"),
            (areas[0], "plan-reads-the-existing-building"),
            (areas[1] // (scale_m * scale_m), "plan-area-from-squares-on-the-plan"),
            (pw + ph, "plan-adds-the-sides"),
            (areas[2], "plan-reads-the-existing-building"),
        ],
    )
    existing_sum = total - areas[1]
    part_d = number_part(
        _TOTAL_ASK,
        total,
        2,
        [
            f"Existing buildings: {' + '.join(str(a) for i, a in enumerate(areas) if i != 1)} = {existing_sum}.",
            f"{existing_sum} + {areas[1]} = {total} square metres.",
        ],
        [
            (existing_sum, "plan-leaves-out-the-proposed-feature"),
            (total - areas[2], "plan-leaves-out-a-building"),
            (total - areas[0], "plan-leaves-out-a-building"),
            (total - areas[3], "plan-leaves-out-a-building"),
            (total + areas[1], "plan-counts-the-proposed-feature-twice"),
            (areas[1] + areas[0], "plan-leaves-out-a-building"),
        ],
        follow=f"{existing_sum}+part_c",
    )
    check(sum(areas) == total, "areas disagree")
    stem = (
        "A site plan is drawn on a square grid with the north arrow shown. Solid outlines are existing and dashed "
        "outlines are proposed. Features are named by abbreviations."
    )
    return finish_parts(fig, stem, [part_a, part_b, part_c, part_d], lower=True)


@generator("exd_x_w7f_plan_conventions_chain", level=11, presets={"exam": {"tier": "exam"}})
def exd_x_w7f_plan_conventions_chain(rng, tier: str = "exam"):
    """The proposed feature, its direction from a building, its area and the total area covered."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry(rng, _conventions_chain, "a site plan with enough distractors for every part")


def _read_conventions_chain(question) -> str:
    fig, parts = question.figure, split_parts(question.text)
    feats = _features_of(fig)
    dashed = [f for f in feats if f["style"] == "dashed"]
    check(len(dashed) == 1 and parts["a"] == _PROPOSED_ASK, "exactly one feature is proposed")
    full = {v.lower(): k for k, v in FEATURES.items()}
    start = next(f for f in feats if f["name"] == full[_DIRECTION_RE.search(parts["b"]).group(1)])
    target = dashed[0]
    cx = lambda f: f["box"][0] + f["box"][2]  # noqa: E731  (doubled centre)
    cy = lambda f: f["box"][1] + f["box"][3]  # noqa: E731
    dx, dy = cx(target) - cx(start), cy(target) - cy(start)
    check(dx == 0 or dy == 0 or abs(dx) == abs(dy), "the two features are not on a compass line")

    def area(f):
        x0, y0, x1, y1 = f["box"]
        return _dim(fig, [(x0 + x1) / 2, y1 + 0.5]) * _dim(fig, [x1 + 1, (y0 + y1) / 2])

    total = sum(area(f) for f in feats)
    return (
        f"(a) {FEATURES[dashed[0]['name']]}; (b) {_BY_VECTOR[(_sign(dx), _sign(dy))]}; (c) {area(target)}; (d) {total}"
    )


figures.register_reader("exd_x_w7f_plan_conventions_chain", _read_conventions_chain)

# ------------------------------------------------------------------------------------------- 3. map landmarks

_MOVES = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
_SQ_ASK = "Which landmark is in square {ref}?"
_REF_ASK = "What is the grid reference of the {name}? Give the column letter first, then the row number."
_GO_ASK = "Start at the {name}. Go {first}, then {second}. Which square do you finish in?"
_AT_ASK = "Which landmark is in the square you finish in?"


def _squares(n: int) -> str:
    return f"{n} square" if n == 1 else f"{n} squares"


def _landmarks_chain(rng):
    width, height = rng.choice((6, 7)), rng.choice((5, 6))
    cells: dict = {}
    _fill(rng, cells, NAMEABLE, width, height, rng.randint(7, 9))
    if len(cells) < 6:
        return None
    sites = sorted(cells)
    probe, named, start = rng.sample(sites, 3)
    # the finishing square holds a landmark, reached by two moves from `start`
    for _ in range(40):
        a, b = rng.randint(1, 4), rng.randint(1, 4)
        sx, sy = rng.choice((1, -1)), rng.choice((1, -1))
        end = (start[0] + sx * a, start[1] + sy * b)
        if end in cells and end != start and end not in (probe,):
            break
    else:
        return None
    c, r = start
    wrong_ends = [
        (c - sx * a, r + sy * b),
        (c + sx * a, r - sy * b),
        (c + sx * b, r + sy * a),
        (c + sx * (a - 1), r + sy * (b - 1)),
        (c - sx * a, r - sy * b),
    ]
    inside = lambda cell: 0 <= cell[0] < width and 0 <= cell[1] < height  # noqa: E731
    wrong_ends = [w for w in wrong_ends if inside(w) and w != end]
    legs = [f"{_squares(a)} {'east' if sx > 0 else 'west'}", f"{_squares(b)} {'north' if sy > 0 else 'south'}"]
    if rng.random() < 0.5:
        legs.reverse()
    fig = _map_figure(rng, cells, width, height, legend=False, answer="zz")
    if fig is None:
        return None
    probe_ref, named_ref, end_ref = _ref(*probe), _ref(*named), _ref(*end)
    swapped = lambda cell: _ref(cell[1], cell[0]) if cell[1] < width and cell[0] < height else None  # noqa: E731
    near = lambda cell: [  # noqa: E731
        (cell[0] + d[0], cell[1] + d[1])
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1))
        if inside((cell[0] + d[0], cell[1] + d[1]))
    ]
    part_a = mcq_part(
        _SQ_ASK.format(ref=probe_ref),
        _name(cells[probe]),
        1,
        [f"Square {probe_ref} is column {LETTERS[probe[0]]} and row {probe[1] + 1} up from the bottom."],
        [
            *[(_name(cells[x]), "map-neighbouring-square") for x in near(probe) if x in cells],
            (_name(cells[(probe[1], probe[0])]), "map-swaps-column-and-row")
            if (probe[1], probe[0]) in cells and probe[1] < width and probe[0] < height
            else (None, ""),
            *[(_name(v), "map-other-landmark") for k, v in sorted(cells.items()) if k != probe],
        ],
    )
    part_b = mcq_part(
        _REF_ASK.format(name=grid.SYMBOLS[cells[named]]),
        named_ref,
        1,
        [
            f"The {grid.SYMBOLS[cells[named]]} is in column {LETTERS[named[0]]} and row {named[1] + 1} up from the bottom."
        ],
        [
            (swapped(named), "map-swaps-column-and-row") if swapped(named) else (None, ""),
            (_ref(named[0], height - 1 - named[1]), "map-counts-rows-from-the-top"),
            *[(_ref(*x), "map-neighbouring-square") for x in near(named)],
        ],
    )
    part_c = mcq_part(
        _GO_ASK.format(name=grid.SYMBOLS[cells[start]], first=legs[0], second=legs[1]),
        end_ref,
        2,
        [
            f"The {grid.SYMBOLS[cells[start]]} is in square {_ref(*start)}.",
            f"Going {legs[0]} and {legs[1]} ends in column {LETTERS[end[0]]}, row {end[1] + 1}: {end_ref}.",
        ],
        [
            (_ref(*wrong_ends[0]), "map-confuses-east-and-west") if len(wrong_ends) > 0 else (None, ""),
            (_ref(*wrong_ends[1]), "map-confuses-north-and-south") if len(wrong_ends) > 1 else (None, ""),
            (_ref(*wrong_ends[2]), "map-swaps-the-two-distances") if len(wrong_ends) > 2 else (None, ""),
            (_ref(*wrong_ends[3]), "map-counts-the-start-square") if len(wrong_ends) > 3 else (None, ""),
            (_ref(*wrong_ends[4]), "map-reverses-both-directions") if len(wrong_ends) > 4 else (None, ""),
        ],
    )
    part_d = mcq_part(
        _AT_ASK,
        _name(cells[end]),
        2,
        [f"The square you finish in is {end_ref}, which holds the {grid.SYMBOLS[cells[end]]}."],
        [
            *[(_name(cells[w]), "map-lands-in-a-wrong-square") for w in wrong_ends if w in cells],
            (_name(cells[start]), "map-names-the-start"),
            *[(_name(v), "map-other-landmark") for k, v in sorted(cells.items()) if k not in (end, start)],
        ],
    )
    stem = (
        "The map is drawn on a square grid. Columns are lettered from the left and rows are numbered from the bottom."
    )
    # The picture shows icons, not names, so an answer that is a name can only leak through a legend; none is drawn.
    out = finish_parts(fig, stem, [part_a, part_b, part_c, part_d], lower=True)
    return out


@generator("exd_x_w7f_map_landmarks_chain", level=3, presets={"exam": {"tier": "exam"}})
def exd_x_w7f_map_landmarks_chain(rng, tier: str = "exam"):
    """A landmark in a square, a grid reference, a two-move journey and the landmark where it ends."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry(rng, _landmarks_chain, "a map with enough distractors for every part")


_MOVE_RE = re.compile(r"Start at the (.+?)\. Go (\d+) squares? (\w+), then (\d+) squares? (\w+)\. ")


def _read_landmarks_chain(question) -> str:
    fig, parts = question.figure, split_parts(question.text)
    cells = _symbol_cells(fig)
    probe = _cell_of(fig, re.fullmatch(r"Which landmark is in square (\w+)\?", parts["a"]).group(1))
    name = re.fullmatch(r"What is the grid reference of the (.+?)\? .*", parts["b"]).group(1)
    m = _MOVE_RE.match(parts["c"] + " ")
    x, y = _symbol_named(fig, m.group(1))
    for count, way in ((m.group(2), m.group(3)), (m.group(4), m.group(5))):
        dx, dy = _MOVES[way]
        x, y = x + dx * int(count), y + dy * int(count)
    check((x, y) in cells, "the journey ends on an empty square")
    return (
        f"(a) {grid.SYMBOLS[cells[probe]].capitalize()}; (b) {_ref_of(fig, _symbol_named(fig, name))}; "
        f"(c) {_ref_of(fig, (x, y))}; (d) {grid.SYMBOLS[cells[(x, y)]].capitalize()}"
    )


figures.register_reader("exd_x_w7f_map_landmarks_chain", _read_landmarks_chain)
