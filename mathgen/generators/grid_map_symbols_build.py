"""Generates three `grid_drawing` multiple-choice skills about maps, plans and their conventions.

Topics (each a `grid_drawing` on a square grid, `mcq` answers, easy / medium / hard):

``grid_plan_symbols`` (skill ``senior-interpret-commonly-used-symbols-and-abbreviations-in-scale``)
    A floor plan of eight rooms. Rooms are named by the abbreviations a plan uses (``BR 1``, ``ENS``, ``L'DRY``,
    ``WIR``) and fixtures by ``DW``, ``HWS``, ``BIR``; windows and toilets are drawn as symbols. Every answer is a
    full name or a count, so decoding an abbreviation is always part of the work.

    - easy: find the room that has the toilets symbol, or the one room with no window symbol, and name it in full.
    - medium: name the room that holds a fixture asked for by its full name (``HWS`` is the hot water system).
    - hard: a count that needs decoded rooms and symbols together (wet areas with a window; rooms a plumber visits).

``grid_plan_conventions`` (skill ``senior-use-symbols-and-conventions-for-the-representation-of``)
    A site plan with solid and dashed outlines, a north arrow, a scale written as a ratio and dimension labels.

    - easy: which feature is the dashed one (proposed, or hidden below ground), named in full from its abbreviation.
    - medium: the compass direction from one feature to another, with the north arrow up the page.
    - hard: the area of the existing (solid) or proposed (dashed) building from the dimension labels in metres.

``grid_map_landmarks`` (skill ``f10-map-feature-and-landmark-representation``)
    A small map with columns A, B, C... and rows 1, 2, 3... counted from the bottom, landmark symbols and a north arrow.

    - easy: which landmark is in a named square.
    - medium: the grid reference of a named landmark, or which landmark lies in a compass direction from a square.
    - hard: follow two moves ("go 2 squares east, then 3 north") from a named landmark to the finishing square.

When the answer is the name of a landmark or room, the legend would hand it over, so those questions draw no
legend (`figures.assert_no_leak` and a lower-case twin of it both run on every figure). Questions whose answer is a
grid reference or a count keep the legend. Each reader re-derives the answer from the figure and the question text.
"""

from __future__ import annotations

import re

from .. import figures
from ..core import attempts, check, generator, pick_distractors
from ..figure_kinds import grid
from . import _space_kit as kit

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _leaks(fig: dict, answer: str) -> bool:
    """The figure shows the answer, in the case written or in lower case."""
    return kit.leaks(fig, answer) or kit.leaks(fig, answer.lower())


def _draw(answer: str, legend: bool, **spec) -> dict | None:
    """The figure with a legend if that does not give the answer away, without one if it would, else ``None``."""
    for with_legend in (True, False) if legend else (False,):
        fig = grid.grid_drawing(legend=with_legend, **spec)
        if not _leaks(fig, answer):
            return fig
    return None


def _question(text, answer, distractors, working, figure):
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer,
        answer_latex=f"plain:{answer}",
        distractors=distractors,
        working=working,
        figure=figure,
    )


def _three(answer: str, candidates) -> list[str]:
    picked = pick_distractors(answer, [str(c) for c in candidates])[:3]
    check(len(picked) == 3, "fewer than three distinct plausible wrong answers")
    return picked


def _rooms(n: int) -> str:
    return f"{n} room" if n == 1 else f"{n} rooms"


def _area(n: int) -> str:
    return f"{n} m\u00b2"


def _squares(n: int) -> str:
    return f"{n} square" if n == 1 else f"{n} squares"


def _tier(table: dict, tier: str):
    if tier not in table:
        raise ValueError(f"unknown tier {tier!r}")
    return table[tier]


def _split(rng, total: int, parts: int, minimum: int) -> list[int]:
    """``parts`` whole numbers, each at least ``minimum``, adding to ``total``."""
    sizes = [minimum] * parts
    for _ in range(total - minimum * parts):
        sizes[rng.randrange(parts)] += 1
    return sizes


def _inside(box, at) -> bool:
    x0, y0, x1, y1 = box
    return x0 <= at[0] <= x1 and y0 <= at[1] <= y1


def _box(vertices) -> tuple:
    xs, ys = [v[0] for v in vertices], [v[1] for v in vertices]
    return min(xs), min(ys), max(xs), max(ys)


def _rect(x0, y0, x1, y1) -> list[list[int]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


# =========================================================================== skill 1: plan symbols and abbreviations

PLAN_W, PLAN_H, WALL = 20, 10, 5
ROOMS = {
    "KIT": "Kitchen",
    "LIV": "Living room",
    "DIN": "Dining room",
    "BATH": "Bathroom",
    "ENS": "Ensuite",
    "WC": "Toilet",
    "L'DRY": "Laundry",
    "GAR": "Garage",
    "WIR": "Walk-in robe",
    "PAN": "Pantry",
    "STU": "Study",
}
FIXTURES = {"DW": "dishwasher", "HWS": "hot water system", "BIR": "built-in robe"}
WET = ("BATH", "ENS", "WC", "L'DRY")
CAN_BE_DARK = ("WIR", "PAN", "BATH", "WC", "ENS", "L'DRY")
_BEDROOM = re.compile(r"BR (\d)$")


def room_full(abbr: str) -> str:
    """The full name of a room abbreviation (``BR 2`` is Bedroom 2)."""
    m = _BEDROOM.match(abbr)
    return f"Bedroom {m.group(1)}" if m else ROOMS[abbr]


def _is_room(text: str) -> bool:
    return text in ROOMS or _BEDROOM.match(text) is not None


def _plan_rooms(rng, wet: str) -> list[dict]:
    """Eight rooms in two rows of four: geometry only (``abbr`` and its box)."""
    bedrooms = [f"BR {i}" for i in range(1, rng.randint(2, 3) + 1)]
    names = ["KIT", "LIV", "BATH", "L'DRY", wet, *bedrooms]
    names += rng.sample([n for n in ("DIN", "WIR", "STU", "PAN", "GAR") if n not in names], 8 - len(names))
    rng.shuffle(names)
    rooms = []
    for row in (0, 1):
        x = 0
        for width in _split(rng, PLAN_W, 4, 4):
            rooms.append({"abbr": names[len(rooms)], "box": (x, row * WALL, x + width, row * WALL + WALL), "row": row})
            x += width
    return rooms


def _plan_figure(rooms, *, dark, toilets, fixtures, legend: bool, answer: str):
    polygons = [{"vertices": _rect(*r["box"]), "style": "solid"} for r in rooms]
    labels = []
    symbols = []
    for r in rooms:
        x0, y0, x1, y1 = r["box"]
        labels.append({"at": [(x0 + x1) / 2, (y0 + y1) / 2], "text": r["abbr"]})
        if r["abbr"] not in dark:
            symbols.append({"at": [(x0 + x1) / 2, PLAN_H if r["row"] else 0], "symbol": "window"})
        if r["abbr"] in toilets:
            symbols.append({"at": [x1 - 1, y0 + 1], "symbol": "toilets"})
        if r["abbr"] in fixtures:
            labels.append({"at": [x0 + 1, y1 - 1], "text": fixtures[r["abbr"]]})
    return _draw(
        answer,
        legend,
        alt="A floor plan drawn on a square grid.",
        grid={"type": "square", "width": PLAN_W, "height": PLAN_H},
        polygons=polygons,
        labels=labels,
        symbols=symbols,
        scale={"text": "1 : 100"},
        north=True,
    )


def _neighbours(room, rooms) -> list[dict]:
    x0, y0, x1, y1 = room["box"]
    out = []
    for other in rooms:
        a0, b0, a1, b1 = other["box"]
        if other is not room and a0 <= x1 and x0 <= a1 and b0 <= y1 and y0 <= b1:
            out.append(other)
    return out


def _room_distractors(answer_room, rooms, prefer=()) -> list[str]:
    order = [*prefer, *_neighbours(answer_room, rooms), *rooms]
    names = [room_full(r["abbr"]) for r in order if r is not answer_room]
    return _three(room_full(answer_room["abbr"]), names)


def _toilet_rooms(rng, rooms, wet: str) -> set[str]:
    toilets = {wet}
    if any(r["abbr"] == "BATH" for r in rooms) and rng.random() < 0.5:
        toilets.add("BATH")
    return toilets


def _plan_easy(rng):
    for _ in attempts(60, "an easy plan question"):
        wet = rng.choice(("ENS", "WC"))
        rooms = _plan_rooms(rng, wet)
        by = {r["abbr"]: r for r in rooms}
        if rng.random() < 0.5:
            target = by[wet]
            dark = set(rng.sample([a for a in CAN_BE_DARK if a in by and a != wet], rng.randint(0, 2)))
            toilets, text = {wet}, "Which room on this plan has the toilets symbol?"
            working = [f"The toilets symbol is in the room marked {wet}, which stands for {room_full(wet)}."]
            prefer = [by[a] for a in WET if a in by]
        else:
            lone = rng.choice([a for a in CAN_BE_DARK if a in by])
            target, dark = by[lone], {lone}
            toilets = _toilet_rooms(rng, rooms, wet)
            text = "Which room on this plan has no window symbol?"
            working = [f"Every room has a window symbol except {lone}, which stands for {room_full(lone)}."]
            prefer = [by[a] for a in CAN_BE_DARK if a in by]
        answer = room_full(target["abbr"])
        fig = _plan_figure(rooms, dark=dark, toilets=toilets, fixtures={}, legend=True, answer=answer)
        if fig is not None:
            break
    return _question(text, answer, _room_distractors(target, rooms, prefer), working, fig)


def _plan_medium(rng):
    for _ in attempts(60, "a medium plan question"):
        wet = rng.choice(("ENS", "WC"))
        rooms = _plan_rooms(rng, wet)
        by = {r["abbr"]: r for r in rooms}
        bedrooms = [a for a in by if _BEDROOM.match(a)]
        hws_homes = [a for a in ("L'DRY", "GAR", "BATH") if a in by]
        homes = {"DW": "KIT", "HWS": rng.choice(hws_homes), "BIR": rng.choice(bedrooms)}
        fixtures = {home: code for code, home in homes.items()}
        if len(fixtures) < 3:
            continue
        asked = rng.choice(sorted(FIXTURES))
        target = by[homes[asked]]
        dark = set(rng.sample([a for a in CAN_BE_DARK if a in by], rng.randint(0, 2)))
        answer = room_full(target["abbr"])
        fig = _plan_figure(
            rooms, dark=dark, toilets=_toilet_rooms(rng, rooms, wet), fixtures=fixtures, legend=True, answer=answer
        )
        if fig is not None:
            break
    text = f"In which room is the {FIXTURES[asked]}?"
    others = [by[home] for code, home in homes.items() if code != asked]
    working = [
        f"{asked} stands for {FIXTURES[asked]}. The label {asked} is in the room marked {target['abbr']}, "
        f"which is {answer}."
    ]
    return _question(text, answer, _room_distractors(target, rooms, others), working, fig)


def _plan_hard(rng):
    for _ in attempts(80, "a hard plan question"):
        wet = rng.choice(("ENS", "WC"))
        rooms = _plan_rooms(rng, wet)
        by = {r["abbr"]: r for r in rooms}
        wet_rooms = [a for a in WET if a in by]
        windowless = rng.sample(wet_rooms, rng.randint(1, len(wet_rooms) - 1))
        extra = [a for a in ("WIR", "PAN") if a in by and rng.random() < 0.5]
        dark = {*windowless, *extra}
        toilets = _toilet_rooms(rng, rooms, wet)
        fixtures = {rng.choice([a for a in ("L'DRY", "GAR", "BATH") if a in by]): "HWS"}
        bedrooms = [a for a in by if _BEDROOM.match(a)]
        fixtures.setdefault("KIT", "DW")
        if rng.random() < 0.5:
            wet_with_window = [a for a in wet_rooms if a not in dark]
            answer = _rooms(len(wet_with_window))
            text = "How many wet areas (bathroom, ensuite, toilet or laundry) on this plan have a window symbol?"
            windows = len([a for a in by if a not in dark])
            cands = [
                len(wet_rooms),
                len(wet_rooms) - len(wet_with_window),
                len([a for a in bedrooms if a not in dark]),
                windows,
                len(wet_with_window) + 1,
            ]
            working = [
                f"The wet areas are {', '.join(wet_rooms)}. Of these, {', '.join(wet_with_window)} "
                f"{'has' if len(wet_with_window) == 1 else 'have'} a window symbol: {answer}."
            ]
        else:
            hws_room = next(a for a, code in fixtures.items() if code == "HWS")
            union = toilets | {hws_room}
            answer = _rooms(len(union))
            text = (
                "A plumber must visit every room that has a toilets symbol or a hot water system. "
                "How many rooms is that?"
            )
            cands = [len(toilets), len(toilets) + 1, 1, len(union) + 1, len(union) - 1, len(union) + 2]
            working = [
                f"Rooms with a toilets symbol: {', '.join(sorted(toilets))}. The hot water system (HWS) is in "
                f"{hws_room}. Together that is {answer}, counting a shared room once."
            ]
        cands = [_rooms(c) for c in cands if c >= 0]
        fig = _plan_figure(rooms, dark=dark, toilets=toilets, fixtures=fixtures, legend=True, answer=answer)
        if fig is not None and len(pick_distractors(answer, cands)) >= 3:
            break
    return _question(text, answer, _three(answer, cands), working, fig)


_PLAN_TIERS = {"easy": _plan_easy, "medium": _plan_medium, "hard": _plan_hard}


@generator(
    "grid_plan_symbols",
    level=11,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def grid_plan_symbols(rng, tier="easy"):
    """A floor plan with abbreviations and symbols. See the module docstring for the three tiers."""
    return _tier(_PLAN_TIERS, tier)(rng)


def _plan_rooms_of(fig: dict) -> list[dict]:
    rooms = []
    for poly in fig["polygons"]:
        box = _box(poly["vertices"])
        names = [lb["text"] for lb in fig["labels"] if _is_room(lb["text"]) and _inside(box, lb["at"])]
        check(len(names) == 1, "a room needs exactly one name label")
        rooms.append({"abbr": names[0], "box": box})
    return rooms


def _rooms_with(fig: dict, rooms, symbol: str) -> list[str]:
    return [
        r["abbr"]
        for r in rooms
        if any(s["symbol"] == symbol and _inside(r["box"], s["at"]) for s in fig.get("symbols", []))
    ]


def _read_plan(question) -> str:
    """The answer from the plan's own rooms, symbols and labels, and the question's words."""
    fig, text = question.figure, question.text
    rooms = _plan_rooms_of(fig)
    windowed = _rooms_with(fig, rooms, "window")
    toilets = _rooms_with(fig, rooms, "toilets")
    if text == "Which room on this plan has the toilets symbol?":
        check(len(toilets) == 1, "the toilets symbol must be in exactly one room")
        return room_full(toilets[0])
    if text == "Which room on this plan has no window symbol?":
        dark = [r["abbr"] for r in rooms if r["abbr"] not in windowed]
        check(len(dark) == 1, "exactly one room must lack a window")
        return room_full(dark[0])
    if (m := re.fullmatch(r"In which room is the (.+)\?", text)) is not None:
        code = next(c for c, name in FIXTURES.items() if name == m.group(1))
        spots = [lb["at"] for lb in fig["labels"] if lb["text"] == code]
        check(len(spots) == 1, "the fixture label must appear once")
        homes = [r["abbr"] for r in rooms if _inside(r["box"], spots[0])]
        check(len(homes) == 1, "the fixture label must sit in one room")
        return room_full(homes[0])
    if text.startswith("How many wet areas"):
        return _rooms(len([r for r in rooms if r["abbr"] in WET and r["abbr"] in windowed]))
    if text.startswith("A plumber must visit"):
        spots = [lb["at"] for lb in fig["labels"] if lb["text"] == "HWS"]
        hws = {r["abbr"] for r in rooms if spots and _inside(r["box"], spots[0])}
        return _rooms(len(set(toilets) | hws))
    raise AssertionError(f"cannot read the question {text!r}")


figures.register_reader("grid_plan_symbols", _read_plan)


# ====================================================== skill 2: symbols and conventions for geometric information

SITE_W, SITE_H = 20, 12
FEATURES = {
    "HSE": "House",
    "GAR": "Garage",
    "SH": "Shed",
    "PERG": "Pergola",
    "DK": "Deck",
    "GH": "Greenhouse",
    "PL": "Pool",
    "CP": "Carport",
    "WT": "Water tank",
    "SEP": "Septic tank",
    "CEL": "Cellar",
}
_BUILT = ("HSE", "GAR", "SH", "CP")
_PLANNED = ("PERG", "DK", "GH", "PL")
_HIDDEN = ("WT", "SEP", "CEL")
COMPASS = {
    "north": (0, 1),
    "north-east": (1, 1),
    "east": (1, 0),
    "south-east": (1, -1),
    "south": (0, -1),
    "south-west": (-1, -1),
    "west": (-1, 0),
    "north-west": (-1, 1),
}
_BY_VECTOR = {v: k for k, v in COMPASS.items()}


def _sign(n: int) -> int:
    return (n > 0) - (n < 0)


def _apart(a, b) -> bool:
    """Two boxes with at least one square between them."""
    return a[2] < b[0] - 1 or b[2] < a[0] - 1 or a[3] < b[1] - 1 or b[3] < a[1] - 1


def _place(rng, sizes, width, height, fixed=()) -> list[tuple] | None:
    """Boxes ``(x0, y0, x1, y1)`` of the given sizes, inside the grid and a square apart from each other.

    ``None`` when a box finds no free place, so the caller draws a new set of sizes.
    """
    boxes = list(fixed)
    out = []
    for w, h in sizes:
        for _ in range(100):
            x, y = rng.randint(1, width - w - 1), rng.randint(1, height - h - 2)
            box = (x, y, x + w, y + h)
            if all(_apart(box, b) for b in boxes):
                break
        else:
            return None
        boxes.append(box)
        out.append(box)
    return out


def _site_figure(boxes, names, *, dashed=(), dim_labels=(), scale="1 : 200", answer: str):
    polygons = [{"vertices": _rect(*b), "style": "dashed" if i in dashed else "solid"} for i, b in enumerate(boxes)]
    labels = [{"at": [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], "text": n} for b, n in zip(boxes, names) if n is not None]
    labels += list(dim_labels)
    return _draw(
        answer,
        False,
        alt="A site plan drawn on a square grid.",
        grid={"type": "square", "width": SITE_W, "height": SITE_H},
        polygons=polygons,
        labels=labels,
        scale={"text": scale},
        north=True,
    )


def _conv_easy(rng):
    hidden = rng.random() < 0.4
    target = rng.choice(_HIDDEN if hidden else _PLANNED)
    pool = _BUILT if hidden else (*_BUILT, *[p for p in _PLANNED if p != target])
    names = [target, *rng.sample([p for p in pool], 3)]
    for _ in attempts(60, "an easy conventions question"):
        boxes = _place(rng, [(rng.randint(3, 5), rng.randint(3, 4)) for _ in names], SITE_W, SITE_H)
        if boxes is None:
            continue
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [names[i] for i in order]
        boxes_by = dict(zip(shuffled, boxes))
        answer = FEATURES[target]
        fig = _site_figure(
            [boxes_by[n] for n in shuffled],
            shuffled,
            dashed={shuffled.index(target)},
            answer=answer,
        )
        if fig is not None:
            break
    text = (
        "This plan shows what is hidden below the ground as well as what can be seen. Which feature is hidden "
        "underground?"
        if hidden
        else "This plan shows what has been built and what is only planned. Which feature is proposed, not yet built?"
    )
    kind = "hidden below the ground" if hidden else "proposed"
    working = [f"A dashed outline shows a feature that is {kind}. Only {target} is dashed: {answer}."]
    others = [FEATURES[n] for n in shuffled if n != target]
    return _question(text, answer, _three(answer, others), working, fig)


def _conv_medium(rng):
    for _ in attempts(200, "a medium conventions question"):
        direction = rng.choice(sorted(COMPASS))
        dx, dy = COMPASS[direction]
        gap = rng.randint(3, 6)
        ax, ay = rng.randint(1, SITE_W - 3), rng.randint(1, SITE_H - 4)
        bx, by = ax + dx * gap, ay + dy * gap
        if not (1 <= bx <= SITE_W - 3 and 1 <= by <= SITE_H - 4):
            continue
        a_box, b_box = (ax, ay, ax + 2, ay + 2), (bx, by, bx + 2, by + 2)
        if not _apart(a_box, b_box):
            continue
        names = rng.sample(sorted(FEATURES), 4)
        extra = _place(rng, [(2, 2), (2, 2)], SITE_W, SITE_H, fixed=[a_box, b_box])
        if extra is None:
            continue
        boxes = [a_box, b_box, *extra]
        fig = _site_figure(boxes, names, answer=direction)
        if fig is not None:
            break
    text = f"From the {FEATURES[names[0]].lower()}, in which direction is the {FEATURES[names[1]].lower()}?"
    mirror_ew = _BY_VECTOR[(-dx, dy)]
    mirror_ns = _BY_VECTOR[(dx, -dy)]
    reverse = _BY_VECTOR[(-dx, -dy)]
    turned = [_BY_VECTOR[(_sign(dx + dy), _sign(dy - dx))], _BY_VECTOR[(_sign(dx - dy), _sign(dx + dy))]]
    working = [
        "The north arrow points up the page, so up is north, down is south, right is east and left is west.",
        f"The {FEATURES[names[1]].lower()} is {direction} of the {FEATURES[names[0]].lower()}.",
    ]
    return _question(text, direction, _three(direction, [reverse, mirror_ew, mirror_ns, *turned]), working, fig)


def _conv_hard(rng):
    scale_m = rng.choice((2, 5))
    for _ in attempts(60, "a hard conventions question"):
        sizes = [(rng.randint(2, 5), rng.randint(2, 4)) for _ in range(2)]
        left_x, right_x = rng.randint(1, 3), rng.randint(11, 13)
        boxes = []
        for (w, h), x in zip(sizes, (left_x, right_x)):
            y = rng.randint(1, SITE_H - h - 3)
            boxes.append((x, y, x + w, y + h))
        dashed_index = rng.randrange(2)
        dims = []
        for x0, y0, x1, y1 in boxes:
            dims.append({"at": [(x0 + x1) / 2, y1 + 0.5], "text": f"{(x1 - x0) * scale_m} m"})
            dims.append({"at": [x1 + 1, (y0 + y1) / 2], "text": f"{(y1 - y0) * scale_m} m"})
        asks_proposed = rng.random() < 0.5
        pick = dashed_index if asks_proposed else 1 - dashed_index
        other = 1 - pick
        (w, h), (w2, h2) = [(s[0] * scale_m, s[1] * scale_m) for s in (sizes[pick], sizes[other])]
        answer = _area(w * h)
        fig = _site_figure(
            boxes,
            [None, None],
            dashed={dashed_index},
            dim_labels=dims,
            scale=f"1 : {100 * scale_m}",
            answer=answer,
        )
        cands = [_area(c) for c in (w2 * h2, 2 * (w + h), w * h2, sizes[pick][0] * sizes[pick][1])]
        if fig is not None and len(pick_distractors(answer, cands)) >= 3:
            break
    text = (
        "The dimensions on this plan are in metres. What area, in square metres, will the proposed addition cover?"
        if asks_proposed
        else "The dimensions on this plan are in metres. What is the area, in square metres, of the existing building?"
    )
    kind = "proposed addition is the dashed outline" if asks_proposed else "existing building is the solid outline"
    working = [f"The {kind}. Its sides are {w} m and {h} m, so its area is {w} x {h} = {w * h} square metres."]
    return _question(text, answer, _three(answer, cands), working, fig)


_CONV_TIERS = {"easy": _conv_easy, "medium": _conv_medium, "hard": _conv_hard}


@generator(
    "grid_plan_conventions",
    level=11,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def grid_plan_conventions(rng, tier="easy"):
    """A site plan read by its drawing conventions. See the module docstring for the three tiers."""
    return _tier(_CONV_TIERS, tier)(rng)


def _features_of(fig: dict) -> list[dict]:
    out = []
    for poly in fig["polygons"]:
        box = _box(poly["vertices"])
        named = [lb["text"] for lb in fig.get("labels", []) if lb["text"] in FEATURES and _inside(box, lb["at"])]
        out.append({"box": box, "name": named[0] if named else None, "style": poly.get("style", "solid")})
    return out


def _read_conventions(question) -> str:
    fig, text = question.figure, question.text
    feats = _features_of(fig)
    if text.endswith("Which feature is hidden underground?") or text.endswith(
        "Which feature is proposed, not yet built?"
    ):
        dashed = [f for f in feats if f["style"] == "dashed"]
        check(len(dashed) == 1, "exactly one feature is dashed")
        return FEATURES[dashed[0]["name"]]
    if (m := re.fullmatch(r"From the (.+), in which direction is the (.+)\?", text)) is not None:
        full = {v.lower(): k for k, v in FEATURES.items()}
        a = next(f for f in feats if f["name"] == full[m.group(1)])
        b = next(f for f in feats if f["name"] == full[m.group(2)])
        dx, dy = b["box"][0] - a["box"][0], b["box"][1] - a["box"][1]
        check(dx == 0 or dy == 0 or abs(dx) == abs(dy), "the two features are not on a compass line")
        return _BY_VECTOR[(_sign(dx), _sign(dy))]
    if "proposed addition" in text or "existing building" in text:
        want = "dashed" if "proposed addition" in text else "solid"
        (shape,) = [f for f in feats if f["style"] == want]
        x0, y0, x1, y1 = shape["box"]
        width = _dim(fig, [(x0 + x1) / 2, y1 + 0.5])
        height = _dim(fig, [x1 + 1, (y0 + y1) / 2])
        return _area(width * height)
    raise AssertionError(f"cannot read the question {text!r}")


def _dim(fig: dict, at) -> int:
    found = [lb["text"] for lb in fig["labels"] if list(lb["at"]) == list(at)]
    check(len(found) == 1 and found[0].endswith(" m"), "a dimension label is missing")
    return int(found[0][:-2])


figures.register_reader("grid_plan_conventions", _read_conventions)


# ========================================================= skill 3: map features and landmark representation

NAMEABLE = ("house", "school", "church", "hospital", "tree", "bridge", "airport", "lighthouse", "railway_station")
KEYED = (*NAMEABLE, "campground", "post_office", "lookout")


def _ref(col: int, row: int, columns=LETTERS) -> str:
    return f"{columns[col]}{row + 1}"


def _map_figure(rng, cells: dict, width: int, height: int, *, legend: bool, answer: str):
    symbols = [{"at": [c + 0.5, r + 0.5], "symbol": s} for (c, r), s in cells.items()]
    rng.shuffle(symbols)
    return _draw(
        answer,
        legend,
        alt="A map drawn on a square grid, with symbols and a north arrow.",
        grid={"type": "square", "width": width, "height": height},
        symbols=symbols,
        column_labels=list(LETTERS[:width]),
        row_labels=[str(i) for i in range(1, height + 1)],
        north=True,
    )


def _name(symbol: str) -> str:
    return grid.SYMBOLS[symbol].capitalize()


def _fill(rng, cells: dict, pool, width, height, extra: int, avoid=()):
    """Add up to ``extra`` more landmarks, each a symbol not yet used, on empty squares."""
    spare = [s for s in pool if s not in cells.values()]
    rng.shuffle(spare)
    for symbol in spare[:extra]:
        free = [(c, r) for c in range(width) for r in range(height) if (c, r) not in cells and (c, r) not in avoid]
        cells[rng.choice(free)] = symbol


def _map_easy(rng):
    for _ in attempts(100, "an easy map question"):
        width, height = rng.choice((6, 7)), rng.choice((5, 6))
        c, r = rng.randrange(width), rng.randrange(height)
        swapped, flipped = (r, c), (c, height - 1 - r)
        wrong = [swapped, flipped, (c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1)]
        wrong = [
            w
            for i, w in enumerate(wrong)
            if 0 <= w[0] < width and 0 <= w[1] < height and w != (c, r) and w not in wrong[:i]
        ]
        if len(wrong) < 3:
            continue
        chosen = wrong[:3]
        cells = dict(zip([(c, r), *chosen], rng.sample(NAMEABLE, 4)))
        _fill(rng, cells, NAMEABLE, width, height, rng.randint(1, 3))
        answer = _name(cells[(c, r)])
        fig = _map_figure(rng, cells, width, height, legend=False, answer=answer)
        if fig is not None:
            break
    ref = _ref(c, r)
    text = f"Which landmark is in square {ref}?"
    working = [
        f"Column {LETTERS[c]} is column {c + 1} from the left and row {r + 1} is row {r + 1} up from the bottom.",
        f"The symbol in square {ref} is the {answer.lower()}.",
    ]
    return _question(text, answer, [_name(cells[w]) for w in chosen], working, fig)


def _map_reference(rng):
    for _ in attempts(100, "a map reference question"):
        width, height = rng.choice((6, 7)), rng.choice((5, 6))
        cells: dict = {}
        _fill(rng, cells, KEYED, width, height, rng.randint(5, 8))
        (c, r), symbol = rng.choice(sorted(cells.items()))
        answer = _ref(c, r)
        fig = _map_figure(rng, cells, width, height, legend=True, answer=answer)
        cands = [
            _ref(c, height - 1 - r) if height - 1 - r != r else None,
            _ref(r, c) if r < width and c < height and r != c else None,
            _ref(c + 1, r) if c + 1 < width else None,
            _ref(c - 1, r) if c > 0 else None,
            _ref(c, r + 1) if r + 1 < height else None,
        ]
        cands = [x for x in cands if x]
        if fig is not None and len(pick_distractors(answer, cands)) >= 3:
            break
    name = grid.SYMBOLS[symbol]
    text = f"What is the grid reference of the {name}? Give the column letter first, then the row number."
    working = [
        f"The {name} is in column {LETTERS[c]} (column {c + 1} from the left) and row {r + 1} up from the bottom.",
        f"So its grid reference is {answer}.",
    ]
    return _question(text, answer, _three(answer, cands), working, fig)


def _map_direction(rng):
    for _ in attempts(100, "a map direction question"):
        width, height = rng.choice((6, 7)), rng.choice((5, 6))
        c, r = rng.randint(1, width - 2), rng.randint(1, height - 2)
        rays = {
            "north": (c, rng.randint(r + 1, height - 1)),
            "south": (c, rng.randint(0, r - 1)),
            "east": (rng.randint(c + 1, width - 1), r),
            "west": (rng.randint(0, c - 1), r),
        }
        names = rng.sample(NAMEABLE, 5)
        cells = {(c, r): names[0], **{cell: names[i + 1] for i, cell in enumerate(rays.values())}}
        _fill(
            rng,
            cells,
            NAMEABLE,
            width,
            height,
            rng.randint(0, 2),
            avoid={(x, y) for x in range(width) for y in range(height) if x == c or y == r},
        )
        direction = rng.choice(sorted(rays))
        answer = _name(cells[rays[direction]])
        fig = _map_figure(rng, cells, width, height, legend=False, answer=answer)
        if fig is not None:
            break
    along = "column" if direction in ("north", "south") else "row"
    text = (
        f"Look {direction}, along the same {along}, from the landmark in square {_ref(c, r)}. Which landmark is there?"
    )
    working = [
        f"North is up the page. Squares to the {direction} of {_ref(c, r)} in the same {along} hold the "
        f"{answer.lower()}."
    ]
    others = [_name(cells[v]) for k, v in rays.items() if k != direction]
    opposite = {"north": "south", "south": "north", "east": "west", "west": "east"}[direction]
    others.sort(key=lambda n: n != _name(cells[rays[opposite]]))
    return _question(text, answer, others, working, fig)


def _map_medium(rng):
    return _map_reference(rng) if rng.random() < 0.5 else _map_direction(rng)


def _map_hard(rng):
    for _ in attempts(200, "a hard map question"):
        width, height = rng.choice((6, 7)), rng.choice((5, 6))
        c, r = rng.randrange(width), rng.randrange(height)
        a, b = rng.randint(1, 4), rng.randint(1, 4)
        sx, sy = rng.choice((1, -1)), rng.choice((1, -1))
        end = (c + sx * a, r + sy * b)

        def ok(cell):
            return 0 <= cell[0] < width and 0 <= cell[1] < height

        if not ok(end):
            continue
        cands = [
            (c - sx * a, r + sy * b),  # east and west confused
            (c + sx * a, r - sy * b),  # north and south confused
            (c + sx * b, r + sy * a),  # the two distances swapped
            (c + sx * (a - 1), r + sy * (b - 1)),  # counts the starting square as the first step
            (c - sx * a, r - sy * b),  # both directions reversed
        ]
        cands = [x for i, x in enumerate(cands) if ok(x) and x != end and x not in cands[:i]]
        if len(cands) < 3:
            continue
        cells: dict = {}
        _fill(rng, cells, KEYED, width, height, rng.randint(4, 7), avoid={end})
        if (c, r) not in cells:
            cells[(c, r)] = rng.choice([s for s in KEYED if s not in cells.values()])
        answer = _ref(*end)
        fig = _map_figure(rng, cells, width, height, legend=True, answer=answer)
        if fig is not None:
            break
    east_west = "east" if sx > 0 else "west"
    north_south = "north" if sy > 0 else "south"
    legs = [f"{_squares(a)} {east_west}", f"{_squares(b)} {north_south}"]
    if rng.random() < 0.5:
        legs.reverse()
    name = grid.SYMBOLS[cells[(c, r)]]
    text = f"Start at the {name}. Go {legs[0]}, then {legs[1]}. Which square do you finish in?"
    working = [
        f"The {name} is in square {_ref(c, r)}.",
        f"Going {legs[0]} and {legs[1]} ends in column {LETTERS[end[0]]}, row {end[1] + 1}: {answer}.",
    ]
    return _question(text, answer, _three(answer, [_ref(*x) for x in cands]), working, fig)


_MAP_TIERS = {"easy": _map_easy, "medium": _map_medium, "hard": _map_hard}


@generator(
    "grid_map_landmarks",
    level=3,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def grid_map_landmarks(rng, tier="easy"):
    """Landmarks on a map with grid references. See the module docstring for the three tiers."""
    return _tier(_MAP_TIERS, tier)(rng)


def _symbol_cells(fig: dict) -> dict:
    return {(int(s["at"][0]), int(s["at"][1])): s["symbol"] for s in fig["symbols"]}


def _cell_of(fig: dict, ref: str) -> tuple[int, int]:
    m = re.fullmatch(r"([A-Z]+)(\d+)", ref)
    check(m is not None, f"{ref!r} is not a grid reference")
    return fig["column_labels"].index(m.group(1)), fig["row_labels"].index(m.group(2))


def _ref_of(fig: dict, cell) -> str:
    return fig["column_labels"][cell[0]] + fig["row_labels"][cell[1]]


def _symbol_named(fig: dict, name: str) -> tuple[int, int]:
    cells = _symbol_cells(fig)
    found = [cell for cell, s in cells.items() if grid.SYMBOLS[s] == name]
    check(len(found) == 1, f"the {name} must be on the map once")
    return found[0]


def _read_map(question) -> str:
    fig, text = question.figure, question.text
    cells = _symbol_cells(fig)
    if (m := re.fullmatch(r"Which landmark is in square (\w+)\?", text)) is not None:
        return grid.SYMBOLS[cells[_cell_of(fig, m.group(1))]].capitalize()
    if (m := re.fullmatch(r"What is the grid reference of the (.+?)\? .*", text)) is not None:
        return _ref_of(fig, _symbol_named(fig, m.group(1)))
    if (
        m := re.fullmatch(r"Look (\w+), along the same (?:row|column), from the landmark in square (\w+)\. .*", text)
    ) is not None:
        c, r = _cell_of(fig, m.group(2))
        dx, dy = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}[m.group(1)]
        ahead = [
            s
            for (x, y), s in cells.items()
            if (x, y) != (c, r) and ((x - c) * dy == 0 and (y - r) * dx == 0) and (x - c) * dx + (y - r) * dy > 0
        ]
        check(len(ahead) == 1, "exactly one landmark lies that way")
        return grid.SYMBOLS[ahead[0]].capitalize()
    if (
        m := re.fullmatch(r"Start at the (.+?)\. Go (\d+) squares? (\w+), then (\d+) squares? (\w+)\. .*", text)
    ) is not None:
        x, y = _symbol_named(fig, m.group(1))
        for count, way in ((m.group(2), m.group(3)), (m.group(4), m.group(5))):
            dx, dy = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}[way]
            x, y = x + dx * int(count), y + dy * int(count)
        return _ref_of(fig, (x, y))
    raise AssertionError(f"cannot read the question {text!r}")


figures.register_reader("grid_map_landmarks", _read_map)
