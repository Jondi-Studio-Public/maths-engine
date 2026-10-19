"""Scale drawings on a square grid: floor plans, garden beds and maps drawn with ``grid_drawing``.

Four topics, one per skill, every answer a ``number`` and every question drawn on a square grid where one cell is
1 cm on the page. The scale is written on the figure (``1 cm : 2 m``, or the ratio ``1 : 200``), so the figure
carries the scale and the shape; the text says what to find and which unit to answer in.

- ``grid_scale_actual`` (``senior-find-actual-measurements-from-scale-drawings-such-as``). Easy: one actual length
  of a rectangle. Medium: the actual perimeter of an L- or U-shaped room. Hard: the actual area, in square metres,
  where the area scales by the square of the factor.
- ``grid_scale_map`` (``senior-obtain-measurements-from-scale-drawings-such-as-maps``). A map with a road, a river
  and symbols. Easy: the distance along one straight road. Medium: the distance along a road of three legs that
  doubles back, so the straight-across distance is not the answer. Hard: the walking time along a road of several
  legs at a stated speed.
- ``grid_scale_materials`` (``senior-estimate-and-compare-quantities-materials-and-costs-using``). Easy: whole
  packs of flooring for a rectangle. Medium: whole turf rolls for an L- or U-shaped lawn, or whole fence panels for
  its boundary. Hard: the saving between two options sold in different packs, rounding each up to whole packs.
- ``grid_scale_conventions`` (``senior-understand-and-apply-drawing-conventions-of-scale-drawings``). Easy: convert
  between ``1 cm : 2 m`` and ``1 : 200``. Medium: find the scale from one labelled wall. Hard: the drawn length
  of a stated real length, with the scale found from the labelled wall.

Every figure reader re-derives the answer from the figure (polygon vertices, road lines, labels, scale text) and
the sentence alone.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from .. import figures
from ..core import VerificationError, attempts, check, exact_decimal_string, generator, phrase, pick_distractors
from ..figure_kinds import grid
from . import _space_kit as kit

# ------------------------------------------------------------------------------------------- shared helpers


def _fr(x) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(x)


def _hour_word(hours: Fraction) -> str:
    return "hour" if hours == 1 else "hours"


def _hours_text(hours: Fraction) -> str:
    """A number of hours as a terminating decimal where it is one, otherwise as a fraction."""
    try:
        return exact_decimal_string(hours, 4)
    except ValueError:
        return f"{hours.numerator}/{hours.denominator}"


def _text(x) -> str:
    """A number as a learner types it: whole, or one decimal place."""
    x = _fr(x)
    if x.denominator == 1:
        return str(int(x))
    check((x * 10).denominator == 1, f"{x} needs more than one decimal place")
    return f"{float(x):.1f}"


def _value(x):
    x = _fr(x)
    return int(x) if x.denominator == 1 else float(x)


def _finish(rng, build):
    """Run ``build(rng)`` until it yields a figure that does not leak and three wrong answers a learner could give.

    ``build`` returns ``(fig, text, answer, candidates, working)``; ``candidates`` are numbers in misconception
    order. Every candidate must differ from the answer by at least one whole unit, so each clears the grader.
    """
    for _ in attempts(60, "a scale-drawing question with three distinct wrong answers"):
        try:
            fig, text, answer, candidates, working = build(rng)
        except VerificationError:
            continue  # this draw had no usable pack size or route: draw again
        answer = _fr(answer)
        if answer <= 0 or kit.leaks(fig, _text(answer)):
            continue
        pool = []
        for c in candidates:
            c = Fraction(c).limit_denominator(100) if isinstance(c, float) else _fr(c)
            if (
                c > 0
                and abs(c - answer) >= 1
                and (c * 10).denominator == 1
                and (c.denominator == 1) == (answer.denominator == 1)
            ):
                pool.append(_text(c))
        distractors = pick_distractors(_text(answer), pool, tolerance=Fraction(1, 2))
        if len(distractors) == 3:
            break
    return dict(
        text=text,
        answer=_value(answer),
        answer_text=_text(answer),
        answer_latex=_text(answer),
        distractors=distractors,
        working=working,
        figure=fig,
    )


def _a(phrase: str) -> str:
    """The article for a phrase: ``an`` before a vowel sound (``L-shaped``, ``office``), else ``a``."""
    return "an" if phrase[0] in "aeioL" else "a"


def _tier(tier, tiers):
    if tier not in tiers:
        raise ValueError(f"unknown tier {tier!r}")
    return tiers[tier]


# ------------------------------------------------------------------------------------------- shapes (cells)


def _rect(w, h):
    return [(0, 0), (w, 0), (w, h), (0, h)]


def _l_shape(rng, w, h):
    h1 = rng.randint(1, h - 1)
    w2 = rng.randint(1, w - 1)
    return [(0, 0), (w, 0), (w, h1), (w2, h1), (w2, h), (0, h)]


def _u_shape(rng, w, h):
    a = rng.randint(1, (w - 1) // 2)
    d = rng.randint(1, h - 1)
    return [(0, 0), (w, 0), (w, h), (w - a, h), (w - a, h - d), (a, h - d), (a, h), (0, h)]


def _area(poly) -> int:
    s = 0
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        s += x1 * y2 - x2 * y1
    return abs(s) // 2


def _edges(poly):
    return list(zip(poly, poly[1:] + poly[:1]))


def _perimeter(poly) -> int:
    return sum(abs(x2 - x1) + abs(y2 - y1) for (x1, y1), (x2, y2) in _edges(poly))


def _box(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return max(xs) - min(xs), max(ys) - min(ys)


def _plan(poly, name, scale_text, *, alt, bar=False, labels=(), margin=1):
    """The plan: ``poly`` (cells, from the origin) shifted onto the grid, with its scale and any wall labels."""
    w, h = _box(poly)
    moved = [[x + margin, y + margin] for x, y in poly]
    scale = {"text": scale_text}
    if bar:
        scale["bar_cells"] = min(2, w + 2 * margin)
    return grid.grid_drawing(
        alt=alt,
        grid={"type": "square", "width": w + 2 * margin, "height": h + 2 * margin},
        polygons=[{"vertices": moved, "style": "solid", "label": name}],
        labels=[{"at": [x + margin, y + margin], "text": t} for (x, y), t in labels],
        scale=scale,
    )


def _drawn_polygon(fig):
    """The polygon's vertices in cells, from the figure."""
    check(len(fig["polygons"]) == 1, "expected one polygon")
    return [tuple(v) for v in fig["polygons"][0]["vertices"]]


_ROOMS = ("hall", "classroom", "workshop", "studio", "library", "gym")
_LAWNS = ("lawn", "garden bed", "courtyard", "play area", "vegetable patch")

# ------------------------------------------------------------------------------------------- scales

_SCALE_METRIC = re.compile(r"^1 cm : (\d+(?:\.\d+)?) (m|km)$")
_SCALE_RATIO = re.compile(r"^1 : (\d+)$")


def _scale_text(k, ratio: bool) -> str:
    """``k`` metres to the centimetre, written as ``1 cm : k m`` or as the ratio ``1 : 100k``."""
    return f"1 : {int(100 * _fr(k))}" if ratio else f"1 cm : {_text(k)} m"


def _metres_per_cm(fig) -> Fraction:
    """The scale written on a figure, as metres to the centimetre."""
    text = fig["scale"]["text"]
    if (m := _SCALE_METRIC.match(text)) is not None:
        return Fraction(m.group(1)) * (1000 if m.group(2) == "km" else 1)
    m = _SCALE_RATIO.match(text)
    check(m is not None, f"cannot read the scale {text!r}")
    return Fraction(int(m.group(1)), 100)


# ------------------------------------------------------------------------------------------- 1. actual measurements


def _actual_easy(rng):
    k = rng.choice((1, 2, 3, 4, 5, 10))
    w = rng.randint(4, 13)
    h = rng.randint(2, min(8, w - 1))
    longer = rng.random() < 0.5
    cells, other = (w, h) if longer else (h, w)
    name = rng.choice(_ROOMS)
    ratio = rng.random() < 0.3
    fig = _plan(
        _rect(w, h),
        name.capitalize(),
        _scale_text(k, ratio),
        alt=f"A scale drawing of a {name} on a square grid, with its scale.",
        bar=rng.random() < 0.4,
    )
    side = "longer" if longer else "shorter"
    text = (
        f"The plan of a {name} is drawn on a square grid. Each square is 1 cm by 1 cm on the plan. "
        f"Find the actual length of the {side} side of the {name}, in metres."
    )
    answer = cells * k
    cands = [cells, other * k, cells * k * 100, (cells + 1) * k, cells + k, other + k]
    working = [
        f"Count the squares along the {side} side: {cells} cm on the plan.",
        f"Each 1 cm on the plan is {_text(k)} m, so the actual length is {cells} × {_text(k)} = {answer} m.",
    ]
    return fig, text, answer, cands, working


def _actual_medium(rng):
    k = rng.choice((1, 2, 3, 4, 5))
    w = rng.randint(5, 12)
    h = rng.randint(4, 9)
    u = rng.random() < 0.3
    poly = _u_shape(rng, w, h) if u else _l_shape(rng, w, h)
    kind = "U-shaped" if u else "L-shaped"
    name = rng.choice(("courtyard", "room", "hall") if u else ("room", "hall", "studio", "workshop"))
    fig = _plan(
        poly,
        name.capitalize(),
        _scale_text(k, rng.random() < 0.3),
        alt=f"A scale drawing of a {kind} {name} on a square grid, with its scale.",
        bar=rng.random() < 0.4,
    )
    p = _perimeter(poly)
    text = (
        f"The plan of {_a(kind)} {kind} {name} is drawn on a square grid. Each square is 1 cm by 1 cm on the plan. "
        f"Find the actual perimeter of the {name}, in metres."
    )
    longest = max(abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in _edges(poly))
    cands = [
        p,  # the drawn perimeter, never scaled
        (p - longest) * k,  # leaves out the longest side
        (w + h) * k,  # half the perimeter
        _area(poly) * k,  # an area where a perimeter was wanted
        2 * (w + h) * k,  # the bounding rectangle (differs from the answer only for the U)
        p * k * 100,
    ]
    working = [
        f"Add the lengths of all {len(poly)} sides on the plan: {p} cm.",
        f"Each 1 cm on the plan is {_text(k)} m, so the actual perimeter is {p} × {_text(k)} = {p * k} m.",
    ]
    return fig, text, p * k, cands, working


def _actual_hard(rng):
    k = rng.choice((2, 3, 4, 5))
    w = rng.randint(4, 12)
    h = rng.randint(3, 9)
    shape = rng.choice(("rect", "l", "l", "u"))
    poly = {"rect": lambda: _rect(w, h), "l": lambda: _l_shape(rng, w, h), "u": lambda: _u_shape(rng, max(w, 5), h)}[
        shape
    ]()
    nice = {"rect": "", "l": "L-shaped ", "u": "U-shaped "}[shape]
    name = rng.choice(_LAWNS + _ROOMS)
    ratio = rng.random() < 0.5
    fig = _plan(
        poly,
        name.capitalize(),
        _scale_text(k, ratio),
        alt=f"A scale drawing of a {nice}{name} on a square grid, with its scale.",
    )
    cells = _area(poly)
    article = _a(nice or name)
    text = (
        f"The plan of {article} {nice}{name} is drawn on a square grid. Each square is 1 cm by 1 cm on the plan. "
        f"Find the actual area of the {name}, in square metres."
    )
    answer = cells * k * k
    cands = [
        cells * k,  # multiplies the area by the linear factor
        cells,  # the drawn area, never scaled
        cells * k * k * 100,
        _perimeter(poly) * k * k,
        cells * 100 * k,
        (cells * k) * (k + 1),
    ]
    working = [
        f"Find the area on the plan: {cells} square centimetres.",
        f"1 cm on the plan is {_text(k)} m, so 1 square centimetre is {_text(k)} × {_text(k)} = {k * k} square metres.",
        f"The actual area is {cells} × {k * k} = {answer} square metres.",
    ]
    return fig, text, answer, cands, working


_ACTUAL = {"easy": _actual_easy, "medium": _actual_medium, "hard": _actual_hard}


@generator(
    "grid_scale_actual",
    level=11,
    presets={t: dict(tier=t) for t in _ACTUAL},
)
def grid_scale_actual(rng, tier="easy"):
    """Actual length, perimeter or area from a scale plan on a square grid. See the module docstring."""
    return _finish(rng, _tier(tier, _ACTUAL))


_ASKS = re.compile(r"actual (length|perimeter|area) of the")


def _read_actual(question) -> str:
    fig, text = question.figure, question.text
    poly = _drawn_polygon(fig)
    k = _metres_per_cm(fig)
    ask = _ASKS.search(text)
    check(ask is not None, "cannot read what is asked")
    if ask.group(1) == "length":
        w, h = _box(poly)
        value = (max(w, h) if "longer side" in text else min(w, h)) * k
    elif ask.group(1) == "perimeter":
        value = _perimeter(poly) * k
    else:
        value = _area(poly) * k * k
    return _text(value)


figures.register_reader("grid_scale_actual", _read_actual)

# ------------------------------------------------------------------------------------------- 2. maps

_MAP_SYMBOLS = (
    "house",
    "school",
    "church",
    "hospital",
    "post_office",
    "railway_station",
    "lighthouse",
    "campground",
    "lookout",
    "picnic_area",
)
_MAP_SCALES = (("1 cm : 1 km", Fraction(1)), ("1 cm : 2 km", Fraction(2)), ("1 cm : 5 km", Fraction(5)))
_MAP_SCALES_HALF = _MAP_SCALES + (("1 cm : 500 m", Fraction(1, 2)),)
_MAP_WALK = (("1 cm : 250 m", Fraction(1, 4)), _MAP_SCALES_HALF[3])
_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _route(rng, gw, gh, legs, lo, hi):
    """A road of ``legs`` axis-aligned legs on the grid that never crosses itself, as a list of corner points."""
    for _ in attempts(400, "a road that stays on the map"):
        x, y = rng.randint(1, gw - 1), rng.randint(1, gh - 1)
        horizontal = rng.random() < 0.5
        pts = [(x, y)]
        seen = {(x, y)}
        ok = True
        for _leg in range(legs):
            n = rng.randint(lo, hi)
            dx, dy = (rng.choice((-1, 1)), 0) if horizontal else (0, rng.choice((-1, 1)))
            for _i in range(n):
                x, y = x + dx, y + dy
                if not (1 <= x <= gw - 1 and 1 <= y <= gh - 1) or (x, y) in seen:
                    ok = False
                    break
                seen.add((x, y))
            if not ok:
                break
            pts.append((x, y))
            horizontal = not horizontal
        if ok:
            return pts, seen
    raise AssertionError("unreachable")


def _length(pts) -> int:
    return sum(abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in zip(pts, pts[1:]))


def _seg_distance(p, a, b) -> float:
    """The distance from point ``p`` to the segment ``a``-``b``."""
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _river(rng, gw, gh, pts, spots):
    """A slanting river that crosses the road at most briefly, never runs beside it and misses every symbol."""
    road = list(zip(pts, pts[1:]))
    for _ in attempts(80, "a river clear of the road and the symbols"):
        ys = [rng.randint(1, gh - 1) for _ in range(3)]
        xs = [0, gw // 2, gw]
        if ys[0] == ys[1] or ys[1] == ys[2]:
            continue
        river = list(zip(xs, ys))
        near = 0
        clear = True
        for (x1, y1), (x2, y2) in zip(river, river[1:]):
            steps = 4 * max(x2 - x1, abs(y2 - y1))
            for i in range(steps + 1):
                q = (x1 + (x2 - x1) * i / steps, y1 + (y2 - y1) * i / steps)
                if min(_seg_distance(q, a, b) for a, b in road) < 1:
                    near += 1
                if min(math.hypot(q[0] - sx, q[1] - sy) for sx, sy in spots) < 1.2:
                    clear = False
        if clear and near <= 12:  # at most about three cells of the river lie within a cell of the road
            return [list(p) for p in river]
    raise VerificationError("no river clear of the road")


def _map_figure(rng, gw, gh, pts, seen, scale_text, names):
    a, b = names
    ends = [pts[0], pts[-1]]
    check(max(abs(ends[0][0] - ends[1][0]), abs(ends[0][1] - ends[1][1])) >= 2, "the two places touch")
    extra = [s for s in _MAP_SYMBOLS if s not in names]
    rng.shuffle(extra)
    spots = list(ends)
    for _ in range(2):  # every symbol at least 2 cells from every other
        free = [
            (x, y)
            for x in range(1, gw)
            for y in range(1, gh)
            if (x, y) not in seen and all(max(abs(x - sx), abs(y - sy)) >= 2 for sx, sy in spots)
        ]
        check(bool(free), "no room for a symbol")
        spots.append(rng.choice(free))
    symbols = [{"at": list(p), "symbol": s} for p, s in zip(spots, [a, b] + extra)]
    river = _river(rng, gw, gh, pts, spots)
    return grid.grid_drawing(
        alt="A map on a square grid with a road, a river and some places marked by symbols, with its scale.",
        grid={"type": "square", "width": gw, "height": gh},
        lines=[{"points": [list(p) for p in pts], "style": "road"}, {"points": river, "style": "river"}],
        symbols=symbols,
        legend=True,
        scale={"text": scale_text, "bar_cells": 2},
        north=True,
    )


def _name(sym: str) -> str:
    return grid.SYMBOLS[sym]


def _map_common(rng, legs, lo, hi, scales, *, detour):
    for _ in attempts(100, "a map route"):
        gw, gh = rng.randint(14, 18), rng.randint(10, 13)
        pts, seen = _route(rng, gw, gh, legs, lo, hi)
        manhattan = abs(pts[-1][0] - pts[0][0]) + abs(pts[-1][1] - pts[0][1])
        if detour and manhattan == _length(pts):
            continue
        break
    scale_text, k = rng.choice(scales)
    names = rng.sample(_MAP_SYMBOLS, 2)
    fig = _map_figure(rng, gw, gh, pts, seen, scale_text, names)
    return fig, pts, k, scale_text, names, manhattan


def _map_easy(rng):
    fig, pts, k, scale_text, names, _m = _map_common(rng, 1, 3, 10, _MAP_SCALES, detour=False)
    cells = _length(pts)
    text = (
        f"The map is drawn on a square grid. Each square is 1 cm by 1 cm on the map. How far is it along the road "
        f"from the {_name(names[0])} to the {_name(names[1])}, in kilometres?"
    )
    answer = cells * k
    cands = [cells, cells + k, (cells + 1) * k, cells * k * 1000, cells * k * 10]
    working = [
        f"Count the squares along the road: {cells} cm on the map.",
        f"{scale_text.replace('1 cm : ', '1 cm on the map is ')}, so the distance is {cells} × {_text(k)} = {_text(answer)} km.",
    ]
    return fig, text, answer, cands, working


def _map_medium(rng):
    fig, pts, k, scale_text, names, manhattan = _map_common(rng, 3, 2, 8, _MAP_SCALES_HALF, detour=True)
    legs = [abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in zip(pts, pts[1:])]
    cells = sum(legs)
    text = (
        f"The map is drawn on a square grid. Each square is 1 cm by 1 cm on the map. Find the distance along the "
        f"road from the {_name(names[0])} to the {_name(names[1])}, in kilometres."
    )
    answer = cells * k
    cands = [
        manhattan * k,  # goes straight across and up, ignoring the doubling back
        (cells - legs[-1]) * k,  # leaves out the last leg
        (cells - legs[0]) * k,  # leaves out the first leg
        cells,  # never scales
        (cells - min(legs)) * k,
        cells * k * 1000,
    ]
    working = [
        f"Add the legs of the road on the map: {' + '.join(str(v) for v in legs)} = {cells} cm.",
        f"Each 1 cm on the map is {_text(k)} km, so the distance is {cells} × {_text(k)} = {_text(answer)} km.",
    ]
    return fig, text, answer, cands, working


def _map_hard(rng):
    for _ in attempts(100, "a route with a whole number of minutes"):
        fig, pts, k, scale_text, names, manhattan = _map_common(rng, rng.choice((4, 5)), 2, 4, _MAP_WALK, detour=True)
        speed = rng.choice((3, 4, 5, 6))
        cells = _length(pts)
        dist = cells * k
        minutes = dist * 60 / speed
        if minutes.denominator == 1 and 1 <= dist <= 6 and (dist * 10).denominator == 1:
            break
    legs = [abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in zip(pts, pts[1:])]
    text = (
        f"The map is drawn on a square grid. Each square is 1 cm by 1 cm on the map. Sam walks along the road from "
        f"the {_name(names[0])} to the {_name(names[1])} at {speed} km/h. How many minutes does the walk take?"
    )
    cands = [
        dist / speed * 100,  # decimal hours read as minutes
        manhattan * k * 60 / speed,  # a straight-across route
        dist / speed,  # hours, not minutes
        dist * speed,
        dist,
        (cells - legs[-1]) * k * 60 / speed,
        cells * 60 / speed,  # never scales
    ]
    working = [
        f"Add the legs of the road on the map: {' + '.join(str(v) for v in legs)} = {cells} cm.",
        f"Each 1 cm on the map is {float(k):g} km, so the road is {cells} × {float(k):g} = {_text(dist)} km long.",
        f"Time = distance ÷ speed = {_text(dist)} ÷ {speed} = {_hours_text(_fr(dist) / speed)} {_hour_word(_fr(dist) / speed)}.",
        f"{_hours_text(_fr(dist) / speed)} {_hour_word(_fr(dist) / speed)} × 60 = {_text(minutes)} minutes.",
    ]
    return fig, text, minutes, cands, working


_MAP = {"easy": _map_easy, "medium": _map_medium, "hard": _map_hard}


@generator("grid_scale_map", level=11, presets={t: dict(tier=t) for t in _MAP})
def grid_scale_map(rng, tier="easy"):
    """Distance or walking time along a road on a scale map. See the module docstring."""
    return _finish(rng, _tier(tier, _MAP))


_SPEED = re.compile(r"at (\d+(?:\.\d+)?) km/h")
_FROM_TO = re.compile(r"from the (.+?) to the (.+?)(?:,| at |\.|$)")


def _read_map(question) -> str:
    fig, text = question.figure, question.text
    by_name = {v: k for k, v in grid.SYMBOLS.items()}
    m = _FROM_TO.search(text)
    check(m is not None, "cannot read the places")
    where = {}
    for sym in fig["symbols"]:
        where.setdefault(sym["symbol"], tuple(sym["at"]))
    start, end = where[by_name[m.group(1)]], where[by_name[m.group(2)]]
    roads = [
        [tuple(p) for p in ln["points"]]
        for ln in fig["lines"]
        if ln["style"] == "road" and {tuple(ln["points"][0]), tuple(ln["points"][-1])} == {start, end}
    ]
    check(len(roads) == 1, "no single road joins the two places")
    km = _length(roads[0]) * _metres_per_cm(fig) / 1000
    if (s := _SPEED.search(text)) is not None:
        return _text(km * 60 / Fraction(s.group(1)))
    return _text(km)


figures.register_reader("grid_scale_map", _read_map)

# ------------------------------------------------------------------------------------------- 3. materials


def _packs(area, cover) -> int:
    return math.ceil(Fraction(area) / cover)


def _materials_easy(rng):
    k = rng.choice((1, 2))
    w = rng.randint(3, 8)
    h = rng.randint(3, 6)
    area = w * h * k * k
    for _ in attempts(100, "a pack size that does not divide the floor"):
        cover = rng.choice((2, 3, 4, 5, 6, 7, 8, 9, 10))
        if area % cover and area / cover >= 2:
            break
    name = rng.choice(("lounge", "bedroom", "study", "hallway", "studio"))
    fig = _plan(
        _rect(w, h),
        name.capitalize(),
        _scale_text(k, rng.random() < 0.3),
        alt=f"A scale drawing of a {name} floor on a square grid, with its scale.",
    )
    text = (
        f"The plan shows the floor of a {name}. Each square is 1 cm by 1 cm on the plan. Laminate flooring is sold "
        f"in packs that each cover {cover} square metres. How many whole packs must be bought to cover the whole "
        f"floor?"
    )
    answer = _packs(area, cover)
    cands = [
        area // cover,  # rounds down
        round(area / cover),  # rounds to the nearest
        _packs(w * h, cover),  # uses the area on the plan
        _packs(w * h * k, cover),  # uses the linear factor
        answer + 1,
        _packs(2 * (w + h) * k, cover),  # uses the perimeter
    ]
    working = [
        f"Area on the plan: {w} × {h} = {w * h} square centimetres, and 1 cm is {_text(k)} m.",
        f"Actual area: {w * h} × {k * k} = {area} square metres.",
        f"{area} / {cover} = {round(area / cover, 2)}, and part of a pack cannot be bought, so round up to {answer} packs.",
    ]
    return fig, text, answer, cands, working


def _materials_medium(rng):
    k = rng.choice((1, 2, 3))
    w = rng.randint(5, 11)
    h = rng.randint(4, 8)
    u = rng.random() < 0.3
    poly = _u_shape(rng, w, h) if u else _l_shape(rng, w, h)
    kind = "U-shaped" if u else "L-shaped"
    name = rng.choice(("lawn", "garden bed", "play area", "paved yard"))
    fig = _plan(
        poly,
        name.capitalize(),
        _scale_text(k, rng.random() < 0.3),
        alt=f"A scale drawing of a {kind} {name} on a square grid, with its scale.",
    )
    cells = _area(poly)
    area = cells * k * k
    perim = _perimeter(poly) * k
    if rng.random() < 0.5:
        for _ in attempts(100, "a roll size that does not divide the lawn"):
            cover = rng.choice((2, 3, 4, 5, 6, 7, 8, 10))
            if area % cover and area / cover >= 4:
                break
        text = (
            f"The plan shows {_a(kind)} {kind} {name} to be covered with turf. Each square is 1 cm by 1 cm on the plan. "
            f"Turf is sold in rolls that each cover {cover} square metres. How many whole rolls must be bought?"
        )
        answer = _packs(area, cover)
        cands = [
            area // cover,
            round(area / cover),
            _packs(cells, cover),
            _packs(cells * k, cover),
            _packs(perim, cover),
            answer + 1,
        ]
        working = [
            f"Count the squares on the plan: {cells} square centimetres. Each is {k * k} square metres, so the "
            f"actual area is {area} square metres.",
            f"{area} / {cover} = {round(area / cover, 2)}, so round up to {answer} rolls.",
        ]
    else:
        for _ in attempts(100, "a panel length that does not divide the boundary"):
            panel = rng.choice((2, 3, 4, 5, 7))
            if perim % panel and perim / panel >= 4:
                break
        text = (
            f"The plan shows {_a(kind)} {kind} {name} to be fenced all the way round its boundary. Each square is 1 cm by "
            f"1 cm on the plan. Fence panels are each {panel} metres long. How many whole panels must be bought?"
        )
        answer = _packs(perim, panel)
        cands = [
            perim // panel,
            round(perim / panel),
            _packs(perim - max(abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in _edges(poly)) * k, panel),
            _packs(_perimeter(poly), panel),
            _packs(cells * k * k, panel),
            answer + 1,
        ]
        working = [
            f"Add every side on the plan: {_perimeter(poly)} cm, so the actual boundary is {perim} m.",
            f"{perim} / {panel} = {round(perim / panel, 2)}, so round up to {answer} panels.",
        ]
    return fig, text, answer, cands, working


def _materials_hard(rng):
    k = rng.choice((1, 2))
    w = rng.randint(5, 11)
    h = rng.randint(4, 8)
    shape = rng.choice(("rect", "l"))
    poly = _rect(w, h) if shape == "rect" else _l_shape(rng, w, h)
    nice = "" if shape == "rect" else "L-shaped "
    name = rng.choice(("lawn", "courtyard", "play area"))
    fig = _plan(
        poly,
        name.capitalize(),
        _scale_text(k, rng.random() < 0.3),
        alt=f"A scale drawing of a {nice}{name} on a square grid, with its scale.",
    )
    area = _area(poly) * k * k
    for _ in attempts(200, "two options with different whole-pack costs"):
        c1, c2 = rng.sample((2, 3, 4, 5, 6, 7, 8, 10), 2)
        p1, p2 = rng.randint(8, 40), rng.randint(8, 40)
        n1, n2 = _packs(area, c1), _packs(area, c2)
        t1, t2 = n1 * p1, n2 * p2
        if area % c1 and area % c2 and abs(t1 - t2) >= 5 and min(n1, n2) >= 2:
            break
    cheap, dear = min(t1, t2), max(t1, t2)
    answer = dear - cheap
    pa, pb = (p1, p2)
    text = (
        f"The plan shows {_a(nice or name)} {nice}{name} to be covered with turf. Each square is 1 cm by 1 cm on the plan. "
        f"Option A: rolls that each cover {c1} square metres and cost ${p1} a roll. "
        f"Option B: rolls that each cover {c2} square metres and cost ${p2} a roll. "
        f"Only whole rolls can be bought. How much less, in dollars, does the cheaper option cost than the other?"
    )
    exact1, exact2 = Fraction(area, c1) * p1, Fraction(area, c2) * p2
    cands = [
        abs(exact1 - exact2),  # never rounds up to whole rolls
        abs(area // c1 * p1 - area // c2 * p2),  # rounds down
        cheap,  # the cost of the cheaper option, not the saving
        abs(pa - pb),  # compares the price of one roll
        abs(n1 - n2) * 10,
        dear,
    ]
    cands = [c if (_fr(c) * 10).denominator == 1 else round(_fr(c)) for c in cands]
    working = [
        f"Actual area: {_area(poly)} square centimetres on the plan × {k * k} = {area} square metres.",
        f"Option A: {area} / {c1} rounds up to {n1} rolls, costing {n1} × {p1} = ${t1}.",
        f"Option B: {area} / {c2} rounds up to {n2} rolls, costing {n2} × {p2} = ${t2}.",
        f"The cheaper option saves ${dear} - ${cheap} = ${answer}.",
    ]
    return fig, text, answer, cands, working


_MATERIALS = {"easy": _materials_easy, "medium": _materials_medium, "hard": _materials_hard}


@generator("grid_scale_materials", level=11, presets={t: dict(tier=t) for t in _MATERIALS})
def grid_scale_materials(rng, tier="easy"):
    """Whole packs, rolls or panels, and a two-option cost comparison, from a scale plan. See the module docstring."""
    return _finish(rng, _tier(tier, _MATERIALS))


_COVER = re.compile(r"each cover (\d+) square metres")
_PANEL = re.compile(r"panels are each (\d+) metres long")
_OPTION = re.compile(r"Option ([AB]): rolls that each cover (\d+) square metres and cost \$(\d+) a roll")


def _read_materials(question) -> str:
    fig, text = question.figure, question.text
    poly = _drawn_polygon(fig)
    k = _metres_per_cm(fig)
    area = _area(poly) * k * k
    options = _OPTION.findall(text)
    if options:
        check(len(options) == 2, "expected two options")
        totals = [_packs(area, int(c)) * int(p) for _label, c, p in options]
        return _text(abs(totals[0] - totals[1]))
    if (m := _PANEL.search(text)) is not None:
        return _text(_packs(_perimeter(poly) * k, int(m.group(1))))
    m = _COVER.search(text)
    check(m is not None, "cannot read the pack size")
    return _text(_packs(area, int(m.group(1))))


figures.register_reader("grid_scale_materials", _read_materials)

# ------------------------------------------------------------------------------------------- 4. conventions

_RATIO_PAIRS = (100, 200, 400, 500, 1000)


def _wall_labelled_plan(rng, k, tier_hard):
    """A plan with one wall labelled with its real length and no scale written. Returns the figure and the wall."""
    for _ in attempts(100, "a plan with a labelled wall"):
        w = rng.randint(4, 12)
        h = rng.randint(3, 8)
        poly = _rect(w, h) if rng.random() < 0.6 else _l_shape(rng, max(w, 4), max(h, 3))
        edges = _edges(poly)
        a, b = rng.choice(edges)
        cells = abs(b[0] - a[0]) + abs(b[1] - a[1])
        if cells < 3:
            continue
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        mid = tuple(int(c) if c == int(c) else c for c in mid)
        real = cells * k
        name = rng.choice(_ROOMS)
        margin = 1
        fig = grid.grid_drawing(
            alt=f"A plan of a {name} on a square grid with one wall labelled with its actual length.",
            grid={"type": "square", "width": _box(poly)[0] + 2 * margin, "height": _box(poly)[1] + 2 * margin},
            polygons=[
                {
                    "vertices": [[x + margin, y + margin] for x, y in poly],
                    "style": "solid",
                    "label": name.capitalize(),
                }
            ],
            labels=[{"at": [mid[0] + margin, mid[1] + margin], "text": f"{_text(real)} m"}],
        )
        return fig, poly, cells, real, name
    raise AssertionError("unreachable")


def _conv_easy(rng):
    to_ratio = rng.random() < 0.5
    n = rng.choice(_RATIO_PAIRS if to_ratio else _RATIO_PAIRS[1:])  # an answer of 1 m would echo the 1 in 1 : n
    k = Fraction(n, 100)
    name = rng.choice(_ROOMS)
    poly = _rect(rng.randint(4, 10), rng.randint(3, 7))
    fig = _plan(
        poly,
        name.capitalize(),
        _scale_text(k, not to_ratio),
        alt=f"A scale drawing of a {name} on a square grid, with its scale written beside it.",
        bar=rng.random() < 0.4,
    )
    if to_ratio:
        text = phrase(
            rng,
            (
                "The scale of the plan is shown beside it. Write the scale as a ratio 1 : n. What is n?",
                "The plan's scale is written beside it. Written as the ratio 1 : n, what is n?",
                "Beside the plan is its scale. Express it as a ratio 1 : n and give the value of n.",
            ),
        )
        answer = n
        cands = [k, k * 10, k * 1000, n * 10, n * 100, Fraction(n, 10)]
        working = [
            f"1 cm on the plan is {_text(k)} m = {n} cm.",
            f"So the ratio is 1 : {n}, and n = {n}.",
        ]
    else:
        text = phrase(
            rng,
            (
                "The scale of the plan is shown beside it as a ratio. How many metres does 1 cm on the plan stand for?",
                "The plan is drawn to the ratio scale shown. Find the actual length, in metres, that 1 cm on the plan "
                "represents.",
                "Beside the plan is its scale as a ratio. What real distance, in metres, is 1 cm on the plan?",
            ),
        )
        answer = k
        cands = [n, Fraction(n, 10), Fraction(n, 1000), n * 100, Fraction(n, 10) * 10]
        working = [f"1 : {n} means 1 cm on the plan is {n} cm in real life.", f"{n} cm = {_text(k)} m."]
    return fig, text, answer, cands, working


def _conv_medium(rng):
    k = rng.choice((1, 2, 3, 4, 5, 10))
    fig, poly, cells, real, name = _wall_labelled_plan(rng, k, False)
    if rng.random() < 0.5:
        text = (
            f"One wall of the {name} is labelled with its actual length. Each square of the plan is 1 cm by 1 cm. "
            f"How many metres does 1 cm on the plan stand for?"
        )
        answer = k
        cands = [real, cells, k * 100, k * 10, real * cells]
    else:
        text = (
            f"One wall of the {name} is labelled with its actual length. Each square of the plan is 1 cm by 1 cm. "
            f"The scale is 1 : n. What is n?"
        )
        answer = 100 * k
        cands = [k, k * 10, k * 1000, real * 100, cells * 100]
    working = [
        f"The labelled wall is {cells} cm on the plan and {_text(real)} m in real life.",
        f"{_text(real)} / {cells} = {k} m for every 1 cm, which is {100 * k} cm for every 1 cm: the scale is 1 : {100 * k}.",
    ]
    return fig, text, answer, cands, working


def _conv_hard(rng):
    k = rng.choice((2, 3, 4, 5, 10))
    fig, poly, cells, real, name = _wall_labelled_plan(rng, k, True)
    for _ in attempts(100, "a real length that is a whole or half number of centimetres"):
        steps = rng.randint(3, 30)
        target = Fraction(steps * k, 2)
        if _text(target) != _text(real) and target.denominator <= 2:
            break
    drawn = target / k
    text = (
        f"One wall of the {name} is labelled with its actual length. Each square of the plan is 1 cm by 1 cm. "
        f"A fence {_text(target)} m long is to be added to the plan. How long, in centimetres, is it drawn?"
    )
    cands = [
        target * k,  # multiplies where it should divide
        target,  # the real length, unscaled
        target * 100 / k,  # a unit slip
        target / (k * 100),
        drawn + 1,
        drawn * 2,
    ]
    working = [
        f"The labelled wall is {cells} cm on the plan and {_text(real)} m in real life, so 1 cm stands for {k} m.",
        f"A length of {_text(target)} m is drawn as {_text(target)} / {k} = {_text(drawn)} cm.",
    ]
    return fig, text, drawn, cands, working


_CONVENTIONS = {"easy": _conv_easy, "medium": _conv_medium, "hard": _conv_hard}


@generator("grid_scale_conventions", level=11, presets={t: dict(tier=t) for t in _CONVENTIONS})
def grid_scale_conventions(rng, tier="easy"):
    """Scales as ratios, the scale from a labelled dimension, and the drawn length of a real one."""
    return _finish(rng, _tier(tier, _CONVENTIONS))


_DIMENSION = re.compile(r"^(\d+(?:\.\d+)?) m$")
_FENCE = re.compile(r"A fence (\d+(?:\.\d+)?) m long")


def _labelled_metres_per_cm(fig) -> Fraction:
    """The scale a labelled wall gives: its real length over the wall's length in cells."""
    poly = _drawn_polygon(fig)
    found = []
    for label in fig.get("labels", []):
        m = _DIMENSION.match(label["text"])
        if m is None:
            continue
        for (x1, y1), (x2, y2) in _edges(poly):
            if ((x1 + x2) / 2, (y1 + y2) / 2) == tuple(label["at"]):
                found.append(Fraction(m.group(1)) / (abs(x2 - x1) + abs(y2 - y1)))
    check(len(found) == 1, "expected one labelled wall")
    return found[0]


def _read_conventions(question) -> str:
    fig, text = question.figure, question.text
    if "scale" in fig:
        k = _metres_per_cm(fig)
        if "1 : n" in text:
            return _text(k * 100)
        return _text(k)
    k = _labelled_metres_per_cm(fig)
    if (m := _FENCE.search(text)) is not None:
        return _text(Fraction(m.group(1)) / k)
    if "The scale is 1 : n" in text:
        return _text(k * 100)
    return _text(k)


figures.register_reader("grid_scale_conventions", _read_conventions)
