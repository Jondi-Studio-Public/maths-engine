"""Exam chains for the scale-drawing and drawing-convention skills (the ``w7f`` figure work).

Four chains, one per skill, each read off one drawn plan or map on a square grid (``grid_drawing``) where one square
is 1 cm on the page. Every part is a ``number``; later parts follow through from earlier ones.

* ``exd_x_w7f_scale_actual_chain`` (``senior-find-actual-measurements-from-scale-drawings-such-as``): an L- or
  U-shaped room. (a) the actual length of its longest wall, (b) its actual perimeter, (c) its actual area, (d) the
  cost of covering it at a stated price per square metre (follows (c)).
* ``exd_x_w7f_scale_map_chain`` (``senior-obtain-measurements-from-scale-drawings-such-as-maps``): a map with a road
  that doubles back. (a) the length of the road, (b) the walking time along it at a stated speed (follows (a)),
  (c) how much shorter the straight-across-then-up route is (follows (a)).
* ``exd_x_w7f_scale_materials_chain`` (``senior-estimate-and-compare-quantities-materials-and-costs-using``): a lawn.
  (a) its actual area, (b) whole rolls of turf (follows (a)), (c) the cost of the rolls (follows (b)), (d) the turf
  left over (follows (a) and (b)).
* ``exd_x_w7f_scale_conventions_chain`` (``senior-understand-and-apply-drawing-conventions-of-scale-drawings``): a
  plan with one wall labelled and no scale written. (a) metres to the centimetre, (b) the ratio 1 : n (follows (a)),
  (c) the drawn length of a stated real length (follows (a)), (d) the actual perimeter (follows (a)).

Each answer is built from the drawn shape and a whole-number scale first and the question derived from it; the
figure reader registered for each topic re-derives every part from the figure and the prompts alone.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from mathgen import figures
from mathgen.core import check, generator, phrase

from ..figure_kinds import grid
from . import _space_kit as kit
from ._w7f_kit import finish_parts, number_part, retry, split_parts
from .grid_scale_drawing_build import (
    _LAWNS,
    _MAP_SCALES,
    _ROOMS,
    _a,
    _area,
    _drawn_polygon,
    _edges,
    _l_shape,
    _length,
    _labelled_metres_per_cm,
    _map_figure,
    _metres_per_cm,
    _name,
    _packs,
    _perimeter,
    _plan,
    _rect,
    _route,
    _scale_text,
    _u_shape,
    _wall_labelled_plan,
)


def _edge_len(edge) -> int:
    (x1, y1), (x2, y2) = edge
    return abs(x2 - x1) + abs(y2 - y1)


# ------------------------------------------------------------------------------------------- 1. actual lengths


def _actual_chain(rng):
    k = rng.choice((2, 3, 4, 5))
    w, h = rng.randint(6, 12), rng.randint(5, 9)
    u = rng.random() < 0.35
    poly = _u_shape(rng, w, h) if u else _l_shape(rng, w, h)
    kind = "U-shaped" if u else "L-shaped"
    lens = sorted((_edge_len(e) for e in _edges(poly)), reverse=True)
    name = rng.choice(("hall", "workshop", "studio", "courtyard") if u else _ROOMS)
    ratio = rng.random() < 0.4
    fig = _plan(poly, name.capitalize(), _scale_text(k, ratio), alt=f"A scale drawing of {_a(kind)} {kind} {name}.")
    cells, perim = _area(poly), _perimeter(poly)
    longest = lens[0]
    price = rng.choice((12, 15, 18, 20, 24, 25, 30, 35, 40, 45))
    a_key, b_key, c_key = longest * k, perim * k, cells * k * k
    d_key = c_key * price
    stem = (
        f"The plan shows {_a(kind)} {kind} {name}. It is drawn on a square grid and each square is 1 cm by 1 cm "
        f"on the plan."
    )
    box_w, box_h = max(x for x, _ in poly), max(y for _, y in poly)
    part_a = number_part(
        phrase(
            rng,
            (
                "Find the actual length, in metres, of the longest wall of the {n}.",
                "What is the actual length, in metres, of the longest wall of the {n}?",
            ),
            n=name,
        ),
        a_key,
        1,
        [f"The longest wall is {longest} cm on the plan.", f"{longest} x {k} = {a_key} m."],
        [
            (longest, "scd-length-unscaled"),
            (lens[1] * k, "scd-wrong-wall"),
            (a_key * 100, "scd-metre-centimetre-slip"),
            (longest + k, "scd-adds-scale"),
            (longest * k * k, "scd-scale-squared-for-length"),
        ],
    )
    part_b = number_part(
        f"Find the actual perimeter of the {name}, in metres.",
        b_key,
        2,
        [f"Add the {len(poly)} walls on the plan: {perim} cm.", f"{perim} x {k} = {b_key} m."],
        [
            (perim, "scd-length-unscaled"),
            ((perim - longest) * k, "scd-perimeter-omits-side"),
            (2 * (box_w + box_h) * k, "scd-perimeter-bounding-rectangle"),
            (cells * k, "scd-area-for-perimeter"),
            (perim * k * k, "scd-scale-squared-for-length"),
            (b_key * 100, "scd-metre-centimetre-slip"),
        ],
    )
    part_c = number_part(
        f"Find the actual area of the {name}, in square metres.",
        c_key,
        2,
        [
            f"Count the squares on the plan: {cells} square centimetres.",
            f"1 square centimetre stands for {k} x {k} = {k * k} square metres, so the area is {cells} x {k * k} = {c_key}.",
        ],
        [
            (cells * k, "scd-area-linear-factor"),
            (cells, "scd-area-unscaled"),
            (c_key * 100, "scd-metre-centimetre-slip"),
            (b_key * k, "scd-area-from-perimeter"),
            (cells * k * 2, "scd-area-linear-factor"),
        ],
    )
    part_d = number_part(
        phrase(
            rng,
            (
                "The floor is to be covered at ${p} per square metre. Find the cost, in dollars.",
                "Covering the floor costs ${p} for each square metre. How much, in dollars, will it cost?",
            ),
            p=price,
        ),
        d_key,
        2,
        [f"Cost = area x price = {c_key} x {price} = ${d_key}."],
        [
            (b_key * price, "scd-price-on-perimeter"),
            (cells * price, "scd-price-on-drawn-area"),
            (cells * k * price, "scd-area-linear-factor"),
            (c_key + price, "scd-cost-adds-price"),
            (d_key * 100, "scd-metre-centimetre-slip"),
        ],
        follow=f"{price}*part_c",
    )
    check(math.isclose(_perimeter(poly) * k, b_key) and _area(poly) * k * k == c_key, "scaled values disagree")
    return finish_parts(fig, stem, [part_a, part_b, part_c, part_d])


@generator("exd_x_w7f_scale_actual_chain", level=11, presets={"exam": {"tier": "exam"}})
def exd_x_w7f_scale_actual_chain(rng, tier: str = "exam"):
    """Longest wall, perimeter, area and floor cost of an L- or U-shaped room from its scale plan."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry(rng, _actual_chain, "a scale plan with three distractors per part")


_PRICE = re.compile(r"\$(\d+) (?:per|for each) square metre")


def _read_actual_chain(question) -> str:
    fig = question.figure
    poly = _drawn_polygon(fig)
    k = _metres_per_cm(fig)
    price = int(_PRICE.search(split_parts(question.text)["d"]).group(1))
    longest = max(_edge_len(e) for e in _edges(poly))
    area = _area(poly) * k * k
    return f"(a) {longest * k}; (b) {_perimeter(poly) * k}; (c) {area}; (d) {area * price}"


figures.register_reader("exd_x_w7f_scale_actual_chain", _read_actual_chain)

# ------------------------------------------------------------------------------------------- 2. maps


def _map_chain(rng):
    scale_text, k = rng.choice(_MAP_SCALES)  # 1, 2 or 5 km to the centimetre
    k = int(k)
    gw, gh = rng.randint(14, 18), rng.randint(10, 13)
    pts, seen = _route(rng, gw, gh, 3, 2, 8)
    manhattan = abs(pts[-1][0] - pts[0][0]) + abs(pts[-1][1] - pts[0][1])
    cells = _length(pts)
    if cells == manhattan:
        return None
    names = rng.sample(
        ("house", "school", "church", "hospital", "post_office", "railway_station", "lighthouse", "campground"), 2
    )
    try:
        fig = _map_figure(rng, gw, gh, pts, seen, scale_text, names)
    except Exception:  # noqa: BLE001 - this draw had no room for a symbol or a river: draw again
        return None
    legs = [_edge_len(e) for e in zip(pts, pts[1:])]
    dist = cells * k
    speed = rng.choice((3, 4, 5, 6, 8, 10, 12))
    minutes = Fraction(dist * 60, speed)
    if minutes.denominator != 1 or not 10 <= minutes <= 600:
        return None
    minutes = int(minutes)
    direct = manhattan * k
    saving = dist - direct
    if saving < 2:
        return None
    a, b = _name(names[0]), _name(names[1])
    stem = (
        f"The map is drawn on a square grid. Each square is 1 cm by 1 cm on the map. A road joins the {a} and the {b}."
    )
    part_a = number_part(
        f"How far is it along the road from the {a} to the {b}, in kilometres?",
        dist,
        1,
        [f"Add the legs of the road: {' + '.join(map(str, legs))} = {cells} cm.", f"{cells} x {k} = {dist} km."],
        [
            (manhattan * k, "scd-straight-across-route"),
            ((cells - legs[-1]) * k, "scd-leaves-out-a-leg"),
            ((cells - legs[0]) * k, "scd-leaves-out-a-leg"),
            (cells, "scd-length-unscaled"),
            (dist * 1000, "scd-kilometre-metre-slip"),
        ],
    )
    part_b = number_part(
        phrase(
            rng,
            (
                "Sam walks along the road at {s} km/h. How many minutes does the walk take?",
                "A walker covers the road at a steady {s} km/h. How long does the walk take, in minutes?",
            ),
            s=speed,
        ),
        minutes,
        2,
        [f"Time = {dist} / {speed} hours.", f"{dist} / {speed} x 60 = {minutes} minutes."],
        [
            (Fraction(dist, speed), "scd-hours-for-minutes"),
            (Fraction(dist * 100, speed), "scd-hours-for-minutes"),
            (Fraction(direct * 60, speed), "scd-straight-across-route"),
            (dist * speed, "scd-multiplies-for-time"),
            (Fraction(cells * 60, speed), "scd-length-unscaled"),
            (Fraction(dist * 60, speed) + 60, "scd-time-off-by-an-hour"),
        ],
        follow=f"part_a*60/{speed}",
    )
    part_c = number_part(
        "A direct track goes straight across the map and then straight up it (or up and then across), along the grid "
        "lines. How many kilometres shorter is the track than the road?",
        saving,
        2,
        [
            f"The track is {abs(pts[-1][0] - pts[0][0])} + {abs(pts[-1][1] - pts[0][1])} = {manhattan} cm on the map, "
            f"which is {direct} km.",
            f"The road is {dist} km, so the track is {dist} - {direct} = {saving} km shorter.",
        ],
        [
            (direct, "scd-gives-track-not-difference"),
            (manhattan, "scd-length-unscaled"),
            (cells - manhattan, "scd-difference-unscaled"),
            (dist + direct, "scd-adds-instead-of-subtracts"),
            (saving * 1000, "scd-kilometre-metre-slip"),
            (saving + k, "scd-adds-scale"),
        ],
        follow=f"part_a-{direct}",
    )
    return finish_parts(fig, stem, [part_a, part_b, part_c])


@generator("exd_x_w7f_scale_map_chain", level=11, presets={"exam": {"tier": "exam"}})
def exd_x_w7f_scale_map_chain(rng, tier: str = "exam"):
    """Road length, walking time and the saving of a direct track from a scale map."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry(rng, _map_chain, "a map with a doubling-back road and three distractors per part")


_JOIN = re.compile(r"A road joins the (.+?) and the (.+?)\.")
_SPEED = re.compile(r"(\d+) km/h")


def _read_map_chain(question) -> str:
    fig, parts = question.figure, split_parts(question.text)
    by_name = {v: s for s, v in grid.SYMBOLS.items()}
    m = _JOIN.search(question.text)
    check(m is not None, "cannot read the two places")
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
    k = _metres_per_cm(fig) / 1000
    dist = _length(roads[0]) * k
    speed = int(_SPEED.search(parts["b"]).group(1))
    direct = (abs(end[0] - start[0]) + abs(end[1] - start[1])) * k
    return f"(a) {dist}; (b) {dist * 60 / speed}; (c) {dist - direct}"


figures.register_reader("exd_x_w7f_scale_map_chain", _read_map_chain)

# ------------------------------------------------------------------------------------------- 3. materials


def _materials_chain(rng):
    k = rng.choice((1, 2, 3))
    w, h = rng.randint(5, 11), rng.randint(4, 8)
    poly = _l_shape(rng, w, h) if rng.random() < 0.6 else _u_shape(rng, max(w, 5), h)
    kind = "U-shaped" if len(poly) == 8 else "L-shaped"
    name = rng.choice(_LAWNS)
    fig = _plan(
        poly,
        name.capitalize(),
        _scale_text(k, rng.random() < 0.3),
        alt=f"A scale drawing of {_a(kind)} {kind} {name} on a square grid, with its scale.",
    )
    cells = _area(poly)
    area = cells * k * k
    cover = rng.choice((3, 4, 5, 6, 7, 8, 9))
    if area % cover == 0 or area / cover < 4:
        return None
    rolls = _packs(area, cover)
    price = rng.choice((18, 24, 28, 32, 36, 45, 52))
    cost = rolls * price
    left = rolls * cover - area
    stem = (
        f"The plan shows {_a(kind)} {kind} {name} that is to be covered with turf. Each square is 1 cm by 1 cm on "
        f"the plan. Turf is sold in rolls that each cover {cover} square metres and cost ${price} a roll."
    )
    part_a = number_part(
        f"Find the actual area of the {name}, in square metres.",
        area,
        2,
        [
            f"{cells} square centimetres on the plan, each standing for {k * k} square metres.",
            f"{cells} x {k * k} = {area}.",
        ],
        [
            (cells * k, "scd-area-linear-factor"),
            (cells, "scd-area-unscaled"),
            (_perimeter(poly) * k, "scd-area-for-perimeter"),
            (area * 100, "scd-metre-centimetre-slip"),
        ],
    )
    part_b = number_part(
        "Only whole rolls can be bought. How many rolls are needed to cover the whole area?",
        rolls,
        1,
        [f"{area} / {cover} = {area / cover:.2f}, so round up to {rolls} rolls."],
        [
            (area // cover, "scd-rounds-down"),
            (round(area / cover) if round(area / cover) != rolls else None, "scd-rounds-to-nearest"),
            (rolls + 1, "scd-one-roll-too-many"),
            (_packs(cells, cover), "scd-area-unscaled"),
            (_packs(cells * k, cover), "scd-area-linear-factor"),
        ],
    )
    part_c = number_part(
        "Find the cost of the rolls, in dollars.",
        cost,
        1,
        [f"{rolls} x ${price} = ${cost}."],
        [
            (area * price // cover if (area * price) % cover == 0 else None, "scd-cost-of-fractional-rolls"),
            ((rolls - 1) * price, "scd-rounds-down"),
            ((rolls + 1) * price, "scd-one-roll-too-many"),
            (rolls + price, "scd-cost-adds-price"),
            (area * price, "scd-price-per-square-metre-for-roll"),
        ],
        follow=f"part_b*{price}",
    )
    part_d = number_part(
        "How many square metres of turf are left over after the whole area is covered?",
        left,
        2,
        [
            f"{rolls} rolls cover {rolls} x {cover} = {rolls * cover} square metres.",
            f"{rolls * cover} - {area} = {left}.",
        ],
        [
            (cover - left if cover - left != left else None, "scd-leftover-complement"),
            (rolls * cover, "scd-gives-turf-bought-not-left"),
            (area % cover, "scd-remainder-for-leftover"),
            (rolls, "scd-gives-rolls"),
            (left + cover, "scd-leftover-extra-roll"),
        ],
        follow=f"{cover}*part_b-part_a",
    )
    check(rolls * cover - area == left and 0 < left < cover, "leftover disagrees")
    return finish_parts(fig, stem, [part_a, part_b, part_c, part_d])


@generator("exd_x_w7f_scale_materials_chain", level=11, presets={"exam": {"tier": "exam"}})
def exd_x_w7f_scale_materials_chain(rng, tier: str = "exam"):
    """Area, whole rolls, cost and turf left over for a lawn drawn to scale."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry(rng, _materials_chain, "a lawn plan with three distractors per part")


_ROLL = re.compile(r"cover (\d+) square metres and cost \$(\d+) a roll")


def _read_materials_chain(question) -> str:
    fig = question.figure
    k = _metres_per_cm(fig)
    area = _area(_drawn_polygon(fig)) * k * k
    cover, price = (int(g) for g in _ROLL.search(question.text).groups())
    rolls = _packs(area, cover)
    return f"(a) {area}; (b) {rolls}; (c) {rolls * price}; (d) {rolls * cover - area}"


figures.register_reader("exd_x_w7f_scale_materials_chain", _read_materials_chain)

# ------------------------------------------------------------------------------------------- 4. conventions


def _conventions_chain(rng):
    k = rng.choice((2, 3, 4, 5, 10))
    fig, poly, cells, real, name = _wall_labelled_plan(rng, k, True)
    target = rng.randint(3, 12) * k
    if target == real:
        return None
    drawn = target // k
    perim = _perimeter(poly)
    stem = (
        f"One wall of the {name} is labelled with its actual length and no scale is written. Each square of the plan "
        f"is 1 cm by 1 cm."
    )
    part_a = number_part(
        "How many metres does 1 cm on the plan stand for?",
        k,
        1,
        [f"The labelled wall is {cells} cm on the plan and {real} m in real life.", f"{real} / {cells} = {k} m."],
        [
            (real, "scd-reads-label-as-scale"),
            (cells, "scd-reads-label-as-scale"),
            (k * 100, "scd-metre-centimetre-slip"),
            (k * 10, "scd-metre-centimetre-slip"),
            (real * cells, "scd-multiplies-instead-of-divides"),
        ],
    )
    part_b = number_part(
        "Write the scale as a ratio 1 : n. What is n?",
        100 * k,
        1,
        [f"1 cm stands for {k} m = {100 * k} cm, so the scale is 1 : {100 * k}."],
        [
            (k, "scd-ratio-not-converted"),
            (k * 10, "scd-metre-centimetre-slip"),
            (k * 1000, "scd-metre-centimetre-slip"),
            (cells * 100, "scd-reads-label-as-scale"),
        ],
        follow="100*part_a",
    )
    part_c = number_part(
        f"A fence {target} m long is to be added to the plan. How long, in centimetres, is it drawn?",
        drawn,
        2,
        [f"{target} / {k} = {drawn} cm."],
        [
            (target * k, "scd-multiplies-instead-of-divides"),
            (target, "scd-length-unscaled"),
            (drawn + 1, "scd-off-by-one"),
            (Fraction(target * 100, k), "scd-metre-centimetre-slip"),
            (drawn * 2, "scd-doubles-drawn-length"),
        ],
        follow=f"{target}/part_a",
    )
    part_d = number_part(
        f"Find the actual perimeter of the {name}, in metres.",
        perim * k,
        2,
        [f"The perimeter on the plan is {perim} cm.", f"{perim} x {k} = {perim * k} m."],
        [
            (perim, "scd-length-unscaled"),
            (perim * k * k, "scd-scale-squared-for-length"),
            ((perim - cells) * k if cells < perim else None, "scd-perimeter-omits-side"),
            (_area(poly) * k, "scd-area-for-perimeter"),
            (perim * k * 100, "scd-metre-centimetre-slip"),
        ],
        follow=f"{perim}*part_a",
    )
    return finish_parts(fig, stem, [part_a, part_b, part_c, part_d])


@generator("exd_x_w7f_scale_conventions_chain", level=11, presets={"exam": {"tier": "exam"}})
def exd_x_w7f_scale_conventions_chain(rng, tier: str = "exam"):
    """Scale from a labelled wall, as a ratio, a drawn length and the actual perimeter."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return retry(rng, _conventions_chain, "a labelled plan with three distractors per part")


_FENCE = re.compile(r"A fence (\d+) m long")


def _read_conventions_chain(question) -> str:
    fig = question.figure
    k = _labelled_metres_per_cm(fig)
    poly = _drawn_polygon(fig)
    fence = Fraction(_FENCE.search(question.text).group(1))
    return f"(a) {k}; (b) {k * 100}; (c) {fence / k}; (d) {_perimeter(poly) * k}"


figures.register_reader("exd_x_w7f_scale_conventions_chain", _read_conventions_chain)
