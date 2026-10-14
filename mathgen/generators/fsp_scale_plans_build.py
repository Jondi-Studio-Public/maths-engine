"""Generates `fsp_scale_plan` and `fsp_scale_plan_exam` questions: real sizes from a drawn plan and its scale.

One skill: ``f10-scale-drawing-floor-plans`` (``answer: number``). The stimulus is a floor plan drawn with the existing
``diagram`` figure kind, to scale, with some plan lengths marked in centimetres (4.5 cm and so on). The question gives the
scale as a ratio, 1 : 50, 1 : 100 or 1 : 200. Three tiers:

- easy: the real length of a named wall, in metres (plan length times the scale, then centimetres to metres). Some plans
  have two rooms side by side.
- medium: the real floor area of a named room, in square metres. The plan area is not scaled once by the ratio: each of the
  two lengths is, so the area grows by the square of the ratio.
- hard: an L-shaped plan with four lengths marked. Two sides are not marked and have to be found from the opposite sides;
  the question asks for the real floor area (the plan is a rectangle with a corner cut out) or the real perimeter (which
  needs those same two sides).

``fsp_scale_plan_exam`` is the exam chain on a plan of one room: (a) the real length of the long wall, (b) the real
length of the short wall, (c) the floor area (follows through from (a) and (b)), (d) the number of square tiles needed
(follows through from (c)).

Every real length is built as an exact fraction. The areas and perimeters are checked again by measuring the drawn
polygon (the shoelace formula) and converting with the drawn-to-marked length ratio, which is what the figure reader
does from the figure alone.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from mathgen import figures
from mathgen.answers import Distractor
from mathgen.core import (
    attempts,
    check,
    generator,
    latex_answer,
    numeric_answer,
    phrase,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)
from mathgen.figure_kinds import space
from . import _fsp_kit as kit

_NAMES = ("Mia", "Tom", "Aiko", "Dev", "Leila", "Sam", "Jess", "Kofi", "Priya", "Marcus", "Zoe", "Ben")
_PLACES = {
    50: ("bedroom", "kitchen", "bathroom", "study", "laundry", "pantry"),
    100: ("classroom", "hall", "staff room", "library", "drama room", "workshop"),
    200: ("gym", "foyer", "canteen", "warehouse bay", "town hall", "garage block"),
}
_ALT_ROOM = "A floor plan drawn to scale, with some of the wall lengths marked in centimetres."
_FRAME = 90.0
_MARGIN = 14.0


# ------------------------------------------------------------------------------------------ helpers


def _dp(value: Fraction) -> int:
    for dp in range(0, 5):
        if (value * 10**dp).denominator == 1:
            return dp
    raise ValueError(f"{value} needs more than four decimal places")


def _cm(v10: int) -> str:
    return f"{Fraction(v10, 10):g} cm" if v10 % 10 else f"{v10 // 10} cm"


def _real_m(plan_v10: int, k: int) -> Fraction:
    """Real length in metres for a plan length of ``plan_v10`` tenths of a centimetre at scale 1 : k."""
    return Fraction(plan_v10, 10) * k / 100


def _answer_fields(value: Fraction, dp: int):
    shown = stated_precision_string(round_half_up(value, dp), dp)
    return numeric_answer(round_half_up(value, dp)), shown


def _distractors(value: Fraction, dp: int, candidates):
    """Tagged distractors at the answer's own precision, none within the grader's tolerance of the answer."""
    answer_text = stated_precision_string(round_half_up(value, dp), dp)
    tolerance = stated_precision_tolerance(dp) if dp else 0
    out = []
    for fraction, tag in candidates:
        if fraction <= 0:
            continue
        rounded = round_half_up(fraction, dp)
        if rounded <= 0:
            continue
        out.append(Distractor(stated_precision_string(rounded, dp), tag, numeric_answer(rounded)))
    return pick_distractors(answer_text, out, 3, tolerance=tolerance)


def _plan_figure(polygons, labelled, *, alt=_ALT_ROOM):
    """A to-scale ``diagram``. ``polygons`` is a list of lists of (letter, x, y) in plan tenths of a centimetre;
    ``labelled`` maps a (letter, letter) side to its marked length text."""
    pts = {}
    for poly in polygons:
        for letter, x, y in poly:
            pts[letter] = (x, y)
    xs = [p[0] for p in pts.values()]
    ys = [p[1] for p in pts.values()]
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    scale = (_FRAME - 2 * _MARGIN) / max(width, height)
    points = [
        {"id": letter, "at": [_MARGIN + (x - min(xs)) * scale, _MARGIN + (y - min(ys)) * scale], "label": letter}
        for letter, (x, y) in pts.items()
    ]
    drawn = set()
    segments = []
    for poly in polygons:
        letters = [p[0] for p in poly]
        for a, b in zip(letters, letters[1:] + letters[:1]):
            key = frozenset((a, b))
            if key in drawn:
                continue
            drawn.add(key)
            text = labelled.get((a, b)) or labelled.get((b, a))
            if text:
                segments.append({"from": a, "to": b, "label": text})
    frame = (width * scale + 2 * _MARGIN, height * scale + 2 * _MARGIN)
    return space.diagram(
        alt=alt,
        points=points,
        segments=segments,
        polygons=[{"points": [p[0] for p in poly]} for poly in polygons],
        frame=frame,
    )


def _single_room(w10, h10, labelled_sides=("AB", "BC")):
    poly = [("A", 0, 0), ("B", w10, 0), ("C", w10, h10), ("D", 0, h10)]
    labels = {}
    lengths = {"AB": w10, "BC": h10}
    for side in labelled_sides:
        labels[(side[0], side[1])] = _cm(lengths[side])
    return _plan_figure([poly], labels)


def _two_rooms(w1, w2, h, labelled_sides=("AB", "BC", "AF")):
    a, b, c = ("A", 0, 0), ("B", w1, 0), ("C", w1 + w2, 0)
    d, e, f = ("D", w1 + w2, h), ("E", w1, h), ("F", 0, h)
    lengths = {"AB": w1, "BC": w2, "AF": h}
    labels = {(s[0], s[1]): _cm(lengths[s]) for s in labelled_sides}
    return _plan_figure([[a, b, e, f], [b, c, d, e]], labels)


# ------------------------------------------------------------------------------------------ easy


def _easy(rng):
    for _ in attempts(500, "a plan whose real lengths need at most one decimal place"):
        k = rng.choice([50, 100, 200])
        two = rng.random() < 0.5
        w1, w2, h = (rng.randint(20, 95) for _ in range(3))
        if _dp(_real_m(w1, k)) > 1 or _dp(_real_m(w2, k)) > 1 or _dp(_real_m(h, k)) > 1:
            continue
        if two:
            ask = rng.choice(["AB", "BC", "AF"])
            fig = _two_rooms(w1, w2, h)
            plan10 = {"AB": w1, "BC": w2, "AF": h}[ask]
            other10 = {"AB": h, "BC": h, "AF": w1}[ask]
        else:
            ask = rng.choice(["AB", "BC"])
            fig = _single_room(w1, h)
            plan10 = {"AB": w1, "BC": h}[ask]
            other10 = {"AB": h, "BC": w1}[ask]
        value = _real_m(plan10, k)
        dp = _dp(value)
        answer, shown = _answer_fields(value, dp)
        if not kit.leaks(fig, shown):
            break
    p = Fraction(plan10, 10)
    cands = [
        (p * k, "scale-units-not-converted"),
        (p * k / 1000, "scale-converted-wrong-power"),
        (_real_m(other10, k), "scale-wrong-wall-read"),
        (p * k / 10, "scale-converted-wrong-power"),
        (p * 100 / k, "scale-applied-the-wrong-way"),
    ]
    place = rng.choice(_PLACES[k])
    text = phrase(
        rng,
        (
            "{who}'s {place} is shown on a plan drawn to a scale of 1 : {k}. What is the real length of wall {wall}, "
            "in metres?",
            "The plan of a {place} is drawn to a scale of 1 : {k}, with some lengths marked. Find the real length "
            "of wall {wall}, in metres.",
            "A {place} is drawn on a plan with a scale of 1 : {k}. {who} wants the real length of wall {wall}. "
            "How long is it, in metres?",
            "On {who}'s plan of a {place}, 1 cm stands for {cm} cm. The marked lengths are plan lengths. How long "
            "is wall {wall} in real life, in metres?",
        ),
        who=rng.choice(_NAMES),
        place=place,
        k=k,
        cm=k,
        wall=ask,
    )
    working = [
        f"At a scale of 1 : {k}, 1 cm on the plan is {k} cm in real life.",
        f"Wall {ask}: {_cm(plan10)} × {k} = {Fraction(plan10, 10) * k:g} cm = {shown} m.",
    ]
    return fig, text, answer, shown, value, dp, cands, working


# ------------------------------------------------------------------------------------------ medium


def _medium(rng):
    for _ in attempts(1000, "a plan whose real room has an area with at most two decimal places"):
        k = rng.choice([50, 100, 200])
        w1, w2, h = (rng.randint(20, 90) for _ in range(3))
        if any(_dp(_real_m(v, k)) > 1 for v in (w1, w2, h)):
            continue
        two = rng.random() < 0.5
        if two:
            room = rng.choice(["ABEF", "BCDE"])
            fig = _two_rooms(w1, w2, h)
            width10 = w1 if room == "ABEF" else w2
        else:
            room = "ABCD"
            fig = _single_room(w1, h)
            width10 = w1
        m_w, m_h = _real_m(width10, k), _real_m(h, k)
        value = m_w * m_h
        if value < 3:
            continue
        dp = _dp(value)
        answer, shown = _answer_fields(value, dp)
        if not kit.leaks(fig, shown):
            break
    plan_area = Fraction(width10, 10) * Fraction(h, 10)  # cm²
    cands = [
        (m_w * Fraction(h, 10), "scale-one-side-scaled"),
        (2 * (m_w + m_h), "perimeter-for-area"),
        (plan_area * k * k / 100, "linear-factor-on-area"),
        (plan_area * k / 10000, "scale-area-linear-factor"),
        (plan_area * k / 100, "scale-area-linear-factor"),
    ]
    place = rng.choice(_PLACES[k])
    text = phrase(
        rng,
        (
            "{who}'s plan of a {place} is drawn to a scale of 1 : {k}. What is the real floor area of room {room}, "
            "in square metres?",
            "The plan shown is drawn to a scale of 1 : {k}. {who} wants to carpet room {room} of the {place}. Find "
            "its real floor area, in square metres.",
            "A {place} is drawn on a plan to a scale of 1 : {k}. Find the real area of the floor of room {room}, in "
            "square metres.",
            "On the plan of {who}'s {place}, drawn at 1 : {k}, what is the floor area of room {room} in real life? "
            "Give your answer in square metres.",
        ),
        who=rng.choice(_NAMES),
        place=place,
        k=k,
        room=room,
    )
    working = [
        f"Real lengths: {_cm(width10)} × {k} = {m_w:g} m and {_cm(h)} × {k} = {m_h:g} m.",
        f"Area = {m_w:g} × {m_h:g} = {shown} m².",
    ]
    return fig, text, answer, shown, value, dp, cands, working


# ------------------------------------------------------------------------------------------ hard


def _l_shape(rng):
    """Plan tenths (a, b, c, d) of an L: bottom a wide, b high; the upper arm c wide, up to a total height d."""
    for _ in attempts(500, "an L-shaped plan with two sides to work out"):
        a = rng.randint(50, 95)
        b = rng.randint(20, 60)
        c = rng.randint(20, a - 20)
        d = rng.randint(b + 20, 95)
        if a - c >= 20 and d - b >= 20:
            return a, b, c, d
    raise AssertionError("unreachable")


def _l_figure(a, b, c, d):
    poly = [("A", 0, 0), ("B", a, 0), ("C", a, b), ("D", c, b), ("E", c, d), ("F", 0, d)]
    labels = {("A", "B"): _cm(a), ("B", "C"): _cm(b), ("E", "F"): _cm(c), ("F", "A"): _cm(d)}
    return _plan_figure([poly], labels)


def _hard(rng):
    for _ in attempts(2000, "an L-shaped plan with a real size of at most two decimal places"):
        k = rng.choice([50, 100, 200])
        a, b, c, d = _l_shape(rng)
        if any(_dp(_real_m(v, k)) > 1 for v in (a, b, c, d)):
            continue
        area_q = rng.random() < 0.6
        ma, mb, mc, md = (_real_m(v, k) for v in (a, b, c, d))
        if area_q:
            value = ma * md - (ma - mc) * (md - mb)
            check(value == ma * mb + mc * (md - mb), "the L-shape area differs when cut the other way")
        else:
            value = 2 * (ma + md)
        dp = _dp(value)
        answer, shown = _answer_fields(value, dp)
        fig = _l_figure(a, b, c, d)
        if not kit.leaks(fig, shown) and value >= 8:
            break
    pa, pb, pc, pd = (Fraction(v, 10) for v in (a, b, c, d))
    if area_q:
        cands = [
            (ma * md, "scale-notch-ignored"),
            (ma * mb + mc * md, "scale-overlap-counted-twice"),
            (ma * md - mc * mb, "scale-notch-misread"),
            ((pa * pb + pc * (pd - pb)) * k * k / 100, "linear-factor-on-area"),
            (pa * pd * k / 10000, "scale-area-linear-factor"),
        ]
        noun = "floor area"
        unit = "square metres"
        working = [
            "The two unmarked sides are found from the opposite sides: the notch is "
            f"{(ma - mc):g} m by {(md - mb):g} m.",
            f"Area = {ma:g} × {md:g} - {(ma - mc):g} × {(md - mb):g} = {shown} m².",
        ]
    else:
        cands = [
            (ma + mb + mc + md, "scale-labelled-sides-only"),
            (2 * (ma + mb) + 2 * (mc + md - mb), "scale-rectangles-perimeters-added"),
            (2 * (ma + md) * k / 100, "scale-units-not-converted"),
            ((pa + pd) * k / 100, "scale-perimeter-halved"),
            (2 * (ma + mb + mc), "scale-labelled-sides-only"),
        ]
        noun = "perimeter"
        unit = "metres"
        working = [
            "The two unmarked sides are found from the opposite sides, and together the sides across and up add to "
            "the same as a rectangle.",
            f"Perimeter = 2 × ({ma:g} + {md:g}) = {shown} m.",
        ]
    text = phrase(
        rng,
        (
            "{who}'s plan of a {place} is L-shaped and drawn to a scale of 1 : {k}. Two of its sides are not "
            "marked. What is the real {noun} of the {place}, in {unit}?",
            "The L-shaped plan shown is drawn to a scale of 1 : {k}, with four lengths marked. Find the real {noun} "
            "of {who}'s {place}, in {unit}.",
            "A {place} is drawn as the L-shaped plan shown, to a scale of 1 : {k}. Not every side is marked. Find "
            "its real {noun}, in {unit}.",
            "{who} has an L-shaped plan of a {place}, drawn at 1 : {k}. Work out the real {noun}, in {unit}, using "
            "the marked lengths.",
        ),
        who=rng.choice(_NAMES),
        place=rng.choice(_PLACES[k]),
        k=k,
        noun=noun,
        unit=unit,
    )
    return fig, text, answer, shown, value, dp, cands, working


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(
    "fsp_scale_plan",
    level=8,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fsp_scale_plan(rng, tier="easy"):
    """A to-scale floor plan with a number answer. See the module docstring for the three tiers."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(40, "a plan question with three distinct wrong answers"):
        fig, text, answer, shown, value, dp, cands, working = _TIERS[tier](rng)
        distractors = _distractors(value, dp, cands)
        if len(distractors) == 3:
            break
    return dict(
        text=text,
        answer=answer,
        answer_text=shown,
        answer_latex=latex_answer(shown),
        answer_kind="number",
        distractors=distractors,
        working=working,
        figure=fig,
    )


# ------------------------------------------------------------------------------------------ reader


def _measure(fig):
    """(position by letter, drawn length per marked centimetre) from the figure alone."""
    pos = {p["label"]: tuple(p["at"]) for p in fig["points"]}
    ratios = []
    for s in fig["segments"]:
        if s.get("label"):
            value = float(s["label"].split()[0])
            ratios.append(math.dist(pos[s["from"]], pos[s["to"]]) / value)
    check(ratios and max(ratios) - min(ratios) < 0.01 * max(ratios), "the marked lengths are not drawn to one scale")
    return pos, sum(ratios) / len(ratios)


def _shoelace(points):
    return abs(sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(points, points[1:] + points[:1]))) / 2


def _plain(value: float) -> str:
    text = f"{round(value, 2):.2f}".rstrip("0").rstrip(".")
    return text


def _read_plan(question) -> str:
    fig = question.figure
    pos, per_cm = _measure(fig)
    text = question.text
    k = int((re.search(r"1 : (\d+)", text) or re.search(r"stands for (\d+) cm", text)).group(1))
    metres_per_drawn = k / 100 / per_cm
    wall = re.search(r"wall ([A-F])([A-F])", text)
    room = re.search(r"room ([A-F]{4})", text)
    if wall:
        a, b = wall.group(1), wall.group(2)
        return _plain(math.dist(pos[a], pos[b]) * metres_per_drawn)
    if room:
        return _plain(_shoelace([pos[c] for c in room.group(1)]) * metres_per_drawn**2)
    corners = [pos[c] for c in "ABCDEF"]
    if "area" in text:
        return _plain(_shoelace(corners) * metres_per_drawn**2)
    sides = [math.dist(p, q) for p, q in zip(corners, corners[1:] + corners[:1])]
    return _plain(sum(sides) * metres_per_drawn)


figures.register_reader("fsp_scale_plan", _read_plan)


# ------------------------------------------------------------------------------------------ exam chain


@generator("fsp_scale_plan_exam", level=9, presets={"exam": {}})
def fsp_scale_plan_exam(rng):
    """One room on a plan: the two real wall lengths, the floor area, then the number of square tiles."""
    for _ in attempts(1000, "a room whose real sides are whole or half metres"):
        k = rng.choice([50, 100, 200])
        step = {50: 10, 100: 5, 200: 5}[k]  # plan lengths that make a whole or half metre in real life
        long10, short10 = step * rng.randint(40 // step + 1, 95 // step), step * rng.randint(25 // step + 1, 70 // step)
        if long10 <= short10 + 10:
            continue
        m_long, m_short = _real_m(long10, k), _real_m(short10, k)
        check((2 * m_long).denominator == 1 and (2 * m_short).denominator == 1, "a real side is not a half metre")
        if m_long * m_short >= 4:
            break
    fig = _single_room(long10, short10, labelled_sides=("AB", "BC"))
    area = m_long * m_short
    tile = Fraction(1, 2)  # tiles are 0.5 m square
    tiles = area / (tile * tile)
    check(tiles.denominator == 1, "the floor does not take a whole number of tiles")
    # second way: measure the drawn rectangle
    pos, per_cm = _measure(fig)
    drawn_area = _shoelace([pos[c] for c in "ABCD"])
    check(abs(drawn_area / per_cm**2 * (k / 100) ** 2 - float(area)) < 1e-6, "drawn area disagrees with the lengths")
    place = rng.choice(_PLACES[k])
    stem = phrase(
        rng,
        (
            "{who} is tiling the floor of a {place}. The plan shown is drawn to a scale of 1 : {k}, and AB and BC are "
            "marked on it.",
            "The plan of {who}'s {place}, drawn to a scale of 1 : {k}, is shown. Two lengths are marked on the plan.",
            "A {place} is drawn on the plan shown, to a scale of 1 : {k}. {who} will cover its floor with tiles.",
            "{who} has drawn the plan shown of a {place} at a scale of 1 : {k}. The marked lengths are plan "
            "lengths. The floor will be tiled.",
        ),
        who=rng.choice(_NAMES),
        place=place,
        k=k,
    )
    dp_long, dp_short = _dp(m_long), _dp(m_short)
    dp_area = _dp(area)

    def number_part(prompt, value, dp, marks, working, cands, **extra):
        answer, shown = _answer_fields(value, dp)
        return dict(
            prompt=prompt,
            answer_kind="number",
            answer=answer,
            answer_text=shown,
            answer_latex=latex_answer(shown),
            marks=marks,
            sensible_range=(0, None),
            working=working,
            distractors=_distractors(value, dp, cands),
            **extra,
        )

    p_long, p_short = Fraction(long10, 10), Fraction(short10, 10)
    parts = [
        number_part(
            "Find the real length of the longer wall AB, in metres.",
            m_long,
            dp_long,
            1,
            [f"{p_long:g} cm × {k} = {p_long * k:g} cm = {m_long:g} m."],
            [
                (p_long * k, "scale-units-not-converted"),
                (p_long * k / 1000, "scale-converted-wrong-power"),
                (m_short, "scale-wrong-wall-read"),
                (p_long * k / 10, "scale-converted-wrong-power"),
            ],
        ),
        number_part(
            "Find the real length of the shorter wall BC, in metres.",
            m_short,
            dp_short,
            1,
            [f"{p_short:g} cm × {k} = {p_short * k:g} cm = {m_short:g} m."],
            [
                (p_short * k, "scale-units-not-converted"),
                (p_short * k / 1000, "scale-converted-wrong-power"),
                (m_long, "scale-wrong-wall-read"),
                (p_short * k / 10, "scale-converted-wrong-power"),
            ],
        ),
        number_part(
            "Hence find the real floor area, in square metres.",
            area,
            dp_area,
            2,
            [f"Area = {m_long:g} × {m_short:g} = {area:g} m²."],
            [
                (2 * (m_long + m_short), "perimeter-for-area"),
                (p_long * p_short * k / 10000, "scale-area-linear-factor"),
                (p_long * p_short * k * k / 100, "linear-factor-on-area"),
                (m_long + m_short, "perimeter-for-area"),
            ],
            follow_through="part_a*part_b",
        ),
        number_part(
            "Each tile is a square of side 0.5 m. Using your floor area, how many tiles cover the floor, with no "
            "gaps and no cutting?",
            tiles,
            0,
            2,
            [f"One tile covers 0.5 × 0.5 = 0.25 m². {area:g} ÷ 0.25 = {tiles:g} tiles."],
            [
                (area / tile, "scale-tile-side-for-area"),
                (area * tile * tile, "scale-tile-area-multiplied"),
                (area, "scale-area-given-as-tiles"),
                (area * tile, "scale-tile-area-multiplied"),
            ],
            follow_through="part_c/0.25",
        ),
    ]
    check(all(len(p["distractors"]) >= 2 for p in parts), "too few distractors in a part")
    return dict(text=stem, figure=fig, parts=parts)
