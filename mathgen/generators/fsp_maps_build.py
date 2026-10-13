"""Generates `fsp_map_route` and `fsp_map_route_exam` questions: distances and bearings on a drawn grid map.

One skill: ``f10-map-grid-routes-bearings`` (``answer: number``). The stimulus is a map drawn with the existing ``cartesian``
figure kind: a grid whose lines are numbered (the grid reference of a place is the pair of numbers on the lines it
sits at, easting first), north up, with places marked as labelled points. Each grid square is a stated number of kilometres
across. Three tiers:

- easy: two places in the same row or column. How far apart are they? Half the questions name both places; the rest give
  one place by grid reference, so the learner has to find it on the map.
- medium: a walk of two legs, each due north, south, east or west, from place to place. How long is the whole walk?
- hard: a walk from one place to a second and on to a third. What is the bearing of the first place from the third (the
  way back, so the learner has to reverse a direction)? The first and third places are in line along one of the eight
  compass directions, so the bearing is a whole multiple of 45 degrees.

``fsp_map_route_exam`` is the exam chain on one map: the two legs of a walk (due east, then due north), the whole walk
(follows through from the two legs) and the bearing of the start from the finish.

Distances are built from the grid positions and checked by measuring the labelled points back off the figure; bearings
are checked by the direction of the vector (its angle from north, clockwise, by atan2) as well as by the eight-point
table the generator draws from.
"""

from __future__ import annotations

import math
import re

from mathgen import figures
from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, latex_answer, phrase, pick_distractors
from mathgen.figures import cartesian
from . import _fsp_kit as kit

_NAMES = ("Mia", "Tom", "Aiko", "Dev", "Leila", "Sam", "Jess", "Kofi", "Priya", "Marcus", "Zoe", "Ben")
_PLACES = (
    "Camp",
    "Lake",
    "Hut",
    "Hill",
    "Pier",
    "Fort",
    "Dam",
    "Mill",
    "Farm",
    "Cave",
    "Inn",
    "Falls",
    "Lodge",
    "Tower",
    "Quarry",
    "Beach",
    "Ridge",
    "Spring",
)
_ALT = "A map drawn on a numbered grid with north at the top and places marked by dots and names."
_SCALES = (2, 5, 10, 20)
_DIRS = {"east": (1, 0), "west": (-1, 0), "north": (0, 1), "south": (0, -1)}
_OPPOSITE = {"east": "west", "west": "east", "north": "south", "south": "north"}
_EIGHT = {  # direction vector -> bearing, clockwise from north
    (0, 1): 0,
    (1, 1): 45,
    (1, 0): 90,
    (1, -1): 135,
    (0, -1): 180,
    (-1, -1): 225,
    (-1, 0): 270,
    (-1, 1): 315,
}


def bearing(dx: int, dy: int) -> int:
    """The bearing of the vector (dx, dy), clockwise from north, to the nearest degree (from atan2)."""
    return round(math.degrees(math.atan2(dx, dy))) % 360


def map_figure(places: dict[str, tuple[int, int]], width: int, height: int):
    return cartesian(
        x_range=(0, width),
        y_range=(0, height),
        alt=_ALT,
        grid=1,
        equal_scale=True,
        x_label="East",
        y_label="North",
        points=[{"at": [x, y], "label": name} for name, (x, y) in places.items()],
    )


def _layout(rng, count):
    """(width, height, [count distinct lattice points not on the edge of the map])."""
    width, height = rng.choice([(8, 6), (9, 6), (10, 7), (9, 7), (10, 8)])
    pts = []
    for _ in attempts(500, "distinct places on the map"):
        p = (rng.randint(1, width - 1), rng.randint(1, height - 1))
        if p not in pts:
            pts.append(p)
        if len(pts) == count:
            return width, height, pts
    raise AssertionError("unreachable")


def _names(rng, count):
    return rng.sample(_PLACES, count)


def _inside(p, width, height):
    return 1 <= p[0] <= width - 1 and 1 <= p[1] <= height - 1


def _dir_word(dx, dy):
    return next(name for name, d in _DIRS.items() if d == ((dx > 0) - (dx < 0), (dy > 0) - (dy < 0)))


# ------------------------------------------------------------------------------------------ tiers


def _easy(rng):
    for _ in attempts(500, "two places in line with a clear gap"):
        width, height, pts = _layout(rng, 4)
        a, b = pts[0], pts[1]
        decoys = pts[2:]
        # put b in line with a, in a random compass direction, a few squares away
        name = rng.choice(list(_DIRS))
        dx, dy = _DIRS[name]
        steps = rng.randint(2, 7)
        b = (a[0] + dx * steps, a[1] + dy * steps)
        if not _inside(b, width, height) or b in decoys:
            continue
        # no decoy shares the row or column of a with a gap that reads like b
        break
    s = rng.choice(_SCALES)
    names = _names(rng, 4)
    places = dict(zip(names, [a, b, *decoys]))
    fig = map_figure(places, width, height)
    distance = steps * s
    by_ref = rng.random() < 0.5
    target = f"the {names[1]}" if not by_ref else f"the place at grid reference ({b[0]}, {b[1]})"
    nearest = min(decoys, key=lambda p: abs(p[0] - a[0]) + abs(p[1] - a[1]))
    squares_to_decoy = abs(nearest[0] - a[0]) + abs(nearest[1] - a[1])
    candidates = [
        Distractor(str(steps), "map-squares-not-scaled", steps),
        Distractor(str((steps + 1) * s), "map-off-by-one-square", (steps + 1) * s),
        Distractor(str((steps - 1) * s), "map-off-by-one-square", (steps - 1) * s),
        Distractor(str(steps + s), "map-scale-added", steps + s),
        Distractor(str(squares_to_decoy * s), "map-wrong-place-read", squares_to_decoy * s),
    ]
    text = phrase(
        rng,
        (
            "{who} is reading the map shown. {target_cap} is directly {name} of the {a}. Each grid square is {s} km "
            "across. How far is it from the {a}, in kilometres?",
            "On the map shown, each grid square is {s} km across. {target_cap} is directly {name} of the {a}, "
            "which {who} has marked. How many kilometres apart are they?",
            "{who} plans a trip using the map shown, where each grid square is {s} km across. How far is it from "
            "the {a} to {target}, which lies directly {name} of the {a}?",
            "Each grid square on the map shown is {s} km across. {who} walks in a straight line from the {a} to "
            "{target}, directly {name} of it. How far does {who} walk, in kilometres?",
        ),
        who=rng.choice(_NAMES),
        target=target,
        target_cap=target[0].upper() + target[1:],
        name=name,
        a=names[0],
        s=s,
    )
    working = [
        f"Count the grid squares between the two places: {steps}.",
        f"Each square is {s} km, so the distance is {steps} × {s} = {distance} km.",
    ]
    return fig, text, distance, candidates, working


def _medium(rng):
    for _ in attempts(1000, "a two-leg walk along grid lines"):
        width, height, pts = _layout(rng, 4)
        a = pts[0]
        decoys = pts[2:]
        first = rng.choice(list(_DIRS))
        second = rng.choice([d for d in _DIRS if d not in (first, _OPPOSITE[first])])
        n1, n2 = rng.randint(2, 6), rng.randint(2, 5)
        b = (a[0] + _DIRS[first][0] * n1, a[1] + _DIRS[first][1] * n1)
        c = (b[0] + _DIRS[second][0] * n2, b[1] + _DIRS[second][1] * n2)
        if _inside(b, width, height) and _inside(c, width, height) and not ({b, c} & set(decoys)):
            break
    s = rng.choice(_SCALES)
    names = _names(rng, 4)
    places = dict(zip(names, [a, b, c, *decoys[:1]]))
    fig = map_figure(places, width, height)
    total = (n1 + n2) * s
    candidates = [
        Distractor(str(n1 * s), "map-route-one-leg", n1 * s),
        Distractor(str(n2 * s), "map-route-one-leg", n2 * s),
        Distractor(str(abs(n1 - n2) * s), "map-legs-subtracted", abs(n1 - n2) * s),
        Distractor(str(n1 + n2), "map-squares-not-scaled", n1 + n2),
        Distractor(str((n1 + n2 + 2) * s), "map-off-by-one-square", (n1 + n2 + 2) * s),
    ]
    text = phrase(
        rng,
        (
            "{who} walks from the {a} to the {b} and then to the {c} on the map shown, each time in a straight line "
            "due north, south, east or west. Each grid square is {s} km across. How far does {who} walk altogether, "
            "in kilometres?",
            "Each grid square on the map shown is {s} km across. A hike goes from the {a} to the {b}, then on to the "
            "{c}, along straight lines running north, south, east or west. How long is the hike, in kilometres?",
            "{who}'s route on the map shown starts at the {a}, goes straight to the {b} and then straight to the "
            "{c}. Each leg runs due north, south, east or west, and each grid square is {s} km across. What is the "
            "total length of the route, in kilometres?",
            "On the map shown, a grid square is {s} km across. {who} travels from the {a} to the {b} and on to the "
            "{c}, each leg in a straight line along the grid. Find the total distance travelled, in kilometres.",
        ),
        who=rng.choice(_NAMES),
        a=names[0],
        b=names[1],
        c=names[2],
        s=s,
    )
    working = [
        f"First leg: {n1} squares. Second leg: {n2} squares.",
        f"Total = ({n1} + {n2}) × {s} = {total} km.",
    ]
    return fig, text, total, candidates, working


def _hard(rng):
    for _ in attempts(2000, "a walk whose way back is along a compass direction"):
        width, height, pts = _layout(rng, 4)
        direction = rng.choice(list(_EIGHT))
        t = rng.randint(2, 5)
        c = pts[0]
        a = (c[0] + direction[0] * t, c[1] + direction[1] * t)  # a lies in `direction` from c
        b = pts[1]
        decoys = pts[2:]
        if not _inside(a, width, height) or len({a, b, c}) < 3 or set(decoys) & {a, b, c}:
            continue
        # the middle place must not sit on the line a-c, or the walk doubles back
        if (b[0] - a[0]) * (c[1] - a[1]) == (b[1] - a[1]) * (c[0] - a[0]):
            continue
        break
    names = _names(rng, 4)
    places = dict(zip(names, [a, b, c, *decoys[:1]]))
    fig = map_figure(places, width, height)
    # second way: the angle of the vector from c to a, clockwise from north
    check(bearing(a[0] - c[0], a[1] - c[1]) == _EIGHT[direction], "atan2 bearing differs from the compass table")
    # the way back: bearing of a seen from c is the key; the bearing of c seen from a is the opposite one
    key = bearing(a[0] - c[0], a[1] - c[1])
    forward = bearing(c[0] - a[0], c[1] - a[1])
    candidates = [
        Distractor(str(forward), "map-bearing-not-reversed", forward),
        Distractor(str((360 - key) % 360), "map-bearing-anticlockwise", (360 - key) % 360),
        Distractor(str((90 - key) % 360), "map-bearing-from-east", (90 - key) % 360),
        Distractor(str((forward + 90) % 360), "map-bearing-reversed-wrong", (forward + 90) % 360),
        Distractor(str((key + 45) % 360), "map-bearing-reversed-wrong", (key + 45) % 360),
        Distractor(str((key + 315) % 360), "map-bearing-reversed-wrong", (key + 315) % 360),
    ]
    text = phrase(
        rng,
        (
            "{who} walks from the {a} to the {b} and then to the {c} on the map shown. What is the bearing of the "
            "{a} from the {c}? Give the bearing in degrees, measured clockwise from north.",
            "On the map shown, a hike goes from the {a} to the {b} and on to the {c}. {who} wants to head straight "
            "back. What is the bearing of the {a} from the {c}, in degrees measured clockwise from north?",
            "{who} travels from the {a} via the {b} to the {c} on the map shown. Find the bearing of the {a} from "
            "the {c}, in degrees clockwise from north.",
            "The map shown has north at the top. {who}'s route is {a}, then {b}, then {c}. What is the bearing of "
            "the {a} from the {c}? Measure the bearing clockwise from north, in degrees.",
        ),
        who=rng.choice(_NAMES),
        a=names[0],
        b=names[1],
        c=names[2],
    )
    working = [
        f"The {names[0]} is {t} squares {_compass(direction)} of the {names[2]}.",
        f"Measured clockwise from north, that direction is {key} degrees.",
    ]
    return fig, text, key, candidates, working


def _compass(direction):
    names = {
        (0, 1): "north",
        (1, 1): "north-east",
        (1, 0): "east",
        (1, -1): "south-east",
        (0, -1): "south",
        (-1, -1): "south-west",
        (-1, 0): "west",
        (-1, 1): "north-west",
    }
    return names[direction]


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(
    "fsp_map_route",
    level=8,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fsp_map_route(rng, tier="easy"):
    """A grid map with a number answer. See the module docstring for the three tiers."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(40, "a map question with three distinct wrong answers"):
        fig, text, answer, candidates, working = _TIERS[tier](rng)
        key = str(answer)
        floor = 0 if tier == "hard" else 1
        distractors = pick_distractors(key, [c for c in candidates if int(c.text) >= floor], 3)
        if len(distractors) == 3:
            break
    extra = {"answer_alt": 360} if tier == "hard" and answer == 0 else {}
    return dict(
        text=text,
        answer=answer,
        answer_text=key,
        answer_latex=latex_answer(key),
        answer_kind="number",
        distractors=distractors,
        working=working,
        figure=fig,
        **extra,
    )


# ------------------------------------------------------------------------------------------ reader


def map_places(fig):
    return {p["label"]: (int(p["at"][0]), int(p["at"][1])) for p in fig["points"]}


def _read_map(question) -> str:
    """Re-derive the key from the places drawn and the question's words."""
    places = map_places(question.figure)
    text = question.text
    if "bearing" in text:
        asked = re.search(r"bearing of the (\w+) from the (\w+)", text)
        a, c = places[asked.group(1)], places[asked.group(2)]
        return str(bearing(a[0] - c[0], a[1] - c[1]))
    s = int(re.search(r"(\d+) km", text).group(1))
    order = sorted((m.start(), name) for name in places for m in re.finditer(rf"\b{name}\b", text))
    named = [name for _pos, name in order]
    ref = re.search(r"grid reference \((\d+), (\d+)\)", text)
    if ref:
        target = (int(ref.group(1)), int(ref.group(2)))
        start = places[named[0]]
        return str((abs(target[0] - start[0]) + abs(target[1] - start[1])) * s)
    walk = [places[n] for n in dict.fromkeys(named)]
    if len(walk) >= 3:
        legs = zip(walk, walk[1:])
        return str(sum(abs(q[0] - p[0]) + abs(q[1] - p[1]) for p, q in legs) * s)
    a, b = walk[0], walk[1]
    return str((abs(b[0] - a[0]) + abs(b[1] - a[1])) * s)


figures.register_reader("fsp_map_route", _read_map)


# ------------------------------------------------------------------------------------------ exam chain


@generator("fsp_map_route_exam", level=9, presets={"exam": {}})
def fsp_map_route_exam(rng):
    """A walk due east and then due north on a grid map: each leg, the whole walk, then the bearing of the start from
    the finish."""
    for _ in attempts(1000, "a walk east then north with equal legs' squares apart from the decoys"):
        width, height, pts = _layout(rng, 4)
        a = pts[0]
        n1 = rng.randint(2, 6)
        n2 = n1  # equal legs make the straight line back a diagonal, so its bearing is a whole 45 degrees
        b = (a[0] + n1, a[1])
        c = (b[0], b[1] + n2)
        decoys = pts[2:]
        if _inside(b, width, height) and _inside(c, width, height) and not ({b, c} & set(decoys)):
            break
    s = rng.choice(_SCALES)
    names = _names(rng, 4)
    fig = map_figure(dict(zip(names, [a, b, c, decoys[0]])), width, height)
    leg = n1 * s
    total = 2 * leg
    back = bearing(a[0] - c[0], a[1] - c[1])
    check(back == 225, "the start is not south-west of the finish")
    stem = phrase(
        rng,
        (
            "{who} walks from the {a} due east to the {b}, then due north to the {c}, on the map shown. Each grid "
            "square is {s} km across.",
            "On the map shown, each grid square is {s} km across. {who}'s walk goes from the {a} due east to the "
            "{b} and then due north to the {c}.",
            "The map shown has grid squares {s} km across. {who} sets out from the {a}, walks due east to the {b} "
            "and finishes by walking due north to the {c}.",
            "{who} hikes from the {a} to the {b}, due east, and on to the {c}, due north. Each grid square on the "
            "map shown is {s} km across.",
        ),
        who=rng.choice(_NAMES),
        a=names[0],
        b=names[1],
        c=names[2],
        s=s,
    )

    def part(prompt, value, marks, working, candidates, **extra):
        text = str(value)
        return dict(
            prompt=prompt,
            answer_kind="number",
            answer=value,
            answer_text=text,
            answer_latex=latex_answer(text),
            marks=marks,
            sensible_range=(0, None),
            working=working,
            distractors=pick_distractors(text, [c for c in candidates if int(c.text) > 0], 3),
            **extra,
        )

    parts = [
        part(
            f"How far is the {names[1]} from the {names[0]}, in kilometres?",
            leg,
            1,
            [f"{n1} squares × {s} km = {leg} km."],
            [
                Distractor(str(n1), "map-squares-not-scaled", n1),
                Distractor(str((n1 + 1) * s), "map-off-by-one-square", (n1 + 1) * s),
                Distractor(str(n1 + s), "map-scale-added", n1 + s),
                Distractor(str((n1 - 1) * s), "map-off-by-one-square", (n1 - 1) * s),
                Distractor(str(n1 * s + 1), "map-scale-added", n1 * s + 1),
            ],
        ),
        part(
            f"How far is the {names[2]} from the {names[1]}, in kilometres?",
            leg,
            1,
            [f"{n2} squares × {s} km = {leg} km."],
            [
                Distractor(str(n2), "map-squares-not-scaled", n2),
                Distractor(str((n2 - 1) * s), "map-off-by-one-square", (n2 - 1) * s),
                Distractor(str(n2 + s), "map-scale-added", n2 + s),
                Distractor(str((n2 + 1) * s), "map-off-by-one-square", (n2 + 1) * s),
                Distractor(str(n2 * s + 1), "map-scale-added", n2 * s + 1),
            ],
        ),
        part(
            "Using your answers to parts (a) and (b), find the total length of the walk, in kilometres.",
            total,
            1,
            [f"{leg} + {leg} = {total} km."],
            [
                Distractor(str(leg), "map-route-one-leg", leg),
                Distractor(str(total + 2 * s), "map-off-by-one-square", total + 2 * s),
                Distractor(str(2 * n1), "map-squares-not-scaled", 2 * n1),
                Distractor(str(total - s), "map-off-by-one-square", total - s),
            ],
            follow_through="part_a + part_b",
        ),
        part(
            f"What is the bearing of the {names[0]} from the {names[2]}? Give the bearing in degrees, measured "
            "clockwise from north.",
            back,
            2,
            [
                f"The {names[0]} is as far south as it is west of the {names[2]}, so it lies south-west of it.",
                "South-west is 225 degrees, measured clockwise from north.",
            ],
            [
                Distractor("45", "map-bearing-not-reversed", 45),
                Distractor("135", "map-bearing-reversed-wrong", 135),
                Distractor("315", "map-bearing-anticlockwise", 315),
            ],
        ),
    ]
    check(all(len(p["distractors"]) >= 2 for p in parts), "too few distractors in a part")
    return dict(text=stem, figure=fig, parts=parts)
