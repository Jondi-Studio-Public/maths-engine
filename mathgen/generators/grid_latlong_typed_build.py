"""Typed latitude and longitude on a map grid (``grid_latlong_typed``, senior, ``pair``).

The same map-grid figure as ``grid_latlong_places`` (`grid_shapes_y36_build._map_figure`: a line every 10 degrees,
latitude up the left side, longitude along the top, places marked with letters), but the learner TYPES the position
instead of choosing it. A position is a ``pair`` written ``(latitude, longitude)`` in signed degrees, and every stem
states the convention: north and east positive, south and west negative. The pair grader reads numbers, not N/S/E/W
letters, so the sign carries the hemisphere.

Easy reads a marked place. Medium defines a place from two others (the latitude of A and the longitude of B), so
each coordinate has to be read from a different place. Hard finds the place halfway between two others, or the place
a stated number of degrees north or south and east or west of one, on a map that crosses the equator or the prime
meridian so the signs matter.

The figure reader re-derives the answer from the figure and the wording alone.
"""

from __future__ import annotations

import re
from itertools import combinations

from .. import figures
from ..core import attempts, generator, paren_negative, phrase
from . import _sr2l_kit as kit
from .grid_shapes_y36_build import _lat, _lon, _map_figure, _read_map, _spots, _tier

_CONVENTION = "Answer as (latitude, longitude) in degrees: north and east positive, south and west negative."
_W, _H = 6, 6


def _origin(rng, n: int, crossing: bool, limit: int) -> int:
    """The lowest line of a map ``n`` squares across: through zero when ``crossing``, else wholly on one side."""
    if crossing:
        return -10 * rng.randint(1, n - 1)
    room = limit // 10 - n
    if rng.random() < 0.5:
        return 10 * rng.randint(0, room)
    return -10 * n - 10 * rng.randint(0, room)


def _fields(key, groups, rng):
    """The pair fields and tagged wrong options for ``key``, from ``groups`` of ``(latitude, longitude, tag)``.

    The groups are, in order, the slips that keep the key's latitude, those that keep its longitude, and the rest, so
    the key always has company in each coordinate. One random member of each group leads, then the others follow."""
    fields = kit.pair_fields(*key)
    heads = [rng.choice(g) for g in groups]
    rest = [c for g in groups for c in g if c not in heads]
    rng.shuffle(rest)  # every slip gets its turn as one of the three wrong options
    valid = kit.tagged("pair", fields, kit.pair_cands(heads + rest), limit=len(heads) + len(rest))
    for trio in combinations(valid, 3):
        if _no_lone_look(fields["answer_text"], [d.text for d in trio]):
            return fields, list(trio)
    return fields, valid[:3]


def _look(text: str) -> tuple[int, bool]:
    """What a learner can see of an option without reading the map: its digit count and whether it has a minus sign."""
    return len(re.findall(r"\d", text)), "-" in text


def _no_lone_look(key: str, wrong: list[str]) -> bool:
    """No option, the key or a wrong one, is the only one with its digit count or the only one with (or without) a minus."""
    looks = [_look(t) for t in (key, *wrong)]
    return all(sum(look[i] == other[i] for other in looks) > 1 for look in looks for i in (0, 1))


def _question(fig, text, key, groups, working, rng):
    if figures.validate(fig):
        raise ValueError(f"invalid figure: {figures.validate(fig)}")
    fields, distractors = _fields(key, groups, rng)
    return dict(text=text, working=working, figure=fig, distractors=distractors, **fields)


def _place(la: int, lo: int) -> str:
    return f"{_lat(la)}, {_lon(lo)}"


# ------------------------------------------------------------------------------------------------------ easy


def _easy(rng):
    la_sign, lo_sign = rng.choice(((-1, 1), (1, 1), (1, -1), (-1, -1)))
    lat0 = -10 * _H - 10 * rng.randint(0, 2) if la_sign < 0 else 10 * rng.randint(0, 2)
    lon0 = 10 * rng.randint(0, 11) if lo_sign > 0 else -10 * _W - 10 * rng.randint(0, 11)
    places = dict(zip("ABC", _spots(rng, lat0, lon0, _W, _H, 3)))
    ask = rng.choice("ABC")
    la, lo = places[ask]
    fig = _map_figure(lat0, lon0, _W, _H, places)
    text = phrase(
        rng,
        (
            "Each square on the map grid is 10 degrees. What is the position of place {p}? " + _CONVENTION,
            "A map has a line every 10 degrees. Type the latitude and longitude of place {p}. " + _CONVENTION,
            "Find place {p} on the map grid. Which latitude and longitude is it at? " + _CONVENTION,
        ),
        p=ask,
    )
    cands = [
        [
            (-la, lo, "llt-wrong-hemisphere-latitude"),
            (la + 10, lo, "llt-one-line-out"),
            (la - 10, lo, "llt-one-line-out"),
        ],
        [(la, -lo, "llt-wrong-side-of-meridian"), (la, lo + 10, "llt-one-line-out"), (la, lo - 10, "llt-one-line-out")],
        [(lo, la, "llt-coordinates-swapped")],
    ]
    working = [
        "Latitude is read up the side of the map and longitude along the top. South and west are negative.",
        f"Place {ask} is on the {_lat(la)} line and the {_lon(lo)} line, so its position is ({la}, {lo}).",
    ]
    return _question(fig, text, (la, lo), cands, working, rng)


# ------------------------------------------------------------------------------------------------------ medium


def _medium(rng):
    lat0 = _origin(rng, _H, rng.random() < 0.6, 90)
    lon0 = _origin(rng, _W, rng.random() < 0.6, 180)
    (la_a, lo_a), (la_b, lo_b) = _spots(rng, lat0, lon0, _W, _H, 2)
    places = {"A": (la_a, lo_a), "B": (la_b, lo_b)}
    fig = _map_figure(lat0, lon0, _W, _H, places)
    first, second = rng.choice((("A", "B"), ("B", "A")))
    (la, _), (_, lo) = places[first], places[second]
    text = phrase(
        rng,
        (
            "Each square on the map grid is 10 degrees. A new place is on the same line of latitude as place {f} and "
            "the same line of longitude as place {s}. Where is it? " + _CONVENTION,
            "A map has a line every 10 degrees. Give the position of the place with the latitude of {f} and the "
            "longitude of {s}. " + _CONVENTION,
            "Place {f} and place {s} are marked on the map grid. What is the position of the place level with {f} "
            "east to west and in line with {s} north to south? " + _CONVENTION,
        ),
        f=first,
        s=second,
    )
    la_f, lo_f = places[first]
    la_s, lo_s = places[second]
    cands = [
        [(la_f, lo_f, "llt-copies-one-place"), (la, -lo, "llt-wrong-side-of-meridian")],
        [(la_s, lo_s, "llt-copies-one-place"), (-la, lo, "llt-wrong-hemisphere-latitude")],
        [(la_s, lo_f, "llt-roles-swapped"), (lo, la, "llt-coordinates-swapped")],
    ]
    working = [
        f"Place {first} is at latitude {la_f}, so the new place is on the {_lat(la_f)} line. "
        f"Place {second} is at longitude {lo_s}, so it is on the {_lon(lo_s)} line.",
        f"The new place is at ({la}, {lo}).",
    ]
    return _question(fig, text, (la, lo), cands, working, rng)


# ------------------------------------------------------------------------------------------------------ hard


def _same_parity_pair(rng, n: int):
    """Two different lines of an ``n``-square map (1 to n - 1) whose positions add to an even number of squares."""
    for _ in attempts(100, "two lines with a line between them"):
        a, b = rng.sample(range(1, n), 2)
        if (a + b) % 2 == 0:
            return a, b


def _midway(rng):
    lat0 = _origin(rng, _H, rng.random() < 0.6, 90)
    lon0 = _origin(rng, _W, rng.random() < 0.6, 180)
    ya, yb = _same_parity_pair(rng, _H)
    xa, xb = _same_parity_pair(rng, _W)
    a, b = (lat0 + 10 * ya, lon0 + 10 * xa), (lat0 + 10 * yb, lon0 + 10 * xb)
    fig = _map_figure(lat0, lon0, _W, _H, {"A": a, "B": b})
    text = phrase(
        rng,
        (
            "Each square on the map grid is 10 degrees. What is the position of the place halfway between place A "
            "and place B? " + _CONVENTION,
            "Places A and B are marked on a map grid with a line every 10 degrees. Find the place midway between "
            "them. " + _CONVENTION,
            "A map has a line every 10 degrees. Give the latitude and longitude of the midpoint of A and B. "
            + _CONVENTION,
        ),
    )
    mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
    size = lambda u, v: (abs(u) + abs(v)) // 2  # noqa: E731
    cands = [
        [(mid[0], a[1], "llt-one-coordinate-halved"), (mid[0], b[1], "llt-one-coordinate-halved")],
        [(a[0], mid[1], "llt-one-coordinate-halved"), (b[0], mid[1], "llt-one-coordinate-halved")],
        [
            (a[0] + b[0], a[1] + b[1], "llt-sum-not-halved"),
            (size(a[0], b[0]), size(a[1], b[1]), "llt-averages-sizes-across-zero"),
            (abs(a[0] - b[0]) // 2, abs(a[1] - b[1]) // 2, "llt-half-the-difference"),
            (mid[1], mid[0], "llt-coordinates-swapped"),
        ],
    ]
    working = [
        f"A is at ({a[0]}, {a[1]}) and B is at ({b[0]}, {b[1]}).",
        f"Halfway: latitude ({a[0]} + {paren_negative(b[0])}) / 2 = {mid[0]}, "
        f"longitude ({a[1]} + {paren_negative(b[1])}) / 2 = {mid[1]}.",
    ]
    return _question(fig, text, mid, cands, working, rng)


def _away(rng):
    for _ in attempts(200, "a start and a finish on the map"):
        lat0 = _origin(rng, _H, rng.random() < 0.7, 90)
        lon0 = _origin(rng, _W, rng.random() < 0.7, 180)
        (la, lo) = _spots(rng, lat0, lon0, _W, _H, 1)[0]
        dy, dx = 10 * rng.randint(1, 5), 10 * rng.randint(1, 5)
        north, east = rng.random() < 0.5, rng.random() < 0.5
        end = (la + dy if north else la - dy, lo + dx if east else lo - dx)
        crosses_lat, crosses_lon = la * end[0] < 0, lo * end[1] < 0
        if not (crosses_lat or crosses_lon) or abs(end[0]) > 90 or abs(end[1]) > 180:
            continue
        break
    fig = _map_figure(lat0, lon0, _W, _H, {"A": (la, lo)})
    ns, ew = ("north" if north else "south"), ("east" if east else "west")
    text = phrase(
        rng,
        (
            "Each square on the map grid is 10 degrees. Place B is {dy} degrees {ns} and {dx} degrees {ew} of place "
            "A. What is the position of B? " + _CONVENTION,
            "A map has a line every 10 degrees. Starting from place A, go {dy} degrees {ns} and {dx} degrees {ew} to "
            "reach place B. Give the position of B. " + _CONVENTION,
            "Place B lies {dx} degrees {ew} of place A and {dy} degrees {ns} of it. Find the latitude and longitude "
            "of B. " + _CONVENTION,
        ),
        dy=dy,
        dx=dx,
        ns=ns,
        ew=ew,
    )
    toward_zero = lambda v, d: (1 if v > 0 else -1) * abs(abs(v) - d)  # noqa: E731
    no_crossing = (toward_zero(la, dy) if crosses_lat else end[0], toward_zero(lo, dx) if crosses_lon else end[1])
    cands = [
        [(end[0], lo, "llt-moves-one-way-only"), (end[0], no_crossing[1], "llt-no-crossing-zero")],
        [(la, end[1], "llt-moves-one-way-only"), (no_crossing[0], end[1], "llt-no-crossing-zero")],
        [
            (la - dy if north else la + dy, lo - dx if east else lo + dx, "llt-direction-reversed"),
            (*no_crossing, "llt-no-crossing-zero"),
            (end[1], end[0], "llt-coordinates-swapped"),
            (la + (dx if east else -dx), lo + (dy if north else -dy), "llt-offsets-on-wrong-axis"),
        ],
    ]
    working = [
        f"A is at ({la}, {lo}). {dy} degrees {ns} changes the latitude by {'+' if north else '-'}{dy}, and {dx} degrees "
        f"{ew} changes the longitude by {'+' if east else '-'}{dx}.",
        f"Latitude {la} {'+' if north else '-'} {dy} = {end[0]}; longitude {lo} {'+' if east else '-'} {dx} = {end[1]}.",
    ]
    return _question(fig, text, end, cands, working, rng)


def _hard(rng):
    return (_midway if rng.random() < 0.5 else _away)(rng)


# ------------------------------------------------------------------------------------------------------ topic

_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator("grid_latlong_typed", level=11, presets={t: dict(tier=t) for t in _TIERS})
def grid_latlong_typed(rng, tier="easy"):
    """Type the (latitude, longitude) of a place read off a map grid, defined from two others, or found by a move."""
    build = _tier(_TIERS, tier)
    return kit.redraw(rng, build)


# ------------------------------------------------------------------------------------------------------ reader

_SIGNED = r"(?:\d+|\d+\.\d+)"


def _read_typed(question) -> str:
    places = _read_map(question.figure)
    text = question.text
    if "halfway between" in text or "midway between" in text or "midpoint of A and B" in text:
        (la1, lo1), (la2, lo2) = places["A"], places["B"]
        return kit.pair_text((la1 + la2) // 2, (lo1 + lo2) // 2)
    if (m := re.search(r"(\d+) degrees (north|south)", text)) is not None:
        dy, ns = int(m.group(1)), m.group(2)
        dx, ew = re.search(r"(\d+) degrees (east|west)", text).groups()
        la, lo = places["A"]
        return kit.pair_text(la + dy if ns == "north" else la - dy, lo + int(dx) if ew == "east" else lo - int(dx))
    if (
        m := re.search(
            r"latitude of (\w) and the longitude of (\w)|latitude as place (\w) and the same line of longitude as place (\w)",
            text,
        )
    ) is not None:
        f, s = [g for g in m.groups() if g]
        return kit.pair_text(places[f][0], places[s][1])
    if (m := re.search(r"level with (\w) east to west and in line with (\w)", text)) is not None:
        return kit.pair_text(places[m.group(1)][0], places[m.group(2)][1])
    letter = re.search(r"place ([ABC])|Find place ([ABC])", text)
    la, lo = places[next(g for g in letter.groups() if g)]
    return kit.pair_text(la, lo)


figures.register_reader("grid_latlong_typed", _read_typed)
