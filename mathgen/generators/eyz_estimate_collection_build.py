"""Estimating the quantity of objects in a collection (AC9M3N05), from a scattered picture.

A picture of trees, houses or tents sits in the cells of a grid, one to a cell, spread at random. The learner
estimates how many there are and writes the estimate to the nearest 10. The key is graded exactly, so every
collection is built so that an honest estimate and an exact count give the same answer:

- the true count is the answer plus or minus at most 2 (it never ends in 3 to 7), so counting, or sizing up
  chunks by eye and being a little out, reaches the same multiple of 10;
- where a benchmark box is drawn, the count inside it times the share it covers also rounds to that answer.

- ``eyz_estimate_collection`` (skill ``ey-estimate-collection-quantity``, ``number``): easy shows a box that is one
  fifth of the picture and holds a few of the objects, so the learner scales it up; medium has no box and a larger
  collection; hard splits the picture by a wall and asks about both sides (the total, or how many more on the
  left), so the learner rounds each side and then combines them.
- ``eyz_estimate_collection_exam`` (exam tier): the hard picture asked as three parts, the left side, the right
  side, and both together, the last following through from the first two.

A registered reader recounts the objects from the figure alone.
"""

from __future__ import annotations

import re

from .. import figures
from ..core import attempts, check, generator
from ..figure_kinds.grid import grid_drawing
from ._spread_kit import tagged_numbers, whole_part, uses_rng

_TIERS = ("easy", "medium", "hard")
_PRESETS = {t: {"tier": t} for t in _TIERS}

# The pictures a collection can be drawn with, and what a stem calls them.
_THINGS = {
    "tree": "trees",
    "house": "houses",
    "campground": "tents",
}
_OFFSETS = (-2, -1, 0, 1, 2)
_MOST = 85  # a picture never fills more than this per cent of its cells, so estimating is not counting every cell
_SIDE_H = 11  # hard and exam: each side is 6 wide by 11 tall, room for 50 and up to 85% full
_WALL_X = 6  # hard and exam: the wall stands on this lattice line, so each side is a 6 by 11 block


def _near(ans: int, rng, *, top: int) -> int:
    """A true count within 2 of the multiple of 10 ``ans``, never above ``top`` (at most 85% of the cells)."""
    return rng.choice([ans + o for o in _OFFSETS if 0 < ans + o <= top])


def _round10(n: int) -> int:
    return (n + 5) // 10 * 10


def _pick(rng, cells: list, n: int) -> list:
    return rng.sample(cells, n)


def _figure(w: int, h: int, symbol: str, cells: list, *, box=None, wall=False) -> dict:
    polygons = []
    if box is not None:
        x0, x1 = box
        polygons.append({"vertices": [[x0, 0], [x1, 0], [x1, h], [x0, h]], "style": "solid"})
    return grid_drawing(
        alt=f"{_THINGS[symbol]} scattered over a grid, one to a square",
        grid={"type": "square", "width": w, "height": h},
        symbols=[{"at": [x + 0.5, y + 0.5], "symbol": symbol} for x, y in cells],
        polygons=polygons,
        lines=[{"points": [[_WALL_X, 0], [_WALL_X, h]], "style": "wall"}] if wall else (),
    )


def _wrongs_single(count: int, ans: int, extra=()) -> list:
    return [
        *extra,
        (count, "eyz-does-not-round-the-estimate"),
        (ans + 10, "eyz-rounds-to-the-wrong-ten"),
        (ans - 10, "eyz-rounds-to-the-wrong-ten"),
        (count // 10 * 10 if count % 10 > 5 else count // 10 * 10 + 10, "eyz-rounds-to-the-wrong-ten"),
    ]


def _tens(candidates):
    """The candidates that are whole tens: every option of an estimate to the nearest 10 is one."""
    return [(v, t) for v, t in candidates if v % 10 == 0]


def _question(text: str, ans: int, cands, working, fig) -> dict:
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=ans,
        answer_text=str(ans),
        answer_latex=str(ans),
        answer_kind="number",
        working=working,
        # Every option is a nearest-10 value, as the stem asks: a raw count would give the key away by not being one.
        distractors=tagged_numbers(ans, [(v, t) for v, t in cands if v % 10 == 0], low=1, high=150, step=10),
        figure=fig,
    )


def _easy(rng) -> dict:
    w, h = 10, 8
    symbol = rng.choice(tuple(_THINGS))
    ans = 10 * rng.randint(1, 6)
    count = _near(ans, rng, top=_MOST * w * h // 100)
    k = round(count / 5)
    check(abs(5 * k - count) <= 2 and _round10(5 * k) == ans, "the box scales to the answer")
    j = rng.randint(0, w - 2)
    box = (j, j + 2)
    inside = [(x, y) for x in range(box[0], box[1]) for y in range(h)]
    outside = [(x, y) for x in range(w) for y in range(h) if not box[0] <= x < box[1]]
    cells = _pick(rng, inside, k) + _pick(rng, outside, count - k)
    fig = _figure(w, h, symbol, cells, box=box)
    things = _THINGS[symbol]
    text = rng.choice(
        (
            f"About how many {things}? The box is one fifth. Round to the nearest 10.",
            f"The box is one fifth of the picture. About how many {things}? Nearest 10.",
        )
    )
    cands = _wrongs_single(
        count,
        ans,
        extra=[
            (_round10(k), "eyz-gives-the-box-count"),
            (_round10(4 * k), "eyz-wrong-share-of-the-whole"),
            (_round10(6 * k), "eyz-wrong-share-of-the-whole"),
        ],
    )
    working = [
        f"The box holds {k} {things}. It is one fifth, so 5 × {k} = {5 * k}.",
        f"{5 * k} is about {ans} to the nearest 10.",
    ]
    return _question(text, ans, cands, working, fig)


def _medium(rng) -> dict:
    w, h = 10, 9
    symbol = rng.choice(tuple(_THINGS))
    ans = 10 * rng.randint(2, 7)
    count = _near(ans, rng, top=_MOST * w * h // 100)
    fig = _figure(w, h, symbol, _pick(rng, [(x, y) for x in range(w) for y in range(h)], count))
    things = _THINGS[symbol]
    text = rng.choice(
        (
            f"About how many {things} are there? Estimate to the nearest 10.",
            f"Estimate how many {things} there are. Round to the nearest 10.",
        )
    )
    cands = _wrongs_single(count, ans, extra=[(count // 2 // 10 * 10 or 10, "eyz-counts-one-part-only")])
    working = [
        f"Count in tens by eye: there are about {ans} {things}.",
        f"The exact count is {count}, which is {ans} to the nearest 10.",
    ]
    return _question(text, ans, cands, working, fig)


def _two_sides(rng, *, difference: bool) -> tuple[int, int]:
    for _ in attempts(500, "two sides that round consistently"):
        if difference:  # the difference is at least 30, so a wrong ten sits on either side of it
            a_ans = 10 * rng.randint(4, 5)
            b_ans = 10 * rng.randint(1, a_ans // 10 - 3)
        else:  # sides of 20 or more, so a wrong ten below the side is a real option (10 is the lowest)
            a_ans, b_ans = 10 * rng.randint(2, 5), 10 * rng.randint(2, 5)
        a, b = (
            _near(a_ans, rng, top=_MOST * _WALL_X * _SIDE_H // 100),
            _near(b_ans, rng, top=_MOST * _WALL_X * _SIDE_H // 100),
        )
        if difference and a <= b:
            continue
        total_ok = _round10(a) + _round10(b) == _round10(a + b)
        diff_ok = _round10(a) - _round10(b) == _round10(a - b)
        if (diff_ok if difference else total_ok) and (not difference or _round10(a) - _round10(b) > 0):
            return a, b
    raise AssertionError("unreachable")


def _hard(rng) -> dict:
    w, h = 12, _SIDE_H
    symbol = rng.choice(tuple(_THINGS))
    difference = rng.random() < 0.4
    a, b = _two_sides(rng, difference=difference)
    left = [(x, y) for x in range(_WALL_X) for y in range(h)]
    right = [(x, y) for x in range(_WALL_X, w) for y in range(h)]
    cells = _pick(rng, left, a) + _pick(rng, right, b)
    fig = _figure(w, h, symbol, cells, wall=True)
    things = _THINGS[symbol]
    ra, rb = _round10(a), _round10(b)
    if difference:
        ans = ra - rb
        text = rng.choice(
            (
                f"About how many more {things} are on the left than the right? Nearest 10.",
                f"Estimate how many more {things} are on the left side than the right. Nearest 10.",
            )
        )
        cands = [
            (ra + rb, "eyz-adds-instead-of-subtracts"),
            (ra, "eyz-reports-one-side-only"),
            (rb, "eyz-reports-one-side-only"),
            (ans + 10, "eyz-rounds-to-the-wrong-ten"),
            (ans - 10, "eyz-rounds-to-the-wrong-ten"),
            (a - b, "eyz-does-not-round-the-estimate"),
        ]
        working = [f"Left is about {ra} and right is about {rb}.", f"{ra} - {rb} = {ans}."]
    else:
        ans = ra + rb
        text = rng.choice(
            (
                f"About how many {things} on both sides altogether? Round to the nearest 10.",
                f"Estimate the total number of {things} on both sides. Nearest 10.",
            )
        )
        cands = [
            (ra, "eyz-reports-one-side-only"),
            (rb, "eyz-reports-one-side-only"),
            (abs(ra - rb), "eyz-subtracts-instead-of-adds"),
            (ans + 10, "eyz-rounds-to-the-wrong-ten"),
            (ans - 10, "eyz-rounds-to-the-wrong-ten"),
            (a + b, "eyz-does-not-round-the-estimate"),
        ]
        working = [f"Left is about {ra} and right is about {rb}.", f"{ra} + {rb} = {ans}."]
    return _question(text, ans, cands, working, fig)


@generator("eyz_estimate_collection", level=3, presets=_PRESETS)
@uses_rng
def eyz_estimate_collection(rng, tier="easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)


@generator("eyz_estimate_collection_exam", level=3, presets={"exam": dict(tier="exam")})
@uses_rng
def eyz_estimate_collection_exam(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    w, h = 12, _SIDE_H
    symbol = rng.choice(tuple(_THINGS))
    a, b = _two_sides(rng, difference=False)
    left = [(x, y) for x in range(_WALL_X) for y in range(h)]
    right = [(x, y) for x in range(_WALL_X, w) for y in range(h)]
    fig = _figure(w, h, symbol, _pick(rng, left, a) + _pick(rng, right, b), wall=True)
    things = _THINGS[symbol]
    ra, rb = _round10(a), _round10(b)
    total = ra + rb
    m = rng.randint(2, 12)
    unit = {
        "tree": f"Each tree needs {m} litres of water. About how many litres in all?",
        "house": f"Each house has {m} windows. About how many windows in all?",
        "campground": f"Each tent sleeps {m} people. About how many people in all?",
    }[symbol]
    stem = rng.choice(
        (
            f"The picture shows {things}, split into two sides by a wall. Estimate to the nearest 10 each time.",
            f"A wall splits the {things} in the picture into a left side and a right side. Round to the nearest 10.",
            f"Look at the {things} on each side of the wall. Estimate each count to the nearest 10.",
        )
    )
    return dict(
        text=stem,
        figure=fig,
        parts=[
            whole_part(
                f"About how many {things} on the left side?",
                ra,
                1,
                _tens(
                    [
                        (a, "eyz-does-not-round-the-estimate"),
                        (ra + 10, "eyz-rounds-to-the-wrong-ten"),
                        (rb, "eyz-other-side"),
                        (ra - 10, "eyz-rounds-to-the-wrong-ten"),
                        (ra + 20, None),
                        (ra + 30, None),
                    ]
                ),
                step=10,
                working=[f"There are {a} on the left, about {ra} to the nearest 10."],
            ),
            whole_part(
                f"About how many {things} on the right side?",
                rb,
                1,
                _tens(
                    [
                        (b, "eyz-does-not-round-the-estimate"),
                        (rb + 10, "eyz-rounds-to-the-wrong-ten"),
                        (ra, "eyz-other-side"),
                        (rb - 10, "eyz-rounds-to-the-wrong-ten"),
                        (rb + 20, None),
                        (rb + 30, None),
                    ]
                ),
                step=10,
                working=[f"There are {b} on the right, about {rb} to the nearest 10."],
            ),
            whole_part(
                f"Hence, about how many {things} on both sides?",
                total,
                2,
                _tens(
                    [
                        (abs(ra - rb), "eyz-subtracts-instead-of-adds"),
                        (total + 10, "eyz-rounds-to-the-wrong-ten"),
                        (a + b, "eyz-does-not-round-the-estimate"),
                        (ra, "eyz-reports-one-side-only"),
                        (total - 10, "eyz-rounds-to-the-wrong-ten"),
                        (total + 20, None),
                    ]
                ),
                step=10,
                follow_through="part_a + part_b",
                working=[f"{ra} + {rb} = {total}."],
            ),
            whole_part(
                unit,
                m * total,
                3,
                [
                    (total, "eyz-forgets-to-multiply"),
                    (m * (a + b), "eyz-does-not-round-the-estimate"),
                    (m * ra, "eyz-reports-one-side-only"),
                    (m * rb, "eyz-reports-one-side-only"),
                    (m * total + m * 10, "eyz-rounds-to-the-wrong-ten"),
                    (m * total - m * 10, "eyz-rounds-to-the-wrong-ten"),
                    (m * total + m, None),
                ],
                follow_through=f"{m}*part_c",
                working=[f"{m} × {total} = {m * total}."],
            ),
        ],
    )


def _count_symbols(fig: dict) -> list[tuple[float, float]]:
    return [tuple(s["at"]) for s in fig.get("symbols", [])]


def _read_estimate(q) -> str:
    fig = q.figure
    pts = _count_symbols(fig)
    text = q.text
    wall = next((ln for ln in fig.get("lines", []) if ln["style"] == "wall"), None)
    if wall is None:
        return str(_round10(len(pts)))
    x = wall["points"][0][0]
    a = _round10(sum(1 for p in pts if p[0] < x))
    b = _round10(sum(1 for p in pts if p[0] > x))
    if q.parts:
        mult = int(
            re.search(r"\d+", q.parts[3]["prompt"] if isinstance(q.parts[3], dict) else q.parts[3].prompt).group()
        )
        return f"(a) {a}; (b) {b}; (c) {a + b}; (d) {mult * (a + b)}"
    return str(a - b if "more" in text else a + b)


figures.register_reader("eyz_estimate_collection", _read_estimate)
figures.register_reader("eyz_estimate_collection_exam", _read_estimate)
