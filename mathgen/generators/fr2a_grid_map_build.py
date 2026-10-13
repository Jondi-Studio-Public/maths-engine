"""``fr2a_grid_map``: a map of a familiar place on a lettered-column, numbered-row grid.

Skill ``f10-grid-map-familiar-place`` (AC9M3SP02 and the reading half of AC9M5SP02). A map of a school, a park, a zoo
or a town centre is drawn on a grid. Columns are lettered along the bottom (A = 1, B = 2, ...), rows are numbered up
the left side, and north is at the top. Each landmark sits in one square and is named in it. Stems stay short.

Tiers:

- easy:   the grid reference of a named landmark, as ``column = _, row = _``.
- medium: start at a named landmark and follow two moves (east or west, north or south) to a square; give its grid
          reference.
- hard:   walk from one landmark to a second and on to a third: how far east and north is the end from the start?
          The legs partly cancel (one of them goes back the way the other came), so adding the moves is not enough.
          Answer ``east = _, north = _`` with a negative number for west or south.

Constructing a grid for a map is a different task and is not built here.
"""

from __future__ import annotations

import re

from mathgen import figures
from mathgen.answers import Distractor
from mathgen.core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    generator,
    latex_answer,
    paren_negative,
    phrase,
    pick_distractors,
)
from mathgen.figures import register_reader

from . import _fr2a_kit as kit
from .exd_x_sc3_support import redraw

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_THEMES = {
    "the school grounds": [
        "Library",
        "Oval",
        "Canteen",
        "Hall",
        "Office",
        "Gate",
        "Garden",
        "Pool",
        "Art room",
        "Courts",
    ],
    "the park": [
        "Playground",
        "Lake",
        "Kiosk",
        "Toilets",
        "Car park",
        "Picnic area",
        "Skate park",
        "Fountain",
        "Gate",
        "Bridge",
    ],
    "the zoo": ["Penguins", "Koalas", "Kiosk", "Aviary", "Reptiles", "Gate", "Lions", "Farm", "Shop", "Pond"],
    "the town centre": [
        "Library",
        "Post office",
        "Bakery",
        "Station",
        "School",
        "Pool",
        "Market",
        "Museum",
        "Cafe",
        "Park",
    ],
}
_WHO = ("Mia", "Sam", "Aiko", "Dev", "Leila", "Tom", "Priya", "Ben", "Zoe", "Kofi")
_STEP = {"east": (1, 0), "west": (-1, 0), "north": (0, 1), "south": (0, -1)}


def _squares(n: int) -> str:
    return f"{n} square" if n == 1 else f"{n} squares"


def _ref(col, row):
    return f"{_LETTERS[col - 1]}{row}"


def _map_figure(theme, cols, rows, places):
    """The grid map: column letters under it, row numbers beside it, each landmark named in its square."""
    points = [{"at": [c - 0.5, -0.5], "label": _LETTERS[c - 1]} for c in range(1, cols + 1)]
    points += [{"at": [-0.5, r - 0.5], "label": str(r)} for r in range(1, rows + 1)]
    points += [{"at": [c - 0.5, r - 0.5], "label": name} for name, (c, r) in places.items()]
    return figures.cartesian(
        x_range=(-1, cols),
        y_range=(-1, rows),
        alt=f"A map of {theme} on a grid with lettered columns and numbered rows, each place named in its square.",
        grid=1,
        equal_scale=True,
        x_label="",
        y_label="",
        tick_labels=False,
        points=points,
        title=f"Map of {theme} (north is at the top)",
    )


def _layout(rng, cols, rows, how_many):
    """A theme and distinct squares for ``how_many`` landmarks, no two in the same column or row."""
    theme = rng.choice(sorted(_THEMES))
    names = rng.sample(_THEMES[theme], how_many)
    for _ in attempts(500, "landmarks in separate columns and rows"):
        cs = rng.sample(range(1, cols + 1), how_many)
        rs = rng.sample(range(1, rows + 1), how_many) if how_many <= rows else None
        if rs is None:
            rs = [rng.randint(1, rows) for _ in range(how_many)]
            if len(set(rs)) < how_many:
                continue
        break
    return theme, dict(zip(names, zip(cs, rs)))


def _ref_pair(col, row):
    return kit.labelled_pair(column=col, row=row)


def _ref_distractor(tag, col, row):
    text, value = _ref_pair(col, row)
    return Distractor(text, tag, value)


# ------------------------------------------------------------------------------------------ tiers

_REF_RULE = "Answer as column = _, row = _ (A = 1, B = 2, ...)."


def _easy(rng):
    cols, rows = 5, 5
    theme, places = _layout(rng, cols, rows, rng.choice((4, 5)))
    name = rng.choice(sorted(places))
    col, row = places[name]
    text = phrase(
        rng,
        (
            f"The map shows {theme}. What is the grid reference of the {name}? " + _REF_RULE,
            f"Look at the map of {theme}. Where is the {name}? " + _REF_RULE,
        ),
    )
    cands = [
        _ref_distractor("swapped-column-row", row, col) if row <= cols else None,
        _ref_distractor("letter-numbered-from-zero", col - 1, row) if col > 1 else None,
        _ref_distractor("fr2a-row-counted-from-top", col, rows + 1 - row),
        _ref_distractor("fr2a-grid-reference-off-by-one", col + 1, row)
        if col < cols
        else _ref_distractor("fr2a-grid-reference-off-by-one", col - 1, row),
        _ref_distractor("fr2a-grid-reference-off-by-one", col, row + 1)
        if row < rows
        else _ref_distractor("fr2a-grid-reference-off-by-one", col, row - 1),
    ]
    working = [
        f"Find the {name} on the map. It is in column {_LETTERS[col - 1]}, which is column {col}, and in row {row}.",
        f"So the grid reference is {_ref(col, row)}: column = {col}, row = {row}.",
    ]
    return theme, cols, rows, places, text, (col, row), [c for c in cands if c], working


def _random_moves(rng, start, cols, rows):
    """One east-west move and one north-south move that stay on the grid, in a random order."""
    col, row = start
    options_h = [(w, n) for w in ("east", "west") for n in range(1, 5) if 1 <= col + _STEP[w][0] * n <= cols]
    options_v = [(w, n) for w in ("north", "south") for n in range(1, 4) if 1 <= row + _STEP[w][1] * n <= rows]
    mh, mv = rng.choice(options_h), rng.choice(options_v)
    moves = [mh, mv]
    rng.shuffle(moves)
    return moves


def _medium(rng):
    cols, rows = 6, 5
    theme, places = _layout(rng, cols, rows, 5)
    name = rng.choice(sorted(places))
    start = places[name]
    moves = _random_moves(rng, start, cols, rows)
    col, row = start
    for word, n in moves:
        col += _STEP[word][0] * n
        row += _STEP[word][1] * n
    # second route: the displacement as signed totals
    dx = sum(_STEP[w][0] * n for w, n in moves)
    dy = sum(_STEP[w][1] * n for w, n in moves)
    check((start[0] + dx, start[1] + dy) == (col, row), "finishing square differs between the two routes")
    who = rng.choice(_WHO)
    pronoun = "she" if who in ("Mia", "Aiko", "Leila", "Priya", "Zoe") else "he"
    legs = f"{_squares(moves[0][1])} {moves[0][0]}, then {_squares(moves[1][1])} {moves[1][0]}"
    text = (
        f"{who} starts at the {name}. {pronoun.capitalize()} walks {legs}. Which square does {pronoun} reach? "
        + _REF_RULE
    )
    flipped_h = [(("east" if w == "west" else "west") if w in ("east", "west") else w, n) for w, n in moves]
    flipped_v = [(("north" if w == "south" else "south") if w in ("north", "south") else w, n) for w, n in moves]

    def land(ms, extra=(0, 0)):
        c, r = start
        for w, n in ms:
            c += _STEP[w][0] * n
            r += _STEP[w][1] * n
        return c + extra[0], r + extra[1]

    last_word, last_n = moves[-1]
    cands = []
    for tag, (c, r) in [
        ("horizontal-direction-reversed", land(flipped_h)),
        ("vertical-direction-reversed", land(flipped_v)),
        ("swapped-column-row", (row, col)),
        ("fr2a-map-leg-dropped", land(moves[:-1])),
        ("fr2a-map-start-square-counted", land(moves[:-1] + [(last_word, last_n + 1)])),
        ("letter-numbered-from-zero", (col - 1, row)),
    ]:
        if 1 <= c <= cols and 1 <= r <= rows and (c, r) != (col, row):
            cands.append(_ref_distractor(tag, c, r))
    working = [
        f"The {name} is in {_ref(*start)}: column {start[0]}, row {start[1]}.",
        f"Moving {legs}: east and west change the column, north and south change the row (north is up, so the row number goes up).",
        f"Finish in column {col}, row {row} ({_ref(col, row)}).",
    ]
    return theme, cols, rows, places, text, (col, row), cands, working


def _hard(rng):
    cols, rows = 7, 6
    for _ in attempts(2000, "three landmarks whose two legs partly cancel"):
        theme, places = _layout(rng, cols, rows, 5)
        a, b, c = rng.sample(sorted(places), 3)
        pa, pb, pc = places[a], places[b], places[c]
        leg1 = (pb[0] - pa[0], pb[1] - pa[1])
        leg2 = (pc[0] - pb[0], pc[1] - pb[1])
        net = (pc[0] - pa[0], pc[1] - pa[1])
        cancels = (leg1[0] * leg2[0] < 0) or (leg1[1] * leg2[1] < 0)
        if cancels and net != (0, 0) and net[0] != 0 and net[1] != 0:
            break
    text = (
        f"{rng.choice(_WHO)} walks from the {a} to the {b}, then to the {c}. Where is the {c} from the {a}? "
        "Answer as east = _, north = _ (negative if west or south)."
    )
    ans_text, ans = kit.labelled_pair(east=net[0], north=net[1])
    total_h = abs(leg1[0]) + abs(leg2[0])
    total_v = abs(leg1[1]) + abs(leg2[1])
    cands = []
    for tag, (e, n) in [
        ("fr2a-map-legs-added-not-netted", (total_h, total_v)),
        ("fr2a-map-first-leg-ignored", leg2),
        ("fr2a-map-second-leg-ignored", leg1),
        ("wrong-direction", (-net[0], -net[1])),
        ("swapped-coordinates", (net[1], net[0])),
        ("fr2a-map-middle-landmark-taken-as-end", (leg1[0], leg1[1])),
    ]:
        if (e, n) != net:
            text_, value = kit.labelled_pair(east=e, north=n)
            cands.append(Distractor(text_, tag, value))
    working = [
        f"{a} is in {_ref(*pa)}, {b} in {_ref(*pb)} and {c} in {_ref(*pc)}.",
        f"First leg: east {leg1[0]}, north {leg1[1]}. Second leg: east {leg2[0]}, north {leg2[1]}.",
        f"Add the legs, keeping signs: east {leg1[0]} + {paren_negative(leg2[0])} = {net[0]}, north {leg1[1]} + {paren_negative(leg2[1])} = {net[1]}.",
    ]
    return theme, cols, rows, places, text, net, cands, working, ans_text, ans


@generator(
    "fr2a_grid_map",
    level=5,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fr2a_grid_map(rng, tier="easy"):
    """Grid references and moves on a map of a familiar place."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return redraw(lambda r: _build(r, tier), rng, "a map question with enough distinct wrong answers")


def _build(rng, tier):
    built = {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)
    theme, cols, rows, places, text, answer, cands, working = built[:8]
    if tier == "hard":
        ans_text, ans = built[8], built[9]
    else:
        ans_text, ans = _ref_pair(*answer)
    fig = _map_figure(theme, cols, rows, places)
    check(not kit.leaks(fig, ans_text), "the figure shows the answer")
    dis = pick_distractors(ans_text, [c for c in cands if c.value != ans], 3, tolerance=GRADER_TOLERANCE)
    check(len(dis) >= 3, f"only {len(dis)} distractors")
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=ans,
        answer_text=ans_text,
        answer_latex=latex_answer(ans_text, plain=True),
        answer_kind="pair",
        working=working,
        distractors=dis,
        figure=fig,
    )


# ------------------------------------------------------------------------------------------ the figure reader


def _read_map(question):
    """Re-derive the answer from the landmarks drawn on the map and the places and moves named in the stem."""
    fig, text = question.figure, question.text
    places = {}
    for pt in fig["points"]:
        label = pt.get("label", "")
        if label and not re.fullmatch(r"[A-Z]|\d+", label):
            places[label] = (int(pt["at"][0] + 0.5), int(pt["at"][1] + 0.5))
    named = sorted(((text.find(f"the {n}"), n) for n in places if f"the {n}" in text))
    if "east = _" in text:
        (_, a), (_, b), (_, c) = named
        return kit.labelled_pair(east=places[c][0] - places[a][0], north=places[c][1] - places[a][1])[0]
    if "starts at" in text:
        col, row = places[named[0][1]]
        for n, word in re.findall(r"(\d+) squares? (east|west|north|south)", text):
            col += _STEP[word][0] * int(n)
            row += _STEP[word][1] * int(n)
        return _ref_pair(col, row)[0]
    col, row = places[named[0][1]]
    return _ref_pair(col, row)[0]


register_reader("fr2a_grid_map", _read_map)
