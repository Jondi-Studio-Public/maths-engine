"""Early-years "place the object": the learner shades the square where a described object goes.

Descriptor: AC9MFSP02 (describe position and location of self and objects using everyday language). The skill is
``f10-place-objects-from-position-words`` (``primary-geometry-time``, answer ``grid_cells``). A small grid shows a few
pictures (the pictograms of ``ey_position_build``); the prompt (at most 15 words) says where a new picture goes, using
the pictures that are drawn. The new picture is not drawn, so the learner has to read the position words and shade the
square or squares themselves, with no options to pick from.

* easy: one line of five squares (a row or a column). The new picture goes just left, right, above or below one
  picture, or between two pictures.
* medium: a scene with pictures all around. The new picture goes where two clues meet ("just above the house and just
  left of the church") or between two pictures in a row or column.
* hard: two new pictures. The second is placed just beside the first, which is itself not drawn, so the learner finds
  the first square before the second. Shade both.

Every key is built backwards from the chosen squares. The figure reader recomputes the squares from the figure's
pictures and the position words in the prompt.
"""

from __future__ import annotations

import re

from .. import figures
from ..core import attempts, check, generator
from ..figure_kinds.grid import grid_drawing
from ..grading.grid import cell_text
from ..skills import READ_ALOUD_MAX_WORDS
from ._spread_kit import uses_rng
from .dg_grid_map_build import _cells_part, _question
from .ey_position_build import OPPOSITE, VEC, WORD, Cell, _add, _inside, _scatter, _tier, symbols_of

SKILL = "f10-place-objects-from-position-words"
_TIERS = ("easy", "medium", "hard")
_PRESETS = {t: {"tier": t} for t in _TIERS}
_KEYS = tuple(WORD)

OPPOSITE_SIDE = "epo-opposite-side"
COVERS = "epo-covers-a-picture"
TOO_FAR = "epo-one-square-too-far"
BESIDE = "epo-beside-not-between"
WRONG_REF = "epo-wrong-reference"
WRONG_AXIS = "epo-wrong-axis"
ONLY_ONE = "epo-places-only-one"
SECOND_FROM_OLD = "epo-second-from-wrong-picture"

MISCONCEPTIONS = [
    (OPPOSITE_SIDE, "Puts the picture on the opposite side, such as below when the clue says above."),
    (COVERS, "Shades the square of a picture that is already there instead of an empty square beside or between."),
    (TOO_FAR, "Goes one square too far, leaving a gap instead of putting the picture just beside the other."),
    (
        BESIDE,
        "Shades a square beside one of the two pictures, outside them or off the line between them, not in the square "
        "between.",
    ),
    (
        WRONG_REF,
        "Uses the right position word about the wrong picture, such as above the second picture named instead of the "
        "first.",
    ),
    (WRONG_AXIS, "Goes left or right when the clue says above or below, or the other way round."),
    (ONLY_ONE, "Shades the square of only the first new picture and leaves out the second."),
    (
        SECOND_FROM_OLD,
        "Puts the second picture beside a picture already drawn instead of beside the first new picture.",
    ),
]

_REL = {"up": "above", "down": "below", "left": "left of", "right": "right of"}
_REL_FIND = {"above": "up", "below": "down", "left of": "left", "right of": "right"}
_HORIZONTAL = ("left", "right")
_VERTICAL = ("up", "down")


def _just(rel: str, name: str) -> str:
    return f"just {_REL[rel]} the {name}"


def _say(rng, templates, **fields) -> str | None:
    """A template filled in, among those short enough to read aloud; None when none is."""
    fits = [t.format(**fields) for t in templates]
    fits = [t for t in fits if len(t.split()) <= READ_ALOUD_MAX_WORDS]
    return rng.choice(fits) if fits else None


def _figure(w: int, h: int, things: dict[Cell, str], plural: bool) -> dict:
    return grid_drawing(
        alt="A small grid of pictures. Shade the square" + ("s" if plural else "") + " where the new picture goes.",
        grid={"type": "square", "width": w, "height": h},
        symbols=[{"at": [x + 0.5, y + 0.5], "symbol": key} for (x, y), key in things.items()],
        canvas="grid",
    )


def _scene(rng, w, h, refs: dict[Cell, str], new: list[str], targets: list[Cell], decoys: int) -> dict[Cell, str]:
    """The pictures drawn: the references and some others, none at a target square and none a new picture."""
    things = dict(refs)
    taken = set(refs) | set(targets)
    things.update(_scatter(rng, w, h, set(refs.values()) | set(new), taken, decoys))
    return things


def _done(text, fig, key_cells, w, h, wrong, working):
    if text is None:
        return None
    wrong = [({got} if isinstance(got, tuple) else got, tag) for got, tag in wrong]  # a lone square or a set
    fields, ds = _cells_part(set(key_cells), w, h, wrong)
    return _question(text, fig, fields, ds, working)


# ---------------------------------------------------------------------------------------------- easy

_EASY = (
    "The {x} goes {clue}. Shade its square.",
    "Shade the {x}. It is {clue}.",
    "Where does the {x} go? {Clue}. Shade it.",
)


def _easy(rng):
    horizontal = rng.random() < 0.5
    w, h = (5, 1) if horizontal else (1, 5)
    cell = (lambda i: (i, 0)) if horizontal else (lambda i: (0, i))
    rels = _HORIZONTAL if horizontal else _VERTICAL
    keys = rng.sample(_KEYS, 3)
    if rng.random() < 0.55:  # just beside one picture
        rel = rng.choice(rels)
        a = cell(rng.randrange(5))
        t = _add(a, VEC[rel])
        if not _inside(t, w, h):
            return None
        refs, clue = {a: keys[0]}, _just(rel, WORD[keys[0]])
        wrong = [
            (_add(a, VEC[OPPOSITE[rel]]), OPPOSITE_SIDE),
            (a, COVERS),
            (_add(a, VEC[rel], 2), TOO_FAR),
            (_add(a, VEC[OPPOSITE[rel]], 2), TOO_FAR),
        ]
        working = [
            f"Find the {WORD[keys[0]]}.",
            f"Just {_REL[rel]} it is the next square on that side. Shade that square.",
        ]
    else:  # between two pictures
        i = rng.randrange(3)
        a, t, b = cell(i), cell(i + 1), cell(i + 2)
        refs = {a: keys[0], b: keys[1]}
        rel = "right" if horizontal else "up"
        clue = f"between the {WORD[keys[0]]} and the {WORD[keys[1]]}"
        wrong = [
            (_add(a, VEC[OPPOSITE[rel]]), BESIDE),
            (_add(b, VEC[rel]), BESIDE),
            (a, COVERS),
            (b, COVERS),
        ]
        working = [
            f"Find the {WORD[keys[0]]} and the {WORD[keys[1]]}.",
            "The square between them has one square on each side that is a picture. Shade it.",
        ]
    things = _scene(rng, w, h, refs, [keys[2]], [t], rng.randint(0, 1))
    text = _say(rng, _EASY, x=WORD[keys[2]], clue=clue, Clue=clue[0].upper() + clue[1:])
    return _done(text, _figure(w, h, things, False), [t], w, h, wrong, working)


# ---------------------------------------------------------------------------------------------- medium

_M_TWO = (
    "Shade where the {x} goes: {c1} and {c2}.",
    "The {x} is {c1} and {c2}. Shade it.",
    "Shade the {x}. It is {c1} and {c2}.",
)
_M_BETWEEN = (
    "Shade where the {x} goes: {clue}.",
    "The {x} is {clue}. Shade it.",
    "Shade the {x}. It is {clue}.",
)


def _medium(rng):
    w, h = rng.choice((4, 5)), rng.choice((3, 4))
    keys = rng.sample(_KEYS, 3)
    x = WORD[keys[2]]
    if rng.random() < 0.6:  # two clues, one up-or-down and one left-or-right
        r1, r2 = rng.choice(_VERTICAL), rng.choice(_HORIZONTAL)
        if rng.random() < 0.5:
            r1, r2 = r2, r1
        t = (rng.randrange(w), rng.randrange(h))
        a, b = _add(t, VEC[OPPOSITE[r1]]), _add(t, VEC[OPPOSITE[r2]])
        if not (_inside(a, w, h) and _inside(b, w, h)):
            return None
        refs = {a: keys[0], b: keys[1]}
        c1, c2 = _just(r1, WORD[keys[0]]), _just(r2, WORD[keys[1]])
        wrong = [
            (_add(a, VEC[OPPOSITE[r1]]), OPPOSITE_SIDE),
            (_add(b, VEC[OPPOSITE[r2]]), OPPOSITE_SIDE),
            (_add(b, VEC[r1]), WRONG_REF),
            (_add(a, VEC[r2]), WRONG_REF),
            (a, COVERS),
            (b, COVERS),
        ]
        text = _say(rng, _M_TWO, x=x, c1=c1, c2=c2)
        working = [
            f"Find the {WORD[keys[0]]}. Just {_REL[r1]} it is one square.",
            f"Find the {WORD[keys[1]]}. Just {_REL[r2]} it is one square.",
            "Shade the one empty square that fits both clues.",
        ]
    else:  # between two pictures in a row or a column, with pictures around
        horizontal = rng.random() < 0.5
        step = (1, 0) if horizontal else (0, 1)
        t = (rng.randrange(w), rng.randrange(h))
        a, b = _add(t, step, -1), _add(t, step, 1)
        if not (_inside(a, w, h) and _inside(b, w, h)):
            return None
        refs = {a: keys[0], b: keys[1]}
        perp = (0, 1) if horizontal else (1, 0)
        wrong = [
            (_add(a, step, -1), BESIDE),
            (_add(b, step), BESIDE),
            (_add(a, perp), BESIDE),
            (_add(a, perp, -1), BESIDE),
            (a, COVERS),
            (b, COVERS),
        ]
        clue = f"between the {WORD[keys[0]]} and the {WORD[keys[1]]}"
        text = _say(rng, _M_BETWEEN, x=x, clue=clue)
        working = [
            f"Find the {WORD[keys[0]]} and the {WORD[keys[1]]}.",
            "Shade the empty square that sits in line between them.",
        ]
    things = _scene(rng, w, h, refs, [keys[2]], [t], rng.randint(2, 3))
    return _done(text, _figure(w, h, things, False), [t], w, h, wrong, working)


# ---------------------------------------------------------------------------------------------- hard

_HARD = (
    "Shade the {x} {c1}, and the {y} {c2}.",
    "Shade the {x} {c1}, then the {y} {c2}.",
    "Shade both: the {x} {c1}, the {y} {c2}.",
)


def _hard(rng):
    w, h = 5, 4
    keys = rng.sample(_KEYS, 4)
    x, y = WORD[keys[2]], WORD[keys[3]]
    r2 = rng.choice(_DIRS_ALL)
    side = _HORIZONTAL if r2 in _VERTICAL else _VERTICAL  # the other axis
    if rng.random() < 0.55:  # the first picture is just beside a drawn one
        r1 = rng.choice(_DIRS_ALL)
        if r1 == OPPOSITE[r2]:  # the second would land back on the drawn picture
            return None
        a = (rng.randrange(w), rng.randrange(h))
        t1 = _add(a, VEC[r1])
        t2 = _add(t1, VEC[r2])
        if not (_inside(t1, w, h) and _inside(t2, w, h)):
            return None
        refs = {a: keys[0]}
        c1 = _just(r1, WORD[keys[0]])
        wrong = [
            ({t1}, ONLY_ONE),
            ({t1, _add(t1, VEC[OPPOSITE[r2]])}, OPPOSITE_SIDE),
            ({t1, _add(a, VEC[r2])}, SECOND_FROM_OLD),
            ({t1, _add(t1, VEC[r2], 2)}, TOO_FAR),
            ({t1, _add(t1, VEC[side[0]])}, WRONG_AXIS),
            ({t1, _add(t1, VEC[side[1]])}, WRONG_AXIS),
            ({_add(a, VEC[OPPOSITE[r1]]), _add(_add(a, VEC[OPPOSITE[r1]]), VEC[r2])}, OPPOSITE_SIDE),
        ]
        working = [
            f"Find the {WORD[keys[0]]}. The {x} is just {_REL[r1]} it.",
            f"The {y} is just {_REL[r2]} the {x}, not the {WORD[keys[0]]}.",
            "Shade both squares.",
        ]
    else:  # the first picture is between two drawn ones
        step = (1, 0) if rng.random() < 0.5 else (0, 1)
        t1 = (rng.randrange(w), rng.randrange(h))
        a, b = _add(t1, step, -1), _add(t1, step, 1)
        t2 = _add(t1, VEC[r2])
        if not (_inside(a, w, h) and _inside(b, w, h) and _inside(t2, w, h)) or t2 in (a, b):
            return None
        refs = {a: keys[0], b: keys[1]}
        c1 = f"between the {WORD[keys[0]]} and the {WORD[keys[1]]}"
        wrong = [
            ({t1}, ONLY_ONE),
            ({t1, _add(t1, VEC[OPPOSITE[r2]])}, OPPOSITE_SIDE),
            ({t1, _add(a, VEC[r2])}, SECOND_FROM_OLD),
            ({t1, _add(b, VEC[r2])}, SECOND_FROM_OLD),
            ({t1, _add(t1, VEC[r2], 2)}, TOO_FAR),
            ({t1, _add(t1, VEC[side[0]])}, WRONG_AXIS),
            ({t1, _add(t1, VEC[side[1]])}, WRONG_AXIS),
            ({a, _add(a, VEC[r2])}, COVERS),
        ]
        working = [
            f"Find the {WORD[keys[0]]} and the {WORD[keys[1]]}. The {x} is the square between them.",
            f"The {y} is just {_REL[r2]} the {x}.",
            "Shade both squares.",
        ]
    if t2 in refs or t1 in refs:
        return None
    text = _say(rng, _HARD, x=x, y=y, c1=c1, c2=_just(r2, x))
    things = _scene(rng, w, h, refs, [keys[2], keys[3]], [t1, t2], rng.randint(2, 3))
    return _done(text, _figure(w, h, things, True), [t1, t2], w, h, wrong, working)


_DIRS_ALL = ("up", "down", "left", "right")


@generator("ey_place_objects", level=1, presets=_PRESETS)
@uses_rng
def ey_place_objects(rng, tier="easy"):
    _tier(tier)
    build = {"easy": _easy, "medium": _medium, "hard": _hard}[tier]
    for _ in attempts(800, f"a {tier} place-the-object question"):
        d = build(rng)
        if d is not None:
            return d
    raise AssertionError  # pragma: no cover


# ---------------------------------------------------------------------------------------------- the reader

_CLUE = re.compile(r"just (above|below|left of|right of) (?:the (\w+)|(it))|between the (\w+) and the (\w+)", re.I)


def _read_place_objects(question) -> str:
    """The squares recomputed from the pictures drawn and the position words in the prompt."""
    where = {word: cell for cell, word in symbols_of(question.figure).items()}
    found: list[Cell] = []  # the square of each new picture, in the order the prompt places them
    for m in _CLUE.finditer(question.text):
        rel, name, it, left, right = m.groups()
        if left:
            (ax, ay), (bx, by) = where[left], where[right]
            check(ax == bx or ay == by, "between needs two pictures in line")
            found.append(((ax + bx) // 2, (ay + by) // 2))
        else:
            vec = VEC[_REL_FIND[rel.lower()]]
            if it or name not in where:  # beside the new picture placed just before
                found.append(_add(found[-1], vec))
            else:
                cell = _add(where[name], vec)
                if found:
                    check(cell == found[-1], "two clues name one square")
                else:
                    found.append(cell)
    check(found, "no clue in the prompt")
    return cell_text(set(found))


figures.register_reader("ey_place_objects", _read_place_objects)
