"""Early-years position, direction, turns and simple maps, drawn on a small square ``grid_drawing``.

Descriptors: AC9MFSP02 (position and location words), AC9M1SP02 (give and follow directions, describe position),
AC9M2SP02 (interpret simple maps, describe position and movement) and AC9M2M05 (half and quarter turns).

Every picture is a few pictograms in the cells of a small grid, with no labels that name them, so the learner reads
the picture and the prompt (at most 15 words). Cell ``[x, y]`` counts from the left and from the bottom; up the page
is up. The four pictures a learner names are house, school, church, hospital, tree, lighthouse, tent and aeroplane.

- ``ey_position_words`` (skill ``ey-position-words-scene``, ``mcq``): easy is one line of five pictures (between, next
  to one at the end, just left or right, just above or below); medium is a 2D scene and asks for the picture just
  above, below, left or right of one; hard asks for a picture by two clues ("just above the A and just left of the
  B"), for the picture between two others in a scene, or for the words that tell where a picture is.
- ``ey_follow_directions`` (skill ``ey-follow-directions-grid``, ``mcq`` and ``number``): the word Start marks a cell;
  easy is one move, medium two moves at right angles, hard three moves with one going back, or a drawn path whose
  steps are counted.
- ``ey_turns`` (skill ``ey-quarter-half-turns``, ``mcq``): a ring marked You with an arrow showing which way it
  faces and a picture at each side; easy turns a quarter left or right from facing up, medium from any facing and
  also a half turn, hard joins two turns or names the turn that takes you from one picture to another.
- ``ey_map_reading`` (skill ``ey-simple-map-reading``, ``mcq`` and ``number``): five places on a small map; easy
  names the place at the top, bottom, far left or far right, medium the place closest to or farthest from one,
  hard counts the steps along the squares between two places.

Each topic registers a figure reader that re-derives the answer from ``question.figure`` and the prompt alone.
"""

from __future__ import annotations

import re

from .. import figures
from ..answers import Distractor
from ..core import attempts, check, generator, pick_distractors
from ..figure_kinds.grid import grid_drawing
from ..slots import NAMES
from ._ey2a_kit import number_question, tagged_numbers

_TIERS = ("easy", "medium", "hard")
_PRESETS = {t: {"tier": t} for t in _TIERS}

# The pictograms learners can name, and the word each is called in a prompt.
WORD = {
    "house": "house",
    "school": "school",
    "church": "church",
    "hospital": "hospital",
    "tree": "tree",
    "lighthouse": "lighthouse",
    "campground": "tent",
    "airport": "aeroplane",
}
KEY_OF = {word: key for key, word in WORD.items()}
_KEYS = tuple(WORD)
_WORD_RE = re.compile(r"\b(" + "|".join(sorted(KEY_OF, key=len, reverse=True)) + r")\b")

# Up the page is +y; right is +x.
VEC = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
_DIRS = ("up", "right", "down", "left")  # clockwise from the top: a quarter turn right is +1


def _tier(tier: str) -> str:
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return tier


Cell = tuple[int, int]


def _add(c: Cell, v: Cell, k: int = 1) -> Cell:
    return (c[0] + k * v[0], c[1] + k * v[1])


def _inside(c: Cell, w: int, h: int) -> bool:
    return 0 <= c[0] < w and 0 <= c[1] < h


def _figure(w: int, h: int, things: dict[Cell, str], *, alt: str, start: Cell | None = None, path=None, labels=()):
    """A grid with the pictures of ``things`` (cell to symbol key); ``start`` writes Start in a cell."""
    labs = list(labels)
    if start is not None:
        labs.append({"at": [start[0] + 0.5, start[1] + 0.5], "text": "Start"})
    return grid_drawing(
        alt=alt,
        grid={"type": "square", "width": w, "height": h},
        symbols=[{"at": [x + 0.5, y + 0.5], "symbol": key} for (x, y), key in things.items()],
        labels=labs,
        path=path or (),
    )


def symbols_of(fig: dict) -> dict[Cell, str]:
    """The picture in each cell of a figure, as ``{(x, y): word}``."""
    return {(int(s["at"][0]), int(s["at"][1])): WORD[s["symbol"]] for s in fig.get("symbols", [])}


def _names_in(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _mcq(text: str, answer: str, cands: list, fig: dict, working: list[str], *, floor: int = 3) -> dict:
    """A multiple-choice question; ``cands`` is the wrong options in priority order, as Distractors or strings."""
    chosen = pick_distractors(answer, cands, n=3)
    check(len(chosen) >= floor, f"only {len(chosen)} distractors for {answer!r}")
    figures.assert_no_leak(fig, [answer, *[str(d) for d in chosen]])
    return dict(
        text=text,
        answer=answer,
        answer_text=answer,
        answer_latex=f"plain:{answer}",
        distractors=chosen,
        working=working,
        figure=fig,
    )


def _pictures(cells, things: dict[Cell, str], tag: str | None) -> list[Distractor]:
    """Distractors naming the picture in each of ``cells`` (skipping empty cells), all with the one tag."""
    return [Distractor(WORD[things[c]], tag) for c in cells if c in things]


def _fill(things: dict[Cell, str], exclude: set[Cell]) -> list[Distractor]:
    """Untagged distractors from every other picture, a last resort so a question always has three."""
    return [Distractor(WORD[k]) for c, k in things.items() if c not in exclude]


def _scatter(rng, w: int, h: int, keys, taken: set[Cell], n: int) -> dict[Cell, str]:
    """``n`` more pictures, of kinds not in ``keys``, at random empty cells."""
    free = [(x, y) for x in range(w) for y in range(h) if (x, y) not in taken]
    spare = [k for k in _KEYS if k not in keys]
    cells = rng.sample(free, min(n, len(free), len(spare)))
    return dict(zip(cells, rng.sample(spare, len(cells))))


# ====================================================================== position words

_REL_WORD = {"above": "above", "below": "below", "left": "left of", "right": "right of"}
_REL_FIND = re.compile(r"\b(above|below|left|right)\b")
_BETWEEN = ("What is between the {a} and the {b}?", "Which picture is between the {a} and the {b}?")
_NEXT = ("What is next to the {x}?", "Which picture is beside the {x}?", "What is just beside the {x}?")
_JUST = (
    "What is {adv} {rel} the {x}?",
    "Which picture is {adv} {rel} the {x}?",
)
_TWO = (
    "What is just {r1} the {a} and just {r2} the {b}?",
    "Which picture is just {r1} the {a}, and just {r2} the {b}?",
)
_WHERE = ("Where is the {x} compared to the {y}?", "Look at the {y}. Where is the {x}?")
_PLACES = {"above": "above", "below": "below", "left": "to the left", "right": "to the right"}
_PLACE_KEY = {v: k for k, v in _PLACES.items()}


def _line_cell(i: int, horizontal: bool, n: int) -> Cell:
    return (i, 0) if horizontal else (0, n - 1 - i)


@generator("ey_position_words", level=1, presets=_PRESETS)
def ey_position_words(rng, tier="easy"):
    _tier(tier)
    return {"easy": _pw_easy, "medium": _pw_medium, "hard": _pw_hard}[tier](rng)


def _pw_easy(rng):
    n = 5
    horizontal = rng.random() < 0.5
    keys = rng.sample(_KEYS, n)
    w, h = (n, 1) if horizontal else (1, n)
    cells = [_line_cell(i, horizontal, n) for i in range(n)]
    things = dict(zip(cells, keys))
    fig = _figure(w, h, things, alt="a row of small pictures" if horizontal else "a column of small pictures")
    word = [WORD[k] for k in keys]
    before, after = ("left", "right") if horizontal else ("above", "below")
    kind = rng.choice(("between", "next", "side"))
    if kind == "between":
        i = rng.randint(1, n - 2)
        a, b = (word[i - 1], word[i + 1]) if rng.random() < 0.5 else (word[i + 1], word[i - 1])
        text = rng.choice(_BETWEEN).format(a=a, b=b)
        cands = _pictures([cells[i - 1], cells[i + 1]], things, "eyp-names-the-reference")
        cands += _pictures([c for j, c in enumerate(cells) if abs(j - i) > 1], things, "eyp-not-next-to")
        working = [f"The {word[i]} is in the middle of the {word[i - 1]} and the {word[i + 1]}."]
        answer = word[i]
    elif kind == "next":
        i = rng.choice((0, n - 1))
        j = 1 if i == 0 else n - 2
        text = rng.choice(_NEXT).format(x=word[i])
        cands = _pictures([c for m, c in enumerate(cells) if abs(m - i) > 1], things, "eyp-not-next-to")
        working = [f"The {word[j]} is right beside the {word[i]}."]
        answer = word[j]
    else:
        i = rng.randint(1, n - 2)
        rel, step = rng.choice(((before, -1), (after, 1)))
        j = i + step
        text = rng.choice(_JUST).format(adv=rng.choice(("just", "directly")), rel=_REL_WORD[rel], x=word[i])
        far = [c for m, c in enumerate(cells) if (m - i) * step > 1]
        cands = _pictures([cells[i - step]], things, "eyp-opposite-direction")
        cands += _pictures(far, things, "eyp-not-next-to")
        cands += _pictures([cells[i]], things, "eyp-names-the-reference")
        cands += _fill(things, {cells[j]})
        working = [f"The {word[j]} is {_REL_WORD[rel]} the {word[i]}."]
        answer = word[j]
    return _mcq(text, answer, cands, fig, working)


def _pw_medium(rng):
    w, h = rng.choice(((4, 3), (3, 4), (4, 4)))
    rel = rng.choice(tuple(_REL_WORD))
    v = VEC[_vec_name(rel)]
    for _ in attempts(200, "a reference with a picture on each side"):
        ref = (rng.randrange(w), rng.randrange(h))
        if _inside(_add(ref, v), w, h) and _inside(_add(ref, v, -1), w, h):
            break
    target, opposite = _add(ref, v), _add(ref, v, -1)
    keys = rng.sample(_KEYS, 6)
    things = {ref: keys[0], target: keys[1], opposite: keys[2]}
    things.update(_scatter(rng, w, h, set(things.values()), set(things), 3))
    fig = _figure(w, h, things, alt="a small scene of pictures on a grid")
    word = {c: WORD[k] for c, k in things.items()}
    text = rng.choice(_JUST).format(adv=rng.choice(("just", "directly")), rel=_REL_WORD[rel], x=word[ref])
    near = [c for c in things if c not in (ref, target, opposite) and max(abs(c[0] - ref[0]), abs(c[1] - ref[1])) <= 2]
    cands = _pictures([opposite], things, "eyp-opposite-direction") + _pictures(near, things, "eyp-not-next-to")
    cands += _fill(things, {ref, target})
    working = [f"Look {_REL_WORD[rel]} the {word[ref]}: it is the {word[target]}."]
    return _mcq(text, word[target], cands, fig, working)


def _vec_name(rel: str) -> str:
    return {"above": "up", "below": "down", "left": "left", "right": "right"}[rel]


def _pw_hard(rng):
    kind = rng.choice(("two", "between", "where"))
    if kind == "two":
        return _pw_two(rng)
    if kind == "between":
        return _pw_between(rng)
    return _pw_where(rng)


def _pw_two(rng):
    w = h = 4
    first = rng.choice(("above", "below"))
    second = rng.choice(("left", "right"))
    r1, r2 = (first, second) if rng.random() < 0.5 else (second, first)
    v1, v2 = VEC[_vec_name(r1)], VEC[_vec_name(r2)]
    t = (rng.randint(1, 2), rng.randint(1, 2))
    a, b = _add(t, v1, -1), _add(t, v2, -1)  # t is r1 of a, and r2 of b
    c1, c2 = _add(b, v1), _add(a, v2)  # r1 used about b, r2 used about a
    keys = rng.sample(_KEYS, 6)
    things = dict(zip((t, a, b, c1, c2), keys))
    things.update(_scatter(rng, w, h, set(things.values()), set(things), 1))
    fig = _figure(w, h, things, alt="a small scene of pictures on a grid")
    word = {c: WORD[k] for c, k in things.items()}
    text = rng.choice(_TWO).format(r1=_REL_WORD[r1], r2=_REL_WORD[r2], a=word[a], b=word[b])
    cands = _pictures([c1, c2], things, "eyp-wrong-reference")
    cands += _pictures([a, b], things, "eyp-names-the-reference")
    working = [f"The {word[t]} is {_REL_WORD[r1]} the {word[a]} and {_REL_WORD[r2]} the {word[b]}."]
    return _mcq(text, word[t], cands, fig, working)


def _pw_between(rng):
    w = h = 4
    horizontal = rng.random() < 0.5
    along = rng.randint(0, 1)
    across = rng.randrange(4)
    line = [((along + i, across) if horizontal else (across, along + i)) for i in range(3)]
    perp = (0, 1) if horizontal else (1, 0)
    mid = line[1]
    keys = rng.sample(_KEYS, 6)
    things = dict(zip(line, keys))
    beside = [c for c in (_add(mid, perp), _add(mid, perp, -1)) if _inside(c, w, h)]
    for c in beside[:1]:
        things[c] = keys[3]
    things.update(_scatter(rng, w, h, set(things.values()), set(things), 6 - len(things)))
    fig = _figure(w, h, things, alt="a small scene of pictures on a grid")
    word = {c: WORD[k] for c, k in things.items()}
    ends = [word[line[0]], word[line[2]]]
    rng.shuffle(ends)
    text = rng.choice(_BETWEEN).format(a=ends[0], b=ends[1])
    cands = _pictures(beside[:1], things, "eyp-next-to-not-between") + _pictures(
        [line[0], line[2]], things, "eyp-names-the-reference"
    )
    cands += _fill(things, {mid})
    working = [f"The {word[mid]} is in the middle of the {ends[0]} and the {ends[1]}."]
    return _mcq(text, word[mid], cands, fig, working)


def _pw_where(rng):
    w = h = 4
    rel = rng.choice(tuple(_PLACES))
    v = VEC[_vec_name(rel)]
    for _ in attempts(200, "a reference with room for the picture"):
        ref = (rng.randrange(w), rng.randrange(h))
        gap = rng.randint(1, 2)
        subject = _add(ref, v, gap)
        if _inside(subject, w, h):
            break
    keys = rng.sample(_KEYS, 5)
    things = {ref: keys[0], subject: keys[1]}
    things.update(_scatter(rng, w, h, set(things.values()), set(things) | _line_through(ref, w, h), 3))
    fig = _figure(w, h, things, alt="a small scene of pictures on a grid")
    x, y = WORD[keys[1]], WORD[keys[0]]
    text = rng.choice(_WHERE).format(x=x, y=y)
    opposite = {"above": "below", "below": "above", "left": "right", "right": "left"}[rel]
    cands = [Distractor(_PLACES[opposite], "eyp-opposite-direction")]
    cands += [Distractor(_PLACES[r], "eyp-wrong-axis") for r in _PLACES if r not in (rel, opposite)]
    working = [f"The {x} is {_PLACES[rel]} the {y}."]
    return _mcq(text, _PLACES[rel], cands, fig, working)


def _line_through(c: Cell, w: int, h: int) -> set[Cell]:
    """Every cell in the row and column of ``c``, so a scatter keeps the relation to ``c`` unambiguous."""
    return {(x, c[1]) for x in range(w)} | {(c[0], y) for y in range(h)}


def _read_position_words(q) -> str:
    text, pics = q.text, symbols_of(q.figure)
    where = {word: cell for cell, word in pics.items()}
    names = _names_in(text)
    if "between" in text:
        a, b = where[names[0]], where[names[1]]
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        return pics[mid]
    if "compared to" in text or "Where is" in text:
        x = _names_in(text.split("Where is")[1])[0]
        (y,) = [n for n in names if n != x]
        dx, dy = where[x][0] - where[y][0], where[x][1] - where[y][1]
        rel = "right" if dx > 0 else "left" if dx < 0 else "up" if dy > 0 else "down"
        return _PLACES[{"up": "above", "down": "below"}.get(rel, rel)]
    if "next to" in text or "beside" in text:
        ref = where[names[0]]
        (near,) = [c for c in pics if abs(c[0] - ref[0]) + abs(c[1] - ref[1]) == 1]
        return pics[near]
    clauses = re.split(r",? and ", text)
    cells = []
    for clause in clauses:
        (rel,) = _REL_FIND.findall(clause)
        (name,) = _names_in(clause)
        cells.append(_add(where[name], VEC[_vec_name(rel)]))
    if len(clauses) == 2:
        check(cells[0] == cells[1], "the two clues name one place")
    return pics[cells[0]]


figures.register_reader("ey_position_words", _read_position_words)


# ====================================================================== follow directions

_MOVE_RE = re.compile(r"\b(up|down|left|right)\s+(\d+)")
_EASY_GO = (
    "Begin at Start. Go {d} {k} {steps}. What do you find?",
    "Go {d} {k} {steps} from Start. What is there?",
    "{n} begins at Start and goes {d} {k} {steps}. What does {n} find?",
)
_MED_GO = (
    "Begin at Start. Go {m}. What do you find?",
    "Go {m} from Start. What is there?",
    "{n} starts at Start and goes {m}. What does {n} find?",
)
_HARD_GO = (
    "From Start, go {m}. What is there?",
    "Begin at Start. Go {m}. What do you find?",
    "{n} goes {m} from Start. What does {n} find?",
)


def _step_word(k: int) -> str:
    return "step" if k == 1 else "steps"


def _moves_text(moves) -> str:
    parts = [f"{d} {k}" for d, k in moves]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + ", then " + parts[-1]


def _swap_tag(direction: str) -> str:
    return "eyp-left-right-swapped" if direction in ("left", "right") else "eyp-opposite-direction"


def _walk(start: Cell, moves) -> Cell:
    c = start
    for d, k in moves:
        c = _add(c, VEC[d], k)
    return c


@generator("ey_follow_directions", level=1, presets=_PRESETS)
def ey_follow_directions(rng, tier="easy"):
    _tier(tier)
    return _fd_moves(rng, tier)


def _perpendicular(d: str) -> tuple[str, str]:
    return ("left", "right") if d in ("up", "down") else ("up", "down")


def _fd_plan(rng, tier: str):
    """``(moves, candidates)``: the moves to follow and, for each wrong reading, ``(moves, tag)``."""
    if tier == "easy":
        d, k = rng.choice(_DIRS), rng.randint(1, 2)
        moves = [(d, k)]
        wrong = [([(d, k + 1)], "eyp-one-step-too-far"), ([(OPPOSITE[d], k)], _swap_tag(d))]
        if k > 1:
            wrong.insert(0, ([(d, k - 1)], "eyp-counts-the-start"))
        return moves, wrong
    d1 = rng.choice(_DIRS)
    d2 = rng.choice(_perpendicular(d1))
    a, b = rng.randint(1, 3), rng.randint(1, 3)
    if tier == "medium":
        moves = [(d1, a), (d2, b)]
        return moves, [
            ([(d1, a)], "eyp-stops-early"),
            ([(d1, a), (OPPOSITE[d2], b)], _swap_tag(d2)),
            ([(OPPOSITE[d1], a), (d2, b)], _swap_tag(d1)),
            ([(d1, a - 1), (d2, b - 1)], "eyp-counts-the-start"),
        ]
    c = rng.choice([x for x in range(1, 4) if x != a])
    moves = [(d1, a), (d2, b), (OPPOSITE[d1], c)]
    return moves, [
        ([(d1, a), (d2, b)], "eyp-stops-early"),
        ([(d1, a), (d2, b), (d1, c)], "eyp-back-moves-forward"),
        ([(d1, a), (OPPOSITE[d2], b), (OPPOSITE[d1], c)], _swap_tag(d2)),
        ([(d1, a), (d2, b - 1), (OPPOSITE[d1], c)], "eyp-counts-the-start"),
    ]


def _fd_moves(rng, tier):
    w = h = 6
    for _ in attempts(500, "moves that stay on the grid with room for the wrong readings"):
        moves, wrong = _fd_plan(rng, tier)
        start = (rng.randrange(w), rng.randrange(h))
        end = _walk(start, moves)
        spots: list[tuple[Cell, str]] = []
        for m, tag in wrong:
            c = _walk(start, m)
            if _inside(c, w, h) and c not in (start, end) and c not in [s for s, _ in spots]:
                spots.append((c, tag))
        if _inside(end, w, h) and end != start and len(spots) >= (2 if tier == "easy" else 3):
            break
    keys = rng.sample(_KEYS, 6)
    things = {end: keys[0]}
    for (c, _), key in zip(spots, keys[1:]):
        things[c] = key
    spare_n = 2 if tier == "easy" else 1
    things.update(_scatter(rng, w, h, set(things.values()), set(things) | {start}, spare_n))
    fig = _figure(w, h, things, alt="a grid with a start square and some pictures", start=start)
    n = rng.choice(NAMES)
    if tier == "easy":
        (d, k) = moves[0]
        text = rng.choice(_EASY_GO).format(d=d, k=k, steps=_step_word(k), n=n)
    else:
        text = rng.choice(_MED_GO if tier == "medium" else _HARD_GO).format(m=_moves_text(moves), n=n)
    cands = [Distractor(WORD[things[c]], tag) for c, tag in spots]
    cands += _fill(things, {end})
    working = [f"Start at the Start square, then go {_moves_text(moves)}. You reach the {WORD[things[end]]}."]
    return _mcq(text, WORD[things[end]], cands, fig, working)


def _read_follow_directions(q) -> str:
    fig = q.figure
    (label,) = fig["labels"]
    start = (int(label["at"][0]), int(label["at"][1]))
    moves = [(d, int(k)) for d, k in _MOVE_RE.findall(q.text)]
    return symbols_of(fig)[_walk(start, moves)]


figures.register_reader("ey_follow_directions", _read_follow_directions)


# ====================================================================== count the steps

_PATH = {
    "easy": (
        "Follow the straight path from the ring. How many steps to the {x}?",
        "How many steps is it along the path to the {x}?",
    ),
    "medium": (
        "Follow the path round the corners. How many steps to the {x}?",
        "Follow the path from the ring. How many steps is it to the {x}?",
    ),
}
_STEPS = (
    "Walk along the squares. How many steps from the {a} to the {b}?",
    "{n} walks along the squares from the {a} to the {b}. How many steps?",
)


def _count_question(text: str, steps: int, cands, fig: dict, working: list[str]) -> dict:
    figures.assert_no_leak(fig, [str(steps)])
    return number_question(
        text=text,
        answer=steps,
        distractors=tagged_numbers(steps, cands, low=1, high=15),
        working=working,
        figure=fig,
    )


def _straight_path(rng, w: int, h: int) -> list[Cell]:
    for _ in attempts(200, "a straight path that fits the grid"):
        d = VEC[rng.choice(_DIRS)]
        length = rng.randint(2, 5)
        start = (rng.randrange(w), rng.randrange(h))
        if _inside(_add(start, d, length), w, h):
            return [_add(start, d, i) for i in range(length + 1)]
    raise AssertionError("unreachable")


def _bent_path(rng, w: int, h: int) -> list[Cell]:
    """A self-avoiding path of 4 to 7 steps that turns at least twice."""
    for _ in attempts(500, "a path that wanders but does not cross itself"):
        length = rng.randint(4, 7)
        cells = [(rng.randrange(w), rng.randrange(h))]
        for _ in range(length):
            options = [c for c in (_add(cells[-1], v) for v in VEC.values()) if _inside(c, w, h) and c not in cells]
            if not options:
                break
            cells.append(rng.choice(options))
        turns = sum(
            1
            for a, b, c in zip(cells, cells[1:], cells[2:])
            if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1])
        )
        if len(cells) == length + 1 and turns >= 2:
            return cells
    raise AssertionError("unreachable")


@generator("ey_count_steps", level=1, presets=_PRESETS)
def ey_count_steps(rng, tier="easy"):
    _tier(tier)
    return _cs_map(rng) if tier == "hard" else _cs_path(rng, tier)


def _cs_path(rng, tier):
    w = h = 6
    cells = _straight_path(rng, w, h) if tier == "easy" else _bent_path(rng, w, h)
    steps = len(cells) - 1
    key = rng.choice(_KEYS)
    things = {cells[-1]: key}
    things.update(_scatter(rng, w, h, {key}, set(cells), 2))
    fig = _figure(w, h, things, alt="a grid with a path drawn from a ring", path=cells)
    dx, dy = abs(cells[-1][0] - cells[0][0]), abs(cells[-1][1] - cells[0][1])
    text = rng.choice(_PATH[tier]).format(x=WORD[key])
    cands = [(steps + 1, "eyp-counts-the-start"), (steps - 1, "eyp-misses-a-step")]
    if tier != "easy":
        cands += [(max(dx, dy), "eyp-cuts-corner"), (dx + dy, "eyp-ignores-the-path")]
    working = [f"Count each step along the path from the ring to the {WORD[key]}: {steps} steps."]
    return _count_question(text, steps, cands, fig, working)


def _cs_map(rng):
    w, h = 6, 5
    for _ in attempts(500, "two places a few steps apart, across and up by different amounts"):
        a = (rng.randrange(w), rng.randrange(h))
        b = (rng.randrange(w), rng.randrange(h))
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        if dx >= 1 and dy >= 1 and dx != dy and dx + dy >= 3:
            break
    keys = rng.sample(_KEYS, 5)
    things = {a: keys[0], b: keys[1]}
    things.update(_scatter(rng, w, h, set(things.values()), set(things), 3))
    fig = _figure(w, h, things, alt="a simple map with five places on a grid")
    answer = dx + dy
    text = rng.choice(_STEPS).format(a=WORD[keys[0]], b=WORD[keys[1]], n=rng.choice(NAMES))
    cands = [
        (answer + 1, "eyp-counts-the-start"),
        (max(dx, dy), "eyp-cuts-corner"),
        (answer - 1, "eyp-misses-a-step"),
        (dx, "eyp-one-direction-only"),
        (dy, "eyp-one-direction-only"),
    ]
    working = [f"Count {dx} across and {dy} up or down: {dx} + {dy} = {answer} steps."]
    return _count_question(text, answer, cands, fig, working)


def _read_count_steps(q) -> str:
    fig = q.figure
    if "path" in fig:
        return str(len(fig["path"]["cells"]) - 1)
    pics = symbols_of(fig)
    where = {word: cell for cell, word in pics.items()}
    a, b = _names_in(q.text)[:2]
    return str(_dist(where[a], where[b]))


figures.register_reader("ey_count_steps", _read_count_steps)


# ====================================================================== quarter and half turns

# Cells of the four pictures on a 5 by 5 grid, clockwise from the top: up, right, down, left.
_SEAT = ((2, 4), (4, 2), (2, 0), (0, 2))
_CENTRE = (2, 2)
_TURN_RE = re.compile(r"\b(quarter|half) turn(?: to the)?(?: (left|right))?")
_ONE = (
    "Turn {t}. What do you face now?",
    "Make {t}. What is in front of you now?",
    "{n} does {t}. What does {n} face now?",
)
_TWO_TURNS = ("Turn {t}. What do you face?", "Make {t}. What is in front now?")
_REVERSE = ("Turn to face the {x}. Which turn do you make?", "You turn to face the {x}. Which turn is it?")
_TURN_NAME = {1: "a quarter turn right", -1: "a quarter turn left", 2: "a half turn", 0: "a full turn"}


def _turn_phrase(t: int, rng=None) -> str:
    if t == 2:
        return "a half turn"
    side = "right" if t == 1 else "left"
    return f"a quarter turn to the {side}" if rng is not None and rng.random() < 0.5 else f"a quarter turn {side}"


def _turn_figure(face: int, pics: list[str]) -> dict:
    things = {_SEAT[i]: pics[i] for i in range(4)}
    return _figure(
        5,
        5,
        things,
        alt="a ring marked You with an arrow showing which way it faces, and a picture on each side",
        path=[_CENTRE, _add(_CENTRE, VEC[_DIRS[face]])],
        labels=[{"at": [2, 2], "text": "You"}],
    )


@generator("ey_turns", level=1, presets=_PRESETS)
def ey_turns(rng, tier="easy"):
    _tier(tier)
    pics = rng.sample(_KEYS, 4)
    face = 0 if tier == "easy" else rng.randrange(4)
    fig = _turn_figure(face, pics)
    word = [WORD[k] for k in pics]
    n = rng.choice(NAMES)
    if tier == "hard" and rng.random() < 0.4:
        target = rng.choice([i for i in range(4) if i != face])
        diff = (target - face) % 4
        answer = _TURN_NAME[1 if diff == 1 else -1 if diff == 3 else 2]
        text = rng.choice(_REVERSE).format(x=word[target])
        if diff == 2:
            cands = [
                Distractor(_TURN_NAME[1], "eyp-half-as-quarter"),
                Distractor(_TURN_NAME[-1], "eyp-half-as-quarter"),
            ]
        else:
            cands = [Distractor(_TURN_NAME[-1 if diff == 1 else 1], "eyp-left-right-swapped")]
            cands.append(Distractor(_TURN_NAME[2], "eyp-quarter-as-half"))
        cands.append(Distractor(_TURN_NAME[0], "eyp-full-turn"))
        working = [f"Face the {word[face]}, then face the {word[target]}: that is {answer}."]
        return _mcq(text, answer, cands, fig, working)
    if tier == "hard":
        for _ in attempts(100, "two turns that do not cancel"):
            t1, t2 = rng.choice((1, -1, 2)), rng.choice((1, -1, 2))
            if (t1 + t2) % 4 != 0:
                break
        turns = [t1, t2]
        phrase_text = f"{_turn_phrase(t1)}, then {_turn_phrase(t2)}"
    elif tier == "medium":
        turns = [rng.choice((1, -1, 2))]
        phrase_text = _turn_phrase(turns[0], rng)
    else:
        turns = [rng.choice((1, -1))]
        phrase_text = _turn_phrase(turns[0], rng)
    final = (face + sum(turns)) % 4
    templates = _TWO_TURNS if len(turns) == 2 else _ONE
    text = rng.choice(templates).format(t=phrase_text, n=n)
    if len(turns) == 2:
        t1, t2 = turns
        cands = [
            Distractor(word[(face + t1) % 4], "eyp-stops-early"),
            Distractor(word[(face + t2) % 4], "eyp-stops-early"),
            Distractor(word[(face - t1 - t2) % 4], "eyp-left-right-swapped"),
            Distractor(word[face], "eyp-no-turn"),
        ]
    elif turns[0] == 2:
        cands = [
            Distractor(word[(face + 1) % 4], "eyp-half-as-quarter"),
            Distractor(word[(face - 1) % 4], "eyp-half-as-quarter"),
            Distractor(word[face], "eyp-no-turn"),
        ]
    else:
        cands = [
            Distractor(word[(face - turns[0]) % 4], "eyp-left-right-swapped"),
            Distractor(word[(face + 2) % 4], "eyp-quarter-as-half"),
            Distractor(word[face], "eyp-no-turn"),
        ]
    cands += [Distractor(w) for w in word]
    working = [f"You face the {word[face]}. After {phrase_text} you face the {word[final]}."]
    return _mcq(text, word[final], cands, fig, working)


def _read_turns(q) -> str:
    fig = q.figure
    a, b = fig["path"]["cells"]
    face = _DIRS.index({v: k for k, v in VEC.items()}[(b[0] - a[0], b[1] - a[1])])
    pics = symbols_of(fig)
    word = [pics[s] for s in _SEAT]
    if "Which turn" in q.text or "which turn" in q.text.lower():
        (x,) = _names_in(q.text)
        diff = (word.index(x) - face) % 4
        return _TURN_NAME[1 if diff == 1 else -1 if diff == 3 else 2]
    net = 0
    for size, side in _TURN_RE.findall(q.text):
        net += 2 if size == "half" else (1 if side == "right" else -1)
    return word[(face + net) % 4]


figures.register_reader("ey_turns", _read_turns)


# ====================================================================== simple maps

_EXTREME_TEXT = {
    "top": ("Which place is at the top of the map?", "What is at the very top of the map?"),
    "bottom": ("Which place is at the bottom of the map?", "What is at the very bottom of the map?"),
    "left": ("Which place is on the far left of the map?", "What is on the far left of the map?"),
    "right": ("Which place is on the far right of the map?", "What is on the far right of the map?"),
}
_NEAR = ("Which place is closest to the {x}?", "What is nearest to the {x}?")
_FAR = ("Which place is farthest from the {x}?", "What is the farthest place from the {x}?")
_CLOSEST_PAIR = (
    "Walk along the squares. Which two places are the closest?",
    "Which two places are the fewest steps apart?",
    "{n} walks along the squares. Which two places are the nearest together?",
)
_MAP_ALT = "a simple map with five places on a grid"


def _dist(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@generator("ey_map_reading", level=1, presets=_PRESETS)
def ey_map_reading(rng, tier="easy"):
    _tier(tier)
    return {"easy": _map_easy, "medium": _map_medium, "hard": _map_hard}[tier](rng)


def _map_easy(rng):
    w, h = 6, 5
    xs, ys = rng.sample(range(w), 5), rng.sample(range(h), 5)
    cells = list(zip(xs, ys))
    keys = rng.sample(_KEYS, 5)
    things = dict(zip(cells, keys))
    fig = _figure(w, h, things, alt=_MAP_ALT)
    edge = rng.choice(tuple(_EXTREME_TEXT))
    axis, sign = {"top": (1, -1), "bottom": (1, 1), "left": (0, 1), "right": (0, -1)}[edge]
    order = sorted(cells, key=lambda c: sign * c[axis])  # the most extreme first
    cands = _pictures([order[-1]], things, "eyp-opposite-direction") + _pictures(
        [order[1]], things, "eyp-nearly-extreme"
    )
    cands += _fill(things, {order[0]})
    where = f"at the {edge}" if edge in ("top", "bottom") else f"on the far {edge}"
    working = [f"The picture {where} of the map is the {WORD[things[order[0]]]}."]
    return _mcq(rng.choice(_EXTREME_TEXT[edge]), WORD[things[order[0]]], cands, fig, working)


def _map_medium(rng):
    w, h = 6, 5
    farthest = rng.random() < 0.4
    for _ in attempts(500, "a unique nearest or farthest place"):
        cells = rng.sample([(x, y) for x in range(w) for y in range(h)], 5)
        ref, others = cells[0], cells[1:]
        ds = sorted(_dist(ref, c) for c in others)
        if (ds[-1] > ds[-2]) if farthest else (ds[0] < ds[1]):
            break
    answer = max(others, key=lambda c: _dist(ref, c)) if farthest else min(others, key=lambda c: _dist(ref, c))
    wrong_end = min(others, key=lambda c: _dist(ref, c)) if farthest else max(others, key=lambda c: _dist(ref, c))
    keys = rng.sample(_KEYS, 5)
    things = dict(zip(cells, keys))
    fig = _figure(w, h, things, alt=_MAP_ALT)
    text = rng.choice(_FAR if farthest else _NEAR).format(x=WORD[things[ref]])
    inline = [c for c in others if c != answer and (c[0] == ref[0] or c[1] == ref[1])]
    cands = _pictures([wrong_end], things, "eyp-farthest-for-closest") + _pictures(inline, things, "eyp-looks-one-way")
    cands += _fill(things, {ref, answer})
    how = "farthest" if farthest else "closest"
    working = [
        f"Count the squares from the {WORD[things[ref]]} to each place. The {how} is the {WORD[things[answer]]}."
    ]
    return _mcq(text, WORD[things[answer]], cands, fig, working)


def _pair_text(a: str, b: str) -> str:
    first, second = sorted((a, b))
    return f"the {first} and the {second}"


def _cheb(a: Cell, b: Cell) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _map_hard(rng):
    w, h = 6, 5
    for _ in attempts(3000, "five places with one closest pair and a pair that only looks close"):
        cells = rng.sample([(x, y) for x in range(w) for y in range(h)], 5)
        pairs = sorted(((_dist(a, b), a, b) for i, a in enumerate(cells) for b in cells[i + 1 :]), key=lambda t: t[0])
        best = pairs[0]
        corner = [p for p in pairs[1:] if p[0] > best[0] and _cheb(p[1], p[2]) <= _cheb(best[1], best[2])]
        if best[0] < pairs[1][0] and best[0] >= 2 and corner and pairs[-1][0] > best[0] + 1:
            break
    keys = rng.sample(_KEYS, 5)
    things = dict(zip(cells, keys))
    fig = _figure(w, h, things, alt=_MAP_ALT)

    def option(pair) -> str:
        return _pair_text(WORD[things[pair[1]]], WORD[things[pair[2]]])

    same_line = [p for p in pairs[1:] if p[1][0] == p[2][0] or p[1][1] == p[2][1]]
    cands = [Distractor(option(pairs[-1]), "eyp-farthest-for-closest")]
    cands += [Distractor(option(p), "eyp-cuts-corner") for p in corner]
    cands += [Distractor(option(p), "eyp-looks-one-way") for p in same_line]
    cands += [Distractor(option(p)) for p in pairs[1:]]
    working = [
        f"Count the steps along the squares for each pair. {option(best).capitalize()} are {best[0]} steps apart."
    ]
    return _mcq(rng.choice(_CLOSEST_PAIR).format(n=rng.choice(NAMES)), option(best), cands, fig, working)


def _read_map(q) -> str:
    pics = symbols_of(q.figure)
    where = {word: cell for cell, word in pics.items()}
    text = q.text
    if "two places" in text:
        cells = list(pics)
        best = min(((_dist(a, b), a, b) for i, a in enumerate(cells) for b in cells[i + 1 :]), key=lambda t: t[0])
        return _pair_text(pics[best[1]], pics[best[2]])
    if "closest" in text or "nearest" in text or "farthest" in text:
        (x,) = _names_in(text)
        others = [c for c in pics if c != where[x]]
        pick = max if "farthest" in text else min
        return pics[pick(others, key=lambda c: _dist(where[x], c))]
    axis, pick = (
        (1, max) if "top" in text else (1, min) if "bottom" in text else (0, min) if "left" in text else (0, max)
    )
    return pics[pick(pics, key=lambda c: c[axis])]


figures.register_reader("ey_map_reading", _read_map)
