"""Exam chains for the Years 3-6 shape, position and transformation skills and the scale-plan skill (``w7f``).

Five chains, each read off one drawing on a square grid (``grid_drawing``). The practice tiers of four of these skills
ask the learner to shade or draw an answer; the chains set the read forms of the same ideas (choose, count, work out),
because a drawn answer cannot yet be marked as part of a multi-part question.

* ``exd_x_w7f_model_plan_chain`` (``f10-model-plans-cube-counts``, Year 3): the plan of a cube model with one stack
  marked ``?`` and the total given. (a) the cubes in the marked stack, (b) the cubes to add so every stack is as tall
  as the tallest (follows (a)), (c) the cubes in the model after that (follows (b)).
* ``exd_x_w7f_plan_views_chain`` (``f10-two-dimensional-object-representations``, Years 3 and 5): the plan of a stack
  with the cubes in each column written on it. (a) the cubes in the stack, (b) the column heights of the front view
  (``mcq``), (c) of the right-side view (``mcq``), (d) the squares in the front view.
* ``exd_x_w7f_turn_pictures_chain`` (``f10-turn-shapes-on-a-grid``, Year 5): four pictures of a shaded shape and a
  hatched image, each with the point P. (a) the picture that shows a stated quarter turn, (b) a half turn, (c) two
  turns one after the other (all ``mcq``).
* ``exd_x_w7f_box_net_chain`` (``f10-box-nets-on-a-grid``, Year 6): the net of a box drawn on a grid. (a) its surface
  area, (b) the squares in the top face, (c) the height of the box, (d) its volume (follows (b) and (c)).
* ``exd_x_w7f_scale_plan_chain`` (``senior-construct-scale-drawings-by-hand-and-by-using``): a room described in
  metres and a blank grid with its scale. (a) the cells across the plan, (b) up the plan, (c) the squares an L-shaped
  plan covers (follows (a) and (b)), (d) the perimeter of the plan in centimetres (follows (a) and (b)).

Each answer is built from the drawn shape first; the reader registered for each topic re-derives every part from the
figure and the prompts alone, with its own turning and counting.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import figures
from mathgen.core import check, generator
from mathgen.figure_kinds.grid import grid_drawing

from . import grid_shading_build as gsb
from ._w7f_kit import finish_parts, mcq_part, number_part, pick, retry, split_parts
from .grid_shapes_y36_build import _asymmetric_tile, _box_net, _model_figure, _model_heights, _turn

Cell = tuple[int, int]


def _only_exam(tier: str) -> None:
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")


# ------------------------------------------------------------------------------------------- 1. a cube model

_MODEL_A = "How many cubes are in the stack marked with a question mark?"
_MODEL_B = (
    "More cubes are added so that every stack is as tall as the tallest stack. Bare squares stay bare. How many "
    "cubes are added?"
)
_MODEL_C = "How many cubes are in the model once those cubes are added?"
_MODEL_TOTAL = re.compile(r"built from (\d+) cubes")


def _model_chain(rng):
    w, d = rng.choice(((3, 3), (4, 2), (4, 3), (3, 4)))
    h = _model_heights(rng, w, d, rng.randint(0, 2), 5)
    if h is None:
        return None
    stacks = {s: v for s, v in h.items() if v}
    if len(stacks) < 4:
        return None
    unknown = rng.choice(sorted(stacks))
    a = stacks[unknown]
    known = {s: v for s, v in stacks.items() if s != unknown}
    tallest = max(known.values())
    if tallest < a:  # the marked stack must not be the tallest, so the tallest is readable from the plan
        return None
    n, total = len(stacks), sum(stacks.values())
    known_sum = total - a
    add = tallest * n - total
    if add < 2:
        return None
    fig = _model_figure(h, w, d, unknown)
    stem = pick(
        rng,
        (
            f"A model is built from {total} cubes. The plan shows the number of cubes in each stack, except for one "
            "stack marked with a question mark. A bare square has no cubes.",
            f"The plan of a model built from {total} cubes shows how many cubes are stacked on each square. One stack "
            "is marked with a question mark, and a bare square has no cubes on it.",
            f"This plan is for a model built from {total} cubes. Each number tells how many cubes are in that stack, "
            "a bare square holds none, and the stack marked with a question mark is not shown.",
        ),
    )
    part_a = number_part(
        _MODEL_A,
        a,
        1,
        [
            f"Add the stacks that show a number: {' + '.join(str(v) for v in known.values())} = {known_sum}.",
            f"{total} - {known_sum} = {a}.",
        ],
        [
            (known_sum, "mod-gives-the-cubes-already-counted"),
            (total, "mod-gives-the-total"),
            (total - n + 1, "mod-treats-every-stack-as-one-cube"),
            (a + 1, "stk-count-off-by-one"),
            (a - 1, "stk-count-off-by-one"),
        ],
    )
    part_b = number_part(
        _MODEL_B,
        add,
        2,
        [
            f"The tallest stack has {tallest} cubes and {n} squares have stacks, so the model would hold {tallest * n}.",
            f"{tallest * n} - {total} = {add} cubes are added.",
        ],
        [
            (tallest * n, "mod-gives-the-total-after-adding"),
            (total, "mod-gives-the-total"),
            (tallest * w * d - total, "mod-builds-on-the-bare-squares"),
            (tallest * n - known_sum, "mod-forgets-the-marked-stack"),
            (add + 1, "stk-count-off-by-one"),
        ],
        follow=f"{tallest * n - known_sum}-part_a",
    )
    part_c = number_part(
        _MODEL_C,
        tallest * n,
        1,
        [f"{total} + {add} = {tallest * n}, which is {tallest} x {n}."],
        [
            (add, "mod-gives-the-cubes-added"),
            (total, "mod-gives-the-total"),
            (tallest * w * d, "mod-builds-on-the-bare-squares"),
            (total - add, "mod-takes-away-the-cubes-added"),
            (tallest * n + 1, "stk-count-off-by-one"),
        ],
        follow=f"{total}+part_b",
    )
    return finish_parts(fig, stem, [part_a, part_b, part_c])


@generator("exd_x_w7f_model_plan_chain", level=3, presets={"exam": {"tier": "exam"}})
def exd_x_w7f_model_plan_chain(rng, tier: str = "exam"):
    """The marked stack, the cubes added to level every stack, and the cubes in the finished model."""
    _only_exam(tier)
    return retry(rng, _model_chain, "a cube model plan with three distractors per part")


def _read_model_chain(question) -> str:
    fig = question.figure
    numbers = {(int(lb["at"][0]), int(lb["at"][1])): lb["text"] for lb in fig["labels"]}
    total = int(_MODEL_TOTAL.search(question.text).group(1))
    known = [int(v) for v in numbers.values() if v != "?"]
    check(sum(1 for v in numbers.values() if v == "?") == 1, "exactly one stack is marked")
    a = total - sum(known)
    n = len(numbers)
    tallest = max(known)
    return f"(a) {a}; (b) {tallest * n - total}; (c) {tallest * n}"


figures.register_reader("exd_x_w7f_model_plan_chain", _read_model_chain)

# ------------------------------------------------------------------------------------------- 2. views from a plan

_PLAN_CUBES = "How many cubes make up the stack?"
_PLAN_FRONT = "Which option gives the column heights of the view from the front, from left to right?"
_PLAN_RIGHT = (
    "Which option gives the column heights of the view from the right side, from left to right (the front of the "
    "stack is on your left)?"
)


def _count(n: int, noun: str) -> str:
    """``n`` with its noun, singular for exactly one."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


_PLAN_FRONT_SQUARES = "How many squares make up the view from the front?"
_PLAN_ALT = (
    "A plan of a stack of cubes on a square grid: each shaded square has the number of cubes in its column written on "
    "it."
)


def _heights_line(heights) -> str:
    return "Column heights, left to right: " + ", ".join(str(v) for v in heights)


def _ranked(rng, key, wrong):
    """The wrong answers ordered so that three of them, with the key at a random rank among the four, come first.

    ``wrong`` is ``(value, misconception id)``. A number part keeps the first wrong answers that survive, so a fixed
    order would put the key at the same rank every time, which a learner could learn without the maths.
    """
    seen, lo, hi = set(), [], []
    for value, tag in wrong:
        if value is None or value == key or value in seen:
            continue
        seen.add(value)
        (lo if value < key else hi).append((value, tag))
    rng.shuffle(lo)
    rng.shuffle(hi)
    ranks = [r for r in range(4) if r <= len(lo) and 3 - r <= len(hi)]
    r = rng.choice(ranks) if ranks else 0
    head = lo[:r] + hi[: 3 - r]
    return head + [w for w in lo + hi if w not in head]


def _plan_views_chain(rng):
    w, d = rng.randint(3, 5), rng.randint(2, 4)
    h = {(x, y): (0 if rng.random() < 0.25 else rng.randint(1, 4)) for x in range(w) for y in range(d)}
    if any(all(h[(x, y)] == 0 for y in range(d)) for x in range(w)):
        return None
    if any(all(h[(x, y)] == 0 for x in range(w)) for y in range(d)):
        return None
    n = sum(h.values())
    front = [max(h[(x, y)] for y in range(d)) for x in range(w)]
    right = [max(h[(x, y)] for x in range(w)) for y in range(d)]
    front_sum = [sum(h[(x, y)] for y in range(d)) for x in range(w)]
    right_sum = [sum(h[(x, y)] for x in range(w)) for y in range(d)]
    near_front = [h[(x, 0)] for x in range(w)]
    near_right = [h[(w - 1, y)] for y in range(d)]
    if len({tuple(front), tuple(front[::-1]), tuple(right), tuple(right[::-1])}) != 4:
        return None
    footprint = [[x, y] for (x, y), v in sorted(h.items()) if v]
    fig = grid_drawing(
        alt=_PLAN_ALT,
        grid={"type": "square", "width": w, "height": d},
        cells=[{"cells": footprint, "style": "shaded"}],
        labels=[{"at": [x + 0.5, y + 0.5], "text": str(v)} for (x, y), v in sorted(h.items()) if v],
    )
    stem = pick(
        rng,
        (
            "The plan of a stack of unit cubes is shown, seen from above. The number in a square is how many cubes "
            "are stacked there, a blank square has none, and the bottom of the plan is the front of the stack.",
            "Looking down on a stack of unit cubes gives this plan. Each number counts the cubes in that column, a "
            "blank square is empty, and the front of the stack is at the bottom of the plan.",
            "A plan view of a cube stack is drawn on a square grid. A number says how many cubes are piled on that "
            "square, an empty square has none, and the bottom edge of the plan is the front.",
        ),
    )
    part_a = number_part(
        _PLAN_CUBES,
        n,
        1,
        [f"Add the numbers: {' + '.join(str(v) for v in h.values() if v)} = {n}."],
        _ranked(
            rng,
            n,
            [
                (len(footprint), "plan-counts-the-squares-not-the-cubes"),
                (max(h.values()) * len(footprint), "plan-makes-every-stack-the-tallest"),
                (max(h.values()) * w * d, "plan-makes-every-stack-the-tallest"),
                (n + 1, "stk-count-off-by-one"),
                (n - 1, "stk-count-off-by-one"),
                (n + 2, "stk-count-off-by-one"),
                (n - 2, "stk-count-off-by-one"),
                (n - 3, "stk-count-off-by-one"),
            ],
        ),
    )
    part_b = mcq_part(
        _PLAN_FRONT,
        _heights_line(front),
        2,
        [f"From the front, each column is as tall as the tallest stack in line with it: {', '.join(map(str, front))}."],
        [
            (_heights_line(front[::-1]), "elev-gives-the-back-view"),
            (_heights_line(right), "elev-gives-the-side-view"),
            (_heights_line(front_sum), "view-adds-the-cubes-in-a-line"),
            (_heights_line(near_front), "view-reads-only-the-nearest-row"),
        ],
    )
    part_c = mcq_part(
        _PLAN_RIGHT,
        _heights_line(right),
        2,
        [
            f"From the right side, each column is as tall as the tallest stack in line with it: {', '.join(map(str, right))}."
        ],
        [
            (_heights_line(right[::-1]), "elev-gives-the-opposite-side-view"),
            (_heights_line(front), "elev-gives-the-front-view"),
            (_heights_line(right_sum), "view-adds-the-cubes-in-a-line"),
            (_heights_line(near_right), "view-reads-only-the-nearest-row"),
        ],
    )
    part_d = number_part(
        _PLAN_FRONT_SQUARES,
        sum(front),
        1,
        [f"Add the column heights of the front view: {' + '.join(map(str, front))} = {sum(front)}."],
        _ranked(
            rng,
            sum(front),
            [
                (n, "elev-counts-every-cube-for-a-view"),
                (len(front), "stk-counts-the-columns"),
                (sum(right), "elev-gives-the-side-view"),
                (max(front) * w, "plan-makes-every-stack-the-tallest"),
                (sum(front) + 1, "stk-count-off-by-one"),
                (sum(front) - 1, "stk-count-off-by-one"),
                (sum(front) + 2, "stk-count-off-by-one"),
                (sum(front) - 2, "stk-count-off-by-one"),
            ],
        ),
    )
    return finish_parts(fig, stem, [part_a, part_b, part_c, part_d])


@generator("exd_x_w7f_plan_views_chain", level=5, presets={"exam": {"tier": "exam"}})
def exd_x_w7f_plan_views_chain(rng, tier: str = "exam"):
    """Cubes in a stack, and the front and right-side views, from its plan with the columns counted."""
    _only_exam(tier)
    return retry(rng, _plan_views_chain, "a plan with differing views and enough distractors")


def _read_plan_views_chain(question) -> str:
    fig = question.figure
    parts = split_parts(question.text)
    check(parts["a"] == _PLAN_CUBES and parts["b"] == _PLAN_FRONT and parts["c"] == _PLAN_RIGHT, "unknown prompts")
    w, d = fig["grid"]["width"], fig["grid"]["height"]
    h = {(x, y): 0 for x in range(w) for y in range(d)}
    for lb in fig["labels"]:
        h[(int(lb["at"][0]), int(lb["at"][1]))] = int(lb["text"])
    front = [max(h[(x, y)] for y in range(d)) for x in range(w)]
    right = [max(h[(x, y)] for x in range(w)) for y in range(d)]
    return f"(a) {sum(h.values())}; (b) {_heights_line(front)}; (c) {_heights_line(right)}; (d) {sum(front)}"


figures.register_reader("exd_x_w7f_plan_views_chain", _read_plan_views_chain)

# ------------------------------------------------------------------------------------------- 3. turn pictures

_TURN_NAMES = {"cw": "a quarter turn clockwise", "ccw": "a quarter turn anticlockwise", "half": "a half turn"}
_TURN_QUARTERS = {"cw": 3, "ccw": 1, "half": 2}  # quarter turns anticlockwise
_COMPOSITIONS = {  # (first, second) -> the single turn it makes
    ("cw", "half"): "ccw",
    ("half", "cw"): "ccw",
    ("ccw", "half"): "cw",
    ("half", "ccw"): "cw",
    ("cw", "cw"): "half",
    ("ccw", "ccw"): "half",
}
_PICTURE_LETTERS = "ABCDEFGHJK"  # no P: the point is called P
_TURN_STEMS = (
    "Each of the four pictures {names} shows the same shaded shape, the same point P and a hatched shape that is a "
    "possible image of the shaded shape.",
    "The four pictures {names} all start from the same shaded shape and the same point P. In each, the hatched shape "
    "might be the image of the shaded shape.",
    "Four pictures, {names}, are drawn on grids. Each one has the shaded shape, the point P and a hatched shape that "
    "may be the shaded shape after a transformation.",
)
_ASK_TURN = "Which picture shows the hatched shape as the image of the shaded shape after {turn} about P?"
_ASK_THEN = (
    "The shaded shape is turned {first} about P and then {second} about P. Which picture shows the hatched shape as "
    "its image after both turns?"
)
_ASK_RE = re.compile(r"after (a quarter turn clockwise|a quarter turn anticlockwise|a half turn) about P\?")
_THEN_RE = re.compile(
    r"turned (a quarter turn clockwise|a quarter turn anticlockwise|a half turn) about P and then "
    r"(a quarter turn clockwise|a quarter turn anticlockwise|a half turn) about P"
)
_NAME_TO_KEY = {v: k for k, v in _TURN_NAMES.items()}


def _turn_chain(rng):
    tile = _asymmetric_tile(rng, rng.choice((4, 5)))
    corners = sorted({(x + a, y + b) for x, y in tile for a in (0, 1) for b in (0, 1)})
    p = rng.choice(corners)
    images = {k: _turn(tile, p, q) for k, q in _TURN_QUARTERS.items()}
    px, py = p
    spare = [
        gsb._reflect_x(tile, px),
        gsb._reflect_y(tile, py),
        _turn(tile, (px + 1, py), 3),
        _turn(tile, (px, py + 1), 1),
        _turn(tile, (px - 1, py), 2),
    ]
    rng.shuffle(spare)
    pool = [s for s in spare if s not in images.values() and s != tile and not (s & tile)]
    if not pool or any(i & tile for i in images.values()):
        return None
    odd = pool[0]
    cells_of = {**images, "odd": odd}
    allx = [x for cs in (tile, *cells_of.values()) for x, _ in cs] + [px]
    ally = [y for cs in (tile, *cells_of.values()) for _, y in cs] + [py]
    lo_x, hi_x, lo_y, hi_y = min(allx), max(allx), min(ally), max(ally)
    pw, ph = hi_x - lo_x + 3, hi_y - lo_y + 3  # one blank square round each picture, room for its letter
    if pw > 14 or ph > 14:
        return None
    order = rng.sample(sorted(cells_of), 4)
    letter_set = sorted(rng.sample(_PICTURE_LETTERS, 4))
    letters = dict(zip(letter_set, order))
    shaded, hatched, labels = [], [], []
    for i, (letter, key) in enumerate(letters.items()):
        ox, oy = (i % 2) * pw, (1 - i // 2) * ph
        sx, sy = ox + 1 - lo_x, oy + 1 - lo_y
        shaded += [[x + sx, y + sy] for x, y in sorted(tile)]
        hatched += [[x + sx, y + sy] for x, y in sorted(cells_of[key])]
        labels.append({"at": [px + sx, py + sy], "text": "P"})
        labels.append({"at": [ox + 0.5, oy + ph - 0.5], "text": letter})
    fig = grid_drawing(
        alt="Four square grids, lettered A to D, each with a shaded shape, a hatched shape and a point marked P.",
        grid={"type": "square", "width": 2 * pw, "height": 2 * ph},
        cells=[{"cells": shaded, "style": "shaded"}, {"cells": hatched, "style": "hatched"}],
        labels=labels,
    )
    letter_of = {key: letter for letter, key in letters.items()}
    names = {key: f"Picture {letter}" for key, letter in letter_of.items()}
    x_key, y_key, z_key = rng.sample(["cw", "ccw", "half"], 3)
    compositions = [pair for pair, out in _COMPOSITIONS.items() if out == z_key]
    first, second = rng.choice(compositions)
    stem = pick(rng, _TURN_STEMS).format(names=", ".join(letter_set[:-1]) + " and " + letter_set[-1])
    wrong_for = {
        "cw": [
            ("ccw", "turn-direction-reversed"),
            ("half", "turn-wrong-amount"),
            ("odd", "turn-reflection-for-a-turn"),
        ],
        "ccw": [
            ("cw", "turn-direction-reversed"),
            ("half", "turn-wrong-amount"),
            ("odd", "turn-reflection-for-a-turn"),
        ],
        "half": [("cw", "turn-wrong-amount"), ("ccw", "turn-wrong-amount"), ("odd", "turn-reflection-for-a-turn")],
    }

    def ask(key, prompt, working, marks, extra=()):
        wrong = [(names[k], tag) for k, tag in [*extra, *wrong_for[key]] if k != key]
        return mcq_part(prompt, names[key], marks, working, wrong)

    part_a = ask(
        x_key,
        _ASK_TURN.format(turn=_TURN_NAMES[x_key]),
        [
            f"Turn each square of the shaded shape about P, not about its own middle: {_TURN_NAMES[x_key]} gives {names[x_key]}."
        ],
        1,
    )
    part_b = ask(
        y_key,
        _ASK_TURN.format(turn=_TURN_NAMES[y_key]),
        [f"{_TURN_NAMES[y_key].capitalize()} about P gives {names[y_key]}."],
        1,
    )
    part_c = ask(
        z_key,
        _ASK_THEN.format(first=_TURN_NAMES[first], second=_TURN_NAMES[second]),
        [
            f"{_TURN_NAMES[first].capitalize()} then {_TURN_NAMES[second]} about the same point is {_TURN_NAMES[z_key]} altogether.",
            f"That is {names[z_key]}.",
        ],
        2,
        extra=[(first, "turn-uses-only-the-first-turn"), (second, "turn-uses-only-the-second-turn")],
    )
    return finish_parts(fig, stem, [part_a, part_b, part_c])


@generator("exd_x_w7f_turn_pictures_chain", level=5, presets={"exam": {"tier": "exam"}})
def exd_x_w7f_turn_pictures_chain(rng, tier: str = "exam"):
    """Pick the picture that shows a quarter turn, a half turn, and two turns one after the other."""
    _only_exam(tier)
    return retry(rng, _turn_chain, "four turn pictures that do not overlap the original")


def _spin(cells, p, quarters):
    """Cells turned ``quarters`` quarter turns anticlockwise about the lattice point ``p``, with arithmetic of its own."""
    out = set()
    for x, y in cells:
        dx, dy = 2 * x + 1 - 2 * p[0], 2 * y + 1 - 2 * p[1]  # the square's centre from P, doubled
        for _ in range(quarters % 4):
            dx, dy = -dy, dx
        out.add(((2 * p[0] + dx - 1) // 2, (2 * p[1] + dy - 1) // 2))
    return out


def _read_turn_chain(question) -> str:
    fig = question.figure
    parts = split_parts(question.text)
    pw, ph = fig["grid"]["width"] // 2, fig["grid"]["height"] // 2
    shaded = {tuple(c) for c in fig["cells"][0]["cells"]}
    hatched = {tuple(c) for c in fig["cells"][1]["cells"]}
    seen = {}
    for lb in fig["labels"]:
        if len(lb["text"]) == 1 and lb["text"] in _PICTURE_LETTERS:
            ox, oy = int(lb["at"][0] - 0.5), int(lb["at"][1] - ph + 0.5)
            inside = lambda c: ox <= c[0] < ox + pw and oy <= c[1] < oy + ph  # noqa: E731
            (p,) = [(int(q["at"][0]), int(q["at"][1])) for q in fig["labels"] if q["text"] == "P" and inside(q["at"])]
            tile = {c for c in shaded if inside(c)}
            image = {c for c in hatched if inside(c)}
            kinds = [k for k, q in _TURN_QUARTERS.items() if _spin(tile, p, q) == image]
            seen[lb["text"]] = kinds[0] if kinds else "odd"
    by_kind = {k: letter for letter, k in seen.items()}
    check(set(by_kind) == {"cw", "ccw", "half", "odd"}, "the four pictures are not the three turns and one other")
    ans = []
    for label in "ab":
        ans.append(f"Picture {by_kind[_NAME_TO_KEY[_ASK_RE.search(parts[label]).group(1)]]}")
    first, second = (_NAME_TO_KEY[g] for g in _THEN_RE.search(parts["c"]).groups())
    ans.append(f"Picture {by_kind[_COMPOSITIONS[(first, second)]]}")
    return f"(a) {ans[0]}; (b) {ans[1]}; (c) {ans[2]}"


figures.register_reader("exd_x_w7f_turn_pictures_chain", _read_turn_chain)

# ------------------------------------------------------------------------------------------- 4. the net of a box

_NET_AREA = "How many squares make up the whole net? This is the surface area of the box, in square units."
_NET_TOP = "The top and the bottom of the box are the same size. How many squares make up the top face, the one joined above the row of four side faces?"
_NET_HEIGHT = "How tall is the box, in squares?"
_NET_VOLUME = (
    "The box is filled with unit cubes, one for each square of height on its base. How many cubes does it hold?"
)


def _box_net_chain(rng):
    a, b = rng.sample(range(1, 9), 2)
    h = rng.randint(1, 6)
    top_on, bottom_on = rng.randrange(4), rng.randrange(4)
    strip, top, bottom, _starts = _box_net(a, b, h, top_on, bottom_on)
    net = strip | top | bottom
    lo_y = min(y for _, y in net)
    shift = -lo_y + rng.randint(0, 1)
    placed = {(x, y + shift) for x, y in net}
    width = max(x for x, _ in placed) + 1 + rng.randint(0, 1)
    height = max(y for _, y in placed) + 1 + rng.randint(0, 1)
    if width > 24 or height > 24:
        return None
    area = len(net)
    top_squares = a * b
    volume = a * b * h
    check(len(top) == top_squares and area == 2 * (a * b + a * h + b * h), "the net does not match the box")
    fig = grid_drawing(
        alt="The net of a box drawn in shaded squares on a square grid.",
        grid={"type": "square", "width": width, "height": height},
        cells=[{"cells": [list(c) for c in sorted(placed)], "style": "shaded"}],
    )
    stem = pick(
        rng,
        (
            "The shaded squares on the grid are the net of a closed box. It folds up with no gaps or overlaps.",
            "A closed box is made by folding the shaded squares of this grid. The net has no gaps or overlaps when it "
            "is folded.",
            "On the grid, the shaded squares form a net. Folded up, it makes a closed box with no gaps or overlaps.",
        ),
    )
    side_strip = 2 * (a + b) * h
    part_a = number_part(
        _NET_AREA,
        area,
        1,
        [
            f"Count the shaded squares: {side_strip} in the row of side faces plus 2 x {top_squares} for the top and bottom."
        ],
        [
            (side_strip, "net-leaves-out-the-top-and-bottom"),
            (area - top_squares, "net-leaves-out-the-top-and-bottom"),
            (width * height, "net-gives-the-sheet-for-the-surface"),
            (area // 2, "net-counts-one-of-each-pair-of-faces"),
            (volume, "net-gives-the-volume-for-the-surface"),
        ],
    )
    part_b = number_part(
        _NET_TOP,
        top_squares,
        1,
        [f"The top is {_count(a, 'square')} by {_count(b, 'square')}: {a} x {b} = {top_squares}."],
        [
            (a + b, "net-adds-the-edges"),
            (2 * (a + b), "net-gives-the-perimeter-of-the-top"),
            (top_squares + a * h, "net-adds-a-face-to-the-volume"),
            (area, "net-gives-the-sheet-for-the-surface"),
            (a * h, "net-gives-a-side-face-for-the-top"),
        ],
    )
    part_c = number_part(
        _NET_HEIGHT,
        h,
        1,
        [f"Each side face in the row is {_count(h, 'square')} tall, so the box is {_count(h, 'square')} tall."],
        [
            (a, "net-gives-another-edge"),
            (b, "net-gives-another-edge"),
            (a + b, "net-adds-the-edges"),
            (h + 1, "stk-count-off-by-one"),
            (2 * (a + b), "net-gives-the-width-of-the-row"),
        ],
    )
    part_d = number_part(
        _NET_VOLUME,
        volume,
        2,
        [
            f"The base has {_count(top_squares, 'square')} and the box is {_count(h, 'square')} tall: {top_squares} x {h} = {volume}."
        ],
        [
            (top_squares + h, "net-adds-a-face-to-the-volume"),
            (area, "net-gives-the-surface-for-the-volume"),
            (side_strip, "net-gives-the-row-for-the-volume"),
            (volume + top_squares, "net-adds-a-face-to-the-volume"),
            (a * b * (h + 1), "stk-count-off-by-one"),
        ],
        follow="part_b*part_c",
    )
    return finish_parts(fig, stem, [part_a, part_b, part_c, part_d])


@generator("exd_x_w7f_box_net_chain", level=6, presets={"exam": {"tier": "exam"}})
def exd_x_w7f_box_net_chain(rng, tier: str = "exam"):
    """Surface area, top face, height and volume of a box read from its net drawn on a grid."""
    _only_exam(tier)
    return retry(rng, _box_net_chain, "a box net that fits the grid")


def _read_box_net_chain(question) -> str:
    cells = {tuple(c) for c in question.figure["cells"][0]["cells"]}
    rows: dict[int, int] = {}
    for _x, y in cells:
        rows[y] = rows.get(y, 0) + 1
    widest = max(rows.values())
    height = sum(1 for n in rows.values() if n == widest)
    top = (len(cells) - widest * height) // 2
    check(2 * top + widest * height == len(cells), "the net is not a row of four sides with a top and a bottom")
    return f"(a) {len(cells)}; (b) {top}; (c) {height}; (d) {top * height}"


figures.register_reader("exd_x_w7f_box_net_chain", _read_box_net_chain)

# ------------------------------------------------------------------------------------------- 5. a scale plan

_PLAN_SCALES = (
    (100, "1 cm : 1 m"),
    (200, "1 cm : 2 m"),
    (500, "1 cm : 5 m"),
    (50, "1 cm : 50 cm"),
    (100, "1 : 100"),
    (200, "1 : 200"),
    (500, "1 : 500"),
)
_CORNERS = ("top left", "top right", "bottom left", "bottom right")
_ACROSS_UP = re.compile(r"(\d+(?:\.\d+)?) m across the page and (\d+(?:\.\d+)?) m up the page")
_MISSING = re.compile(r"A rectangle (\d+(?:\.\d+)?) m across the page and (\d+(?:\.\d+)?) m up the page is missing")
_PLAN_ASK_A = "How many cells across the grid is the plan, at its widest?"
_PLAN_ASK_B = "How many cells up the grid is the plan, at its tallest?"
_PLAN_ASK_C = "How many cells of the grid does the L-shaped plan cover?"
_PLAN_ASK_D = "What is the distance all the way round the plan, in centimetres?"


def _m_text(value: Fraction) -> str:
    return str(int(value)) if value.denominator == 1 else f"{float(value):g}"


def _scale_plan_chain(rng):
    per_cm, scale_text = rng.choice(_PLAN_SCALES)
    step = Fraction(per_cm, 100)  # metres of the real room for each cell
    a, b = rng.randint(5, 9), rng.randint(4, 7)
    c, d = rng.randint(1, a - 2), rng.randint(1, b - 2)
    corner = rng.choice(_CORNERS)
    gw, gh = rng.randint(11, 14), rng.randint(9, 11)
    thing = rng.choice(("workshop", "classroom", "garden bed", "courtyard", "storeroom"))
    stem = (
        f"An L-shaped {thing} is to be drawn to scale on the grid, using the scale shown. Each cell of the grid is "
        f"1 cm on the drawing. The {thing} is {_m_text(a * step)} m across the page and {_m_text(b * step)} m up "
        f"the page at its widest and tallest. A rectangle {_m_text(c * step)} m across the page and "
        f"{_m_text(d * step)} m up the page is missing from the {corner} corner."
    )
    fig = grid_drawing(
        alt="A blank square grid with a scale written beside it, ready for a scale drawing.",
        grid={"type": "square", "width": gw, "height": gh},
        scale={"text": scale_text},
    )
    covered = a * b - c * d
    perimeter = 2 * (a + b)
    real_a = a * step
    part_a = number_part(
        _PLAN_ASK_A,
        a,
        1,
        [
            f"{_m_text(a * step)} m at {scale_text} is {_m_text(a * step)} / {_m_text(step)} = {a} cm, which is {a} cells."
        ],
        [
            (int(real_a) if real_a.denominator == 1 else None, "scd-length-unscaled"),
            (a * per_cm if per_cm else None, "scd-multiplies-instead-of-divides"),
            (a + 1, "scd-off-by-one"),
            (a - 1, "scd-off-by-one"),
            (b, "scd-wrong-wall"),
        ],
    )
    part_b = number_part(
        _PLAN_ASK_B,
        b,
        1,
        [f"{_m_text(b * step)} m at {scale_text} is {b} cm, which is {b} cells."],
        [
            (int(b * step) if (b * step).denominator == 1 else None, "scd-length-unscaled"),
            (b + 1, "scd-off-by-one"),
            (b - 1, "scd-off-by-one"),
            (a, "scd-wrong-wall"),
            (b * 2, "scd-doubles-drawn-length"),
        ],
    )
    part_c = number_part(
        _PLAN_ASK_C,
        covered,
        2,
        [
            f"The rectangle round the plan is {a} by {b} = {_count(a * b, 'cell')}.",
            f"The missing corner is {c} by {d} = {_count(c * d, 'cell')}, so the plan covers {a * b} - {c * d} = {_count(covered, 'cell')}.",
        ],
        [
            (a * b, "plan-forgets-the-missing-corner"),
            (c * d, "plan-counts-the-missing-corner"),
            (a * b + c * d, "scd-adds-instead-of-subtracts"),
            (a * b - c - d, "plan-takes-away-the-sides-of-the-corner"),
            (perimeter, "plan-perimeter-for-area"),
        ],
        follow=f"part_a*part_b-{c * d}",
    )
    part_d = number_part(
        _PLAN_ASK_D,
        perimeter,
        2,
        [
            "Cutting a rectangular corner out of a rectangle does not change the distance round it: the two edges of "
            "the notch are as long as the two edges they replace.",
            f"2 x ({a} + {b}) = {perimeter} cm.",
        ],
        [
            (perimeter + 2 * (c + d), "plan-adds-the-notch-to-the-perimeter"),
            (perimeter - 2 * (c + d), "plan-takes-the-notch-off-the-perimeter"),
            (a + b, "plan-gives-half-the-perimeter"),
            (covered, "plan-area-for-perimeter"),
            (a * b, "plan-area-for-perimeter"),
        ],
        follow="2*(part_a+part_b)",
    )
    return finish_parts(fig, stem, [part_a, part_b, part_c, part_d])


@generator("exd_x_w7f_scale_plan_chain", level=11, presets={"exam": {"tier": "exam"}})
def exd_x_w7f_scale_plan_chain(rng, tier: str = "exam"):
    """Cells across and up for an L-shaped room at a stated scale, the cells it covers and the distance round it."""
    _only_exam(tier)
    return retry(rng, _scale_plan_chain, "a scale plan with three distractors per part")


_SCALE_WORDS = re.compile(r"^1 cm : (\d+) (cm|m)$|^1 : (\d+)$")


def _read_scale_plan_chain(question) -> str:
    text = question.figure["scale"]["text"]
    m = _SCALE_WORDS.match(text)
    check(m is not None, f"cannot read the scale {text!r}")
    per_cm = int(m.group(3)) if m.group(3) else int(m.group(1)) * (100 if m.group(2) == "m" else 1)
    pairs = [(Fraction(x), Fraction(y)) for x, y in _ACROSS_UP.findall(question.text)]
    cells = lambda metres: int(metres * 100 / per_cm)  # noqa: E731
    a, b = cells(pairs[0][0]), cells(pairs[0][1])
    mc, md = _MISSING.search(question.text).groups()
    c, d = cells(Fraction(mc)), cells(Fraction(md))
    return f"(a) {a}; (b) {b}; (c) {a * b - c * d}; (d) {2 * (a + b)}"


figures.register_reader("exd_x_w7f_scale_plan_chain", _read_scale_plan_chain)
