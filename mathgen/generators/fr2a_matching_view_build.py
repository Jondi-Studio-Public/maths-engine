"""``fr2a_matching_view`` and ``exd_x_fr2a_matching_view_chain``: match a plan view and its elevations.

Skill ``f10-pick-the-matching-view`` (AC9M7SP01). A structure of identical cubes is described by a plan view: a grid of
squares with the number of cubes stacked in each square, the front of the structure being the bottom edge of the plan.
Everything is drawn in one picture with the existing ``cartesian`` kind: the plan, and four lettered panels of squares
(the options). The answer is ``Option A`` to ``Option D``.

Tiers:

- easy:   pick the front elevation (the view from the bottom edge: the tallest stack in each column).
- medium: pick the right-side elevation. Looking from the right, the nearest column hides the ones behind it, the
          squares left to right are the rows from the front to the back, and taller stacks behind a short one still show.
- hard:   the reverse: the front and right-side elevations are drawn, and four plans are offered. Only one plan gives
          both elevations (every wrong plan fails at least one of them).

Each wrong panel is built from one named slip: the other side's view, the view from behind (mirror image), column totals
in place of the tallest stack, only the nearest stacks, and so on.

``exd_x_fr2a_matching_view_chain`` is the exam tier: (a) pick the front elevation, (b) the number of squares in the
right-side elevation, (c) the total number of cubes, (d) the number of cubes that are not shown as a square in the
right-side elevation (follow-through from (b) and (c)).
"""

from __future__ import annotations

import re

from mathgen import figures
from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, latex_answer, phrase, pick_distractors
from mathgen.figures import cartesian, register_reader

from . import _fr2a_kit as kit
from .exd_x_sc3_support import number_part, redraw, tagged

_THINGS = ("tower", "wall", "stepped model", "block model", "structure", "sculpture", "display", "building")
_WHO = ("Mia", "Tom", "Aiko", "Dev", "Leila", "Sam", "Jess", "Kofi", "Priya", "Marcus", "Zoe", "Ben")
_LETTERS = "ABCD"
_GAP = 2  # between panels
_Y0 = 0.0


# ------------------------------------------------------------------------------------------ views from the grid
# grid[y][x]: y = 0 is the front row (the bottom edge of the plan), x = 0 the left column.


def front(grid):
    """Front elevation: the tallest stack in each column, left to right."""
    return tuple(max(grid[y][x] for y in range(len(grid))) for x in range(len(grid[0])))


def right(grid):
    """Right-side elevation: the tallest stack in each row; left to right on the page is front row to back row."""
    return tuple(max(row) for row in grid)


def front_by_layers(grid):
    """The front elevation built layer by layer: at each height, which columns have a cube."""
    top = max(max(row) for row in grid)
    out = [0] * len(grid[0])
    for level in range(1, top + 1):
        for x in range(len(grid[0])):
            if any(grid[y][x] >= level for y in range(len(grid))):
                out[x] += 1
    return tuple(out)


def right_by_layers(grid):
    top = max(max(row) for row in grid)
    out = [0] * len(grid)
    for level in range(1, top + 1):
        for y in range(len(grid)):
            if any(grid[y][x] >= level for x in range(len(grid[0]))):
                out[y] += 1
    return tuple(out)


def total(grid) -> int:
    return sum(sum(row) for row in grid)


# ------------------------------------------------------------------------------------------ drawing


def _cell_segments(x0, y0, heights):
    """Outline segments of the unit squares of an elevation, each shared edge once."""
    seen = set()
    for j, h in enumerate(heights):
        for level in range(h):
            x, y = x0 + j, y0 + level
            for a, b in (
                ((x, y), (x + 1, y)),
                ((x + 1, y), (x + 1, y + 1)),
                ((x + 1, y + 1), (x, y + 1)),
                ((x, y + 1), (x, y)),
            ):
                seen.add((a, b) if a <= b else (b, a))
    return [{"from": list(a), "to": list(b)} for a, b in sorted(seen)]


def _plan_parts(x0, y0, grid):
    depth, width = len(grid), len(grid[0])
    segs = _cell_segments(x0, y0, [depth] * width)  # a full rectangle of squares
    points = [
        {"at": [x0 + x + 0.5, y0 + y + 0.5], "label": str(grid[y][x])} for y in range(depth) for x in range(width)
    ]
    return segs, points


def _extent(item) -> int:
    kind, data = item[0], item[1]
    return len(data) if kind == "plan" else max(data)


def _layout_figure(top_row, options, *, alt, title, panel_w):
    """Four lettered option panels in a row at the bottom and, above them, the stimulus: ``top_row`` is a list of
    ``('plan', grid, label)`` or ``('elev', heights, label)``; ``options`` is four ``('elev', heights)`` or
    ``('plan', grid)``. Each panel is ``panel_w`` squares wide; the panels are ``_GAP`` apart."""
    segments, points = [], []
    band = max(1, *(_extent(o) for o in options))
    top_y = band + 2.0
    x = 0.0
    for item in top_row:
        kind, data, label = item
        if kind == "plan":
            seg, pts = _plan_parts(x, top_y, data)
            segments += seg
            points += pts
        else:
            segments += _cell_segments(x, top_y, data)
        points.append({"at": [x + panel_w / 2, top_y - 0.6], "label": label})
        x += panel_w + _GAP
    x = 0.0
    for i, (kind, data) in enumerate(options):
        if kind == "elev":
            segments += _cell_segments(x, _Y0, data)
        else:
            seg, pts = _plan_parts(x, _Y0, data)
            segments += seg
            points += pts
        points.append({"at": [x + panel_w / 2, _Y0 - 0.6], "label": _LETTERS[i]})
        x += panel_w + _GAP
    top = top_y + max(_extent(item) for item in top_row) + 0.5
    width = 4 * panel_w + 3 * _GAP
    return cartesian(
        x_range=(0, int(width + 0.99)),
        y_range=(-1, int(top + 0.99)),
        alt=alt,
        grid=1,
        equal_scale=True,
        x_label="",
        y_label="",
        tick_labels=False,
        points=points,
        segments=segments,
        title=title,
    )


# ------------------------------------------------------------------------------------------ slips


def _front_panels(grid):
    """(heights, misconception id) for each wrong front elevation."""
    f, r = front(grid), right(grid)
    depth, width = len(grid), len(grid[0])
    return [
        (r, "fr2a-view-wrong-side"),
        (tuple(reversed(f)), "fr2a-view-from-the-back"),
        (tuple(sum(grid[y][x] for y in range(depth)) for x in range(width)), "fr2a-view-stacks-added"),
        (tuple(grid[0]), "fr2a-view-nearest-stacks-only"),
        (tuple(grid[-1]), "fr2a-view-back-stacks-only"),
        (tuple(min(grid[y][x] for y in range(depth)) for x in range(width)), "fr2a-view-shortest-stack-shown"),
    ]


def _right_panels(grid):
    f, r = front(grid), right(grid)
    depth, width = len(grid), len(grid[0])
    return [
        (f, "fr2a-view-wrong-side"),
        (tuple(reversed(r)), "fr2a-view-from-the-left"),
        (tuple(sum(row) for row in grid), "fr2a-view-stacks-added"),
        (tuple(grid[y][-1] for y in range(depth)), "fr2a-view-nearest-stacks-only"),
        (tuple(grid[y][0] for y in range(depth)), "fr2a-view-far-stacks-only"),
        (tuple(min(row) for row in grid), "fr2a-view-shortest-stack-shown"),
    ]


def _plan_panels(f, r, grid):
    """(grid, misconception id) for wrong plans: each fails the front or the right-side elevation."""
    depth, width = len(grid), len(grid[0])
    out = []
    # rows and columns swapped
    out.append(([list(col) for col in zip(*grid)], "fr2a-plan-rows-and-columns-swapped"))
    # left to right mirror
    out.append(([list(reversed(row)) for row in grid], "fr2a-plan-mirrored"))
    # front to back mirror
    out.append(([list(row) for row in reversed(grid)], "fr2a-plan-front-and-back-swapped"))
    # every square as tall as its column's tallest stack
    out.append(([[f[x] for x in range(width)] for _ in range(depth)], "fr2a-plan-every-stack-as-tall-as-column"))
    # the elevations' heights placed as a single row
    flat = [[grid[y][x] for x in range(width)] for y in range(depth)]
    flat[0] = list(f)
    out.append((flat, "fr2a-plan-elevation-copied-to-front-row"))
    return out


def _distinct_panels(truth, candidates, kind, n=3):
    """Up to ``n`` wrong panels that differ from the truth and from each other."""
    chosen, seen = [], {tuple(map(tuple, truth)) if kind == "plan" else tuple(truth)}
    for data, tag in candidates:
        key = tuple(map(tuple, data)) if kind == "plan" else tuple(data)
        if key in seen:
            continue
        if kind == "elev" and (min(key, default=0) < 0 or max(key, default=0) == 0):
            continue
        seen.add(key)
        chosen.append((data, tag))
        if len(chosen) == n:
            break
    return chosen


def _random_grid(rng, width, depth, top):
    return [[rng.randint(0, top) for _ in range(width)] for _ in range(depth)]


def _good_grid(grid, side):
    """Every column and row has a cube; the wanted elevation is not just the nearest stacks."""
    if any(v == 0 for v in front(grid)) or any(v == 0 for v in right(grid)):
        return False
    if side == "front" and tuple(grid[0]) == front(grid):
        return False
    if side == "right" and tuple(grid[y][-1] for y in range(len(grid))) == right(grid):
        return False
    return len(set(front(grid))) > 1 and len(set(right(grid))) > 1


def _stem(rng, who, thing, ask):
    return phrase(
        rng,
        (
            "{who} builds a {thing} from identical cubes. The plan view shows the number of cubes in each stack, and "
            "the front is the bottom edge of the plan. {ask}",
            "The plan view of {who}'s {thing} is shown, with the number of cubes stacked in each square. The front "
            "is the bottom edge. {ask}",
        ),
        who=who,
        thing=thing,
        ask=ask,
    )


def _option_d(i):
    return f"Option {_LETTERS[i]}"


def _finish_mcq(rng, text, fig, truth_index, tags, working):
    texts = [_option_d(i) for i in range(4)]
    key = texts[truth_index]
    dis = [Distractor(texts[i], tags[i], texts[i]) for i in range(4) if i != truth_index]
    check(not kit.leaks(fig, key), "the figure shows the answer")
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=key,
        answer_text=key,
        answer_latex=latex_answer(key, plain=True),
        answer_kind="mcq",
        working=working,
        distractors=dis,
        figure=fig,
    )


# ------------------------------------------------------------------------------------------ the tiers


def _elevation_question(rng, side):
    for _ in attempts(3000, "a plan, an elevation and three distinct wrong panels"):
        width, depth = (
            rng.choice([(3, 2), (3, 3), (4, 2), (4, 3)])
            if side == "front"
            else rng.choice([(3, 3), (3, 4), (2, 4), (4, 4)])
        )
        top = 3 if side == "front" and depth == 2 else 4
        grid = _random_grid(rng, width, depth, top)
        if not _good_grid(grid, side):
            continue
        truth = front(grid) if side == "front" else right(grid)
        wrong = _distinct_panels(truth, _front_panels(grid) if side == "front" else _right_panels(grid), "elev")
        if len(wrong) < 3:
            continue
        break
    check(truth == (front_by_layers(grid) if side == "front" else right_by_layers(grid)), "elevation differs by layers")
    panel_w = max(width, depth)
    options = [("elev", truth)] + [("elev", w) for w, _ in wrong]
    tags = [None] + [t for _, t in wrong]
    order = list(range(4))
    rng.shuffle(order)
    ordered = [options[i] for i in order]
    ordered_tags = [tags[i] for i in order]
    truth_index = order.index(0)
    fig = _layout_figure(
        [("plan", grid, "Plan view")],
        ordered,
        alt="A plan view of a structure of cubes, with a number in each square, above four lettered drawings of squares.",
        title="Plan view (the front is the bottom edge) and four views to choose from",
        panel_w=panel_w,
    )
    who, thing = rng.choice(_WHO), rng.choice(_THINGS)
    if side == "front":
        ask = "Which option shows the front elevation, the view from the front?"
        how = (
            "Looking from the front, the tallest stack in each column shows, because lower stacks behind it are hidden."
        )
    else:
        ask = "Which option shows the right-side elevation, the view from the right-hand side?"
        how = (
            "Looking from the right, each row shows its tallest stack; the front row is on the left of the view "
            "and the back row is on the right."
        )
    text = _stem(rng, who, thing, ask)
    heights = ", ".join(str(h) for h in truth)
    working = [
        how,
        f"The squares in each position, left to right, are {heights}.",
        f"That is {_option_d(truth_index)}.",
    ]
    return _finish_mcq(rng, text, fig, truth_index, ordered_tags, working)


def _easy(rng):
    return _elevation_question(rng, "front")


def _medium(rng):
    return _elevation_question(rng, "right")


def _hard(rng):
    for _ in attempts(3000, "a plan whose two elevations pick out only one of four plans"):
        width, depth = rng.choice([(3, 3), (4, 3), (3, 4)])
        grid = _random_grid(rng, width, depth, 4)
        if not _good_grid(grid, "front"):
            continue
        f, r = front(grid), right(grid)
        wrong = []
        for g, tag in _plan_panels(f, r, grid):
            if front(g) == f and right(g) == r:
                continue  # also fits both elevations, so it cannot be a wrong option
            wrong.append((g, tag))
        wrong = _distinct_panels(grid, wrong, "plan")
        if len(wrong) >= 3:
            break
    check(front(grid) == front_by_layers(grid) and right(grid) == right_by_layers(grid), "elevations differ by layers")
    check(all(front(g) != f or right(g) != r for g, _ in wrong), "a wrong plan fits both elevations")
    panel_w = max(width, depth)
    options = [("plan", grid)] + [("plan", g) for g, _ in wrong]
    tags = [None] + [t for _, t in wrong]
    order = list(range(4))
    rng.shuffle(order)
    ordered = [options[i] for i in order]
    ordered_tags = [tags[i] for i in order]
    truth_index = order.index(0)
    fig = _layout_figure(
        [("elev", f, "Front elevation"), ("elev", r, "Right-side elevation")],
        ordered,
        alt="A front elevation and a right-side elevation of a structure of cubes, above four lettered plan views.",
        title="Front and right-side elevations, and four plan views to choose from",
        panel_w=panel_w,
    )
    who, thing = rng.choice(_WHO), rng.choice(_THINGS)
    text = phrase(
        rng,
        (
            "{who} builds a {thing} from identical cubes. Its front elevation and right-side elevation are shown. "
            "Each plan view has the front at its bottom edge. Which plan view gives both elevations?",
            "The front elevation and the right-side elevation of {who}'s {thing} are drawn. In each plan view the "
            "front is the bottom edge. Which plan could be the plan view of the {thing}?",
        ),
        who=who,
        thing=thing,
    )
    working = [
        "The front elevation shows the tallest stack in each column of the plan: " + ", ".join(map(str, f)) + ".",
        "The right-side elevation shows the tallest stack in each row, front row first: "
        + ", ".join(map(str, r))
        + ".",
        f"Only {_option_d(truth_index)} has these tallest stacks in its columns and in its rows.",
    ]
    return _finish_mcq(rng, text, fig, truth_index, ordered_tags, working)


@generator(
    "fr2a_matching_view",
    level=7,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fr2a_matching_view(rng, tier="easy"):
    """Pick the elevation that matches a plan view, or the plan that matches two elevations."""
    check(tier in ("easy", "medium", "hard"), f"unknown tier {tier!r}")
    return redraw({"easy": _easy, "medium": _medium, "hard": _hard}[tier], rng, "a view question that fits")


# ------------------------------------------------------------------------------------------ the figure reader


def _cells(fig):
    """The unit squares (lower-left corner) whose four edges are all drawn."""
    edges = set()
    for seg in fig["segments"]:
        a, b = tuple(seg["from"]), tuple(seg["to"])
        edges.add((a, b) if a <= b else (b, a))

    def has(a, b):
        return ((a, b) if a <= b else (b, a)) in edges

    cells = set()
    for a, b in edges:
        x, y = int(a[0]), int(a[1])
        if (
            has((x, y), (x + 1, y))
            and has((x + 1, y), (x + 1, y + 1))
            and has((x, y + 1), (x + 1, y + 1))
            and has((x, y), (x, y + 1))
        ):
            cells.add((x, y))
    return cells


def _column_height(cells, x, y0):
    h = 0
    while (x, y0 + h) in cells:
        h += 1
    return h


def _grid_at(numbers, x0, y0, width, depth):
    return [[numbers[(x0 + x, y0 + y)] for x in range(width)] for y in range(depth)]


def _read_matching_view(question):
    """Re-derive the lettered option from the picture alone: rebuild the plan (from its numbers) or the two elevations
    (from their squares), then find the lettered panel that fits."""
    fig, text = question.figure, question.text
    at = {p["label"]: p["at"] for p in fig["points"] if p.get("label")}
    letters = {c: at[c] for c in _LETTERS}
    panel_w = round(letters["B"][0] - letters["A"][0] - _GAP)
    cells = _cells(fig)
    numbers = {
        (int(p["at"][0] - 0.5), int(p["at"][1] - 0.5)): int(p["label"])
        for p in fig["points"]
        if re.fullmatch(r"\d+", p.get("label", ""))
    }

    def panel_x(letter):
        return int(round(letters[letter][0] - panel_w / 2))

    if "Plan view" in at:
        top_y = int(round(at["Plan view"][1] + 0.6))
        x0 = int(round(at["Plan view"][0] - panel_w / 2))
        width = 1 + max(k[0] for k in numbers) - x0
        depth = 1 + max(k[1] for k in numbers) - top_y
        grid = _grid_at(numbers, x0, top_y, width, depth)
        want = front(grid) if "front elevation" in text else right(grid)
        want = want + (0,) * (panel_w - len(want))
        for i, letter in enumerate(_LETTERS):
            heights = tuple(_column_height(cells, panel_x(letter) + j, 0) for j in range(panel_w))
            if heights == want:
                return _option_d(i)
        raise AssertionError("no option matches the drawn plan")
    top_y = int(round(at["Front elevation"][1] + 0.6))
    fx = int(round(at["Front elevation"][0] - panel_w / 2))
    rx = int(round(at["Right-side elevation"][0] - panel_w / 2))
    f = tuple(_column_height(cells, fx + j, top_y) for j in range(panel_w))
    r = tuple(_column_height(cells, rx + j, top_y) for j in range(panel_w))
    f, r = tuple(h for h in f if h), tuple(h for h in r if h)
    low = {k: v for k, v in numbers.items() if k[1] < top_y}
    for i, letter in enumerate(_LETTERS):
        x0 = panel_x(letter)
        mine = {k: v for k, v in low.items() if x0 <= k[0] < x0 + panel_w}
        width = 1 + max(k[0] for k in mine) - x0
        depth = 1 + max(k[1] for k in mine)
        grid = _grid_at(low, x0, 0, width, depth)
        if front(grid) == f and right(grid) == r:
            return _option_d(i)
    raise AssertionError("no option matches the drawn elevations")


register_reader("fr2a_matching_view", _read_matching_view)


# ------------------------------------------------------------------------------------------ the exam chain


@generator("exd_x_fr2a_matching_view_chain", level=9, presets={"exam": dict(tier="exam")})
def exd_x_fr2a_matching_view_chain(rng, tier="exam"):
    """Exam tier: the front elevation, squares in the right-side elevation, cubes in all, cubes not shown."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return redraw(_fr2a_matching_view_chain, rng, "a chain with enough distinct wrong answers")


def _fr2a_matching_view_chain(rng):
    for _ in attempts(3000, "a plan with three distinct wrong front elevations and different counts"):
        width, depth = rng.choice([(3, 3), (4, 3), (4, 4)])
        grid = _random_grid(rng, width, depth, 4)
        if not _good_grid(grid, "front"):
            continue
        wrong = _distinct_panels(front(grid), _front_panels(grid), "elev")
        if len(wrong) < 3:
            continue
        seen_squares, tot, shown = sum(front(grid)), total(grid), sum(right(grid))
        if len({shown, tot, seen_squares}) < 3 or tot - shown < 2:
            continue
        break
    check(shown == sum(right_by_layers(grid)) and seen_squares == sum(front_by_layers(grid)), "counts differ by layers")
    panel_w = max(width, depth)
    options = [("elev", front(grid))] + [("elev", w) for w, _ in wrong]
    tags = [None] + [t for _, t in wrong]
    order = list(range(4))
    rng.shuffle(order)
    ordered = [options[i] for i in order]
    ordered_tags = [tags[i] for i in order]
    truth_index = order.index(0)
    fig = _layout_figure(
        [("plan", grid, "Plan view")],
        ordered,
        alt="A plan view of a structure of cubes, with a number in each square, above four lettered drawings of squares.",
        title="Plan view (the front is the bottom edge) and four views to choose from",
        panel_w=panel_w,
    )
    who, thing = rng.choice(_WHO), rng.choice(_THINGS)
    stem = (
        f"{who} builds a {thing} from identical cubes. The plan view shows the number of cubes in each stack, and the "
        "front is the bottom edge of the plan."
    )
    key = _option_d(truth_index)
    part_a = dict(
        prompt="Which option shows the front elevation?",
        answer_kind="mcq",
        answer=key,
        answer_text=key,
        answer_latex=latex_answer(key, plain=True),
        marks=1,
        working=[
            "The front elevation shows the tallest stack in each column: " + ", ".join(map(str, front(grid))) + "."
        ],
        distractors=[Distractor(_option_d(i), ordered_tags[i], _option_d(i)) for i in range(4) if i != truth_index],
    )
    from fractions import Fraction

    cols_tallest = max(max(row) for row in grid)
    occupied = sum(1 for row in grid for v in row if v)
    b_cands = [
        (Fraction(tot), "fr2a-view-total-for-elevation"),
        (Fraction(seen_squares), "fr2a-view-wrong-side"),
        (Fraction(cols_tallest * depth), "fr2a-view-tallest-times-rows"),
        (Fraction(sum(grid[y][-1] for y in range(depth))), "fr2a-view-nearest-stacks-only"),
    ]
    part_b = number_part(
        "How many squares make up the right-side elevation?",
        Fraction(shown),
        0,
        2,
        [
            "The right-side elevation shows the tallest stack in each row: "
            + " + ".join(map(str, right(grid)))
            + f" = {shown} squares."
        ],
        tagged(Fraction(shown), b_cands, 0, lo=0),
        sensible_range=(0, None),
    )
    c_cands = [
        (Fraction(occupied), "fr2a-view-squares-counted"),
        (Fraction(cols_tallest * width * depth), "fr2a-view-every-stack-tallest"),
        (Fraction(seen_squares), "fr2a-view-elevation-for-total"),
        (Fraction(shown), "fr2a-view-elevation-for-total"),
    ]
    part_c = number_part(
        "How many cubes are used in the whole structure?",
        Fraction(tot),
        0,
        2,
        ["Add the cubes in every stack of the plan: " + f"{tot} cubes."],
        tagged(Fraction(tot), c_cands, 0, lo=0),
        sensible_range=(0, None),
    )
    d_cands = [
        (Fraction(shown), "fr2a-view-hidden-count-is-visible-count"),
        (Fraction(tot + shown), "fr2a-view-hidden-added"),
        (Fraction(seen_squares), "fr2a-view-wrong-side"),
    ]
    part_d = number_part(
        "How many cubes are not shown as a square in the right-side elevation?",
        Fraction(tot - shown),
        0,
        2,
        [
            f"Cubes in all, {tot}, less the {shown} squares that show: {tot} − {shown} = {tot - shown} cubes are hidden behind others."
        ],
        tagged(Fraction(tot - shown), kit.far_from(tot - shown, d_cands, 1), 0, lo=0),
        follow_through="part_c-part_b",
        sensible_range=(0, None),
    )
    parts = [part_a, part_b, part_c, part_d]
    kit.require_distractors(parts)
    return dict(text=stem, figure=fig, parts=parts)
