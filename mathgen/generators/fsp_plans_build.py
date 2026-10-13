"""Generates `fsp_plan_views` and `fsp_plan_views_exam` questions: plans and elevations of a stack of cubes.

One skill: ``f10-plan-views-cube-stacks`` (``answer: number``). The stimulus is the plan view of a structure built from
identical cubes, drawn with the existing ``cartesian`` figure kind as a grid: the number in each square is how many
cubes are stacked there, and the front of the structure is the bottom edge of the plan. Three tiers:

- easy: how many cubes are used in all? (add every stack.)
- medium: how many squares make up the front elevation (or the right-side elevation)? An elevation shows only the
  tallest stack in each column (or row) of the plan, because the lower stacks are hidden behind it.
- hard: one stack is marked ? and the number of squares in an elevation is given. Which number of cubes makes the
  elevation that size? The marked stack is the tallest in its column (or row), so it is the only unknown in the sum of
  the tallest stacks.

``fsp_plan_views_exam`` is the exam chain on one plan: (a) the number of cubes, (b) the squares in the front
elevation, (c) the squares in the right-side elevation, (d) the volume when each cube has a stated edge (follows
through from (a)).

Elevations are computed from the grid (the maximum of each column or row) and again by building the structure up layer
by layer and counting the squares each layer shows. The figure reader rebuilds the grid from the drawn labels alone.
"""

from __future__ import annotations

import re

from mathgen import figures
from mathgen.answers import Distractor
from mathgen.core import attempts, check, generator, latex_answer, phrase, pick_distractors
from mathgen.figure_kinds import space  # noqa: F401  (registers the figure kinds)
from mathgen.figures import cartesian
from . import _fsp_kit as kit

_NAMES = ("Mia", "Tom", "Aiko", "Dev", "Leila", "Sam", "Jess", "Kofi", "Priya", "Marcus", "Zoe", "Ben")
_THINGS = ("tower", "wall", "stepped model", "block model", "structure", "sculpture", "display", "building")
_ALT = "A plan view drawn as a grid of squares, with a number written in each square."
_TITLE = "Plan view (the front is the bottom edge)"


# ------------------------------------------------------------------------------------------ the grid


def plan_figure(grid):
    """The plan view of ``grid`` (rows from the front; ``None`` marks the unknown stack) as a ``cartesian`` figure."""
    depth, width = len(grid), len(grid[0])
    points = [
        {"at": [x + 0.5, y + 0.5], "label": "?" if grid[y][x] is None else str(grid[y][x])}
        for y in range(depth)
        for x in range(width)
    ]
    return cartesian(
        x_range=(0, width),
        y_range=(0, depth),
        alt=_ALT,
        grid=1,
        equal_scale=True,
        x_label="",
        y_label="",
        tick_labels=False,
        points=points,
        title=_TITLE,
    )


def front_squares(grid) -> int:
    """Squares in the front elevation: the tallest stack in each column."""
    return sum(max(grid[y][x] for y in range(len(grid))) for x in range(len(grid[0])))


def right_squares(grid) -> int:
    """Squares in the right-side elevation: the tallest stack in each row."""
    return sum(max(row) for row in grid)


def _layer_view_squares(grid, view: str) -> int:
    """The elevation counted layer by layer: at each height, the columns (or rows) that have a cube at that height."""
    depth, width = len(grid), len(grid[0])
    top = max(max(row) for row in grid)
    total = 0
    for level in range(1, top + 1):
        if view == "front":
            total += sum(1 for x in range(width) if any(grid[y][x] >= level for y in range(depth)))
        else:
            total += sum(1 for y in range(depth) if any(grid[y][x] >= level for x in range(width)))
    return total


def _total(grid) -> int:
    return sum(sum(row) for row in grid)


def _random_grid(rng, width, depth, top=4, lowest=0):
    return [[rng.randint(lowest, top) for _ in range(width)] for _ in range(depth)]


def _flat(grid):
    return [v for row in grid for v in row]


# ------------------------------------------------------------------------------------------ tiers


def _easy(rng):
    for _ in attempts(500, "a plan whose total is not drawn on it"):
        width, depth = rng.choice([(3, 2), (3, 3), (4, 2), (4, 3), (5, 2)])
        grid = _random_grid(rng, width, depth, 4, 0)
        total = _total(grid)
        if total >= 8 and sum(1 for v in _flat(grid) if v) >= 4:
            fig = plan_figure(grid)
            if not kit.leaks(fig, str(total)):
                break
    stacks = sum(1 for v in _flat(grid) if v)
    biggest = max(_flat(grid))
    candidates = [
        Distractor(str(stacks), "plan-squares-counted", stacks),
        Distractor(str(biggest * len(_flat(grid))), "plan-tallest-times-squares", biggest * len(_flat(grid))),
        Distractor(str(front_squares(grid)), "plan-elevation-for-total", front_squares(grid)),
        Distractor(str(total - sum(grid[-1])), "plan-stack-left-out", total - sum(grid[-1])),
        Distractor(str(total - biggest), "plan-stack-left-out", total - biggest),
    ]
    text = phrase(
        rng,
        (
            "{who} builds a {thing} from identical cubes. The plan view shows how many cubes are stacked in each "
            "square. How many cubes does {who} use altogether?",
            "The plan view of {who}'s {thing} is shown. The number in each square is the number of cubes stacked "
            "there. How many cubes make the {thing}?",
            "A {thing} is made of identical cubes stacked in columns. The plan view shows the number of cubes in "
            "each stack. How many cubes are used in all?",
            "{who} stacks identical cubes to make a {thing}. Each number on the plan view is the height of a stack, "
            "in cubes. What is the total number of cubes?",
        ),
        who=rng.choice(_NAMES),
        thing=rng.choice(_THINGS),
    )
    working = ["Add the number of cubes in every square of the plan.", f"Total = {total} cubes."]
    return fig, text, total, candidates, working


def _medium(rng):
    for _ in attempts(1000, "a plan whose two elevations and total all differ"):
        width, depth = rng.choice([(3, 3), (4, 3), (4, 4), (3, 4), (5, 3)])
        grid = _random_grid(rng, width, depth, 4, 0)
        view = rng.choice(["front", "right"])
        front, right, total = front_squares(grid), right_squares(grid), _total(grid)
        answer = front if view == "front" else right
        if len({front, right, total}) < 3 or answer < 5:
            continue
        # the tallest stack must not be the only one that matters: some column or row hides a lower stack
        if answer == total:
            continue
        fig = plan_figure(grid)
        if not kit.leaks(fig, str(answer)):
            break
    other = right if view == "front" else front
    near = sum(grid[0]) if view == "front" else sum(row[-1] for row in grid)
    columns = width if view == "front" else depth
    biggest = max(_flat(grid))
    candidates = [
        Distractor(str(total), "plan-total-for-elevation", total),
        Distractor(str(other), "plan-wrong-side-viewed", other),
        Distractor(str(biggest * columns), "plan-tallest-times-columns", biggest * columns),
        Distractor(str(near), "plan-nearest-stacks-only", near),
    ]
    if view == "front":
        phrase_view = "front elevation (the view from the front, along the bottom edge of the plan)"
    else:
        phrase_view = "right-side elevation (the view from the right-hand side of the plan)"
    text = phrase(
        rng,
        (
            "{who}'s {thing} is made of identical cubes. The plan view shows the number of cubes stacked in each "
            "square. How many squares make up the {view}?",
            "The plan view of a {thing} built by {who} is shown, with the number of cubes in each stack. How many "
            "squares does the {view} show?",
            "A {thing} is built from identical cubes, as the plan view shows. Looking at the {view}, how many "
            "squares can be seen?",
            "{who} draws the {view} of the {thing} whose plan view is shown. How many squares does it have?",
        ),
        who=rng.choice(_NAMES),
        thing=rng.choice(_THINGS),
        view=phrase_view,
    )
    check(answer == _layer_view_squares(grid, view), "the elevation count differs when counted layer by layer")
    working = [
        "An elevation shows only the tallest stack in each " + ("column" if view == "front" else "row") + ".",
        "Add those tallest stacks: " + str(answer) + " squares.",
    ]
    return fig, text, answer, candidates, working


def _hard(rng):
    for _ in attempts(2000, "a plan with one unknown stack that is the tallest in its line"):
        width, depth = rng.choice([(3, 3), (4, 3), (4, 4), (3, 4)])
        view = rng.choice(["front", "right"])
        unknown = rng.randint(4, 6)
        grid = _random_grid(rng, width, depth, 3, 0)
        ux, uy = rng.randrange(width), rng.randrange(depth)
        grid[uy][ux] = unknown
        # every other label avoids the unknown's value, so the figure does not show the answer
        if unknown in [v for i, v in enumerate(_flat(grid)) if i != uy * width + ux]:
            continue
        line = [grid[y][ux] for y in range(depth)] if view == "front" else list(grid[uy])
        if sorted(line)[-1] != unknown or line.count(unknown) != 1:
            continue
        shown = front_squares(grid) if view == "front" else right_squares(grid)
        shown_grid = [row[:] for row in grid]
        shown_grid[uy][ux] = None
        fig = plan_figure(shown_grid)
        if kit.leaks(fig, str(unknown)):
            continue
        break
    total_other = _total(grid) - unknown
    others_columns = (
        sum(max(grid[y][x] for y in range(depth)) for x in range(width) if x != ux)
        if view == "front"
        else sum(max(grid[y]) for y in range(depth) if y != uy)
    )
    check(shown - others_columns == unknown, "the elevation does not fix the unknown stack")
    column_others = sum(line) - unknown
    candidates = [
        Distractor(str(shown - total_other), "plan-elevation-for-total", shown - total_other),
        Distractor(str(unknown - column_others), "plan-line-added-for-tallest", unknown - column_others),
        Distractor(
            str(shown - (total_other - column_others)),
            "plan-line-added-for-tallest",
            shown - (total_other - column_others),
        ),
        Distractor(
            str(shown - others_columns + column_others),
            "plan-line-added-for-tallest",
            shown - others_columns + column_others,
        ),
        Distractor(str(unknown - 1), "plan-count-off-by-one", unknown - 1),
        Distractor(str(unknown + 1), "plan-count-off-by-one", unknown + 1),
    ]
    name = "front elevation" if view == "front" else "right-side elevation"
    where = (
        "from the front, along the bottom edge of the plan"
        if view == "front"
        else "from the right-hand side of the plan"
    )
    text = phrase(
        rng,
        (
            "{who}'s {thing} is made of identical cubes. The plan view shows the number of cubes in each stack, "
            "except one marked ?. The {name} (the view {where}) has {shown} squares. How many cubes are in the "
            "stack marked ?",
            "In the plan view of a {thing} built by {who}, one stack is marked ?. The {name}, seen {where}, is made "
            "of {shown} squares. How many cubes are in the stack marked ?",
            "A {thing} is built from identical cubes. One stack on the plan view is marked ?. Seen {where}, the "
            "{name} has {shown} squares. Find the number of cubes in the stack marked ?",
            "{who} knows that the {name} of the {thing}, seen {where}, has {shown} squares. The plan view shows "
            "every stack except one, marked ?. How many cubes are in that stack?",
        ),
        who=rng.choice(_NAMES),
        thing=rng.choice(_THINGS),
        name=name,
        where=where,
        shown=shown,
    )
    working = [
        f"The {name} shows the tallest stack in each "
        + ("column" if view == "front" else "row")
        + f". The other tallest stacks add to {others_columns}.",
        f"The stack marked ? is the tallest in its {'column' if view == 'front' else 'row'}: {shown} - {others_columns} = {unknown}.",
    ]
    return fig, text, unknown, candidates, working


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(
    "fsp_plan_views",
    level=8,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def fsp_plan_views(rng, tier="easy"):
    """A plan view of cube stacks with a number answer. See the module docstring for the tiers."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(40, "a plan question with three distinct wrong answers"):
        fig, text, answer, candidates, working = _TIERS[tier](rng)
        key = str(answer)
        distractors = pick_distractors(key, [c for c in candidates if int(c.text) > 0], 3)
        if len(distractors) == 3:
            break
    return dict(
        text=text,
        answer=answer,
        answer_text=key,
        answer_latex=latex_answer(key),
        answer_kind="number",
        distractors=distractors,
        working=working,
        figure=fig,
    )


# ------------------------------------------------------------------------------------------ reader


def read_grid(fig):
    """The grid of stacks (rows from the front, ``None`` for the unknown) read from the drawn labels alone."""
    width = int(fig["x_range"][1])
    depth = int(fig["y_range"][1])
    grid = [[None] * width for _ in range(depth)]
    for p in fig["points"]:
        x, y = int(p["at"][0]), int(p["at"][1])
        grid[y][x] = None if p["label"] == "?" else int(p["label"])
    return grid


def _read_plan(question) -> str:
    fig = question.figure
    grid = read_grid(fig)
    text = question.text
    if any(v is None for row in grid for v in row):
        shown = int(re.search(r"(\d+) squares", text).group(1))
        right = "right-side elevation" in text
        uy, ux = next((y, x) for y, row in enumerate(grid) for x, v in enumerate(row) if v is None)
        depth, width = len(grid), len(grid[0])
        if right:
            others = sum(max(grid[y]) for y in range(depth) if y != uy)
        else:
            others = sum(max(grid[y][x] for y in range(depth)) for x in range(width) if x != ux)
        return str(shown - others)
    if "How many cubes" in text or "total number of cubes" in text:
        return str(_total(grid))
    if "front elevation" in text:
        return str(front_squares(grid))
    return str(right_squares(grid))


figures.register_reader("fsp_plan_views", _read_plan)


# ------------------------------------------------------------------------------------------ exam chain


@generator("fsp_plan_views_exam", level=9, presets={"exam": {}})
def fsp_plan_views_exam(rng):
    """One plan: count the cubes, the squares of two elevations, then the volume (follows through from the count)."""
    for _ in attempts(1000, "a plan whose total and two elevations all differ"):
        width, depth = rng.choice([(4, 3), (3, 3), (4, 4), (5, 3)])
        grid = _random_grid(rng, width, depth, 4, 0)
        total, front, right = _total(grid), front_squares(grid), right_squares(grid)
        if len({total, front, right}) == 3 and min(front, right) >= 5 and total >= 12:
            break
    edge = rng.choice([2, 3, 4, 5])
    volume = edge**3 * total
    check(
        front == _layer_view_squares(grid, "front") and right == _layer_view_squares(grid, "right"),
        "elevations disagree",
    )
    fig = plan_figure(grid)
    stem = phrase(
        rng,
        (
            "{who} builds a {thing} from identical cubes with edges of {edge} cm. The plan view shows the number of "
            "cubes stacked in each square, and the front is the bottom edge of the plan.",
            "A {thing} is made from identical cubes, each with an edge of {edge} cm. The plan view, with the front "
            "at the bottom, shows how many cubes {who} has stacked in each square.",
            "{who}'s {thing} is built from cubes with {edge} cm edges. The plan view shows the stack in each square; "
            "the front of the {thing} is the bottom edge of the plan.",
            "The plan view of a {thing} made by {who} from {edge} cm cubes is shown. Each number is a stack of "
            "cubes, and the front is the bottom edge of the plan.",
        ),
        who=rng.choice(_NAMES),
        thing=rng.choice(_THINGS),
        edge=edge,
    )
    stacks = sum(1 for v in _flat(grid) if v)
    biggest = max(_flat(grid))
    a_cands = [
        Distractor(str(stacks), "plan-squares-counted", stacks),
        Distractor(str(biggest * width * depth), "plan-tallest-times-squares", biggest * width * depth),
        Distractor(str(front), "plan-elevation-for-total", front),
    ]
    b_cands = [
        Distractor(str(total), "plan-total-for-elevation", total),
        Distractor(str(biggest * width), "plan-tallest-times-columns", biggest * width),
        Distractor(str(sum(grid[0])), "plan-nearest-stacks-only", sum(grid[0])),
        Distractor(str(right), "plan-wrong-side-viewed", right),
    ]
    c_cands = [
        Distractor(str(total), "plan-total-for-elevation", total),
        Distractor(str(biggest * depth), "plan-tallest-times-columns", biggest * depth),
        Distractor(str(sum(row[-1] for row in grid)), "plan-nearest-stacks-only", sum(row[-1] for row in grid)),
        Distractor(str(front), "plan-wrong-side-viewed", front),
    ]
    d_cands = [
        Distractor(str(edge**2 * total), "plan-volume-edge-squared", edge**2 * total),
        Distractor(str(edge * total), "plan-volume-edge-not-cubed", edge * total),
        Distractor(str(6 * edge**2 * total), "plan-volume-for-surface", 6 * edge**2 * total),
        Distractor(str(3 * edge * total), "plan-volume-edge-not-cubed", 3 * edge * total),
    ]
    parts = [
        dict(
            prompt="How many cubes are in the structure?",
            answer_kind="number",
            answer=total,
            answer_text=str(total),
            answer_latex=latex_answer(str(total)),
            marks=1,
            sensible_range=(0, None),
            working=[f"Add every stack on the plan: {total} cubes."],
            distractors=pick_distractors(str(total), [c for c in a_cands if int(c.text) > 0], 3),
        ),
        dict(
            prompt="How many squares make up the front elevation?",
            answer_kind="number",
            answer=front,
            answer_text=str(front),
            answer_latex=latex_answer(str(front)),
            marks=2,
            sensible_range=(0, None),
            working=["The front elevation shows the tallest stack in each column.", f"Squares: {front}."],
            distractors=pick_distractors(str(front), [c for c in b_cands if int(c.text) > 0], 3),
        ),
        dict(
            prompt="How many squares make up the right-side elevation?",
            answer_kind="number",
            answer=right,
            answer_text=str(right),
            answer_latex=latex_answer(str(right)),
            marks=2,
            sensible_range=(0, None),
            working=["The right-side elevation shows the tallest stack in each row.", f"Squares: {right}."],
            distractors=pick_distractors(str(right), [c for c in c_cands if int(c.text) > 0], 3),
        ),
        dict(
            prompt="Using your answer to part (a), find the volume of the structure, in cubic centimetres.",
            answer_kind="number",
            answer=volume,
            answer_text=str(volume),
            answer_latex=latex_answer(str(volume)),
            marks=2,
            sensible_range=(0, None),
            follow_through=f"{edge**3}*part_a",
            working=[f"One cube has volume {edge}³ = {edge**3} cm³.", f"{total} × {edge**3} = {volume} cm³."],
            distractors=pick_distractors(str(volume), [c for c in d_cands if int(c.text) > 0], 3),
        ),
    ]
    check(all(len(p["distractors"]) >= 2 for p in parts), "too few distractors in a part")
    return dict(text=stem, figure=fig, parts=parts)
