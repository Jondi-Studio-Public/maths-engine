"""Early-years shapes from a picture: the ``shape_scene`` figure kind and three skills built on it.

- ``ey_shape_recognise`` (skill ``name-basic-shapes``, AC9MFSP01): easy names the one shape shown. Medium counts one kind of shape
  in a mixed scene where some are turned, sized differently or shaded. Hard finds the shape in an
  everyday object (the roof of a house, a can), or the object a solid is shaped like.
- ``ey_shape_classify`` (skill ``f10-sort-classify-familiar-shapes``, AC9M1SP01): easy counts the solid
  or flat shapes in a mix; medium sorts flat shapes by corners or a curved side; hard picks the odd
  one out together with the reason, as one multiple-choice option.
- ``ey_shape_sides`` (skill ``f10-shape-sides-spatial-terms``, AC9M2SP01): easy counts the sides or
  corners of one drawn shape; medium picks the labelled shape that is curved only, straight only or
  has a pair of parallel sides; hard counts the shapes in a scene with exactly N sides or a parallel pair.

Every question is multiple choice (``answer: mcq``): a count is offered as four numbered options, so a skill
grades one way at every tier.

The picture is the only place the shapes appear: no prompt names a shape it asks the learner to
find, and no label shows a count. Every prompt is a short sentence for reading aloud. Each topic
registers a figure reader that works from ``question.figure`` and the prompt, never from the key.
Facts about a shape (sides, corners, curved, parallel) come from
:data:`mathgen.figure_kinds.scene.FACTS`, so a question and its reader share one source.
"""

from __future__ import annotations

import re

from .. import figures
from ..core import attempts, check, generator, phrase, pick_distractors
from .early_math import _balanced
from ..figure_kinds.scene import (
    DISPLAY,
    FACTS,
    FLAT,
    is_also,
    OBJECTS,
    PLURAL,
    SIZES,
    SOLIDS,
    STYLES,
    object_part_shape,
    scene_item,
    shape_scene,
)

_TIERS = ("easy", "medium", "hard")
_PRESETS = {t: {"tier": t} for t in _TIERS}
_POLYGONS = tuple(s for s in FLAT if FACTS[s]["straight_only"])
_TURNS = (15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180)
_LETTERS = "ABCD"


# ---------------------------------------------------------------- shared helpers


def _item(rng, shape: str, *, turned: bool = False, size=None, style=None, upright: bool = False, label: str = ""):
    """A drawn shape. ``turned`` forces a clear turn; otherwise a flat shape is turned about one time in three."""
    turn = 0
    if shape in FLAT and not upright and shape != "circle":
        turn = rng.choice(_TURNS) if turned or rng.random() < 0.34 else 0
    return scene_item(
        shape,
        size=size or rng.choice(SIZES),
        style=style or rng.choice(STYLES),
        turn=turn,
        label=label,
    )


def _scene(rng, items, alt="a picture of some shapes"):
    n = len(items)
    if n <= 4 or rng.random() < 0.5:
        return shape_scene(items=items, alt=alt, layout="row")
    return shape_scene(items=items, alt=alt, layout="grid", cols=3 if n <= 6 else 4)


def _word(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


def _numeric_distractors(rng, answer: int, cands, fig, floor: int = 1) -> list[str]:
    """Three wrong counts. The first named slip is always offered; the key's rank among the four options is
    drawn at random so it is not predictably the second-smallest."""
    named = [c for c in cands if isinstance(c, int) and c >= floor and c != answer]
    pool = list(dict.fromkeys(named + [v for v in range(answer - 2, answer + 4) if v >= floor and v != answer]))
    chosen = _balanced(rng, answer, pool, tuple(named), must=named[:1])
    check(len(chosen) == 3, "needs three distractors")
    figures.assert_no_leak(fig, [str(answer), *chosen])
    return chosen


def _result(text, answer_text, distractors, working, fig):
    """A multiple-choice question: a count is offered as numbered options, so every tier grades the same way."""
    return dict(
        text=text,
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=f"plain:{answer_text}",
        distractors=distractors,
        working=working,
        figure=fig,
    )


def _check_tier(tier: str) -> None:
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")


def _noun_in(text: str) -> str | None:
    """The shape named (as a plural) in a prompt, longest name first so 'rectangles' is not 'rectangle'."""
    for shape in sorted(PLURAL, key=lambda s: -len(PLURAL[s])):
        if PLURAL[shape] in text:
            return shape
    return None


# ====================================================================== AC9MFSP01: recognise

_NAME_TEXT = ("What shape is this?", "Which shape can you see?", "What is the name of this shape?")
_NAME_SHAPES = ("circle", "triangle", "square", "rectangle")
_COUNT_TEXT = (
    "Count the {p}.",
    "How many {p} can you see?",
    "How many {p} are in the picture?",
)
_COUNT_SHAPES = ("triangle", "circle", "square", "rectangle")
# What a learner most often mixes each shape up with. A flat name is never offered beside the solid whose
# outline it is (no Circle with Sphere), and a cube is never offered beside the prism: a cube IS one.
_CONFUSED = {
    "triangle": ("pentagon", "square", "rhombus"),
    "square": ("triangle", "pentagon", "hexagon"),
    "circle": ("semicircle", "hexagon", "triangle"),
    "sphere": ("cylinder", "cone", "square pyramid"),
    "rectangular prism": ("cylinder", "cone", "square pyramid"),
    "cylinder": ("cone", "sphere", "square pyramid"),
}
_OBJECT_TEXT = {
    1: (
        "Which shape is this {o}?",
        "Which shape does this {o} look like?",
        "This is a {o}. Which shape is it?",
        "What shape is this {o}?",
        "Which shape is the {o} in the picture?",
    ),
    2: (
        "Which shape is the {r} of this {o}?",
        "What shape is the {r} of this {o}?",
        "Look at the {o}. Which shape is the {r}?",
        "Which shape is the {r} on this {o}?",
        "Which shape makes the {r} of this {o}?",
    ),
}
# A solid and the object it is shaped like; the reverse question names the object, not the solid.
_LIKE = {"sphere": "ball", "rectangular prism": "box", "cylinder": "can", "cone": "party hat"}
_LIKE_TEXT = (
    "Which object is shaped like this?",
    "Which object looks like this shape?",
    "This shape is like which object?",
)


def _recognise_name(rng):
    shape = rng.choice(_NAME_SHAPES)
    fig = shape_scene(
        items=[_item(rng, shape, upright=True, size=rng.choice(SIZES))],
        alt="a picture of a shape",
    )
    answer = DISPLAY[shape]
    # never offer a second right name: a square is also a rectangle
    options = [DISPLAY[s] for s in _NAME_SHAPES + ("hexagon",) if not is_also(shape, s)]
    distractors = pick_distractors(answer, options)
    figures.assert_no_leak(fig, [answer, *distractors])
    text = rng.choice(_NAME_TEXT)
    return _result(text, answer, distractors, [f"This shape is a {shape}."], fig)


def _recognise_count(rng):
    target = rng.choice(_COUNT_SHAPES)
    k = rng.randint(1, 4)
    decoys_pool = [s for s in FLAT if not is_also(s, target)]  # no square among the rectangles, no circle among ovals
    # a near neighbour of the target is always in the scene, so only the right shape counts
    near = {"triangle": "pentagon", "square": "rectangle", "circle": "oval", "rectangle": "kite"}[target]
    for _ in attempts(500, "a scene with a clear count"):
        total = rng.randint(max(5, k + 2), 8)
        decoys = [near] + [rng.choice(decoys_pool) for _ in range(total - k - 1)]
        items = [_item(rng, target) for _ in range(k)] + [_item(rng, s) for s in decoys]
        if target != "circle" and not any(i["turn"] for i in items[:k]):
            items[0] = _item(rng, target, turned=True)
        rng.shuffle(items)
        fig = _scene(rng, items)
        text = phrase(rng, _COUNT_TEXT, p=PLURAL[target])
        try:
            figures.assert_no_leak(fig, [str(k)])
        except AssertionError:
            continue
        break
    upright = sum(1 for i in items if i["shape"] == target and not i["turn"])
    other = {s: sum(1 for i in items if i["shape"] == s) for s in set(decoys)}
    cands = [k + 1, k - 1, len(items), upright if upright != k else k + 2, max(other.values()), k + 2]
    distractors = _numeric_distractors(rng, k, cands, fig, floor=0)
    working = [f"There {_word(k, 'is 1', f'are {k}')} {target if k == 1 else PLURAL[target]}, turned or not."]
    return _result(text, str(k), distractors, working, fig)


def _recognise_object(rng):
    if rng.random() < 0.25:
        solid = rng.choice(tuple(_LIKE))
        fig = shape_scene(items=[_item(rng, solid)], alt="a picture of a solid shape")
        answer = f"A {_LIKE[solid]}"
        options = [f"A {o}" for o in _LIKE.values()]
        distractors = pick_distractors(answer, options)
        figures.assert_no_leak(fig, [answer, *distractors])
        return _result(rng.choice(_LIKE_TEXT), answer, distractors, [f"A {_LIKE[solid]} is a {solid}."], fig)
    # the ice cream is not asked: a flat triangle and a scoop read as a triangle and a circle, not a cone and a semicircle
    name = rng.choice([n for n in OBJECTS if n != "ice cream"])
    parts = OBJECTS[name]["parts"]
    role, shape = rng.choice(parts)
    fig = shape_scene(items=[], object=name, alt=f"a picture of {OBJECTS[name]['alt']}")
    text = phrase(rng, _OBJECT_TEXT[min(len(parts), 2)], o=name, r=role)
    plural_role = len(parts) > 1 and role.endswith("s")
    if plural_role:  # "the walls" takes "are"/"make", never "is"/"makes"
        text = text.replace(f" is the {role}", f" are the {role}").replace(f" makes the {role}", f" make the {role}")
    answer = DISPLAY[shape]
    if len(parts) == 1:
        working = f"A {name} is shaped like a {shape}."
    elif plural_role:
        working = f"The {role} of the {name} are {PLURAL.get(shape, shape + 's')}."
    else:
        working = f"The {role} of the {name} is a {shape}."
    distractors = pick_distractors(answer, [DISPLAY[s] for s in _CONFUSED[shape]])
    check(len(distractors) == 3)
    figures.assert_no_leak(fig, [answer, *distractors])
    return _result(text, answer, distractors, [working], fig)


@generator("ey_shape_recognise", level=1, presets=_PRESETS)
def ey_shape_recognise(rng, tier="easy"):
    _check_tier(tier)
    built = {"easy": _recognise_name, "medium": _recognise_count, "hard": _recognise_object}[tier](rng)
    return built


def _read_recognise(q) -> str:
    fig, text = q.figure, q.text
    if fig["object"] is not None:
        obj = fig["object"]
        if len(obj["parts"]) == 1:
            return DISPLAY[obj["parts"][0]["shape"]]
        role = next(
            p["role"] for p in obj["parts"] if f" {p['role']} " in f" {text.replace('.', ' ').replace('?', ' ')} "
        )
        return DISPLAY[object_part_shape(fig, role)]
    items = fig["items"]
    if text.startswith("Which object") or text.startswith("This shape is like"):
        return f"A {_LIKE[items[0]['shape']]}"
    if len(items) == 1:
        return DISPLAY[items[0]["shape"]]
    target = _noun_in(text)
    return str(sum(1 for i in items if i["shape"] == target))


figures.register_reader("ey_shape_recognise", _read_recognise)


# ====================================================================== AC9M1SP01: sort and classify

_SOLID_TEXT = (
    "How many solid shapes can you see?",
    "Count the solid shapes.",
    "How many shapes are solid?",
)
_FLAT_TEXT = (
    "How many flat shapes can you see?",
    "Count the flat shapes.",
    "How many shapes are flat?",
)
# mode -> (prompt patterns, predicate). Semicircle is left out of the corner modes: its two corners are
# arguable for a young learner, and it is a clear case only for a curved side.
_SORT_MODES = {
    "no corners": (
        ("How many shapes have no corners?", "Count the shapes with no corners."),
        lambda f: f["corners"] == 0,
    ),
    "3 corners": (("How many shapes have 3 corners?", "Count the shapes with 3 corners."), lambda f: f["corners"] == 3),
    "4 corners": (("How many shapes have 4 corners?", "Count the shapes with 4 corners."), lambda f: f["corners"] == 4),
    "curved": (("How many shapes have a curved side?", "Count the shapes with a curved side."), lambda f: f["curved"]),
}
_ODD_TEXT = (
    "Which shape is the odd one out? Choose the shape and the reason.",
    "One shape does not belong. Which shape, and why?",
    "Which shape is different? Pick the shape and the reason.",
)
_BIGGER, _TURNED, _SHADED = "it is bigger", "it is turned", "it is shaded"


def _classify_solid(rng):
    want_solid = rng.random() < 0.5
    for _ in attempts(500, "a flat and solid mix"):
        k = rng.randint(1, 4)
        total = rng.randint(max(4, k + 1), 7)
        n_other = total - k
        if n_other == k:  # equal groups would make "how many of the other kind" the same number
            continue
        pool_t, pool_o = (SOLIDS, FLAT) if want_solid else (FLAT, SOLIDS)
        items = [_item(rng, rng.choice(pool_t)) for _ in range(k)] + [
            _item(rng, rng.choice(pool_o)) for _ in range(n_other)
        ]
        rng.shuffle(items)
        fig = _scene(rng, items, alt="a picture of flat shapes and solid shapes")
        try:
            figures.assert_no_leak(fig, [str(k), str(n_other)])
        except AssertionError:
            continue
        break
    text = rng.choice(_SOLID_TEXT if want_solid else _FLAT_TEXT)
    cands = [n_other, k + 1, k - 1, total, k + 2]
    distractors = _numeric_distractors(rng, k, cands, fig)
    word = "solid" if want_solid else "flat"
    return _result(text, str(k), distractors, [f"{k} of the {total} shapes {'is' if k == 1 else 'are'} {word}."], fig)


def _sort_pool(mode: str):
    pred = _SORT_MODES[mode][1]
    yes = [s for s in FLAT if s != "semicircle" or mode == "curved"]
    return [s for s in yes if pred(FACTS[s])], [s for s in yes if not pred(FACTS[s])]


def _classify_sort(rng):
    mode = rng.choice(tuple(_SORT_MODES))
    pool_yes, pool_no = _sort_pool(mode)
    k = rng.randint(1, 4)
    total = rng.randint(max(5, k + 2), 8)
    for _ in attempts(500, "a clear sort"):
        items = [_item(rng, rng.choice(pool_yes)) for _ in range(k)] + [
            _item(rng, rng.choice(pool_no)) for _ in range(total - k)
        ]
        rng.shuffle(items)
        fig = _scene(rng, items)
        try:
            figures.assert_no_leak(fig, [str(k)])
        except AssertionError:
            continue
        break
    text = phrase(rng, _SORT_MODES[mode][0])
    # misconceptions: counts the squares only for 4 corners; counts the shapes that are not in the group
    square_only = sum(1 for i in items if i["shape"] == "square") if mode == "4 corners" else k + 2
    cands = [total - k, k + 1, k - 1, square_only, total]
    distractors = _numeric_distractors(rng, k, cands, fig)
    return _result(text, str(k), distractors, [f"{k} of the {total} shapes: {mode}."], fig)


def _odd_set(rng):
    """(three matching shapes, the odd shape, reason) for the odd-one-out question."""
    kind = rng.choice(("curved", "corners", "sides"))
    if kind == "curved":
        return [rng.choice(_POLYGONS) for _ in range(3)], rng.choice(("circle", "oval")), "it has a curved side"
    if kind == "corners":
        return [rng.choice(("circle", "oval")) for _ in range(3)], rng.choice(_POLYGONS), "it has corners"
    by_sides: dict[int, list[str]] = {}
    for s in _POLYGONS:
        by_sides.setdefault(FACTS[s]["sides"], []).append(s)
    n, m = rng.sample(sorted(by_sides), 2)
    return [rng.choice(by_sides[n]) for _ in range(3)], rng.choice(by_sides[m]), f"it has {m} sides"


def _classify_odd(rng):
    """Four labelled shapes alike in size, style and turn, so only the stated rule singles one out.

    Every reason offered is false of the item it names: a wrong letter with the right reason, and the odd
    letter with a reason about size, turn or shading (all four are the same, so none of those is true).
    """
    same, odd, reason = _odd_set(rng)
    odd_at = rng.randrange(4)
    others = [i for i in range(4) if i != odd_at]
    shapes = [None] * 4
    shapes[odd_at] = odd
    for slot, s in zip(others, same):
        shapes[slot] = s
    size, style = rng.choice(SIZES), rng.choice(("plain", "dotted"))
    items = [_item(rng, s, size=size, style=style, upright=True, label=_LETTERS[i]) for i, s in enumerate(shapes)]
    fig = shape_scene(items=items, alt="a picture of four shapes labelled A to D", layout="row")
    answer = f"{_LETTERS[odd_at]}, {reason}"
    if "curved" in reason:
        wrong = "it has corners"
    elif "corners" in reason:
        wrong = "it has a curved side"
    else:
        wrong = f"it has {FACTS[same[0]]['sides']} sides"
    visuals = [_BIGGER, _TURNED, _SHADED]
    rng.shuffle(visuals)
    cands = [
        f"{_LETTERS[others[0]]}, {reason}",
        f"{_LETTERS[odd_at]}, {visuals[0]}",
        f"{_LETTERS[others[1]]}, {reason}",
        f"{_LETTERS[odd_at]}, {visuals[1]}",
        f"{_LETTERS[odd_at]}, {wrong}",
    ]
    distractors = pick_distractors(answer, cands)
    check(len(distractors) == 3)
    figures.assert_no_leak(fig, [answer, *distractors])
    working = [f"Three shapes match. {_LETTERS[odd_at]} is the odd one out: {reason}."]
    return _result(rng.choice(_ODD_TEXT), answer, distractors, working, fig)


@generator("ey_shape_classify", level=2, presets=_PRESETS)
def ey_shape_classify(rng, tier="easy"):
    _check_tier(tier)
    return {"easy": _classify_solid, "medium": _classify_sort, "hard": _classify_odd}[tier](rng)


def _odd_reading(items) -> str:
    """Re-derive the odd one out and its reason from the drawn shapes alone."""
    shapes = [i["shape"] for i in items]
    curved = [FACTS[s]["curved"] for s in shapes]
    if curved.count(True) == 1:
        return f"{items[curved.index(True)]['label']}, it has a curved side"
    if curved.count(False) == 1:
        return f"{items[curved.index(False)]['label']}, it has corners"
    sides = [FACTS[s]["sides"] for s in shapes]
    lone = [i for i, n in enumerate(sides) if sides.count(n) == 1]
    check(len(lone) == 1, "no single odd shape")
    return f"{items[lone[0]]['label']}, it has {sides[lone[0]]} sides"


def _read_classify(q) -> str:
    fig, text = q.figure, q.text
    items = fig["items"]
    if items and items[0]["label"]:
        return _odd_reading(items)
    if "solid" in text:
        return str(sum(1 for i in items if i["shape"] in SOLIDS))
    if "flat" in text:
        return str(sum(1 for i in items if i["shape"] in FLAT))
    for mode, (patterns, pred) in _SORT_MODES.items():
        if any(text == p for p in patterns):
            return str(sum(1 for i in items if pred(FACTS[i["shape"]])))
    raise ValueError(f"cannot read {text!r}")


figures.register_reader("ey_shape_classify", _read_classify)


# ====================================================================== AC9M2SP01: sides and spatial terms

_SIDES_TEXT = (
    "How many sides does this shape have?",
    "Count the sides of this shape.",
    "How many sides can you count?",
)
_CORNERS_TEXT = (
    "How many corners does this shape have?",
    "Count the corners of this shape.",
    "How many corners can you count?",
)
_PICK_TEXT = {
    "curved": ("Which shape has only curved sides?", "Which shape is curved all the way round?"),
    "straight": ("Which shape has only straight sides?", "Which shape has no curved sides?"),
    "parallel": (
        "Which shape has at least one pair of parallel sides?",
        "Which shape has one or more pairs of parallel sides?",
        "Which shape has a pair of opposite sides that are parallel?",
    ),
}
_PARALLEL_SHAPES = tuple(s for s in FLAT if FACTS[s]["parallel"])
_CURVED_ONLY = ("circle", "oval")
_COUNT_SIDES_TEXT = (
    "How many shapes have exactly {n} sides?",
    "Count the shapes with {n} sides.",
    "How many of these shapes have {n} sides?",
)
_COUNT_PARALLEL_TEXT = (
    "How many shapes have at least one pair of parallel sides?",
    "Count the shapes with at least one pair of parallel sides.",
)


def _sides_count(rng):
    shape = rng.choice(_POLYGONS)
    n = FACTS[shape]["sides"]
    fig = shape_scene(items=[_item(rng, shape)], alt="a picture of a shape")
    corners = rng.random() < 0.4
    text = rng.choice(_CORNERS_TEXT if corners else _SIDES_TEXT)
    cands = [4 if n != 4 else 3, n + 1, n - 1, n + 2, 3, 6]
    distractors = _numeric_distractors(rng, n, cands, fig)
    word = "corners" if corners else "sides"
    return _result(text, str(n), distractors, [f"This shape has {n} {word}."], fig)


def _pick_property(rng):
    mode = rng.choice(("curved", "straight", "parallel"))
    if mode == "curved":
        key = rng.choice(_CURVED_ONLY)
        rest = rng.sample([s for s in _POLYGONS if s != "square"] + ["semicircle"], 3)
        if "semicircle" not in rest:  # the half-curved shape is the misconception, so it is usually there
            rest[0] = "semicircle" if rng.random() < 0.7 else rest[0]
    elif mode == "straight":
        key = rng.choice(_POLYGONS)
        rest = ["circle", "oval", "semicircle"]
    else:
        key = rng.choice(_PARALLEL_SHAPES)
        rest = rng.sample([s for s in FLAT if not FACTS[s]["parallel"] and s != "semicircle"], 3)
    shapes = [key, *rest]
    rng.shuffle(shapes)
    check(len(set(shapes)) == 4)
    items = [_item(rng, s, label=_LETTERS[i]) for i, s in enumerate(shapes)]
    fig = shape_scene(items=items, alt="a picture of four shapes labelled A to D", layout="row")
    answer = f"Shape {_LETTERS[shapes.index(key)]}"
    options = [f"Shape {c}" for c in _LETTERS]
    distractors = pick_distractors(answer, options)
    figures.assert_no_leak(fig, [answer, *distractors])
    working = [f"Shape {_LETTERS[shapes.index(key)]} is the {key}: it has {_PROPERTY_WORDS[mode]}."]
    return _result(phrase(rng, _PICK_TEXT[mode]), answer, distractors, working, fig)


_PROPERTY_WORDS = {
    "curved": "only curved sides",
    "straight": "only straight sides",
    "parallel": "a pair of parallel sides",
}


def _count_sides_scene(rng):
    by_mode = rng.choice(("sides", "sides", "parallel"))
    if by_mode == "sides":
        n = rng.choice((3, 4, 4, 5, 6))
        pool_yes = [s for s in _POLYGONS if FACTS[s]["sides"] == n]
        pred = lambda s: FACTS[s]["sides"] == n  # noqa: E731
        text = phrase(rng, _COUNT_SIDES_TEXT, n=n)
    else:
        pool_yes = list(_PARALLEL_SHAPES)
        pred = lambda s: FACTS[s]["parallel"]  # noqa: E731
        text = rng.choice(_COUNT_PARALLEL_TEXT)
    pool_no = [s for s in FLAT if not pred(s) and s != "semicircle"]  # its edges are not cleanly countable
    k = rng.randint(2, 5)
    for _ in attempts(800, "a scene with a clear count"):
        total = rng.randint(max(7, k + 3), 10)
        yes = [rng.choice(pool_yes) for _ in range(k)]
        no = [rng.choice(pool_no) for _ in range(total - k)]
        if len(pool_yes) > 1 and len(set(yes)) < 2:
            continue  # a mix of kinds, so counting only one kind (the squares) gives a different number
        items = [_item(rng, s) for s in yes + no]
        if not any(i["turn"] for i in items[:k]):
            items[0] = _item(rng, yes[0], turned=True)
        rng.shuffle(items)
        fig = _scene(rng, items)
        try:
            figures.assert_no_leak(fig, [str(k)])
        except AssertionError:
            continue
        break
    first = yes[0]
    one_kind = sum(1 for s in yes if s == first)
    upright = sum(1 for i in items if pred(i["shape"]) and not i["turn"])
    cands = [one_kind if one_kind != k else k + 2, k + 1, k - 1, upright if upright != k else k + 2, total - k]
    distractors = _numeric_distractors(rng, k, cands, fig)
    return _result(text, str(k), distractors, [f"{k} of the {total} shapes match."], fig)


@generator("ey_shape_sides", level=2, presets=_PRESETS)
def ey_shape_sides(rng, tier="easy"):
    _check_tier(tier)
    return {"easy": _sides_count, "medium": _pick_property, "hard": _count_sides_scene}[tier](rng)


def _read_sides(q) -> str:
    fig, text = q.figure, q.text
    items = fig["items"]
    if len(items) == 1:
        return str(FACTS[items[0]["shape"]]["sides"])
    if items[0]["label"]:
        if "curved" in text and "no curved" not in text:
            hits = [i["label"] for i in items if FACTS[i["shape"]]["curved"] and FACTS[i["shape"]]["corners"] == 0]
        elif "straight" in text or "no curved" in text:
            hits = [i["label"] for i in items if FACTS[i["shape"]]["straight_only"]]
        else:
            hits = [i["label"] for i in items if FACTS[i["shape"]]["parallel"]]
        check(len(hits) == 1, "not exactly one matching shape")
        return f"Shape {hits[0]}"
    if "parallel" in text:
        return str(sum(1 for i in items if FACTS[i["shape"]]["parallel"]))
    n = int(re.search(r"\d+", text).group())
    return str(sum(1 for i in items if FACTS[i["shape"]]["sides"] == n))


figures.register_reader("ey_shape_sides", _read_sides)
