"""Deciding whether a solid is a right prism from the data given about its cross-sections.

``prism_cross_sections_forced`` replaces the two-end-widths wording of ``prism_cross_sections`` at the medium and
hard tiers. Every question states enough for exactly one of three answers, and each answer is correct in about a
third of draws:

- ``It is a right prism``: the solid is built by sweeping or stacking one shape, so every cross-section is the same
  (medium: a shape pushed straight along its length; hard: identical slices stacked, every slice measured).
- ``It is not a right prism``: two cross-sections that are given differ in size (or, at hard, in shape), and a right
  prism has the same cross-section everywhere.
- ``There is not enough information to tell``: only the cross-sections at the two ends are given and they match, so
  the solid might bulge in between (medium); or only the top and bottom slices of a longer stack are measured
  (hard).

Medium reads one stated fact; hard has to notice which slices were measured and compare all of them.
Distractors are the two other answers, so three options are always shown.
"""

from __future__ import annotations

from mathgen.core import check, generator, latex_answer

SHAPES = ("rectangular", "triangular", "hexagonal", "pentagonal", "square")
PRISM = "It is a right prism"
NOT_PRISM = "It is not a right prism"
UNKNOWN = "There is not enough information to tell"
_OPTIONS = (PRISM, NOT_PRISM, UNKNOWN)

_ASKS = (
    "What can you say about the solid?",
    "Which statement about the solid is correct?",
    "Can you tell whether the solid is a right prism?",
)
_ORDINALS = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}


def _verdict(widths, shapes, ends_only):
    """Independent rule: a right prism has one cross-section throughout, so a difference settles it, and equal
    cross-sections settle it only if every one was seen."""
    if len(set(widths)) > 1 or len(set(shapes)) > 1:
        return NOT_PRISM
    return UNKNOWN if ends_only else PRISM


def _other_width(rng, m):
    return rng.choice([w for w in range(2, 13) if w != m])


def _medium(rng):
    shape = rng.choice(SHAPES)
    length = rng.randint(6, 20)
    m = rng.randint(2, 12)
    family = rng.choice(("ends_differ", "ends_equal", "pushed"))
    if family == "pushed":
        text = (
            f"A {shape} flat shape measuring {m} cm across is pushed straight forward {length} cm, at right "
            f"angles to its flat surface, and leaves a solid behind. {rng.choice(_ASKS)}"
        )
        widths, shapes, ends_only = [m, m], [shape, shape], False
        working = [f"Pushing one shape straight along its length makes every cross-section a {m} cm {shape} shape."]
    else:
        m2 = _other_width(rng, m) if family == "ends_differ" else m
        left, right = (m, m2) if rng.random() < 0.5 else (m2, m)
        text = (
            f"A solid is {length} cm long. Its cross-sections, cut at right angles to its length, are {shape} "
            f"shapes. The one at the left end measures {left} cm across and the one at the right end measures "
            f"{right} cm across. {rng.choice(_ASKS)}"
        )
        widths, shapes, ends_only = [left, right], [shape, shape], True
        working = (
            [f"The end cross-sections are {left} cm and {right} cm across. A right prism has the same one everywhere."]
            if family == "ends_differ"
            else [f"Both ends measure {m} cm, but nothing is said about the middle, so it could bulge or pinch."]
        )
    answer = _verdict(widths, shapes, ends_only)
    return text, answer, working


def _hard(rng):
    shape = rng.choice(SHAPES)
    k = rng.randint(4, 6)
    thick = rng.randint(2, 5)
    m = rng.randint(2, 12)
    family = rng.choice(("all_equal", "one_differs", "ends_only", "shape_differs"))
    widths = [m] * k
    shapes = [shape] * k
    ends_only = False
    if family == "one_differs":
        i = rng.randrange(k)
        widths[i] = _other_width(rng, m)
        measured = f"The slices measure, from the bottom up, {', '.join(str(w) for w in widths)} cm across."
        working = [
            f"The {_ORDINALS[i + 1]} slice is {widths[i]} cm across but the others are {m} cm, so the cross-sections differ."
        ]
    elif family == "shape_differs":
        i = rng.randrange(k)
        other = rng.choice([s for s in SHAPES if s != shape])
        shapes[i] = other
        measured = (
            f"Every slice measures {m} cm across, but the {_ORDINALS[i + 1]} slice from the bottom is a {other} shape "
            f"and the rest are {shape}."
        )
        working = [f"The {_ORDINALS[i + 1]} slice is a different shape from the rest, so the cross-sections differ."]
    elif family == "ends_only":
        ends_only = True
        top = m
        measured = (
            f"Only the bottom slice and the top slice were measured: each is {top} cm across. "
            f"The slices in between were not measured."
        )
        working = ["The bottom and top slices match, but the slices between them were not measured."]
    else:
        measured = f"All {k} slices were measured and each is {m} cm across."
        working = [f"All {k} slices are the same {m} cm {shape} shape stacked directly on top of each other."]
    if family in ("all_equal", "ends_only"):
        intro_shape = f"Every slice is a {shape} shape."
    elif family == "one_differs":
        intro_shape = f"Every slice is a {shape} shape."
    else:
        intro_shape = "The slices are flat shapes."
    text = (
        f"A solid is made by stacking {k} flat slices, each {thick} cm thick, directly on top of each other. "
        f"{intro_shape} {measured} {rng.choice(_ASKS)}"
    )
    if family == "ends_only":
        answer = _verdict([m, m], shapes[:2], True)
    elif family == "shape_differs":
        answer = _verdict(widths, shapes, False)
    else:
        answer = _verdict(widths, shapes, False)
    return text, answer, working


@generator(
    "prism_cross_sections_forced",
    level=6,
    presets={"medium": {"tier": "medium"}, "hard": {"tier": "hard"}},
)
def prism_cross_sections_forced(rng, tier="medium"):
    if tier not in ("medium", "hard"):
        raise ValueError("tier must be 'medium' or 'hard'")
    text, answer, working = (_medium if tier == "medium" else _hard)(rng)
    check(answer in _OPTIONS, "unknown verdict")
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer,
        answer_latex=latex_answer(answer, plain=True),
        working=working,
        distractors=[o for o in _OPTIONS if o != answer],
    )
