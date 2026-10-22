"""Equal groups, sharing and halves with pictures: counters in rings and arrays, and shapes cut into equal parts.

Descriptors: AC9MFN06 (recognise and describe halves), AC9M1N06 (share and group collections equally), AC9M2N03
(partition collections into equal groups and fractions of a collection).

Every group size here is 20 or fewer counters in all. Difficulty escalates in the operation, not the numbers:

- easy: equal rings of counters; type how many counters there are in all, or how many are in each ring when the
  total is stated (the number of rings has to be read from the picture).
- medium: counters in an array shared equally among 2, 3 or 4 children with none left over (the total is counted from
  the picture); or half of a group of counters, or how many parts make up half of a shape cut into equal parts.
- hard: sharing that leaves counters over (how many each, or how many are left); or a quarter of a group or of a
  shape cut into equal parts.
"""

from __future__ import annotations

from .. import figures
from ..core import generator
from ..figure_kinds.early import counters, counters_total, fraction_model, fraction_shape
from ..slots import NAMES
from ._ey2a_kit import SHAPE_WORD, number_question, scatter20, tagged_numbers

_PRESETS = {"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")}

_WHOLE_WORD = {"bar": "bar", "rectangle": "rectangle"}
_BAR_PARTS = (4, 6, 8, 10, 12)  # even, so a half is a whole number of parts
_QUARTER_PARTS = (4, 8, 12)
_SHAPE_KINDS = ("bar", "rectangle")
_RECT = {4: (2, 2), 6: (2, 3), 8: (2, 4), 10: (2, 5), 12: (3, 4)}


def _model(rng, kind: str, parts: int) -> dict:
    """One whole cut into ``parts`` equal parts with a few shaded at random (the shading is not part of the
    question: it only keeps the picture from being a bare outline)."""
    shaded = rng.sample(range(parts), rng.randint(1, parts - 1))
    if kind == "rectangle":
        rows, cols = _RECT[parts]
        return fraction_shape("rectangle", parts, shaded, rows=rows, cols=cols)
    return fraction_shape("bar", parts, shaded)


def _shape_question(rng, name: str, divisor: int) -> dict:
    """How many parts make a half (``divisor`` 2) or a quarter (4) of a whole cut into equal parts."""
    kind = rng.choice(_SHAPE_KINDS)
    parts = rng.choice(_BAR_PARTS if divisor == 2 else _QUARTER_PARTS)
    fig = fraction_model(models=[_model(rng, kind, parts)], alt=f"a {_WHOLE_WORD[kind]} cut into equal parts")
    answer = parts // divisor
    fraction = "half" if divisor == 2 else "a quarter"
    text = rng.choice(
        (
            "The {w} is cut into equal parts. How many parts make {f} of the {w}?",
            "{n} wants {f} of the {w}. How many parts is that?",
        )
    ).format(n=name, w=_WHOLE_WORD[kind], f=fraction)
    cands = [
        (parts, "ey2a-reports-the-whole"),
        (answer + 1, "ey2a-count-one-over"),
        (answer - 1, "ey2a-count-one-short"),
        (parts - answer, "ey2a-fraction-as-the-rest") if divisor == 4 else (parts + 1, "ey2a-count-one-over"),
        (parts // 2 if divisor == 4 else parts // 4, "ey2a-mixes-half-and-quarter"),
        (answer + 2, "ey2a-count-one-over"),
    ]
    working = [
        f"The {_WHOLE_WORD[kind]} has {parts} equal parts. {fraction.capitalize()} is {parts} shared in {divisor}: {answer}."
    ]
    return dict(fig=fig, answer=answer, text=text, cands=cands, working=working, reads=("shape", parts, divisor))


def _share_question(rng, name: str, tier: str, shape: str) -> dict:
    word = SHAPE_WORD[shape]
    children = rng.choice((2, 3, 4))
    if tier == "medium":
        each = rng.randint(2, 20 // children)
        total = children * each
        left = 0
    else:
        each = rng.randint(2, (19 - (children - 1)) // children)
        left = rng.randint(1, children - 1)
        total = children * each + left
    dots = scatter20(rng, total)
    fig = counters(layout="scatter", alt="some counters to share", shape=shape, dots=dots)
    if tier == "medium":
        answer = each
        text = rng.choice(
            (
                "{n} shares the {w} equally among {c} children. How many does each get?",
                "Share the {w} equally among {c} children. How many does each child get?",
            )
        ).format(n=name, w=word, c=children)
        cands = [
            (total, "ey2a-reports-the-whole"),
            (children, "ey2a-reports-the-divisor"),
            (each + 1, "ey2a-count-one-over"),
            (each - 1, "ey2a-count-one-short"),
            (total - children, "ey2a-subtracts-for-sharing"),
            (each + 2, "ey2a-count-one-over"),
        ]
        working = [f"There are {total} {word}. {total} shared among {children} is {each} each."]
    else:
        if rng.random() < 0.5:
            answer = each
            text = rng.choice(
                (
                    "{n} deals the {w} equally among {c} children, one at a time. How many does each get?",
                    "Share the {w} equally among {c} children. How many does each child get?",
                )
            ).format(n=name, w=word, c=children)
            cands = [
                (left, "ey2a-reports-the-left-over"),
                (each + 1, "ey2a-rounds-sharing-up"),
                (total // (children - 1), "ey2a-wrong-divisor"),
                (each - 1, "ey2a-count-one-short"),
                (total, "ey2a-reports-the-whole"),
                (each + 2, "ey2a-count-one-over"),
            ]
            working = [
                f"{children} x {each} = {children * each}. Each child gets {each} and {left} {'is' if left == 1 else 'are'} left over."
            ]
        else:
            answer = left
            text = rng.choice(
                (
                    "{n} deals the {w} equally among {c} children. How many are left over?",
                    "Share the {w} equally among {c} children. How many are left over?",
                )
            ).format(n=name, w=word, c=children)
            cands = [
                (each, "ey2a-reports-each-share"),
                (children - left, "ey2a-left-over-the-wrong-way"),
                (total, "ey2a-reports-the-whole"),
                (children, "ey2a-reports-the-divisor"),
                (answer + 2, "ey2a-count-one-over"),
                (answer + 3, "ey2a-count-one-over"),
            ]
            working = [f"{children} x {each} = {children * each}. {total} - {children * each} = {left} left over."]
    return dict(fig=fig, answer=answer, text=text, cands=cands, working=working, reads=("share", children))


def _half_group_question(rng, name: str, shape: str, divisor: int) -> dict:
    word = SHAPE_WORD[shape]
    each = rng.randint(1, 10 if divisor == 2 else 5)
    total = each * divisor
    dots = scatter20(rng, total)
    fig = counters(layout="scatter", alt="some counters in a group", shape=shape, dots=dots)
    answer = total // divisor
    fraction = "half" if divisor == 2 else "a quarter"
    text = rng.choice(("{n} takes {f} of the {w}. How many does {n} take?", "What is {f} of the group of {w}?")).format(
        n=name, w=word, f=fraction
    )
    cands = [
        (total, "ey2a-reports-the-whole"),
        (answer + 1, "ey2a-count-one-over"),
        (answer - 1, "ey2a-count-one-short"),
        (total - answer, "ey2a-fraction-as-the-rest"),
        (total // 2 if divisor == 4 else total // 4, "ey2a-mixes-half-and-quarter"),
        (answer + 2, "ey2a-count-one-over"),
    ]
    working = [f"There are {total} {word}. {fraction.capitalize()} is {total} shared in {divisor}: {answer}."]
    return dict(fig=fig, answer=answer, text=text, cands=cands, working=working, reads=("group", divisor))


def _rings_question(rng, name: str, shape: str) -> dict:
    word = SHAPE_WORD[shape]
    rings = rng.randint(2, 5)
    each = rng.randint(2, min(9, 20 // rings))
    fig = counters(
        layout="groups", alt="rings with the same number of counters in each", shape=shape, groups=[each] * rings
    )
    if rng.random() < 0.5:
        answer = rings * each
        text = rng.choice(
            (
                "Each ring has the same number of {w}. How many {w} in all?",
                "{n} counts all the {w} in the rings. How many?",
            )
        ).format(n=name, w=word)
        cands = [
            (rings + each, "ey2a-adds-rings-and-size"),
            (each, "ey2a-reports-group-size"),
            (rings, "ey2a-counts-rings-not-counters"),
            (answer + 1, "ey2a-count-one-over"),
            (answer - 1, "ey2a-count-one-short"),
            (answer + each, "ey2a-count-one-over"),
        ]
        working = [f"There are {rings} rings with {each} in each: {rings} x {each} = {answer}."]
        reads = ("rings-total",)
    else:
        total = rings * each
        answer = each
        text = rng.choice(
            (
                "There are {t} {w} in all, in equal rings. How many are in each ring?",
                "{n} puts {t} {w} into the rings equally. How many are in each ring?",
            )
        ).format(n=name, t=total, w=word)
        cands = [
            (rings, "ey2a-counts-rings-not-counters"),
            (total, "ey2a-reports-the-whole"),
            (total - rings, "ey2a-subtracts-for-sharing"),
            (answer + 1, "ey2a-count-one-over"),
            (answer - 1, "ey2a-count-one-short"),
            (answer + 2, "ey2a-count-one-over"),
        ]
        working = [f"{total} counters in {rings} equal rings: {total} shared in {rings} is {each}."]
        reads = ("rings-each", total)
    return dict(fig=fig, answer=answer, text=text, cands=cands, working=working, reads=reads)


@generator("ey2a_equal_groups_sharing_halves", level=1, presets=_PRESETS)
def ey2a_equal_groups_sharing_halves(rng, tier="easy"):
    shape = rng.choice(sorted(SHAPE_WORD))
    name = rng.choice(NAMES)
    if tier == "easy":
        built = _rings_question(rng, name, shape)
    elif tier == "medium":
        pick = rng.choice(("share", "half-group", "half-shape"))
        if pick == "share":
            built = _share_question(rng, name, "medium", shape)
        elif pick == "half-group":
            built = _half_group_question(rng, name, shape, 2)
        else:
            built = _shape_question(rng, name, 2)
    elif tier == "hard":
        pick = rng.choice(("share", "quarter-group", "quarter-shape"))
        if pick == "share":
            built = _share_question(rng, name, "hard", shape)
        elif pick == "quarter-group":
            built = _half_group_question(rng, name, shape, 4)
        else:
            built = _shape_question(rng, name, 4)
    else:
        raise ValueError(f"unknown tier {tier!r}")
    answer, fig = built["answer"], built["fig"]
    figures.assert_no_leak(fig, [str(answer)])
    return number_question(
        text=built["text"],
        answer=answer,
        distractors=tagged_numbers(answer, built["cands"], low=1, high=30),
        working=built["working"],
        figure=fig,
    )


def _read(q) -> str:
    """Re-derive the answer from the drawn figure and the stem alone."""
    fig, text = q.figure, q.text
    nums = [int(t) for t in text.replace("?", " ").replace(".", " ").split() if t.isdigit()]
    if fig["kind"] == "fraction_model":
        parts = fig["models"][0]["parts"]
        return str(parts // (4 if "quarter" in text else 2))
    if fig["layout"] == "groups":
        rings = fig["groups"]
        return str(sum(rings)) if not nums else str(nums[0] // len(rings))
    total = counters_total(fig)
    if "left over" in text:
        return str(total % nums[0])
    if "half" in text:
        return str(total // 2)
    if "quarter" in text:
        return str(total // 4)
    return str(total // nums[0])


figures.register_reader("ey2a_equal_groups_sharing_halves", _read)
