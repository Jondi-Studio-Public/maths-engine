"""Exam chain on a shape-sorting procedure applied to a set of shapes: trace each shape, count and total the
perimeters of those landing in one bin, find their mean, then see how a late shape moves it.

The stem states the ten-bin procedure of the exam tree and the side lengths and facts of six shapes. Four parts:

(a) how many of the six shapes are placed in Bin X;
(b) the total perimeter of those shapes;
(c) hence their mean perimeter, to 1 decimal place;
(d) the mean perimeter once a seventh shape (which the procedure also sends to Bin X) is sorted too.

Built backwards from the bin: k shapes of the bin's type, the rest drawn from other bins, then shuffled. Each
shape is traced by the shared tree walker and the count is re-derived from the rendered facts.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import check, generator, phrase

from .exd_x_r2j_support import number_part
from .p15d_build import (
    _ALL_TYPES,
    _TEST_PROMPT_VARIANTS,
    _TREE_EXAM,
    _all_leaves,
    _all_tests,
    _attrs_for,
    _describe_tree,
    _quad_sides,
    _trace,
    _triangle_sides,
    _types_reaching,
)

_LABELS = "PQRSTU"


def _sides(rng: random.Random, shape: str, type_name: str) -> list[int]:
    return _triangle_sides(rng, type_name) if shape == "triangle" else _quad_sides(rng, type_name)


def _shape_sentence(label: str, shape: str, type_name: str, sides: list[int]) -> str:
    attrs = _attrs_for(shape, type_name)
    word = "triangle" if shape == "triangle" else "quadrilateral"
    parts = [f"Shape {label} is a {word} with sides {', '.join(str(s) for s in sides)} units."]
    parts.append(
        "One of its angles is a right angle." if attrs["right_angle"] else "None of its angles is a right angle."
    )
    if shape == "quad":
        parts.append(
            {
                "both": "Both pairs of opposite sides are parallel.",
                "one": "Exactly one pair of opposite sides is parallel.",
                "none": "No pair of opposite sides is parallel.",
            }[attrs["parallel"]]
        )
    return " ".join(parts)


def _neighbour(leaves: list[str], target: str) -> str:
    """The bin next to `target` in the tree's leaf order, the one a mis-read branch lands in."""
    i = leaves.index(target)
    return leaves[i - 1] if i else leaves[1]


@generator("exd_x_r2j_shape_sorting_chain", level=12, presets={"exam": dict(tier="exam")})
def exd_x_r2j_shape_sorting_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Count, total and average the perimeters of the shapes a stated procedure sends to one bin."""
    check(tier == "exam", "exd_x_r2j_shape_sorting_chain: only the exam tier exists")
    leaves = _all_leaves(_TREE_EXAM)
    by_leaf = _types_reaching(_TREE_EXAM)
    target = rng.choice(leaves)
    k = rng.randint(2, 4)
    drawn = [(*rng.choice(by_leaf[target]), target) for _ in range(k)]
    others = [leaf for leaf in leaves if leaf != target]
    neighbour = _neighbour(leaves, target)
    drawn.append((*rng.choice(by_leaf[neighbour]), neighbour))  # one shape in the bin a mis-read lands in
    for _ in range(6 - len(drawn)):
        leaf = rng.choice(others)
        drawn.append((*rng.choice(by_leaf[leaf]), leaf))
    rng.shuffle(drawn)
    shapes = [(shape, type_name, _sides(rng, shape, type_name)) for shape, type_name, _ in drawn]
    late = ("triangle" if by_leaf[target][0][0] == "triangle" else "quad", by_leaf[target][0][1])
    late_sides = _sides(rng, *late)

    # trace every shape twice: the walker and the leaf the shape was drawn for
    bins = []
    for (shape, type_name, sides), (_s, _t, leaf) in zip(shapes, drawn):
        got = _trace(_TREE_EXAM, _attrs_for(shape, type_name))
        check(got == leaf, "traced bin disagrees with the bin the shape was drawn for")
        bins.append(got)
    check(_trace(_TREE_EXAM, _attrs_for(*late)) == target, "late shape does not land in the target bin")

    in_bin = [i for i, b in enumerate(bins) if b == target]
    count = len(in_bin)
    total = sum(sum(shapes[i][2]) for i in in_bin)
    mean = Fraction(total, count)
    late_perimeter = sum(late_sides)
    new_mean = Fraction(total + late_perimeter, count + 1)
    check(count == bins.count(target) and 2 <= count <= 4, "count of the target bin")

    tests = sorted(_all_tests(_TREE_EXAM))
    prompts = {t: rng.choice(_TEST_PROMPT_VARIANTS[t]) for t in tests}
    procedure = _describe_tree(_TREE_EXAM, prompts)
    sentences = " ".join(_shape_sentence(_LABELS[i], *shapes[i]) for i in range(6))
    late_sentence = _shape_sentence("V", *late, late_sides).replace("Shape V is", "A seventh shape, Shape V, is")
    intro = phrase(
        rng,
        [
            "A shape-sorting procedure is defined by these numbered tests: {procedure}",
            "Shapes are sorted into bins by a procedure with these numbered tests: {procedure}",
        ],
        procedure=procedure,
    )
    stem = f"{intro} Six shapes are sorted by this procedure. {sentences}"

    neighbour_in = [i for i, b in enumerate(bins) if b == neighbour]
    all_total = sum(sum(s[2]) for s in shapes)
    neighbour_total = sum(sum(shapes[i][2]) for i in neighbour_in)

    part_a = number_part(
        prompt=f"How many of the six shapes are placed in {target}?",
        exact=count,
        places=0,
        marks=2,
        working=[f"Tracing each shape through the tests, {', '.join(_LABELS[i] for i in in_bin)} reach {target}."],
        wrong=[
            (len(neighbour_in), "misreads-a-branch"),
            (count + 1, "counted-a-shape-from-the-wrong-bin"),
            (6 - count, "counted-the-shapes-left-out"),
            (count - 1, "left-out-a-shape"),
        ],
        sensible_range=(0, 6),
    )
    part_b = number_part(
        prompt=f"Find the total perimeter of the shapes placed in {target}, in units.",
        exact=total,
        places=0,
        marks=2,
        working=[
            "Perimeters: "
            + ", ".join(f"{_LABELS[i]} = {sum(shapes[i][2])}" for i in in_bin)
            + f". Total = {total} units."
        ],
        wrong=[
            (all_total, "ignored-the-sorting"),
            (neighbour_total, "misreads-a-branch"),
            (total - min(sum(shapes[i][2]) for i in in_bin), "left-out-a-shape"),
            (total + min(sum(shapes[i][2]) for i in neighbour_in), "counted-a-shape-from-the-wrong-bin"),
        ],
        sensible_range=(0, None),
    )
    part_c = number_part(
        prompt=f"Hence find the mean perimeter of the shapes in {target}, correct to 1 decimal place.",
        exact=mean,
        places=1,
        marks=1,
        working=[f"Mean = {total} / {count} = {float(mean):.1f} units."],
        wrong=[
            (Fraction(total, 6), "divided-by-all-the-shapes"),
            (Fraction(all_total, 6), "ignored-the-sorting"),
            (Fraction(total, count + 1), "divided-by-wrong-amount"),
            (total, "total-instead-of-mean"),
        ],
        follow_through="part_b/part_a",
        sensible_range=(0, None),
    )
    part_d = number_part(
        prompt=f"{late_sentence} It is sorted by the same procedure. Find the mean perimeter of the shapes in "
        f"{target} once Shape V has been sorted too, correct to 1 decimal place.",
        exact=new_mean,
        places=1,
        marks=2,
        working=[
            f"Shape V also reaches {target}, with perimeter {late_perimeter}.",
            f"New mean = ({total} + {late_perimeter}) / {count + 1} = {float(new_mean):.1f} units.",
        ],
        wrong=[
            (mean, "ignored-the-new-shape"),
            (Fraction(total + late_perimeter, count), "new-shape-not-counted"),
            (Fraction(total + late_perimeter, 7), "divided-by-all-the-shapes"),
            ((mean + late_perimeter) / 2, "average-of-averages"),
        ],
        follow_through=f"(part_b + {late_perimeter})/(part_a + 1)",
        sensible_range=(0, None),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
