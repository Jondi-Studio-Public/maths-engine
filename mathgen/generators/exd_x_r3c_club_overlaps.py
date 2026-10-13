"""Exam source for `senior-determine-and-use-the-formulas-for-finding-the`: a three-set count worked out in stages.

The stem gives a group of N people, how many are in none of three activities, the size of each activity's group and
the size of each pairwise overlap (every overlap count includes the people in all three). It is built backwards from
the eight regions of the Venn diagram.

(a) the number in at least one activity: N less the number in none;
(b) the number in all three, from the three-set formula and (a);
(c) the number in exactly two, from the overlaps and (b);
(d) the number in exactly one, from (a), (b) and (c).

Parts (b), (c) and (d) carry `follow_through`. The key is re-derived by adding the eight region counts the draw
started from.
"""

from __future__ import annotations

from mathgen.core import check, generator

from .exd_x_r3c_parts import number_part
from ._exd_distinct_parts import distinct_part_answers

# (people noun, opening sentence, plural for the groups, singular for one group, three group names)
_CONTEXTS = (
    ("students", "A school has {n} students.", "clubs", "club", ("film", "photography", "journalism")),
    ("students", "A school with {n} students runs three clubs.", "clubs", "club", ("chess", "debating", "robotics")),
    ("students", "{n} students are enrolled in a year level.", "groups", "group", ("drama", "choir", "orchestra")),
    ("members", "A community centre has {n} members.", "classes", "class", ("yoga", "pottery", "swimming")),
    (
        "people",
        "{n} people at a camp were asked about the activities they joined.",
        "activities",
        "activity",
        ("kayaking", "archery", "climbing"),
    ),
)


@generator("exd_x_r3c_club_overlaps", level=15, presets={"exam": {"tier": "exam"}})
@distinct_part_answers
def exd_x_r3c_club_overlaps(rng, tier="exam"):
    """N, the number in none, the set sizes and the pairwise overlaps; then all three, exactly two, exactly one."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    noun, intro, group, one, names = rng.choice(_CONTEXTS)
    only = [rng.randint(3, 14) for _ in range(3)]
    two_only = [rng.randint(1, 9) for _ in range(3)]  # AB, AC, BC only
    triple = rng.randint(1, 6)
    none = rng.randint(2, 15)
    ab_only, ac_only, bc_only = two_only
    total = sum(only) + sum(two_only) + triple + none
    size = [
        only[0] + ab_only + ac_only + triple,
        only[1] + ab_only + bc_only + triple,
        only[2] + ac_only + bc_only + triple,
    ]
    ab, ac, bc = ab_only + triple, ac_only + triple, bc_only + triple
    sizes, pairs = sum(size), ab + ac + bc

    union = total - none
    check(union == sum(only) + sum(two_only) + triple, "the regions do not add to the union")
    check(union == sizes - pairs + triple, "the three-set formula disagrees with the regions")
    exactly_two = sum(two_only)
    exactly_one = sum(only)
    check(exactly_two == pairs - 3 * triple and exactly_one == union - exactly_two - triple, "region counts disagree")

    stem = (
        f"{intro.format(n=total)} In all, {none} of them are in none of the {group}. "
        f"The {names[0]} {one} has {size[0]} {noun}, the {names[1]} {one} has {size[1]} and the {names[2]} {one} has "
        f"{size[2]}. {ab} are in both {names[0]} and {names[1]}, {ac} are in both {names[0]} and {names[2]}, and "
        f"{bc} are in both {names[1]} and {names[2]}. Each of these three counts includes the {noun} who are in "
        f"all three {group}."
    )

    def wrongs(*pairs_):
        return [(v, tag) for v, tag in pairs_ if v >= 0]

    part_a = number_part(
        f"How many of the {noun} are in at least one of the {group}?",
        union,
        working=[f"{total} - {none} = {union}."],
        wrongs=wrongs(
            (total + none, "confuses-neither-with-union"),
            (none, "confuses-neither-with-union"),
            (total, "ignored-neither-count"),
        ),
    )
    part_b = number_part(
        f"Using your answer to (a), find the number of {noun} who are in all three.",
        triple,
        marks=2,
        working=[
            f"{union} = {size[0]} + {size[1]} + {size[2]} - ({ab} + {ac} + {bc}) + (number in all three).",
            f"So the number in all three is {union} - {sizes} + {pairs} = {triple}.",
        ],
        wrongs=wrongs(
            (total - sizes + pairs, "confuses-neither-with-union"),
            (pairs - triple, "reported-total-overlap-as-triple"),
            (sizes - pairs, "inclusion-exclusion-sign-error"),
        ),
        follow_through=f"part_a - {sizes} + {pairs}",
    )
    part_c = number_part(
        f"Using your answer to (b), find the number of {noun} who are in exactly two of the {group}.",
        exactly_two,
        working=[
            f"Each overlap count includes those in all three, so exactly two is {ab} + {ac} + {bc} - 3 × {triple} = {exactly_two}."
        ],
        wrongs=wrongs(
            (pairs - triple, "subtracts-triple-once"),
            (pairs, "forgets-triple-region"),
            (pairs - 2 * triple, "subtracts-triple-twice"),
        ),
        follow_through=f"{pairs} - 3*part_b",
    )
    part_d = number_part(
        f"Using your answers to (a), (b) and (c), find the number of {noun} who are in exactly one of the {group}.",
        exactly_one,
        working=[f"{union} - {exactly_two} - {triple} = {exactly_one}."],
        wrongs=wrongs(
            (union - exactly_two, "omits-triple-from-exactly-one"),
            (sizes - pairs, "single-counted-overlaps"),
            (sizes - 2 * exactly_two, "single-counted-overlaps"),
        ),
        follow_through="part_a - part_b - part_c",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
