"""Generates `p14b_polygon_classification` questions for `f10-polygon-classification`: classify a triangle or
quadrilateral from its stated side lengths, angle measures and parallel-side relationships.

Full answer space, stated up front:

- **easy** (triangle, sides only): ``{scalene, isosceles, equilateral}`` --
  3 classes, single feature (equal-side count).
- **medium** (triangle, sides AND angles together): ``{scalene, not
  right-angled; isosceles, not right-angled; scalene right-angled; isosceles
  right-angled}`` -- 4 classes, ALL FOUR of them full two-fact compounds. Every
  offered option at this tier now names BOTH the side-equality fact and the
  right-angle fact, because both facts are always stated and the prompt asks
  for "the most specific type" of both -- so the tier's answer space is
  exactly the 2 (side class) x 2 (right-angle presence) partition, and each
  cell has exactly one true label among the four offered every draw.
- **hard** (quadrilateral, side-equalities AND parallel-pair count):
  ``{parallelogram, rhombus, trapezium}`` -- 3 classes, exactly the pool the
  worked example for this tier gives.
- **exam** (quadrilateral, side-equalities AND right-angle presence, both
  pairs of opposite sides always stated parallel): ``{parallelogram,
  rectangle, rhombus, square}`` -- 4 classes. The genuinely different
  reasoning step over hard: every exam shape already has 2 parallel pairs
  (so parallelism is never the deciding fact here), and the learner must
  notice a SECOND promoting fact (right angles) on top of the equal-sides
  fact hard already tests, catching the "stopped at rhombus, missed the
  right-angle fact that promotes it to square" under-read.

The independent in-generator check (`_classify_*_independent` functions) is
a genuinely different route from construction: it recomputes the class from
the drawn `sides`/`angles` values by counting equal sides and (for the
tiers that draw them) checking for a stated right angle / parallel-pair
count, deliberately never sharing the branch construction itself took to
choose the target class. The oracle
 goes one step further
and never imports this module at all -- it parses `question.text` and
enumerates every template x property-slot combination.
"""

from __future__ import annotations

import math

from ..core import check, generator, phrase, pick_distractors

# -- Answer-space vocabulary (construction side) -----------------------------

EASY_CLASSES = ["scalene", "isosceles", "equilateral"]
MEDIUM_CLASSES = [
    "scalene, not right-angled",
    "isosceles, not right-angled",
    "scalene right-angled",
    "isosceles right-angled",
]
HARD_CLASSES = ["parallelogram", "rhombus", "trapezium"]
EXAM_CLASSES = ["parallelogram", "rectangle", "rhombus", "square"]

_SIDE_LO, _SIDE_HI = 3, 15


def _format_list3(values) -> str:
    return f"{values[0]}, {values[1]} and {values[2]}"


def _format_list4(values) -> str:
    return f"{values[0]}, {values[1]}, {values[2]} and {values[3]}"


# -- Easy: triangle, sides only ----------------------------------------------


def _classify_by_sides_independent(sides) -> str:
    """Independent re-derivation: count how many of the three sides share the
    same length via pairwise comparison (never the branch construction used
    to pick the target class)."""
    a, b, c = sides
    pairs_equal = (a == b) + (b == c) + (a == c)
    if pairs_equal == 3:
        return "equilateral"
    if pairs_equal >= 1:
        return "isosceles"
    return "scalene"


def _draw_easy(rng):
    cls = rng.choice(EASY_CLASSES)
    if cls == "equilateral":
        s = rng.randint(_SIDE_LO, _SIDE_HI)
        sides = [s, s, s]
    elif cls == "isosceles":
        a = rng.randint(_SIDE_LO, _SIDE_HI)
        b = rng.randint(_SIDE_LO, _SIDE_HI)
        while b == a:
            b = rng.randint(_SIDE_LO, _SIDE_HI)
        sides = [a, a, b]
        rng.shuffle(sides)
    else:
        while True:
            sides = [rng.randint(_SIDE_LO, _SIDE_HI) for _ in range(3)]
            if len(set(sides)) == 3:
                break
    return cls, sides


_EASY_TEMPLATES = [
    "A triangle has side lengths {sides} units. Classify the triangle by its sides, "
    "giving the most specific type that applies.",
    "The three sides of a triangle measure {sides} units. What is the most specific type of "
    "triangle this is, based on its sides?",
    "A triangle's sides are {sides} units long. Classify it by its side lengths, choosing the "
    "single most specific type.",
]


def _build_easy(rng):
    cls, sides = _draw_easy(rng)
    check(
        _classify_by_sides_independent(sides) == cls,
        f"p14b_polygon_classification: easy-tier independent check disagrees ({sides} -> "
        f"{_classify_by_sides_independent(sides)}, expected {cls})",
    )

    text = phrase(rng, _EASY_TEMPLATES, sides=_format_list3(sides))
    answer = cls
    distractors_list = [c for c in EASY_CLASSES if c != cls]

    working = [
        f"Side lengths: {_format_list3(sides)}.",
        f"Equal-side count: {'all three equal' if cls == 'equilateral' else 'exactly two equal' if cls == 'isosceles' else 'no two equal'}.",
        f"Classification: {answer}.",
    ]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer,
        answer_latex=f"plain:{answer}",
        working=working,
        distractors=pick_distractors(answer, distractors_list, n=3),
    )


# -- Medium: triangle, sides AND angles together -----------------------------


def _classify_side_angle_independent(sides, angles) -> str:
    """Independent re-derivation from the drawn side/angle values: an
    equal-side pair count and a stated-right-angle check, combined -- never
    the branch construction itself used to pick the target class. Every
    returned label names BOTH facts."""
    a, b, c = sides
    is_isosceles = (a == b) or (b == c) or (a == c)
    is_right = 90 in angles
    if is_isosceles and is_right:
        return "isosceles right-angled"
    if is_isosceles:
        return "isosceles, not right-angled"
    if is_right:
        return "scalene right-angled"
    return "scalene, not right-angled"


def _sides_from_angles(rng, angles):
    """Derive plausible integer side lengths FROM the drawn angles via the
    law of sines (side proportional to sin(opposite angle)), scaled so the
    largest side lands in a readable range and rounded to the nearest
    integer -- never independently redrawn. Two equal angles always produce
    exactly equal sides (same sin value, same rounding), preserving the
    isosceles/scalene structure the angles already encode; and, critically,
    the biggest stated side always corresponds to the biggest stated angle
    (real triangle geometry), unlike drawing sides and angles independently
    at random, which can state a side/angle combination no triangle could
    have (found during development: sides [7, 6, 6] stated alongside angles
    [65, 65, 50] -- the unique side 7 is the LARGEST side but its angle 50 is
    the SMALLEST angle, backwards from every real triangle). Retries if
    rounding happens to collapse two genuinely different angles onto the
    same side length."""
    for attempt in range(200):
        # Escalate resolution on retry: two angles close together (e.g. 90 and
        # 76) need a bigger scale before their sin values round to different
        # integers.
        target_max = rng.randint(20, 30 + 20 * (attempt // 20))
        k = target_max / max(math.sin(math.radians(a)) for a in angles)
        sides = [max(3, round(k * math.sin(math.radians(a)))) for a in angles]
        # equal angles must give equal sides, and distinct angles distinct sides
        ok = True
        for i in range(3):
            for j in range(i + 1, 3):
                same_angle = angles[i] == angles[j]
                same_side = sides[i] == sides[j]
                if same_angle != same_side:
                    ok = False
        if ok:
            return sides
    raise RuntimeError(f"p14b_polygon_classification: could not derive consistent sides for angles {angles}")


def _draw_medium(rng):
    cls = rng.choice(MEDIUM_CLASSES)
    if cls == "isosceles right-angled":
        angles = [45, 45, 90]
    elif cls == "isosceles, not right-angled":
        # Exclude 60 alongside 45: t=60 gives apex 180-2*60=60, i.e. angles
        # [60, 60, 60] -- a genuinely equilateral triangle, still correctly
        # keyed "isosceles, not right-angled" (equilateral is a subset of
        # isosceles: this tier's side axis is only 2-valued, scalene vs
        # isosceles, never its own 3rd "equilateral" class), but excluded
        # anyway to keep every drawn triangle strictly 2-equal-sided at this
        # tier and the enumeration table's cells textbook-clean.
        t = rng.choice([v for v in range(10, 80) if v not in (45, 60)])
        apex = 180 - 2 * t
        angles = [t, t, apex]
    elif cls == "scalene right-angled":
        x = rng.choice([v for v in range(10, 80) if v != 45])
        angles = [90, x, 90 - x]
    else:  # scalene, not right-angled
        while True:
            x = rng.randint(20, 70)
            y = rng.randint(20, 70)
            z = 180 - x - y
            if z <= 0 or z >= 180 or z == 90 or len({x, y, z}) != 3:
                continue
            angles = [x, y, z]
            break
    sides = _sides_from_angles(rng, angles)
    order = list(range(3))
    rng.shuffle(order)
    angles = [angles[i] for i in order]
    sides = [sides[i] for i in order]
    return cls, sides, angles


_MEDIUM_TEMPLATES = [
    "A triangle has side lengths {sides} units, and angle measures {angles} degrees. "
    "Classify the triangle, giving the single most specific type that applies.",
    "A triangle's sides measure {sides} units and its angles measure {angles} degrees. "
    "What is the most specific classification for this triangle?",
    "The sides of a triangle are {sides} units, and the angles are {angles} degrees. "
    "Classify the triangle by both its sides and its angles, giving the most specific type.",
]


def _build_medium(rng):
    cls, sides, angles = _draw_medium(rng)
    check(sum(angles) == 180, f"p14b_polygon_classification: medium-tier angles do not sum to 180: {angles}")
    check(
        _classify_side_angle_independent(sides, angles) == cls,
        f"p14b_polygon_classification: medium-tier independent check disagrees ({sides}, {angles} -> "
        f"{_classify_side_angle_independent(sides, angles)}, expected {cls})",
    )

    text = phrase(rng, _MEDIUM_TEMPLATES, sides=_format_list3(sides), angles=_format_list3(angles))
    answer = cls
    distractors_list = [c for c in MEDIUM_CLASSES if c != cls]

    working = [
        f"Side lengths: {_format_list3(sides)}; angle measures: {_format_list3(angles)} degrees.",
        f"Equal sides present: {'yes' if (sides[0] == sides[1] or sides[1] == sides[2] or sides[0] == sides[2]) else 'no'}.",
        f"A right angle present: {'yes' if 90 in angles else 'no'}.",
        f"Most specific classification: {answer}.",
    ]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer,
        answer_latex=f"plain:{answer}",
        working=working,
        distractors=pick_distractors(answer, distractors_list, n=3),
    )


# -- Hard: quadrilateral, side-equalities AND parallel-pair count ------------


def _classify_quad_hard_independent(sides, parallel_pairs) -> str:
    """Independent re-derivation: the parallel-pair count decides trapezium
    outright (it is the one property a parallelogram/rhombus never has), and
    only then does the equal-side count decide rhombus vs parallelogram --
    never the branch construction used to pick the target class."""
    if parallel_pairs == 1:
        return "trapezium"
    if len(set(sides)) == 1:
        return "rhombus"
    return "parallelogram"


def _build_hard(rng):
    cls = rng.choice(HARD_CLASSES)
    if cls == "rhombus":
        s = rng.randint(_SIDE_LO, _SIDE_HI)
        sides = [s, s, s, s]
        parallel_pairs = 2
        sides_desc = f"all four sides are equal, each {s} units"
    elif cls == "parallelogram":
        a = rng.randint(_SIDE_LO, _SIDE_HI)
        b = rng.randint(_SIDE_LO, _SIDE_HI)
        while b == a:
            b = rng.randint(_SIDE_LO, _SIDE_HI)
        sides = [a, b, a, b]
        parallel_pairs = 2
        sides_desc = f"its opposite sides are equal in pairs: {a} units and {a} units, {b} units and {b} units"
    else:  # trapezium
        while True:
            sides = [rng.randint(_SIDE_LO, _SIDE_HI) for _ in range(4)]
            if len(set(sides)) >= 3:
                break
        parallel_pairs = 1
        sides_desc = f"its side lengths are {_format_list4(sides)} units, with no two pairs of opposite sides equal"

    check(
        _classify_quad_hard_independent(sides, parallel_pairs) == cls,
        f"p14b_polygon_classification: hard-tier independent check disagrees ({sides}, "
        f"parallel_pairs={parallel_pairs} -> {_classify_quad_hard_independent(sides, parallel_pairs)}, "
        f"expected {cls})",
    )

    parallel_clause = (
        "both pairs of opposite sides are parallel"
        if parallel_pairs == 2
        else "only one pair of opposite sides is parallel"
    )
    templates = [
        "A quadrilateral is such that {parallel}, and {sides_desc}. Classify the quadrilateral, "
        "giving its most specific type.",
        "Consider a quadrilateral where {parallel}, and {sides_desc}. What is the most specific "
        "classification for this quadrilateral?",
        "In a quadrilateral, {parallel}, and {sides_desc}. Classify it as specifically as the stated facts allow.",
    ]
    text = phrase(rng, templates, parallel=parallel_clause, sides_desc=sides_desc)
    answer = cls
    distractors_list = [c for c in HARD_CLASSES if c != cls]

    working = [
        f"Parallel sides: {parallel_clause}.",
        f"Side lengths: {sides}.",
        f"Classification: {answer}.",
    ]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer,
        answer_latex=f"plain:{answer}",
        working=working,
        distractors=pick_distractors(answer, distractors_list, n=3),
    )


# -- Exam: quadrilateral, side-equalities AND right-angle presence ----------
# (Both pairs of opposite sides are always stated parallel here, so
# parallelism is never the deciding fact -- the genuinely new step over hard
# is noticing a SECOND promoting fact (right angles) on top of the equal-
# sides fact hard already tests, catching an under-read that stops at the
# less-specific class.)


def _classify_quad_exam_independent(sides, all_right) -> str:
    eq = len(set(sides)) == 1
    if eq and all_right:
        return "square"
    if eq:
        return "rhombus"
    if all_right:
        return "rectangle"
    return "parallelogram"


def _build_exam(rng):
    cls = rng.choice(EXAM_CLASSES)
    all_sides_equal = cls in ("square", "rhombus")
    all_angles_right = cls in ("square", "rectangle")

    if all_sides_equal:
        s = rng.randint(_SIDE_LO, _SIDE_HI)
        sides = [s, s, s, s]
        sides_desc = f"all four sides are equal, each {s} units"
    else:
        a = rng.randint(_SIDE_LO, _SIDE_HI)
        b = rng.randint(_SIDE_LO, _SIDE_HI)
        while b == a:
            b = rng.randint(_SIDE_LO, _SIDE_HI)
        sides = [a, b, a, b]
        sides_desc = f"its opposite sides are equal in pairs: {a} units and {a} units, {b} units and {b} units"

    if all_angles_right:
        angle_desc = "all four interior angles are right angles (90 degrees each)"
    else:
        x = rng.choice([v for v in range(10, 170) if v != 90])
        angle_desc = f"one interior angle measures {x} degrees, so not every angle is a right angle"

    check(
        _classify_quad_exam_independent(sides, all_angles_right) == cls,
        f"p14b_polygon_classification: exam-tier independent check disagrees ({sides}, "
        f"all_right={all_angles_right} -> {_classify_quad_exam_independent(sides, all_angles_right)}, "
        f"expected {cls})",
    )

    templates = [
        "A quadrilateral is such that both pairs of opposite sides are parallel, {sides_desc}, and {angle_desc}. "
        "Classify the quadrilateral, giving its most specific type.",
        "Consider a quadrilateral where both pairs of opposite sides are parallel, {sides_desc}, and {angle_desc}. "
        "What is the most specific classification?",
        "In a quadrilateral whose opposite sides are always parallel, {sides_desc}, and {angle_desc}. "
        "Classify it as specifically as the stated facts allow.",
    ]
    text = phrase(rng, templates, sides_desc=sides_desc, angle_desc=angle_desc)
    answer = cls
    distractors_list = [c for c in EXAM_CLASSES if c != cls]

    working = [
        f"Side lengths: {sides}.",
        f"All sides equal: {'yes' if all_sides_equal else 'no'}.",
        f"All angles right angles: {'yes' if all_angles_right else 'no'}.",
        f"Most specific classification: {answer}.",
    ]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer,
        answer_text=answer,
        answer_latex=f"plain:{answer}",
        working=working,
        distractors=pick_distractors(answer, distractors_list, n=3),
    )


_BUILDERS = {
    "easy": _build_easy,
    "medium": _build_medium,
    "hard": _build_hard,
    "exam": _build_exam,
}


@generator(
    "p14b_polygon_classification",
    level=7,
    presets={
        "easy": dict(shape="easy"),
        "medium": dict(shape="medium"),
        "hard": dict(shape="hard"),
        "exam": dict(shape="exam"),
    },
)
def p14b_polygon_classification(rng, shape="easy"):
    """Classify a triangle or quadrilateral from its stated side lengths,
    angle measures and parallel-side relationships (see the module
    docstring for the full answer space, one tier at a time)."""
    builder = _BUILDERS.get(shape)
    if builder is None:
        raise ValueError(f"unknown shape {shape!r}; expected one of {sorted(_BUILDERS)}")
    return builder(rng)
