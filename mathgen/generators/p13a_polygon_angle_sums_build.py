"""Generates `p13a_polygon_angle_sum` questions for `f10-polygon-angle-sums`: given a stated polygon and all
interior angles except one, find the missing interior angle in degrees.

Four tiers, escalating by DECISION, never by a bigger polygon:

- easy (``shape="numeral_individual"``): ``n`` (3-6) stated as a numeral, its
  ``n-1`` known angles listed individually as plain numbers. Direct
  sum-then-subtract.
- medium (``shape="numeral_grouped"``): ``n`` (4-6, see below) still a
  numeral, but the known angles include a GROUPED restatement -- "g of the
  angles measure v degrees each" -- plus the remaining known angles listed
  individually. The learner must multiply (count x value) before subtracting,
  a genuine added step over easy.
- hard (``shape="name_individual"``): the polygon is named (triangle .. octagon,
  n 3-8) instead of given as a numeral, forcing a name-to-n lookup before the
  sum formula applies. Known angles listed individually (easy's
  representation) -- a different KIND of extra step (vocabulary lookup) from
  medium's (arithmetic grouping).
- exam (``shape="name_grouped"``): COMBINES both extra moves -- the polygon is
  named (hard's move) AND its known angles include a grouped restatement
  (medium's move) in the same question.

``medium``/``exam`` restrict ``n`` to 4-8 (not 3-8): a grouped restatement
needs at least one individually-listed angle left over once the group is
removed ("g of the angles measure v degrees each, and the rest measure ...")
-- for a triangle (n=3, only 2 known angles) there is no room for both a
group of >=2 AND a non-empty individual remainder, so the shape is not built
for n=3. This is a scope decision, not a defect: the "plus the rest listed
individually" wording in the medium/exam descriptions presupposes
a non-empty remainder.

Construct-to-target throughout: every known angle is drawn from a NARROWED, n-dependent plausibility band
centred on that polygon's own average interior angle (``(n-2)*180/n``), and the whole draw is RESAMPLED (never
clamped) whenever the computed missing angle falls outside a sensible convex-polygon range, or whenever a
candidate distractor would coincide with the answer. A fixed band such as 80-160 degrees for every ``n`` was
proven unusable for a triangle (average interior angle 60) during development: it makes almost every draw sum
past 180 degrees before the third angle is even placed.

The independent in-generator check (``_check_*`` functions below) re-derives
the answer from the RENDERED text via a regex parse -- never a replay of
construction's own running subtraction -- using its OWN independent
name -> n table and its own independent count * value expansion, deliberately
NOT shared with construction's ``_NAME_TO_N``/grouping code (a genuinely
different route, not the same code called twice). The standalone oracle test
file repeats this
independence one level further out: it never imports this module at all.
"""

from __future__ import annotations

import re

from ..core import (
    check,
    check_equivalent,
    check_pairwise_distinct,
    generator,
    pick_distractors,
)

# -- Shared vocabulary (construction side) ----------------------------------

_NAME_TO_N = {
    "triangle": 3,
    "quadrilateral": 4,
    "pentagon": 5,
    "hexagon": 6,
    "heptagon": 7,
    "octagon": 8,
}
_N_TO_NAME = {n: name for name, n in _NAME_TO_N.items()}

_MISSING_LO, _MISSING_HI = 15, 170
_RETRY_LIMIT = 400


def _known_angle_band(n: int) -> tuple[int, int]:
    """An n-dependent plausibility band for a KNOWN interior angle, centred on the regular n-gon's own average
    interior angle ((n-2)*180/n), +/- 45 degrees, clipped to a sensible convex range."""
    avg = ((n - 2) * 180) / n
    lo = max(10, round(avg - 45))
    hi = min(170, round(avg + 45))
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _draw_individual(rng, n: int):
    """n-1 known angles, listed individually. Resample the WHOLE draw (never
    clamp) until the computed missing angle lands in the sensible convex
    range."""
    lo, hi = _known_angle_band(n)
    for _ in range(_RETRY_LIMIT):
        knowns = [rng.randint(lo, hi) for _ in range(n - 1)]
        missing = (n - 2) * 180 - sum(knowns)
        if _MISSING_LO <= missing <= _MISSING_HI:
            return knowns, missing
    raise RuntimeError(f"p13a_polygon_angle_sum: could not draw a plausible individual set for n={n}")


def _draw_grouped(rng, n: int):
    """A grouped restatement: g angles (2 <= g <= n-2) each measuring v
    degrees, plus the remaining (n-1-g) known angles listed individually
    (always >= 1 by construction, see the module docstring). Resamples the
    whole draw until the computed missing angle lands in the sensible convex
    range."""
    lo, hi = _known_angle_band(n)
    max_g = n - 2  # leaves at least one individually-listed angle (n-1-g >= 1)
    for _ in range(_RETRY_LIMIT):
        g = rng.randint(2, max_g)
        v = rng.randint(lo, hi)
        rest_count = n - 1 - g
        rest = [rng.randint(lo, hi) for _ in range(rest_count)]
        missing = (n - 2) * 180 - g * v - sum(rest)
        if _MISSING_LO <= missing <= _MISSING_HI:
            return g, v, rest, missing
    raise RuntimeError(f"p13a_polygon_angle_sum: could not draw a plausible grouped set for n={n}")


def _format_list(values: list[int]) -> str:
    """ "80, 95 and 105 degrees" / "80 and 95 degrees" / "80 degrees"."""
    if len(values) == 1:
        return f"{values[0]} degrees"
    return ", ".join(str(v) for v in values[:-1]) + f" and {values[-1]} degrees"


def _article(name: str) -> str:
    """ "A hexagon" / "An octagon" -- the only vowel-initial name in the
    six-name pool is "octagon"."""
    return "An" if name[0] in "aeiou" else "A"


# -- Independent in-generator checks (own name->n table, own expansion) -----

_CHECK_NAME_TO_N = {
    "triangle": 3,
    "quadrilateral": 4,
    "pentagon": 5,
    "hexagon": 6,
    "heptagon": 7,
    "octagon": 8,
}

_EASY_RE = re.compile(r"A polygon has (\d+) sides\. (\d+) of its interior angles measure (.+?) degrees\.")
_MEDIUM_RE = re.compile(
    r"A polygon has (\d+) sides\. (\d+) of the angles measure (\d+) degrees each, "
    r"and the rest measure (.+?) degrees\."
)
_HARD_RE = re.compile(r"An? (\w+) has (.+?) known interior angles: (.+?) degrees, with one angle still unknown\.")
_EXAM_RE = re.compile(r"An? (\w+) has (\d+) angles measuring (\d+) degrees each and the rest measuring (.+?)\.")


def _check_easy(text: str, answer_text: str):
    m = _EASY_RE.search(text)
    check(m is not None, "p13a_polygon_angle_sum: could not parse easy-tier text")
    n, stated_count = int(m.group(1)), int(m.group(2))
    knowns = [int(x) for x in re.findall(r"\d+", m.group(3))]
    check(
        stated_count == len(knowns) == n - 1,
        "p13a_polygon_angle_sum: easy-tier stated count does not match the listed angles "
        f"(stated {stated_count}, listed {len(knowns)}, n-1 {n - 1})",
    )
    expected = (n - 2) * 180 - sum(knowns)
    check_equivalent(expected, int(answer_text), "p13a_polygon_angle_sum: easy-tier check disagrees with answer_text")


def _check_medium(text: str, answer_text: str):
    m = _MEDIUM_RE.search(text)
    check(m is not None, "p13a_polygon_angle_sum: could not parse medium-tier text")
    n, g, v = int(m.group(1)), int(m.group(2)), int(m.group(3))
    rest = [int(x) for x in re.findall(r"\d+", m.group(4))]
    check(
        len(rest) == n - 1 - g,
        "p13a_polygon_angle_sum: medium-tier rest count does not match n-1-g "
        f"(listed {len(rest)}, expected {n - 1 - g})",
    )
    expected = (n - 2) * 180 - g * v - sum(rest)
    check_equivalent(expected, int(answer_text), "p13a_polygon_angle_sum: medium-tier check disagrees with answer_text")


def _check_hard(text: str, answer_text: str):
    m = _HARD_RE.search(text)
    check(m is not None, "p13a_polygon_angle_sum: could not parse hard-tier text")
    name, stated_count = m.group(1), int(m.group(2))
    check(name in _CHECK_NAME_TO_N, f"p13a_polygon_angle_sum: unknown polygon name {name!r}")
    n = _CHECK_NAME_TO_N[name]
    knowns = [int(x) for x in re.findall(r"\d+", m.group(3))]
    check(
        stated_count == len(knowns) == n - 1,
        "p13a_polygon_angle_sum: hard-tier stated count does not match the listed angles "
        f"(stated {stated_count}, listed {len(knowns)}, n-1 {n - 1})",
    )
    expected = (n - 2) * 180 - sum(knowns)
    check_equivalent(expected, int(answer_text), "p13a_polygon_angle_sum: hard-tier check disagrees with answer_text")


def _check_exam(text: str, answer_text: str):
    m = _EXAM_RE.search(text)
    check(m is not None, "p13a_polygon_angle_sum: could not parse exam-tier text")
    name = m.group(1)
    check(name in _CHECK_NAME_TO_N, f"p13a_polygon_angle_sum: unknown polygon name {name!r}")
    n = _CHECK_NAME_TO_N[name]
    g, v = int(m.group(2)), int(m.group(3))
    rest = [int(x) for x in re.findall(r"\d+", m.group(4))]
    check(
        len(rest) == n - 1 - g,
        f"p13a_polygon_angle_sum: exam-tier rest count does not match n-1-g (listed {len(rest)}, expected {n - 1 - g})",
    )
    expected = (n - 2) * 180 - g * v - sum(rest)
    check_equivalent(expected, int(answer_text), "p13a_polygon_angle_sum: exam-tier check disagrees with answer_text")


# -- Distractors --------------------------------------------------------------


def _wrong_n_adjacent(n: int) -> int:
    """(Name-lookup tiers) an adjacent wrong side count: n-1 if that is still
    a valid polygon (>= 3), else n+1 -- always constructible, unlike a fixed
    "always n-1" rule which degenerates for a triangle (n=3 -> n=2, not a
    polygon)."""
    return n - 1 if n - 1 >= 3 else n + 1


_ANGLE_RANGE_LO, _ANGLE_RANGE_HI = 1, 179


_OFFSET_MAGNITUDES = (5, 7, 9, 11, 13, 15, 17)


def _bounded_offsets(rng, missing: int) -> list[int]:
    """Small, generic wrong-answer offsets that stay inside a physically
    sensible interior-angle range (1-179 degrees), unlike the misconception
    candidates below: ``n*180 - known_sum`` is always ``missing + 360``, an
    off-by-one side count is always ``missing - 180`` or ``missing + 180``,
    and "omits one known angle" adds back a whole angle value -- all three
    routinely land outside [1, 179] and would be eliminable by a learner on
    SHAPE alone (a negative or 300+ "degrees" answer), never by doing the
    maths.

    The magnitude AND sign are drawn from ``rng`` (called after `text`/ `answer_text` are already fixed, so
    this cannot perturb the question or answer, only which wrong options accompany it) rather than a fixed
    ``[-11, +11, -7, +7]`` sequence: a fixed sequence made the answer land at the same RANK (3rd of 4, sorted)
    on every seed where no misconception value survived the range filter -- itself a shape a learner could
    read off without any arithmetic. Returns up to 3 values, deduplicated and distinct from `missing`."""
    magnitudes = list(_OFFSET_MAGNITUDES)
    rng.shuffle(magnitudes)
    out = []
    for mag in magnitudes:
        sign = rng.choice((-1, 1))
        v = missing + sign * mag
        if v == missing or v in out:
            continue
        if _ANGLE_RANGE_LO <= v <= _ANGLE_RANGE_HI:
            out.append(v)
        if len(out) >= 3:
            break
    return out


def _order_in_range_first(missing: int, candidates: list[int]) -> list[int]:
    """Reorder survivor candidates so every value inside the sensible
    interior-angle range [1, 179] is offered to `pick_distractors` before any
    value outside it, preserving each bucket's own relative order -- so a
    rendered question's wrong options are, wherever enough in-range
    candidates exist, never eliminable by shape (a negative or 300+ degree
    "angle") alone."""
    in_range = [c for c in candidates if _ANGLE_RANGE_LO <= c <= _ANGLE_RANGE_HI]
    out_range = [c for c in candidates if not (_ANGLE_RANGE_LO <= c <= _ANGLE_RANGE_HI)]
    return in_range + out_range


_CLOSER = "What is the size of the missing interior angle, in degrees?"


# -- Tier builders ------------------------------------------------------------


def _build_numeral_individual(rng):
    n = rng.randint(3, 6)
    knowns, missing = _draw_individual(rng, n)

    text = f"A polygon has {n} sides. {n - 1} of its interior angles measure {_format_list(knowns)}. {_CLOSER}"
    answer_text = str(missing)
    _check_easy(text, answer_text)

    d_wrong_formula = n * 180 - sum(knowns)  # forgets the "-2"
    d_omit_one = missing + knowns[0]  # omits the first listed angle from the subtraction

    survivors = []
    for c in (d_wrong_formula, d_omit_one, *_bounded_offsets(rng, missing)):
        if c == missing or c in survivors:
            continue
        survivors.append(c)
    check_pairwise_distinct(missing, *survivors, msg="p13a_polygon_angle_sum: easy-tier distractor collision")

    working = [
        f"Interior angle sum = ({n} - 2) x 180 = {(n - 2) * 180} degrees.",
        f"Known angles sum to {sum(knowns)} degrees.",
        f"Missing angle = {(n - 2) * 180} - {sum(knowns)} = {missing} degrees.",
    ]
    ordered = _order_in_range_first(missing, survivors)
    return dict(
        text=text,
        answer=missing,
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, [str(c) for c in ordered]),
    )


def _build_numeral_grouped(rng):
    n = rng.randint(4, 6)
    g, v, rest, missing = _draw_grouped(rng, n)

    text = (
        f"A polygon has {n} sides. {g} of the angles measure {v} degrees each, "
        f"and the rest measure {_format_list(rest)}. One more angle is unknown. {_CLOSER}"
    )
    answer_text = str(missing)
    _check_medium(text, answer_text)

    known_sum = g * v + sum(rest)
    d_wrong_formula = n * 180 - known_sum  # forgets the "-2"
    d_group_once = (n - 2) * 180 - v - sum(rest)  # adds the grouped angle only once, not g times
    d_omit_one = missing + rest[0]  # omits one individually-listed (rest) angle

    survivors = []
    for c in (d_wrong_formula, d_group_once, d_omit_one, *_bounded_offsets(rng, missing)):
        if c == missing or c in survivors:
            continue
        survivors.append(c)
    check_pairwise_distinct(missing, *survivors, msg="p13a_polygon_angle_sum: medium-tier distractor collision")

    working = [
        f"Interior angle sum = ({n} - 2) x 180 = {(n - 2) * 180} degrees.",
        f"Grouped angles: {g} x {v} = {g * v} degrees.",
        f"All known angles sum to {known_sum} degrees.",
        f"Missing angle = {(n - 2) * 180} - {known_sum} = {missing} degrees.",
    ]
    ordered = _order_in_range_first(missing, survivors)
    return dict(
        text=text,
        answer=missing,
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, [str(c) for c in ordered]),
    )


def _build_name_individual(rng):
    name = rng.choice(list(_NAME_TO_N))
    n = _NAME_TO_N[name]
    knowns, missing = _draw_individual(rng, n)

    text = (
        f"{_article(name)} {name} has {n - 1} known interior angles: {_format_list(knowns)}, "
        f"with one angle still unknown. {_CLOSER}"
    )
    answer_text = str(missing)
    _check_hard(text, answer_text)

    wrong_n = _wrong_n_adjacent(n)
    d_wrong_formula = n * 180 - sum(knowns)  # forgets the "-2"
    d_wrong_n = (wrong_n - 2) * 180 - sum(knowns)  # off-by-one side count
    d_omit_one = missing + knowns[0]  # omits the first listed angle

    survivors = []
    for c in (d_wrong_formula, d_wrong_n, d_omit_one, *_bounded_offsets(rng, missing)):
        if c == missing or c in survivors:
            continue
        survivors.append(c)
    check_pairwise_distinct(missing, *survivors, msg="p13a_polygon_angle_sum: hard-tier distractor collision")

    working = [
        f"{_article(name)} {name} has {n} sides, so the interior angle sum = ({n} - 2) × 180 = {(n - 2) * 180} degrees.",
        f"Known angles sum to {sum(knowns)} degrees.",
        f"Missing angle = {(n - 2) * 180} - {sum(knowns)} = {missing} degrees.",
    ]
    ordered = _order_in_range_first(missing, survivors)
    return dict(
        text=text,
        answer=missing,
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, [str(c) for c in ordered]),
    )


def _build_name_grouped(rng):
    name = rng.choice([nm for nm, n in _NAME_TO_N.items() if n >= 4])
    n = _NAME_TO_N[name]
    g, v, rest, missing = _draw_grouped(rng, n)

    text = (
        f"{_article(name)} {name} has {g} angles measuring {v} degrees each and the rest measuring "
        f"{_format_list(rest)}. {_CLOSER}"
    )
    answer_text = str(missing)
    _check_exam(text, answer_text)

    known_sum = g * v + sum(rest)
    wrong_n = _wrong_n_adjacent(n)
    d_wrong_formula = n * 180 - known_sum  # forgets the "-2"
    d_group_once = (n - 2) * 180 - v - sum(rest)  # adds the grouped angle only once
    d_wrong_n = (wrong_n - 2) * 180 - known_sum  # off-by-one side count
    d_omit_one = missing + rest[0]  # omits one individually-listed (rest) angle

    survivors = []
    for c in (d_wrong_formula, d_group_once, d_wrong_n, d_omit_one, *_bounded_offsets(rng, missing)):
        if c == missing or c in survivors:
            continue
        survivors.append(c)
    check_pairwise_distinct(missing, *survivors, msg="p13a_polygon_angle_sum: exam-tier distractor collision")

    working = [
        f"{_article(name)} {name} has {n} sides, so the interior angle sum = ({n} - 2) × 180 = {(n - 2) * 180} degrees.",
        f"Grouped angles: {g} x {v} = {g * v} degrees.",
        f"All known angles sum to {known_sum} degrees.",
        f"Missing angle = {(n - 2) * 180} - {known_sum} = {missing} degrees.",
    ]
    ordered = _order_in_range_first(missing, survivors)
    return dict(
        text=text,
        answer=missing,
        answer_text=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, [str(c) for c in ordered]),
    )


_BUILDERS = {
    "numeral_individual": _build_numeral_individual,
    "numeral_grouped": _build_numeral_grouped,
    "name_individual": _build_name_individual,
    "name_grouped": _build_name_grouped,
}


@generator(
    "p13a_polygon_angle_sum",
    level=7,
    presets={
        "easy": dict(shape="numeral_individual"),
        "medium": dict(shape="numeral_grouped"),
        "hard": dict(shape="name_individual"),
        "exam": dict(shape="name_grouped"),
    },
)
def p13a_polygon_angle_sum(rng, shape="numeral_individual"):
    """Given a stated polygon (numeral or common name) and all interior
    angles except one (listed individually, or with a grouped restatement),
    determine the missing interior angle in degrees. `shape` selects one of
    the four genuinely different question forms (see module docstring)."""
    builder = _BUILDERS.get(shape)
    if builder is None:
        raise ValueError(f"unknown shape {shape!r}; expected one of {sorted(_BUILDERS)}")
    return builder(rng)
