"""Builds the `f10-enlargement-invariants` skill: what an enlargement changes (lengths, area, perimeter, the
distance from the centre of enlargement) and what it keeps (angles).

Topic: ``enlargement_invariants`` (``answer: number``). Every question states the centre of enlargement, the
scale factor and the measurements of the original; the answer is a single number the learner can work out from
the stem alone, so no drawing is needed. Whole-number answers are written as integers; the one decimal kind says
"Give your answer to 1 decimal place" and its answer and every distractor carry exactly one decimal place, so each
number is graded to its own stated precision.

Three tiers, escalating in reasoning rather than size:

- easy (whole-number scale factor 2 to 5): the image of a stated length (multiply), or the image of a stated angle
  (unchanged, with a side length in the stem as a lure).
- medium (whole-number scale factor 2 to 6): the area or perimeter of an enlarged rectangle, the number the area is
  multiplied by (the scale factor squared), or the original length from the image length (divide).
- hard: the scale factor has to be found first (from an area ratio), or the quantity is the gap between a point and
  its image, an area under a reduction (scale factor 1/2, 1/3 or 1/4), or a length under a decimal scale factor.

Each answer is re-derived by enlarging actual coordinates about the centre with exact fractions.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import check, generator, numeric_answer, pick_distractors, round_half_up, stated_precision_string

ADDS = "adds-instead-of-multiplies"
ANGLE_SCALED = "angle-scaled-with-lengths"
COPIES_LENGTH = "copies-a-stated-length"
AREA_BY_K = "area-scaled-by-k-not-k-squared"
PERIMETER_FOR_AREA = "finds-perimeter-instead-of-area"
AREA_FOR_PERIMETER = "finds-area-instead-of-perimeter"
PERIMETER_BY_K2 = "perimeter-scaled-by-k-squared"
DOUBLES = "doubles-instead-of-squares"
VOLUME_FACTOR = "uses-k-cubed-for-area"
MULTIPLIES_BACK = "multiplies-instead-of-divides"
SUBTRACTS_BACK = "subtracts-instead-of-divides"
LENGTH_BY_K2 = "length-scaled-by-k-squared"
AREA_RATIO_AS_K = "uses-area-ratio-as-scale-factor"
FACTOR_ONLY = "reports-scale-factor-not-length"
DISTANCE_FROM_CENTRE = "gives-distance-from-centre-not-gap"
OPPOSITE_SIDE = "puts-image-on-opposite-side-of-centre"
REDUCTION_AS_ENLARGEMENT = "treats-reduction-as-enlargement"
ROUNDS_FACTOR = "rounds-the-scale-factor"

_SHAPES = ["triangle", "kite", "parallelogram", "trapezium", "pentagon", "hexagon"]
_FRAMES = [
    "A {shape} is enlarged from a centre of enlargement O with a scale factor of {k}.",
    "A scale factor of {k} is used to enlarge a {shape} from a centre of enlargement O.",
    "The image of a {shape} is drawn by enlarging it from a centre of enlargement O with a scale factor of {k}.",
]


def _frame(rng, shape, k):
    return rng.choice(_FRAMES).format(shape=shape, k=k)


# -- exact enlargement of coordinates (the independent route) --------------------------------------------


def _enlarge(points, centre, k):
    cx, cy = centre
    return [(cx + k * (x - cx), cy + k * (y - cy)) for x, y in points]


def _rect(l, w, corner=(3, 2)):
    x, y = corner
    return [(x, y), (x + l, y), (x + l, y + w), (x, y + w)]


def _shoelace(points):
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]))) / 2


def _axis_perimeter(points):
    return sum(abs(x1 - x2) + abs(y1 - y2) for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]))


def _text(value, places):
    return stated_precision_string(Fraction(value), places) if places else str(int(value))


def _finish(rng, text, value, places, candidates, working):
    """candidates: [(Fraction value, misconception id)] in priority order. The answer's and every distractor's
    text carry the same number of decimal places."""
    answer_text = _text(value, places)
    seen, picked = {Fraction(value)}, []
    for cand, note in candidates:
        cand = Fraction(cand)
        if cand in seen or cand <= 0:
            continue
        if places == 0 and cand.denominator != 1:
            continue
        if places and (cand * 10**places).denominator != 1:
            continue
        seen.add(cand)
        picked.append((_text(cand, places), note))
    keep = pick_distractors(answer_text, [t for t, _ in picked], n=3)
    check(len(keep) == 3, f"fewer than three distractors: {text} {candidates}")
    notes = {t: n for t, n in picked if t in keep}
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=numeric_answer(Fraction(value)),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=keep,
        option_notes=notes,
    )


# -- tiers -------------------------------------------------------------------------------------------------


@generator(
    "enlargement_invariants",
    level=21,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def enlargement_invariants(rng, tier="easy"):
    """Apply a stated scale factor from a centre of enlargement and report the image's length, angle, area,
    perimeter or distance from the centre."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    return _hard(rng)


def _easy(rng):
    k = rng.randint(2, 5)
    shape = rng.choice(_SHAPES)
    frame = _frame(rng, shape, k)
    if rng.random() < 0.5:
        length = rng.randint(3, 19)
        centre, pts = (0, 0), [(1, 1), (1 + length, 1)]
        image = _enlarge(pts, centre, k)
        answer = abs(image[1][0] - image[0][0])
        check(answer == k * length, "image length disagrees with k times the length")
        text = (
            f"{frame} One side of the original is {length} cm long. How long, in cm, is the matching side of the image?"
        )
        cands = [
            (length + k, ADDS),
            (length * k * k, LENGTH_BY_K2),
            (k, FACTOR_ONLY),
            (length * (k + 1), ADDS),
            (length * k + k, ADDS),
        ]
        working = [f"Every length is multiplied by the scale factor: {length} x {k} = {answer} cm."]
        return _finish(rng, text, answer, 0, cands, working)
    angle = rng.randint(4, 17) * 5
    side = rng.randint(3, 19)
    check(0 < angle < 180, "angle out of range")
    text = (
        f"{frame} One angle of the original is {angle}°, and one of its sides is {side} cm long. "
        f"What size, in degrees, is the matching angle in the image?"
    )
    cands = [(angle * k, ANGLE_SCALED), (angle + k, ADDS), (side, COPIES_LENGTH), (side * k, COPIES_LENGTH)]
    working = [f"An enlargement keeps every angle, so the matching angle is still {angle}°."]
    return _finish(rng, text, angle, 0, cands, working)


def _medium(rng):
    k = rng.randint(2, 6)
    kind = rng.choice(["area", "perimeter", "area_factor", "original_length", "area_from_area"])
    shape = "rectangle" if kind in ("area", "perimeter") else rng.choice(_SHAPES)
    frame = _frame(rng, shape, k)
    l, w = rng.randint(2, 9), rng.randint(2, 9)
    while w == l:
        w = rng.randint(2, 9)
    if kind in ("area", "perimeter"):
        image = _enlarge(_rect(l, w), (0, 0), Fraction(k))
        if kind == "area":
            answer = _shoelace(image)
            check(answer == k * k * l * w, "image area disagrees with k squared times the area")
            text = f"{frame} The original is {l} cm by {w} cm. What is the area of the image, in square centimetres?"
            cands = [
                (k * l * w, AREA_BY_K),
                (k * 2 * (l + w), PERIMETER_FOR_AREA),
                (l * w + k, ADDS),
                (k * k * (l + w), AREA_BY_K),
                (2 * k * l * w, DOUBLES),
                (k * (l + w), PERIMETER_FOR_AREA),
                (k * k + l * w, ADDS),
            ]
            working = [f"Original area {l} x {w} = {l * w}.", f"Area is multiplied by {k} x {k} = {k * k}: {answer}."]
        else:
            answer = _axis_perimeter(image)
            check(answer == k * 2 * (l + w), "image perimeter disagrees with k times the perimeter")
            text = f"{frame} The original is {l} cm by {w} cm. What is the perimeter of the image, in cm?"
            cands = [
                (k * k * 2 * (l + w), PERIMETER_BY_K2),
                (k * k * l * w, AREA_FOR_PERIMETER),
                (2 * (l + w) + k, ADDS),
                (k * (l + w), ADDS),
                (2 * (l + w) + 4 * k, ADDS),
            ]
            working = [
                f"Original perimeter 2 × ({l} + {w}) = {2 * (l + w)}.",
                f"Perimeter is multiplied by {k}: {answer}.",
            ]
        return _finish(rng, text, answer, 0, cands, working)
    if kind == "area_factor":
        text = (
            f"{frame} The area of the image is how many times the area of the original? "
            f"Give the number the area is multiplied by."
        )
        answer = k * k
        cands = [(k, AREA_BY_K), (2 * k, DOUBLES), (k**3, VOLUME_FACTOR), (k * k + k, ADDS), (k + 2, ADDS)]
        return _finish(
            rng, text, answer, 0, cands, [f"Area is multiplied by the scale factor squared: {k} x {k} = {answer}."]
        )
    if kind == "area_from_area":
        area = rng.randint(3, 40)
        image = _enlarge(_rect(area, 1), (0, 0), Fraction(k))
        answer = _shoelace(image)
        check(answer == k * k * area, "image area disagrees with k squared times the area")
        text = f"{frame} The original has an area of {area} cm². What is the area of the image, in square centimetres?"
        cands = [(k * area, AREA_BY_K), (area + k * k, ADDS), (area * k * k * k, VOLUME_FACTOR), (area + k, ADDS)]
        return _finish(rng, text, answer, 0, cands, [f"{area} x {k} x {k} = {answer}."])
    # original_length: the image length is given
    original = rng.randint(2, 15)
    image_len = original * k
    answer = original
    text = f"{frame} A side of the image is {image_len} cm long. How long, in cm, is the matching side of the original?"
    cands = [
        (image_len * k, MULTIPLIES_BACK),
        (image_len - k, SUBTRACTS_BACK),
        (image_len + k, SUBTRACTS_BACK),
        (image_len * k * k, MULTIPLIES_BACK),
        (image_len - 2 * k, SUBTRACTS_BACK),
    ]
    return _finish(
        rng, text, answer, 0, cands, [f"Divide the image length by the scale factor: {image_len} / {k} = {answer} cm."]
    )


def _hard(rng):
    kind = rng.choice(["factor_from_areas", "gap_to_image", "reduction_area", "decimal_factor"])
    shape = rng.choice(_SHAPES)
    if kind == "factor_from_areas":
        k = rng.randint(2, 6)
        area = rng.randint(2, 12)
        side = rng.randint(2, 12)
        image_area = area * k * k
        # independent route: enlarge a w x l rectangle of that area by the factor found from the areas
        ratio = Fraction(image_area, area)
        found = next(f for f in range(1, 40) if f * f == ratio)
        check(found == k, "scale factor from the area ratio disagrees")
        answer = k * side
        text = (
            f"A {shape} is enlarged from a centre of enlargement O. The original has an area of {area} cm² and the "
            f"image has an area of {image_area} cm². One side of the original is {side} cm long. "
            f"How long, in cm, is the matching side of the image?"
        )
        cands = [
            (side * k * k, AREA_RATIO_AS_K),
            (k, FACTOR_ONLY),
            (side + k, ADDS),
            (side * (k * k - 1), AREA_RATIO_AS_K),
        ]
        working = [
            f"Area ratio {image_area} / {area} = {k * k}, so the scale factor is {k}.",
            f"Length of the image side: {side} x {k} = {answer} cm.",
        ]
        return _finish(rng, text, answer, 0, cands, working)
    if kind == "gap_to_image":
        k = rng.randint(2, 5)
        d = rng.randint(2, 12)
        image_point = _enlarge([(d, 0)], (0, 0), Fraction(k))[0]
        answer = image_point[0] - d
        check(answer == (k - 1) * d, "gap disagrees with (k - 1) times the distance")
        text = (
            f"A point P is {d} cm from the centre of enlargement O. P is enlarged with a scale factor of {k}, and its "
            f"image P' lies on the same side of O as P, on the line through O and P. "
            f"How far apart, in cm, are P and P'?"
        )
        cands = [
            (k * d, DISTANCE_FROM_CENTRE),
            ((k + 1) * d, OPPOSITE_SIDE),
            (d + k, ADDS),
            (k * d - k, ADDS),
            (k * k * d, LENGTH_BY_K2),
            (k * d + k, ADDS),
            (d * (k - 1) + k, ADDS),
        ]
        working = [
            f"P' is {k} x {d} = {k * d} cm from O.",
            f"P is {d} cm from O on the same side, so the gap is {k * d} - {d} = {answer} cm.",
        ]
        return _finish(rng, text, answer, 0, cands, working)
    if kind == "reduction_area":
        m = rng.randint(2, 4)
        area = m * m * rng.randint(2, 12)
        image = _enlarge(_rect(area, 1), (0, 0), Fraction(1, m))
        answer = _shoelace(image)
        check(answer == Fraction(area, m * m), "reduced area disagrees")
        text = (
            f"A {shape} with an area of {area} cm² is enlarged from a centre of enlargement O with a scale factor of 1/{m}. "
            f"What is the area of the image, in square centimetres?"
        )
        cands = [
            (Fraction(area, m), AREA_BY_K),
            (area * m * m, REDUCTION_AS_ENLARGEMENT),
            (area - m, ADDS),
            (Fraction(area, m * m * m), VOLUME_FACTOR),
        ]
        working = [f"Area is multiplied by (1/{m})^2 = 1/{m * m}: {area} / {m * m} = {answer}."]
        return _finish(rng, text, answer, 0, cands, working)
    # decimal_factor: length under a decimal scale factor, answer to 1 decimal place
    k = rng.choice([Fraction(3, 2), Fraction(5, 2), Fraction(7, 2)])
    length = rng.choice([5, 7, 9, 11, 13])
    image = _enlarge([(0, 0), (length, 0)], (0, 0), k)
    answer = image[1][0] - image[0][0]
    check(answer == k * length and answer.denominator == 2, "decimal image length disagrees")
    k_text = _text(k, 1)
    text = (
        f"A {shape} is enlarged from a centre of enlargement O with a scale factor of {k_text}. One side of the original "
        f"is {length} cm long. How long, in cm, is the matching side of the image? Give your answer to 1 decimal place."
    )
    cands = [
        (length + k, ADDS),
        (round_half_up(k) * length, ROUNDS_FACTOR),
        (Fraction(length) * k * k, LENGTH_BY_K2),
        (length * (k + 1), ADDS),
    ]
    working = [f"{length} x {k_text} = {_text(answer, 1)} cm."]
    return _finish(rng, text, answer, 1, cands, working)
