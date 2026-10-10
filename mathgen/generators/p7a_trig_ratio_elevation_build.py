"""Senior trig-ratio review (with tangent) and angle of elevation/depression.

- ``trig_ratio_review_tan`` (``senior-review-sine-cosine-and-tangent-as-ratios-of``,
  ACMMM028, answer:
  fraction) -- states a named ratio (sine, cosine OR tangent) as an exact
  fraction of given/derived side lengths in a right-angled triangle, or
  derives a second named ratio from one stated ratio at the same angle. This
  is a *different operation* from the sibling
  ``senior-apply-the-cosine-and-sine-ratios-to-find`` (ACMEM119,
  ``mathgen/generators/p5b_trig_ratio_build.py``), which solves an applied
  word problem for an unknown side or angle and deliberately excludes
  tangent. Tangent is made load-bearing here by construction: it is the
  asked-for or given ratio in 100% of hard-tier questions and at least 1/3
  of easy/medium-tier questions (measured in the test file, not
  merely assumed).
- ``elevation_depression_problem`` (``senior-work-with-the-concepts-of-angle-of-elevation``,
  ACMEM118, domain senior-mathematics, answer: number) -- angle of elevation
  (easy), angle of depression with the alternate-angle correction (medium),
  and a two-observation-point surveying problem (hard).

Both topics use an independent-route in-generator check (``mathgen.core.check``)
distinct from the construction route, and both ship a companion oracle test
 that never imports this
module.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import check, check_distinct, generator, phrase, pick_distractors, rational_text, VerificationError


# ---------------------------------------------------------------------------
# Skill A: trig_ratio_review_tan
# ---------------------------------------------------------------------------

# A right-triangle side-length pool independent of p5b_trig_ratio_build.py's
# table: six primitive triples not present there, plus two non-primitive
# scalings of two of them (18-80-82 = 2x(9-40-41), 24-70-74 = 2x(12-35-37))
# so "in lowest terms" is genuinely load-bearing on some easy/medium draws.
TRIPLES = [
    (9, 40, 41),
    (12, 35, 37),
    (28, 45, 53),
    (33, 56, 65),
    (16, 63, 65),
    (48, 55, 73),
    (18, 80, 82),
    (24, 70, 74),
]

_RATIO_FORMULA = {
    "sin": ("opposite", "hypotenuse"),
    "cos": ("adjacent", "hypotenuse"),
    "tan": ("opposite", "adjacent"),
}


def _pick_triangle(rng) -> tuple[int, int, int]:
    """One triple from TRIPLES, with the two legs randomly assigned to the
    opposite/adjacent roles (i.e. theta is at one acute vertex or the other)."""
    a, b, hyp = rng.choice(TRIPLES)
    if rng.choice((True, False)):
        return a, b, hyp
    return b, a, hyp


def _ratio_value(name: str, opp: int, adj: int, hyp: int) -> Fraction:
    if name == "sin":
        return Fraction(opp, hyp)
    if name == "cos":
        return Fraction(adj, hyp)
    return Fraction(opp, adj)


def _side_value(side: str, opp: int, adj: int, hyp: int) -> int:
    return {"opposite": opp, "adjacent": adj, "hypotenuse": hyp}[side]


def _check_ratio(name: str, value: Fraction, opp: int, adj: int, hyp: int) -> None:
    """Independent check for the easy/medium tiers.

    Route 1: opp/adj/hyp genuinely form a right triangle (integer Pythagoras,
    never reused by the ratio arithmetic itself).
    Route 2: sin and cos of the SAME angle satisfy the Pythagorean identity
    sin^2 + cos^2 = 1 -- a cross-check that only holds if opp, adj and hyp
    are mutually consistent, independent of which one ``value`` was asked
    for.
    """
    check(opp * opp + adj * adj == hyp * hyp, f"_check_ratio: {opp},{adj},{hyp} is not a right triangle")
    sin_v = Fraction(opp, hyp)
    cos_v = Fraction(adj, hyp)
    check(sin_v * sin_v + cos_v * cos_v == 1, f"_check_ratio: Pythagorean identity fails for sin={sin_v}, cos={cos_v}")
    if name == "sin":
        check(value == sin_v, f"_check_ratio: sin mismatch, stated {value} vs {sin_v}")
    elif name == "cos":
        check(value == cos_v, f"_check_ratio: cos mismatch, stated {value} vs {cos_v}")
    else:
        check(cos_v != 0, "_check_ratio: cos is zero")
        check(value == sin_v / cos_v, f"_check_ratio: tan mismatch, stated {value} vs {sin_v / cos_v}")


def _reconstruct_from_ratio(name: str, value: Fraction) -> tuple[int, int, int]:
    """Reconstruct integer (opp, adj, hyp) from a ratio's reduced fraction
    ``value``, assuming ``value`` is already in lowest terms (guaranteed:
    every triple in TRIPLES is pairwise coprime -- either primitive, or,
    for the two scaled triples, the ratio itself still reduces to the
    primitive triple's fraction, so ``value.numerator``/``value.denominator``
    are always the primitive pair)."""
    p, q = value.numerator, value.denominator
    if name == "sin":
        opp, hyp = p, q
        adj_sq = hyp * hyp - opp * opp
        adj = math.isqrt(adj_sq)
        check(adj * adj == adj_sq, "_reconstruct_from_ratio: sin gives no integer adjacent")
        return opp, adj, hyp
    if name == "cos":
        adj, hyp = p, q
        opp_sq = hyp * hyp - adj * adj
        opp = math.isqrt(opp_sq)
        check(opp * opp == opp_sq, "_reconstruct_from_ratio: cos gives no integer opposite")
        return opp, adj, hyp
    opp, adj = p, q
    hyp_sq = opp * opp + adj * adj
    hyp = math.isqrt(hyp_sq)
    check(hyp * hyp == hyp_sq, "_reconstruct_from_ratio: tan gives no integer hypotenuse")
    return opp, adj, hyp


def _weighted_ratio(rng, weights: dict[str, float]) -> str:
    """Pick a ratio name from ``weights`` (must sum to ~1.0)."""
    r = rng.random()
    cum = 0.0
    last = None
    for name, w in weights.items():
        cum += w
        last = name
        if r < cum:
            return name
    return last


def _named_distractors(name: str, opp: int, adj: int, hyp: int) -> list[str]:
    """Wrong side pairing (the other two named ratios) + the reciprocal of
    the asked ratio. For a scalene right triangle every named/reciprocal
    ratio is a genuinely different value; the two other named ratios also
    equal the cofunction values for the OTHER acute angle (sin<->cos), so
    this single family covers both "wrong pairing" and "other angle"
    misconceptions."""
    values = {"sin": Fraction(opp, hyp), "cos": Fraction(adj, hyp), "tan": Fraction(opp, adj)}
    correct = values[name]
    others = [v for k, v in values.items() if k != name]
    reciprocal = Fraction(correct.denominator, correct.numerator)
    candidates = others + [reciprocal]
    check_distinct(correct, *candidates, msg="trig_ratio_review_tan: a distractor equals the answer")
    return [rational_text(c) for c in candidates]


def _easy_tier(rng) -> dict:
    opp, adj, hyp = _pick_triangle(rng)
    name = _weighted_ratio(rng, {"sin": 0.25, "cos": 0.25, "tan": 0.5})
    value = _ratio_value(name, opp, adj, hyp)
    _check_ratio(name, value, opp, adj, hyp)
    answer_text = rational_text(value)

    num_side, den_side = _RATIO_FORMULA[name]
    num_val, den_val = _side_value(num_side, opp, adj, hyp), _side_value(den_side, opp, adj, hyp)

    question_text = (
        f"A right-angled triangle has an angle theta at one vertex. The side opposite theta "
        f"is {opp} units, the side adjacent to theta is {adj} units, and the hypotenuse is "
        f"{hyp} units. What is {name}(theta), as a fraction in lowest terms?"
    )
    working = [
        f"{name}(theta) uses the {num_side} and {den_side} sides.",
        f"{name}(theta) = {num_side}/{den_side} = {num_val}/{den_val} = {answer_text}.",
    ]
    distractors = pick_distractors(answer_text, _named_distractors(name, opp, adj, hyp))
    return dict(
        text=question_text,
        latex=f"plain:{question_text}",
        answer={"p": value.numerator, "q": value.denominator},
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


def _medium_tier(rng) -> dict:
    opp, adj, hyp = _pick_triangle(rng)

    # Which pair of sides is stated. Weighted so the branches that make tan
    # a genuine two-step question (opp_hyp, adj_hyp -- tan needs the missing
    # side) are drawn more often than the branch that cannot ask tan without
    # trivialising the tier (legs -- tan is directly opp/adj, no Pythagoras
    # needed, so that branch only ever asks sin or cos).
    case_r = rng.random()
    if case_r < 0.2:
        case = "legs"
    elif case_r < 0.6:
        case = "opp_hyp"
    else:
        case = "adj_hyp"

    if case == "legs":
        name = "sin" if rng.random() < 0.5 else "cos"
        given_desc = (
            f"The side opposite theta is {opp} units and the side adjacent to theta is {adj} "
            f"units. (The hypotenuse is not given.)"
        )
        missing, known_a, known_b = "hypotenuse", ("opposite", opp), ("adjacent", adj)
    elif case == "opp_hyp":
        name = "tan" if rng.random() < 0.8 else "cos"
        given_desc = (
            f"The side opposite theta is {opp} units and the hypotenuse is {hyp} units. "
            f"(The adjacent side is not given.)"
        )
        missing, known_a, known_b = "adjacent", ("opposite", opp), ("hypotenuse", hyp)
    else:
        name = "tan" if rng.random() < 0.8 else "sin"
        given_desc = (
            f"The side adjacent to theta is {adj} units and the hypotenuse is {hyp} units. "
            f"(The opposite side is not given.)"
        )
        missing, known_a, known_b = "opposite", ("adjacent", adj), ("hypotenuse", hyp)

    value = _ratio_value(name, opp, adj, hyp)
    _check_ratio(name, value, opp, adj, hyp)
    answer_text = rational_text(value)
    missing_val = _side_value(missing, opp, adj, hyp)

    question_text = (
        f"A right-angled triangle has an angle theta at one vertex. {given_desc} "
        f"What is {name}(theta), as a fraction in lowest terms?"
    )
    num_side, den_side = _RATIO_FORMULA[name]
    num_val, den_val = _side_value(num_side, opp, adj, hyp), _side_value(den_side, opp, adj, hyp)
    if missing == "hypotenuse":
        pythagoras_step = (
            f"First find the missing hypotenuse: hypotenuse^2 = {known_a[1]}^2 + {known_b[1]}^2, "
            f"so hypotenuse = {missing_val}."
        )
    else:
        pythagoras_step = (
            f"First find the missing {missing} side: {missing}^2 = {hyp}^2 - "
            f"{known_a[1] if known_a[0] != 'hypotenuse' else known_b[1]}^2, so {missing} = {missing_val}."
        )
    working = [
        pythagoras_step,
        f"{name}(theta) = {num_side}/{den_side} = {num_val}/{den_val} = {answer_text}.",
    ]
    distractors = pick_distractors(answer_text, _named_distractors(name, opp, adj, hyp))
    return dict(
        text=question_text,
        latex=f"plain:{question_text}",
        answer={"p": value.numerator, "q": value.denominator},
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


_HARD_PAIRS = [("sin", "tan"), ("tan", "sin"), ("cos", "tan"), ("tan", "cos")]


def _hard_tier(rng) -> dict:
    opp, adj, hyp = _pick_triangle(rng)
    given_name, asked_name = rng.choice(_HARD_PAIRS)
    given_value = _ratio_value(given_name, opp, adj, hyp)
    correct = _ratio_value(asked_name, opp, adj, hyp)
    _check_ratio(given_name, given_value, opp, adj, hyp)
    _check_ratio(asked_name, correct, opp, adj, hyp)

    # Independent check: reconstruct the triangle from the GIVEN ratio alone
    # (a different computation route from the direct construction above --
    # it never looks at opp/adj/hyp, only at given_value's own numerator/
    # denominator) and confirm the asked ratio recomputed from that
    # reconstruction agrees with the value construction produced.
    r_opp, r_adj, r_hyp = _reconstruct_from_ratio(given_name, given_value)
    check(r_opp * r_opp + r_adj * r_adj == r_hyp * r_hyp, "hard: reconstructed sides are not a right triangle")
    reconstructed_answer = _ratio_value(asked_name, r_opp, r_adj, r_hyp)
    check(
        reconstructed_answer == correct,
        f"hard: reconstruction {reconstructed_answer} disagrees with construction {correct}",
    )

    given_text = rational_text(given_value)
    answer_text = rational_text(correct)

    question_text = (
        f"For an acute angle theta in a right-angled triangle, {given_name}(theta) = "
        f"{given_text}. Find {asked_name}(theta), as a fraction in lowest terms."
    )
    working = [
        f"Reconstruct the triangle from {given_name}(theta) = {given_text}: opposite = "
        f"{r_opp}, adjacent = {r_adj}, hypotenuse = {r_hyp}.",
        f"{asked_name}(theta) = {answer_text}.",
    ]

    all_names = {"sin", "cos", "tan"}
    third_name = next(iter(all_names - {given_name, asked_name}))
    third_value = _ratio_value(third_name, opp, adj, hyp)
    reciprocal = Fraction(correct.denominator, correct.numerator)
    candidates_fr = [given_value, reciprocal, third_value]
    check_distinct(correct, *candidates_fr, msg="trig_ratio_review_tan/hard: a distractor equals the answer")
    candidates = [rational_text(c) for c in candidates_fr]
    distractors = pick_distractors(answer_text, candidates)

    return dict(
        text=question_text,
        latex=f"plain:{question_text}",
        answer={"p": correct.numerator, "q": correct.denominator},
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


@generator(
    "trig_ratio_review_tan",
    level=13,
    presets={
        "easy": dict(variant="direct"),
        "medium": dict(variant="pythagoras"),
        "hard": dict(variant="cofunction"),
    },
)
def trig_ratio_review_tan(rng, variant="direct"):
    """Review sine, cosine and tangent as ratios of side lengths in
    right-angled triangles (ACMMM028). Unlike the sibling
    senior-apply-the-cosine-and-sine-ratios-to-find (ACMEM119), which solves
    an applied word problem for an unknown side/angle and never uses
    tangent, this skill STATES a named ratio as an exact fraction of given
    or derived side lengths (or derives a second ratio from a stated one).
    "In lowest terms" is a display convention only: grade_fraction compares
    by exact value, so an unreduced but numerically equal typed answer is
    still accepted."""
    if variant == "direct":
        return _easy_tier(rng)
    if variant == "pythagoras":
        return _medium_tier(rng)
    if variant == "cofunction":
        return _hard_tier(rng)
    raise ValueError(f"unknown variant: {variant}")


# ---------------------------------------------------------------------------
# Skill B: elevation_depression_problem
# ---------------------------------------------------------------------------

# Theta pool excludes 45 degrees, which makes height == distance exactly
# (tan(45) = 1) and so renders a trailing-zero "X.0" answer at 1 decimal
# place every time -- the same defect class p5b_trig_ratio_build.py guards
# against with its 40..50 exclusion band.
THETA_POOL = [t for t in range(10, 81) if t != 45]

_RESAMPLE_LIMIT = 80


def _round1(v: float) -> str:
    return f"{v:.1f}"


def _on_rounding_boundary(v: float) -> bool:
    """True when v*10 sits within 1e-9 of a half-integer -- the case where
    round-half-to-even and the learner's own calculator could disagree on
    the last digit."""
    scaled = v * 10
    frac = scaled - math.floor(scaled)
    return abs(frac - 0.5) < 1e-9


def _bad_trailing_zero(text: str) -> bool:
    return text.endswith(".0")


def _check_elevation(d: float, theta_deg: float, h: float) -> None:
    """Independent check for the easy tier: a different route (hypotenuse
    via cosine, then height via sine) from the direct tan multiplication
    used to construct ``h``."""
    hyp = d / math.cos(math.radians(theta_deg))
    h_check = hyp * math.sin(math.radians(theta_deg))
    check(abs(h_check - h) < 1e-6, "elevation: sin/cos route disagrees with tan route")


def _check_depression(H: float, theta_deg: float, dist: float) -> None:
    """Independent check for the medium tier: a different route (hypotenuse
    via sine, then distance via cosine) from the direct division used to
    construct ``dist``."""
    hyp = H / math.sin(math.radians(theta_deg))
    dist_check = hyp * math.cos(math.radians(theta_deg))
    check(abs(dist_check - dist) < 1e-6, "depression: sin/hyp route disagrees with tan route")


def _check_two_observation(theta1_deg: float, theta2_deg: float, d: float, h: float) -> None:
    """Independent check for the hard tier: back-substitute ``h`` into the
    two ORIGINAL tan equations (a direct re-derivation from first
    principles, not the closed form used to construct ``h``) and confirm
    the ground-point separation they imply reproduces ``d``."""
    x1 = h / math.tan(math.radians(theta1_deg))  # near point's distance to the base
    x2 = h / math.tan(math.radians(theta2_deg))  # far point's distance to the base
    check(abs((x2 - x1) - d) < 1e-6, "two-observation: back-substitution does not reproduce d")


def _easy_tier_elevation(rng) -> dict:
    d = theta = h = answer_text = None
    for _ in range(_RESAMPLE_LIMIT):
        d = rng.randint(5, 60)
        theta = rng.choice(THETA_POOL)
        h = d * math.tan(math.radians(theta))
        if _on_rounding_boundary(h):
            continue
        answer_text = _round1(h)
        if _bad_trailing_zero(answer_text):
            continue
        break
    else:
        check(False, "elevation easy: no non-degenerate draw found")
    answer = float(answer_text)
    _check_elevation(d, theta, h)

    question_text = (
        f"From a point on level ground {d} m from the base of a tower, the angle of "
        f"elevation to the top of the tower is {theta} degrees. Find the height of the "
        f"tower, to 1 decimal place."
    )
    working = [
        f"height = distance x tan(angle) = {d} x tan({theta} degrees)",
        f"height = {answer_text} m (to 1 decimal place).",
    ]
    candidates = [
        _round1(d * math.sin(math.radians(theta))),  # swaps tan for sin
        _round1(d * math.cos(math.radians(theta))),  # swaps tan for cos
        _round1(d / math.tan(math.radians(theta))),  # inverts the ratio
    ]
    distractors = pick_distractors(answer_text, candidates, tolerance=Fraction(1, 20))
    return dict(
        text=question_text,
        latex=f"plain:{question_text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


def _medium_tier_depression(rng) -> dict:
    H = theta = dist = answer_text = None
    for _ in range(_RESAMPLE_LIMIT):
        H = rng.randint(5, 60)
        theta = rng.choice(THETA_POOL)
        dist = H / math.tan(math.radians(theta))
        if dist < 5:  # a boat closer than this to the cliff base is a contrived scenario
            continue
        if _on_rounding_boundary(dist):
            continue
        answer_text = _round1(dist)
        if _bad_trailing_zero(answer_text):
            continue
        break
    else:
        check(False, "depression medium: no non-degenerate draw found")
    answer = float(answer_text)
    _check_depression(H, theta, dist)

    question_text = (
        f"An observer stands at the top of a cliff {H} m high. The angle of depression from "
        f"the observer to a boat on the water is {theta} degrees. The angle of depression "
        f"from the top equals the angle of elevation from the boat, so find the horizontal "
        f"distance from the base of the cliff to the boat, to 1 decimal place."
    )
    working = [
        f"The angle of elevation from the boat to the top of the cliff also equals {theta} degrees (alternate angles).",
        f"distance = height / tan(angle) = {H} / tan({theta} degrees) = {answer_text} m (to 1 decimal place).",
    ]
    candidates = [
        _round1(H * math.tan(math.radians(theta))),  # multiplies instead of divides
        _round1(H / math.sin(math.radians(theta))),  # uses sine instead of tangent
        _round1(H / math.cos(math.radians(theta))),  # uses cosine instead of tangent
    ]
    distractors = pick_distractors(answer_text, candidates, tolerance=Fraction(1, 20))
    return dict(
        text=question_text,
        latex=f"plain:{question_text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


def _check_twin_base_distance(H: float, theta1_deg: float, theta2_deg: float, d: float) -> None:
    """Independent check for the exam tier: recompute each observer's
    distance to the tower's base via the hypotenuse route (H/sin(theta),
    then x hyp*cos(theta)) -- the same alternate route ``_check_elevation``
    uses for a single triangle -- rather than the direct H/tan(theta)
    division used at construction, applied separately to BOTH triangles, and
    confirm the difference between the two reproduces ``d``."""
    hyp1 = H / math.sin(math.radians(theta1_deg))
    x1_check = hyp1 * math.cos(math.radians(theta1_deg))
    hyp2 = H / math.sin(math.radians(theta2_deg))
    x2_check = hyp2 * math.cos(math.radians(theta2_deg))
    check(abs((x2_check - x1_check) - d) < 1e-6, "twin_base_distance: hyp/sin/cos route disagrees with tan route")


def _hard_tier_two_observation(rng) -> dict:
    theta1 = theta2 = d = h = cot1 = cot2 = answer_text = None
    for _ in range(_RESAMPLE_LIMIT):
        theta2 = rng.randint(15, 40)
        delta = rng.randint(12, 40)
        theta1 = theta2 + delta
        if theta1 >= 80:
            continue
        d = rng.randint(10, 80)

        t1, t2 = math.radians(theta1), math.radians(theta2)
        cot1, cot2 = 1 / math.tan(t1), 1 / math.tan(t2)
        denom = cot2 - cot1
        if denom < 0.15:  # keep the closed form well-conditioned
            continue
        h = d / denom
        if _on_rounding_boundary(h):
            continue
        answer_text = _round1(h)
        if _bad_trailing_zero(answer_text):
            continue
        break
    else:
        check(False, "two-observation hard: no well-conditioned draw found")
    answer = float(answer_text)
    _check_two_observation(theta1, theta2, d, h)
    t1, t2 = math.radians(theta1), math.radians(theta2)

    question_text = (
        f"Standing on level ground, an observer measures the angle of elevation to the top "
        f"of a tower as {theta1} degrees. Walking directly away from the tower, along a "
        f"straight line through the tower's base, the observer stops {d} m further away and "
        f"measures the angle of elevation again, now {theta2} degrees. The two observation "
        f"points and the base of the tower are collinear, on the same side of the tower. "
        f"Find the height of the tower, to 1 decimal place."
    )
    working = [
        f"Let x be the far point's distance to the base: height = x tan({theta2} degrees) = "
        f"(x - {d}) tan({theta1} degrees).",
        f"Solving: height = {d} / (cot({theta2} degrees) - cot({theta1} degrees)) = "
        f"{answer_text} m (to 1 decimal place).",
    ]
    near_only = _round1(d * math.tan(t1))  # single-observation height, near angle
    far_only = _round1(d * math.tan(t2))  # single-observation height, far angle
    opposite_sides = _round1(d / (cot1 + cot2))  # the "opposite sides" misreading of "d apart"
    candidates = [near_only, far_only, opposite_sides]
    distractors = pick_distractors(answer_text, candidates, tolerance=Fraction(1, 20))
    return dict(
        text=question_text,
        latex=f"plain:{question_text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


# Three structurally distinct phrasings of the same exam-tier scenario, so a `prompt_shape` sweep sees
# genuine sentence variety rather than only the drawn numbers -- every other source on this topic renders a
# single fixed template, so the exam source is the first to need this. Each states H, theta1 (near/steeper)
# and theta2 (far/shallower) in a different order and with different vocabulary/labelling, never reusing
# the same clause structure.
_EXAM_TWIN_BASE_DISTANCE_TEMPLATES = (
    "A tower is {H} m tall. Observers A and B stand on level ground on the same side "
    "of the tower, with A, B and the base of the tower collinear. From A, the angle of "
    "elevation to the top of the tower is {theta1} degrees. From B, further from the "
    "tower than A, the angle of elevation to the top is {theta2} degrees. Find the "
    "distance AB, to 1 decimal place.",
    "Two survey markers, P and Q, lie on level ground on the same side of a flagpole, "
    "with P, Q and the base of the flagpole collinear. The angle of elevation from P to "
    "the top of the flagpole is {theta1} degrees, and from Q, which is further from the "
    "flagpole than P, it is {theta2} degrees. The flagpole is {H} m tall. Find the "
    "distance PQ, to 1 decimal place.",
    "A lighthouse is {H} m tall. A ship's navigator measures the angle of elevation to the "
    "top of the lighthouse while sailing directly away from it along a single straight "
    "line, on the same side: {theta1} degrees from the nearer position, then {theta2} "
    "degrees from a farther position. How far apart, to 1 decimal place, are the two "
    "positions?",
)


def _exam_tier_twin_base_distance(rng) -> dict:
    """Exam tier: the mirror of the hard tier's "which quantity is unknown"
    -- same two right triangles sharing the tower's vertical side, but now
    the HEIGHT is given and both base angles are given, and the unknown is
    the HORIZONTAL DISTANCE between the two observers (the hard tier gives
    that distance and asks for the height). Genuinely needs both triangles:
    with only one angle the distance from that one observer to the base is
    directly computable, but the gap between the two observers is not."""
    H = theta1 = theta2 = x1 = x2 = d = answer_text = None
    for _ in range(_RESAMPLE_LIMIT):
        H = rng.randint(5, 60)
        theta2 = rng.randint(15, 40)  # far observer: shallower angle
        delta = rng.randint(12, 40)
        theta1 = theta2 + delta  # near observer: steeper angle
        if theta1 >= 80:
            continue
        x1 = H / math.tan(math.radians(theta1))
        x2 = H / math.tan(math.radians(theta2))
        d = x2 - x1
        if d < 5:  # two observers closer together than this is a contrived scenario
            continue
        if _on_rounding_boundary(d):
            continue
        answer_text = _round1(d)
        if _bad_trailing_zero(answer_text):
            continue
        break
    else:
        check(False, "twin_base_distance exam: no well-conditioned draw found")
    answer = float(answer_text)
    _check_twin_base_distance(H, theta1, theta2, d)

    question_text = phrase(rng, _EXAM_TWIN_BASE_DISTANCE_TEMPLATES, H=H, theta1=theta1, theta2=theta2)
    working = [
        f"Let x1 and x2 be the horizontal distances from the two positions to the base: "
        f"{H} = x1 x tan({theta1} degrees) = x2 x tan({theta2} degrees).",
        f"distance = x2 - x1 = {H}/tan({theta2} degrees) - {H}/tan({theta1} degrees) = "
        f"{answer_text} m (to 1 decimal place).",
    ]
    near_only = _round1(x1)  # single-observation shortcut: A's own distance to the base
    far_only = _round1(x2)  # single-observation shortcut: B's own distance to the base
    opposite_sides = _round1(x1 + x2)  # opposite-sides misreading: sums instead of subtracting
    candidates = [near_only, far_only, opposite_sides]
    distractors = pick_distractors(answer_text, candidates, tolerance=Fraction(1, 20))
    return dict(
        text=question_text,
        latex=f"plain:{question_text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


@generator(
    "elevation_depression_problem",
    level=13,
    presets={
        "easy": dict(variant="elevation"),
        "medium": dict(variant="depression"),
        "hard": dict(variant="two_observation"),
        "exam": dict(variant="twin_base_distance"),
    },
)
def elevation_depression_problem(rng, variant="elevation"):
    """Work with the concepts of angle of elevation and angle of depression
    (ACMEM118). Easy: elevation from ground level. Medium: depression from
    a height, using the alternate-angle correction. Hard: a two-observation
    surveying problem, solving two tan equations together; the question
    states explicitly that the two ground points and the object's base are
    collinear and on the same side, since "d apart" also has a legitimate
    opposite-sides reading with a different answer. Exam: the same two
    right triangles sharing the tower's vertical side, but the OTHER
    quantity unknown -- height and both base angles are given, and the
    question asks for the horizontal distance between the two observers,
    not the height."""
    if variant == "elevation":
        return _easy_tier_elevation(rng)
    if variant == "depression":
        return _medium_tier_depression(rng)
    if variant == "two_observation":
        return _hard_tier_two_observation(rng)
    if variant == "twin_base_distance":
        return _exam_tier_twin_base_distance(rng)
    raise ValueError(f"unknown variant: {variant}")
