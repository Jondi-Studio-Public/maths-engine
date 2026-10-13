"""Generates `fy1ca_applied_optimisation` and `fy1ca_applied_optimisation_exam` questions.

One first-year skill, ``ucv1-applied-optimisation``: turn a word problem into a function of one variable on a closed
interval, find its stationary points and compare them with the ends. The senior library already covers stationary
points of a given function and the second derivative test; this skill is the step up, where the function has to be
built from a constraint and the answer may sit at an end of the allowed interval. Every key is a whole number or a
decimal shown to its stated places, graded as a number.

Tiers (the reasoning escalates; the numbers stay small):

- ``easy``: a fenced rectangle with a fixed length of fencing, one side on a creek or divided by cross fences. Eliminate
  one variable, differentiate a quadratic and give the greatest area.
- ``medium``: a cubic or rational function that needs more set-up: an open box folded from a square sheet, an open-top
  tank of fixed volume with least material, or the largest rectangle that fits in a right-angled gable.
- ``hard``: the same set-up, but the allowed interval matters: a cable that crosses a river at one cost and runs along
  the bank at another, or a paddock whose width is capped. The stationary point may lie outside the interval, so the
  least cost or greatest area is at an end.
- ``exam`` (topic ``fy1ca_applied_optimisation_exam``): a four-part chain on an open-top square-based tank of fixed
  volume: (a) the height at a trial side, (b) the surface area there, (c) the side that uses least material,
  (d) the least cost of the sheet metal.

Every key is re-derived by a second route: a fine grid search over the allowed interval finds the same optimum.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import check, generator, latex_answer, numeric_answer, phrase, stated_precision_string
from ._fy1ca_kit import (
    Redraw,
    build_with_redraws,
    close,
    grid,
    need_distractors,
    number_distractors,
    number_part,
)


def _grid_optimum(fn, lo: float, hi: float, maximise: bool, n: int = 4000) -> float:
    """The best value of `fn` on a fine grid over [lo, hi]: the numerical second route."""
    values = [fn(lo + (hi - lo) * i / n) for i in range(n + 1)]
    return max(values) if maximise else min(values)


def _number_question(text: str, answer, wrong, working, dp: int = 0) -> dict:
    distractors = number_distractors(answer, wrong, dp)
    need_distractors(distractors)
    rounded = grid(answer, dp)
    answer_text = stated_precision_string(rounded, dp)
    return dict(
        text=text,
        answer=numeric_answer(rounded),
        answer_text=answer_text,
        answer_kind="number",
        answer_latex=latex_answer(answer_text),
        working=working,
        distractors=distractors,
    )


# -- easy ------------------------------------------------------------------------------------------------------------


def _creek(rng: random.Random) -> dict:
    """P metres of fencing for three sides of a rectangle; the fourth side is a creek. Width x, length P - 2x."""
    fence = rng.choice([40, 60, 80, 100, 120, 160, 200])
    x_star = Fraction(fence, 4)
    best = x_star * (fence - 2 * x_star)
    area = lambda x: x * (fence - 2 * x)  # noqa: E731
    check(close(_grid_optimum(area, 0, fence / 2, True), float(best)), "grid search disagrees")
    wrong = [
        (x_star, "fy1ca-opt-variable-reported"),
        (Fraction(fence, 4) ** 2, "fy1ca-opt-square-assumed"),
        (Fraction(fence**2, 16), "fy1ca-opt-all-sides-fenced"),
        (x_star * (fence - x_star), "fy1ca-opt-length-not-eliminated"),
    ]
    text = phrase(
        rng,
        (
            "A farmer has {p} m of fencing to make a rectangular paddock beside a creek. No fence is needed along the creek. What is the greatest area, in m², that can be enclosed?",
            "A rectangular sheep yard is to be built against a long shed wall using {p} m of fencing for the other three sides. Find the greatest possible area of the yard, in m².",
        ),
        p=fence,
    )
    working = [
        f"Let the width (the two sides at right angles to the unfenced side) be x, so the length is {fence} - 2x.",
        f"A(x) = x({fence} - 2x), so A'(x) = {fence} - 4x = 0 gives x = {x_star}.",
        f"A({x_star}) = {x_star} × {fence - 2 * x_star} = {best}.",
    ]
    return _number_question(text, best, wrong, working)


def _dividers(rng: random.Random) -> dict:
    """A rectangle with k cross fences: 2L + (k + 2)W = P, area L*W."""
    k = rng.choice([1, 2, 3])
    sides = k + 2
    width = rng.choice([4, 6, 10, 12, 20])
    fence = 2 * sides * width  # W = P / (2 * sides) at the optimum
    length = Fraction(fence - sides * width, 2)
    best = length * width
    area = lambda w: w * (fence - sides * w) / 2  # noqa: E731
    check(close(_grid_optimum(area, 0, fence / sides, True), float(best)), "grid search disagrees")
    wrong = [
        (Fraction(width), "fy1ca-opt-variable-reported"),
        (Fraction(fence, 4) ** 2, "fy1ca-opt-square-assumed"),
        (Fraction(fence**2, 16), "fy1ca-opt-all-sides-fenced"),
        (length * length, "fy1ca-opt-square-assumed"),
        (Fraction(width) * (fence - width) / 2, "fy1ca-opt-length-not-eliminated"),
    ]
    text = phrase(
        rng,
        (
            "A rectangular enclosure for a school garden uses {p} m of fencing, including {k} fence{s} parallel to one side that divide{d} it into {n} equal beds. What is the greatest total area, in m², that the enclosure can have?",
            "A market gardener fences a rectangle with {p} m of fencing and adds {k} internal fence{s} parallel to a shorter side, making {n} equal plots. Find the greatest area, in m², of the rectangle.",
        ),
        p=fence,
        k=k,
        s="" if k == 1 else "s",
        d="s" if k == 1 else "",
        n=k + 1,
    )
    working = [
        f"Let the {sides} fences parallel to the width have length W, so the other two sides have length L = ({fence} - {sides}W)/2.",
        f"A(W) = W({fence} - {sides}W)/2, and A'(W) = ({fence} - {2 * sides}W)/2 = 0 gives W = {width}.",
        f"L = {length}, so the greatest area is {length} × {width} = {best}.",
    ]
    return _number_question(text, best, wrong, working)


def _easy(rng: random.Random) -> dict:
    return rng.choice([_creek, _dividers])(rng)


# -- medium --------------------------------------------------------------------------------------------------------------


def _open_box(rng: random.Random) -> dict:
    side = rng.choice([12, 18, 24, 30, 36])
    cut = Fraction(side, 6)
    best = cut * (side - 2 * cut) ** 2
    volume = lambda x: x * (side - 2 * x) ** 2  # noqa: E731
    check(close(_grid_optimum(volume, 0, side / 2, True), float(best)), "grid search disagrees")
    wrong = [
        (cut, "fy1ca-opt-variable-reported"),
        (Fraction(side, 4) * (side - Fraction(side, 2)) ** 2, "fy1ca-opt-derivative-slip"),
        (Fraction(side) ** 3 / 27, "fy1ca-opt-wrong-coefficient"),
        (Fraction(side, 3) * (side - Fraction(2 * side, 3)) ** 2, "fy1ca-opt-derivative-slip"),
        (cut * (side - cut) ** 2, "fy1ca-opt-length-not-eliminated"),
    ]
    text = phrase(
        rng,
        (
            "An open box is made from a square sheet of aluminium {a} cm by {a} cm by cutting a square of side x cm from each corner and folding up the sides. What is the greatest volume, in cm³, that the box can have?",
            "A tray with no lid is folded from a {a} cm by {a} cm square of cardboard after squares of side x are cut from the corners. Find the maximum volume of the tray, in cm³.",
        ),
        a=side,
    )
    working = [
        f"The base is ({side} - 2x) by ({side} - 2x) and the height is x, so V(x) = x({side} - 2x)², for 0 < x < {side // 2}.",
        f"V'(x) = ({side} - 2x)({side} - 6x) = 0 gives x = {cut} (x = {side // 2} gives zero volume).",
        f"V({cut}) = {cut} × {side - 2 * cut}² = {best}.",
    ]
    return _number_question(text, best, wrong, working)


def _tank(rng: random.Random) -> dict:
    x_star = rng.choice([2, 4, 6, 8, 10])
    volume = Fraction(x_star**3, 2)
    best = Fraction(x_star) ** 2 + 4 * volume / x_star
    surface = lambda x: x * x + 4 * float(volume) / x  # noqa: E731
    check(close(_grid_optimum(surface, 0.2, 4 * x_star, False, 3000), float(best), 1e-4), "grid search disagrees")
    wrong = [
        (Fraction(x_star), "fy1ca-opt-variable-reported"),
        (Fraction(x_star) ** 2 + 2 * volume / x_star, "fy1ca-opt-wrong-faces"),
        (2 * Fraction(x_star) ** 2 + 4 * volume / x_star, "fy1ca-opt-closed-top-counted"),
        (Fraction(round(float(volume) ** (1 / 3) * 100), 100) ** 2 * 3, "fy1ca-opt-derivative-slip"),
        (3 * Fraction(x_star) ** 2, "fy1ca-opt-wrong-faces"),
    ]
    text = phrase(
        rng,
        (
            "A farmer wants an open-top rainwater tank with a square base and a capacity of {v} m³. What is the least area of sheet steel, in m², needed to make it?",
            "A square-based tank with no lid must hold {v} m³ of water. Find the smallest possible total area of the base and four sides, in m².",
        ),
        v=volume,
    )
    working = [
        f"With base side x and height h, V = x²h = {volume}, so h = {volume}/x².",
        f"The surface area is S = x² + 4xh = x² + {4 * volume}/x, so S'(x) = 2x - {4 * volume}/x² = 0 gives x³ = {2 * volume}, x = {x_star}.",
        f"S({x_star}) = {x_star}² + {4 * volume}/{x_star} = {best}.",
    ]
    return _number_question(text, best, wrong, working)


def _gable(rng: random.Random) -> dict:
    base, height = rng.choice([(8, 6), (12, 9), (10, 8), (6, 4), (12, 5), (16, 6), (20, 8)])
    best = Fraction(base * height, 4)
    area = lambda x: x * height * (1 - x / base)  # noqa: E731
    check(close(_grid_optimum(area, 0, base, True), float(best)), "grid search disagrees")
    wrong = [
        (Fraction(base, 2), "fy1ca-opt-variable-reported"),
        (Fraction(base * height, 2), "fy1ca-opt-whole-triangle"),
        (Fraction(base * height, 3), "fy1ca-opt-wrong-coefficient"),
        (Fraction(base * height, 6), "fy1ca-opt-wrong-coefficient"),
        (Fraction(base, 3) * height * (1 - Fraction(1, 3)), "fy1ca-opt-derivative-slip"),
    ]
    text = phrase(
        rng,
        (
            "The gable end of a shed is a right-angled triangle {b} m wide at the floor and {h} m high at the tall side. A rectangular window is to fit inside it with one corner on the sloping edge and two sides along the floor and the tall side. What is the greatest area of window, in m²?",
            "A right-angled triangular panel has legs {b} m (horizontal) and {h} m (vertical). A rectangle is cut from it with two sides along the legs and the opposite corner on the hypotenuse. Find the largest area the rectangle can have, in m².",
        ),
        b=base,
        h=height,
    )
    working = [
        f"The hypotenuse is y = {height}(1 - x/{base}), so a rectangle of width x has height y and area A(x) = {height}x(1 - x/{base}).",
        f"A'(x) = {height}(1 - 2x/{base}) = 0 gives x = {Fraction(base, 2)}.",
        f"A = {Fraction(base, 2)} × {Fraction(height, 2)} = {best}.",
    ]
    return _number_question(text, best, wrong, working)


def _medium(rng: random.Random) -> dict:
    return rng.choice([_open_box, _tank, _gable])(rng)


# -- hard ----------------------------------------------------------------------------------------------------------------

# (c1, c2, q): relative costs of the two routes and q = sqrt(c1^2 - c2^2)
_COSTS = ((5, 3, 4), (5, 4, 3), (13, 12, 5), (13, 5, 12), (10, 8, 6), (25, 24, 7))
# (a, d, e): a river of width a, bank distance d, a^2 + d^2 = e^2
_RIVERS = (
    (3, 4, 5),
    (4, 3, 5),
    (5, 12, 13),
    (12, 5, 13),
    (6, 8, 10),
    (8, 6, 10),
    (8, 15, 17),
    (15, 8, 17),
    (7, 24, 25),
)


def _cable(rng: random.Random) -> dict:
    c1r, c2r, q = rng.choice(_COSTS)
    kappa = rng.choice([2, 4, 5, 10])
    c1, c2 = c1r * kappa, c2r * kappa
    unit = rng.choice([10, 20])
    interior = rng.random() < 0.5
    if interior:
        # the width is a multiple of q so that the stationary point x* = c2 a / q and the slant length are whole
        a = q * rng.choice([1, 2]) * unit
        x_star = Fraction(c2r * a, q)
        slant = Fraction(c1r * a, q)
        gap = rng.choice([1, 2, 3]) * unit
        d = int(x_star) + gap
        best = c1 * slant + c2 * (d - x_star)
        endpoint = False
    else:
        a0, d0, e0 = rng.choice(_RIVERS)
        m = rng.choice([1, 2]) * unit // 2
        a, d, e = a0 * m, d0 * m, e0 * m
        x_star = Fraction(c2r * a, q)
        if not x_star > d:
            raise Redraw
        best = Fraction(c1 * e)
        endpoint = True
    cost = lambda x: c1 * math.sqrt(a * a + x * x) + c2 * (d - x)  # noqa: E731
    check(close(_grid_optimum(cost, 0, d, False, 3000), float(best), 1e-4), "grid search disagrees")
    at_zero = Fraction(c1 * a + c2 * d)
    all_water = Fraction(c1 * math.isqrt(a * a + d * d)) if math.isqrt(a * a + d * d) ** 2 == a * a + d * d else None
    slant_at_xstar = Fraction(c1r * a, q)
    interior_cost = c1 * slant_at_xstar + c2 * (d - x_star)
    wrong = [
        (at_zero, "fy1ca-opt-endpoint-not-compared"),
        (all_water, "fy1ca-opt-endpoint-not-compared")
        if not endpoint
        else (interior_cost, "fy1ca-opt-ignores-constraint"),
        (Fraction(c2 * a + c1 * d), "fy1ca-opt-costs-swapped"),
        (Fraction(c1 * (a + d)), "fy1ca-opt-water-only"),
        (Fraction(c2 * (a + d)), "fy1ca-opt-land-only"),
    ]
    wrong = [w for w in wrong if w is not None and w[0] is not None]
    text = phrase(
        rng,
        (
            "A power cable must run from a point A on one bank of a straight river {a} m wide to a point B on the other bank, {d} m downstream from the point directly opposite A. Laying cable under water costs ${c1} per metre and along the bank costs ${c2} per metre. The cable may leave the water at any point between the point opposite A and B. What is the least possible cost, in dollars?",
            "A pipeline is to join a pump A on the north bank of a {a} m wide river to a tank B on the south bank, {d} m further along the river than the point directly opposite A. It costs ${c1} per metre to lay under the river and ${c2} per metre on land. The pipe may come ashore anywhere between the point opposite A and B. Find the minimum cost, in dollars.",
        ),
        a=a,
        d=d,
        c1=c1,
        c2=c2,
    )
    if endpoint:
        working = [
            f"Let x be the distance along the bank at which the line leaves the water, 0 ≤ x ≤ {d}. C(x) = {c1}√({a}² + x²) + {c2}({d} - x).",
            f"C'(x) = 0 at x = {x_star}, which is beyond {d}, so C is decreasing on the whole interval and the least cost is at x = {d}.",
            f"C({d}) = {c1} × √({a}² + {d}²) = {c1} × {Fraction(math.isqrt(a * a + d * d))} = {best}.",
        ]
    else:
        working = [
            f"Let x be the distance along the bank at which the line leaves the water, 0 ≤ x ≤ {d}. C(x) = {c1}√({a}² + x²) + {c2}({d} - x).",
            f"C'(x) = {c1}x/√({a}² + x²) - {c2} = 0 gives x = {x_star}, inside the interval, with √({a}² + x²) = {slant_at_xstar}.",
            f"C({x_star}) = {c1} × {slant_at_xstar} + {c2} × {d - x_star} = {best}; this is below C(0) = {at_zero} and C({d}).",
        ]
    return _number_question(text, best, wrong, working)


def _capped_paddock(rng: random.Random) -> dict:
    fence = rng.choice([60, 80, 100, 120, 160])
    x_star = Fraction(fence, 4)
    cap = Fraction(rng.choice([k for k in range(5, int(x_star)) if k % 5 == 0] or [int(x_star) - 5]))
    best = cap * (fence - 2 * cap)
    area = lambda x: x * (fence - 2 * x)  # noqa: E731
    check(close(_grid_optimum(area, 0, float(cap), True), float(best)), "grid search disagrees")
    unconstrained = x_star * (fence - 2 * x_star)
    wrong = [
        (unconstrained, "fy1ca-opt-ignores-constraint"),
        (cap, "fy1ca-opt-variable-reported"),
        (x_star, "fy1ca-opt-variable-reported"),
        (cap * (fence - cap), "fy1ca-opt-length-not-eliminated"),
        (cap * cap, "fy1ca-opt-square-assumed"),
    ]
    text = phrase(
        rng,
        (
            "A paddock beside a creek is fenced on three sides with {p} m of fencing. Because of a drainage easement, the sides at right angles to the creek can be at most {cap} m long. What is the greatest area, in m², that can be enclosed?",
            "Using {p} m of fencing, a rectangular yard is built against a wall, so only three sides need fencing. The width (the sides at right angles to the wall) cannot exceed {cap} m. Find the greatest possible area of the yard, in m².",
        ),
        p=fence,
        cap=cap,
    )
    working = [
        f"Let the width be x, so the length is {fence} - 2x and A(x) = x({fence} - 2x) for 0 < x ≤ {cap}.",
        f"A'(x) = {fence} - 4x = 0 gives x = {x_star}, which is outside the allowed range, so A is increasing on it.",
        f"The greatest area is at x = {cap}: A = {cap} × {fence - 2 * cap} = {best}.",
    ]
    return _number_question(text, best, wrong, working)


def _hard(rng: random.Random) -> dict:
    return rng.choice([_cable, _cable, _capped_paddock])(rng)


_TIERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator("fy1ca_applied_optimisation", level=19, presets={t: dict(tier=t) for t in _TIERS})
def fy1ca_applied_optimisation(rng: random.Random, tier: str = "easy") -> dict:
    """Constrained optimisation: fenced areas (easy), boxes, tanks and gables (medium), end-point cases (hard)."""
    check(tier in _TIERS, f"unknown tier {tier!r}")
    return build_with_redraws(rng, _TIERS[tier], f"a {tier} optimisation question")


# -- exam ----------------------------------------------------------------------------------------------------------------


def _exam_build(rng: random.Random) -> dict:
    x_star = rng.choice([2, 4, 6])
    volume = Fraction(x_star**3, 2)
    price = rng.choice([25, 30, 40, 50])
    trial = rng.choice([1, 2, 3, 4, 5, 8, 10])
    height = volume / trial**2
    if trial == x_star or (height * 100).denominator != 1 or height < Fraction(1, 10):
        raise Redraw
    surface_trial = trial**2 + 4 * trial * height
    least = Fraction(x_star) ** 2 + 4 * volume / x_star
    cost = price * least
    # second route: a grid search for the side that uses least material
    surface = lambda x: x * x + 4 * float(volume) / x  # noqa: E731
    best_x = min((0.01 * i for i in range(20, 2000)), key=surface)
    check(abs(best_x - x_star) < 0.011, "grid search disagrees on the best side")
    check(close(surface(trial), float(surface_trial)), "surface area at the trial side disagrees")

    a_dist = number_distractors(
        height,
        [
            (volume / trial, "fy1ca-opt-area-not-squared"),
            (volume * trial**2, "fy1ca-opt-height-inverted"),
            (volume / (4 * trial**2), "fy1ca-opt-wrong-faces"),
            (Fraction(trial**2) / volume, "fy1ca-opt-height-inverted"),
        ],
        2,
    )
    b_dist = number_distractors(
        surface_trial,
        [
            (2 * trial**2 + 4 * trial * height, "fy1ca-opt-closed-top-counted"),
            (trial**2 + trial * height, "fy1ca-opt-one-side-only"),
            (trial**2 + 2 * trial * height, "fy1ca-opt-wrong-faces"),
            (trial**2 * height, "fy1ca-opt-volume-for-area"),
        ],
        2,
    )
    c_dist = number_distractors(
        Fraction(x_star),
        [
            (Fraction(round((float(volume) / 2) ** (1 / 3))), "fy1ca-opt-derivative-slip"),
            (Fraction(round(float(volume) ** (1 / 3))), "fy1ca-opt-factor-dropped"),
            (Fraction(round(math.sqrt(float(volume)))), "fy1ca-opt-wrong-root"),
            (Fraction(round(float(volume))), "fy1ca-opt-volume-for-side"),
            (Fraction(x_star + 1), "fy1ca-opt-stopped-near"),
        ],
        0,
    )
    d_dist = number_distractors(
        cost,
        [
            (price * surface_trial, "fy1ca-opt-cost-at-trial-value"),
            (price * Fraction(x_star) ** 2, "fy1ca-opt-wrong-faces"),
            (price * volume, "fy1ca-opt-volume-for-area"),
            (Fraction(x_star) * price * 4, "fy1ca-opt-wrong-faces"),
        ],
        0,
    )
    for dist in (a_dist, b_dist, c_dist, d_dist):
        need_distractors(dist)
    stem = phrase(
        rng,
        (
            "A farmer is designing an open-top rainwater tank with a square base of side x metres and height h metres. It must hold {v} m³ of water, and the steel for the base and sides costs ${c} per square metre.",
            "A shed owner wants a tank with no lid, a square base of side x metres, height h metres and a capacity of {v} m³. The base and sides are cut from sheet steel at ${c} per square metre.",
            "A school garden needs a water tank with a square base of side x metres and no lid. Its height is h metres and it must hold {v} m³. Sheet steel for the base and four sides costs ${c} per square metre.",
        ),
        v=volume,
        c=price,
    )
    parts = [
        number_part(
            f"Find the height of the tank, in metres, when x = {trial}. Give your answer correct to 2 decimal places.",
            height,
            2,
            1,
            [f"V = x²h, so h = {volume}/{trial}² = {float(height):.2f} m."],
            a_dist,
            sensible_range=(0, None),
        ),
        number_part(
            f"Hence find the area of steel, in m², needed when x = {trial}. Give your answer correct to 2 decimal places.",
            surface_trial,
            2,
            2,
            [f"S = x² + 4xh = {trial}² + 4({trial})({float(height):.2f}) = {float(surface_trial):.2f} m²."],
            b_dist,
            follow_through=f"{trial}**2 + 4*{trial}*part_a",
            sensible_range=(0, None),
        ),
        number_part(
            "Find the value of x for which the least area of steel is used.",
            Fraction(x_star),
            0,
            3,
            [
                f"S(x) = x² + {4 * volume}/x, so S'(x) = 2x - {4 * volume}/x² = 0 gives x³ = {2 * volume}.",
                f"Hence x = {x_star}; S'' > 0 there, so this is a minimum.",
            ],
            c_dist,
            sensible_range=(0, None),
        ),
        number_part(
            phrase(
                rng,
                (
                    "Hence find the least possible cost of the steel, in dollars.",
                    "Find the least possible cost of the steel for the tank, in dollars.",
                ),
            ),
            cost,
            0,
            2,
            [
                f"S({x_star}) = {x_star}² + {4 * volume}/{x_star} = {least} m², so the cost is {price} × {least} = ${cost}."
            ],
            d_dist,
            follow_through=f"{price}*(part_c**2 + {4 * volume}/part_c)",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, parts=parts)


@generator("fy1ca_applied_optimisation_exam", level=19, presets={"exam": {}})
def fy1ca_applied_optimisation_exam(rng: random.Random) -> dict:
    """Open-top square-based tank of fixed volume: height at a trial side, surface there, best side, least cost."""
    return build_with_redraws(rng, _exam_build, "an optimisation exam chain")
