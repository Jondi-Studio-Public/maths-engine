"""Generates `limits-of-a-bivariate-claim` questions: what a fitted line can and cannot be trusted to say.

Two families of scenario:

  line and range   A line of fit is stated, with the range of x-values it was fitted to. The learner predicts a value
                   from it, then judges whether that prediction can be trusted (inside the range: interpolation; outside:
                   extrapolation), and, where the line predicts an impossible value, finds where the line first does.
  lurking variable Two measured variables rise together along a stated line. The learner predicts from the line, finds
                   the change in y for a stated change in x, and names the variable that most likely drives both.

Every multiple-choice part has five options in one sentence shape; each wrong option makes one named error.

Ladder (reasoning escalation, not size):
  easy:   a prediction inside the range of the data, then judge how far to trust it.
  medium: a prediction outside the range (or, for the other family, a prediction and the lurking variable).
  hard:   the prediction outside the range is impossible; find the x at which the line first reaches the limit, then say
          what the impossible value shows (or: prediction, change for a larger step, lurking variable).
  exam:   a four-part chain: the prediction, the x at the limit, how far the prediction overshoots (follow-through) and
          the reason; or the prediction, the change for a step, the lurking variable and the conclusion.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import attempts, check, generator

from ._ir2a_kit import NAMES, chain, mcq_part, number_part

_TIERS = ("easy", "medium", "hard")

_LINES = (
    dict(
        x="hours of study each week",
        y="exam mark (%)",
        who="students",
        bs=(3, 4, 5, 6),
        xb=(12, 17),
        lo=1,
        hi=8,
        limit=100,
        limit_note="an exam mark cannot be more than 100%",
        ylabel="mark",
    ),
    dict(
        x="age of a car in years",
        y="advertised price in hundreds of dollars",
        who="cars",
        bs=(-10, -12, -15, -18),
        xb=(15, 20),
        lo=1,
        hi=12,
        limit=0,
        limit_note="a price cannot be negative",
        ylabel="price",
    ),
    dict(
        x="age of a child in years",
        y="height in centimetres",
        who="children",
        bs=(4, 5, 6),
        xb=(20, 30),
        lo=2,
        hi=10,
        limit=200,
        limit_note="a person is not more than about 200 cm tall",
        ylabel="height",
    ),
    dict(
        x="daily maximum temperature in degrees Celsius",
        y="cups of hot chocolate sold",
        who="days",
        bs=(-4, -5, -6, -8),
        xb=(30, 40),
        lo=8,
        hi=26,
        limit=0,
        limit_note="a cafe cannot sell a negative number of cups",
        ylabel="cups sold",
    ),
    dict(
        x="age of a sunflower in weeks",
        y="height in centimetres",
        who="sunflowers",
        bs=(12, 14, 15, 18),
        xb=(20, 28),
        lo=2,
        hi=9,
        limit=310,
        limit_note="a sunflower does not grow taller than about 310 cm",
        ylabel="height",
    ),
)

_LURK = (
    dict(
        x="ice creams sold at a beach kiosk each day (tens)",
        y="people treated for sunburn each day",
        a=(2, 6),
        b=(2, 3),
        lo=2,
        hi=12,
        key="the daily maximum temperature",
        decoys=[
            "the number of lifeguards on duty",
            "the number of surfboards on the sand",
            "the price of an ice cream",
            "the length of the beach",
        ],
        short=("ice creams sold", "sunburn treatments"),
    ),
    dict(
        x="shoe size of a primary school child",
        y="score on a reading test",
        a=(5, 15),
        b=(6, 10),
        lo=1,
        hi=6,
        key="the age of the child",
        decoys=[
            "the colour of the child's school bag",
            "the number of books in the classroom",
            "the brand of the child's shoes",
            "the child's favourite sport",
        ],
        short=("shoe size", "reading score"),
    ),
    dict(
        x="number of fire trucks sent to a fire",
        y="damage in thousands of dollars",
        a=(10, 30),
        b=(30, 50),
        lo=1,
        hi=8,
        key="the size of the fire",
        decoys=[
            "the colour of the fire trucks",
            "the age of the fire chief",
            "the brand of hose used",
            "the colour of the fire station",
        ],
        short=("fire trucks sent", "damage"),
    ),
    dict(
        x="umbrellas sold in a city each day (hundreds)",
        y="road accidents reported in the city each day",
        a=(3, 8),
        b=(2, 4),
        lo=1,
        hi=9,
        key="the amount of rain that fell",
        decoys=[
            "the price of umbrellas",
            "the number of cars sold",
            "the number of letters posted",
            "the number of shops in the city",
        ],
        short=("umbrellas sold", "accidents"),
    ),
)

_DECOY_TAG = "ir2a-names-a-variable-with-no-link-to-both"


def _term(sc, x):
    a, b = sc["a"], sc["b"]
    sign = "+" if b > 0 else "-"
    return f"y = {a} {sign} {abs(b)}x"


def _line_text(sc):
    return _term(sc, None)


def _interp_part(sc, x):
    y = sc["a"] + sc["b"] * x
    return number_part(
        prompt=f"Use the line to predict the value of y when x is {x}.",
        exact=y,
        places=0,
        marks=2,
        working=[f"y = {sc['a']} {'+' if sc['b'] > 0 else '-'} {abs(sc['b'])} x {x} = {y}."],
        wrong=[
            (sc["b"] * x, "ir2a-forgets-the-intercept"),
            (sc["a"] - sc["b"] * x, "ir2a-flips-the-sign-of-the-slope-term"),
            (sc["a"] + sc["b"] + x, "ir2a-adds-instead-of-multiplying"),
            (sc["a"] * x + sc["b"], "ir2a-swaps-the-slope-and-intercept"),
        ],
        sensible_range=None,
    )


def _stem(sc, name):
    return (
        f"{name} collected data on {sc['who']}, where x is the {sc['x']} and y is the {sc['y']}, and fitted the line "
        f"{_line_text(sc)}. The data used had x-values from {sc['lo']} to {sc['hi']}."
    )


_RELI = {
    "inside": "Reasonably reliable: the value of x is inside the range of the data, so this is interpolation.",
    "outside": "Unreliable: the value of x is outside the range of the data, so the pattern may not continue.",
}


def _reliability_part(inside: bool, marks=1):
    if inside:
        key = _RELI["inside"]
        wrongs = [
            (
                "Unreliable: the value of x is outside the range of the data, so this is extrapolation.",
                "ir2a-calls-interpolation-extrapolation",
            ),
            (
                "Reliable: the line passes through every data point, so it is exact.",
                "ir2a-thinks-the-line-passes-through-every-point",
            ),
            (
                "Unreliable: the line can only predict values of x that appear in the data.",
                "ir2a-thinks-only-recorded-values-can-be-predicted",
            ),
            ("Unreliable: a line of fit can never be used to make a prediction.", "ir2a-rejects-all-predictions"),
        ]
    else:
        key = _RELI["outside"]
        wrongs = [
            (
                "Reasonably reliable: the value of x is inside the range of the data, so this is interpolation.",
                "ir2a-calls-extrapolation-interpolation",
            ),
            (
                "Reliable: the predicted value is a sensible number, so the line must still hold.",
                "ir2a-trusts-a-sensible-looking-prediction",
            ),
            (
                "Unreliable: the line does not pass through the point where x and y are both 0.",
                "ir2a-rejects-a-prediction-for-the-wrong-reason",
            ),
            (
                "Unreliable: the predicted value is not one of the y-values in the data.",
                "ir2a-rejects-a-prediction-for-the-wrong-reason",
            ),
            (
                "Reliable: the line fits the data well, so it fits at every value of x.",
                "ir2a-assumes-the-line-holds-everywhere",
            ),
        ]
    return mcq_part(
        "How far can the prediction be trusted?",
        key,
        wrongs,
        marks=marks,
        working=["Check whether the value of x lies between the smallest and largest x-values in the data."],
        keep=4,
    )


def _impossible_part(marks=2):
    key = (
        "The line cannot keep going that far: the relationship must change before then, so the prediction is not valid."
    )
    wrongs = [
        (
            "The data must contain an error, because a correct line never gives an impossible value.",
            "ir2a-blames-the-data-for-an-impossible-value",
        ),
        (
            "The slope must be wrong and should be smaller, so that the value stays possible.",
            "ir2a-blames-the-slope-for-an-impossible-value",
        ),
        (
            "The prediction is valid: a line has to be followed wherever it leads.",
            "ir2a-assumes-the-line-holds-everywhere",
        ),
        (
            "The prediction is valid, because the line fits the data in the range it was fitted to.",
            "ir2a-trusts-a-sensible-looking-prediction",
        ),
    ]
    return mcq_part(
        "What does the impossible value show?",
        key,
        wrongs,
        marks=marks,
        working=["A line fitted to a range of x-values describes that range only."],
        keep=4,
    )


def _lurk_part(sc, marks=2):
    wrongs = [(d, _DECOY_TAG) for d in sc["decoys"]]
    return mcq_part(
        f"Which of these most likely drives both the {sc['short'][0]} and the {sc['short'][1]}?",
        sc["key"],
        wrongs,
        marks=marks,
        working=[f"{sc['key'].capitalize()} would raise both, so neither one has to cause the other."],
        keep=4,
    )


def _conclusion_part(sc, marks=1):
    a, b = sc["short"]
    key = (
        f"The two move together, but a rise in {a} does not have to cause a rise in {b}: something else may drive both."
    )
    wrongs = [
        (f"A rise in {a} causes a rise in {b}, because the line has a positive slope.", "ir2a-reads-cause-from-a-line"),
        (f"A rise in {b} causes a rise in {a}, because the two are linked.", "ir2a-reverses-cause-and-effect"),
        (
            f"There is no link between {a} and {b}, because a line of fit cannot show one.",
            "ir2a-rejects-all-predictions",
        ),
        (
            f"The link is a coincidence that disappears when the data are collected again.",
            "ir2a-calls-a-link-a-coincidence",
        ),
    ]
    return mcq_part(
        "Which conclusion is justified?", key, wrongs, marks=marks, working=["Association does not show cause."], keep=4
    )


def _pick_line(rng):
    base = rng.choice(_LINES)
    for _ in attempts(500, "a line that reaches its limit at a whole x"):
        b = rng.choice(base["bs"])
        xb = rng.randint(*base["xb"])
        a = base["limit"] - b * xb
        y_lo, y_hi = a + b * base["lo"], a + b * base["hi"]
        room = abs(base["limit"] - y_hi)
        if a > 0 and y_lo > 0 and y_hi > 0 and room >= max(8, abs(b) * 3):
            break
    sc = dict(base, a=a, b=b)
    sc["xbreak"] = xb
    return sc, rng.choice(NAMES)


def _x_out(rng, sc, impossible=True):
    """An x beyond the data: past the limit's x when `impossible`, else short of it (a sensible-looking value)."""
    if impossible:
        return rng.randint(sc["xbreak"] + 2, sc["xbreak"] + 8)
    return rng.randint(sc["hi"] + 2, sc["xbreak"] - 1)


def _draw_lurk(rng):
    base = rng.choice(_LURK)
    return dict(base, a=rng.randint(*base["a"]), b=rng.randint(*base["b"]))


def _break_x(sc):
    x = sc["xbreak"]
    check(sc["a"] + sc["b"] * x == sc["limit"], "limit must be reached at a whole x")
    return x


def _easy(rng):
    sc, name = _pick_line(rng)
    x = rng.randint(sc["lo"] + 1, sc["hi"] - 1)
    return chain(_stem(sc, name), _interp_part(sc, x), _reliability_part(True))


def _lurk_stem(sc, name):
    return (
        f"{name} recorded the {sc['x']} and the {sc['y']} on a number of occasions. Let x be the first and y the second, "
        f"and let {_line_text(sc)} be the line fitted to the data. The data used had x-values from {sc['lo']} to {sc['hi']}."
    )


def _medium(rng):
    if rng.random() < 0.5:
        sc, name = _pick_line(rng)
        x = _x_out(rng, sc, impossible=False)
        return chain(_stem(sc, name), _interp_part(sc, x), _reliability_part(False))
    sc = _draw_lurk(rng)
    name = rng.choice(NAMES)
    x = rng.randint(sc["lo"] + 1, sc["hi"] - 1)
    return chain(_lurk_stem(sc, name), _interp_part(sc, x), _lurk_part(sc))


def _break_part(sc, marks=3):
    xb = _break_x(sc)
    a, b = sc["a"], sc["b"]
    return number_part(
        prompt=f"For what value of x does the line first predict a y-value of {sc['limit']}?",
        exact=xb,
        places=0,
        marks=marks,
        working=[f"{a} {'+' if b > 0 else '-'} {abs(b)}x = {sc['limit']}", f"x = {xb}."],
        wrong=[
            (Fraction(sc["limit"], abs(b)), "ir2a-forgets-the-intercept"),
            (xb + 1, "ir2a-solves-for-one-step-past"),
            ((sc["limit"] + a) // b if b else 0, "ir2a-adds-the-intercept-instead-of-subtracting"),
            (xb - 1, "ir2a-solves-for-one-step-short"),
        ],
        sensible_range=None,
    )


def _hard(rng):
    if rng.random() < 0.6:
        sc, name = _pick_line(rng)
        x = _x_out(rng, sc)
        pred = _interp_part(sc, x)
        return chain(_stem(sc, name) + f" {sc['limit_note'].capitalize()}.", pred, _break_part(sc), _impossible_part())
    sc = _draw_lurk(rng)
    name = rng.choice(NAMES)
    x = rng.randint(sc["lo"] + 1, sc["hi"] - 1)
    step = rng.choice([4, 5, 10])
    y0 = sc["a"] + sc["b"] * x
    pb = number_part(
        prompt=f"What does the line predict for y when x is {step} greater than {x}?",
        exact=y0 + sc["b"] * step,
        places=0,
        marks=2,
        working=[
            f"Each increase of {step} in x adds {sc['b']} x {step} = {sc['b'] * step} to y.",
            f"{y0} + {sc['b'] * step} = {y0 + sc['b'] * step}.",
        ],
        wrong=[
            (sc["b"] * step, "ir2a-gives-the-change-not-the-prediction"),
            (y0 + step, "ir2a-adds-instead-of-multiplying"),
            (sc["a"] + sc["b"] * step, "ir2a-gives-a-prediction-not-a-change"),
        ],
        follow_through=f"part_a + {sc['b'] * step}",
        sensible_range=None,
    )
    return chain(_lurk_stem(sc, name), _interp_part(sc, x), pb, _lurk_part(sc))


@generator("ir2a_limits_of_a_bivariate_claim", level=11, presets={t: dict(tier=t) for t in _TIERS})
def ir2a_limits_of_a_bivariate_claim(rng: random.Random, tier: str = "easy") -> dict:
    check(tier in _TIERS, f"unknown tier {tier!r}")
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)


@generator("exd_x_ir2a_limits_of_a_bivariate_claim", level=12, presets={"exam": dict(tier="exam")})
def exd_x_ir2a_limits_of_a_bivariate_claim(rng: random.Random, tier: str = "exam") -> dict:
    """Prediction, x at the limit, overshoot and reason; or prediction, change for a step, lurking variable, conclusion."""
    check(tier == "exam", "only the exam tier exists")
    if rng.random() < 0.5:
        sc, name = _pick_line(rng)
        x = _x_out(rng, sc)
        y = sc["a"] + sc["b"] * x
        pa = _interp_part(sc, x)
        pb = _break_part(sc, marks=2)
        over = abs(y - sc["limit"])
        pc = number_part(
            prompt=f"By how much does the predicted y-value go past {sc['limit']}?",
            exact=over,
            places=0,
            marks=2,
            working=[f"|{y} - {sc['limit']}| = {over}."],
            wrong=[
                (y + sc["limit"], "ir2a-adds-instead-of-subtracting"),
                (abs(y), "ir2a-reports-the-prediction-not-the-excess"),
                (abs(x - _break_x(sc)), "ir2a-gives-the-excess-in-x-not-y"),
            ],
            follow_through=f"Abs(part_a - {sc['limit']})",
            sensible_range=(0, None),
        )
        stem = _stem(sc, name) + f" {sc['limit_note'].capitalize()}."
        return chain(stem, pa, pb, pc, _impossible_part(marks=1))
    sc = _draw_lurk(rng)
    name = rng.choice(NAMES)
    x = rng.randint(sc["lo"] + 1, sc["hi"] - 1)
    step = rng.choice([4, 5, 10])
    y0 = sc["a"] + sc["b"] * x
    pb = number_part(
        prompt=f"What does the line predict for y when x is {step} greater than {x}?",
        exact=y0 + sc["b"] * step,
        places=0,
        marks=2,
        working=[
            f"Each increase of {step} in x adds {sc['b']} x {step} = {sc['b'] * step} to y.",
            f"{y0} + {sc['b'] * step} = {y0 + sc['b'] * step}.",
        ],
        wrong=[
            (sc["b"] * step, "ir2a-gives-the-change-not-the-prediction"),
            (y0 + step, "ir2a-adds-instead-of-multiplying"),
            (sc["a"] + sc["b"] * step, "ir2a-gives-a-prediction-not-a-change"),
        ],
        follow_through=f"part_a + {sc['b'] * step}",
        sensible_range=None,
    )
    return chain(_lurk_stem(sc, name), _interp_part(sc, x), pb, _lurk_part(sc, 1), _conclusion_part(sc, 1))
