"""Exam questions on fitted lines: two topics sharing this module.

* `exd_x_s5_fitted_line_threshold` -- the fitted line is given either as an equation in a time variable
  measured from a base year, or only as two (year, value) points it passes through. The learner must
  recover the line when it is not given, solve an inequality for the time variable, and turn the
  solution into the first calendar year in which the prediction crosses a stated value. The answer is
  a whole year, so "round up, not to nearest" and "add the base year back" both matter.

* `exd_x_s5_best_fit_from_summary` -- the line of best fit is not given at all. The learner is told the
  mean and standard deviation of each variable and the correlation coefficient, and must build the
  least-squares line (gradient `r * s_y / s_x`, through the point of means) before predicting.

Both are built backwards from exact rationals so the key is exact. The second is graded as a typed
number to one decimal place; its draws are required to land exactly on the tenths grid so the key is
never a rounded value, and every distractor must clear half a tenth.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..core import check, generator, phrase, pick_distractors, stated_precision_string, stated_precision_tolerance

# ---------------------------------------------------------------------------
# Threshold year from a fitted line
# ---------------------------------------------------------------------------

# (quantity, unit text, rising?) -- a rising quantity "first exceeds" a level, a falling one "first falls below".
_CONTEXTS = (
    ("the number of subscribers to a streaming service", "thousand subscribers", True),
    ("the average price of a house in a suburb", "thousand dollars", True),
    ("the annual visitors to a national park", "thousand visitors", True),
    ("the number of paper tickets sold at a station", "hundred tickets", False),
    ("the annual rainfall at a weather station", "mm", False),
    ("the percentage of households with a landline", "percent", False),
)

_THRESHOLD_EQUATION = (
    "A line fitted to data predicts {quantity}, in {unit}, as V = {m} t + {c}, where t is the number of "
    "years after {base}. Assuming the line continues to hold, in which calendar year does the predicted "
    "value first {verb} {level} {unit}?",
    "For {quantity}, in {unit}, a fitted line is V = {m} t + {c} with t measured in years since {base}. "
    "If the trend continues, what is the first calendar year in which the predicted value is {adj} "
    "{level}?",
)
_THRESHOLD_POINTS = (
    "A line fitted to data passes through the points ({y1}, {v1}) and ({y2}, {v2}), where the first "
    "coordinate is the calendar year and the second is {quantity} in {unit}. If the fitted line continues "
    "to hold, in which calendar year does the predicted value first {verb} {level} {unit}?",
    "Using a line fitted to {quantity} (in {unit}), the predicted value was {v1} in {y1} and {v2} in {y2}. "
    "Assuming the same line applies in later years, what is the first calendar year in which the "
    "prediction is {adj} {level}?",
)


def _num(x: Fraction) -> str:
    """A tenths-grid value as the shortest decimal string (no trailing zero)."""
    check((x * 10).denominator == 1, "fitline: value off the tenths grid")
    s = f"{float(x):.1f}"
    return s[:-2] if s.endswith(".0") else s


def _threshold_build(rng: random.Random) -> dict:
    for _ in range(500):
        quantity, unit, rising = rng.choice(_CONTEXTS)
        m = Fraction(rng.randint(5, 40), 10) * (1 if rising else -1)
        c = Fraction(rng.randint(50, 600), 10)
        base = rng.choice((1990, 1995, 2000, 2005, 2010, 2015))
        use_points = rng.random() < 0.5
        # t = last_before does NOT yet cross the level; given points always lie before the crossing
        last_before = rng.randint(6 if use_points else 2, 24)
        # the level sits strictly between the values at last_before and last_before + 1
        step = Fraction(rng.randint(1, int(abs(m) * 10) - 1), 10) * (1 if rising else -1)
        if abs(step) >= abs(m) or step == 0:
            continue
        level = m * last_before + c + step
        if level <= 0 or (level * 10).denominator != 1:
            continue
        t_star = (level - c) / m
        check(last_before < t_star < last_before + 1, "fitline: crossing time is not between the two years")
        first_t = last_before + 1  # first whole t past the crossing
        key = base + first_t

        verb = "exceed" if rising else "fall below"
        adj = "greater than" if rising else "less than"
        if use_points:
            t1 = rng.randint(0, 3)
            t2 = rng.randint(t1 + 3, last_before)
            text = phrase(
                rng,
                _THRESHOLD_POINTS,
                y1=base + t1,
                v1=_num(m * t1 + c),
                y2=base + t2,
                v2=_num(m * t2 + c),
                quantity=quantity,
                unit=unit,
                verb=verb,
                adj=adj,
                level=_num(level),
            )
            working = [
                f"Gradient = ({_num(m * t2 + c)} - {_num(m * t1 + c)}) / ({base + t2} - {base + t1}) = {_num(m)} per year.",
                f"Using t = years after {base}: V = {_num(m)} t + {_num(c)}.",
            ]
        else:
            text = phrase(
                rng,
                _THRESHOLD_EQUATION,
                quantity=quantity,
                unit=unit,
                m=_num(m),
                c=_num(c),
                base=base,
                verb=verb,
                adj=adj,
                level=_num(level),
            )
            working = [f"V = {_num(m)} t + {_num(c)}, with t in years after {base}."]
        working += [
            f"Solve {_num(m)} t + {_num(c)} = {_num(level)}: t = {float(t_star):.3f}.",
            f"The first whole year past the crossing has t = {first_t}, so the year is {base} + {first_t} = {key}.",
        ]

        wrong = [
            base + last_before,  # last year that has NOT yet crossed (rounds the crossing down)
            first_t,  # forgets to add the base year back
            key + 1,  # takes the year after the first crossing
            base + math.floor(t_star + Fraction(1, 2)),  # rounds the crossing time to nearest instead of up
        ]
        answer_text = str(key)
        distractors = pick_distractors(answer_text, [str(w) for w in wrong])
        if len(distractors) < 3:
            continue

        # Independent check: scan years forward until the prediction crosses the level.
        t = 0
        while (m * t + c <= level) if rising else (m * t + c >= level):
            t += 1
        check(base + t == key, "fitline: scan disagrees with the closed-form crossing year")
        return dict(
            text=text,
            latex=f"plain:{text}",
            answer=key,
            answer_text=answer_text,
            answer_latex=answer_text,
            distractors=distractors[:3],
            working=working,
        )
    raise RuntimeError("exd_x_s5_fitted_line_threshold: no draw satisfied the constraints in 500 attempts")


@generator("exd_x_s5_fitted_line_threshold", level=12, presets={"exam": dict(tier="exam")})
def exd_x_s5_fitted_line_threshold(rng: random.Random, tier: str = "exam") -> dict:
    """First calendar year in which a fitted line crosses a stated level."""
    check(tier == "exam", "exd_x_s5_fitted_line_threshold: only the exam tier exists")
    return _threshold_build(rng)


# ---------------------------------------------------------------------------
# Line of best fit from summary statistics
# ---------------------------------------------------------------------------

# (x description, y description, sign of the correlation in real data)
_PAIRS = (
    ("weekly hours of study", "exam mark", 1),
    ("daily maximum temperature (degrees Celsius)", "number of ice creams sold", 1),
    ("age of a car (years)", "resale value (thousand dollars)", -1),
    ("weekly advertising spend (hundred dollars)", "weekly sales (thousand dollars)", 1),
    ("daily maximum temperature (degrees Celsius)", "number of hot drinks sold", -1),
)

_BESTFIT_TEMPLATES = (
    "For a sample, x is the {xd} and y is the {yd}. The mean of x is {xb} and its standard deviation is {sx}; the mean "
    "of y is {yb} and its standard deviation is {sy}. The correlation coefficient is r = {r}. The line of "
    "best fit of y on x has gradient r multiplied by s_y / s_x and passes through the point of means. Use it to "
    "predict y when x = {x0}, giving your answer to 1 decimal place.",
    "Summary statistics for the {xd} (x) and the {yd} (y): mean of x = {xb}, standard deviation of x = {sx}, mean of "
    "y = {yb}, standard deviation of y = {sy}, correlation coefficient r = {r}. Find the least-squares "
    "regression line of y on x and use it to predict y when x = {x0}. Give your answer to 1 decimal place.",
    "A study of the {xd} (x) and the {yd} (y) gives r = {r}. For x the mean is {xb} and the standard deviation is "
    "{sx}; for y the mean is {yb} and the standard deviation is {sy}. Predict y when x = {x0} from the "
    "least-squares line of y on x, correct to 1 decimal place.",
)


def _dec(x: Fraction, places: int = 2) -> str:
    s = f"{float(x):.{places}f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def _bestfit_build(rng: random.Random) -> dict:
    for _ in range(2000):
        a, b, sign = _PAIRS[rng.randrange(len(_PAIRS))]
        r = Fraction(rng.choice((4, 5, 6, 7, 8, 9)), 10) * sign
        sx = rng.choice((2, 4, 5, 8, 10, 12, 15, 20))
        sy = rng.choice((3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30))
        m = r * sy / sx
        if (m * 100).denominator != 1 or m == 0 or abs(m) == abs(r):
            continue
        xb = rng.randint(10, 60)
        yb = rng.randint(30, 90)
        d = rng.choice((-1, 1)) * rng.randint(1, 15)
        x0 = xb + d
        if x0 <= 0:
            continue
        y0 = yb + m * d
        if y0 <= 0 or (y0 * 10).denominator != 1:
            continue
        # Principled wrong predictions.
        inv = r * sx / sy  # gradient inverted: regresses x on y
        wrong = {
            "inverted gradient": yb + inv * d,
            "gradient taken as r": yb + r * d,
            "line through the origin": m * x0,
            "forgets to centre x": yb + m * x0,
        }
        tol = stated_precision_tolerance(1)
        if any(abs(w - y0) <= tol for w in wrong.values()):
            continue
        if len({round(w, 2) for w in wrong.values()}) < 4:
            continue
        text = phrase(
            rng,
            _BESTFIT_TEMPLATES,
            xd=a,
            yd=b,
            xb=xb,
            yb=yb,
            sx=sx,
            sy=sy,
            r=_dec(r),
            x0=x0,
        )
        answer_text = stated_precision_string(y0, 1)

        # Independent check via the equivalent form y = intercept + m x.
        intercept = yb - m * xb
        check(intercept + m * x0 == y0, "bestfit: slope-intercept form disagrees with point-slope form")
        working = [
            f"Gradient = r times s_y / s_x = {_dec(r)} times {sy} / {sx} = {_dec(m)}.",
            f"The line passes through ({xb}, {yb}): y - {yb} = {_dec(m)} (x - {xb}).",
            f"At x = {x0}: y = {yb} + {_dec(m)} times ({d}) = {answer_text}.",
        ]
        cands = [stated_precision_string(round_tenth(w), 1) for w in wrong.values()]
        distractors = pick_distractors(answer_text, cands, tolerance=tol)
        if len(distractors) < 3:
            continue
        return dict(
            text=text,
            latex=f"plain:{text}",
            answer=float(y0),
            answer_text=answer_text,
            answer_latex=answer_text,
            distractors=distractors[:3],
            working=working,
        )
    raise RuntimeError("exd_x_s5_best_fit_from_summary: no draw satisfied the constraints in 2000 attempts")


def round_tenth(x: Fraction) -> Fraction:
    """Half-up rounding to one decimal place for a non-negative `Fraction`."""
    return Fraction(math.floor(x * 10 + Fraction(1, 2)), 10)


@generator("exd_x_s5_best_fit_from_summary", level=12, presets={"exam": dict(tier="exam")})
def exd_x_s5_best_fit_from_summary(rng: random.Random, tier: str = "exam") -> dict:
    """Predict y from the least-squares line built from means, standard deviations and r."""
    check(tier == "exam", "exd_x_s5_best_fit_from_summary: only the exam tier exists")
    return _bestfit_build(rng)
