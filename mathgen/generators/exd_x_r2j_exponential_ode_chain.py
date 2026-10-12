"""Exam chain on dy/dt = k y: the constant from two readings, the value at another time, the rate there, and the
time to reach a stated amount.

The stem gives y(0) = y0 and y(t1) = y1 (growth or decay). Four parts:

(a) k = ln(y1 / y0) / t1, as an exact expression;
(b) y(t2) = y0 e^(k t2), to 1 decimal place;
(c) the rate of change at t2, dy/dt = k y(t2), to 1 decimal place;
(d) the time at which y reaches Y, t = ln(Y / y0) / k, to 2 decimal places.

Each key is worked out from exact logarithms and rounded once. The draw is kept only when (c), worked from the
rounded (b), displays the same answer.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction
from math import gcd

from sympy import sympify

from mathgen.answers import Distractor
from mathgen.core import check, draw_until, generator, latex_answer, phrase

from .exd_x_r2j_support import clear_of_boundary, number_part, shown

_CONTEXTS = [
    ("The mass y grams of a radioactive sample", "grams", "hour", "decay"),
    ("The number y (in thousands) of bacteria in a culture", "thousand bacteria", "hour", "growth"),
    ("The amount y mg of a drug in a patient's bloodstream", "mg", "hour", "decay"),
    ("The population y of a fish colony in a lake", "fish", "month", "either"),
    ("The balance y dollars of an account earning interest continuously", "dollars", "year", "growth"),
]

_STEM = [
    "{what} satisfies the differential equation dy/dt = ky, where k is a constant and t is the time in {tunit}s. "
    "Initially y = {y0}, and after {t1} {tunit}s y = {y1}.",
    "{what} changes at a rate proportional to the amount present, so dy/dt = ky for a constant k, with t in "
    "{tunit}s. At t = 0, y = {y0}. At t = {t1}, y = {y1}.",
    "For {what_lower}, dy/dt = ky, where t is measured in {tunit}s and k is a constant. The initial value is "
    "y = {y0}, and y = {y1} when t = {t1}.",
]


def _ratio_text(num: int, den: int) -> str:
    return f"{num}" if den == 1 else f"{num}/{den}"


def _draw(rng: random.Random, direction: str):
    y0 = rng.randrange(20, 900, 5)
    growth = direction == "growth" or (direction == "either" and rng.random() < 0.5)
    factor = rng.uniform(1.4, 3.6) if growth else rng.uniform(0.25, 0.75)
    y1 = max(5, round(y0 * factor / 5) * 5)
    t1 = rng.randint(2, 8)
    t2 = rng.choice([t for t in range(1, 13) if t != t1])
    if growth:
        target = rng.randint(int(y0 * 1.2), int(y1 * 1.8))
    else:
        target = rng.randint(max(2, int(y1 * 0.5)), int(y0 * 0.85))
    return y0, y1, t1, t2, target


def _accept(draw) -> bool:
    y0, y1, t1, t2, target = draw
    if y1 == y0 or target in (y0, y1) or not (0.2 < y1 / y0 < 5):
        return False
    if (target > y0) != (y1 > y0) or (y1 > y0 and target <= y0) or (y1 < y0 and target >= y0):
        return False
    k = math.log(y1 / y0) / t1
    yb = y0 * math.exp(k * t2)
    rate = k * yb
    time = math.log(target / y0) / k
    if not all(clear_of_boundary(Fraction(v), p) for v, p in ((yb, 1), (rate, 1), (time, 2))):
        return False
    return shown(Fraction(k) * shown(Fraction(yb), 1), 1) == shown(Fraction(rate), 1) and yb < 5000


def _numeric(text: str) -> float:
    """The value of a constant expression written with ln, + - * / and brackets (a cheap stand-in for grading)."""
    return float(sympify(text.replace("ln", "log")))


def _expression_part(*, prompt, text, marks, working, wrong):
    """An `expression` part: the key and tagged wrong expressions. A wrong expression is kept only when it differs
    from the key in value (the test file grades every one with the real grader)."""
    key = _numeric(text)
    distractors, seen = [], {text}
    for candidate, tag in wrong:
        if candidate in seen or abs(_numeric(candidate) - key) < 1e-9:
            continue
        seen.add(candidate)
        distractors.append(Distractor(candidate, tag, candidate))
        if len(distractors) == 3:
            break
    check(len(distractors) >= 2, "fewer than two usable expression distractors")
    return dict(
        prompt=prompt,
        answer_kind="expression",
        answer=text,
        answer_text=text,
        answer_latex=latex_answer(text),
        marks=marks,
        working=working,
        distractors=distractors,
    )


@generator("exd_x_r2j_exponential_ode_chain", level=16, presets={"exam": dict(tier="exam")})
def exd_x_r2j_exponential_ode_chain(rng: random.Random, tier: str = "exam") -> dict:
    """k from two readings, then the value, the rate and the time to reach a stated amount."""
    check(tier == "exam", "exd_x_r2j_exponential_ode_chain: only the exam tier exists")
    what, unit, tunit, direction = rng.choice(_CONTEXTS)
    y0, y1, t1, t2, target = draw_until(
        rng, lambda g: _draw(g, direction), _accept, what="a stable exponential-ode chain"
    )
    stem = phrase(rng, _STEM, what=what, what_lower=what[0].lower() + what[1:], y0=y0, y1=y1, t1=t1, tunit=tunit)

    g = gcd(y1, y0)
    r_num, r_den = y1 // g, y0 // g
    k = math.log(y1 / y0) / t1
    y2 = y0 * math.exp(k * t2)
    rate = k * y2
    time = math.log(target / y0) / k
    check(abs(y2 - y0 * (y1 / y0) ** (t2 / t1)) < 1e-9 * y2, "y(t2) disagrees with the ratio route")
    check(abs(y0 * math.exp(k * time) - target) < 1e-9 * target, "the time found does not reach the target")

    k_text = f"ln({_ratio_text(r_num, r_den)})/{t1}"
    ln_ratio = f"ln({_ratio_text(r_num, r_den)})"
    ft_y = f"{y0}*exp(part_a*{t2})"
    ft_rate = "part_a*part_b"
    ft_time = f"log({_ratio_text(target // gcd(target, y0), y0 // gcd(target, y0))})/part_a"
    check(abs(float(target // gcd(target, y0)) / (y0 // gcd(target, y0)) - target / y0) < 1e-12, "target ratio text")

    parts = [
        _expression_part(
            prompt="Find the exact value of k.",
            text=k_text,
            marks=2,
            working=[
                f"y = {y0}e^(kt). Using y({t1}) = {y1}: e^({t1}k) = {_ratio_text(r_num, r_den)}.",
                f"k = {k_text}.",
            ],
            wrong=[
                (ln_ratio, "forgot-to-divide-by-time"),
                (f"({y1} - {y0})/{t1}", "average-rate-for-k"),
                (f"ln({y1})/{t1}", "log-of-value-not-ratio"),
                (f"{t1}*ln({_ratio_text(r_num, r_den)})", "multiplies-instead-of-dividing"),
            ],
        ),
        number_part(
            prompt=f"Hence find the value of y when t = {t2}, correct to 1 decimal place.",
            exact=Fraction(y2),
            places=1,
            marks=1,
            working=[f"y = {y0}e^(kt) with k = {k_text}.", f"y({t2}) = {y2:.1f}."],
            wrong=[
                (Fraction(y0 + (y1 - y0) * t2 / t1), "linear-instead-of-exponential"),
                (Fraction(y0 * math.exp(k)), "forgot-time-in-exponent"),
                (Fraction(y1 * math.exp(k * t2)), "wrong-starting-value"),
                (Fraction(y0 * math.exp(-k * t2)), "sign-error-in-exponent"),
            ],
            follow_through=ft_y,
            sensible_range=(0, None),
        ),
        number_part(
            prompt=f"Hence find the rate of change of y when t = {t2}, correct to 1 decimal place.",
            exact=Fraction(rate),
            places=1,
            marks=2,
            working=[f"dy/dt = ky, so at t = {t2} the rate is k y({t2}) = {rate:.1f} {unit} per {tunit}."],
            wrong=[
                (Fraction(k * y0), "rate-at-start-for-rate-now"),
                (Fraction(y2), "evaluated-function-not-derivative"),
                (Fraction(-rate), "dropped-negative-from-derivative"),
                (Fraction(k * y2 * t2), "multiplied-by-time-once-more"),
            ],
            follow_through=ft_rate,
        ),
        number_part(
            prompt=f"Find the time at which y reaches {target}, correct to 2 decimal places.",
            exact=Fraction(time),
            places=2,
            marks=2,
            working=[f"{y0}e^(kt) = {target}, so t = ln({target}/{y0}) / k = {time:.2f}."],
            wrong=[
                (Fraction(math.log(target / y0)), "forgot-to-divide-by-rate"),
                (Fraction(math.log(target) / k), "log-of-value-not-ratio"),
                (Fraction((target - y0) / (y1 - y0) * t1), "linear-instead-of-exponential"),
                (Fraction(t1 * target / y1), "proportional-instead-of-exponential"),
            ],
            follow_through=ft_time,
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, parts=parts)
