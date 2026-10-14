"""Mean, variance and standard deviation of aX + b: `r3a_linear_transform` and its exam chain.

Skill `expectation-variance-linear-transform`.

Tiers (topic `r3a_linear_transform`):

* easy   -- Y = aX + b is a stated rule in a context (a pay rule, a profit, a countdown with a negative a); the
  mean or the variance of Y from the mean or the variance of X;
* medium -- a change of units (Celsius to Fahrenheit, centimetres to inches, minutes to seconds, a shift of origin):
  the mean, the standard deviation or the variance of the new measure, where a shift moves the mean and leaves the
  spread alone;
* hard   -- X is given by a table of four outcomes, so its mean and variance must be found first: the standard
  deviation of aX + b, or E(aX^2 + bX + c).

The exam chain (topic `exd_x_r3a_linear_transform`) is (a) E(X) from the table, (b) Var(X) from the table, (c) E(Y)
for Y = aX + b from (a), (d) SD(Y) from (b). Parts (b) to (d) carry `follow_through`.

Every key is derived on the exact distribution of Y itself: Y's values and probabilities are listed and its moments
summed, a route that uses none of the rules for linear transforms.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import attempts, check, generator, round_half_up
from ._r3a_kit import build_until, clear_of_boundary, number_part, single, sqrt_fraction


def _moments(values, probs):
    mean = sum((v * p for v, p in zip(values, probs)), Fraction(0))
    second = sum((v * v * p for v, p in zip(values, probs)), Fraction(0))
    return mean, second - mean * mean, second


def _sign_b(b) -> str:
    return f"+ {b}" if b >= 0 else f"- {-b}"


def _poly(a, b, c) -> str:
    out = f"{a}X^2"
    for coeff, var in ((b, "X"), (c, "")):
        if coeff:
            out += f" {'+' if coeff > 0 else '-'} {abs(coeff) if (abs(coeff) != 1 or not var) else ''}{var}"
    return out


def _rule(a, b) -> str:
    coeff = "-" if a == -1 else "" if a == 1 else f"{a}"
    text = f"{coeff}X"
    return text if b == 0 else f"{text} {_sign_b(b)}"


# (context sentence with {m} {s}, rule sentence with {rule}, name of Y)
_RULES = (
    (
        "The number of parcels X that a courier delivers in a day has mean {m} and {spread}.",
        "The courier is paid ${a} for each parcel plus a daily allowance of ${b}, so the daily pay in dollars is "
        "Y = {rule}.",
        (3, 7),
        (20, 60),
    ),
    (
        "The number of coffees X that a cafe sells in a day has mean {m} and {spread}.",
        "Each coffee brings in ${a} after ingredient costs, and fixed costs are ${b} a day, so the daily profit in "
        "dollars is Y = {rule}.",
        (2, 5),
        (-300, -120),
    ),
    (
        "The number of reports X that an analyst finishes in a week has mean {m} and {spread}.",
        "Each report takes {a_abs} minutes of a {b} minute weekly allocation, so the minutes left over are Y = {rule}.",
        (-25, -10),
        (480, 900),
    ),
    (
        "The number of tickets X sold per show has mean {m} and {spread}.",
        "The venue keeps ${a} per ticket and pays ${b} to hire the hall, so the profit per show in dollars is "
        "Y = {rule}.",
        (6, 14),
        (-900, -350),
    ),
)


def _draw_rule(rng: random.Random):
    sentence, rule_text, a_range, b_range = rng.choice(_RULES)
    a = rng.randint(*a_range)
    b = rng.randint(*b_range)
    return sentence, rule_text, a, b


# -- easy ------------------------------------------------------------------------------------------------------


def _easy(rng: random.Random) -> dict:
    sentence, rule_text, a, b = _draw_rule(rng)
    ask = rng.choice(("mean", "variance"))
    mean = Fraction(rng.randint(40, 400), rng.choice((1, 2, 4)))
    sd = Fraction(rng.randint(10, 80), rng.choice((1, 2)))
    var_x = sd * sd
    rule = _rule(a, b)
    spread = f"standard deviation {float(sd):g}" if ask == "mean" else f"variance {float(var_x):g}"
    context = sentence.format(m=f"{float(mean):g}", spread=spread)
    rule_sentence = rule_text.format(a=abs(a), a_abs=abs(a), b=abs(b), rule=rule)
    stem = f"{context} {rule_sentence}"
    if ask == "mean":
        value = a * mean + b
        part = number_part(
            "Find E(Y), correct to 2 decimal places.",
            value,
            2,
            1,
            [f"E(Y) = a E(X) + b = {a} x {float(mean):g} {_sign_b(b)}."],
            [
                (a * mean, "r3a-linear-mean-omits-the-shift"),
                (mean + b, "r3a-linear-mean-omits-the-multiplier"),
                (a * (mean + b), "r3a-linear-mean-multiplies-the-shift"),
            ],
        )
    else:
        value = a * a * var_x
        part = number_part(
            "Find Var(Y), correct to 2 decimal places.",
            value,
            2,
            1,
            [f"Var(Y) = a^2 Var(X) = {a}^2 x {float(var_x):g}; the shift {b} does not change the spread."],
            [
                (a * var_x, "r3a-linear-variance-omits-the-square"),
                (a * a * var_x + b, "r3a-linear-variance-adds-the-shift"),
                (abs(a) * sd, "r3a-linear-variance-scales-the-sd-only"),
            ],
        )
    return single(f"{stem} {part['prompt']}", part)


# -- medium ----------------------------------------------------------------------------------------------------

# (description of C, unit of C, new unit, a, b, plausible mean range, sd range, decimals on the mean)
_UNITS = (
    (
        "The daily maximum temperature C in a town has mean {m} degrees Celsius and standard deviation {s} degrees. "
        "The temperature in degrees Fahrenheit is F = 1.8C + 32.",
        Fraction(9, 5),
        32,
        (12, 34),
        (2, 6),
    ),
    (
        "The length L of a timber offcut has mean {m} centimetres and standard deviation {s} centimetres. A supplier "
        "quotes lengths in millimetres, M = 10L.",
        Fraction(10),
        0,
        (40, 220),
        (3, 14),
    ),
    (
        "The time T that a customer waits in a queue has mean {m} minutes and standard deviation {s} minutes. The "
        "system logs waits in seconds, S = 60T.",
        Fraction(60),
        0,
        (3, 12),
        (1, 4),
    ),
    (
        "The water temperature C in a pool has mean {m} degrees Celsius and standard deviation {s} degrees. A sensor "
        "reports kelvin, K = C + 273.15.",
        Fraction(1),
        Fraction(27315, 100),
        (22, 30),
        (1, 3),
    ),
    (
        "The mass W of a parcel has mean {m} kilograms and standard deviation {s} kilograms. The courier charges "
        "$4.50 plus $2.20 per kilogram, so the charge in dollars is Q = 2.2W + 4.5.",
        Fraction(11, 5),
        Fraction(9, 2),
        (3, 18),
        (1, 5),
    ),
)


def _medium(rng: random.Random) -> dict:
    text, a, b, m_range, s_range = rng.choice(_UNITS)
    ask = rng.choice(("mean", "sd", "variance"))
    for _ in attempts(500, "a mean and standard deviation whose converted value rounds cleanly"):
        mean = Fraction(rng.randint(m_range[0] * 10, m_range[1] * 10), 10)
        sd = Fraction(rng.randint(s_range[0] * 10 + 1, s_range[1] * 10), 10)
        if clear_of_boundary(a * a * sd * sd, 2) and clear_of_boundary(a * sd, 2):
            break
    stem = text.format(m=float(mean), s=float(sd))
    shift = b
    if ask == "mean":
        value, label = a * mean + shift, "mean"
        slips = [
            (a * mean, "r3a-linear-mean-omits-the-shift"),
            (mean + shift, "r3a-linear-mean-omits-the-multiplier"),
            (a * (mean + shift), "r3a-linear-mean-multiplies-the-shift"),
        ]
        working = [f"The mean converts like a single value: {float(a):g} x {float(mean):g} + {float(shift):g}."]
    elif ask == "sd":
        value, label = abs(a) * sd, "standard deviation"
        slips = [
            (a * sd + shift, "r3a-linear-sd-applies-the-shift"),
            (a * a * sd, "r3a-linear-sd-squares-the-multiplier"),
            (sd, "r3a-linear-sd-unchanged-by-scaling"),
            (a * a * sd * sd, "r3a-linear-sd-reports-the-variance"),
        ]
        working = [f"SD(aX + b) = |a| SD(X) = {float(a):g} x {float(sd):g}; the shift does not change the spread."]
    else:
        value, label = a * a * sd * sd, "variance"
        slips = [
            (a * sd * sd, "r3a-linear-variance-omits-the-square"),
            (a * sd * sd + shift, "r3a-linear-variance-adds-the-shift"),
            (a * sd, "r3a-linear-variance-scales-the-sd-only"),
        ]
        working = [f"Var(aX + b) = a^2 Var(X) = {float(a):g}^2 x {float(sd * sd):g}."]
    # the transformed distribution itself: two equally likely points one standard deviation either side of the mean
    points = [mean - sd, mean + sd]
    half = [Fraction(1, 2)] * 2
    t_mean, t_var, _ = _moments([a * v + shift for v in points], half)
    check(t_mean == a * mean + shift and t_var == a * a * sd * sd, "the transformed distribution disagrees")
    dp = 2
    part = number_part(
        f"Find the {label} of the new measure, correct to {dp} decimal places.", value, dp, 2, working, slips
    )
    return single(f"{stem} {part['prompt']}", part)


# -- hard ------------------------------------------------------------------------------------------------------


def _draw_table(rng: random.Random):
    for _ in attempts(500, "four outcomes"):
        values = sorted(rng.sample(range(0, 7), 4))
        cuts = sorted(rng.sample(range(1, 20), 3))
        counts = [cuts[0], cuts[1] - cuts[0], cuts[2] - cuts[1], 20 - cuts[2]]
        if max(counts) <= 10 and len(set(counts)) >= 3:
            return values, [Fraction(c, 20) for c in counts]
    raise AssertionError("unreachable")  # pragma: no cover


def _table_text(values, probs) -> str:
    return ", ".join(f"P(X = {v}) = {float(p):.2f}" for v, p in zip(values, probs))


def _hard(rng: random.Random) -> dict:
    variant = rng.choice(("sd", "quadratic"))
    for _ in attempts(1000, "a table whose answer sits clear of a rounding boundary"):
        values, probs = _draw_table(rng)
        mean, var, second = _moments(values, probs)
        a = rng.choice((-4, -3, -2, 2, 3, 4, 5))
        b = rng.randint(-20, 30)
        if variant == "sd":
            ys = [a * v + b for v in values]
            _ym, yvar, _ys2 = _moments(ys, probs)
            check(yvar == a * a * var, "listing Y's values disagrees with a^2 Var(X)")
            value = sqrt_fraction(yvar)
            ok = clear_of_boundary(value, 2)
        else:
            c = rng.randint(-15, 15)
            ys = [a * v * v + b * v + c for v in values]
            value = sum((y * p for y, p in zip(ys, probs)), Fraction(0))
            check(value == a * second + b * mean + c, "listing Y's values disagrees with the linear rule for E")
            ok = True
        if ok and var > Fraction(1, 2):
            break
    table = _table_text(values, probs)
    if variant == "sd":
        stem = f"A random variable X has the distribution {table}. Let Y = {_rule(a, b)}."
        part = number_part(
            "Find the standard deviation of Y, correct to 2 decimal places.",
            value,
            2,
            3,
            [
                f"E(X) = {float(mean):.2f}, E(X^2) = {float(second):.2f}, so Var(X) = {float(var):.4f}.",
                f"SD(Y) = |{a}| x sqrt({float(var):.4f}).",
            ],
            [
                (sqrt_fraction(abs(a) * var), "r3a-linear-sd-omits-the-square-in-the-variance"),
                (abs(a) * sqrt_fraction(second), "r3a-variance-without-subtracting-the-mean-squared"),
                (abs(a) * var, "r3a-linear-sd-omits-the-square-root"),
                (a * sqrt_fraction(var) + b, "r3a-linear-sd-applies-the-shift"),
            ],
        )
    else:
        quad = _poly(a, b, c)
        stem = f"A random variable X has the distribution {table}. Find E({quad})."
        part = number_part(
            "Give your answer correct to 2 decimal places.",
            value,
            2,
            3,
            [
                f"E(X) = {float(mean):.2f}, E(X^2) = {float(second):.2f}, so E = {a} x {float(second):.2f} {_sign_b(b)} x {float(mean):.2f} {_sign_b(c)}."
            ],
            [
                (a * mean * mean + b * mean + c, "r3a-second-moment-squares-the-mean"),
                (a * second + b * mean, "r3a-linear-mean-omits-the-shift"),
                (a * second + b * mean * mean + c, "r3a-second-moment-squares-the-mean"),
                (a * mean + b * mean + c, "r3a-second-moment-is-the-mean"),
            ],
        )
    return single(f"{stem} {part['prompt']}", part)


@generator(
    "r3a_linear_transform",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3a_linear_transform(rng: random.Random, tier: str = "easy") -> dict:
    """E, Var and SD of aX + b: a stated rule (easy), a change of units (medium), a table first (hard)."""
    if tier == "easy":
        return build_until(rng, _easy)
    if tier == "medium":
        return build_until(rng, _medium)
    if tier == "hard":
        return build_until(rng, _hard)
    raise ValueError(f"unknown tier {tier!r}")


# -- exam chain ------------------------------------------------------------------------------------------------

_EXAM_CONTEXTS = (
    (
        "A workshop records the number of faults X found in a batch of machine parts.",
        "The repair bill for a batch is Y = {rule} dollars.",
        (20, 60),
        (40, 150),
    ),
    (
        "A bakery records the number of cakes X returned unsold at the end of a day.",
        "The loss on a day is Y = {rule} dollars.",
        (3, 9),
        (10, 80),
    ),
    (
        "A school records the number of late arrivals X in a class each morning.",
        "The time lost in minutes is Y = {rule}.",
        (2, 6),
        (5, 30),
    ),
)


def _exam(rng: random.Random) -> dict:
    """E(X) and Var(X) from a table, then E(aX + b) and SD(aX + b)."""
    intro, rule_text, a_range, b_range = rng.choice(_EXAM_CONTEXTS)
    for _ in attempts(3000, "a table whose chain carries forward cleanly"):
        values, probs = _draw_table(rng)
        mean, var, second = _moments(values, probs)
        a = rng.randint(*a_range)
        b = rng.randint(*b_range)
        mean_y = a * mean + b
        sd_y = a * sqrt_fraction(var)
        var_2 = round_half_up(var, 2)
        sd_from_b = a * sqrt_fraction(var_2)
        if not (clear_of_boundary(var, 2) and clear_of_boundary(sd_y, 2) and var > Fraction(1, 2)):
            continue
        if round_half_up(sd_y, 2) != round_half_up(sd_from_b, 2):
            continue
        break
    ys = [a * v + b for v in values]
    ym, yvar, _ = _moments(ys, probs)
    check(ym == mean_y and yvar == a * a * var, "Y's own distribution disagrees with the linear rules")
    stem = f"{intro} Its distribution is {_table_text(values, probs)}. {rule_text.format(rule=_rule(a, b))}"
    parts = [
        number_part(
            "Find E(X), correct to 2 decimal places.",
            mean,
            2,
            1,
            [f"E(X) = sum of x P(X = x) = {float(mean):.2f}."],
            [
                (sum(values, Fraction(0)) / 4, "r3a-expectation-averages-the-outcomes"),
                (second, "r3a-expectation-squares-the-outcomes"),
                (sum(probs[i] * (i + 1) for i in range(4)), "r3a-expectation-uses-positions-not-values"),
            ],
            sensible_range=(min(values), max(values)),
        ),
        number_part(
            "Find Var(X), correct to 2 decimal places.",
            var,
            2,
            2,
            [f"Var(X) = E(X^2) - (E(X))^2 = {float(second):.2f} - {float(mean):.2f}^2 = {float(var):.4f}."],
            [
                (second, "r3a-variance-without-subtracting-the-mean-squared"),
                (second - mean, "r3a-variance-subtracts-the-mean-not-its-square"),
                (sqrt_fraction(var), "r3a-variance-reported-as-sd"),
            ],
            follow_through=f"{second.numerator}/{second.denominator} - part_a**2",
            sensible_range=(0, None),
        ),
        number_part(
            "Find E(Y), correct to 2 decimal places.",
            mean_y,
            2,
            1,
            [f"E(Y) = {a} E(X) {_sign_b(b)} = {float(mean_y):.2f}."],
            [
                (a * mean, "r3a-linear-mean-omits-the-shift"),
                (mean + b, "r3a-linear-mean-omits-the-multiplier"),
                (a * (mean + b), "r3a-linear-mean-multiplies-the-shift"),
            ],
            follow_through=f"{a}*part_a + {b}",
            sensible_range=(0, None),
        ),
        number_part(
            "Find the standard deviation of Y, correct to 2 decimal places.",
            sd_y,
            2,
            2,
            [f"SD(Y) = {a} x SD(X) = {a} x sqrt({float(var):.4f}) = {float(sd_y):.2f}."],
            [
                (sqrt_fraction(a * var), "r3a-linear-sd-omits-the-square-in-the-variance"),
                (a * a * var, "r3a-linear-sd-squares-the-multiplier"),
                (a * sqrt_fraction(var) + b, "r3a-linear-sd-applies-the-shift"),
                (sqrt_fraction(var), "r3a-linear-sd-unchanged-by-scaling"),
            ],
            follow_through=f"{a}*sqrt(part_b)",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, parts=parts)


@generator("exd_x_r3a_linear_transform", level=12, presets={"exam": dict(tier="exam")})
def exd_x_r3a_linear_transform(rng: random.Random, tier: str = "exam") -> dict:
    """E(X) and Var(X) from a table, then E(aX + b) and SD(aX + b)."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return build_until(rng, _exam)
