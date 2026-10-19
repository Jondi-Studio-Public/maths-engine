"""Generators for curriculum expansion batches 2–6."""

from decimal import Decimal
from fractions import Fraction
from math import comb, factorial, sqrt

from .. import figures
from ..answers import Distractor
from ..core import (
    VerificationError,
    attempts,
    generator,
    latex_answer,
    pick_distractors,
    rational_text,
    round_half_up,
    stated_precision_string,
)
from ..figure_kinds import space
from . import _space_kit as kit
from .primary_math import _aligned


def _number(text, answer, candidates, *, latex=None, working=None, aligned=False, places=None):
    if aligned and not isinstance(answer, Fraction):
        shown, candidates = _aligned(answer, candidates, places)
    else:
        shown = rational_text(answer) if isinstance(answer, Fraction) else str(answer)
    structured = {"p": answer.numerator, "q": answer.denominator} if isinstance(answer, Fraction) else answer
    d = dict(
        text=text,
        answer=structured,
        answer_text=shown,
        answer_latex=shown,
        working=working or [],
        distractors=pick_distractors(shown, candidates),
    )
    if latex is not None:
        d["latex"] = latex
    return d


def _mcq(text, answer, candidates, *, latex=None, working=None):
    d = dict(
        text=text,
        answer=answer,
        answer_text=answer,
        answer_latex=answer,
        working=working or [],
        distractors=pick_distractors(answer, candidates),
    )
    if latex is not None:
        d["latex"] = latex
    return d


# Batch 2: primary geometry, time and money


_ANGLE_CLASSES = ("Acute", "Right", "Obtuse", "Straight", "Reflex")
_ANGLE_STEMS = (
    "Classify an angle measuring {angle}°.",
    "What type of angle measures {angle}°?",
    "An angle turns through {angle}°. Is it acute, right, obtuse, straight or reflex?",
    "Which name describes an angle of {angle}°?",
)


def angle_type(angle: int) -> str:
    """The name of an angle's size class; only exactly 90 is right and exactly 180 straight."""
    if angle < 90:
        return "Acute"
    if angle == 90:
        return "Right"
    if angle < 180:
        return "Obtuse"
    if angle == 180:
        return "Straight"
    return "Reflex"


@generator("classify_angle", level=6, presets={"easy": {}, "medium": {"near_boundary": True}})
def classify_angle(rng, near_boundary=False):
    """All five names are always offered, so no option gives the answer away by being the odd one out.

    By default the angle is drawn from anywhere in the class, 10 to 350 degrees. With ``near_boundary`` it is drawn
    within ten degrees of 90 or 180 (acute against right against obtuse, and obtuse against straight against
    reflex), where the measure has to be compared with the boundary rather than judged by eye."""
    if near_boundary:
        centre = rng.choice((90, 180))
        angle = centre + rng.randint(-10, 10)
    else:
        name = rng.choice(_ANGLE_CLASSES)
        angle = {
            "Acute": rng.randint(10, 85),
            "Right": 90,
            "Obtuse": rng.randint(95, 175),
            "Straight": 180,
            "Reflex": rng.randint(185, 350),
        }[name]
    answer = angle_type(angle)
    d = _mcq(
        rng.choice(_ANGLE_STEMS).format(angle=angle),
        answer,
        [x for x in _ANGLE_CLASSES if x != answer],
        latex=rf"{angle}^\circ",
        working=[f"An angle of {angle}° is {answer.lower()}."],
    )
    d["distractors"] = pick_distractors(answer, [x for x in _ANGLE_CLASSES if x != answer], n=4)
    d["answer_latex"] = latex_answer(answer, plain=True)
    return d


@generator("straight_line_angle", level=7)
def straight_line_angle(rng):
    """Two angles side by side on a straight line, drawn (not to scale) with one marked x. A right angle is not
    drawn (it would need its own mark), and a draw is redone if the answer's digits show inside the other angle's
    label."""
    for _ in attempts(80, "a straight-line picture that does not show its own answer"):
        known = rng.randint(20, 160)
        if known == 90:
            continue
        answer = 180 - known
        first = rng.randrange(2)  # which of the two angles is the unknown one
        true = [answer, known] if first == 0 else [known, answer]
        try:
            sweeps = kit.drawn_sweeps(rng, true, hide=[first])
        except VerificationError:
            continue
        letter = kit.pick_unknown(rng)
        fig = space.angles_on_line(
            sweeps=sweeps,
            labels=[letter, kit.degrees(known)] if first == 0 else [kit.degrees(known), letter],
            tilt=rng.randrange(360),
            alt="Two angles side by side on a straight line, one of them marked with a letter.",
        )
        if not kit.leaks(fig, str(answer)):
            break
    d = _number(
        f"{rng.choice(kit.LINE_OPENERS)} {rng.choice(kit.SIZE_CLOSERS).format(u=letter)}",
        answer,
        # the given angle itself; 90 as the total (the size off a right angle either way round); 360 as the total;
        # a slip of ten. All positive: a negative angle is never a plausible option.
        [known, abs(90 - known), 360 - known, answer + 10],
        working=[f"Angles on a straight line total 180°, so 180 − {known} = {answer}°."],
    )
    d["figure"] = fig
    return d


figures.register_reader("straight_line_angle", lambda q: str(kit.read_missing_angle(q.figure)))


def _clock12(total_minutes):
    total_minutes %= 24 * 60
    hour24, minute = divmod(total_minutes, 60)
    suffix = "am" if hour24 < 12 else "pm"
    hour = hour24 % 12 or 12
    return f"{hour}:{minute:02d} {suffix}"


@generator("time_system_conversion", level=6)
def time_system_conversion(rng):
    hour = rng.randint(0, 23)
    minute = rng.choice((0, 5, 10, 15, 20, 30, 45, 50, 55))
    total = hour * 60 + minute
    if rng.choice((True, False)):
        prompt, answer = f"Convert {_clock12(total)} to 24-hour time.", f"{hour:02d}:{minute:02d}"
        alternatives = [f"{(hour + shift) % 24:02d}:{minute:02d}" for shift in (12, 1, -1)]
    else:
        prompt, answer = f"Convert {hour:02d}:{minute:02d} to 12-hour time.", _clock12(total)
        alternatives = [_clock12(total + shift * 60) for shift in (12, 1, -1)]
    d = _mcq(
        prompt,
        answer,
        alternatives,
        working=[f"The same instant is written as {hour:02d}:{minute:02d} or {_clock12(total)}."],
    )
    d["answer_latex"] = latex_answer(answer, plain=True)  # "2:05 pm" would render as italic prose
    return d


@generator("elapsed_time", level=6)
def elapsed_time(rng, duration_hi=180):
    start = rng.randint(6 * 12, 18 * 12) * 5
    duration = rng.randint(3, max(3, duration_hi // 5)) * 5
    finish = start + duration
    return _number(
        f"How many minutes pass from {_clock12(start)} to {_clock12(finish)}?",
        duration,
        [finish - start + 60, abs((finish % 60) - (start % 60)), duration - 5, duration + 5],
        working=[f"Count from {_clock12(start)} to {_clock12(finish)}: {duration} minutes."],
    )


@generator("money_change", level=5)
def money_change(rng, price_hi=80):
    price_cents = rng.randint(5, price_hi * 20) * 5
    paid_dollars = (price_cents + 999) // 100
    paid_dollars += rng.randint(0, 2) * 5
    change = round(paid_dollars - price_cents / 100, 2)
    price = price_cents / 100
    return _number(
        f"An item costs ${price:.2f}. You pay ${paid_dollars:.2f}. How much change do you receive?",
        change,
        [paid_dollars + price, price, change + 1, max(0, change - 1)],
        aligned=True,
        places=2,
        working=[f"${paid_dollars:.2f} − ${price:.2f} = ${change:.2f}."],
    )


@generator("financial_estimation", level=5)
def financial_estimation(rng):
    prices = [rng.randint(120, 499) / 100 for _ in range(3)]
    rounded = [int(p + 0.5) for p in prices]
    answer = sum(rounded)
    listing = ", ".join(f"${p:.2f}" for p in prices)
    return _number(
        f"Estimate the total of {listing} by rounding each price to the nearest dollar.",
        answer,
        [round(sum(prices), 2), answer - 1, answer + 1, sum(int(p) for p in prices)],
        aligned=True,
        working=[f"Round to {', '.join('$' + str(x) for x in rounded)}; the estimated total is ${answer}."],
    )


@generator("l_shape_perimeter", level=7)
def l_shape_perimeter(rng, side_hi=20):
    width, height = rng.randint(5, side_hi), rng.randint(5, side_hi)
    notch_w, notch_h = rng.randint(1, width - 2), rng.randint(1, height - 2)
    sides = (width, height - notch_h, notch_w, notch_h, width - notch_w, height)
    answer = sum(sides)
    return _number(
        f"An L-shaped boundary has consecutive side lengths {', '.join(map(str, sides))} m. Find its perimeter.",
        answer,
        [width * height, 2 * (width + height) - notch_w - notch_h, sum(sides[:-1]), answer + notch_w],
        working=[f"Add all six boundary lengths: {' + '.join(map(str, sides))} = {answer} m."],
    )


@generator("rectangular_prism_volume", level=7)
def rectangular_prism_volume(rng, side_hi=12):
    length, width, height = (rng.randint(2, side_hi) for _ in range(3))
    answer = length * width * height
    return _number(
        f"A rectangular prism is {length} cm by {width} cm by {height} cm. Find its volume in cm³.",
        answer,
        [
            length + width + height,
            2 * (length * width + length * height + width * height),
            length * width,
            answer + height,
        ],
        working=[f"Volume = {length} × {width} × {height} = {answer} cm³."],
    )


# Batch 3: statistics and probability


@generator("classify_data", level=7)
def classify_data(rng):
    cases = (
        ("students' favourite fruit", "Categorical", "names or labels"),
        ("the number of siblings each student has", "Discrete numerical", "counted whole-number values"),
        ("students' heights measured in centimetres", "Continuous numerical", "measured values on a scale"),
    )
    text, answer, reason = rng.choice(cases)
    d = _mcq(
        f"Classify the data: {text}.",
        answer,
        [x for x in ("Categorical", "Discrete numerical", "Continuous numerical") if x != answer],
        working=[f"These are {reason}, so the data are {answer.lower()}."],
    )
    d["answer_latex"] = latex_answer(answer, plain=True)
    return d


@generator("summary_statistic", level=7)
def summary_statistic(rng, kind="mean", count=5, hi=20):
    if kind == "mean":
        centre = rng.randint(4, hi - 4)
        a, b = rng.randint(1, 3), rng.randint(1, 3)
        values = [centre - a, centre + a, centre - b, centre + b, centre]
        answer = centre
        working = [f"The total is {sum(values)}; divide by {len(values)} to get {answer}."]
    elif kind == "median":
        values = sorted(rng.randint(1, hi) for _ in range(count if count % 2 else count + 1))
        answer = values[len(values) // 2]
        working = [f"Order the values; the middle value is {answer}."]
    elif kind == "mode":
        answer = rng.randint(3, hi - 3)
        others = [answer - 2, answer - 1, answer + 1, answer + 2]
        values = others + [answer, answer, answer]
        rng.shuffle(values)
        working = [f"{answer} occurs most often."]
    elif kind == "range":
        values = [rng.randint(1, hi) for _ in range(count)]
        answer = max(values) - min(values)
        working = [f"Range = maximum − minimum = {max(values)} − {min(values)} = {answer}."]
    else:
        raise ValueError(f"unknown statistic: {kind}")
    return _number(
        f"Find the {kind} of: {', '.join(map(str, values))}.",
        answer,
        [sum(values), max(values), min(values), answer + 1],
        working=working,
    )


@generator("simple_probability", level=7)
def simple_probability(rng):
    favourable, other = rng.randint(1, 8), rng.randint(1, 8)
    total = favourable + other
    answer = Fraction(favourable, total)
    return _number(
        f"A bag has {favourable} red {'counter' if favourable == 1 else 'counters'} and "
        f"{other} blue {'counter' if other == 1 else 'counters'}. What is P(red)?",
        answer,
        [
            Fraction(other, total),
            Fraction(favourable, other) if favourable < other else Fraction(favourable, total + 1),
            Fraction(1, total),
            Fraction(favourable, total + 1),
            Fraction(other, total + 1),
        ],
        working=[f"P(red) = favourable ÷ total = {favourable}/{total} = {rational_text(answer)}."],
    )


@generator("expected_frequency", level=8)
def expected_frequency(rng):
    denominator = rng.choice((2, 4, 5, 10))
    numerator = rng.randint(1, denominator - 1)
    trials = denominator * rng.randint(5, 20)
    answer = trials * numerator // denominator
    return _number(
        f"An event has probability {numerator}/{denominator}. How many times is it expected in {trials} trials?",
        answer,
        [trials // denominator, trials * numerator, trials - answer, answer + denominator],
        working=[f"Expected frequency = {trials} × {numerator}/{denominator} = {answer}."],
    )


@generator("probability_representation", level=7)
def probability_representation(rng):
    value = rng.choice((Fraction(1, 10), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(4, 5)))
    percent, decimal = int(value * 100), float(value)
    answer = f"{rational_text(value)}, {decimal}, {percent}%"
    candidates = [
        f"{rational_text(value)}, {decimal * 10:g}, {percent}%",
        f"{rational_text(value)}, {decimal:g}, {percent + 10}%",
        f"{rational_text(value)}, {decimal / 10:g}, {percent}%",
    ]
    d = _mcq(
        "Which row gives three equivalent probabilities?",
        answer,
        candidates,
        working=[f"{rational_text(value)} = {decimal} = {percent}%."],
    )
    d["answer_latex"] = latex_answer(answer, plain=True)
    return d


# Batch 4: secondary number


@generator("factor_list", level=7)
def factor_list(rng, hi=60):
    n = rng.randint(12, hi)
    factors = [x for x in range(1, n + 1) if n % x == 0]
    text = ", ".join(map(str, factors))
    distractors = [
        ", ".join(map(str, factors[:-1])),
        ", ".join(map(str, factors + [n + 1])),
        ", ".join(map(str, factors[1:])),
    ]
    return dict(
        text=f"List all positive factors of {n}.",
        latex=rf"\text{{Factors of }}{n}",
        answer=factors,
        answer_text=text,
        answer_latex=text,
        working=[f"The positive divisors of {n} are {text}."],
        distractors=distractors,
    )


def _prime_factors(n):
    factors, divisor = [], 2
    while divisor * divisor <= n:
        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        divisor += 1
    if n > 1:
        factors.append(n)
    return factors


@generator("prime_factorisation", level=7)
def prime_factorisation(rng):
    primes = (2, 3, 5, 7)
    factors = sorted(rng.choices(primes, k=rng.randint(2, 4)))
    n = 1
    for factor in factors:
        n *= factor
    text = " × ".join(map(str, factors))
    return _mcq(
        f"Which is the prime factorisation of {n}?",
        text,
        [f"{factors[0]} × {n // factors[0] + 1}", str(n), f"1 × {n}"],
        working=[f"Dividing by primes gives {n} = {text}."],
    )


def _positional_text(coefficient_text: str, exponent: int) -> str:
    """The ordinary (never exponential) decimal for coefficient * 10**exponent,
    with trailing zeros trimmed. `coefficient_text` is already a short decimal
    string (e.g. "3.1"), so `Decimal.scaleb` only moves the point -- it never
    rounds -- giving an exact "0.000031" or "3800000" instead of Python's `:g`,
    which switches to exponent notation once the value's own exponent goes
    below -4 or reaches 6, defeating a question that asks the learner to
    produce exactly that notation themselves."""
    text = format(Decimal(coefficient_text).scaleb(exponent), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


@generator("scientific_notation", level=9)
def scientific_notation(rng):
    coefficient = rng.randint(10, 99) / 10
    exponent = rng.choice((-5, -4, -3, 3, 4, 5, 6))
    coefficient_text = f"{coefficient:g}"
    ordinary = _positional_text(coefficient_text, exponent)
    answer = f"{coefficient_text} × 10^{exponent}"
    return _mcq(
        f"Write {ordinary} in scientific notation.",
        answer,
        [
            f"{coefficient * 10:g} × 10^{exponent - 1}",
            f"{coefficient:g} × 10^{-exponent}",
            f"{coefficient / 10:g} × 10^{exponent}",
        ],
        working=[f"Move the decimal point to make a coefficient from 1 to 10: {answer}."],
    )


@generator("classify_real_number", level=9)
def classify_real_number(rng):
    cases = (
        ("√2", "Irrational"),
        ("π", "Irrational"),
        ("0.375", "Rational"),
        ("-7/9", "Rational"),
        ("√49", "Rational"),
    )
    shown, answer = rng.choice(cases)
    d = _mcq(
        f"Is {shown} rational or irrational?",
        answer,
        ["Irrational" if answer == "Rational" else "Rational", "Not a real number"],
        working=[f"{shown} is {answer.lower()}."],
    )
    d["answer_latex"] = latex_answer(answer, plain=True)
    return d


@generator("recurring_decimal_fraction", level=8)
def recurring_decimal_fraction(rng):
    digit = rng.randint(1, 8)
    answer = Fraction(digit, 9)
    return _number(
        f"Write 0.{digit}{digit}{digit}... as a fraction in lowest terms.",
        answer,
        [Fraction(digit, 10), Fraction(digit, 99), Fraction(1, digit), Fraction(9, digit)],
        latex=rf"\text{{Write }} 0.\overline{{{digit}}} \text{{ as a fraction in lowest terms.}}",
        working=[
            f"A single recurring digit gives {digit}/9"
            + (f", which simplifies to {rational_text(answer)}." if rational_text(answer) != f"{digit}/9" else ".")
        ],
    )


# Australian settings for `percentage_change`, each a singular quantity that can sit between 40 and 600. The prefix is
# the unit written before the number.
_PERCENTAGE_CHANGE_CONTEXTS = (
    ("the weekly rent on a unit in Brisbane", "$"),
    ("a family's weekly grocery bill in Adelaide", "$"),
    ("the price of a season pass to the public pool in Darwin", "$"),
    ("the number of members of a surf life saving club on the Gold Coast", ""),
    ("the number of daily visitors to a national park in Tasmania", ""),
)


def _decimal_option(value: Fraction) -> tuple[str, int | float]:
    """A distractor's text and value: whole numbers as they are, anything else to one decimal place."""
    if value.denominator == 1:
        return str(value.numerator), value.numerator
    shown = stated_precision_string(round_half_up(value, 1), 1)
    return shown, float(shown)


@generator("percentage_change", level=8)
def percentage_change(rng, reverse=False):
    percent = rng.choice((10, 20, 25, 50))
    subject, unit = rng.choice(_PERCENTAGE_CHANGE_CONTEXTS)
    if reverse:
        # A step that keeps both the final value and the percentage taken off it whole, so no option is a decimal
        # beside a whole-number answer (a decimal there would give the answer away). 100 is left out: from it,
        # taking the percentage off as a number lands back on the answer, leaving one wrong option.
        step, low, high = {10: (100, 2, 7), 20: (25, 2, 16), 25: (16, 3, 25), 50: (20, 2, 20)}[percent]
        original = rng.choice([k * step for k in range(low, high + 1) if k * step != 100])
    else:
        original = rng.randint(2, 20) * 20
    final = original * (100 + percent) // 100
    if reverse:
        if percent == 10 and rng.random() < 0.5:
            text = f"A price including 10% GST is ${final}. What was the price before GST was added?"
        else:
            text = f"After a {percent}% increase, {subject} is {unit}{final}. What was it before the increase?"
        # Misconceptions: took the percentage off the final value; took the percentage off as a number.
        slips = [
            (Fraction(final * (100 - percent), 100), "subtracts-percent"),
            (Fraction(final - percent), "percent-as-absolute"),
        ]
        cands = []
        for value, tag in slips:
            shown, number = _decimal_option(value)
            cands.append(Distractor(shown, tag, number))
        return dict(
            text=text,
            answer=original,
            answer_text=str(original),
            answer_latex=str(original),
            answer_kind="number",
            working=[f"Divide by {1 + percent / 100:g}: {final} ÷ {1 + percent / 100:g} = {original}."],
            distractors=pick_distractors(str(original), cands),
        )
    text = f"{subject[0].upper()}{subject[1:]} rose from {unit}{original} to {unit}{final}. What is the percentage increase?"
    # Misconceptions: gave the change itself; gave the new value as a percentage of the old; divided by the new value.
    slips = [
        (Fraction(final - original), "gave-change-not-percent"),
        (Fraction(final * 100, original), "new-as-percent-of-original"),
        # Rounded to a whole percent like the answer, so the option's form does not mark it as wrong.
        (Fraction(round_half_up(Fraction((final - original) * 100, final), 0)), "divides-by-final"),
    ]
    cands = []
    for value, tag in slips:
        shown, number = _decimal_option(value)
        cands.append(Distractor(shown, tag, number))
    return dict(
        text=text,
        answer=percent,
        answer_text=str(percent),
        answer_latex=str(percent),
        answer_kind="number",
        working=[f"Increase = {final - original}; ({final - original}/{original}) × 100 = {percent}%."],
        distractors=pick_distractors(str(percent), cands),
    )


@generator("logarithm_law", level=11)
def logarithm_law(rng):
    cases = (("log(a) + log(b)", "log(ab)"), ("log(a) − log(b)", "log(a/b)"), ("k log(a)", "log(a^k)"))
    prompt, answer = rng.choice(cases)
    d = _mcq(
        f"Which expression is equivalent to {prompt}?",
        answer,
        ["log(a+b)", "log(a−b)", "log(a)/log(b)", "k+log(a)"],
        working=[f"Apply the logarithm laws: {prompt} = {answer}."],
    )
    d["answer_latex"] = latex_answer(answer)
    return d


# Batch 5: financial and discrete mathematics


@generator("earnings", level=11)
def earnings(rng, kind="wages"):
    rate = rng.randint(18, 45)
    hours = rng.randint(10, 40)
    if kind == "wages":
        answer = rate * hours
        text = f"A worker earns ${rate} per hour for {hours} hours. Find the gross wage."
        working = [f"Gross wage = {rate} × {hours} = ${answer}."]
    else:
        overtime = rng.randint(2, 10)
        multiplier = rng.choice((1.5, 2))
        answer = rate * hours + rate * multiplier * overtime
        text = f"A worker earns ${rate}/h for {hours} regular hours plus {overtime} hours at {multiplier:g}× rates. Find gross pay."
        working = [
            f"Regular pay plus overtime pay = ${rate * hours} + ${rate * multiplier * overtime:g} = ${answer:g}."
        ]
    return _number(
        text,
        answer,
        [rate + hours, rate * (hours + (overtime if kind != "wages" else 0)), answer + rate, answer - rate],
        aligned=True,
        working=working,
    )


@generator("financial_interest", level=11)
def financial_interest(rng, kind="simple"):
    principal = rng.randint(10, 50) * 100
    rate = rng.choice((2, 3, 4, 5, 6, 8))
    years = rng.randint(2, 5)
    if kind == "simple":
        answer = principal * rate * years / 100
        text = f"Find the simple interest on ${principal} at {rate}% p.a. for {years} years."
        working = [f"I = Prt = {principal} × {rate / 100:g} × {years} = ${answer:g}."]
    else:
        answer = round(principal * (1 + rate / 100) ** years, 2)
        text = f"Find the balance of ${principal} invested at {rate}% p.a., compounded annually for {years} years. Give your answer to the nearest cent."
        working = [f"A = P(1+r)^n = {principal}(1+{rate / 100:g})^{years} = ${answer:.2f}."]
    return _number(
        text,
        answer,
        [principal * rate / 100, principal + rate * years, principal, round(answer * 1.1, 2)],
        aligned=True,
        places=2 if kind != "simple" else None,
        working=working,
    )


@generator("unit_pricing", level=8)
def unit_pricing(rng):
    quantity = rng.choice((2, 4, 5, 10))
    unit = rng.randint(2, 20) / 2
    price = quantity * unit
    return _number(
        f"A pack of {quantity} items costs ${price:.2f}. What is the price per item?",
        unit,
        [price, price / (quantity + 1), quantity / price, unit + 1],
        aligned=True,
        places=2,
        working=[f"Unit price = ${price:.2f} ÷ {quantity} = ${unit:.2f}."],
    )


@generator("flat_loan_repayment", level=11)
def flat_loan_repayment(rng):
    principal = rng.randint(10, 40) * 1000
    annual_rate = rng.choice((3, 4, 5, 6))
    years = rng.choice((2, 3, 4, 5))
    months = years * 12
    total = principal * (1 + annual_rate * years / 100)
    answer = round(total / months, 2)
    return _number(
        f"A ${principal} flat-interest loan charges {annual_rate}% of principal per year for {years} years. Find the monthly repayment to the nearest cent.",
        answer,
        [round(principal / months, 2), round(total / years, 2), round(answer * 12, 2), round(answer * 1.1, 2)],
        aligned=True,
        places=2,
        working=[f"Total repayable is ${total:.2f}; divide by {months} months to get ${answer:.2f}."],
    )


@generator("moving_average", level=11)
def moving_average(rng, window=3):
    values = [rng.randint(10, 40) for _ in range(window)]
    total = sum(values)
    remainder = total % window
    values[-1] += (window - remainder) % window
    answer = sum(values) / window
    return _number(
        f"Find the {window}-point moving average for {', '.join(map(str, values))}.",
        answer,
        [sum(values), max(values), min(values), answer + 1],
        aligned=True,
        working=[f"Add the {window} values and divide by {window}: {sum(values)} ÷ {window} = {answer:g}."],
    )


@generator("shortest_path", level=11)
def shortest_path(rng):
    ab, bd, ac, cd, bc = (rng.randint(2, 12) for _ in range(5))
    routes = {"A-B-D": ab + bd, "A-C-D": ac + cd, "A-B-C-D": ab + bc + cd, "A-C-B-D": ac + bc + bd}
    answer = min(routes.values())
    edges = f"AB={ab}, BD={bd}, AC={ac}, CD={cd}, BC={bc}"
    return _number(
        f"A weighted network has edges {edges}. Find the shortest distance from A to D.",
        answer,
        list(routes.values()),
        working=["Compare the total weight of each possible simple route from A to D."],
    )


# Batch 6: senior mathematics


@generator("combinatorics", level=12)
def combinatorics(rng, kind="multiplication"):
    n, r = rng.randint(5, 10), rng.randint(2, 4)
    if kind == "multiplication":
        a, b, c = rng.randint(2, 8), rng.randint(2, 8), rng.randint(2, 8)
        answer, text = (
            a * b * c,
            f"An outfit has {a} shirt choices, {b} trouser choices and {c} shoe choices. How many outfits?",
        )
        working = [f"Use the multiplication principle: {a} × {b} × {c} = {answer}."]
    elif kind == "permutation":
        answer, text = (
            factorial(n) // factorial(n - r),
            f"How many ordered arrangements of {r} objects can be chosen from {n} distinct objects?",
        )
        working = [f"{n}P{r} = {answer}."]
    elif kind == "combination":
        answer, text = comb(n, r), f"How many unordered groups of {r} can be chosen from {n} distinct objects?"
        working = [f"{n}C{r} = {answer}."]
    else:
        raise ValueError(f"unknown combinatorics kind: {kind}")
    return _number(text, answer, [n * r, factorial(r), answer + r, max(1, answer - r)], working=working)


def _cx(re: int, im: int) -> str:
    """re + im i as a student writes it: 3 + i, 4 − 5i, −5i, 7 (no 1i, no 0i, no "0 + ")."""
    imag = "i" if abs(im) == 1 else f"{abs(im)}i"
    if im == 0:
        return str(re).replace("-", "−")
    if re == 0:
        return ("−" if im < 0 else "") + imag
    return f"{re} {'+' if im > 0 else '−'} {imag}".replace("-", "−")


@generator("complex_number", level=12)
def complex_number(rng, kind="real_part"):
    a, b, c, d = (rng.randint(-8, 8) for _ in range(4))
    z = _cx(a, b)
    if kind == "real_part":
        return _number(
            f"Find the real part of z = {z}.",
            a,
            [b, -a, -b, a + 1, a - 1, a + b],
            aligned=True,
            working=[f"The real component is {a}."],
        )
    if kind == "imaginary_part":
        return _number(
            f"Find the imaginary coefficient of z = {z}.",
            b,
            [a, -a, -b, b + 1, b - 1, a + b],
            aligned=True,
            working=[f"The coefficient of i is {b}."],
        )
    if kind == "conjugate":
        answer = _cx(a, -b)
        return _mcq(
            f"Find the conjugate of {z}.",
            answer,
            [z, _cx(-a, b), _cx(-a, -b)],
            working=["Keep the real part and reverse the sign of the imaginary part."],
        )
    if kind == "add":
        answer = _cx(a + c, b + d)
        other = _cx(c, d)
        return _mcq(
            f"Calculate ({z}) + ({other}).",
            answer,
            [_cx(a + c, abs(b - d)), _cx(a * c, abs(b * d)), _cx(a - c, abs(b - d))],
            working=["Add real parts together and imaginary coefficients together."],
        )
    if kind == "multiply":
        real, imag = a * c - b * d, a * d + b * c
        answer = _cx(real, imag)
        other = _cx(c, d)
        return _mcq(
            f"Calculate ({z})({other}).",
            answer,
            [
                _cx(a * c, abs(b * d)),
                _cx(a * c + b * d, abs(imag)),
                _cx(real, abs(a * d - b * c)),
                _cx(real + 1, imag),
                _cx(real, imag + (1 if imag >= 0 else -1)),
            ],
            working=["Expand and use i² = −1, then collect real and imaginary parts."],
        )
    if kind == "modulus":
        triples = ((3, 4, 5), (5, 12, 13), (8, 15, 17))
        x, y, answer = rng.choice(triples)
        return _number(
            f"Find the modulus of {x} + {y}i.",
            answer,
            [x + y, sqrt(x + y), x * y],
            aligned=True,
            working=[f"|z| = √({x}² + {y}²) = {answer}."],
        )
    raise ValueError(f"unknown complex-number kind: {kind}")


@generator("matrix_product_mcq", level=12)
def matrix_product_mcq(rng):
    a, b, c, d, e, f, g, h = (rng.randint(-4, 5) for _ in range(8))
    product = [[a * e + b * g, a * f + b * h], [c * e + d * g, c * f + d * h]]
    answer = str(product)
    candidates = [
        str([[a * e, b * f], [c * g, d * h]]),
        str([[e * a + f * c, e * b + f * d], [g * a + h * c, g * b + h * d]]),
        str([[a + e, b + f], [c + g, d + h]]),
    ]
    return _mcq(
        f"Multiply A=[[{a}, {b}], [{c}, {d}]] by B=[[{e}, {f}], [{g}, {h}]].",
        answer,
        candidates,
        working=["Each entry is a row-by-column dot product."],
    )


@generator("binomial_probability", level=12)
def binomial_probability(rng):
    n = rng.randint(3, 7)
    x = rng.randint(0, n)
    p = rng.choice((0.2, 0.25, 0.5, 0.75))
    answer = round(comb(n, x) * p**x * (1 - p) ** (n - x), 4)
    return _number(
        f"For X ~ Bin({n}, {p}), find P(X = {x}) to 4 d.p.",
        answer,
        [
            round(p**x, 4),
            round(comb(n, x) * p**x, 4),
            round(1 - answer, 4),
            round(answer + 0.1, 4),
            round(abs(answer - 0.1), 4),
        ],
        aligned=True,
        places=4,
        working=[f"P(X={x}) = C({n},{x})({p})^{x}({1 - p})^{n - x} = {answer:.4f}."],
    )


@generator("z_score", level=12)
def z_score(rng):
    mean = rng.randint(40, 80)
    sd = rng.choice((2, 4, 5, 10))
    z = rng.randint(-3, 3)
    value = mean + z * sd
    return _number(
        f"A distribution has mean {mean} and standard deviation {sd}. Find the z-score for {value}.",
        z,
        [z + 1, z - 1, -z, 2 * z + 1],
        working=[f"z = (x − μ)/σ = ({value} − {mean})/{sd} = {z}."],
    )
