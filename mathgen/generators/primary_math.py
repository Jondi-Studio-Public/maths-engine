"""Primary fractions, decimals, percentages and metric measurement."""

import math
from fractions import Fraction

from ..answers import Distractor
from ..core import generator, latex_answer, pick_distractors, rational_text, round_half_up, stated_precision_string


def _as_fraction(value):
    return Fraction(str(round(float(value), 9)))


def _aligned(answer, candidates, places=None, *, exact_only=False):
    """Show the answer and every candidate at one shared number of decimal
    places, so the options cannot be told apart by shape ("423.0" against
    "34", "0.1406" against "0.75"). Candidates that land exactly on the
    answer's grid come first; the rest are rounded to it and used only as
    fallbacks, so a draw keeps its distractors without showing float noise.
    ``exact_only`` drops the rounded fallbacks (a value that does not sit on
    the grid has no honest display at it)."""
    value = _as_fraction(answer)
    if places is None:
        places = next(p for p in range(7) if (value * 10**p).denominator == 1)

    def show(v):
        return stated_precision_string(round_half_up(_as_fraction(v), places), places)

    exact = [c for c in candidates if (_as_fraction(c) * 10**places).denominator == 1]
    rounded = [] if exact_only else [c for c in candidates if c not in exact]
    return show(value), [show(c) for c in exact + rounded]


def _number_question(
    text, answer, candidates, *, latex=None, working=None, answer_text=None, aligned=False, exact_only=False
):
    if aligned and answer_text is None:
        answer_text, candidates = _aligned(answer, candidates, exact_only=exact_only)
    if answer_text is None:
        answer_text = rational_text(answer) if isinstance(answer, Fraction) else str(answer)
    d = dict(
        text=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working or [],
        distractors=pick_distractors(answer_text, candidates),
    )
    if latex is not None:
        d["latex"] = latex
    return d


@generator("fraction_of_quantity", level=6)
def fraction_of_quantity(rng, max_den=8, quantity_hi=80, find_whole=False):
    denominators = [d for d in range(2, max_den + 1)]
    denominator = rng.choice(denominators)
    numerator = rng.randint(1, denominator - 1)
    groups = rng.randint(2, max(2, quantity_hi // denominator))
    quantity = denominator * groups
    answer = numerator * groups
    if find_whole:
        return _number_question(
            f"{numerator}/{denominator} of a number is {answer}. What is the number?",
            quantity,
            [answer, numerator * quantity, max(0, quantity - groups), quantity + groups],
            latex=rf"\frac{{{numerator}}}{{{denominator}}}\text{{ of a number is }}{answer}",
            working=[
                (
                    f"{numerator} parts are worth {answer}, so 1 part is worth {answer} ÷ {numerator} = {groups}."
                    if numerator > 1
                    else f"1 part is worth {answer}."
                ),
                f"The whole is {denominator} parts: {denominator} × {groups} = {quantity}.",
            ],
        )
    return _number_question(
        f"What is {numerator}/{denominator} of {quantity}?",
        answer,
        [groups, numerator * quantity, max(0, answer - groups), answer + groups],
        latex=rf"\frac{{{numerator}}}{{{denominator}}}\text{{ of }}{quantity}",
        working=[
            f"Divide {quantity} into {denominator} equal parts: {quantity} ÷ {denominator} = {groups}.",
            f"Take {numerator} {'part' if numerator == 1 else 'parts'}: {numerator} × {groups} = {answer}.",
        ],
    )


@generator("compare_common_fractions", level=6)
def compare_common_fractions(rng, max_den=12, include_equal=True, benchmark=False):
    if benchmark:
        half = Fraction(1, 2)
        while True:
            d1 = rng.randint(3, max_den)
            n1 = rng.randint(1, d1 - 1)
            d2 = rng.randint(3, max_den)
            n2 = rng.randint(1, d2 - 1)
            left, right = Fraction(n1, d1), Fraction(n2, d2)
            dist_left, dist_right = abs(left - half), abs(right - half)
            if dist_left != dist_right:
                break
        answer_text = f"{n1}/{d1}" if dist_left < dist_right else f"{n2}/{d2}"
        return dict(
            text=f"Which fraction is closer to one half: {n1}/{d1} or {n2}/{d2}?",
            latex=rf"\left|\frac{{{n1}}}{{{d1}}}-\tfrac12\right|\;\text{{vs}}\;\left|\frac{{{n2}}}{{{d2}}}-\tfrac12\right|",
            answer=answer_text,
            answer_text=answer_text,
            answer_latex=answer_text,
            working=[
                f"{n1}/{d1} is {dist_left} away from one half; {n2}/{d2} is {dist_right} away. The closer one is {answer_text}."
            ],
            distractors=[f"{n2}/{d2}" if answer_text == f"{n1}/{d1}" else f"{n1}/{d1}", "They are equally close"],
        )
    d1 = rng.randint(2, max_den)
    n1 = rng.randint(1, d1 - 1)
    if include_equal and rng.random() < 0.2:
        scale = rng.randint(2, max(2, max_den // d1)) if d1 <= max_den // 2 else 1
        n2, d2 = n1 * scale, d1 * scale
    else:
        d2 = rng.randint(2, max_den)
        n2 = rng.randint(1, d2 - 1)
    left, right = Fraction(n1, d1), Fraction(n2, d2)
    answer = "=" if left == right else ("<" if left < right else ">")
    common = d1 * d2 // math.gcd(d1, d2)
    if d1 == d2 and left == right:
        working = [f"The denominators are the same and the numerators are both {n1}, so the fractions are equal (=)."]
    elif d1 == d2:
        working = [
            f"The denominators are the same, so compare the numerators: {n1} and {n2}, so {n1}/{d1} {answer} {n2}/{d2}."
        ]
    else:
        conv1 = f"{n1}/{d1} = {n1 * common // d1}/{common}" if d1 != common else None
        conv2 = f"{n2}/{d2} = {n2 * common // d2}/{common}" if d2 != common else None
        converted = " and ".join(c for c in (conv1, conv2) if c)
        working = [f"Use a common denominator of {common}: {converted}, so {n1}/{d1} {answer} {n2}/{d2}."]
    return dict(
        text=f"Choose the correct symbol: {n1}/{d1} __ {n2}/{d2}",
        latex=rf"\frac{{{n1}}}{{{d1}}}\;\square\;\frac{{{n2}}}{{{d2}}}",
        answer=answer,
        answer_text=answer,
        answer_latex=answer,
        working=working,
        distractors=[symbol for symbol in ("<", "=", ">") if symbol != answer],
    )


@generator("decimal_place_value", level=4)
def decimal_place_value(rng, max_places=2, whole_hi=50, ask_place=False):
    places = rng.randint(1, max_places)
    digits = [rng.randint(0, 9) for _ in range(places)]
    chosen = rng.randrange(places)
    digits[chosen] = rng.randint(1, 9)
    if ask_place and places >= 2:
        zero_idx = (chosen + 1) % places
        digits[zero_idx] = 0
    whole = rng.randint(0, whole_hi)
    shown = f"{whole}." + "".join(str(d) for d in digits)
    digit = digits[chosen]
    place = chosen + 1
    place_name = ("tenths", "hundredths", "thousandths")[place - 1]
    if ask_place:
        wrong_digits = [d for d in set(digits) if d != digit]
        candidates = wrong_digits + [(digit + 1) % 10, (digit + 5) % 10, whole % 10]
        return _number_question(
            f"Which digit is in the {place_name} place of {shown}?",
            digit,
            candidates,
            latex=rf"\text{{Digit in the {place_name} place of }}{shown}",
            working=[f"Counting from the decimal point, the {place_name} place is digit number {place}: {digit}."],
        )
    answer = digit / (10**place)
    return _number_question(
        f"What is the value of the digit {digit} in {shown}?",
        answer,
        [
            digit,
            digit * (10**place),
            digit / (10 ** max(1, place - 1)),
            digit / (10 ** (place + 1)),
            digit * (10 ** (place + 1)),
            (digit % 9 + 1) / (10**place),
        ],
        aligned=True,
        exact_only=True,
        latex=rf"\text{{Value of digit }}{digit}\text{{ in }}{shown}",
        working=[f"The digit {digit} is in the {place_name} place, so its value is {answer}."],
    )


@generator("round_decimal", level=6)
def round_decimal(rng, target_places=1, whole_hi=100):
    # Every value is held as an integer count of the target place and formatted
    # to exactly that many decimals. Working in floats dropped the trailing
    # zero that carries the stated precision -- "Round 16.201 to the nearest
    # hundredth" answered "16.2" rather than "16.20", and the nearest-whole-
    # number tier answered "26.0" rather than "26" -- and produced distractors
    # such as "20.400000000000002" for the learner to read.
    scale = 10 ** (target_places + 1)
    raw = rng.randint(1, whole_hi * scale - 1)
    rounded_units = (raw + 5) // 10

    def at_target(units: int) -> str:
        if target_places == 0:
            return str(units)
        whole, part = divmod(abs(units), 10**target_places)
        sign = "-" if units < 0 else ""
        return f"{sign}{whole}.{part:0{target_places}d}"

    shown = f"{raw / scale:.{target_places + 1}f}"
    answer_text = at_target(rounded_units)
    label = (
        "nearest whole number"
        if target_places == 0
        else ("nearest tenth" if target_places == 1 else "nearest hundredth")
    )
    candidates = [
        at_target(raw // 10 + 1),  # always rounded up
        at_target(rounded_units + 1),  # rounded the wrong way
        at_target(rounded_units - 1),
        at_target(raw // 10),  # truncated instead of rounded
        at_target(((rounded_units + 5) // 10) * 10),  # rounded to the place above
    ]
    return _number_question(
        f"Round {shown} to the {label}.",
        rounded_units if target_places == 0 else rounded_units / 10**target_places,
        candidates,
        working=[f"Look at the next digit and round to the {label}: {answer_text}."],
        answer_text=answer_text,
    )


def _fdp_dec(x) -> str:
    """A decimal the way str(float) prints it ("0.5", "0.05", "5.0"), never in exponent form."""
    text = f"{float(x):.6f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def _fdp_pct(x) -> str:
    v = Fraction(x) * 100
    return f"{v.numerator}%" if v.denominator == 1 else f"{float(v):g}%"


def _fdp_swap(digits: str) -> str | None:
    """The two digits in the other order, or None when that changes nothing or leaves a trailing zero."""
    if len(digits) != 2 or digits[0] == digits[1] or digits[1] == "0":
        return None
    return digits[1] + digits[0]


_FDP_STEMS = {
    "percent": (
        "Which percentage is equal to {f}?",
        "Write {f} as a percentage.",
        "{f} of a tank is full. What percentage of the tank is full?",
        "A class has {f} of its students on the bus. What percentage is that?",
    ),
    "decimal": (
        "Which decimal is equal to {f}?",
        "Write {f} as a decimal.",
        "A runner finished {f} of the course. Which decimal shows that?",
        "{f} of the ribbon was used. Which decimal is that?",
    ),
    "fraction": (
        "Which fraction is equal to {d}?",
        "Write {d} as a fraction.",
        "A jug is filled to {d} of its capacity. Which fraction shows that?",
        "Which fraction has the same value as {d}?",
    ),
    "percent_from_decimal": (
        "Which percentage is equal to the decimal {d}?",
        "Write the decimal {d} as a percentage.",
        "A battery is charged to {d} of its full charge. What percentage is that?",
    ),
    "decimal_from_percent": (
        "Which decimal is equal to {p}?",
        "Write {p} as a decimal.",
        "A phone battery shows {p}. Which decimal shows the charge?",
    ),
}


@generator("fraction_decimal_percent", level=6)
def fraction_decimal_percent(rng, denominators=(2, 4, 5, 10, 20, 25), reverse_directions=False, allow_improper=False):
    denominator = rng.choice(tuple(denominators))
    top = 2 * denominator - 1 if allow_improper else denominator - 1
    numerator = rng.randint(1, top)
    value = Fraction(numerator, denominator)
    n, d = value.numerator, value.denominator
    fraction_text = rational_text(value)
    decimal = _fdp_dec(value)
    percent_text = _fdp_pct(value)
    modes = ("percent", "decimal", "fraction")
    if reverse_directions:
        modes = modes + ("percent_from_decimal", "decimal_from_percent")
    mode = rng.choice(modes)
    # The scene stems (a tank, a ribbon, a battery) are parts of a whole, so an value of 1 or more only gets the bare
    # "write it as" stems, the first two of each mode.
    stems = _FDP_STEMS[mode][:2] if value >= 1 else _FDP_STEMS[mode]
    stem = rng.choice(stems)
    if mode in ("percent", "percent_from_decimal"):
        render, answer = _fdp_pct, percent_text
    elif mode in ("decimal", "decimal_from_percent"):
        render, answer = _fdp_dec, decimal
    else:
        render, answer = rational_text, fraction_text
    text = stem.format(f=fraction_text, d=decimal, p=percent_text)

    # Near misses, each a slip a learner makes, rendered in the answer's own form so no option stands out by being
    # a "nice" benchmark. (text, value, tag): value is what the option is worth, so a slip that lands on the right
    # value is dropped rather than offered.
    near = []  # place-value slips
    digit_slips = []  # digits read straight off the other form
    neighbours = []  # wrong numerator or denominator
    for k in (-1, 1, 2):
        if numerator + k > 0:
            neighbours.append(Fraction(numerator + k, denominator))
    for k in (-1, 1):
        if d + k > 1 and n < d + k:
            neighbours.append(Fraction(n, d + k))
    near += [value / 10, value * 10]
    if render is _fdp_pct:
        digit_slips += [Fraction(int(f"{n}{d}"), 100), Fraction(d, 100)]
        swapped = _fdp_swap(str(int(value * 100))) if (value * 100).denominator == 1 else None
        if swapped:
            digit_slips.append(Fraction(int(swapped), 100))
    elif render is _fdp_dec:
        digit_slips += [Fraction(f"0.{n}{d}"), Fraction(f"0.{d}")]
        frac_digits = decimal.split(".")[1]
        swapped = _fdp_swap(frac_digits) if decimal.startswith("0.") else None
        if swapped:
            digit_slips.append(Fraction(f"0.{swapped}"))
    else:
        if n != d:
            digit_slips.append(Fraction(d, n))  # numerator and denominator swapped
        frac_digits = decimal.split(".")[1] if decimal.startswith("0.") else ""
        if frac_digits and int(frac_digits) > 1:
            digit_slips.append(Fraction(1, int(frac_digits)))  # the decimal's digits used as a denominator
        if len(frac_digits) == 2 and frac_digits[1] != "0" and frac_digits[0] != "0":
            digit_slips.append(Fraction(int(frac_digits[0]), int(frac_digits[1])))
    groups = [near, digit_slips, neighbours]
    for g in groups:
        rng.shuffle(g)
    picked: list[Fraction] = []
    seen = {value}

    def take(candidate):
        if candidate in seen or candidate <= 0:
            return False
        if render is _fdp_dec and candidate.denominator == 1 and value.denominator != 1:
            return False  # a "12.0" beside "1.2" reads oddly
        q = candidate.denominator
        while q % 2 == 0:
            q //= 2
        while q % 5 == 0:
            q //= 5
        if q != 1 and render is not rational_text:
            return False  # a recurring decimal is not a near miss a learner could write down
        seen.add(candidate)
        picked.append(candidate)
        return True

    # one of each kind first, in a shuffled order, then top up from whatever is left
    order = [0, 1, 2]
    rng.shuffle(order)
    for gi in order:
        for c in groups[gi]:
            if take(c):
                break
    for gi in order:
        for c in groups[gi]:
            if len(picked) >= 3:
                break
            take(c)
    near_set = set(near)
    candidates = [
        Distractor(
            render(c), misconception="decimal-place-shift" if c in near_set and render is not rational_text else None
        )
        for c in picked
    ]
    rng.shuffle(candidates)
    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer,
        answer_latex=latex_answer(answer),
        working=[f"{fraction_text} = {decimal} = {percent_text}."],
        distractors=pick_distractors(answer, candidates),
    )


@generator("percentage_quantity", level=6)
def percentage_quantity(
    rng, kind="amount", percents=(10, 20, 25, 50, 75), units_hi=20, quantity_hi=100, find_whole=False, chained=False
):
    if kind == "discount" and chained:
        percent = rng.choice(tuple(percents))
        percent2 = rng.choice(tuple(percents))
        fraction = Fraction(percent, 100)
        sale_fraction = Fraction(100 - percent, 100)
        fraction2 = Fraction(percent2, 100)
        from math import lcm

        base = lcm(fraction.denominator, sale_fraction.denominator) * fraction2.denominator
        max_t = min(units_hi, quantity_hi // base)
        if max_t < 1:
            raise ValueError("quantity_hi is too small for the requested chained percentages")
        t = rng.randint(1, max_t)
        quantity = base * t
        portion = int(fraction * quantity)
        sale_price = quantity - portion
        second_discount = int(fraction2 * sale_price)
        answer = sale_price - second_discount
        return _number_question(
            f"An item costs ${quantity} and is discounted by {percent}%, then a further {percent2}% off the new price. What is the final price?",
            answer,
            [sale_price, portion + second_discount, quantity - second_discount, quantity - portion],
            working=[
                f"First discount: {percent}% of ${quantity} = ${portion}; new price ${sale_price}.",
                f"Second discount: {percent2}% of ${sale_price} = ${second_discount}; final price ${answer}.",
            ],
        )
    percent = rng.choice(tuple(percents))
    fraction = Fraction(percent, 100)
    max_units = min(units_hi, quantity_hi // fraction.denominator)
    if max_units < 2:
        raise ValueError("quantity_hi is too small for the requested percentages")
    units = rng.randint(2, max_units)
    quantity = fraction.denominator * units
    portion = fraction.numerator * units
    if kind == "amount":
        if find_whole:
            return _number_question(
                f"{percent}% of a number is {portion}. What is the number?",
                quantity,
                [portion, units, max(0, quantity - units), quantity + units],
                working=[
                    f"{percent}% = {rational_text(fraction)}; {portion} ÷ {fraction.numerator} = {units} for one part.",
                    f"The whole is {fraction.denominator} parts: {fraction.denominator} × {units} = {quantity}.",
                ],
            )
        answer = portion
        text = f"What is {percent}% of {quantity}?"
        working = [f"{percent}% = {rational_text(fraction)}, so {rational_text(fraction)} of {quantity} is {answer}."]
        candidates = [units, quantity - portion, percent * quantity, portion + units]
    elif kind == "discount":
        answer = quantity - portion
        text = f"An item costs ${quantity} and is discounted by {percent}%. What is the sale price?"
        working = [f"The discount is ${portion}; subtract it from ${quantity} to get ${answer}."]
        candidates = [portion, quantity + portion, quantity - units, percent]
    else:
        raise ValueError(f"unknown percentage kind: {kind}")
    return _number_question(text, answer, candidates, working=working)


_METRIC_FACTORS = {
    "length": (("mm", 1), ("cm", 10), ("m", 1000), ("km", 1_000_000)),
    "mass": (("g", 1), ("kg", 1000)),
    "capacity": (("mL", 1), ("L", 1000)),
}


@generator("metric_conversion", level=6)
def metric_conversion(rng, kind="length", max_steps=2, value_hi=25, combine=False):
    units = _METRIC_FACTORS.get(kind)
    if units is None:
        raise ValueError(f"unknown metric kind: {kind}")
    pairs = [
        (units[i], units[j]) for i in range(len(units)) for j in range(len(units)) if i != j and abs(i - j) <= max_steps
    ]
    if combine:
        (a_unit, a_factor), (b_unit, b_factor) = rng.choice(pairs)
        ratio = Fraction(a_factor, b_factor)
        a_value = ratio.denominator * rng.randint(1, value_hi)
        b_value = rng.randint(1, value_hi)
        converted_a = a_value * ratio
        total = converted_a + b_value
        answer = int(total) if total.denominator == 1 else float(total)
        return _number_question(
            f"One amount is {a_value} {a_unit} and another is {b_value} {b_unit}. What is the total in {b_unit}?",
            answer,
            [a_value + b_value, float(converted_a), b_value, float(converted_a - b_value), float(total * 10)],
            aligned=True,
            exact_only=True,
            latex=rf"{a_value}\,\mathrm{{{a_unit}}}+{b_value}\,\mathrm{{{b_unit}}}=\square\,\mathrm{{{b_unit}}}",
            working=[
                f"Convert {a_value} {a_unit} to {b_unit}: {a_value} {a_unit} = {converted_a} {b_unit}. Then add: {converted_a} + {b_value} = {answer} {b_unit}."
            ],
        )
    (source_unit, source_factor), (target_unit, target_factor) = rng.choice(pairs)
    ratio = Fraction(source_factor, target_factor)
    source_value = ratio.denominator * rng.randint(1, value_hi)
    target_value = source_value * ratio
    answer = int(target_value) if target_value.denominator == 1 else float(target_value)
    inverse_error = Fraction(source_value, 1) / ratio
    return _number_question(
        f"Convert {source_value} {source_unit} to {target_unit}.",
        answer,
        [
            float(inverse_error),
            source_value,
            float(target_value * 10),
            float(target_value / 10),
            float(target_value * 100),
        ],
        aligned=True,
        exact_only=True,
        latex=rf"{source_value}\,\mathrm{{{source_unit}}}=\square\,\mathrm{{{target_unit}}}",
        working=[
            f"Use the metric relationship between {source_unit} and {target_unit}: {source_value} {source_unit} = {answer} {target_unit}."
        ],
    )


@generator("rectangle_measure", level=6)
def rectangle_measure(rng, kind="area", side_hi=30, find_side=False):
    length = rng.randint(2, side_hi)
    width = rng.randint(2, side_hi)
    if kind == "area":
        if find_side:
            area = length * width
            return _number_question(
                f"A rectangle has area {area} cm² and length {length} cm. What is its width in cm?",
                width,
                [length, max(0, area - length), area + length, length * width - width],
                working=[f"Width = area ÷ length = {area} ÷ {length} = {width} cm."],
            )
        answer = length * width
        text = f"A rectangle is {length} cm long and {width} cm wide. What is its area in cm²?"
        working = [f"Area = length × width = {length} × {width} = {answer} cm²."]
        candidates = [2 * (length + width), length + width, answer + length, answer - width]
    elif kind == "perimeter":
        if find_side:
            perimeter = 2 * (length + width)
            half = perimeter // 2
            return _number_question(
                f"A rectangle has perimeter {perimeter} cm and length {length} cm. What is its width in cm?",
                width,
                [length, max(0, perimeter - length), half, perimeter],
                working=[f"Half the perimeter is {perimeter} ÷ 2 = {half} cm; width = {half} − {length} = {width} cm."],
            )
        answer = 2 * (length + width)
        text = f"A rectangle is {length} cm long and {width} cm wide. What is its perimeter in cm?"
        working = [f"Perimeter = 2 × (length + width) = 2 × ({length} + {width}) = {answer} cm."]
        candidates = [length * width, length + width, 2 * length + width, length + 2 * width]
    else:
        raise ValueError(f"unknown rectangle measure: {kind}")
    return _number_question(text, answer, candidates, working=working)
