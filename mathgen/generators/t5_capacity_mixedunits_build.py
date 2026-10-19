"""Rectangular-prism capacity and mixed-unit rectangle area/perimeter -- both compose an EXACT metric conversion
(1 L = 1000 cm^3, 1 m^3 = 1000 L = 1,000,000 cm^3; the mm/cm/m linear ladder) into a geometry computation.

Two generators:

- ``prism_capacity_conversion`` (skill ``prism-capacity-metric-units``) --
  a rectangular prism's capacity in litres or cubic metres from its three
  dimensions. Difficulty escalates from a single conversion step (easy) to
  the SAME step run twice in a chained litres-then-m^3 route (medium,
  mixed input units that must be reconciled to a common base before the
  volume formula even applies (hard).
- ``rectangle_mixed_unit_area_perimeter`` (skill
  ``rectangle-area-perimeter-mixed-units``) -- a rectangle's area or
  perimeter from two sides given in different units; the unit
  reconciliation is the skill itself.

Exactness discipline: every conversion factor in this module is a power of
ten, so every quantity is a `Fraction` whose denominator, once reduced, is a
product of 2s and 5s only -- it always terminates in base 10.
`_decimal_places_needed`/`_format_exact` make that guarantee explicit and
raise (never silently round) if a *canonical-answer* value fails to
terminate within a small bound; a raise there is a construction bug
(resample the draw), never a runtime path. `_format_rounded` is the
distractor-only counterpart -- rounds half-up instead of raising, since a
misconception is often a different order of magnitude and may need more
digits than the true answer's own stated precision to state exactly. This
mirrors `p3b_measurement_volume_build.py`'s own `_format_fixed`/
`_format_display` split; copied locally here (not imported) matching that
module's own precedent of one private formatting helper per generator
module rather than a shared `core.py` utility.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    fixed_point_string,
    generator,
    numeric_answer,
    pick_distractors,
)


# --- shared exact-decimal formatting (local copy; see module docstring) ---


def _decimal_places_needed(value: Fraction) -> int:
    """Fewest decimal places that represent `value` exactly. Every value in
    this module has a denominator that is a product of 2s and 5s only, so
    this always terminates; raises otherwise, which is a construction bug."""
    d = Fraction(value).denominator
    twos = 0
    while d % 2 == 0:
        d //= 2
        twos += 1
    fives = 0
    while d % 5 == 0:
        d //= 5
        fives += 1
    check(d == 1, f"{value} does not terminate in base 10")
    return max(twos, fives)


def _format_exact(value: Fraction, places: int) -> str:
    """CANONICAL-answer formatting: exact at `places`, raises (never rounds)
    if `value` is not exactly representable there."""
    scaled = Fraction(value) * (10**places)
    check(scaled.denominator == 1, f"{value} is not exact to {places} decimal places")
    return fixed_point_string(scaled.numerator, places)


def _format_rounded(value: Fraction, places: int) -> str:
    """DISTRACTOR-only formatting: rounds half-up to `places` instead of
    raising. `check_distinct`/`pick_distractors`'s `tolerance=` are what
    guard against a rounded distractor collapsing onto the answer, not this
    function."""
    scale = 10**places
    scaled = Fraction(value) * scale
    n, d = scaled.numerator, scaled.denominator
    if n >= 0:
        rounded = (2 * n + d) // (2 * d)
    else:
        rounded = -((-2 * n + d) // (2 * d))
    return fixed_point_string(rounded, places)


def _fmt_value(value) -> str:
    """Render a dimension's own given value (int or exact Fraction)."""
    fr = Fraction(value)
    if fr.denominator == 1:
        return str(fr.numerator)
    return _format_exact(fr, _decimal_places_needed(fr))


def _bare_label(label: str) -> str:
    return label.split(" (")[0]


def _precision_phrase(places: int, bare_label: str) -> str:
    if places == 0:
        return f"This is a whole number of {bare_label}; give your answer exactly, with no rounding needed."
    word = "place" if places == 1 else "places"
    return f"Give your answer to {places} decimal {word}."


# ===========================================================================
# Skill 1: prism-capacity-metric-units
# ===========================================================================

_CAPACITY_FACTORS = {"L": Fraction(1000), "m3": Fraction(1_000_000)}
_CAPACITY_LABELS = {"L": "litres (L)", "m3": "cubic metres (m³)"}
# mm^3-pivot equivalents, for the independent check (1 cm^3 = 1000 mm^3).
_MM3_CAPACITY_FACTORS = {"L": Fraction(1_000_000), "m3": Fraction(1_000_000_000)}

_LINEAR_TO_CM = {"cm": Fraction(1), "mm": Fraction(1, 10), "m": Fraction(100)}
_LINEAR_TO_MM = {"cm": Fraction(10), "mm": Fraction(1), "m": Fraction(1000)}


def _volume_cm3(dims) -> Fraction:
    """Construction's own pivot: convert every dimension to cm, then
    multiply. `dims`: iterable of (value, unit)."""
    total = Fraction(1)
    for value, unit in dims:
        total *= Fraction(value) * _LINEAR_TO_CM[unit]
    return total


def _volume_mm3_independent(dims) -> Fraction:
    """Independent-check pivot: mm (a different base unit from construction,
    and a fresh lookup of each raw factor -- never reuses construction's
    cm-equivalent values)."""
    total = Fraction(1)
    for value, unit in dims:
        total *= Fraction(value) * _LINEAR_TO_MM[unit]
    return total


def _capacity_misconceptions(volume_cm3: Fraction, target: str) -> dict[str, Fraction]:
    target_factor = _CAPACITY_FACTORS[target]
    other = "m3" if target == "L" else "L"
    other_factor = _CAPACITY_FACTORS[other]
    return {
        "wrong-order-of-magnitude": volume_cm3 / (target_factor / 10),
        "reports-unconverted-volume": volume_cm3,
        "confuses-L-and-m3-factor": volume_cm3 / other_factor,
    }


def _draw_easy_prism(rng):
    """All three dims in cm; target litres. Resample until the litres
    answer is a genuine (non-whole) decimal -- i.e. volume_cm3 is not a
    multiple of 1000."""
    for _ in range(2000):
        l = rng.randint(2, 9)
        w = rng.randint(2, 9)
        h = 10 * rng.randint(1, 9)
        volume_cm3 = l * w * h
        if (Fraction(volume_cm3, 1000)).denominator != 1:
            return [(l, "cm", "long"), (w, "cm", "wide"), (h, "cm", "high")]
    check(False, "could not draw easy-tier prism dims with a non-whole litres answer")


def _draw_medium_prism(rng):
    """All three dims in cm; target m^3, via a litres intermediate.
    Resample until volume_cm3's product carries EXACTLY 3 trailing zeros
    (distributed across the three dims, not from any one dim being 1000) --
    that makes the litres intermediate a whole number and the m^3 answer
    exact to exactly 3 decimal places."""
    for _ in range(2000):
        l = rng.randint(2, 20)
        w = rng.randint(2, 20)
        h = rng.randint(2, 20)
        volume_cm3 = l * w * h
        if _decimal_places_needed(Fraction(volume_cm3, 1_000_000)) == 3:
            return [(l, "cm", "long"), (w, "cm", "wide"), (h, "cm", "high")]
    check(False, "could not draw medium-tier prism dims with an exact-3dp m^3 answer")


def _draw_hard_prism(rng):
    """Two dims in cm, one dim in a different unit (mm or m), reconciled to
    cm before the volume formula applies. Target randomised between L/m^3.
    Resample until the capacity answer is a genuine (non-whole) decimal."""
    roles = ("long", "wide", "high")
    for _ in range(2000):
        odd_index = rng.randrange(3)
        odd_unit = rng.choice(("mm", "m"))
        target = rng.choice(("L", "m3"))
        cm_vals = [rng.randint(2, 12) for _ in range(3)]
        if odd_unit == "m":
            cm_vals[odd_index] = rng.randint(20, 120)
        dims = []
        for i, cm_val in enumerate(cm_vals):
            if i == odd_index:
                if odd_unit == "mm":
                    dims.append((cm_val * 10, "mm", roles[i]))
                else:
                    dims.append((Fraction(cm_val, 100), "m", roles[i]))
            else:
                dims.append((cm_val, "cm", roles[i]))
        dim_values = [(v, u) for v, u, _r in dims]
        volume_cm3 = _volume_cm3(dim_values)
        answer_value = volume_cm3 / _CAPACITY_FACTORS[target]
        if answer_value.denominator != 1:
            return dims, target
    check(False, "could not draw hard-tier prism dims with a non-whole capacity answer")


@generator("prism_capacity_conversion", level=12)
def prism_capacity_conversion(rng, tier="easy"):
    if tier == "easy":
        dims = _draw_easy_prism(rng)
        target = "L"
    elif tier == "medium":
        dims = _draw_medium_prism(rng)
        target = "m3"
    elif tier == "hard":
        dims, target = _draw_hard_prism(rng)
    else:
        raise ValueError(f"unknown tier: {tier!r}")

    dim_values = [(v, u) for v, u, _r in dims]
    volume_cm3 = _volume_cm3(dim_values)
    answer_value = volume_cm3 / _CAPACITY_FACTORS[target]
    places = _decimal_places_needed(answer_value)
    answer_text = _format_exact(answer_value, places)

    check_equivalent(
        answer_value,
        _volume_mm3_independent(dim_values) / _MM3_CAPACITY_FACTORS[target],
        "capacity mismatch: cm-pivot construction route vs mm-pivot independent route",
    )

    misconceptions = _capacity_misconceptions(volume_cm3, target)
    for name, value in misconceptions.items():
        check_distinct(answer_value, value, msg=f"{name} distractor equals the answer")
    distractor_strings = [_format_rounded(v, places) for v in misconceptions.values()]

    unit_label = _CAPACITY_LABELS[target]
    precision = _precision_phrase(places, _bare_label(unit_label))
    dims_phrase = (
        f"{_fmt_value(dims[0][0])} {dims[0][1]} long, "
        f"{_fmt_value(dims[1][0])} {dims[1][1]} wide and "
        f"{_fmt_value(dims[2][0])} {dims[2][1]} high"
    )

    if tier == "medium":
        litres_value = volume_cm3 / _CAPACITY_FACTORS["L"]
        text = (
            f"A rectangular prism-shaped container is {dims_phrase}. "
            f"Find its capacity in cubic metres (m³), by first finding the capacity "
            f"in litres (L) as a check, then converting that to cubic metres. {precision}"
        )
        working = [
            f"Volume = {_fmt_value(dims[0][0])} × {_fmt_value(dims[1][0])} × "
            f"{_fmt_value(dims[2][0])} = {int(volume_cm3)} cm³.",
            f"Capacity in litres = {int(volume_cm3)} ÷ 1000 = {int(litres_value)} L.",
            f"Capacity in cubic metres = {int(litres_value)} ÷ 1000 = {answer_text} m³.",
        ]
    else:
        text = (
            f"A rectangular prism-shaped container is {dims_phrase}. What is its capacity in {unit_label}? {precision}"
        )
        working = [
            f"Volume = {_fmt_value(dims[0][0])} × {_fmt_value(dims[1][0])} × "
            f"{_fmt_value(dims[2][0])} = {int(volume_cm3)} cm³.",
            f"Capacity in {_bare_label(unit_label)} = {int(volume_cm3)} ÷ "
            f"{int(_CAPACITY_FACTORS[target])} = {answer_text}.",
        ]

    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer=numeric_answer(answer_value),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, distractor_strings, tolerance=Fraction(1, 2 * 10**places)),
    )


# ===========================================================================
# Skill 2: rectangle-area-perimeter-mixed-units
# ===========================================================================

_LENGTH_UNITS_MM = {"mm": Fraction(1), "cm": Fraction(10), "m": Fraction(1000)}
_LENGTH_LABELS = {
    "mm": "millimetres (mm)",
    "cm": "centimetres (cm)",
    "m": "metres (m)",
}
_AREA_LABELS = {
    "mm": "square millimetres (mm²)",
    "cm": "square centimetres (cm²)",
    "m": "square metres (m²)",
}


def _linear_ratio(from_unit: str, to_unit: str) -> Fraction:
    return _LENGTH_UNITS_MM[from_unit] / _LENGTH_UNITS_MM[to_unit]


def _rectangle_answer(side1, side2, target, quantity, base_unit) -> Fraction:
    s1 = Fraction(side1[0]) * _linear_ratio(side1[1], base_unit)
    s2 = Fraction(side2[0]) * _linear_ratio(side2[1], base_unit)
    if quantity == "perimeter":
        raw = 2 * (s1 + s2)
        return raw * _linear_ratio(base_unit, target)
    raw = s1 * s2
    return raw * _linear_ratio(base_unit, target) ** 2


def _rectangle_no_conversion(side1, side2, quantity) -> Fraction:
    """Multiplies/adds the two RAW numbers directly, ignoring the unit
    difference entirely."""
    v1, v2 = Fraction(side1[0]), Fraction(side2[0])
    if quantity == "perimeter":
        return 2 * (v1 + v2)
    return v1 * v2


def _rectangle_partial_conversion(side1, side2, target, quantity, base_unit) -> Fraction:
    """Converts whichever side's own unit differs from `base_unit`
    CORRECTLY; leaves the other side's raw value untouched (as if it were
    already in `base_unit`) before combining and converting to `target` as
    normal. Deliberately keyed on UNIT, not on which argument is `side1`
    vs `side2`: this generator randomises which physical side plays which
    role, and a version that always "forgets" `side2` degenerates to the
    fully-correct answer whenever the random draw happens to put the
    side already matching `base_unit` in that slot (found empirically: the
    hard tier's mm-vs-m pairing collided on ~half of all seeds before this
    fix, exactly the draws where the mm side -- already at `base_unit` --
    landed as `side2`, making "forgetting" it a no-op)."""
    if side1[1] != base_unit:
        s1 = Fraction(side1[0])  # forgot to convert this one
        s2 = Fraction(side2[0]) * _linear_ratio(side2[1], base_unit)
    else:
        s1 = Fraction(side1[0]) * _linear_ratio(side1[1], base_unit)
        s2 = Fraction(side2[0])  # forgot to convert this one
    if quantity == "perimeter":
        raw = 2 * (s1 + s2)
        return raw * _linear_ratio(base_unit, target)
    raw = s1 * s2
    return raw * _linear_ratio(base_unit, target) ** 2


def _rectangle_linear_factor_on_area(side1, side2, target, base_unit) -> Fraction:
    """Converts the final area with the LINEAR factor instead of its
    square. Area quantity only."""
    s1 = Fraction(side1[0]) * _linear_ratio(side1[1], base_unit)
    s2 = Fraction(side2[0]) * _linear_ratio(side2[1], base_unit)
    raw = s1 * s2
    return raw * _linear_ratio(base_unit, target)


def _rectangle_wrong_direction(side1, side2, target, quantity, base_unit) -> Fraction:
    """Inverts every conversion ratio used in `_rectangle_answer` -- a systematic direction confusion (divides
    where a multiplication was needed and vice versa), the same shape as the library's existing
    `convert-metric-length` `conversion-direction` misconception."""
    s1 = Fraction(side1[0]) / _linear_ratio(side1[1], base_unit)
    s2 = Fraction(side2[0]) / _linear_ratio(side2[1], base_unit)
    if quantity == "perimeter":
        raw = 2 * (s1 + s2)
        return raw / _linear_ratio(base_unit, target)
    raw = s1 * s2
    return raw / (_linear_ratio(base_unit, target) ** 2)


def _draw_easy_rectangle(rng):
    """mm & cm (factor of 10 apart); perimeter; target = mm (the smaller)."""
    mm_val = rng.randint(15, 95)
    cm_val = rng.randint(2, 9)
    if rng.random() < 0.5:
        side1, side2 = (mm_val, "mm"), (cm_val, "cm")
    else:
        side1, side2 = (cm_val, "cm"), (mm_val, "mm")
    return side1, side2, "mm", "perimeter"


def _draw_medium_rectangle(rng):
    """m & cm (factor of 100 apart); area; target = cm^2 (the smaller,
    squared). The m-side is drawn to 3 decimal places -- one more place
    than the x100 m->cm conversion can absorb -- and resampled until the
    area is a genuine (non-whole) decimal, so the squared-unit conversion
    step is not disguised behind a trivially whole answer."""
    for _ in range(2000):
        m_val = Fraction(rng.randint(1000, 4000), 1000)
        cm_val = rng.randint(20, 90)
        area_cm2 = m_val * 100 * cm_val
        if area_cm2.denominator != 1:
            if rng.random() < 0.5:
                side1, side2 = (m_val, "m"), (cm_val, "cm")
            else:
                side1, side2 = (cm_val, "cm"), (m_val, "m")
            return side1, side2, "cm", "area"
    check(False, "could not draw medium-tier rectangle dims with a non-whole area answer")


def _draw_hard_rectangle(rng):
    """mm & m (two steps apart); area; target = cm^2 -- the one unit in {mm, cm, m} that matches neither input,
    so the target-unit "draw" has exactly one candidate. The m-side is drawn to 3 decimal places for the same
    reason as medium, and resampled until the area is a genuine (non-whole) decimal."""
    for _ in range(2000):
        mm_val = rng.randint(50, 950)
        m_val = Fraction(rng.randint(1000, 4000), 1000)
        used_units = {"mm", "m"}
        candidates = [u for u in ("mm", "cm", "m") if u not in used_units]
        check(candidates == ["cm"], "hard tier's target-unit draw space should be exactly {'cm'}")
        target = candidates[0]
        area_cm2 = _rectangle_answer((mm_val, "mm"), (m_val, "m"), target, "area", "mm")
        if area_cm2.denominator != 1:
            if rng.random() < 0.5:
                side1, side2 = (mm_val, "mm"), (m_val, "m")
            else:
                side1, side2 = (m_val, "m"), (mm_val, "mm")
            return side1, side2, target, "area"
    check(False, "could not draw hard-tier rectangle dims with a non-whole area answer")


@generator("rectangle_mixed_unit_area_perimeter", level=12)
def rectangle_mixed_unit_area_perimeter(rng, tier="easy"):
    if tier == "easy":
        side1, side2, target, quantity = _draw_easy_rectangle(rng)
    elif tier == "medium":
        side1, side2, target, quantity = _draw_medium_rectangle(rng)
    elif tier == "hard":
        side1, side2, target, quantity = _draw_hard_rectangle(rng)
    else:
        raise ValueError(f"unknown tier: {tier!r}")

    base_unit = "mm"
    answer_value = _rectangle_answer(side1, side2, target, quantity, base_unit)
    check_equivalent(
        answer_value,
        _rectangle_answer(side1, side2, target, quantity, "m"),
        "rectangle answer mismatch: mm-pivot construction route vs m-pivot independent route",
    )
    places = _decimal_places_needed(answer_value)
    answer_text = _format_exact(answer_value, places)

    candidates = {
        "no-conversion-before-computing": _rectangle_no_conversion(side1, side2, quantity),
    }
    # "partial-conversion" (convert one side, forget the other) is only a
    # genuinely distinct error when BOTH sides actually need converting to
    # reach `target` -- true at hard tier only (mm & m, target cm, neither
    # input already matches). At easy (mm & cm, target mm) and medium (m &
    # cm, target cm) exactly one side already IS the target unit, so
    # "forgetting to convert the other one" is the same operation as
    # `no-conversion-before-computing` and would collide with it (or, if the
    # already-native side is the one arbitrarily "left unconverted", with
    # the correct answer itself). Skip it there rather than force a
    # collision, the same honesty used with
    # `linear-factor-on-area` at easy.
    if side1[1] != target and side2[1] != target:
        candidates["partial-conversion"] = _rectangle_partial_conversion(side1, side2, target, quantity, base_unit)
    if quantity == "area":
        candidates["linear-factor-on-area"] = _rectangle_linear_factor_on_area(side1, side2, target, base_unit)
    candidates["wrong-direction-conversion"] = _rectangle_wrong_direction(side1, side2, target, quantity, base_unit)

    for name, value in candidates.items():
        check_distinct(answer_value, value, msg=f"{name} distractor equals the answer")
    distractor_strings = [_format_rounded(v, places) for v in candidates.values()]

    quantity_label = quantity
    target_label = _LENGTH_LABELS[target] if quantity == "perimeter" else _AREA_LABELS[target]
    precision = _precision_phrase(places, _bare_label(target_label))
    text = (
        f"A rectangle has one side of {_fmt_value(side1[0])} {side1[1]} and the other side of "
        f"{_fmt_value(side2[0])} {side2[1]}. What is its {quantity_label}, in {target_label}? {precision}"
    )
    converted = []
    working = []
    for value, unit in (side1, side2):
        c = Fraction(value) * _linear_ratio(unit, target)
        converted.append(c)
        if unit != target:
            working.append(f"{_fmt_value(value)} {unit} = {_fmt_value(c)} {target}.")
    c1, c2 = (_fmt_value(c) for c in converted)
    if quantity == "perimeter":
        working.append(f"Perimeter = 2 × ({c1} + {c2}) = {answer_text} {_bare_label(target_label)}.")
    else:
        working.append(f"Area = {c1} × {c2} = {answer_text} {_bare_label(target_label)}.")

    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer=numeric_answer(answer_value),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=pick_distractors(answer_text, distractor_strings, tolerance=Fraction(1, 2 * 10**places)),
    )
