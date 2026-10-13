"""Descriptions for the measure figure kinds: clocks, a protractor, informal units and scales.

Each says where things sit against the marks the picture draws (hands against numerals and minute marks, rays
against the printed scale, a pointer against labelled and small ticks) and never reads the time, the angle or
the measurement off for the learner.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.describe import count, join, num, register_describer

_WORDS = "zero one two three four five six seven eight nine ten eleven twelve".split()
_DENOMS = {
    2: "half",
    3: "third",
    4: "quarter",
    5: "fifth",
    6: "sixth",
    8: "eighth",
    10: "tenth",
    12: "twelfth",
    15: "fifteenth",
    20: "twentieth",
    30: "thirtieth",
    60: "sixtieth",
}


def _opening(n: int, noun: str) -> str:
    """``Two clocks``: a count in words that can begin a sentence."""
    text = count(n, noun)
    _, _, rest = text.partition(" ")
    return f"{_WORDS[n].capitalize()} {rest}" if n < len(_WORDS) else text


def _fraction_words(f: Fraction) -> str:
    """``a quarter``, ``three quarters``, ``17 sixtieths``, or ``5 out of 7`` when no common name fits."""
    p, q = f.numerator, f.denominator
    name = _DENOMS.get(q)
    if name is None:
        return f"{p} out of {q}"
    top = "a" if p == 1 else (_WORDS[p] if p < len(_WORDS) else str(p))
    return f"{top} {name}" if p == 1 else f"{top} {name}s"


def _compass(deg: float) -> str:
    """A screen direction, degrees anticlockwise from pointing right, as words."""
    words = [
        "to the right",
        "to the upper right",
        "straight up",
        "to the upper left",
        "to the left",
        "to the lower left",
        "straight down",
        "to the lower right",
    ]
    d = deg % 360
    word = words[round(d / 45) % 8]
    return word if d % 45 == 0 else f"roughly {word}"


def _place(deg: float) -> str:
    """A position round a dial, degrees clockwise from 12 o'clock, as words."""
    words = [
        "the top",
        "the upper right",
        "the right",
        "the lower right",
        "the bottom",
        "the lower left",
        "the left",
        "the upper left",
    ]
    d = deg % 360
    word = words[round(d / 45) % 8]
    return word if d % 45 == 0 else f"roughly {word}"


# ---------------------------------------------------------------- clocks


def _mark(spec: dict, n: int) -> str:
    """Hour position ``n`` (0 is the top) in the terms this face lets a learner use."""
    n %= 12
    numeral = n or 12
    style = spec["numerals"]
    if style == "all" or (style == "quarters" and n % 3 == 0):
        return f"the {numeral}"
    if style == "quarters":
        base = n // 3 * 3
        return f"the long tick {count(n - base, 'step')} past the {base or 12}"
    return "the long tick at the top" if n == 0 else f"the long tick {count(n, 'step')} clockwise from the top"


def _face(spec: dict) -> str:
    numerals = {
        "all": "The numerals 1 to 12 are written round the rim.",
        "quarters": "Only the numerals 12, 3, 6 and 9 are written. The other hour positions are marked by long ticks.",
        "none": "No numerals are written. The 12 hour positions are marked by long ticks, one at the very top.",
    }[spec["numerals"]]
    ticks = (
        "There are 60 small ticks round the rim, with a longer tick at each hour position."
        if spec["minute_ticks"]
        else "Only the 12 hour ticks are drawn, with no minute marks."
    )
    return f"{numerals} {ticks}"


def _hands(spec: dict) -> str:
    m, h = spec["minute"], spec["hour"] % 12
    k, idx = m % 5, m // 5
    if m == 0:
        minute = f"The minute hand points straight up, at {_mark(spec, 0)}."
    elif k == 0:
        minute = f"The minute hand points at {_mark(spec, idx)}."
    else:
        minute = (
            f"The minute hand points {count(k, 'minute mark')} past {_mark(spec, idx)}, "
            f"on the way to {_mark(spec, idx + 1)}."
        )
    if m == 0:
        hour = f"The hour hand points at {_mark(spec, h)}."
    else:
        f = Fraction(m, 60)
        if m == 30:
            how = "halfway"
        elif f.denominator in (3, 4, 6, 12):
            how = f"{_fraction_words(f)} of the way"
        else:
            how = f"{m} sixtieths of the way"
        hour = f"The hour hand is {how} from {_mark(spec, h)} to {_mark(spec, h + 1)}."
    return f"{minute} {hour}"


def _describe_clock(spec: dict) -> str:
    out = [
        "An analogue clock face with two hands, the short thick hour hand and the long thin minute hand, "
        "turning about the centre."
    ]
    if spec.get("caption"):
        out.append(f"The caption under the clock reads {spec['caption']}.")
    out.append(_face(spec))
    out.append(_hands(spec))
    return " ".join(out)


def _describe_clock_set(spec: dict) -> str:
    clocks = spec["clocks"]
    caps = [c["caption"] for c in clocks]
    out = [
        f"{_opening(len(clocks), 'analogue clock face')} drawn side by side, captioned {join(caps)}. "
        "On every clock the hour hand is short and thick and the minute hand is long and thin."
    ]
    faces = {_face(c) for c in clocks}
    if len(faces) == 1:
        out.append(f"All the faces are marked the same way. {next(iter(faces))}")
    for c in clocks:
        parts = [f"Clock {c['caption']}."]
        if len(faces) > 1:
            parts.append(_face(c))
        parts.append(_hands(c))
        out.append(" ".join(parts))
    return " ".join(out)


# ---------------------------------------------------------------- protractor


def _scale_position(reading: int) -> str:
    if reading % 10 == 0:
        return f"exactly at the long tick labelled {reading}"
    return f"at the short tick halfway between the labelled {reading - 5} and {reading + 5}"


def _describe_protractor(spec: dict) -> str:
    rot, zero_on = spec["rotation"], spec["zero_on"]
    local = round((spec["second_direction"] - rot) % 360)
    outer = local
    inner = 180 - local
    out = [
        "A semicircular protractor laid over an angle. The vertex of the angle is at the middle of the "
        "protractor's straight edge, and two rays are drawn from it, each longer than the protractor's radius."
    ]
    if rot == 0:
        out.append("The protractor is upright, with its straight edge along the bottom and its curved edge above.")
    else:
        out.append(f"The protractor is tilted, so that its curved edge bulges towards {_place(-rot)}.")
    out.append(
        "Two scales of numbers run round the curved edge. Seen with the protractor upright, the outer scale starts "
        "at 0 at the right-hand end of the straight edge and counts up anticlockwise to 180 at the left-hand end, "
        "and the inner scale starts at 0 at the left-hand end and counts up clockwise to 180 at the right-hand end. "
        "Every 10 is a longer labelled tick, with a short tick at each 5 between."
    )
    if zero_on == "outer":
        out.append(
            f"The base ray points {_compass(spec['base_direction'])}, along the end of the straight edge where "
            "the outer scale reads 0 and the inner scale reads 180."
        )
    else:
        out.append(
            f"The base ray points {_compass(spec['base_direction'])}, along the end of the straight edge where "
            "the inner scale reads 0 and the outer scale reads 180."
        )
    out.append(
        f"The second ray points {_compass(spec['second_direction'])}. It crosses the outer scale {_scale_position(outer)}, "
        f"and it crosses the inner scale {_scale_position(inner)}."
    )
    if spec["reflex"]:
        out.append(
            "An arc outside the protractor marks the larger angle, running the long way round from the base ray "
            "to the second ray."
        )
    else:
        out.append("A small arc inside the protractor marks the angle between the base ray and the second ray.")
    names = spec.get("names") or {}
    placed = []
    if names.get("vertex"):
        placed.append(f"{names['vertex']} at the vertex")
    if names.get("base"):
        placed.append(f"{names['base']} at the end of the base ray")
    if names.get("second"):
        placed.append(f"{names['second']} at the end of the second ray")
    if placed:
        out.append(f"The letters written on the figure are {join(placed)}.")
    return " ".join(out)


# ---------------------------------------------------------------- unit_strip


def _frac(v) -> Fraction:
    return Fraction(str(round(float(v), 6))).limit_denominator(10**6)


def _of_a_unit(r: Fraction) -> str:
    if r == 1:
        return "as wide as a unit"
    if r < 1:
        return f"{_fraction_words(r)} of a unit's width"
    return f"{num(float(r))} times a unit's width"


def _strip_row(i: int, row: dict, unit: str) -> str:
    units = [(_frac(u["start"]), _frac(u["length"])) for u in row["units"]]
    n = len(units)
    out = [
        f"Row {i + 1} is labelled {row['label']}. It has a bar with a short end line at each end, and "
        f"{count(n, unit)} drawn directly below the bar, the first starting at the left end line."
    ]
    first_w = units[0][1]
    widths = [w for _s, w in units]
    if len(set(widths)) > 1:
        rel = ["the same" if w == first_w else ("wider" if w > first_w else "narrower") for w in widths[1:]]
        out.append(
            f"The units are not all the same width. Compared with the first, the others in order are {join(rel)}."
        )
    else:
        out.append("The units are all the same width.")
    if n > 1:
        gaps = [units[k][0] - (units[k - 1][0] + units[k - 1][1]) for k in range(1, n)]
        if all(g == 0 for g in gaps):
            out.append("Each unit touches the next, end to end, with no gaps and no overlaps.")
        elif all(g > 0 for g in gaps) and len(set(gaps)) == 1:
            out.append(f"There is a gap between each pair of units, {_of_a_unit(gaps[0] / first_w)}.")
        elif all(g < 0 for g in gaps) and len(set(gaps)) == 1:
            out.append(
                f"Each unit overlaps the one before it by {_of_a_unit(-gaps[0] / first_w)}, with the later unit drawn on top."
            )
        else:
            pos = [f"after unit {k}" for k, g in enumerate(gaps, 1) if g > 0]
            neg = [f"between units {k} and {k + 1}" for k, g in enumerate(gaps, 1) if g < 0]
            if pos:
                out.append(f"There is a gap {join(pos)}.")
            if neg:
                out.append(f"Units overlap {join(neg)}, with the later unit drawn on top.")
    last_s, last_w = units[-1]
    if _frac(row["length"]) == last_s + last_w:
        out.append("The right-hand end line of the bar falls at the right-hand edge of the last unit.")
    else:
        out.append("The right-hand end line of the bar passes through the middle of the last unit.")
    return " ".join(out)


def _describe_unit_strip(spec: dict) -> str:
    unit = spec["unit"]
    rows = spec["rows"]
    out = [
        f"{_opening(len(rows), 'object')} measured with {unit}s laid along {'it' if len(rows) == 1 else 'them'}. The rows are drawn from top to "
        "bottom, all to the same scale with their left ends lined up. "
        f"The caption at the bottom reads Unit: {unit}."
    ]
    out.extend(_strip_row(i, r, unit) for i, r in enumerate(rows))
    return " ".join(out)


# ---------------------------------------------------------------- scale


def _locator(spec: dict, v) -> str:
    """Where value ``v`` sits against the labelled and small ticks."""
    lo, major, minor = _frac(spec["min"]), _frac(spec["major"]), _frac(spec["minor"])
    ratio = int(major / minor)
    idx = (_frac(v) - lo) / minor
    q = int(idx // ratio)
    off = idx - q * ratio
    base = num(float(lo + q * major))
    nxt = num(float(lo + (q + 1) * major))
    if off == 0:
        return f"the long tick labelled {base}"
    if ratio == 1:
        return f"halfway between the long ticks labelled {base} and {nxt}"
    way = {"ruler": "to the right of", "dial": "clockwise from"}.get(spec["variant"], "above")
    whole = int(off)
    if off == whole:
        return f"{count(whole, 'small tick')} {way} the long tick labelled {base}"
    if whole == 0:
        return f"half a small interval {way} the long tick labelled {base}"
    return f"{whole} and a half small intervals {way} the long tick labelled {base}"


def _scale_core(spec: dict) -> str:
    lo, hi = _frac(spec["min"]), _frac(spec["max"])
    major, minor = _frac(spec["major"]), _frac(spec["minor"])
    ratio = int(major / minor)
    n_major = int((hi - lo) / major)
    labelled = [num(float(lo + i * major)) for i in range(n_major + 1)]
    if len(labelled) > 9:
        out = [f"The long ticks are labelled from {labelled[0]} to {labelled[-1]} in steps of {num(float(major))}."]
    else:
        out = [f"The long ticks are labelled {join(labelled)}."]
    if ratio > 1:
        out.append(
            f"Between each pair of neighbouring labelled ticks there are {ratio} equal small intervals, marked by small ticks."
        )
    else:
        out.append("There are no small ticks between the labelled ones.")
    n_small = int((hi - lo) / minor)
    if len(spec["extra_ticks"]) == n_small and ratio > 0:
        out.append("An extra very short tick sits halfway between every pair of neighbouring small ticks.")
    elif spec["extra_ticks"]:
        out.append(
            f"Extra very short ticks sit halfway between two small ticks, at: "
            f"{join([_locator(spec, t) for t in spec['extra_ticks']], 'and')}."
        )
    return " ".join(out)


def _marks_text(spec: dict, lead: str) -> str:
    items = [f"{m['label']} at {_locator(spec, m['value'])}" for m in spec["marks"]]
    return f"{lead} {join(items)}."


def _describe_scale(spec: dict) -> str:
    variant, unit = spec["variant"], spec["unit"]
    lo, hi = num(spec["min"]), num(spec["max"])
    level = spec.get("level")
    obj = spec.get("object")
    marks = spec["marks"]
    out: list[str] = []
    if variant == "ruler":
        out.append(
            f"A horizontal ruler with the ticks along its lower edge, the numbers increasing from {lo} at the left "
            f"end to {hi} at the right end. The unit {unit} is written on the ruler."
        )
        out.append(_scale_core(spec))
        if obj:
            name = f" labelled {obj['label']}" if obj.get("label") else ""
            out.append(
                f"A bar{name} lies along the ruler, directly above it, with a short vertical line dropping from each "
                f"end of the bar down to the scale. The left line meets the scale at {_locator(spec, obj['start'])}. "
                f"The right line meets the scale at {_locator(spec, obj['end'])}."
            )
        if marks:
            out.append(_marks_text(spec, "Above the ruler, lettered arrows point down to the scale. They point to:"))
    elif variant == "dial":
        start, sweep = spec["start_angle"], spec["sweep"]
        out.append(
            f"A round dial. The labelled numbers run clockwise round the face, from {lo} at {_place(start)} "
            f"to {hi} at {_place(start + sweep)}. The unit {unit} is written below the centre of the face."
        )
        out.append(_scale_core(spec))
        if level is not None:
            out.append(f"A needle from the centre points at {_locator(spec, level)}.")
        if marks:
            out.append(_marks_text(spec, "Small lettered arrow markers on the outside of the rim point to:"))
    else:
        if variant == "thermometer":
            out.append(
                f"A vertical thermometer with a round bulb at the bottom, the unit {unit} written above the tube. "
                f"The scale runs up the right-hand side of the tube, with {lo} at the bottom and {hi} at the top, "
                "and its numbers are written to the right of the ticks."
            )
        else:
            out.append(
                f"A vertical measuring jug with a spout at the top left, the unit {unit} written above the numbers. "
                f"The scale runs up the right-hand side of the jug from {lo} at the base to {hi} at the top, "
                "and its numbers are written to the right of the ticks."
            )
        out.append(_scale_core(spec))
        if level is not None:
            if variant == "thermometer":
                out.append(f"Liquid fills the bulb and the tube, and its top is level with {_locator(spec, level)}.")
            else:
                out.append(
                    f"Liquid fills the jug from the base up to a flat surface level with {_locator(spec, level)}."
                )
        if marks:
            out.append(
                _marks_text(spec, "On the left-hand side, lettered arrows point across to the scale. They point to:")
            )
    return " ".join(out)


register_describer("clock", _describe_clock)
register_describer("clock_set", _describe_clock_set)
register_describer("protractor", _describe_protractor)
register_describer("unit_strip", _describe_unit_strip)
register_describer("scale", _describe_scale)
