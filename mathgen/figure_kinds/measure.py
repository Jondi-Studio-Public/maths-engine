"""Measurement figure kinds: clocks, a protractor, informal units and scales.

Every spec carries everything that is drawn (hand angles, tick values, unit
positions); the front end draws it and never evaluates maths. Each kind also
has a pure ``read_*`` function that re-derives what a learner reads from the
spec alone, which a generator's plausibility reader calls.

Kinds: ``clock``, ``clock_set``, ``protractor``, ``unit_strip``, ``scale``.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Iterable

from mathgen.figures import _clean, _is_num, check, register_kind

Number = int | float

_EPS = 1e-6


def _frac(v: Any) -> Fraction:
    """An exact Fraction for a number that came from a spec (rounded to 6 places)."""
    return Fraction(str(round(float(v), 6))).limit_denominator(10**6)


def _is_int_valued(v: Any) -> bool:
    return _is_num(v) and abs(v - round(v)) < _EPS


# ------------------------------------------------------------------- clock

CLOCK_NUMERALS = ("all", "quarters", "none")

# Hands closer than this (degrees) hide each other; 12:00 is the one conventional exception.
MIN_HAND_SEPARATION = 7


def hand_angles(hour: int, minute: int) -> tuple[Number, Number]:
    """(hour hand, minute hand) in degrees clockwise from 12 o'clock.

    The hour hand moves continuously: 30 degrees per hour plus half a degree per minute.
    """
    return _clean(30 * (hour % 12) + 0.5 * minute), _clean(6 * minute)


def clock(
    *,
    hour: int,
    minute: int,
    alt: str,
    numerals: str = "all",
    minute_ticks: bool = True,
    caption: str = "",
) -> dict:
    """An analogue clock face.

    - ``hour`` 1-12 and ``minute`` 0-59: the time the hands show.
    - ``hour_angle`` and ``minute_angle``: where each hand points, in degrees
      clockwise from 12 o'clock (0 = straight up, 90 = pointing at the 3). The
      renderer draws the hands from these. ``hour_angle`` is ``30 * (hour % 12) + 0.5 * minute``
      (the hour hand sits proportionally between numerals: at 3:45 it is three quarters of
      the way from the 3 to the 4) and ``minute_angle`` is ``6 * minute``; the validator
      rejects any other value.
    - The hour hand is drawn shorter and thicker than the minute hand. Both turn about the
      centre of a circular face whose 12 is at the top.
    - ``numerals``: ``"all"`` writes 1-12 inside the rim; ``"quarters"`` writes only 12, 3, 6, 9;
      ``"none"`` writes none (the 12 positions still get a long tick).
    - ``minute_ticks``: ``True`` draws 60 ticks round the rim (every fifth one longer, at each
      numeral); ``False`` draws only the 12 hour ticks. A minute that is not a multiple of 5
      needs the ticks, and so does a face with no numerals.
    - ``caption``: an optional short label shown under the face (``"A"``, ``"Start"``).
    """
    hour_angle, minute_angle = hand_angles(hour, minute)
    spec = {
        "kind": "clock",
        "alt": alt,
        "hour": hour,
        "minute": minute,
        "hour_angle": hour_angle,
        "minute_angle": minute_angle,
        "numerals": numerals,
        "minute_ticks": bool(minute_ticks),
    }
    if caption:
        spec["caption"] = caption
    return check(spec)


def _validate_clock(spec: dict) -> list[str]:
    problems = []
    hour, minute = spec.get("hour"), spec.get("minute")
    if not (isinstance(hour, int) and not isinstance(hour, bool) and 1 <= hour <= 12):
        problems.append("hour must be an integer from 1 to 12")
    if not (isinstance(minute, int) and not isinstance(minute, bool) and 0 <= minute <= 59):
        problems.append("minute must be an integer from 0 to 59")
    if spec.get("numerals") not in CLOCK_NUMERALS:
        problems.append(f"numerals must be one of {CLOCK_NUMERALS}")
    if not isinstance(spec.get("minute_ticks"), bool):
        problems.append("minute_ticks must be a bool")
    if "caption" in spec and not isinstance(spec["caption"], str):
        problems.append("caption must be a string")
    if problems:
        return problems
    want_h, want_m = hand_angles(hour, minute)
    if not (_is_num(spec.get("hour_angle")) and abs(spec["hour_angle"] - want_h) < _EPS):
        problems.append(f"hour_angle must be 30*hour + 0.5*minute = {want_h}")
    if not (_is_num(spec.get("minute_angle")) and abs(spec["minute_angle"] - want_m) < _EPS):
        problems.append(f"minute_angle must be 6*minute = {want_m}")
    if minute % 5 and not spec["minute_ticks"]:
        problems.append("a minute that is not a multiple of 5 needs minute_ticks")
    if spec["numerals"] == "none" and not spec["minute_ticks"]:
        problems.append("a face with no numerals needs minute_ticks")
    gap = abs(want_h - want_m)
    gap = min(gap, 360 - gap)
    if gap < MIN_HAND_SEPARATION and not (hour == 12 and minute == 0):
        problems.append(f"the hands are only {gap} degrees apart and hide each other")
    return problems


def read_clock(spec: dict) -> tuple[int, int]:
    """(hour 1-12, minute) read from the hand angles alone.

    The minute comes from the minute hand; the hour is the numeral the hour hand has
    last passed, and the hour hand's offset into that hour must agree with the minute
    (half a degree per minute), otherwise the face is implausible.
    """
    minute = round(spec["minute_angle"] / 6) % 60
    hour_angle = spec["hour_angle"] % 360
    hour = int(hour_angle // 30) % 12 or 12
    if abs((hour_angle - 30 * (hour % 12)) - 0.5 * minute) > 0.01:
        raise ValueError("the hour hand disagrees with the minute hand")
    return hour, minute


def _clock_texts(spec: dict) -> list[str]:
    return [spec["caption"]] if spec.get("caption") else []


# ---------------------------------------------------------------- clock_set


def clock_set(*, clocks: Iterable[dict], alt: str) -> dict:
    """Two to four clock faces drawn side by side (or in a grid) for a comparison.

    - ``clocks``: a list of 2-4 ``clock`` specs (each exactly as :func:`clock` returns it,
      ``kind`` and ``alt`` included), each with a distinct non-empty ``caption`` (``"A"``,
      ``"B"``, ... or ``"Start"``/``"End"``), drawn in list order, left to right. Faces share
      one size.
    """
    return check({"kind": "clock_set", "alt": alt, "clocks": [dict(c) for c in clocks]})


def _validate_clock_set(spec: dict) -> list[str]:
    clocks = spec.get("clocks")
    if not isinstance(clocks, list) or not 2 <= len(clocks) <= 4:
        return ["clocks must be a list of 2 to 4 clock specs"]
    problems = []
    captions = []
    for i, c in enumerate(clocks):
        if not isinstance(c, dict) or c.get("kind") != "clock":
            problems.append(f"clocks[{i}] must be a clock spec")
            continue
        problems.extend(f"clocks[{i}]: {p}" for p in _validate_clock(c))
        if not c.get("caption"):
            problems.append(f"clocks[{i}] needs a caption")
        captions.append(c.get("caption"))
    if len(set(captions)) != len(captions):
        problems.append("clock captions must be distinct")
    return problems


def _clock_set_texts(spec: dict) -> list[str]:
    return [c["caption"] for c in spec.get("clocks", ()) if c.get("caption")]


# --------------------------------------------------------------- protractor

MIN_PROTRACTOR_ANGLE = 10
MAX_PROTRACTOR_ANGLE = 170


def protractor(
    *,
    angle: int,
    rotation: int,
    zero_on: str,
    alt: str,
    reflex: bool = False,
    names: dict | None = None,
) -> dict:
    """A semicircular protractor laid on an angle, with the angle's two rays drawn.

    Frame. Work in the protractor's own frame first: its centre (the vertex of the angle) is
    the middle of the straight edge, the straight edge is the baseline, and the semicircle
    bulges upwards. Two scales run round the arc, both 0 to 180 in steps of 5 (a short tick),
    with every 10th a longer, labelled tick. The OUTER scale starts at 0 on the right-hand end
    of the baseline and counts up anticlockwise to 180 at the left end; the INNER scale
    starts at 0 on the left-hand end and counts up clockwise to 180 at the right end. So a
    ray at local anticlockwise angle ``t`` from the right-hand end reads ``t`` on the outer
    scale and ``180 - t`` on the inner scale.

    Placement. The whole protractor, rays included, is then rotated anticlockwise by
    ``rotation`` degrees about the vertex (``0`` = baseline horizontal, upright as above).
    All directions below are SCREEN directions in degrees anticlockwise from the positive
    x-axis (pointing right).

    - ``angle``: the size, in degrees, of the angle between the two rays that the scale
      shows, an integer multiple of 5 from 10 to 170 other than 90 (where both scales agree).
    - ``zero_on``: ``"outer"`` or ``"inner"``, the scale whose 0 lies on the BASE ray.
      ``"outer"`` puts the base ray along the right-hand end of the baseline, so the angle is
      read on the outer scale; ``"inner"`` puts it along the left-hand end, so the angle is
      read on the inner scale (and the other scale gives ``180 - angle``).
    - ``rotation``: integer degrees 0-359 as above.
    - ``base_direction``: screen direction of the base ray, ``rotation`` for ``"outer"`` and
      ``rotation + 180`` for ``"inner"`` (mod 360).
    - ``second_direction``: screen direction of the second ray, ``rotation + angle`` for
      ``"outer"`` and ``rotation + 180 - angle`` for ``"inner"`` (mod 360).
    - Both rays are drawn from the vertex, longer than the protractor's radius, so each
      crosses the scales.
    - ``reflex``: ``False`` marks the small angle between the rays with an arc; ``True`` marks
      the reflex angle instead (the arc runs the long way round, outside the protractor),
      whose size is ``360 - angle``.
    - ``names``: optional ``{"vertex": "O", "base": "A", "second": "B"}``, single capitals
      written at the vertex and the ray ends.
    """
    base = rotation if zero_on == "outer" else rotation + 180
    second = rotation + angle if zero_on == "outer" else rotation + 180 - angle
    spec = {
        "kind": "protractor",
        "alt": alt,
        "angle": angle,
        "zero_on": zero_on,
        "rotation": rotation,
        "base_direction": base % 360,
        "second_direction": second % 360,
        "reflex": bool(reflex),
    }
    if names:
        spec["names"] = dict(names)
    return check(spec)


def _validate_protractor(spec: dict) -> list[str]:
    problems = []
    angle, rotation = spec.get("angle"), spec.get("rotation")
    if not (_is_int_valued(angle) and angle % 5 == 0 and MIN_PROTRACTOR_ANGLE <= angle <= MAX_PROTRACTOR_ANGLE):
        problems.append("angle must be a multiple of 5 from 10 to 170")
    elif angle == 90:
        problems.append("angle 90 reads the same on both scales")
    if spec.get("zero_on") not in ("outer", "inner"):
        problems.append("zero_on must be 'outer' or 'inner'")
    if not (_is_int_valued(rotation) and 0 <= rotation < 360):
        problems.append("rotation must be an integer from 0 to 359")
    if not isinstance(spec.get("reflex"), bool):
        problems.append("reflex must be a bool")
    names = spec.get("names")
    if names is not None:
        if not isinstance(names, dict) or not set(names) <= {"vertex", "base", "second"}:
            problems.append("names may hold only vertex, base and second")
        else:
            for k, v in names.items():
                if not (isinstance(v, str) and len(v) == 1 and v.isupper()):
                    problems.append(f"names[{k!r}] must be a single capital letter")
    if problems:
        return problems
    for key in ("base_direction", "second_direction"):
        if not _is_num(spec.get(key)):
            return [f"{key} must be a number"]
    try:
        reading = read_protractor_scale(spec)
    except ValueError as exc:
        return [str(exc)]
    if reading != angle:
        problems.append("the rays do not make the stated angle on the scale that starts at the base ray")
    return problems


def read_protractor_scale(spec: dict) -> int:
    """The number on the scale whose 0 is on the base ray, under the second ray.

    Derived from ``rotation``, ``base_direction`` and ``second_direction`` alone. A reflex angle's
    size is ``360`` minus this (see :func:`read_protractor`).
    """
    rot = spec["rotation"]
    local = (spec["second_direction"] - rot) % 360  # anticlockwise from the protractor's right-hand end
    base_local = (spec["base_direction"] - rot) % 360
    if base_local not in (0, 180):
        raise ValueError("the base ray does not lie along the baseline")
    if not 0 < local < 180:
        raise ValueError("the second ray is not over the protractor")
    reading = local if base_local == 0 else 180 - local
    if abs(reading - round(reading)) > _EPS:
        raise ValueError("the second ray does not land on a whole degree")
    return int(round(reading))


def read_protractor(spec: dict) -> int:
    """The size of the angle the figure marks: the scale reading, or ``360`` minus it if ``reflex``."""
    reading = read_protractor_scale(spec)
    return 360 - reading if spec["reflex"] else reading


def _protractor_texts(spec: dict) -> list[str]:
    return list((spec.get("names") or {}).values())


# --------------------------------------------------------------- unit_strip

UNIT_STYLES = ("paper_clip", "cube", "block", "footprint", "counter")
MAX_STRIP_UNITS = 14
MAX_STRIP_ROWS = 3
MAX_STRIP_EXTENT = 14  # in unit lengths


def unit_strip(*, unit: str, unit_style: str, rows: Iterable[dict], alt: str) -> dict:
    """Objects measured by informal units laid along them.

    All horizontal distances are in UNIT LENGTHS: 1 is the nominal width of one unit.

    - ``unit``: the singular name of the unit ("paper clip", "cube"), written as a small
      caption under the figure.
    - ``unit_style``: which picture the renderer repeats, one of
      ``"paper_clip"``, ``"cube"``, ``"block"``, ``"footprint"``, ``"counter"``.
    - ``rows``: 1-3 rows drawn top to bottom, each ``{"label": str, "length": number,
      "units": [{"start": number, "length": number}, ...]}``:

      * ``label``: the object's name ("pencil"), written at the left of its bar.
      * ``length``: how long the object is, in unit lengths: a multiple of 0.5. Its bar
        runs from x = 0 to x = ``length`` and a short end line is drawn at both ends.
      * ``units``: the repeated units drawn directly beneath the bar, left to right,
        each a picture ``length`` wide starting at x = ``start`` (a multiple of 0.25).
        The first unit starts at 0. A unit may overhang the end of the object: the bar's
        end line then crosses the unit, and it must cross at the unit's end or its exact
        middle. A gap between units shows as empty space, an overlap as units drawn over
        one another (the later unit on top).
    """
    spec = {
        "kind": "unit_strip",
        "alt": alt,
        "unit": unit,
        "unit_style": unit_style,
        "rows": [
            {
                "label": r["label"],
                "length": _clean(r["length"]),
                "units": [{"start": _clean(u["start"]), "length": _clean(u["length"])} for u in r["units"]],
            }
            for r in rows
        ],
    }
    return check(spec)


def _on_quarter(v: Any) -> bool:
    return _is_num(v) and (_frac(v) * 4).denominator == 1


def _validate_unit_strip(spec: dict) -> list[str]:
    problems = []
    if not (isinstance(spec.get("unit"), str) and spec["unit"].strip()):
        problems.append("unit needs a name")
    if spec.get("unit_style") not in UNIT_STYLES:
        problems.append(f"unit_style must be one of {UNIT_STYLES}")
    rows = spec.get("rows")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_STRIP_ROWS:
        return problems + [f"rows must be a list of 1 to {MAX_STRIP_ROWS}"]
    labels = []
    for i, row in enumerate(rows):
        where = f"rows[{i}]"
        if not (isinstance(row.get("label"), str) and row["label"].strip()):
            problems.append(f"{where} needs a label")
        labels.append(row.get("label"))
        length, units = row.get("length"), row.get("units")
        if not (_is_num(length) and 0 < length <= MAX_STRIP_EXTENT and (_frac(length) * 2).denominator == 1):
            problems.append(f"{where}.length must be a positive multiple of 0.5 up to {MAX_STRIP_EXTENT}")
            continue
        if not isinstance(units, list) or not 1 <= len(units) <= MAX_STRIP_UNITS:
            problems.append(f"{where}.units must hold 1 to {MAX_STRIP_UNITS} units")
            continue
        if not all(
            isinstance(u, dict) and _on_quarter(u.get("start")) and _on_quarter(u.get("length")) and u["length"] > 0
            for u in units
        ):
            problems.append(f"{where}: every unit needs a start and a length that are multiples of 0.25")
            continue
        problems.extend(f"{where}: {p}" for p in _strip_row_problems(row))
    if len(set(labels)) != len(labels):
        problems.append("row labels must be distinct")
    return problems


def _strip_row_problems(row: dict) -> list[str]:
    problems = []
    length = _frac(row["length"])
    units = [(_frac(u["start"]), _frac(u["length"])) for u in row["units"]]
    if units[0][0] != 0:
        problems.append("the first unit must start at 0")
    if any(units[i][0] < units[i - 1][0] for i in range(1, len(units))):
        problems.append("units must be listed left to right")
    if any(w > 2 or w < Fraction(1, 2) for _s, w in units):
        problems.append("a unit must be between 0.5 and 2 unit lengths wide")
    gaps = [units[i][0] - (units[i - 1][0] + units[i - 1][1]) for i in range(1, len(units))]
    if any(g > 0 for g in gaps) and any(g < 0 for g in gaps):
        problems.append("a row cannot mix a gap with an overlap")
    last_start, last_w = units[-1]
    if not last_start < length:
        problems.append("a unit lies wholly beyond the end of the object")
    if last_start + last_w < length:
        problems.append("the units stop short of the end of the object")
    elif length not in (last_start + last_w, last_start + last_w / 2):
        problems.append("the object must end at the end or the exact middle of its last unit")
    return problems


def classify_unit_row(row: dict) -> str:
    """How fair a row's measurement is, from its units alone.

    ``"unequal"`` if the units differ in width, else ``"gaps"`` if any space separates two
    units, else ``"overlaps"`` if any two overlap, else ``"fair"`` (equal units end to end).
    """
    units = [(_frac(u["start"]), _frac(u["length"])) for u in row["units"]]
    if len({w for _s, w in units}) > 1:
        return "unequal"
    ends = [s + w for s, w in units]
    gaps = [units[i][0] - ends[i - 1] for i in range(1, len(units))]
    if any(g > 0 for g in gaps):
        return "gaps"
    if any(g < 0 for g in gaps):
        return "overlaps"
    return "fair"


def read_unit_row(row: dict) -> Fraction:
    """The object's length in units for a ``"fair"`` row: its ``length`` over the unit width, so
    whole units to its end plus a half if it ends in the middle of a unit. Raises ValueError for
    an unfair row."""
    if classify_unit_row(row) != "fair":
        raise ValueError("the units are not laid fairly end to end")
    return _frac(row["length"]) / _frac(row["units"][0]["length"])


def _unit_strip_texts(spec: dict) -> list[str]:
    return [spec.get("unit", ""), *(r.get("label", "") for r in spec.get("rows", ()))]


# -------------------------------------------------------------------- scale

SCALE_VARIANTS = ("ruler", "thermometer", "jug", "dial")
# the most minor intervals a variant can draw legibly at phone width, and the most labelled majors
_SCALE_LIMITS = {"ruler": (80, 20), "thermometer": (60, 14), "jug": (60, 12), "dial": (60, 12)}


def scale(
    *,
    variant: str,
    unit: str,
    min_: Number,
    max_: Number,
    major: Number,
    minor: Number,
    alt: str,
    level: Number | None = None,
    object_: dict | None = None,
    marks: Iterable[dict] = (),
    extra_ticks: Iterable[Number] = (),
    start_angle: int | None = None,
    sweep: int | None = None,
) -> dict:
    """A measuring scale with something indicated on it.

    Ticks run from ``min`` to ``max``: a minor tick at every ``minor`` and a longer tick at
    every ``major`` counted from ``min``. Every major tick is labelled with its value; minor
    ticks are not. ``major`` is a whole multiple (1, 2, 4, 5 or 10) of ``minor`` and ``max - min``
    a whole number of majors. ``unit`` (``"cm"``, ``"mL"``, ``"kg"``, ``"°C"``) is written once beside
    the scale.

    - ``variant``:

      * ``"ruler"``: horizontal, ticks along the lower edge of the ruler, ``min`` at the left,
        increasing to the right. ``min`` need not be 0 (a broken or offset ruler). ``object``
        ``{"start": v, "end": v, "label"?: name}`` draws a bar lying along the ruler, directly
        above it, from value ``start`` to value ``end``, with a short vertical line down to the
        scale at each end.
      * ``"thermometer"``: vertical, ``min`` at the bottom, with a bulb. The scale may be
        negative. ``level`` fills the tube with liquid up to that value.
      * ``"jug"``: a vertical measuring jug or cylinder, ticks up one side from ``min`` (0) at the
        base. ``level`` draws the liquid surface at that value, the liquid filled below it.
      * ``"dial"``: a round kitchen-scale or gauge face. ``min`` sits at ``start_angle`` and the
        values run clockwise round the face through ``sweep`` degrees to ``max``; angles are
        degrees clockwise from 12 o'clock (``start_angle=-135, sweep=270`` puts 0 at the lower
        left and the maximum at the lower right). ``level`` is where the needle points.
    - ``level`` (thermometer, jug, dial) or ``object`` (ruler): the single indicated value(s).
    - ``marks``: 0-2 labelled arrow markers ``{"value": v, "label": "A"}`` drawn beside the
      scale pointing at ``value`` (on a dial, a small marker on the rim). Use them instead of
      ``level``/``object`` when a question compares two readings of the same scale.
    - ``extra_ticks``: values halfway between two minor ticks that get an extra short tick, so
      a reading of that value is still read from a mark.

    Exactly one of ``level``/``object`` and ``marks`` is given. Every indicated value must sit
    on a minor tick or an extra tick.
    """
    spec: dict = {
        "kind": "scale",
        "alt": alt,
        "variant": variant,
        "unit": unit,
        "min": _clean(min_),
        "max": _clean(max_),
        "major": _clean(major),
        "minor": _clean(minor),
        "extra_ticks": [_clean(t) for t in extra_ticks],
        "marks": [{"value": _clean(m["value"]), "label": m["label"]} for m in marks],
    }
    if level is not None:
        spec["level"] = _clean(level)
    if object_ is not None:
        spec["object"] = {k: (_clean(v) if k in ("start", "end") else v) for k, v in object_.items()}
    if variant == "dial":
        spec["start_angle"] = start_angle
        spec["sweep"] = sweep
    return check(spec)


def _on_grid(v: Any, origin: Fraction, step: Fraction, divisor: int = 1) -> bool:
    return _is_num(v) and ((_frac(v) - origin) / (step / divisor)).denominator == 1


def _validate_scale(spec: dict) -> list[str]:
    problems = []
    variant = spec.get("variant")
    if variant not in SCALE_VARIANTS:
        return [f"variant must be one of {SCALE_VARIANTS}"]
    if not (isinstance(spec.get("unit"), str) and spec["unit"].strip()):
        problems.append("unit needs a name")
    numbers = [spec.get(k) for k in ("min", "max", "major", "minor")]
    if (
        not all(_is_num(v) for v in numbers)
        or not spec["min"] < spec["max"]
        or spec["major"] <= 0
        or spec["minor"] <= 0
    ):
        return problems + ["min < max with positive major and minor steps are required"]
    lo, hi, major, minor = (_frac(v) for v in numbers)
    ratio = major / minor
    if ratio.denominator != 1 or int(ratio) not in (1, 2, 4, 5, 10):
        problems.append("major must be 1, 2, 4, 5 or 10 times minor")
    n_major = (hi - lo) / major
    if n_major.denominator != 1:
        problems.append("max - min must be a whole number of majors")
    if problems:
        return problems
    max_minor, max_major = _SCALE_LIMITS[variant]
    if (hi - lo) / minor > max_minor:
        problems.append(f"more than {max_minor} minor intervals; the ticks would be unreadable")
    if n_major > max_major:
        problems.append(f"more than {max_major} labelled majors; the labels would overlap")
    if variant == "jug" and lo != 0:
        problems.append("a jug scale starts at 0")
    if variant == "dial":
        sa, sw = spec.get("start_angle"), spec.get("sweep")
        if not (_is_int_valued(sa) and -180 <= sa <= 180):
            problems.append("a dial needs an integer start_angle from -180 to 180")
        if not (_is_int_valued(sw) and 180 <= sw <= 340):
            problems.append("a dial needs an integer sweep from 180 to 340")
    elif spec.get("start_angle") is not None or spec.get("sweep") is not None:
        problems.append("start_angle and sweep belong to a dial only")

    extras = spec.get("extra_ticks", [])
    if not isinstance(extras, list):
        return problems + ["extra_ticks must be a list"]
    extra_set = set()
    for t in extras:
        if not (_is_num(t) and lo < _frac(t) < hi and _on_grid(t, lo, minor, 2) and not _on_grid(t, lo, minor)):
            problems.append(f"extra tick {t} must lie halfway between two minor ticks")
        else:
            extra_set.add(_frac(t))

    def on_scale(v: Any, what: str) -> None:
        if not _is_num(v) or not lo <= _frac(v) <= hi:
            problems.append(f"{what} is not a value on the scale")
        elif not (_on_grid(v, lo, minor) or _frac(v) in extra_set):
            problems.append(f"{what} = {v} sits between ticks; it must be on a tick or a marked half")

    marks = spec.get("marks", [])
    if not isinstance(marks, list) or len(marks) > 2:
        return problems + ["marks must be a list of at most 2"]
    obj, level = spec.get("object"), spec.get("level")
    if (obj is not None) + (level is not None) > 1:
        problems.append("give one indicated thing, not several")
    if variant == "ruler":
        if level is not None:
            problems.append("a ruler indicates an object, not a level")
        if (obj is not None) == bool(marks):
            problems.append("a ruler needs exactly one of object and marks")
    else:
        if obj is not None:
            problems.append("only a ruler carries an object")
        if (level is not None) == bool(marks):
            problems.append(f"a {variant} needs exactly one of level and marks")
    if level is not None:
        on_scale(level, "level")
    if obj is not None:
        if not (isinstance(obj, dict) and _is_num(obj.get("start")) and _is_num(obj.get("end"))):
            problems.append("object needs numeric start and end")
        else:
            on_scale(obj["start"], "object start")
            on_scale(obj["end"], "object end")
            if not obj["start"] < obj["end"]:
                problems.append("object start must be before its end")
            if "label" in obj and not isinstance(obj["label"], str):
                problems.append("object label must be a string")
    seen = set()
    for i, m in enumerate(marks):
        if not (
            isinstance(m, dict) and isinstance(m.get("label"), str) and len(m["label"]) == 1 and m["label"].isupper()
        ):
            problems.append(f"marks[{i}] needs a single capital letter label")
            continue
        seen.add(m["label"])
        on_scale(m.get("value"), f"marks[{i}]")
    if len(seen) != len(marks):
        problems.append("mark labels must be distinct")
    return problems


def scale_readings(spec: dict) -> list[Number]:
    """The values the figure indicates, derived from the spec: ``[start, end]`` for a ruler's
    object, ``[level]`` for a liquid level or needle, else each mark's value in listed order."""
    if spec.get("object") is not None:
        return [spec["object"]["start"], spec["object"]["end"]]
    if spec.get("level") is not None:
        return [spec["level"]]
    return [m["value"] for m in spec.get("marks", ())]


def _scale_texts(spec: dict) -> list[str]:
    out = [spec.get("unit", "")]
    if (spec.get("object") or {}).get("label"):
        out.append(spec["object"]["label"])
    out.extend(m["label"] for m in spec.get("marks", ()))
    return out


register_kind("clock", _validate_clock, texts=_clock_texts)
register_kind("clock_set", _validate_clock_set, texts=_clock_set_texts)
register_kind("protractor", _validate_protractor, texts=_protractor_texts)
register_kind("unit_strip", _validate_unit_strip, texts=_unit_strip_texts)
register_kind("scale", _validate_scale, texts=_scale_texts)
