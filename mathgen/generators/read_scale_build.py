"""Reading a measuring scale drawn on the figure: a ruler, thermometer, jug or kitchen dial.

- ``scale_read_value`` reads ONE value (``answer: number``, graded to the precision of the scale):
  easy reads a positive thermometer, a jug or a ruler whose minor ticks are worth 1/2 or 1/5 of a
  labelled step; medium adds a thermometer below zero, ticks worth 1/4 or 1/10 of a step and a
  kitchen dial; hard reads a value on a marked half tick, or reads a centimetre ruler in
  millimetres.
- ``scale_compare_readings`` combines TWO readings: how much a thermometer rose or a jug filled
  (easy), how long an object is when it does not start at 0 on the ruler (medium), and the same
  across zero or on a ruler that itself starts part way along (hard).

Ticks are labelled only at the majors, so the value of a minor tick has to be worked out from the
two labels either side. Every indicated value sits on a tick (the validator insists), and every
distractor is a named misreading: counting each minor tick as 1, counting down from the label above,
dropping the minus sign, reading the nearest label, being one tick out. All arithmetic is in exact
fractions; answers are written to the scale's own number of decimal places so options look alike.
The readers work from the figure alone (and from the unit the prompt asks for).
"""

from __future__ import annotations

import re
from fractions import Fraction as F

from mathgen import figures
from mathgen.core import check, generator, pick_distractors
from mathgen.figure_kinds import measure

_TIERS = ("easy", "medium", "hard")
_PRESETS = {t: {"tier": t} for t in _TIERS}

# variant -> (scale unit drawn, unit words for a prompt)
_UNIT_WORDS = {
    "°C": "degrees Celsius",
    "mL": "millilitres",
    "cm": "centimetres",
    "mm": "millimetres",
    "kg": "kilograms",
    "g": "grams",
}

# (variant, unit, lo, hi, major, minor); the dial also gets 270 degrees of sweep
_DESIGNS = {
    "easy": [
        ("thermometer", "°C", 0, 50, 10, 5),
        ("thermometer", "°C", 0, 40, 10, 2),
        ("thermometer", "°C", 0, 30, 5, 1),
        ("jug", "mL", 0, 500, 100, 50),
        ("jug", "mL", 0, 1000, 100, 20),
        ("jug", "mL", 0, 250, 50, 10),
        ("ruler", "cm", 0, 10, 1, F(1, 2)),
        ("ruler", "cm", 0, 12, 1, F(1, 2)),
        ("ruler", "cm", 0, 8, 1, F(1, 5)),
    ],
    "medium": [
        ("thermometer", "°C", -10, 40, 10, 2),
        ("thermometer", "°C", -20, 30, 10, 5),
        ("thermometer", "°C", -10, 30, 5, 1),
        ("thermometer", "°C", -20, 20, 10, 2),
        ("jug", "mL", 0, 400, 100, 25),
        ("jug", "mL", 0, 1000, 250, 50),
        ("dial", "kg", 0, 5, 1, F(1, 5)),
        ("dial", "kg", 0, 2, F(1, 2), F(1, 10)),
        ("dial", "g", 0, 1000, 250, 50),
        ("ruler", "cm", 0, 6, 1, F(1, 10)),
        ("ruler", "cm", 0, 10, 1, F(1, 5)),
    ],
    "hard_half": [
        ("thermometer", "°C", -10, 40, 10, 2),
        ("thermometer", "°C", 0, 40, 10, 2),
        ("jug", "mL", 0, 500, 100, 50),
        ("jug", "mL", 0, 250, 50, 10),
        ("dial", "kg", 0, 5, 1, F(1, 5)),
        ("dial", "g", 0, 500, 100, 20),
    ],
    "hard_mm": [
        ("ruler", "cm", 0, 8, 1, F(1, 10)),
        ("ruler", "cm", 0, 7, 1, F(1, 10)),
        ("ruler", "cm", 0, 6, 1, F(1, 10)),
    ],
}

_OBJECTS = ("pencil", "stick", "ribbon", "leaf", "crayon", "straw", "feather", "strip of card")

_NAME = {"thermometer": "thermometer", "jug": "jug", "dial": "kitchen scale", "ruler": "ruler"}

_READ_POOLS = {
    "thermometer": (
        (
            "A thermometer is shown.",
            "The thermometer outside the window is shown.",
            "Here is a room thermometer.",
            "A weather thermometer is drawn below.",
            "The thermometer in the classroom is shown.",
        ),
        (
            "What temperature does it show, in degrees Celsius?",
            "Read the temperature in degrees Celsius.",
            "What is the temperature shown, in degrees Celsius?",
        ),
    ),
    "jug": (
        (
            "Water is in a measuring jug.",
            "A measuring jug has some juice in it.",
            "A jug is used to measure some milk.",
            "Here is a measuring jug with water in it.",
            "Some water is poured into a measuring cylinder.",
        ),
        (
            "How much liquid is in the jug, in millilitres?",
            "Read the amount of liquid in millilitres.",
            "What is the volume of liquid shown, in millilitres?",
        ),
    ),
    "dial": (
        (
            "A kitchen scale is weighing some flour.",
            "The needle on a kitchen scale is shown.",
            "Some fruit is weighed on a dial scale.",
            "A dial scale is weighing a parcel.",
            "A cook weighs some rice on a dial scale.",
        ),
        (
            "What mass does the scale show, in {w}?",
            "Read the mass shown, in {w}.",
            "How heavy is it, in {w}?",
        ),
    ),
    "ruler": (
        (
            "A {obj} lies along a ruler.",
            "The ruler is used to measure a {obj}.",
            "A {obj} is placed beside the ruler.",
            "Here a {obj} is measured with a ruler.",
            "The length of a {obj} is shown against a ruler.",
        ),
        (
            "How long is the {obj}, in centimetres?",
            "Find the length of the {obj} in centimetres.",
            "What is the length of the {obj}, in centimetres?",
        ),
    ),
}
_HARD_POOLS = {
    "thermometer": (
        (
            "Look at this thermometer.",
            "The thermometer on the patio is drawn below.",
            "Zoe reads a thermometer in the garden.",
            "A freezer thermometer is shown.",
        ),
        (
            "Give the temperature it shows in degrees Celsius.",
            "State the reading in degrees Celsius.",
            "What temperature is marked by the liquid, in degrees Celsius?",
        ),
    ),
    "jug": (
        (
            "Look at this measuring jug.",
            "Sam pours some water into a jug.",
            "The level of the water in a jug is shown.",
            "A cook has measured some stock in a jug.",
        ),
        (
            "Give the volume in millilitres.",
            "State the reading of the liquid level in millilitres.",
            "What volume does the jug hold, in millilitres?",
        ),
    ),
    "dial": (
        (
            "Look at this dial scale.",
            "Ali puts some potatoes on a dial scale.",
            "The needle of a dial scale is drawn below.",
            "A parcel is weighed on a dial scale.",
        ),
        (
            "Give the mass in {w}.",
            "State the reading in {w}.",
            "What mass does the needle point to, in {w}?",
        ),
    ),
    "ruler": (
        (
            "A {obj} is measured with a ruler marked in centimetres and tenths of a centimetre.",
            "Look at the {obj} beside the ruler. Each small mark is a tenth of a centimetre.",
            "The ruler has a mark for each tenth of a centimetre. A {obj} lies along it.",
            "Each small mark on this ruler is one millimetre. A {obj} lies along it.",
        ),
        (
            "How long is the {obj} in millimetres?",
            "Give the length of the {obj} in millimetres.",
            "Write the length of the {obj} in millimetres.",
        ),
    ),
}


def _pick(rng, pool):
    return pool[rng.randrange(len(pool))]


def _dp(step: F) -> int:
    d = 0
    while (step * 10**d).denominator != 1:
        d += 1
    return d


def _num(v: F, dp: int) -> str:
    return f"{float(v):.{dp}f}"


def _build_scale(variant, unit, lo, hi, major, minor, alt, *, level=None, obj=None, marks=(), halves=False):
    lo, hi, major, minor = F(lo), F(hi), F(major), F(minor)
    extras = []
    if halves:
        x = lo + minor / 2
        while x < hi:
            extras.append(float(x))
            x += minor
    kwargs = {}
    if variant == "dial":
        kwargs = dict(start_angle=-135, sweep=270)
    return measure.scale(
        variant=variant,
        unit=unit,
        min_=float(lo),
        max_=float(hi),
        major=float(major),
        minor=float(minor),
        alt=alt,
        level=None if level is None else float(level),
        object_=obj,
        marks=[{"value": float(v), "label": lab} for lab, v in marks],
        extra_ticks=extras,
        **kwargs,
    )


def _ticks(lo, hi, step):
    out, x = [], F(lo)
    while x <= hi:
        out.append(x)
        x += step
    return out


def _slot_text(rng, pools, variant, **slots):
    opener, question = pools[variant]
    return " ".join(_pick(rng, p).format(**slots) for p in (opener, question))


def _read_misreads(v, lo, hi, major, minor, dp):
    """Wrong readings of v: each named in the skill's misconceptions."""
    m_lo = lo + ((v - lo) // major) * major
    m_hi = m_lo + major
    k = (v - m_lo) / minor
    cands = [
        m_lo + k,  # counts every minor tick as 1
        m_hi - k * minor,  # counts down from the label above instead of up from the one below
        v + minor,
        v - minor,
        m_lo,  # reads the label below and ignores the ticks
        m_hi,
    ]
    if v < 0:
        cands.append(-v)  # drops the minus sign
    cands += [v + 2 * minor, v - 2 * minor, v + major, v - major]  # two ticks out, a whole label out
    return [c for c in cands if lo <= c <= hi and c != v]


def _scenes_for_temperature(openers, v):
    """Keep only the thermometer scenes that can show the reading v: an indoor room sits at 10 to 30 degrees
    and a freezer at or below zero."""
    keep = []
    for opener in openers:
        indoor = "room" in opener or "classroom" in opener
        if indoor and not 10 <= v <= 30:
            continue
        if "freezer" in opener and v > 0:
            continue
        keep.append(opener)
    return tuple(keep)


def _build_read(rng, tier: str):
    if tier == "hard":
        kind = _pick(rng, ("half", "half", "mm"))
        design = _pick(rng, _DESIGNS["hard_half" if kind == "half" else "hard_mm"])
    else:
        kind = None
        design = _pick(rng, _DESIGNS[tier])
    variant, unit, lo, hi, major, minor = design
    lo, hi, major, minor = F(lo), F(hi), F(major), F(minor)
    obj = _pick(rng, _OBJECTS)
    alt = f"a {obj} beside a ruler" if variant == "ruler" else f"a {_NAME[variant]}"
    if kind == "half":
        halves = _ticks(lo + minor / 2, hi - minor / 2, minor)
        v = _pick(rng, halves)
    else:
        ticks = [t for t in _ticks(lo, hi, minor) if lo < t < hi]
        non_major = [t for t in ticks if (t - lo) % major != 0]
        pool = non_major
        if tier == "medium" and variant == "thermometer":
            neg = [t for t in non_major if t < 0]
            pool = neg if neg and rng.randrange(3) else non_major
        if variant == "ruler":
            pool = [t for t in pool if t >= 2 * minor + lo]
        v = _pick(rng, pool)
    dp = _dp(minor / 2 if kind == "half" else minor)
    object_ = {"start": 0, "end": float(v), "label": obj} if variant == "ruler" else None
    fig = _build_scale(
        variant,
        unit,
        lo,
        hi,
        major,
        minor,
        alt,
        level=None if variant == "ruler" else v,
        obj=object_,
        halves=kind == "half",
    )
    w = _UNIT_WORDS[unit]
    pools = _HARD_POOLS if tier == "hard" else _READ_POOLS
    if variant == "thermometer":
        pools = {**pools, "thermometer": (_scenes_for_temperature(pools["thermometer"][0], v), pools["thermometer"][1])}
    text = _slot_text(rng, pools, variant, obj=obj, w=w)
    if kind == "mm":
        answer_v, dp = v * 10, 0
        cands = [c * 10 for c in _read_misreads(v, lo, hi, major, minor, dp)]
        cands += [answer_v + 10, answer_v - 10, answer_v + 1, answer_v - 1]
        cands = [c for c in cands if c != answer_v and c > 0]
    else:
        answer_v = v
        cands = _read_misreads(v, lo, hi, major, minor, dp)
    if kind == "mm":
        note = f"The ruler reads {_num(v, _dp(minor))} cm, which is {_num(answer_v, 0)} mm"
    elif kind == "half":
        note = (
            f"Each small step is {_num(minor, _dp(minor))} {unit}. The level is halfway between two small marks, "
            f"so the reading is {_num(answer_v, dp)} {unit}"
        )
    else:
        note = f"Each small step is {_num(minor, _dp(minor))} {unit}; the reading is {_num(answer_v, dp)} {unit}"
    return fig, text, _num(answer_v, dp), [_num(c, dp) for c in cands], note


@generator("scale_read_value", level=4, presets=_PRESETS)
def scale_read_value(rng, tier="easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    fig, text, answer, cands, note = _build_read(rng, tier)
    rng.shuffle(cands)
    distractors = pick_distractors(answer, cands, n=3)
    check(len(distractors) == 3, "the question needs three distinct distractors")
    figures.assert_no_leak(fig, [answer, *distractors])
    return dict(
        text=text,
        answer=float(answer) if "." in answer else int(answer),
        answer_text=answer,
        answer_latex=f"plain:{answer}",
        distractors=distractors,
        working=[note + "."],
        figure=fig,
    )


def _dp_of_figure(fig: dict) -> int:
    step = F(str(fig["minor"]))
    return _dp(step / 2 if fig["extra_ticks"] else step)


def _read_value(question) -> str:
    fig = question.figure
    start, *rest = measure.scale_readings(fig)
    value = F(str(rest[0])) - F(str(start)) if rest else F(str(start))
    if "millimetres" in question.text:
        return _num(value * 10, 0)
    return _num(value, _dp_of_figure(fig))


figures.register_reader("scale_read_value", _read_value)


# ------------------------------------------------------------------ two readings

_CMP_POOLS = {
    "thermometer": (
        (
            "The thermometer shows the temperature at dawn, marked A, and at noon, marked B.",
            "A thermometer shows two readings: A in the morning and B in the afternoon.",
            "The temperature on a thermometer is marked at two times: A at breakfast and B at lunch.",
            "Marks A and B show the temperature at two different times of the day.",
        ),
        (
            "By how many degrees did the temperature {verb} from A to B?",
            "How many degrees did the temperature {verb} between A and B?",
            "The temperature {verb2} from A to B. By how many degrees?",
        ),
    ),
    "jug": (
        (
            "The level in a measuring jug is marked A before more water is poured in and B afterwards.",
            "Mark A shows the water in a jug, and mark B shows it after some more is added.",
            "A jug holds water to mark A. Then more is poured in, up to mark B.",
            "Some juice is added to a jug, raising the level from A to B.",
        ),
        (
            "How many millilitres were poured in?",
            "How much more liquid, in millilitres, is in the jug at B than at A?",
            "Find the volume added, in millilitres.",
        ),
    ),
    "ruler": (
        (
            "A {obj} is lined up against a ruler.",
            "The {obj} lies along a ruler, but not from the end of it.",
            "The ruler measures a {obj} that starts part way along it.",
            "A ruler is held beside a {obj}.",
        ),
        (
            "How long is the {obj}, in centimetres?",
            "What is the length of the {obj}, in centimetres?",
            "Find the length of the {obj} in centimetres.",
        ),
    ),
}
_CMP_HARD_RULER = (
    (
        "The end of this ruler has broken off, so it starts part way along. A {obj} is lined up against it.",
        "A ruler with its first few centimetres snapped off measures a {obj}.",
        "This ruler does not start at 0. It is used to measure a {obj}.",
        "Part of the ruler has worn away. A {obj} lies along what is left.",
    ),
    _CMP_POOLS["ruler"][1],
)


def _cmp_design(rng, tier):
    if tier == "easy":
        return _pick(
            rng,
            [
                ("thermometer", "°C", 0, 50, 10, 5),
                ("thermometer", "°C", 0, 40, 10, 2),
                ("jug", "mL", 0, 500, 100, 50),
                ("jug", "mL", 0, 1000, 100, 20),
            ],
        )
    if tier == "medium":
        return _pick(
            rng,
            [("ruler", "cm", 0, 10, 1, F(1, 2)), ("ruler", "cm", 0, 12, 1, F(1, 2)), ("ruler", "cm", 0, 8, 1, F(1, 5))],
        )
    return _pick(
        rng,
        [
            ("thermometer", "°C", -10, 40, 10, 2),
            ("thermometer", "°C", -20, 30, 10, 5),
            ("thermometer", "°C", -10, 30, 5, 1),
            ("ruler", "cm", 3, 15, 1, F(1, 2)),
            ("ruler", "cm", 2, 12, 1, F(1, 5)),
            ("ruler", "cm", 4, 14, 1, F(1, 2)),
        ],
    )


def _cmp_misreads(a, b, d, lo, hi, minor):
    out = [abs(a) + abs(b) if (a < 0) == (b < 0) else abs(abs(a) - abs(b)), b, a, d + minor, d - minor, abs(a), abs(b)]
    return [c for c in out if c > 0 and c != d]


def _build_cmp(rng, tier: str):
    variant, unit, lo, hi, major, minor = _cmp_design(rng, tier)
    lo, hi, major, minor = F(lo), F(hi), F(major), F(minor)
    dp = _dp(minor)
    obj = _pick(rng, _OBJECTS)
    ticks = _ticks(lo, hi, minor)
    if variant == "ruler":
        majors = [t for t in ticks if (t - lo) % major == 0 and t < hi]
        for _ in range(500):
            starts = [t for t in majors if t > lo] if tier == "medium" else [t for t in ticks if lo < t]
            a = _pick(rng, [t for t in starts if t + 2 <= hi])
            ends = [t for t in ticks if a + 2 <= t <= a + 8 and (t - lo) % major != 0]
            if ends:
                b = _pick(rng, ends)
                break
        else:
            raise AssertionError("no ruler object found")
        check(tier != "hard" or lo > 0, "the hard ruler must not start at 0")
        obj_spec = {"start": float(a), "end": float(b), "label": obj}
        fig = _build_scale(variant, unit, lo, hi, major, minor, f"a {obj} beside a ruler", obj=obj_spec)
        pools = _CMP_HARD_RULER if tier == "hard" else _CMP_POOLS["ruler"]
        text = " ".join(_pick(rng, p).format(obj=obj) for p in pools)
        d = b - a
        cands = [c for c in [b, a + b, d + minor, d - minor, d + major, d - major, b - lo] if c > 0 and c != d]
        note = f"The {obj} runs from {_num(a, dp)} to {_num(b, dp)} on the ruler: {_num(b, dp)} - {_num(a, dp)} = {_num(d, dp)} {unit}."
        return fig, text, _num(d, dp), [_num(c, dp) for c in cands], note
    # two readings on a thermometer or a jug
    interior = [t for t in ticks if lo < t < hi]
    cross = rng.randrange(3) > 0
    for _ in range(1000):
        a, b = rng.sample(interior, 2)
        if tier == "easy":
            ok = b > a and b - a >= 2 * minor
        else:
            ok = a != 0 and b != 0 and abs(b - a) >= 3 * minor and ((a < 0) != (b < 0)) == cross
        if ok:
            break
    else:
        raise AssertionError("no pair of readings found")
    d = abs(b - a)
    if variant == "jug":
        verb = verb2 = None
    else:
        verb, verb2 = ("rise", "rose") if b > a else ("fall", "fell")
    fig = _build_scale(variant, unit, lo, hi, major, minor, f"a {_NAME[variant]}", marks=[("A", a), ("B", b)])
    opener, question = _CMP_POOLS[variant]
    text = " ".join(_pick(rng, p).format(verb=verb, verb2=verb2) for p in (opener, question))
    cands = [c for c in _cmp_misreads(a, b, d, lo, hi, minor)]
    note = f"A reads {_num(a, dp)} and B reads {_num(b, dp)}; the change is {_num(d, dp)} {unit}."
    return fig, text, _num(d, dp), [_num(c, dp) for c in cands], note


@generator("scale_compare_readings", level=5, presets=_PRESETS)
def scale_compare_readings(rng, tier="easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    fig, text, answer, cands, note = _build_cmp(rng, tier)
    rng.shuffle(cands)
    distractors = pick_distractors(answer, cands, n=3)
    check(len(distractors) == 3, "the question needs three distinct distractors")
    figures.assert_no_leak(fig, [answer, *distractors])
    return dict(
        text=text,
        answer=float(answer) if "." in answer else int(answer),
        answer_text=answer,
        answer_latex=f"plain:{answer}",
        distractors=distractors,
        working=[note],
        figure=fig,
    )


def _read_difference(question) -> str:
    fig = question.figure
    vals = measure.scale_readings(fig)
    a, b = F(str(vals[0])), F(str(vals[1]))
    if fig["marks"]:
        text = question.text
        if re.search(r"\b(rise|rose)\b", text):
            check(b > a, "the prompt says the temperature rose but B is not above A")
        elif re.search(r"\b(fall|fell)\b", text):
            check(b < a, "the prompt says the temperature fell but B is not below A")
        else:
            check(b > a, "more liquid was added, so B must be above A")
    return _num(abs(b - a), _dp(F(str(fig["minor"]))))


figures.register_reader("scale_compare_readings", _read_difference)
