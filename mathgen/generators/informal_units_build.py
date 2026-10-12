"""Measuring with informal units laid end to end, from the figure alone.

``informal_units_measure`` draws an object with identical units (paper clips, cubes,
footprints...) laid along it and asks a multiple-choice question about the measurement.

- easy: count how many units long the object is; the units are laid exactly end to end and
  the last one finishes at the end of the object.
- medium: either the object ends in the middle of a unit ("4 and a half") or two objects are
  measured with the same unit and the question asks how many more units one is than the other.
- hard: a learner says "it is N units long"; decide whether that measurement is fair. The units
  may be spaced with gaps (the object is really longer than N units), overlap (really shorter),
  differ in size (no count of them is a length), or be laid correctly. Exactly one of the four
  options is true.

Distractors come from named mistakes: counting the unit that is only part covered as a whole one,
dropping the part unit, adding the two lengths instead of subtracting, counting the end marks
instead of the gaps between them. The reader works from the figure alone, through
``classify_unit_row`` and ``read_unit_row``.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen import figures
from mathgen.core import check, generator, pick_distractors
from mathgen.figure_kinds import measure

_TIERS = ("easy", "medium", "hard")
_PRESETS = {t: {"tier": t} for t in _TIERS}

# unit name -> (figure style, plural, objects it is sensibly used on)
_UNITS = {
    "paper clip": ("paper_clip", "paper clips", ("pencil", "crayon", "ribbon", "stick", "key")),
    "cube": ("cube", "cubes", ("pencil", "strip of card", "book", "toy car")),
    "block": ("block", "blocks", ("ribbon", "book", "shelf", "strip of card")),
    "footprint": ("footprint", "footprints", ("rug", "path", "bench", "mat")),
    "counter": ("counter", "counters", ("ribbon", "strip of card", "pencil", "book")),
}
_NAMES = ("Priya", "Sam", "Mia", "Leo", "Zoe", "Jack", "Amira", "Ben")

_POOLS = {
    "easy": (
        (
            "The {obj} is measured with {units}.",
            "{Units} are laid along the {obj}.",
            "A row of {units} lies along the {obj}.",
            "The {obj} is measured using {units}, one after the other.",
            "Look at the {obj} and the {units} beside it.",
            "{Units} are placed along the length of the {obj}.",
        ),
        (
            "How many {units} long is the {obj}?",
            "How long is the {obj}, in {units}?",
            "Count the {units}: how long is the {obj}?",
        ),
    ),
    "medium_half": (
        (
            "The {obj} is measured with {units}, and its end falls in the middle of the last {unit}.",
            "A row of {units} runs along the {obj}. The {obj} stops halfway along the last {unit}.",
            "The last {unit} is only half covered by the {obj}.",
            "The {obj} ends halfway along a {unit}.",
        ),
        (
            "How long is the {obj}, in {units}?",
            "Find the length of the {obj} in {units}.",
            "What is the length of the {obj} measured in {units}?",
        ),
    ),
    "medium_compare": (
        (
            "The {a} and the {b} are each measured with {units}.",
            "Two objects, the {a} and the {b}, are measured with the same {units}.",
            "{Units} are used to measure a {a} and a {b}.",
            "A {a} and a {b} are measured with {units} laid end to end.",
        ),
        (
            "How many more {units} long is the {a} than the {b}?",
            "The {a} is longer than the {b}. By how many {units}?",
            "How many {units} longer is the {a} than the {b}?",
        ),
    ),
    "hard": (
        (
            "{name} measures the {obj} with {units} and says it is {n} {units} long.",
            "{name} lays {units} along the {obj}. {name} says the {obj} is {n} {units} long.",
            "To find the length of the {obj}, {name} uses {units}. {name} says the answer is {n} {units}.",
            "{name} claims the {obj} is {n} {units} long, measured with {units}.",
        ),
        (
            "Is the measurement fair?",
            "Which statement about {name}'s measurement is true?",
            "Look at how the {units} are laid. Which statement is true?",
        ),
    ),
}

_OPTION_TEXT = {
    "fair": "Yes, the {units} are the same size and are laid end to end",
    "gaps": "No, there are gaps between the {units}, so the {obj} is longer than {n} {units}",
    "overlaps": "No, the {units} overlap, so the {obj} is shorter than {n} {units}",
    "unequal": "No, the {units} are not all the same size",
}
_OPTION_ORDER = ("fair", "gaps", "overlaps", "unequal")


def _pick(rng, pool):
    return pool[rng.randrange(len(pool))]


def _units_row(label: str, length, units: list[tuple]) -> dict:
    return {"label": label, "length": length, "units": [{"start": s, "length": w} for s, w in units]}


def _even(n: int) -> list[tuple]:
    return [(i, 1) for i in range(n)]


def _half_text(x: Fraction) -> str:
    whole, part = divmod(x, 1)
    check(part in (0, Fraction(1, 2)), "a measurement that is not a whole or a half")
    return str(int(whole)) if part == 0 else f"{int(whole)} and a half"


def _strip(unit: str, rows: list[dict], alt: str) -> dict:
    return measure.unit_strip(unit=unit, unit_style=_UNITS[unit][0], rows=rows, alt=alt)


def _fill(rng, pools, **slots) -> str:
    """One sentence from each pool, with ``{Units}`` the capitalised plural for a sentence start."""
    slots["Units"] = slots["units"][0].upper() + slots["units"][1:]
    return " ".join(_pick(rng, p).format(**slots) for p in pools)


def _slots(rng):
    unit = _pick(rng, tuple(_UNITS))
    _style, plural, objs = _UNITS[unit]
    return unit, plural, objs


def _build_easy(rng):
    unit, plural, objs = _slots(rng)
    obj = _pick(rng, objs)
    n = 3 + rng.randrange(8)
    fig = _strip(unit, [_units_row(obj, n, _even(n))], f"a {obj} with {plural} laid along it")
    text = _fill(rng, _POOLS["easy"], obj=obj, units=plural, unit=unit)
    answer = str(n)
    cands = [str(n + 1), str(n - 1), str(n + 2), str(n - 2)]
    note = f"Count the {plural} from one end of the {obj} to the other: {n}."
    return fig, text, answer, cands, note


def _build_medium(rng):
    unit, plural, objs = _slots(rng)
    if rng.randrange(2):
        obj = _pick(rng, objs)
        n = 2 + rng.randrange(7)  # whole units before the half
        fig = _strip(
            unit,
            [_units_row(obj, Fraction(2 * n + 1, 2), _even(n + 1))],
            f"a {obj} with {plural} laid along it",
        )
        text = _fill(rng, _POOLS["medium_half"], obj=obj, units=plural, unit=unit)
        answer = f"{n} and a half"
        cands = [str(n + 1), str(n), f"{n + 1} and a half", f"{n - 1} and a half"]
        note = f"{n} whole {plural} and half of the next one: {answer}."
        return fig, text, answer, cands, note
    a_obj, b_obj = rng.sample(objs, 2)
    b = 2 + rng.randrange(6)
    a = b + 1 + rng.randrange(4)
    fig = _strip(
        unit,
        [_units_row(a_obj, a, _even(a)), _units_row(b_obj, b, _even(b))],
        f"a {a_obj} and a {b_obj}, each with {plural} laid along it",
    )
    text = _fill(rng, _POOLS["medium_compare"], a=a_obj, b=b_obj, units=plural, unit=unit)
    answer = str(a - b)
    cands = [str(a + b), str(b), str(a), str(a - b + 1), str(a - b - 1)]
    note = f"The {a_obj} is {a} {plural} long and the {b_obj} is {b}: {a} - {b} = {a - b}."
    return fig, text, answer, cands, note


def _hard_row(rng, kind: str, label: str) -> dict:
    n = 4 + rng.randrange(5)
    if kind == "fair":
        return _units_row(label, n, _even(n))
    if kind == "gaps":
        joints = rng.sample(range(1, n), 1 + rng.randrange(2))
        units, x = [], Fraction(0)
        for i in range(n):
            if i in joints:
                x += Fraction(1, 2)
            units.append((x, 1))
            x += 1
        return _units_row(label, x, units)
    if kind == "overlaps":
        joints = rng.sample(range(1, n), 1 + rng.randrange(2))
        units, x = [], Fraction(0)
        for i in range(n):
            if i in joints:
                x -= Fraction(1, 2)
            units.append((x, 1))
            x += 1
        return _units_row(label, x, units)
    widths = [1] * n
    for i in rng.sample(range(n), 1 + rng.randrange(2)):
        widths[i] = Fraction(3, 2) if rng.randrange(2) else Fraction(1, 2)
    units, x = [], Fraction(0)
    for w in widths:
        units.append((x, w))
        x += w
    return _units_row(label, x, units)


def _build_hard(rng):
    unit, plural, objs = _slots(rng)
    obj = _pick(rng, objs)
    kind = _pick(rng, _OPTION_ORDER)
    row = _hard_row(rng, kind, obj)
    n = len(row["units"])
    fig = _strip(unit, [row], f"a {obj} with {plural} laid along it")
    check(measure.classify_unit_row(fig["rows"][0]) == kind, "the row is not the kind of measurement intended")
    slots = dict(name=_pick(rng, _NAMES), obj=obj, units=plural, unit=unit, n=n)
    text = _fill(rng, _POOLS["hard"], **slots)
    options = {k: _OPTION_TEXT[k].format(**slots) for k in _OPTION_ORDER}
    answer = options[kind]
    cands = [options[k] for k in _OPTION_ORDER if k != kind]
    notes = {
        "fair": f"The {plural} are the same size and touch end to end with no gaps or overlaps, so {n} {plural} is fair.",
        "gaps": f"The gaps use up some of the {obj}'s length, so it is longer than {n} {plural}.",
        "overlaps": f"The overlaps use the same length twice, so the {obj} is shorter than {n} {plural}.",
        "unequal": f"The {plural} are different sizes, so counting them does not give a length.",
    }
    return fig, text, answer, cands, notes[kind]


_BUILDERS = {"easy": _build_easy, "medium": _build_medium, "hard": _build_hard}


@generator("informal_units_measure", level=2, presets=_PRESETS)
def informal_units_measure(rng, tier="easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    fig, text, answer, cands, note = _BUILDERS[tier](rng)
    cands = [c for c in cands if not c.startswith("-") and not c.startswith("0")]
    rng.shuffle(cands)  # so the key is not always the middle of the options
    distractors = pick_distractors(answer, cands, n=3)
    check(len(distractors) == 3, "the question needs three distinct distractors")
    figures.assert_no_leak(fig, [answer, *distractors])
    return dict(
        text=text,
        answer=answer,
        answer_text=answer,
        answer_latex=f"plain:{answer}",
        distractors=distractors,
        working=[note],
        figure=fig,
    )


def _read_units(question) -> str:
    fig = question.figure
    rows = fig["rows"]
    if len(rows) == 2:
        (a, b) = (measure.read_unit_row(r) for r in rows)
        return str(int(a - b))
    row = rows[0]
    kind = measure.classify_unit_row(row)
    if question.text.endswith(("fair?", "true?")):
        plural = fig["unit"] + "s"
        slots = dict(units=plural, obj=row["label"], n=len(row["units"]))
        return _OPTION_TEXT[kind].format(**slots)
    return _half_text(measure.read_unit_row(row))


figures.register_reader("informal_units_measure", _read_units)
