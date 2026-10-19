"""Generates `fr2b_integer_situations` questions for `f10-integer-situations-number-line` (AC9M6N01). `answer: number`.

Scope: temperature, height above or below sea level, building levels and a bank balance, each shown on a drawn
`number_line` figure with the whole numbers printed. The learner reads a marked value, finds the change between two
marks, or finds the value after a move. Ordering integers is the neighbouring skill `f10-integer-numberline-
coordinates`; nothing here orders numbers.

Ladder (reasoning escalation, not size):
  easy:   read the value at one marked point, with a negative reading about half of the time.
  medium: the rise from A to B, where A is below zero and B is above it (the change is a sum of sizes) or both are
          below zero (the change is a difference of sizes).
  hard:   either two moves from A, one up and one down, where the line is crossed on the way; or the value halfway
          between A and B, where the two have the same parity so the midpoint is a whole number.

Each key is signed integer arithmetic on the values drawn, and the figure's marks are read back by
`_read_marks` (a plausibility reader that sees the figure and the amounts in the wording, never the key).
"""

from __future__ import annotations

import re

from .. import figures
from ..core import attempts, check, generator
from ._fr2b_kit import build, int_dist, tagged

_TIERS = ("easy", "medium", "hard")

#: name -> (tick step, plausible lowest value, plausible highest value, range half-width) for each setting.
_SETTINGS = {
    "temperature": dict(step=1, low=-10, high=10, half=10),
    "height": dict(step=1, low=-12, high=12, half=12),
    "level": dict(step=1, low=-5, high=10, half=10),
    "balance": dict(step=5, low=-50, high=50, half=50),
}

_SCENE = {
    "temperature": "The number line shows temperatures in degrees Celsius on a mountain.",
    "height": "The number line shows heights in metres, where 0 is sea level and negative heights are below it.",
    "level": "The number line shows the levels of a building, where 0 is the ground floor and negative levels are basements.",
    "balance": "The number line shows the balance of a bank account in dollars, where a negative balance means money is owed.",
}

_EASY_ASK = {
    "temperature": "A marks the temperature at dawn. What is the temperature at dawn, in degrees Celsius?",
    "height": "A marks the position of a diver. What is the diver's height in metres?",
    "level": "A marks the level of the car park. What is the level number of the car park?",
    "balance": "A marks the balance on Monday. What is the balance in dollars?",
}

_MEDIUM_ASK = {
    "temperature": "A marks the overnight low and B marks the afternoon high. By how many degrees did the temperature rise from A to B?",
    "height": "A marks the sea floor and B marks the top of a cliff. How many metres higher is B than A?",
    "level": "A marks the car park level and B marks the cafe level. How many levels does the lift travel from A to B?",
    "balance": "A marks the balance before a deposit and B marks the balance after it. How many dollars were deposited?",
}

# (rise phrase, fall phrase, unit word, subject) for the two-move question.
_MOVES = {
    "temperature": ("rises by", "falls by", "degrees", "The temperature at A is the overnight low."),
    "height": ("swims up by", "dives down by", "metres", "A diver starts at the height marked A."),
    "level": ("goes up by", "goes down by", "levels", "A lift starts at the level marked A."),
    "balance": ("goes up by", "goes down by", "dollars", "The balance at A is the balance at the start of the week."),
}

_MOVE_ASK = {
    "temperature": "What is the temperature after both changes, in degrees Celsius?",
    "height": "What is the diver's height after both moves, in metres?",
    "level": "What level is the lift at after both moves?",
    "balance": "What is the balance after both changes, in dollars?",
}

_MID_ASK = {
    "temperature": "A and B mark the overnight low and the afternoon high. The temperature at noon was halfway between them. What was the temperature at noon, in degrees Celsius?",
    "height": "A and B mark the sea floor and the top of a cliff. A ledge is halfway between them. What is the height of the ledge, in metres?",
    "level": "A and B mark the car park and the roof garden. The cafe is halfway between them. What is the level number of the cafe?",
    "balance": "A and B mark the balance at the start and at the end of a year. The balance in the middle of the year was halfway between them. What was it, in dollars?",
}

_MOVE_RE = re.compile(r"(rises|swims up|goes up|falls|dives down|goes down) by (\d+)")
_UP_WORDS = {"rises", "swims up", "goes up"}


def _line(setting, marks, alt):
    cfg = _SETTINGS[setting]
    half, step = cfg["half"], cfg["step"]
    return figures.number_line(range_=(-half, half), alt=alt, tick=step, marks=marks)


def _value(rng, setting, *, even=False, low=None, high=None):
    cfg = _SETTINGS[setting]
    lo, hi, step = cfg["low"] if low is None else low, cfg["high"] if high is None else high, cfg["step"]
    n = rng.randint(lo // step, hi // step)
    if even and n % 2:
        n += 1 if n < hi // step else -1
    return n * step


def _result(text, answer, distractors, working, fig):
    return dict(
        text=text,
        latex="",
        figure=fig,
        answer_kind="number",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=working,
        distractors=distractors,
    )


def _easy(rng):
    setting = rng.choice(list(_SETTINGS))
    step = _SETTINGS[setting]["step"]
    for _ in attempts(200, "a non-zero reading"):
        a = _value(rng, setting)
        if a != 0 and (a < 0) == (rng.random() < 0.55):
            break
    fig = _line(setting, [{"at": a, "label": "A"}], "A number line with point A marked on it.")
    cands = [
        int_dist(-a, "fr2b-reads-the-wrong-side-of-zero"),
        int_dist(a + _SETTINGS[setting]["half"], "fr2b-counts-steps-from-the-left-end"),
        int_dist(a + step, "fr2b-reads-the-wrong-tick"),
        int_dist(a - step, "fr2b-reads-the-wrong-tick"),
        int_dist(a + 2 * step, "fr2b-reads-the-wrong-tick"),
    ]
    text = f"{_SCENE[setting]} {_EASY_ASK[setting]}"
    working = [f"A is {abs(a)} {'to the left of' if a < 0 else 'to the right of'} zero, so its value is {a}."]
    return _result(text, a, tagged(str(a), cands), working, fig)


def _medium(rng):
    setting = rng.choice(list(_SETTINGS))
    cfg = _SETTINGS[setting]
    for _ in attempts(500, "two marks"):
        cross = rng.random() < 0.6
        if cross:
            a = _value(rng, setting, low=cfg["low"], high=-cfg["step"])
            b = _value(rng, setting, low=cfg["step"], high=cfg["high"])
        else:
            a = _value(rng, setting, low=cfg["low"], high=-2 * cfg["step"])
            b = _value(rng, setting, low=a + cfg["step"], high=-cfg["step"])
        if a < b and (cfg["low"] <= a) and b - a >= 2 * cfg["step"] and abs(a) != abs(b):
            break
    change = b - a
    check(change == (abs(a) + abs(b) if a < 0 < b else abs(a) - abs(b)), "change disagrees with sizes")
    fig = _line(
        setting, [{"at": a, "label": "A"}, {"at": b, "label": "B"}], "A number line with points A and B marked on it."
    )
    cands = [
        int_dist(abs(a) - abs(b), "fr2b-subtracts-sizes-across-zero")
        if a < 0 < b
        else int_dist(abs(a) + abs(b), "fr2b-adds-sizes-on-one-side-of-zero"),
        int_dist(a + b, "fr2b-adds-the-readings"),
        int_dist(-change, "fr2b-change-direction-reversed"),
        int_dist(b - abs(a) if a < 0 < b else abs(b) - abs(a), "fr2b-subtracts-sizes-across-zero"),
        int_dist(change + _SETTINGS[setting]["step"], "fr2b-reads-the-wrong-tick"),
    ]
    text = f"{_SCENE[setting]} {_MEDIUM_ASK[setting]}"
    if a < 0 < b:
        working = [f"A is {abs(a)} below zero and B is {b} above it, so the distance is {abs(a)} + {b} = {change}."]
    else:
        working = [f"Both are below zero. B is closer to zero, so the distance is {abs(a)} - {abs(b)} = {change}."]
    return _result(text, change, tagged(str(change), cands), working, fig)


def _hard_moves(rng):
    setting = rng.choice(list(_SETTINGS))
    cfg = _SETTINGS[setting]
    step = cfg["step"]
    rise, fall, unit, subject = _MOVES[setting]
    who = {"height": "The diver", "level": "The lift"}.get(setting, "It")
    for _ in attempts(2000, "two moves that cross zero and stay on the line"):
        a = rng.randint(cfg["low"] // step, cfg["high"] // step) * step
        x, y = rng.randint(2, 14) * step, rng.randint(2, 14) * step
        up_first = rng.random() < 0.5
        mid = a + x if up_first else a - y
        end = a + x - y
        stops = (a, mid, end)
        if all(cfg["low"] <= v <= cfg["high"] for v in stops) and min(stops) < 0 < max(stops) and a != 0 and end != 0:
            break
    first = f"{rise} {x}" if up_first else f"{fall} {y}"
    second = f"{fall} {y}" if up_first else f"{rise} {x}"
    fig = _line(setting, [{"at": a, "label": "A"}], "A number line with point A marked on it.")
    text = f"{_SCENE[setting]} {subject} {who} {first} {unit}, then {second} {unit}. {_MOVE_ASK[setting]}"
    cands = [
        int_dist(a - x - y, "fr2b-rise-and-fall-treated-the-same"),
        int_dist(a + x + y, "fr2b-rise-and-fall-treated-the-same"),
        int_dist(a + y - x, "fr2b-rise-and-fall-swapped"),
        int_dist(abs(a) + x - y, "fr2b-sign-of-the-start-ignored"),
        int_dist(-end, "fr2b-negative-sign-dropped-at-the-end"),
    ]
    working = [
        f"Start at {a}. After the first move: {mid}. After the second move: {end}.",
        f"{a} {'+' if up_first else '-'} {x if up_first else y} {'-' if up_first else '+'} {y if up_first else x} = {end}.",
    ]
    return _result(text, end, tagged(str(end), cands), working, fig)


def _hard_midpoint(rng):
    setting = rng.choice(list(_SETTINGS))
    cfg = _SETTINGS[setting]
    step = cfg["step"]
    for _ in attempts(1000, "two marks with a whole-number midpoint"):
        a = _value(rng, setting, low=cfg["low"], high=-2 * step)
        b = _value(rng, setting, low=2 * step, high=cfg["high"])
        if ((b - a) // step) % 2 == 0 and abs(a) != abs(b):
            break
    mid = (a + b) // 2
    check(2 * mid == a + b and a < mid < b, "midpoint is wrong")
    fig = _line(
        setting, [{"at": a, "label": "A"}, {"at": b, "label": "B"}], "A number line with points A and B marked on it."
    )
    cands = [
        int_dist((abs(a) + abs(b)) // 2, "fr2b-sign-of-the-start-ignored"),
        int_dist((b - a) // 2, "fr2b-half-the-distance-given-as-the-position"),
        int_dist(a + b, "fr2b-no-halving"),
        int_dist(-mid, "fr2b-reads-the-wrong-side-of-zero"),
        int_dist(mid + step, "fr2b-reads-the-wrong-tick"),
    ]
    text = f"{_SCENE[setting]} {_MID_ASK[setting]}"
    working = [f"The halfway value is ({a} + {b}) / 2 = {a + b} / 2 = {mid}."]
    return _result(text, mid, tagged(str(mid), cands), working, fig)


def _hard(rng):
    return _hard_moves(rng) if rng.random() < 0.55 else _hard_midpoint(rng)


def _marks(question):
    return {m["label"]: m["at"] for m in question.figure["marks"]}


def _read_answer(question) -> str:
    marks = _marks(question)
    text = question.text
    if "B" not in marks:
        moves = _MOVE_RE.findall(text)
        if moves:
            return str(marks["A"] + sum(int(n) if verb in _UP_WORDS else -int(n) for verb, n in moves))
        return str(marks["A"])
    if "halfway" in text:
        return str((marks["A"] + marks["B"]) // 2)
    return str(marks["B"] - marks["A"])


figures.register_reader("fr2b_integer_situations", _read_answer)

_BUILDERS = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator("fr2b_integer_situations", level=6, presets={tier: dict(tier=tier) for tier in _TIERS})
def fr2b_integer_situations(rng, tier="easy"):
    """Integers in temperature, height, level and balance settings on a number line; tiers in the module docstring."""
    check(tier in _BUILDERS, f"unknown tier {tier!r}")
    return build(_BUILDERS[tier], rng, tier)
