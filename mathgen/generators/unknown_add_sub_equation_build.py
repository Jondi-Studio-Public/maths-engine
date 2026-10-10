"""Generates `unknown_add_sub_equation` questions: find the missing value in a numerical addition or
subtraction equation, using operation properties.

Scope: one blank, whole numbers, additions and subtractions only. No letters standing for numbers and no
multiplicative unknowns. The answer is the missing number.

## Boundary

`f10-addition-facts-related-subtraction` is a fact-family skill: it gives a complete addition fact and asks for
a related subtraction. Here the equation is given with a value missing and the learner has to find it, by
using the inverse operation, by seeing that both sides must stay equal, or by shifting a number and its partner
together. The written equation is the only thing in the question: there is no story and the blank is a square.

## Difficulty progression (genuine reasoning escalation, not size)

- easy: one operation and the blank stands alone against a total (`27 + □ = 61`, `□ + 18 = 61`, `90 - □ = 34`,
  `□ - 18 = 34`), numbers to 100. DECISION: choose the inverse operation. Two of the four shapes (`90 - □ = 34`
  and `□ - 18 = 34`) look alike and need opposite operations.
- medium: an expression on both sides (`27 + 15 = □ + 20`, `□ + 15 = 30 + 12`, `63 - 28 = □ - 20`,
  `63 - 28 = 80 - □`), numbers to about 100. DECISION: the equals sign says "has the same value as", so the
  learner must find the value of the side that is complete before working back to the blank; adding up
  every number shown is the typical wrong answer.
- hard: three-digit numbers where a property does the work: a partner number shifted to a round number
  (`398 + 245 = 400 + □`, `597 - 248 = 600 - □`), or a blank sitting between two operations
  (`135 + □ - 48 = 400`, `□ - 120 + 45 = 300`). DECISION: use how a change to one number must be matched
  by a change to the blank, or undo two operations in the right order.

## Independent check for every draw

The learner receives a rendered equation, so the check reads that string. `_solve_equation_text` splits the
equation at the equals sign, reads each side as a list of signed terms, and finds the value of the blank from
how each side grows with the blank (its value with the blank as 0 and as 1), which is a different route from
the formula each shape was built with. The key must equal that solution and every distractor candidate must
differ from it (`check_equivalent`, `check_distinct`).

The oracle parses the equation from `question.text` and solves it by trying every whole number for the blank
until both sides agree, and never imports this module.
"""

from __future__ import annotations

import re

from ..core import check, check_distinct, check_equivalent, generator, phrase
from ._option_rank import pick_ranked_distractors

TOPIC = "unknown_add_sub_equation"
_TIERS = ("easy", "medium", "hard")
BLANK = "□"

_PROMPT = (
    "Find the missing number: {eq}",
    "What number makes this equation true? {eq}",
    "Work out the number that goes in the {blank}: {eq}",
    "Find the value of the {blank} that makes this true: {eq}",
)

_EASY_SHAPES = ("add_right", "add_left", "sub_result", "sub_start")
_MEDIUM_SHAPES = ("balance_add_right", "balance_add_left", "balance_sub_left", "balance_sub_right")
_HARD_SHAPES = ("compensate_add", "compensate_sub", "between_ops")

# A side is a list of (sign, value) with value an int or None (the blank). The first sign is always "+".
Side = list[tuple[str, int | None]]


def _render_side(side: Side) -> str:
    out = ""
    for i, (sign, value) in enumerate(side):
        cell = BLANK if value is None else str(value)
        out += cell if i == 0 else f" {sign} {cell}"
    return out


def _render(left: Side, right: Side) -> str:
    return f"{_render_side(left)} = {_render_side(right)}"


_TERM_RE = re.compile(r"([+-]) (\d+|" + BLANK + ")")


def _side_coefficients(side_text: str) -> tuple[int, int]:
    """(value with the blank as 0, change in value per unit of blank) for a rendered side."""
    parts = side_text.split(" ", 1)
    first = parts[0]
    rest = " " + parts[1] if len(parts) > 1 else ""
    const, slope = 0, 0
    terms = [("+", first)] + _TERM_RE.findall(rest)
    for sign, cell in terms:
        factor = 1 if sign == "+" else -1
        if cell == BLANK:
            slope += factor
        else:
            const += factor * int(cell)
    return const, slope


def _solve_equation_text(equation: str) -> int:
    """Solve a RENDERED equation for its blank. Reads the string the learner sees."""
    left, right = equation.split(" = ")
    check(equation.count(BLANK) == 1, f"{TOPIC}: {equation!r} must have exactly one blank")
    l0, ls = _side_coefficients(left)
    r0, rs = _side_coefficients(right)
    check(ls != rs, f"{TOPIC}: the blank cancels out of {equation!r}")
    numerator, denominator = r0 - l0, ls - rs
    check(numerator % denominator == 0, f"{TOPIC}: {equation!r} has no whole-number solution")
    return numerator // denominator


# -- shapes --------------------------------------------------------------------------------------------------
# Each returns (left, right, key, candidates). The candidates are the errors the shape invites; the shared tail
# adds the off-by-one and off-by-ten slips.


def _add_right(rng):
    a = rng.randint(6, 60)
    key = rng.randint(6, 99 - a)
    c = a + key
    return [("+", a), ("+", None)], [("+", c)], key, [a + c, c, a]


def _add_left(rng):
    b = rng.randint(6, 60)
    key = rng.randint(6, 99 - b)
    c = b + key
    return [("+", None), ("+", b)], [("+", c)], key, [b + c, c, b]


def _sub_result(rng):
    a = rng.randint(6, 60)
    key = rng.randint(6, 99 - a)
    c = a + key
    return [("+", c), ("-", None)], [("+", a)], key, [c + a, a, c]


def _sub_start(rng):
    b, c = rng.randint(6, 45), rng.randint(6, 45)
    key = c + b
    return [("+", None), ("-", b)], [("+", c)], key, [c - b, b, c]


def _balance_add_right(rng):
    a, b = rng.randint(11, 60), rng.randint(11, 60)
    c = rng.randint(11, min(a + b - 6, 90))
    key = a + b - c
    return [("+", a), ("+", b)], [("+", None), ("+", c)], key, [a + b, a + b + c, c]


def _balance_add_left(rng):
    b = rng.randint(11, 60)
    c, d = rng.randint(11, 60), rng.randint(11, 60)
    key = c + d - b
    return [("+", None), ("+", b)], [("+", c), ("+", d)], key, [c + d, c + d + b, d]


def _balance_sub_left(rng):
    a = rng.randint(40, 95)
    b = rng.randint(11, a - 11)
    c = rng.randint(11, 60)
    key = a - b + c
    return [("+", a), ("-", b)], [("+", None), ("-", c)], key, [a - b, a - b - c, b + c]


def _balance_sub_right(rng):
    a = rng.randint(40, 95)
    b = rng.randint(11, a - 11)
    diff = a - b
    c = rng.randint(diff + 6, 99)
    key = c - diff
    return [("+", a), ("-", b)], [("+", c), ("-", None)], key, [diff, c + diff, c - a - b]


def _compensate_add(rng):
    round_a = rng.choice(range(200, 1000, 100))
    k = rng.randint(1, 9)
    a = round_a - k
    b = rng.randint(120, 800)
    key = b - k
    return [("+", a), ("+", b)], [("+", round_a), ("+", None)], key, [b, b + k, a + b, round_a + b]


def _compensate_sub(rng):
    round_a = rng.choice(range(300, 1000, 100))
    k = rng.randint(1, 9)
    a = round_a - k
    b = rng.randint(120, a - 100)
    key = b + k
    return [("+", a), ("-", b)], [("+", round_a), ("-", None)], key, [b, b - k, a - b, round_a - b]


def _between_ops(rng):
    if rng.random() < 0.5:
        a, b = rng.randint(100, 400), rng.randint(21, 99)
        key = rng.randint(100, 600)
        c = a + key - b
        return (
            [("+", a), ("+", None), ("-", b)],
            [("+", c)],
            key,
            [c - a - b, c + a - b, c - a, c + b],
        )
    a, b = rng.randint(21, 99), rng.randint(21, 99)
    key = rng.randint(200, 800)
    c = key - a + b
    return (
        [("+", None), ("-", a), ("+", b)],
        [("+", c)],
        key,
        [c - a + b, c + a + b, c + a, c - b],
    )


_BUILDERS = {
    "add_right": _add_right,
    "add_left": _add_left,
    "sub_result": _sub_result,
    "sub_start": _sub_start,
    "balance_add_right": _balance_add_right,
    "balance_add_left": _balance_add_left,
    "balance_sub_left": _balance_sub_left,
    "balance_sub_right": _balance_sub_right,
    "compensate_add": _compensate_add,
    "compensate_sub": _compensate_sub,
    "between_ops": _between_ops,
}
_CAP = {"easy": 199, "medium": 199, "hard": 1199}
_SHAPES = {"easy": _EASY_SHAPES, "medium": _MEDIUM_SHAPES, "hard": _HARD_SHAPES}


def _numbers(left: Side, right: Side) -> set[int]:
    return {v for _s, v in left + right if v is not None}


def _build(rng, tier: str = "easy") -> dict:
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    for _ in range(200):
        left, right, key, raw = _BUILDERS[rng.choice(_SHAPES[tier])](rng)
        if key >= 2 and key not in _numbers(left, right):  # an answer that repeats a shown number blurs the options
            break
    else:
        check(False, f"{TOPIC} {tier}: could not draw an equation whose answer is not already shown")
    equation = _render(left, right)

    # the rendered equation, solved a different way from the formula that built it
    check_equivalent(_solve_equation_text(equation), key, f"{TOPIC} {tier}: {equation!r} does not solve to {key}")

    slips = [key - 1, key + 1, key - 10, key + 10]
    chosen = pick_ranked_distractors(rng, key, [*raw, *slips], lo=1, hi=_CAP[tier], fillers=(2, 3, 5))
    check_distinct(key, *chosen, msg=f"{TOPIC} {tier}: a distractor equals the key")

    text = phrase(rng, _PROMPT, eq=equation, blank=BLANK)
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=key,
        answer_text=str(key),
        answer_latex=str(key),
        distractors=[str(c) for c in chosen],
        working=[f"{equation.replace(BLANK, str(key))} is true, so the missing number is {key}."],
    )


@generator(
    TOPIC,
    level=3,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def unknown_add_sub_equation(rng, tier: str = "easy"):
    return _build(rng, tier=tier)
