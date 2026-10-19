"""Generates `add_sub_within_20_strategies` questions: solve an addition or subtraction problem within 20, where
the numbers are chosen so that a mental strategy beats counting.

Scope: solve an addition or subtraction problem within 20. The answer is the number; the strategy is a hint in
the easier tiers and the learner's own choice in the hardest.

## Boundary with `add-within-20` / `subtract-within-20`

Those skills ask for a single bare sum or difference and let any method through. Here the numbers are drawn so
that the sum or difference crosses ten, or sits next to a double, or sits next to a ten, so the quick
strategies (make a ten, a near double, add 10 and adjust, go back through ten, think addition) are the
efficient route, and the wrong answers are the specific slips each strategy invites (a forgotten
leftover, an adjustment taken the wrong way).

## Difficulty progression (genuine reasoning escalation, not size)

- easy: the strategy is named in the prompt and the problem is an addition across ten (`8 + 6`, make a ten) or
  a double (`7 + 7`). DECISION: carry out one named strategy.
- medium: the strategy is named, but it is a different one for each problem and some are subtractions:
  a near double (`7 + 8`), adding 9 or 8 by adding 10 and adjusting, taking away through ten (`15 - 8`), and
  finding the difference by thinking addition (`14 - 6`). DECISION: carry out the strategy that is named,
  including its adjustment step, which easy never needs.
- hard: no strategy is named and there are three numbers (`6 + 9 + 4`, `8 + 7 - 6`, `15 - 8 + 5`) with every
  stage staying within 20. DECISION: spot which two numbers to combine first (a pair that makes ten, or the
  term that cancels part of another) rather than working left to right, and read the signs correctly.

## Independent check for every draw

Each question has its key worked out two ways: by the plain arithmetic of the displayed numbers, and by the
named strategy's own steps (make a ten: `10 + (b - (10 - a))`; add 9: `a + 10 - 1`; and so on) with
`check_equivalent`. Every partial result of a hard problem is checked to stay between 0 and 20.
`check_distinct` runs on every distractor candidate unconditionally. Answers are whole numbers, so the
typed-answer grader requires an exact match and any value different from the key is rejected.

The oracle parses the displayed numbers and signs from `question.text` and evaluates them, and never imports
this module.
"""

from __future__ import annotations

from ..core import check, check_distinct, check_equivalent, generator, phrase
from ._option_rank import pick_ranked_distractors

TOPIC = "add_sub_within_20_strategies"
_TIERS = ("easy", "medium", "hard")

_EASY_SHAPES = ("make_ten", "make_ten", "double")
_MEDIUM_SHAPES = ("near_double", "add_near_ten", "back_through_ten", "think_addition")
_HARD_SHAPES = ("ten_pair", "add_then_subtract", "subtract_then_add")

_MAKE_TEN = (
    "Make a ten to work out {x} + {y}.",
    "Work out {x} + {y} by making a ten first.",
    "Use the make-a-ten strategy to find {x} + {y}.",
    "What is {x} + {y}? Try making a ten.",
)
_DOUBLE = (
    "Work out {a} + {a}. Use your doubles facts.",
    "What is {a} + {a}? Use a doubles fact.",
    "Use a double to find {a} + {a}.",
)
_NEAR_DOUBLE = (
    "Use a double to work out {x} + {y}.",
    "Work out {x} + {y} using a near double.",
    "What is {x} + {y}? Think of a double that is close.",
)
_ADD_NEAR_TEN = (
    "To add {k}, add 10 and then take {j} off. Work out {x} + {y}.",
    "Add {k} by adding 10 and then taking {j} off. What is {x} + {y}?",
    "Use the add-10-then-adjust strategy to find {x} + {y}.",
)
_BACK_THROUGH_TEN = (
    "Go back through ten to work out {a} - {b}: first subtract to get to 10.",
    "Work out {a} - {b} by taking away to 10 first.",
    "Subtract through ten to find {a} - {b}.",
)
_THINK_ADDITION = (
    "Think {b} + ? = {a} to work out {a} - {b}.",
    "What is {a} - {b}? Think about what to add to {b} to make {a}.",
    "Use a related addition fact to work out {a} - {b}.",
)
_HARD = (
    "Work out {expr}. Look for a quick way.",
    "What is {expr}? Choose a strategy that suits the numbers.",
    "Find the value of {expr} in your head.",
)


def _finalise(text: str, key: int, candidates: list[int], rng, working: str) -> dict:
    check(0 <= key <= 20, f"{TOPIC}: the key {key} is outside 0 to 20")
    chosen = pick_ranked_distractors(rng, key, candidates, lo=0, hi=20)
    check_distinct(key, *chosen, msg=f"{TOPIC}: a distractor equals the key")
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=key,
        answer_text=str(key),
        answer_latex=str(key),
        distractors=[str(c) for c in chosen],
        working=[working],
    )


def _shuffled_pair(rng, a: int, b: int) -> tuple[int, int]:
    return (a, b) if rng.random() < 0.5 else (b, a)


# -- easy ----------------------------------------------------------------------------------------------------


def _make_ten(rng) -> dict:
    for _ in range(200):
        a = rng.randint(6, 9)
        b = rng.randint(3, 9)
        need = 10 - a
        if a + b >= 11 and b > need:
            break
    else:
        check(False, f"{TOPIC}: could not draw a make-ten pair")
    key = a + b
    rest = b - need
    check_equivalent(key, 10 + rest, f"{TOPIC} make_ten: the make-ten route disagrees with {a} + {b}")
    x, y = _shuffled_pair(rng, a, b)
    text = phrase(rng, _MAKE_TEN, x=x, y=y)
    # 10 (forgets the leftover), 10 + b (adds all of b after making ten), a + rest (adds the leftover to a),
    # and an off-by-one slip each way.
    working = f"Make ten first: {a} + {need} = 10, which leaves {b} - {need} = {rest}. Then 10 + {rest} = {key}."
    return _finalise(text, key, [10, 10 + b, a + rest, key - 1, key + 1], rng, working)


def _double(rng) -> dict:
    a = rng.randint(6, 9)
    key = 2 * a
    check_equivalent(key, a + a, f"{TOPIC} double: {a} doubled disagrees with {a} + {a}")
    text = phrase(rng, _DOUBLE, a=a)
    return _finalise(text, key, [key - 1, key + 1, key - 2, key + 2, a + 10], rng, f"{a} + {a} = {key}.")


# -- medium --------------------------------------------------------------------------------------------------


def _near_double(rng) -> dict:
    x = rng.randint(4, 9)
    y = x + 1
    key = x + y
    check_equivalent(key, 2 * x + 1, f"{TOPIC} near_double: {x} + {y} disagrees with its double plus one")
    first, second = _shuffled_pair(rng, x, y)
    text = phrase(rng, _NEAR_DOUBLE, x=first, y=second)
    # the lower double, the higher double, a double shifted by two, and adding ten to one number
    working = f"{x} + {x} = {2 * x}, and {y} is one more than {x}, so {x} + {y} = {2 * x} + 1 = {key}."
    return _finalise(text, key, [2 * x, 2 * y, key - 2, key + 2, x + 10, y + 10], rng, working)


def _add_near_ten(rng) -> dict:
    k = rng.choice((8, 9))
    j = 10 - k
    a = rng.randint(3, 9)
    key = a + k
    check_equivalent(key, a + 10 - j, f"{TOPIC} add_near_ten: {a} + 10 - {j} disagrees with {a} + {k}")
    x, y = (a, k) if rng.random() < 0.5 else (k, a)
    text = phrase(rng, _ADD_NEAR_TEN, k=k, j=j, x=x, y=y)
    # stops after adding ten, adjusts the wrong way, and the off-by-one slips
    working = f"{a} + 10 = {a + 10}, then take {j} off: {a + 10} - {j} = {key}."
    return _finalise(text, key, [a + 10, a + 10 + j, key - 1, key + 1, a + k + j], rng, working)


def _back_through_ten(rng) -> dict:
    for _ in range(200):
        a = rng.randint(12, 18)
        b = rng.randint(3, 9)
        to_ten = a - 10
        if b > to_ten and a - b >= 3:
            break
    else:
        check(False, f"{TOPIC}: could not draw a through-ten subtraction")
    key = a - b
    leftover = b - to_ten
    check_equivalent(key, 10 - leftover, f"{TOPIC} back_through_ten: the through-ten route disagrees with {a} - {b}")
    text = phrase(rng, _BACK_THROUGH_TEN, a=a, b=b)
    # stops at ten, reports the leftover, adds instead of subtracting, and the off-by-one slips
    working = f"{a} - {to_ten} = 10, then take away the other {leftover}: 10 - {leftover} = {key}."
    return _finalise(text, key, [10, leftover, a + b, key - 1, key + 1], rng, working)


def _think_addition(rng) -> dict:
    for _ in range(200):
        a = rng.randint(12, 18)
        b = rng.randint(4, 9)
        if a - b >= 3 and a - b != b:
            break
    else:
        check(False, f"{TOPIC}: could not draw a think-addition subtraction")
    key = a - b
    check_equivalent(b + key, a, f"{TOPIC} think_addition: {b} + {key} should make {a}")
    text = phrase(rng, _THINK_ADDITION, a=a, b=b)
    # repeats the number taken away, adds instead of subtracting, and the off-by-one slips
    working = f"{b} + {key} = {a}, so {a} - {b} = {key}."
    return _finalise(text, key, [b, a + b, key - 1, key + 1, key + 2], rng, working)


# -- hard ----------------------------------------------------------------------------------------------------


def _expr_text(terms: list[int], signs: list[str]) -> str:
    out = str(terms[0])
    for sign, t in zip(signs, terms[1:]):
        out += f" {sign} {t}"
    return out


def _evaluate(terms: list[int], signs: list[str]) -> list[int]:
    """The running total after each term, left to right."""
    total = terms[0]
    running = [total]
    for sign, t in zip(signs, terms[1:]):
        total = total + t if sign == "+" else total - t
        running.append(total)
    return running


def _hard_finalise(rng, terms, signs, key, candidates) -> dict:
    running = _evaluate(terms, signs)
    check(all(0 <= r <= 20 for r in running), f"{TOPIC} hard: a stage of {terms} {signs} leaves 0 to 20")
    check_equivalent(running[-1], key, f"{TOPIC} hard: the strategy route disagrees with left-to-right arithmetic")
    text = phrase(rng, _HARD, expr=_expr_text(terms, signs))
    steps = [f"{running[i]} {sign} {t} = {running[i + 1]}" for i, (sign, t) in enumerate(zip(signs, terms[1:]))]
    return _finalise(text, key, candidates, rng, "Work from left to right: " + ", then ".join(steps) + ".")


def _ten_pair(rng) -> dict:
    x = rng.randint(2, 8)
    pair = [x, 10 - x]
    z = rng.randint(3, 9)
    terms = pair + [z]
    rng.shuffle(terms)
    key = 10 + z
    check_equivalent(key, sum(terms), f"{TOPIC} ten_pair: 10 + {z} disagrees with the sum of {terms}")
    # leaves out the last term, leaves out the first term, stops at the pair, and off-by-one slips
    return _hard_finalise(
        rng,
        terms,
        ["+", "+"],
        key,
        [terms[0] + terms[1], terms[1] + terms[2], 10, key - 1, key + 1],
    )


def _add_then_subtract(rng) -> dict:
    for _ in range(200):
        a = rng.randint(5, 9)
        b = rng.randint(5, 9)
        c = rng.randint(3, 9)
        if c not in (a, b) and a + b - c >= 4 and a + b >= 11:
            break
    else:
        check(False, f"{TOPIC}: could not draw an add-then-subtract problem")
    key = a + b - c
    check_equivalent(key, a + (b - c), f"{TOPIC} add_then_subtract: regrouped route disagrees with {a} + {b} - {c}")
    # leaves out the subtraction, adds the last number instead, reads it as a - b + c, and off-by-one slips
    return _hard_finalise(rng, [a, b, c], ["+", "-"], key, [a + b, a + b + c, abs(a - b) + c, key - 1, key + 1])


def _subtract_then_add(rng) -> dict:
    for _ in range(200):
        a = rng.randint(12, 19)
        b = rng.randint(4, 9)
        c = rng.randint(3, 9)
        if c != b and a - b >= 3 and a - b + c <= 19:
            break
    else:
        check(False, f"{TOPIC}: could not draw a subtract-then-add problem")
    key = a - b + c
    check_equivalent(key, (a + c) - b, f"{TOPIC} subtract_then_add: regrouped route disagrees with {a} - {b} + {c}")
    # subtracts both numbers, leaves out the addition, subtracts after adding the wrong way, and off-by-one slips
    return _hard_finalise(rng, [a, b, c], ["-", "+"], key, [a - b - c, a - b, a + b - c, key - 1, key + 1])


_BUILDERS = {
    "make_ten": _make_ten,
    "double": _double,
    "near_double": _near_double,
    "add_near_ten": _add_near_ten,
    "back_through_ten": _back_through_ten,
    "think_addition": _think_addition,
    "ten_pair": _ten_pair,
    "add_then_subtract": _add_then_subtract,
    "subtract_then_add": _subtract_then_add,
}
_SHAPES = {"easy": _EASY_SHAPES, "medium": _MEDIUM_SHAPES, "hard": _HARD_SHAPES}


@generator(
    TOPIC,
    level=1,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def add_sub_within_20_strategies(rng, tier: str = "easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return _BUILDERS[rng.choice(_SHAPES[tier])](rng)
