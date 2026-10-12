"""Independent oracle for `senior-use-set-language-and-notation-for-events-including`.

Reads only `question.text` and the public `set` grader. It reads the sample space and each event out of the wording
(an explicit list, or a worded event it decodes with its own table), parses the set expression with its own small
parser, evaluates it, and compares the result with the question's key. It does not import the generator module.
"""

from __future__ import annotations

import re

import pytest

from mathgen import make
from mathgen.grading.typed import grade_set

TOPIC = "senior_event_set_notation"
TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = 250

_PRIMES = {2, 3, 5, 7, 11}


def _decode(words, outcome):
    """Whether `outcome` satisfies the worded event."""
    table = {
        "the number is even": lambda x: x % 2 == 0,
        "the number is odd": lambda x: x % 2 == 1,
        "the number is prime": lambda x: x in _PRIMES,
        "the number is a multiple of 3": lambda x: x % 3 == 0,
        "the number is a perfect square": lambda x: int(x**0.5) ** 2 == x,
        "the number is greater than 3": lambda x: x > 3,
        "the number is less than 4": lambda x: x < 4,
        "the number is at least 5": lambda x: x >= 5,
        "the number is at most 2": lambda x: x <= 2,
        "the number is a factor of 6": lambda x: 6 % x == 0,
        "at least one head is shown": lambda o: "H" in o,
        "both coins show the same face": lambda o: o[0] == o[1],
        "exactly one tail is shown": lambda o: sum(1 for c in o if c == "T") == 1,
        "the first coin shows a head": lambda o: o[0] == "H",
        "the second coin shows a tail": lambda o: o[1] == "T",
        "at least two heads are shown": lambda o: sum(1 for c in o if c == "H") >= 2,
        "the first coin shows a tail": lambda o: o[0] == "T",
        "all three coins show the same face": lambda o: o[0] == o[1] == o[2],
        "exactly one head is shown": lambda o: sum(1 for c in o if c == "H") == 1,
        "the last coin shows a head": lambda o: o[2] == "H",
        "no heads are shown": lambda o: "H" not in o,
        "the sum is 8": lambda o: o[0] + o[1] == 8,
        "the sum is 7": lambda o: o[0] + o[1] == 7,
        "the two dice show the same number": lambda o: o[0] == o[1],
        "the first die shows 6": lambda o: o[0] == 6,
        "the second die shows 2": lambda o: o[1] == 2,
        "the sum is at least 10": lambda o: o[0] + o[1] >= 10,
        "the first die is odd": lambda o: o[0] % 2 == 1,
        "the difference between the dice is 4": lambda o: abs(o[0] - o[1]) == 4,
        "the product of the two numbers is 12": lambda o: o[0] * o[1] == 12,
    }
    return table[words](outcome)


def _space(text):
    if "all 36 ordered pairs" in text:
        return [(a, b) for a in range(1, 7) for b in range(1, 7)]
    body = re.search(r"S = \{([^}]*)\}", text).group(1)
    items = [i.strip() for i in body.split(",")]
    return [int(i) if i.isdigit() else i for i in items]


def _events(text, space):
    events = {}
    for name, body in re.findall(r"\b([ABC]) = \{([^}]*)\}", text):
        events[name] = {int(i) for i in body.split(",")}
    for name, words in re.findall(r"Event ([ABC]) is: (.+?)\.", text):
        events[name] = {o for o in space if _decode(words, o)}
    return events


def _tokens(expr):
    return re.findall(r"[ABC]|'|∩|∪|\(|\)", expr)


def _evaluate(expr, events, space):
    toks = _tokens(expr)
    full = set(space)
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def atom():
        nonlocal pos
        tok = toks[pos]
        pos += 1
        if tok == "(":
            value = union()
            assert toks[pos] == ")"
            pos += 1
        else:
            value = set(events[tok])
        while peek() == "'":
            pos += 1
            value = full - value
        return value

    def inter():
        nonlocal pos
        value = atom()
        while peek() == "∩":
            pos += 1
            value = value & atom()
        return value

    def union():
        nonlocal pos
        value = inter()
        while peek() == "∪":
            pos += 1
            value = value | atom_or_inter()
        return value

    def atom_or_inter():
        return inter()

    value = union()
    assert pos == len(toks), expr
    return value


def _expression(text):
    return re.search(r"List the outcomes in (.+?)\. Write your answer", text).group(1)


@pytest.mark.parametrize("tier", TIERS)
def test_the_key_is_the_set_the_notation_names(tier):
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        space = _space(q.text)
        events = _events(q.text, space)
        expected = _evaluate(_expression(q.text), events, space)
        key = {tuple(i) if isinstance(i, list) else i for i in q.answer}
        assert key == expected, (q.text, sorted(key, key=str), sorted(expected, key=str))
        assert len(q.answer) == len(expected)


@pytest.mark.parametrize("tier", TIERS)
def test_the_grader_accepts_the_oracle_set_and_rejects_a_changed_one(tier):
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=tier)
        space = _space(q.text)
        expected = _evaluate(_expression(q.text), _events(q.text, space), space)
        items = [str(o) if not isinstance(o, tuple) else f"({o[0]}, {o[1]})" for o in sorted(expected, key=str)]
        assert grade_set(q, "{" + ", ".join(reversed(items)) + "}").correct, q.text
        outsider = next(o for o in space if o not in expected) if len(expected) < len(space) else None
        if outsider is not None:
            extra = str(outsider) if not isinstance(outsider, tuple) else f"({outsider[0]}, {outsider[1]})"
            assert not grade_set(q, ", ".join(items + [extra])).correct, q.text


@pytest.mark.parametrize("tier", TIERS)
def test_every_distractor_is_a_different_set_from_the_oracle_answer(tier):
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        space = _space(q.text)
        expected = _evaluate(_expression(q.text), _events(q.text, space), space)
        for text in q.distractors:
            body = text.strip("{}")
            if not body.strip():
                wrong = set()
            elif "(" in body:
                wrong = {tuple(int(x) for x in m.split(",")) for m in re.findall(r"\(([^)]*)\)", body)}
            else:
                wrong = {int(i) if i.strip().isdigit() else i.strip() for i in body.split(",")}
            assert wrong != expected, (q.text, text)
            assert wrong <= set(space), (q.text, text)
