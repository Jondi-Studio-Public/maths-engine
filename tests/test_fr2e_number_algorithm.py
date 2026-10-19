"""Tests `f10-number-algorithm-choose-complete` (topic `fr2e_number_algorithm`, exam chain `exd_x_fr2e_number_algorithm`).

The oracle reads the steps, start values and finishing values back out of the question TEXT and runs the algorithms
itself: it simulates every written list-making instruction, tries all 24 orders of a number machine, and replays the
`n` sequence for each candidate missing rule.
"""

from __future__ import annotations

import itertools
import re
from fractions import Fraction

import pytest
from sympy import Symbol
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.fr2e_number_algorithm_build  # noqa: F401  (registers the topics)
from tests.fr2e_support import (
    SEEDS,
    TIERS,
    assert_chain_shape,
    assert_clean_and_deterministic,
    assert_exam_grading,
    assert_follow_through,
    assert_registered,
    assert_single_grading,
    assert_tiers_differ,
    exam,
    single,
)

SKILL = "f10-number-algorithm-choose-complete"
TOPIC = "fr2e_number_algorithm"
EXAM = "exd_x_fr2e_number_algorithm"
N = Symbol("n")
_STEP = re.compile(r"([A-D])\. (Add|Subtract|Multiply by|Divide by) (\d+)")
_OPS = {
    "Add": lambda v, k: v + k,
    "Subtract": lambda v, k: v - k,
    "Multiply by": lambda v, k: v * k,
    "Divide by": lambda v, k: Fraction(v) / k,
}


def _steps(text: str) -> dict:
    found = {letter: (op, int(k)) for letter, op, k in _STEP.findall(text)}
    assert len(found) == 4, text
    return found


def _run(order, steps, start):
    value = Fraction(start)
    for letter in order:
        op, k = steps[letter]
        value = _OPS[op](value, k)
    return value


def _letters(answer: str):
    return [part.strip() for part in answer.split(",")]


def _orders_that_work(steps, start, finish):
    return [p for p in itertools.permutations(steps) if _run(p, steps, start) == finish]


# -- easy: which written instructions make the list ----------------------------------------------------------------

_INSTRUCTION = re.compile(
    r"Write (\d+)\. Then repeat: add (\d+) to the last number written; if the new number is (more than (\d+)|(\d+) or more), stop; otherwise write it\."
)


def _written(option: str):
    first, step, _, above, at_least = _INSTRUCTION.fullmatch(option).groups()
    out, value = [int(first)], int(first)
    while True:
        value += int(step)
        if (above and value > int(above)) or (at_least and value >= int(at_least)):
            return out
        out.append(value)
        assert len(out) < 200


def _wanted(text: str):
    if m := re.search(r"start at (\d+), go up by (\d+) each time and never go above (\d+)", text):
        s, d, top = (int(v) for v in m.groups())
        return list(range(s, top + 1, d))
    if m := re.search(r"start at (\d+), go up by (\d+) each time and stay below (\d+)", text):
        s, d, top = (int(v) for v in m.groups())
        return list(range(s, top, d))
    if m := re.search(r"positive multiples of (\d+) up to and including (\d+)", text):
        d, top = (int(v) for v in m.groups())
        return list(range(d, top + 1, d))
    if m := re.search(r"positive multiples of (\d+) that are less than (\d+)", text):
        d, top = (int(v) for v in m.groups())
        return list(range(d, top, d))
    raise AssertionError(text)


def test_registered_sources():
    assert_registered(SKILL, TOPIC, EXAM)


def test_easy_only_the_key_writes_the_list():
    for seed in SEEDS:
        q = single(TOPIC, "easy", seed)
        wanted = _wanted(q.text)
        assert _written(q.answer_text) == wanted, (seed, q.text, q.answer_text)
        for wrong in q.distractors:
            assert _written(wrong) != wanted, (seed, q.text, wrong)


def test_medium_exactly_one_order_works():
    for seed in SEEDS:
        q = single(TOPIC, "medium", seed)
        steps = _steps(q.text)
        start, finish = (
            int(v) for v in re.search(r"Starting with (\d+), the machine must finish with (\d+)", q.text).groups()
        )
        assert _orders_that_work(steps, start, finish) == [tuple(_letters(q.answer_text))], (seed, q.text)
        for wrong in q.distractors:
            assert _run(_letters(wrong), steps, start) != finish, (seed, wrong)


def _rule(text: str):
    body = text.removeprefix("replace n by ").replace("×", "*")
    return parse_expr(
        body, local_dict={"n": N}, transformations=standard_transformations + (implicit_multiplication_application,)
    )


def _replay(text: str, rule) -> list[int]:
    start, q = (
        int(v)
        for v in re.search(
            r"starts with n = (\d+) and repeats these steps: if n is (?:even|a multiple of (\d+))", text
        ).groups(default="2")
    )
    seq, n = [start], start
    for _ in range(40):
        if n == 1:
            break
        n = n // q if n % q == 0 else int(rule.subs(N, n))
        seq.append(n)
    return seq


def test_hard_the_key_rule_replays_the_listed_numbers():
    for seed in SEEDS:
        q = single(TOPIC, "hard", seed)
        listed = [int(v) for v in re.search(r"in order, are ([\d, ]+)\.", q.text).group(1).split(",")]
        assert _replay(q.text, _rule(q.answer_text)) == listed, (seed, q.text, q.answer_text)
        for wrong in q.distractors:
            assert _replay(q.text, _rule(wrong)) != listed, (seed, q.text, wrong)


@pytest.mark.parametrize("tier", TIERS)
def test_tiers_differ_in_shape(tier):
    assert_tiers_differ(TOPIC)
    questions = [single(TOPIC, tier, seed) for seed in SEEDS]
    marker = {
        "easy": "Which instructions write exactly that list?",
        "medium": "Which order of the steps works?",
        "hard": "Which rule is the missing step",
    }
    assert all(marker[tier] in q.text for q in questions)


def test_clean_and_deterministic():
    assert_clean_and_deterministic(TOPIC, TIERS)
    assert_clean_and_deterministic(EXAM, ("exam",))


def test_single_grading():
    assert_single_grading(SKILL, TOPIC)


def test_exam_matches_oracle():
    for seed in range(80):
        q = exam(EXAM, seed)
        steps = _steps(q.text)
        start, finish = (int(v) for v in re.search(r"Starting with (\d+).*?finishes with (\d+)", q.text).groups())
        works = _orders_that_work(steps, start, finish)
        assert len(works) == 1, (seed, q.text)
        order = works[0]
        a, b, c, d = q.parts
        assert _letters(a.answer_text) == list(order)
        again = int(re.search(r"started again with (\d+)", q.text).group(1))
        out = _run(order, steps, again)
        assert int(b.answer_text) == out
        assert int(c.answer_text) == _run(order, steps, out)
        target = int(re.search(r"finishes with (\d+)\. Find the input", q.text).group(1))
        assert _run(order, steps, int(d.answer_text)) == target, (seed, q.text)


def test_exam_chain_shape():
    assert_chain_shape(EXAM, ["ordered_list", "number", "number", "number"], [2, 1, 1, 2], [False, False, True, False])


def test_exam_grading():
    assert_exam_grading(SKILL, EXAM)


def test_second_run_follows_through_from_a_wrong_first_output():
    def second(q, canon, wrong_output):
        steps = _steps(q.text)
        order = _letters(canon["a"])
        value = _run(order, steps, int(wrong_output))
        return str(int(value)) if value.denominator == 1 else None

    assert_follow_through(SKILL, EXAM, "c", "b", second)
