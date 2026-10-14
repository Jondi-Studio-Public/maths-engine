"""Independent oracle for `f10-linear-conjecture-testing`.

Reads only `question.text`, the public option list and the public `mcq` grader. It reads the table and the
conjecture out of the wording, checks every row or step itself, decides which of the four options is true, and
compares that with the question's key. It does not import the generator module.
"""

from __future__ import annotations

import re

import pytest

from mathgen import make
from mathgen.grading.typed import grade_mcq, mcq_options

TOPIC = "f10_linear_conjecture"
TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = 250


def _ints(s):
    return [int(v) for v in s.split(", ")]


def _rule(text):
    m = re.search(r"the rule y = (-?\d*)x(?: ([+-]) (\d+))? gives", text)
    coef = {"": 1, "-": -1}.get(m.group(1)) or int(m.group(1))
    const = int(m.group(3)) * (1 if m.group(2) == "+" else -1) if m.group(3) else 0
    return coef, const


def _broken_places(text):
    """('row'|'step', [labels of broken places]) re-derived from the wording alone."""
    xs = _ints(re.search(r"x: ([-\d, ]+); (?:table A )?y: ", text).group(1))
    if "gives every y-value" in text:
        ys = _ints(re.search(r"; y: ([-\d, ]+)\. Which", text).group(1))
        m, c = _rule(text)
        return "row", [f"x = {x}" for x, y in zip(xs, ys) if y != m * x + c]
    if "table B is" in text:
        k = int(re.search(r"table B is (\d+) times", text).group(1))
        ya = _ints(re.search(r"table A y: ([-\d, ]+); table B", text).group(1))
        yb = _ints(re.search(r"table B y: ([-\d, ]+)\. Which", text).group(1))
        bad = [
            f"from x = {xs[i]} to x = {xs[i + 1]}"
            for i in range(len(xs) - 1)
            if yb[i + 1] - yb[i] != k * (ya[i + 1] - ya[i])
        ]
        return "step", bad
    ys = _ints(re.search(r"; y: ([-\d, ]+)\. Which", text).group(1))
    if "each time x goes up by 1" in text:
        up = re.search(r"y goes (up|down) by (\d+) each time", text)
        rate = int(up.group(2)) * (1 if up.group(1) == "up" else -1)
    else:
        rate = int(re.search(r"y changes by (-?\d+) for every 1", text).group(1))
    bad = [
        f"from x = {xs[i]} to x = {xs[i + 1]}"
        for i in range(len(xs) - 1)
        if (ys[i + 1] - ys[i]) != rate * (xs[i + 1] - xs[i])
    ]
    return "step", bad


def _true(option, noun, broken):
    """Whether one option is a true claim about the table."""
    m = re.match(rf"The {noun} (.+?) breaks the (?:rule|pattern), so the table does not support it\.", option)
    if m:
        return m.group(1) in broken
    assert re.match(rf"No {noun} breaks the (?:rule|pattern): every {noun} agrees, so the table supports", option), (
        option
    )
    return not broken


@pytest.mark.parametrize("tier", TIERS)
def test_exactly_one_option_is_true_and_it_is_the_key(tier):
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        noun, broken = _broken_places(q.text)
        assert len(broken) <= 1, q.text
        options = mcq_options(q)
        truths = [_true(o, noun, broken) for o in options]
        assert sum(truths) == 1, (q.text, options, broken)
        assert options[truths.index(True)] == q.answer_text, (q.text, q.answer_text)


@pytest.mark.parametrize("tier", TIERS)
def test_the_grader_marks_the_true_option_correct_and_the_others_wrong(tier):
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=tier)
        noun, broken = _broken_places(q.text)
        for index, option in enumerate(mcq_options(q)):
            assert grade_mcq(q, index).correct == _true(option, noun, broken), (q.text, option)


def test_hard_and_exam_tables_have_uneven_x_steps_and_wide_steps_appear_among_the_distractors():
    for tier in ("hard", "exam"):
        saw_wide_distractor = False
        for seed in range(80):
            q = make(TOPIC, seed=seed, difficulty=tier)
            xs = _ints(re.search(r"x: ([-\d, ]+); (?:table A )?y: ", q.text).group(1))
            gaps = [b - a for a, b in zip(xs, xs[1:])]
            assert len(set(gaps)) >= 2 and max(gaps) >= 2, q.text
            for d in q.distractors:
                m = re.search(r"from x = (-?\d+) to x = (-?\d+)", d)
                if m and int(m.group(2)) - int(m.group(1)) >= 2:
                    saw_wide_distractor = True
        assert saw_wide_distractor
