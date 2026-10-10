"""W5b: formula_plugin `unknown="input"` (hard tier of use-formulas): rearrange before substituting."""

import re
from fractions import Fraction

import pytest

from mathgen.grading import GRADERS
from mathgen.skills import load_skills
from mathgen import make


def _q(seed):
    return make("formula_plugin", seed=seed, unknown="input")


def _nums(text):
    return [int(x) for x in re.findall(r"\d+", text.split(" Use ")[0])]


def _val(s):
    return Fraction(s)


def _rederive(text):
    n = _nums(text)
    if "has area" in text and "rectangle" in text:
        A, l = n
        return Fraction(A, l)
    if "has perimeter" in text:
        P, l = n
        return Fraction(P, 2) - l
    if "triangle" in text:
        A, b = n
        return Fraction(2 * A, b)
    if "km/h" in text:
        s, t = n
        return Fraction(s * t)
    if "Simple interest" in text:
        I, P, T = n
        return Fraction(100 * I, P * T)
    raise AssertionError(text)


def test_default_paths_unchanged():
    expected = {
        0: "A rectangle has length 3 and width 28. Use A = l × w to find its area.",
        1: "A triangle has base 20 and height 22. Use A = ½bh to find its area.",
        3: "A rectangle has length 15 and width 20. Use P = 2(l + w) to find its perimeter.",
        4: "A car travels 27 km in 9 hours. Use speed = distance / time to find its speed (km/h).",
        5: "Find the simple interest on £1000 at 3% per year for 3 years. Use I = PRT/100.",
    }
    for seed, text in expected.items():
        assert make("formula_plugin", seed=seed, difficulty="hard").text == text
        assert make("formula_plugin", seed=seed, difficulty="hard", unknown="result").text == text


def test_answer_rederived_from_prompt():
    kinds = set()
    for seed in range(400):
        q = _q(seed)
        a = _rederive(q.text)
        assert a.denominator == 1, q.text
        assert a == _val(q.answer_text), (seed, q.text, q.answer_text)
        kinds.add(q.latex)
    assert len(kinds) == 5


def test_distractors_distinct_in_value():
    for seed in range(400):
        q = _q(seed)
        ans = _val(q.answer_text)
        vals = [_val(d) for d in q.distractors]
        assert len(q.distractors) == 3, (seed, q.distractors)
        assert ans not in vals
        assert len(set(vals)) == len(vals)


def test_grading_accepts_key_rejects_distractors():
    _, skills = load_skills()
    skill = next(s for s in skills if s.id == "use-formulas")
    hard = next(s for s in skill.sources if s.difficulty == "hard")
    assert hard.params == {"unknown": "input"}
    grader = GRADERS[skill.answer]
    for seed in range(150):
        q = make(hard.topic, seed=seed, difficulty="hard", **hard.params)
        assert grader(q, q.answer_text).correct, (seed, q.answer_text)
        for d in q.distractors:
            assert not grader(q, d).correct, (seed, d)
        assert not grader(q, str(int(q.answer) + 1)).correct


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_deterministic(seed):
    assert _q(seed).text == _q(seed).text
