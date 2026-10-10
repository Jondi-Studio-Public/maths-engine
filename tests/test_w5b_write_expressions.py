"""W5b: expression_from_words `structure` kind (hard tier of write-expressions-from-words)."""

import re

import sympy

from mathgen.grading import GRADERS
from mathgen.skills import load_skills
from mathgen import make

X = sympy.Symbol("x")
T1 = re.compile(r"^Write an expression for: subtract (\d+) times a number from (\d+)$")
T2 = re.compile(r"^Write an expression for: (\d+) times the sum of a number and (\d+)$")
T3 = re.compile(r"^Write an expression for: (\d+) times the difference between a number and (\d+)$")


def _q(seed):
    return make("expression_from_words", seed=seed, kind="structure")


def _expected(text):
    if m := T1.match(text):
        a, b = map(int, m.groups())
        return b - a * X
    if m := T2.match(text):
        a, b = map(int, m.groups())
        return a * (X + b)
    m = T3.match(text)
    assert m, text
    a, b = map(int, m.groups())
    return a * (X - b)


def _val(s):
    return sympy.expand(sympy.sympify(s, locals={"x": X}))


def test_default_path_unchanged():
    expected = {
        ("easy", 0): "Write an expression for: 4 times a number, plus 3",
        ("easy", 1): "Write an expression for: 3 less than 5 times a number",
        ("easy", 2): "Write an expression for: 2 times a number, minus 2",
        ("medium", 0): "Write an expression for: 6 less than 6 times a number",
        ("medium", 1): "Write an expression for: 6 less than 4 times a number",
        ("medium", 2): "Write an expression for: a number multiplied by 6, then decreased by 12",
        ("hard", 0): "Write an expression for: a number multiplied by 11, then increased by 18",
        ("hard", 1): "Write an expression for: 5 less than 11 times a number",
        ("hard", 2): "Write an expression for: 3 times a number, minus 6",
    }
    for (d, seed), text in expected.items():
        assert make("expression_from_words", seed=seed, difficulty=d).text == text


def test_answer_rederived_from_prompt():
    templates = set()
    for seed in range(300):
        q = _q(seed)
        assert sympy.expand(_val(q.answer_text) - sympy.expand(_expected(q.text))) == 0, (seed, q.text)
        templates.add(next(i for i, r in enumerate((T1, T2, T3)) if r.match(q.text)))
    assert templates == {0, 1, 2}


def test_distractors_differ_in_value():
    for seed in range(300):
        q = _q(seed)
        ans = _val(q.answer_text)
        assert len(q.distractors) == 3, seed
        vals = [_val(d) for d in q.distractors]
        for v in vals:
            assert sympy.expand(v - ans) != 0, (seed, q.text, q.distractors)
        assert len(set(vals)) == 3, seed
        assert q.answer_text not in q.distractors


def test_real_grader_accepts_expanded_form_and_rejects_distractors():
    _, skills = load_skills()
    skill = next(s for s in skills if s.id == "write-expressions-from-words")
    hard = next(s for s in skill.sources if s.difficulty == "hard")
    assert hard.params == {"kind": "structure"}
    grader = GRADERS[skill.answer]
    for seed in range(120):
        q = make(hard.topic, seed=seed, difficulty="hard", **hard.params)
        assert grader(q, q.answer_text).correct, (seed, q.answer_text)
        expanded = str(sympy.expand(_expected(q.text)))
        assert grader(q, expanded).correct, (seed, q.text, expanded)
        for d in q.distractors:
            assert not grader(q, d).correct, (seed, q.text, d)
        assert not grader(q, f"{q.answer_text}+1").correct
