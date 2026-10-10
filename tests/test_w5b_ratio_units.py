"""W5b: ratio `mixed_units` kind (hard tier of ratio-rate-proportion)."""

import re
from fractions import Fraction

import pytest

from mathgen.grading import GRADERS
from mathgen.skills import load_skills
from mathgen import make

FACTORS = {("m", "cm"): 100, ("kg", "g"): 1000, ("km", "m"): 1000, ("hours", "minutes"): 60}
PROMPT_RE = re.compile(r"^Write the ratio (\d+) (\w+) to (\d+) (\w+) in simplest form\.$")


def _q(seed):
    return make("ratio", seed=seed, kind="mixed_units")


def _val(text):
    left, right = text.split(":")
    return Fraction(int(left), int(right))


def test_default_paths_unchanged():
    expected = {
        ("simplify", 0): "Simplify the ratio 28:20",
        ("simplify", 1): "Simplify the ratio 108:9",
        ("simplify", 2): "Simplify the ratio 35:49",
        ("share", 0): "Share 40 in the ratio 2:3. Find the first share.",
        ("share", 1): "Share 140 in the ratio 6:1. Find the first share.",
        ("unit_rate", 0): "4 items cost 20 in total. Find the cost per item (unit rate).",
        ("unit_rate", 1): "2 items cost 26 in total. Find the cost per item (unit rate).",
    }
    for (kind, seed), text in expected.items():
        assert make("ratio", seed=seed, kind=kind).text == text


def test_answer_rederived_from_prompt():
    for seed in range(300):
        q = _q(seed)
        m = PROMPT_RE.match(q.text)
        assert m, q.text
        a, big, b, small = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
        f = FACTORS[(big, small)]
        assert "convert" not in q.text.lower()
        assert Fraction(a * f, b) == _val(q.answer_text)
        assert q.answer == {"p": _val(q.answer_text).numerator, "q": _val(q.answer_text).denominator}


def test_distractors_distinct_in_value():
    for seed in range(300):
        q = _q(seed)
        ans = _val(q.answer_text)
        vals = [_val(d) for d in q.distractors]
        assert len(q.distractors) == 3, (seed, q.distractors)
        assert ans not in vals
        assert len(set(vals)) == len(vals)


def test_grading_accepts_key_rejects_distractors():
    _, skills = load_skills()
    skill = next(s for s in skills if s.id == "ratio-rate-proportion")
    hard = next(s for s in skill.sources if s.difficulty == "hard")
    assert hard.params == {"kind": "mixed_units"}
    grader = GRADERS[skill.answer]
    for seed in range(150):
        q = make(hard.topic, seed=seed, difficulty="hard", **hard.params)
        assert grader(q, q.answer_text).correct, (seed, q.answer_text)
        for d in q.distractors:
            assert not grader(q, d).correct, (seed, d)
        # a different value is also rejected
        assert not grader(q, f"{q.answer['p'] + 1}:{q.answer['q']}").correct


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_deterministic(seed):
    assert _q(seed).text == _q(seed).text
