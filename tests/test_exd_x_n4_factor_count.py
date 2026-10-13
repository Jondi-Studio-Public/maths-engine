"""Exam tier of `list-factors`: counting factors from a prime factorisation."""

from __future__ import annotations

import math
import re

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_n4_factor_count"
SKILL = next(s for s in load_skills()[1] if s.id == "list-factors")


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def oracle(q):
    """Re-derive all three answers from the factorisation printed in the prompt, by listing the divisors."""
    factorisation = re.search(r"N = ([^.]+)\.", q.stem).group(1)
    value = 1
    for term in factorisation.split(" × "):
        base, _, exponent = term.partition("^")
        value *= int(base) ** int(exponent or 1)
    divisors = _divisors(value)
    if "perfect squares" in q.text:
        kept = [d for d in divisors if math.isqrt(d) ** 2 != d]
    else:
        prime = int(re.search(r"not multiples of (\d+)", q.text).group(1))
        kept = [d for d in divisors if d % prime]
    return {"a": str(value), "b": str(len(divisors)), "c": str(len(kept))}


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_source_wired():
    exam = [s for s in SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC]
    assert exam[0].stages == ["exam"]


def test_prompt_shapes_are_new():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make("factor_list", seed=s, hi=60)) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = _exam(seed)
        assert oracle(q) == _canonical(q), (seed, q.text)


def test_shape():
    q = _exam(0)
    assert [p.label for p in q.parts] == ["a", "b", "c"]
    assert [p.marks for p in q.parts] == [1, 2, 2]
    assert [bool(p.follow_through) for p in q.parts] == [False, False, True]
    assert all(p.answer_kind == "number" and len(p.distractors) >= 2 for p in q.parts)


def test_answer_space():
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 15


def test_key_accepted_and_distractors_rejected():
    shown = 0
    for seed in range(120):
        q = _exam(seed)
        canon = _canonical(q)
        full = grade(SKILL, q, canon)
        assert full.correct and full.marks == full.max_marks == 5, (seed, q.text)
        for part in q.parts:
            for d in part.distractors:
                shown += 1
                result = grade(SKILL, q, {**canon, part.label: d})
                assert result.parts[part.label].marks == 0, (seed, part.label, d)
                assert part.option_notes.get(d)
    assert shown >= 600


def test_follow_through_from_wrong_factor_count():
    carried = 0
    for seed in range(100):
        q = _exam(seed)
        canon = _canonical(q)
        unwanted = int(canon["b"]) - int(canon["c"])
        wrong_b = int(q.parts[1].distractors[0])
        wrong_c = wrong_b - unwanted
        result = grade(SKILL, q, {"a": canon["a"], "b": str(wrong_b), "c": str(wrong_c)})
        assert result.parts["b"].marks == 0
        if wrong_c > 0 and wrong_c != wrong_b:  # a reply must be sensible and not repeat the earlier one
            carried += 1
            assert result.parts["c"].marks == q.parts[2].marks, (seed, q.text, wrong_b, wrong_c)
            assert result.parts["c"].follow_through
    assert carried >= 50


def test_all_wrong_earns_nothing():
    result = grade(SKILL, _exam(3), {"a": "7", "b": "7", "c": "7"})
    assert result.marks == 0 and not result.correct


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
