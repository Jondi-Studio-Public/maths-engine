"""Tests for the multi-part exam source `exd_x_sc7_counterexample_search` (senior-use-examples-and-counter-examples).

The oracle re-derives every key from the rendered stem and part prompts alone, with plain integer arithmetic, and
never imports the generator module.
"""

from __future__ import annotations

import math
import re

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_sc7_counterexample_search"
SKILL_ID = "senior-use-examples-and-counter-examples"
HARD = ("counterexample_identify", {"variant": "find_counterexample"})
SEEDS = range(300)

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _is_prime(n: int) -> bool:
    return n > 1 and all(n % d for d in range(2, math.isqrt(n) + 1))


def oracle(q) -> dict[str, str]:
    """(a) and (b) re-derived from the stem and the part prompts."""
    stem = q.stem
    if "prime" in stem:
        m = re.search(r"n\^2 \+ (?:(\d+))?n \+ (\d+) is", stem)
        b, c = int(m.group(1) or 1), int(m.group(2))
        n = next(k for k in range(1, 5000) if not _is_prime(k * k + b * k + c))
        return {"a": str(n), "b": str(n * n + b * n + c)}
    if "n^2 is divisible by" in stem:
        m = int(re.search(r"n\^2 is divisible by (\d+)", stem).group(1))
        n = next(k for k in range(1, 5000) if k * k % m == 0 and k % m)
        return {"a": str(n), "b": str(n * n // m)}
    nums = re.search(r"divisible by (?:both )?(\d+) and (?:by )?(\d+)", stem)
    x, y = int(nums.group(1)), int(nums.group(2))
    prod = int(re.search(r"divisible by (\d+)\.", stem).group(1))
    assert prod == x * y
    limit = int(re.search(r"from 1 to (\d+)", q.parts[1].prompt).group(1))

    def counter(k):
        return k % x == 0 and k % y == 0 and k % prod != 0

    return {
        "a": str(next(k for k in range(1, 5000) if counter(k))),
        "b": str(sum(counter(k) for k in range(1, limit + 1))),
    }


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_source_is_wired_into_the_skill():
    last = _SKILL.sources[-1]
    assert last.topic == TOPIC and last.difficulty == "exam" and list(last.stages) == ["exam"]
    assert last.params == {"tier": "exam"}


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b"]
        assert [p.follow_through is not None for p in q.parts] == [False, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_oracle_agrees_300_seeds(questions):
    bad = [(s, q.text) for s, q in zip(SEEDS, questions) if oracle(q) != _canonical(q)]
    assert bad == []


def test_all_three_families_occur(questions):
    kinds = {
        "prime" if "prime" in q.stem else "square" if "n^2 is divisible" in q.stem else "product" for q in questions
    }
    assert kinds == {"prime", "square", "product"}


def test_key_accepted_and_distractors_rejected_100_seeds(questions):
    for q in questions[:100]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) == 3
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_distractors_are_tagged_where_an_error_exists(questions):
    tagged = [d for q in questions[:100] for p in q.parts for d in p.distractors if d in p.option_notes]
    assert len(tagged) >= 300


def _b_from_a(q, a_value: int) -> int:
    """What a correct (b) is when (a) was answered `a_value` (re-derived from the prompts)."""
    stem = q.stem
    if "prime" in stem:
        m = re.search(r"n\^2 \+ (?:(\d+))?n \+ (\d+) is", stem)
        b, c = int(m.group(1) or 1), int(m.group(2))
        return a_value * a_value + b * a_value + c
    if "n^2 is divisible by" in stem:
        m = int(re.search(r"n\^2 is divisible by (\d+)", stem).group(1))
        return a_value * a_value // m
    prod = int(re.search(r"divisible by (\d+)\.", stem).group(1))
    limit = int(re.search(r"from 1 to (\d+)", q.parts[1].prompt).group(1))
    return limit // a_value - limit // prod


def test_follow_through_from_a_wrong_part_a(questions):
    checked = 0
    for q in questions:
        canon = int(q.parts[0].answer_text)
        wrong = 2 * canon  # a multiple of the true answer: whole quotients, and positive counts, follow from it
        b_wrong = _b_from_a(q, wrong)
        if b_wrong in (int(q.parts[1].answer_text), wrong) or b_wrong <= 0:  # a repeated value earns no follow-through
            continue
        result = grade(_SKILL, q, {"a": str(wrong), "b": str(b_wrong)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == result.parts["b"].max_marks, (q.text, wrong, b_wrong)
        assert result.parts["b"].follow_through is True
        assert result.marks < result.max_marks and not result.correct
        checked += 1
    assert checked >= 150


def test_all_wrong_earns_zero(questions):
    for q in questions[:60]:
        result = grade(_SKILL, q, {"a": "999999", "b": "-7"})
        assert result.marks == 0 and not result.correct
