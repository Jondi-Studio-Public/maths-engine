"""Exam tier of `f10-number-represent-order-to-1000`: described numbers are worked out, then ordered."""

from __future__ import annotations

import re

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_pr7_order_to_1000_regrouped"
SKILL_ID = "f10-number-represent-order-to-1000"
HARD = ("order_to_1000", {"count": 5, "digit_lengths": [2, 3], "min_same_hundreds_pairs": 1})
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _value(item: str) -> int:
    item = item.strip()
    if m := re.fullmatch(r"(\d+) more than (\d+)", item):
        return int(m[2]) + int(m[1])
    if m := re.fullmatch(r"(\d+) less than (\d+)", item):
        return int(m[2]) - int(m[1])
    if "hundred" in item:
        nums = [int(x) for x in re.findall(r"\d+", item)]
        return nums[0] * 100 + nums[1] * 10 + (nums[2] if len(nums) > 2 else 0)
    return int(item)


def oracle(q):
    body = re.search(r"ways: (.*?)\. (?:Write|Put)", q.text).group(1)
    return sorted(_value(x) for x in body.split(";"))


def test_key_matches_independent_derivation_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _q(seed)
        want = oracle(q)
        if len(set(want)) != 5 or not all(100 <= v <= 999 for v in want) or want != q.answer:
            bad.append((seed, q.text, q.answer))
    assert bad == []


def test_wording_is_grammatical():
    for seed in range(300):
        text = _q(seed).text
        assert " 1 hundreds" not in text and " 1 ones" not in text and "0 ones" not in text


def test_grader_accepts_key_and_rejects_every_distractor():
    for seed in range(150):
        q = _q(seed)
        assert grade(_SKILL, q, q.answer_text).correct
        assert len(q.distractors) >= 2
        for d in q.distractors:
            assert not grade(_SKILL, q, d).correct, (seed, d)


def test_distractors_tagged_with_values():
    for seed in range(60):
        q = _q(seed)
        assert set(q.distractor_values) == set(q.distractors)
