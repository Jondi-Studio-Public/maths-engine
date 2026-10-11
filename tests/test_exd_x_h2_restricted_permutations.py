"""Exam tier of the restricted-permutations skill: letters of a word with repeats under a positional rule.

The oracle reads the word and the rule from the rendered prompt, lists every distinct arrangement with the
standard library, and tests the rule with regular expressions. It shares no code with the generator.
"""

from __future__ import annotations

import re
from itertools import permutations

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_h2_restricted_permutations"


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def test_no_text_defects():
    for seed in range(100):
        q = _exam(seed)
        assert find_text_defects(q, include_working=True) == [], (seed, q.text)


def test_prompt_variety_and_new_shapes_over_hard_tier():
    shapes = {prompt_shape(_exam(s)) for s in range(60)}
    assert len(shapes) >= 3, shapes
    hard = {prompt_shape(make("p11c_restricted_permutations", seed=s, difficulty="hard")) for s in range(60)}
    assert len(shapes - hard) >= 3, (shapes, hard)


def _oracle(text: str) -> int:
    word = re.search(r"\b([A-Z]{5,})\b", text).group(1)
    arrangements = {"".join(p) for p in permutations(word)}
    if "no two vowels" in text:
        return sum(1 for w in arrangements if not re.search(r"[AEIOU]{2}", w))
    if "unbroken block" in text:
        return sum(1 for w in arrangements if len(re.findall(r"[AEIOU]+", w)) == 1)
    assert "begins and ends with a consonant" in text, text
    return sum(1 for w in arrangements if w[0] not in "AEIOU" and w[-1] not in "AEIOU")


def test_key_matches_independent_oracle_over_300_seeds():
    mismatches = []
    for seed in range(300):
        q = _exam(seed)
        expected = _oracle(q.text)
        if q.answer != expected or q.answer_text != str(expected):
            mismatches.append((seed, q.text, expected, q.answer_text))
    assert mismatches == [], mismatches[:3]


def test_grader_accepts_key_and_rejects_every_distractor():
    grade = GRADERS["number"]
    bad = []
    for seed in range(150):
        q = _exam(seed)
        assert q.distractors, seed
        if not grade(q, q.answer_text).correct:
            bad.append(("key rejected", seed))
        for d in q.distractors:
            if grade(q, d).correct:
                bad.append(("distractor accepted", seed, d))
    assert bad == [], bad[:3]


def test_deterministic_and_all_three_rules_occur():
    for seed in (0, 1, 7, 99):
        assert _exam(seed).to_json() == _exam(seed).to_json()
    rules = set()
    for seed in range(60):
        t = _exam(seed).text
        rules.add(("no two vowels" in t, "unbroken block" in t))
    assert rules == {(True, False), (False, True), (False, False)}
