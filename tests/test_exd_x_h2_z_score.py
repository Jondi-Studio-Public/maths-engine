"""Exam tier of the z-score skill: recover the mean or standard deviation, or move a score between two scales.

The oracle reads the numbers out of the rendered prompt and solves with `fractions`. It shares no code with the
generator.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_h2_z_score"


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def test_no_text_defects():
    for seed in range(100):
        q = _exam(seed)
        assert find_text_defects(q, include_working=True) == [], (seed, q.text)


def test_prompt_variety_and_new_shapes_over_easy_tier():
    shapes = {prompt_shape(_exam(s)) for s in range(60)}
    assert len(shapes) >= 3, shapes
    easy = {prompt_shape(make("z_score", seed=s, difficulty="easy")) for s in range(60)}
    assert len(shapes - easy) >= 3, (shapes, easy)


def _numbers(text: str) -> list:
    return [Fraction(v) for v in re.findall(r"-?\d+(?:\.\d+)?", text)]


def _oracle(text: str) -> Fraction:
    if "Test A" in text or "exam 1" in text:
        t = re.sub(r"exam [12]", "", text)
        mean_a, sd_a, mean_b, sd_b, score_a = _numbers(t)
        return mean_b + (score_a - mean_a) / sd_a * sd_b
    nums = _numbers(text)
    if "(z =" in text:
        x1, z1, x2, z2 = nums[0], nums[1], nums[3], nums[4]
    else:
        x1, z1, x2, z2 = nums[:4]
    sd = (x2 - x1) / (z2 - z1)
    mean = x1 - z1 * sd
    return sd if "standard deviation of" in text else mean


def test_key_matches_independent_oracle_over_300_seeds():
    mismatches = []
    for seed in range(300):
        q = _exam(seed)
        expected = _oracle(q.text)
        if expected.denominator != 1 or str(int(expected)) != q.answer_text or q.answer != int(expected):
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


def test_deterministic_and_all_three_kinds_occur():
    for seed in (0, 1, 7, 99):
        assert _exam(seed).to_json() == _exam(seed).to_json()
    kinds = set()
    for seed in range(60):
        t = _exam(seed).text
        kinds.add("scales" if ("Test A" in t or "exam 1" in t) else ("sd" if "standard deviation of" in t else "mean"))
    assert kinds == {"scales", "sd", "mean"}
