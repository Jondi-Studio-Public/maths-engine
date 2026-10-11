"""Tests for the exam source of `senior-an-angle-in-a-semicircle-is-a-right` (composed length questions)."""

from __future__ import annotations

from pathlib import Path
import math
import re
from mathgen import make
from mathgen.grading.typed import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_h4_semicircle_composed"
HARD_TOPIC = "angle_in_semicircle"
HARD_PARAMS = {"variant": "ratio"}
_GRADER = GRADERS["number"]


def _make(seed):
    return make(TOPIC, seed=seed, difficulty="exam", tier="exam")


def _hard(seed):
    return make(HARD_TOPIC, seed=seed, **HARD_PARAMS)


def test_prompt_variety_and_new_shapes_over_hard_tier():
    shapes = {prompt_shape(_make(s)) for s in range(60)}
    hard_shapes = {prompt_shape(_hard(s)) for s in range(60)}
    assert len(shapes) >= 3, shapes
    assert len(shapes - hard_shapes) >= 3, (shapes, hard_shapes)


def test_no_text_defects_and_deterministic():
    for seed in range(60):
        q = _make(seed)
        assert find_text_defects(q) == [], (seed, q.text)
        assert q.to_json() == _make(seed).to_json()


def test_skill_yaml_has_the_exam_source_last():
    import yaml

    path = next(Path(__file__).resolve().parents[1].glob("skills/*/senior-an-angle-in-a-semicircle-is-a-right.yaml"))
    skill = yaml.safe_load(path.read_text(encoding="utf-8"))
    last = skill["sources"][-1]
    assert last["topic"] == TOPIC
    assert last["difficulty"] == "exam"
    assert last["stages"] == ["exam"]
    assert last["params"] == {"tier": "exam"}
    assert [s["difficulty"] for s in skill["sources"]] == ["easy", "medium", "hard", "exam"]


def _exact_sqrt(n):
    r = math.isqrt(n)
    assert r * r == n, n
    return r


def _oracle(text):
    """Key re-derived from the rendered prompt alone, with integer arithmetic."""
    if "AH = " in text:
        p, q = (int(v) for v in re.findall(r"(?:AH|HB) = (\d+) cm", text))
        return _exact_sqrt(p * q)
    if "radius of the circle" in text:
        a, b = (int(v) for v in re.search(r"AC = (\d+) cm.*?BC = (\d+) cm", text).groups())
        return _exact_sqrt(a * a + b * b) // 2
    r = int(re.search(r"radius (\d+) cm", text).group(1))
    a = int(re.search(r"AC (?:=|is) (\d+)", text).group(1))
    return a * _exact_sqrt(4 * r * r - a * a) // 2


def test_key_matches_independent_rederivation_from_the_prompt_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _make(seed)
        expected = _oracle(q.text)
        if expected != q.answer:
            bad.append((seed, q.text, q.answer_text, expected))
    assert bad == [], bad[:3]


def test_grader_accepts_key_and_rejects_every_distractor_over_150_seeds():
    bad = []
    for seed in range(150):
        q = _make(seed)
        if not _GRADER(q, q.answer_text).correct:
            bad.append(("key rejected", seed, q.answer_text))
        assert len(q.distractors) >= 2, (seed, q.distractors)
        for d in q.distractors:
            if _GRADER(q, d).correct:
                bad.append(("distractor accepted", seed, d, q.answer_text))
    assert bad == [], bad[:3]
