"""Tests for the exam source of `senior-use-the-scale-factor-for-two-similar-figures` (area and volume ratios)."""

from __future__ import annotations

from pathlib import Path
import re

import sympy
from mathgen import make
from mathgen.grading.typed import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_h4_similar_area_volume_scale"
HARD_TOPIC = "similar_figure_scale_factor_length"
HARD_PARAMS = {"mode": "derive_perimeter"}
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

    path = next(
        Path(__file__).resolve().parents[1].glob("skills/*/senior-use-the-scale-factor-for-two-similar-figures.yaml")
    )
    skill = yaml.safe_load(path.read_text(encoding="utf-8"))
    last = skill["sources"][-1]
    assert last["topic"] == "exd_x_r3b_similar_scale"  # the exam source moved to the multi-part chain
    assert last["difficulty"] == "exam"
    assert last["stages"] == ["exam"]
    assert last["params"] == {"tier": "exam"}
    assert [s["difficulty"] for s in skill["sources"]] == ["easy", "medium", "hard", "exam"]


def _unique(values):
    out = []
    for v in values:
        if v not in out:
            out.append(v)
    return out


def _oracle(text):
    solids = "cm^3" in text
    measures = [int(v) for v in _unique(re.findall(r"(\d+) cm\^" + ("3" if solids else "2"), text))]
    plain = [int(v) for v in _unique(re.findall(r"(\d+) cm(?!\^)", text))]
    if text.rstrip().endswith("in cm^2."):  # asks for an area
        area1, (l1, l2) = measures[0], plain[:2]
        return int(sympy.Rational(l2, l1) ** 2 * area1)
    m1, m2 = measures
    ratio = sympy.Rational(m2, m1)
    factor = sympy.root(ratio, 3 if solids else 2)
    value = plain[0] * factor
    assert value.is_Integer, value
    return int(value)


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
