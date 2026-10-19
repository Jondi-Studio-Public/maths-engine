"""Tests for the exam source of `senior-the-angle-at-the-centre-subtended-by-an` (isosceles radius triangles)."""

from __future__ import annotations

from pathlib import Path
import re
from mathgen import make
from mathgen.grading.typed import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_h4_centre_isosceles_chords"
HARD_TOPIC = "the_angle_at_the_centre_subtended_by_an_arc"
HARD_PARAMS = {"variant": "reflex"}
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


def test_skill_yaml_has_the_exam_source_last():
    import yaml

    path = next(
        Path(__file__).resolve().parents[1].glob("skills/*/senior-the-angle-at-the-centre-subtended-by-an.yaml")
    )
    skill = yaml.safe_load(path.read_text(encoding="utf-8"))
    last = skill["sources"][-1]
    # The skill's exam source is now the multi-part chain; this module's generator stays registered on its own.
    assert last["topic"] == "exd_x_r3e_circle_centre_chain"
    assert last["difficulty"] == "exam"
    assert last["stages"] == ["exam"]
    assert last["params"] == {"tier": "exam"}
    assert [s["difficulty"] for s in skill["sources"]] == ["easy", "medium", "hard", "exam"]


def _oracle(text):
    a = int(re.search(r"[Aa]ngle OAB (?:=|is) (\d+)", text).group(1))
    if "Calculate angle OCB" in text or "Find angle OCB" in text:
        m = int(re.search(r"(?:[Nn]on-reflex )?angle AOC (?:=|is) (\d+)", text).group(1))
        assert "major arc" in text and m % 2 == 0
        return m // 2 - a
    c = int(re.search(r"[Aa]ngle OCB (?:=|is) (\d+)", text).group(1))
    abc = a + c  # base angles of the two isosceles triangles add to angle ABC
    reflex_or_not = 2 * abc
    if "minor arc" in text:
        assert abc > 90
        return 360 - reflex_or_not
    assert abc < 90 and "major arc" in text
    return reflex_or_not


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
