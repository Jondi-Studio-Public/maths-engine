"""Tests for the exam source of `senior-the-alternate-segment-theorem` (two tangents from an external point)."""

from __future__ import annotations

from pathlib import Path
import re

import sympy
from mathgen import make
from mathgen.grading.typed import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_h4_tangent_pair_alternate_segment"
HARD_TOPIC = "alternate_segment_theorem"
HARD_PARAMS = {"variant": "algebraic"}
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

    path = next(Path(__file__).resolve().parents[1].glob("skills/*/senior-the-alternate-segment-theorem.yaml"))
    skill = yaml.safe_load(path.read_text(encoding="utf-8"))
    last = skill["sources"][-1]
    assert last["topic"] == TOPIC
    assert last["difficulty"] == "exam"
    assert last["stages"] == ["exam"]
    assert last["params"] == {"tier": "exam"}
    assert [s["difficulty"] for s in skill["sources"]] == ["easy", "medium", "hard", "exam"]


_X = sympy.Symbol("x")


def _expr(raw):
    raw = raw.replace(" ", "")
    return sympy.sympify(re.sub(r"(\d)x", r"\1*x", raw), locals={"x": _X})


def _oracle(text):
    """Angle APB + 2 x angle AQB = 180 (isosceles triangle PAB and the alternate segment theorem)."""
    algebraic = re.search(r"Angle APB (?:=|is) \(([^)]*)\).*?angle AQB (?:=|is) \(([^)]*)\)", text)
    if algebraic:
        apb, aqb = _expr(algebraic.group(1)), _expr(algebraic.group(2))
        (x,) = sympy.solve(apb + 2 * aqb - 180, _X)
        return int(apb.subs(_X, x))
    if "Find angle AQB" in text or "Calculate angle AQB" in text:
        apb = int(re.search(r"[Aa]ngle APB (?:=|is) (\d+)", text).group(1))
        assert apb % 2 == 0
        return (180 - apb) // 2
    aqb = int(re.search(r"angle AQB (?:=|equal to|is) (?:equal to )?(\d+)", text).group(1))
    return 180 - 2 * aqb


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
