"""Tests for the exam source of `senior-angles-at-the-circumference-of-a-circle-subtended` (crossing chords)."""

from __future__ import annotations

from pathlib import Path
import re

import sympy
from mathgen import make
from mathgen.grading.typed import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_h4_chords_intersecting"
HARD_TOPIC = "angles_same_arc"
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


def test_skill_yaml_has_the_exam_source_last():
    import yaml

    path = next(
        Path(__file__)
        .resolve()
        .parents[1]
        .glob("skills/*/senior-angles-at-the-circumference-of-a-circle-subtended.yaml")
    )
    skill = yaml.safe_load(path.read_text(encoding="utf-8"))
    # The multi-part chain is second to last; this single-part topic was re-added after it as key exam-single (bf2fca7).
    chain, last = skill["sources"][-2:]
    assert chain["topic"] == "exd_x_r2f_chords_chain"
    assert last["topic"] == "exd_x_h4_chords_intersecting"
    assert last["key"] == "exam-single"
    for source in (chain, last):
        assert source["difficulty"] == "exam"
        assert source["stages"] == ["exam"]
        assert source["params"] == {"tier": "exam"}
    assert [s["difficulty"] for s in skill["sources"]] == ["easy", "medium", "hard", "exam", "exam"]


_X = sympy.Symbol("x")


def _expr(s):
    return sympy.sympify(re.sub(r"(\d)x", r"\1*x", s), locals={"x": _X})


def _angle(text, name):
    m = re.search(rf"[Aa]ngle {name} (?:=|is) (\([^)]*\)|\d+)", text)
    raw = m.group(1).strip("()")
    return _expr(raw.replace(" ", ""))


def _oracle(text):
    bac, bdc = _angle(text, "BAC"), _angle(text, "BDC")
    (x,) = sympy.solve(bac - bdc, _X)
    abd = _angle(text, "ABD").subs(_X, x)
    if "angle CPD" in text:
        return int(180 - bac.subs(_X, x) - abd)
    dac = _angle(text, "DAC")
    return int(abd + dac)


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
