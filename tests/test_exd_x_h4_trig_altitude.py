"""Tests for the exam source of `senior-apply-the-cosine-and-sine-ratios-to-find` (altitude to the hypotenuse)."""

from __future__ import annotations

from pathlib import Path
import math
import re
from decimal import ROUND_HALF_UP, Decimal
from mathgen import make
from mathgen.grading.typed import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_h4_trig_altitude"
HARD_TOPIC = "sine_cosine_ratio_triangle"
HARD_PARAMS = {"variant": "chained"}
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
        Path(__file__).resolve().parents[1].glob("skills/*/senior-apply-the-cosine-and-sine-ratios-to-find.yaml")
    )
    skill = yaml.safe_load(path.read_text(encoding="utf-8"))
    # The chain is the default exam source; the old generator follows it as the extra `exam-single` source.
    chain, last = skill["sources"][-2:]
    assert chain["topic"] == "exd_x_r2f_trig_altitude"
    assert chain["difficulty"] == "exam"
    assert chain["stages"] == ["exam"]
    assert chain["params"] == {"tier": "exam"}
    assert "key" not in chain
    assert last["topic"] == "exd_x_h4_trig_altitude"
    assert last["difficulty"] == "exam"
    assert last["stages"] == ["exam"]
    assert last["params"] == {"tier": "exam"}
    assert last["key"] == "exam-single"
    assert [s["difficulty"] for s in skill["sources"]] == ["easy", "medium", "hard", "exam", "exam"]


def _oracle(text):
    """Place the triangle on coordinate axes and read the length off, rather than using the ratio formulas."""
    h = int(re.search(r"(?:AC = |hypotenuse AC = )(\d+) cm", text).group(1))
    alpha = math.radians(int(re.search(r"angle BAC = (\d+) degrees", text).group(1)))
    # A at the origin, C on the x-axis; B is on the circle with diameter AC.
    bx, by = h * math.cos(alpha) * math.cos(alpha), h * math.cos(alpha) * math.sin(alpha)
    asked = re.search(r"(?:length of|Calculate) (AD|DC|BD)", text).group(1)
    value = {"AD": bx, "DC": h - bx, "BD": by}[asked]
    return str(Decimal(repr(round(value, 9))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def test_key_matches_independent_rederivation_from_the_prompt_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _make(seed)
        expected = _oracle(q.text)
        if expected != q.answer_text:
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
