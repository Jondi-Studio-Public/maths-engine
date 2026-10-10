"""Tests for the exam source of `senior-midpoints-of-the-sides-of-a-quadrilateral-join` (rebuild a vertex)."""

from __future__ import annotations

from pathlib import Path
import re
from mathgen import make
from mathgen.grading.typed import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_h4_midpoints_reconstruct"
HARD_TOPIC = "midpoints_of_the_sides_of_a_quadrilateral_join"
HARD_PARAMS = {"variant": "hard", "coord_lo": -14, "coord_hi": 14}
_GRADER = GRADERS["pair"]


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
        Path(__file__).resolve().parents[1].glob("skills/*/senior-midpoints-of-the-sides-of-a-quadrilateral-join.yaml")
    )
    skill = yaml.safe_load(path.read_text(encoding="utf-8"))
    last = skill["sources"][-1]
    assert last["topic"] == TOPIC
    assert last["difficulty"] == "exam"
    assert last["stages"] == ["exam"]
    assert last["params"] == {"tier": "exam"}
    assert [s["difficulty"] for s in skill["sources"]] == ["easy", "medium", "hard", "exam"]


def _oracle(text):
    pts = {n: (int(x), int(y)) for n, x, y in re.findall(r"\b([APQRS])(?: is)? ?\((-?\d+), (-?\d+)\)", text)}
    a, p, q = pts["A"], pts["P"], pts["Q"]
    b = (2 * p[0] - a[0], 2 * p[1] - a[1])
    c = (2 * q[0] - b[0], 2 * q[1] - b[1])
    d = (2 * pts["R"][0] - c[0], 2 * pts["R"][1] - c[1]) if "R" in pts else None
    asked = re.search(r"coordinates of (?:vertex )?([CD])", text).group(1)
    if "S(" in text:  # S is redundant: it must agree with the vertices found
        s = pts["S"]
        assert (d[0] + a[0], d[1] + a[1]) == (2 * s[0], 2 * s[1])
    x, y = {"C": c, "D": d}[asked]
    return f"x = {x}, y = {y}"


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
