"""Exam source of the complement and union rules skill: `exd_x_s4_union`."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_s4_union"


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam", tier="exam")


def _hard(seed):
    return make("probability_complement_union", seed=seed, difficulty="hard", tier="hard")


def _fracs(text):
    return [Fraction(int(a), int(b)) for a, b in re.findall(r"(\d+)/(\d+)", text)]


def _oracle(text):
    """Re-derive the key from the rendered prompt alone, with exact fractions."""
    v = _fracs(text)
    assert len(v) == 3, text
    if "not A" in text or "does not occur" in text:
        not_a, pb, union = v
        return (1 - not_a) + pb - union
    if "neither" in text:
        pa, pb, neither = v
        return pa + pb - (1 - neither)
    pa, pb, union = v
    return 2 * union - pa - pb


def test_no_text_defects():
    for seed in range(60):
        q = _exam(seed)
        assert find_text_defects(q) == [], q.text


def test_shapes_and_escalation_in_kind():
    shapes = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(_hard(s)) for s in range(60)}
    assert len(shapes) >= 3
    assert shapes - hard, "the exam source must add a shape the hard tier lacks"


def test_independent_oracle_matches_key():
    bad = []
    for seed in range(400):
        q = _exam(seed)
        if _oracle(q.text) != Fraction(q.answer["p"], q.answer["q"]):
            bad.append((seed, q.text, q.answer_text))
    assert bad == []


def test_all_three_variants_occur():
    texts = [_exam(s).text for s in range(60)]
    assert any("neither" in t for t in texts)
    assert any("not A" in t or "does not occur" in t for t in texts)
    assert any("exactly one" in t or "not both" in t or "one and only one" in t for t in texts)


def test_grader_accepts_key_and_rejects_every_distractor():
    grader = GRADERS["fraction"]
    bad = []
    for seed in range(150):
        q = _exam(seed)
        assert len(q.distractors) >= 2
        if not grader(q, q.answer_text).correct:
            bad.append(("key rejected", seed))
        for d in q.distractors:
            if grader(q, d).correct:
                bad.append(("distractor accepted", seed, d))
    assert bad == []


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
