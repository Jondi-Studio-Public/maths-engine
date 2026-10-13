"""Tests the exam tier of `f10-trig-ratios-similarity`: platform height, second ramp length, gap between feet."""

from __future__ import annotations

import math
import re
from decimal import ROUND_HALF_UP, Decimal

import mathgen.generators.exd_x_u7_two_ramps_chain  # noqa: F401  (registers the topic)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u7_two_ramps_chain"
SKILL_ID = "f10-trig-ratios-similarity"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _dp1(x: float) -> str:
    return str(Decimal(repr(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _givens(q):
    stem = q.text.split(" (a)")[0]
    L = float(re.search(r"(\d+\.\d) m long", stem).group(1))
    theta, phi = (int(v) for v in re.findall(r"(\d+)°", stem))
    return L, theta, phi, "same side" in stem


def oracle(q) -> dict[str, str]:
    """Re-derive all three keys from the stem with floats and the radians functions."""
    L, theta, phi, same = _givens(q)
    t, p = math.radians(theta), math.radians(phi)
    h = L * math.sin(t)
    L2 = h / math.sin(p)
    far = L2 * math.cos(p) - L * math.cos(t) if same else L2 * math.cos(p) + L * math.cos(t)
    return {"a": _dp1(h), "b": _dp1(L2), "c": _dp1(far)}


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_source_wired_last():
    src = _SKILL.sources[-1]
    assert (src.topic, src.difficulty, src.stages, src.params) == (TOPIC, "exam", ["exam"], {"tier": "exam"})


def test_shape_and_marks():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [2, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert find_text_defects(q) == []


def test_prompt_shapes_beyond_the_hard_tier():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make("f10_trig_ratios_similarity", seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3 and exam - hard


def test_key_matches_independent_derivation_over_320_seeds():
    bad = [s for s in range(320) if oracle(_exam(s)) != _canonical(_exam(s))]
    assert bad == []


def test_both_layouts_appear():
    layouts = {_givens(_exam(s))[3] for s in range(60)}
    assert layouts == {True, False}


def test_key_accepted_and_every_distractor_rejected():
    for seed in range(120):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 6
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2
            for text in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: text}).parts[p.label].marks == 0, (seed, p.label, text)


def test_follow_through_from_a_wrong_height():
    checked = 0
    for seed in range(120):
        q = _exam(seed)
        L, theta, phi, same = _givens(q)
        h_wrong = float(q.parts[0].distractors[0])
        p, t = math.radians(phi), math.radians(theta)
        L2 = h_wrong / math.sin(p)
        far = L2 * math.cos(p) - L * math.cos(t) if same else L2 * math.cos(p) + L * math.cos(t)
        if far <= 0:
            continue
        # Use the rounded earlier answer, as a learner who rounds as asked does.
        L2_r = float(_dp1(L2))
        far_r = L2_r * math.cos(p) - L * math.cos(t) if same else L2_r * math.cos(p) + L * math.cos(t)
        result = grade(_SKILL, q, {"a": f"{h_wrong:.1f}", "b": _dp1(L2), "c": _dp1(far_r)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through is True, (seed, result)
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True, (seed, result)
        checked += 1
    assert checked >= 80


def test_correct_chain_rounded_as_asked_is_full_marks():
    for seed in range(60):
        q = _exam(seed)
        canon = _canonical(q)
        assert grade(_SKILL, q, canon).marks == 6


def test_answer_space():
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 15


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
