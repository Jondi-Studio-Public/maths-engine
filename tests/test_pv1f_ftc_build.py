"""Tests `pv1f_ftc_derivative_read`: the skill loads with the figure-stimulus shape and every tier draws a valid
figure.

``pv1f_ftc_derivative_read`` (``answer: number``, ACMMM130, filed under ``measurement``). The skill is ``status: deferred`` (visual): every tier of
every source is generated here through the skill's own sources.
"""

import collections
import functools
import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import figures, make
from mathgen.core import VerificationError
from mathgen.generators import pv1f_ftc_build as gen
from mathgen.text_quality import prompt_shape
from scripts.measure_tiers import classify_skill

SKILL_ID = "senior-understand-and-use-the-theorem-f-left-x"
TOPIC = "pv1f_ftc_derivative_read"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)


@functools.lru_cache(maxsize=1)
def _skill():
    _domains, skills = load_skills()
    return {s.id: s for s in skills}[SKILL_ID]


def _gen(tier, seed):
    spec = next(s for s in _skill().sources if s.difficulty == tier)
    return make(spec.topic, seed=seed, difficulty=spec.difficulty, **(spec.params or {}))


def _limits_in(text):
    a = int(re.search(r"t = (-?\d+)", text).group(1))
    x0 = int(re.search(r"F'\((-?\d+)\)", text).group(1))
    return a, x0


# ------------------------------------------------------------------ the skill file


def test_skill_loads_with_the_figure_stimulus_shape():
    skill = _skill()
    assert skill.answer == "number" and skill.domain == "measurement"
    assert skill.status == "deferred" and skill.deferred_reason == "visual"
    assert skill.needs_figure is True and skill.capability_gates == ["figure_stimulus"]
    assert skill.placement_eligible is False and skill.mental is None
    assert [s.difficulty for s in skill.sources] == TIERS
    assert [list(s.stages) for s in skill.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert skill.misconceptions


def test_every_tier_draws_a_valid_figure_with_no_leak():
    for tier in TIERS:
        for seed in SEEDS:
            q = _gen(tier, seed)
            assert q.figure is not None and figures.validate(q.figure) == [], (tier, seed)
            figures.assert_no_leak(q.figure, [q.answer_text, *q.distractors])
            assert all(len(p.get("label", "")) <= 1 for p in q.figure["points"])
            assert len(q.figure["curves"]) == (2 if tier == "exam" else 1)


def test_deterministic_at_every_tier():
    for tier in TIERS:
        for seed in SEEDS:
            assert _gen(tier, seed).to_json() == _gen(tier, seed).to_json()


def test_unknown_tier_raises():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, tier="bogus")


# ------------------------------------------------------------------ the real grader


@pytest.mark.parametrize("tier", TIERS)
def test_number_grader_accepts_the_key_and_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = _gen(tier, seed)
        v = q.answer
        assert isinstance(v, int) and 0 < abs(v) <= 10 and q.answer_text == str(v)
        assert len(q.distractors) == 3, (tier, seed, q.distractors)
        for d in q.distractors:
            assert int(d) != 0
        for wrong in (v + 1, v - 1, -v):
            assert not grade_number(q, str(wrong)).correct
        assert not grade_number(q, "").correct


@pytest.mark.parametrize("tier", TIERS)
def test_answer_concentration_and_distractor_distinctness(tier):
    counts = collections.Counter()
    n = 200
    for seed in range(n):
        q = _gen(tier, seed)
        counts[q.answer_text] += 1
        values = [q.answer] + [int(d) for d in q.distractors]
        assert len(set(values)) == len(values), (tier, seed, values)
    assert max(counts.values()) / n < 0.25, (tier, counts.most_common(3))
    assert len(counts) >= 9, (tier, len(counts))


# ------------------------------------------------------------------ tier decisions


def test_each_tier_makes_its_decision_in_the_prompt_numbers():
    for seed in range(150):
        easy, med, hard, exam = (_gen(t, seed) for t in TIERS)
        a, x0 = _limits_in(easy.text)
        assert easy.answer > 0 and a < x0 and abs(a - x0) >= 2
        a, x0 = _limits_in(med.text)
        assert med.answer < 0 and a < x0
        a, x0 = _limits_in(hard.text)
        assert a > x0 and abs(a - x0) >= 2
        a, x0 = _limits_in(exam.text)
        assert abs(a - x0) >= 2 and exam.figure["points"][0]["label"] == "P"


def test_no_printed_number_is_the_key_and_prompt_never_prints_f():
    for tier in TIERS:
        for seed in range(150):
            q = _gen(tier, seed)
            a, x0 = _limits_in(q.text)
            assert q.answer not in (a, x0), (tier, seed)
            assert "f(x) =" not in q.text and "f(x) = " not in q.text.replace("y = f(x)", "")
            assert "**" not in q.text and "*" not in q.text


# ------------------------------------------------------------------ the reader


@pytest.mark.parametrize("tier", TIERS)
def test_registered_reader_equals_the_key_1000_seeds(tier):
    reader = figures.get_reader(TOPIC)
    assert reader is not None
    for seed in range(1000):
        q = _gen(tier, seed)
        assert reader(q) == q.answer_text, (tier, seed)


def test_reader_does_not_peek_at_the_answer():
    q = _gen("exam", 4)
    key = q.answer_text
    q.answer_text, q.answer, q.distractors = "99", 99, []
    assert figures.get_reader(TOPIC)(q) == key


# ------------------------------------------------------------------ the in-generator checks fire


def test_in_generator_area_function_check_fires_when_the_area_is_wrong(monkeypatch):
    monkeypatch.setattr(gen, "_area", lambda spec, a, x0: Fraction(5))
    with pytest.raises(VerificationError):
        make(TOPIC, seed=1, tier="easy")


def test_figure_reading_check_fires_on_a_shifted_curve():
    q = _gen("hard", 3)
    shifted = {
        **q.figure,
        "curves": [{"pieces": [[[x, y + 1] for x, y in piece] for piece in q.figure["curves"][0]["pieces"]]}],
    }
    assert gen._read_from_figure(shifted, q.text) != q.answer_text


def test_reader_refuses_a_curve_off_the_grid_at_x0():
    q = _gen("easy", 2)
    half = {
        **q.figure,
        "curves": [{"pieces": [[[x, y + 0.5] for x, y in piece] for piece in q.figure["curves"][0]["pieces"]]}],
    }
    with pytest.raises(VerificationError):
        gen._read_from_figure(half, q.text)


# ------------------------------------------------------------------ variety and escalation


@pytest.mark.parametrize("tier", TIERS)
def test_each_tier_has_three_shapes_and_twenty_distinct_questions(tier):
    qs = [make(TOPIC, seed=s, difficulty=tier) for s in range(60)]
    assert len({prompt_shape(q) for q in qs}) >= 3
    assert len({q.text for q in qs}) >= 20


def test_every_tier_adds_a_prompt_shape():
    verdict, adds = classify_skill(_skill(), 60)
    assert verdict == "escalates" and adds == [False, True, True, True], adds
