"""The sideways-parabola axis skill and the circle-centre skill.

Covers ``pv1d_parabola_axis_y`` (senior-recognise-features-of-the-graph-of-y-2) and
``pv1d_circle_centre`` (senior-recognise-features-of-the-graphs-of-x-2), both in
``mathgen/generators/pv1d_sideways_circle_build.py``. Both skills are ``status: deferred``
(``figure_stimulus``), so the loader never generates their questions itself: every tier is
generated here through the skill's own sources. The independent figure-to-key derivation
lives in ``tests/test_pv1d_sideways_circle_oracle.py``.
"""

import functools
from collections import Counter

import pytest

import mathgen.generators.pv1d_sideways_circle_build as gen
from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills
from mathgen import figures, make
from mathgen.core import VerificationError

DIFFICULTIES = ["easy", "medium", "hard", "exam"]
PARABOLA = "senior-recognise-features-of-the-graph-of-y-2"
CIRCLE = "senior-recognise-features-of-the-graphs-of-x-2"
SKILLS = {PARABOLA: "pv1d_parabola_axis_y", CIRCLE: "pv1d_circle_centre"}
CASES = [(skill, tier) for skill in SKILLS for tier in DIFFICULTIES]
GRADE = GRADERS["pair"]


@functools.lru_cache(maxsize=None)
def _skills():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make(skill_id, difficulty, seed):
    spec = next(s for s in _skills()[skill_id].sources if s.difficulty == difficulty)
    return make(spec.topic, seed=seed, difficulty=spec.difficulty, **(spec.params or {}))


# ------------------------------------------------------------------ the skill files


@pytest.mark.parametrize("skill_id", sorted(SKILLS))
def test_skill_loads_deferred_with_figure_gate(skill_id):
    skill = _skills()[skill_id]
    assert skill.answer == "pair"
    assert skill.status == "deferred" and skill.deferred_reason == "visual"
    assert skill.needs_figure is True
    assert skill.capability_gates == ["figure_stimulus"]
    assert skill.placement_eligible is False
    assert skill.mental is None
    assert skill.prerequisites == []
    assert [s.difficulty for s in skill.sources] == DIFFICULTIES
    assert [s.stages for s in skill.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert {s.topic for s in skill.sources} == {SKILLS[skill_id]}
    assert skill.misconceptions, "every distractor family must be named"


def test_registered_readers():
    for topic in SKILLS.values():
        assert figures.get_reader(topic) is not None


# ------------------------------------------------------------------ determinism, figure, leak


@pytest.mark.parametrize("skill_id,tier", CASES)
def test_every_seed_has_a_valid_figure_with_no_leak(skill_id, tier):
    """F1: the deferred skill is skipped by the loader, so check the figure here."""
    for seed in range(300):
        q = _make(skill_id, tier, seed)
        assert q.figure is not None, (skill_id, tier, seed)
        assert figures.validate(q.figure) == [], (skill_id, tier, seed)
        assert q.answer_text and q.text and q.latex
        figures.assert_no_leak(q.figure, [q.answer_text, *q.distractors])
        assert sum(len(p) for c in q.figure["curves"] for p in c["pieces"]) <= 400
        for c in q.figure["curves"]:
            assert sum(len(p) for p in c["pieces"]) <= 200
        for p in q.figure["points"]:
            assert len(p["label"]) == 1 and p["label"].isupper(), "a point label is one capital letter"


@pytest.mark.parametrize("skill_id,tier", CASES)
def test_figure_reader_equals_the_key_on_1000_seeds(skill_id, tier):
    reader = figures.get_reader(SKILLS[skill_id])
    for seed in range(1000):
        q = _make(skill_id, tier, seed)
        assert reader(q) == q.answer_text, (skill_id, tier, seed, q.text)


# ------------------------------------------------------------------ the real grader


def _wrong_parabola(q):
    k = q.answer["y"]
    return f"y = {k + 7}"


def _wrong_circle(q):
    return f"({q.answer['x'] + 7}, {q.answer['y']})"


@pytest.mark.parametrize("skill_id,tier", CASES)
def test_real_grader_accepts_key_and_rejects_every_distractor(skill_id, tier):
    for seed in range(200):
        q = _make(skill_id, tier, seed)
        assert GRADE(q, q.answer_text).correct, (skill_id, tier, seed, q.answer_text)
        assert len(q.distractors) == 3, (skill_id, tier, seed, q.distractors)
        assert len(set(q.distractors)) == 3 and q.answer_text not in q.distractors
        for d in q.distractors:
            assert not GRADE(q, d).correct, (skill_id, tier, seed, d, q.answer_text)
        wrong = _wrong_parabola(q) if skill_id == PARABOLA else _wrong_circle(q)
        assert not GRADE(q, wrong).correct, (skill_id, tier, seed, wrong)


@pytest.mark.parametrize("tier", DIFFICULTIES)
def test_parabola_every_distractor_keeps_the_label_y(tier):
    """Every distractor keeps the key's own label 'y =' (and is still rejected)."""
    for seed in range(200):
        q = _make(PARABOLA, tier, seed)
        assert all(d.startswith("y = ") for d in q.distractors), (tier, seed, q.distractors)


@pytest.mark.parametrize("tier", DIFFICULTIES)
def test_parabola_spelling_variants(tier):
    """Accepted: 'y=3' and 'y = 3'. Rejected: a bare '3', the other axis ('x = 3') and the sign flip."""
    for seed in range(40):
        q = _make(PARABOLA, tier, seed)
        k = q.answer["y"]
        for given in (f"y = {k}", f"y={k}"):
            assert GRADE(q, given).correct, (tier, seed, given)
        for given in (str(k), f"x = {k}", f"y = {-k}"):
            assert not GRADE(q, given).correct, (tier, seed, given)


@pytest.mark.parametrize("tier", DIFFICULTIES)
def test_circle_spelling_variants_and_swap(tier):
    for seed in range(60):
        q = _make(CIRCLE, tier, seed)
        a, b = q.answer["x"], q.answer["y"]
        for given in (f"({a}, {b})", f"({a},{b})", f"{a}, {b}"):
            assert GRADE(q, given).correct, (tier, seed, given)
        if a != b:
            assert not GRADE(q, f"({b}, {a})").correct, (tier, seed)


# ------------------------------------------------------------------ construction targets


@pytest.mark.parametrize("skill_id,tier", CASES)
def test_answer_concentration_at_200_seeds(skill_id, tier):
    counts = Counter(_make(skill_id, tier, s).answer_text for s in range(200))
    assert max(counts.values()) / 200 <= 0.25, counts.most_common(3)
    assert len(counts) >= (8 if skill_id == PARABOLA else 25), len(counts)


@pytest.mark.parametrize("skill_id,tier", CASES)
def test_at_least_20_distinct_questions_and_3_templates_over_60_seeds(skill_id, tier):
    from mathgen.text_quality import prompt_shape

    shapes = {prompt_shape(_make(skill_id, tier, s)) for s in range(60)}
    assert len(shapes) >= 3


def test_circle_centre_class_shares():
    """a = 0 or b = 0 on a small stated share (<= 20%); a != b on >= 85% of draws."""
    for tier in DIFFICULTIES:
        qs = [_make(CIRCLE, tier, s) for s in range(400)]
        axis = sum(1 for q in qs if q.answer["x"] == 0 or q.answer["y"] == 0) / 400
        diff = sum(1 for q in qs if q.answer["x"] != q.answer["y"]) / 400
        if tier != "exam":
            assert axis <= 0.20, (tier, axis)
        assert diff >= 0.85, (tier, diff)


# ------------------------------------------------------------------ the in-generator check fires


def _shift_up(fig):
    """Move the curve and every marked point up one unit: still a symmetric figure, but its
    reading is now one more than the key."""
    for c in fig["curves"]:
        for piece in c["pieces"]:
            for p in piece:
                p[1] += 1
    for p in fig["points"]:
        p["at"][1] += 1
    fig["y_range"] = [fig["y_range"][0] + 1, fig["y_range"][1] + 1]
    return fig


@pytest.mark.parametrize("tier", ["easy", "medium", "hard", "exam"])
def test_parabola_generator_check_fires_on_a_corrupted_figure(tier, monkeypatch):
    real = gen._parabola_figure
    monkeypatch.setattr(gen, "_parabola_figure", lambda *a, **kw: _shift_up(real(*a, **kw)))
    with pytest.raises((VerificationError, ValueError)):
        make("pv1d_parabola_axis_y", seed=1, difficulty=tier, variant=tier)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard", "exam"])
def test_circle_generator_check_fires_on_a_corrupted_figure(tier, monkeypatch):
    real = gen._circle_figure
    monkeypatch.setattr(gen, "_circle_figure", lambda *a, **kw: _shift_up(real(*a, **kw)))
    with pytest.raises((VerificationError, ValueError)):
        make("pv1d_circle_centre", seed=1, difficulty=tier, variant=tier)


def test_generator_check_fires_when_a_distractor_collides_with_the_key():
    with pytest.raises(VerificationError):
        gen.check_pairwise_distinct(3, 3, -3, 0)


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        make("pv1d_parabola_axis_y", seed=1, variant="nightmare")
    with pytest.raises(ValueError):
        make("pv1d_circle_centre", seed=1, variant="nightmare")


def test_hard_circle_never_labels_a_circumference_point_C():
    """Easy teaches that C marks a centre; hard/exam must not reuse C for a point on a circle."""
    for tier in ("hard", "exam"):
        for seed in range(300):
            labels = {p["label"] for p in _make(CIRCLE, tier, seed).figure["points"]}
            assert "C" not in labels, (tier, seed)


def test_default_variant_is_easy():
    assert make("pv1d_parabola_axis_y", seed=4).figure["points"][0]["label"] == "V"
    assert make("pv1d_circle_centre", seed=4).figure["points"][0]["label"] == "C"
