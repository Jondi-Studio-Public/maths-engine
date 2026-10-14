"""Tests `pv1d_quadratic_root_count` (ACMMM011) and `pv1d_parabola_axis_x` (ACMMM007), two parabola figure
skills.

``pv1d_quadratic_root_count`` (mcq over {0, 1, 2}, ACMMM011) and
``pv1d_parabola_axis_x`` (``x = b`` as a single-equation pair, ACMMM007). Both skills are
``status: active`` (figure_stimulus passed), and the figure is still checked here: every
tier of every source is generated here through the skill's own sources.
"""

import collections
import functools

import pytest

from mathgen import figures, make
from mathgen.capabilities import load_capability_gates
from mathgen.core import VerificationError
from mathgen.generators import pv1d_parabola_build as gen
from mathgen.grading.typed import GRADERS, grade_mcq, grade_pair, mcq_options
from mathgen.skills import load_skills
from mathgen.text_quality import prompt_shape
from scripts.measure_tiers import classify_skill

ROOT_ID = "senior-recognise-features-of-the-graph-of-the-general"
AXIS_ID = "senior-recognise-features-of-the-graphs-of-y-x"
ROOT_TOPIC = "pv1d_quadratic_root_count"
AXIS_TOPIC = "pv1d_parabola_axis_x"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)


@functools.lru_cache(maxsize=1)
def _skills():
    _domains, skills = load_skills()
    return {s.id: s for s in skills}


def _gen(skill_id, tier, seed):
    spec = next(s for s in _skills()[skill_id].sources if s.difficulty == tier)
    return make(spec.topic, seed=seed, difficulty=spec.difficulty, **(spec.params or {}))


# ------------------------------------------------------------------ the skill files


@pytest.mark.parametrize(
    "skill_id,answer,domain",
    [
        (ROOT_ID, "mcq", "quadratics"),
        (AXIS_ID, "pair", "functions-sequences"),
    ],
)
def test_skill_loads_with_the_figure_stimulus_shape(skill_id, answer, domain):
    skill = _skills()[skill_id]
    assert skill.answer == answer and skill.domain == domain
    assert skill.status == "active" and skill.deferred_reason is None
    assert skill.needs_figure is True and skill.capability_gates == ["figure_stimulus"]
    assert load_capability_gates()["figure_stimulus"]["status"] == "passed"
    assert skill.placement_eligible is False and skill.mental is None
    assert [s.difficulty for s in skill.sources] == TIERS
    assert [list(s.stages) for s in skill.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert skill.misconceptions


@pytest.mark.parametrize("skill_id", [ROOT_ID, AXIS_ID])
def test_every_tier_of_every_source_draws_a_valid_figure(skill_id):
    for tier in TIERS:
        for seed in SEEDS:
            q = _gen(skill_id, tier, seed)
            assert q.figure is not None, (tier, seed)
            assert figures.validate(q.figure) == [], (tier, seed)
            figures.assert_no_leak(q.figure, [q.answer_text, *q.distractors])
            assert all(len(p.get("label", "")) <= 1 for p in q.figure["points"])


@pytest.mark.parametrize("skill_id", [ROOT_ID, AXIS_ID])
def test_deterministic_at_every_tier(skill_id):
    for tier in TIERS:
        for seed in SEEDS:
            assert _gen(skill_id, tier, seed).to_json() == _gen(skill_id, tier, seed).to_json()


def test_unknown_tier_raises():
    for topic in (ROOT_TOPIC, AXIS_TOPIC):
        with pytest.raises(ValueError):
            make(topic, seed=1, tier="bogus")


# ------------------------------------------------------------------ the real graders: root count


@pytest.mark.parametrize("tier", TIERS)
def test_mcq_grader_accepts_the_key_and_rejects_every_other_option(tier):
    for seed in SEEDS:
        q = _gen(ROOT_ID, tier, seed)
        options = mcq_options(q)
        assert sorted(options) == ["0", "1", "2"], (tier, seed, options)
        assert len(q.distractors) == 2 and q.answer_text not in q.distractors
        for idx, opt in enumerate(options):
            got = grade_mcq(q, idx).correct
            assert got == (opt == q.answer_text), (tier, seed, idx, opt, q.answer_text)
        wrong = (int(q.answer_text) + 1) % 3
        assert not grade_mcq(q, options.index(str(wrong))).correct


@pytest.mark.parametrize("tier", TIERS)
def test_root_count_options_are_plain_alike_and_unlabelled(tier):
    for seed in SEEDS:
        q = _gen(ROOT_ID, tier, seed)
        assert q.answer_latex == f"plain:{q.answer_text}"
        assert all(o in ("0", "1", "2") for o in [q.answer_text, *q.distractors])
        assert q.answer == q.answer_text


@pytest.mark.parametrize("tier", TIERS)
def test_root_count_key_position_is_balanced(tier):
    pos = collections.Counter()
    cls = collections.Counter()
    n = 400
    for seed in range(n):
        q = _gen(ROOT_ID, tier, seed)
        pos[mcq_options(q).index(q.answer_text)] += 1
        cls[q.answer_text] += 1
    assert max(pos.values()) / n < 0.42, (tier, dict(pos))
    assert max(cls.values()) / n < 0.42, (tier, dict(cls))


# ------------------------------------------------------------------ the real graders: axis


@pytest.mark.parametrize("tier", TIERS)
def test_pair_grader_accepts_the_key_and_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = _gen(AXIS_ID, tier, seed)
        b = q.answer["x"]
        assert q.answer_text == f"x = {b}" and b != 0 and isinstance(b, int)
        assert grade_pair(q, f"x={b}").correct and grade_pair(q, f" x = {b} ").correct
        assert not grade_pair(q, str(b)).correct  # the line has to be written as "x = b"
        assert len(q.distractors) == 3, (tier, seed, q.distractors)
        for d in q.distractors:
            assert d.startswith("x = "), d  # every distractor keeps the key's own label
            assert not grade_pair(q, d.replace(" ", "")).correct
        for wrong in (b + 1, b - 1, -b):
            assert not grade_pair(q, f"x = {wrong}").correct, (tier, seed, wrong)
        assert not grade_pair(q, "").correct


def test_a_wrong_label_is_rejected():
    """ "y = b" is a different line from the key "x = b", so it is wrong even with the same number."""
    q = _gen(AXIS_ID, "easy", 0)
    assert not grade_pair(q, f"y = {q.answer['x']}").correct


@pytest.mark.parametrize("tier", TIERS)
def test_axis_answer_concentration_and_distractor_distinctness(tier):
    counts = collections.Counter()
    n = 200
    for seed in range(n):
        q = _gen(AXIS_ID, tier, seed)
        counts[q.answer_text] += 1
        values = [q.answer["x"]] + [int(d.split("=")[1]) for d in q.distractors]
        assert len(set(values)) == len(values), (tier, seed, values)
    assert max(counts.values()) / n < 0.25, (tier, counts.most_common(3))


# ------------------------------------------------------------------ the readers


def _reader_text(topic, q):
    return figures.get_reader(topic)(q)


@pytest.mark.parametrize("topic,skill_id", [(ROOT_TOPIC, ROOT_ID), (AXIS_TOPIC, AXIS_ID)])
@pytest.mark.parametrize("tier", TIERS)
def test_registered_reader_equals_the_key_1000_seeds(topic, skill_id, tier):
    assert figures.get_reader(topic) is not None
    for seed in range(1000):
        q = _gen(skill_id, tier, seed)
        assert _reader_text(topic, q) == q.answer_text, (tier, seed)


def test_reader_does_not_peek_at_the_answer():
    q = _gen(AXIS_ID, "hard", 4)
    q.answer_text, q.answer, q.distractors = "x = 99", {"x": 99}, []
    assert figures.get_reader(AXIS_TOPIC)(q) != "x = 99"


# ------------------------------------------------------------------ the in-generator checks fire


def test_in_generator_root_count_check_fires_on_a_corrupted_answer():
    q = _gen(ROOT_ID, "exam", 3)
    key = int(q.answer_text)
    gen._verify_root_count(q.figure, q.text, key)
    with pytest.raises(VerificationError):
        gen._verify_root_count(q.figure, q.text, (key + 1) % 3)


def test_in_generator_discriminant_check_fires_when_the_class_is_wrong(monkeypatch):
    monkeypatch.setattr(gen, "_discriminant_class", lambda a, h, c: 3)
    with pytest.raises(VerificationError):
        make(ROOT_TOPIC, seed=1, tier="easy")


def test_in_generator_axis_check_fires_when_the_axis_is_wrong(monkeypatch):
    monkeypatch.setattr(gen, "_axis_from_coefficients", lambda a, h, c: h + 1)
    with pytest.raises(VerificationError):
        make(AXIS_TOPIC, seed=1, tier="hard")


def test_axis_reader_refuses_a_figure_with_no_equal_height_pair():
    q = _gen(AXIS_ID, "hard", 2)
    fig = {
        **q.figure,
        "points": [
            dict(q.figure["points"][0]),
            {**q.figure["points"][1], "at": [q.figure["points"][1]["at"][0], q.figure["points"][1]["at"][1] + 1]},
        ],
    }
    with pytest.raises(VerificationError):
        gen._read_axis_text(fig)


# ------------------------------------------------------------------ variety and escalation


@pytest.mark.parametrize("topic", [ROOT_TOPIC, AXIS_TOPIC])
@pytest.mark.parametrize("tier", TIERS)
def test_each_tier_has_three_shapes_and_twenty_distinct_questions(topic, tier):
    qs = [make(topic, seed=s, difficulty=tier) for s in range(60)]
    assert len({prompt_shape(q) for q in qs}) >= 3
    assert len({q.text for q in qs}) >= 20


@pytest.mark.parametrize("skill_id", [ROOT_ID, AXIS_ID])
def test_every_tier_adds_a_prompt_shape(skill_id):
    verdict, adds = classify_skill(_skills()[skill_id], 60)
    assert verdict == "escalates" and adds == [False, True, True, True], adds


# ------------------------------------------------------------------ prompts never state the answer


@pytest.mark.parametrize("tier", TIERS)
def test_prompts_state_no_number_and_no_python_notation(tier):
    for seed in SEEDS:
        for q in (_gen(ROOT_ID, tier, seed), _gen(AXIS_ID, tier, seed)):
            assert "**" not in q.text and "tangent" not in q.text.lower() and "touch" not in q.text.lower()
        qa = _gen(AXIS_ID, tier, seed)
        assert qa.answer_text not in qa.text
        assert not any(ch.isdigit() for ch in qa.text.replace("1 unit", "")), qa.text.replace("x^2", "")


def test_graders_registry_has_both_kinds():
    assert "mcq" in GRADERS and "pair" in GRADERS
