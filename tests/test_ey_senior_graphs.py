"""Seven senior figure skills built on `cartesian`, `dot_plot` and `histogram` figures (no new figure kind).

Transformed graphs (ACMMM025, ACMMM026, ACMSM099), a modelled straight line (ACMGM042), a position-time curve with a
tangent (ACMMM094), simulated-sample histograms (ACMMM173) and the spread of a dot plot (ACMEM054). Every skill is
generated here through its own sources at every tier and 60 seeds: the figure is valid, does not show the key, the
registered reader re-derives the key from the figure, and the real graders accept the key and reject every distractor.
The exam tiers are connected chains of parts; each part's distractors are tried inside the full labelled reply.
"""

from __future__ import annotations

import functools

import pytest

from mathgen import figures, make
from mathgen.capabilities import load_capability_gates
from mathgen.core import VerificationError
from mathgen.generators import (
    ey_senior_context_graphs_build as ctx,
)
from mathgen.generators import (
    ey_senior_spread_build as spread,
)
from mathgen.generators import (
    ey_senior_transform_build as tf,
)
from mathgen.grading.typed import grade, mcq_options
from mathgen.skills import load_skills
from scripts import skill_contract

SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]
FIGURE_KINDS = {"cartesian", "dot_plot", "histogram"}

# skill id -> (topic, exam topic, domain, answer kind, ACARA code tag, figure kind)
SKILLS = {
    "senior-graph-translation-read-shift": (
        "ey_graph_translation",
        "ey_graph_translation_exam",
        "functions-sequences",
        "mcq",
        "acmmm025",
        "cartesian",
    ),
    "senior-graph-dilation-read-factor": (
        "ey_graph_dilation",
        "ey_graph_dilation_exam",
        "functions-sequences",
        "mcq",
        "acmmm026",
        "cartesian",
    ),
    "senior-graph-reciprocal-abs-compare": (
        "ey_graph_reciprocal_abs",
        "ey_graph_reciprocal_abs_exam",
        "functions-sequences",
        "mcq",
        "acmsm099",
        "cartesian",
    ),
    "senior-graph-slope-intercept-meaning": (
        "ey_context_line",
        "ey_context_line_exam",
        "linear-relations",
        "mcq",
        "acmgm042",
        "cartesian",
    ),
    "senior-position-time-tangent-velocity": (
        "ey_position_tangent",
        "ey_position_tangent_exam",
        "senior-mathematics",
        "number",
        "acmmm094",
        "cartesian",
    ),
    "senior-simulated-samples-variability": (
        "ey_sample_variability",
        "ey_sample_variability_exam",
        "statistics-probability",
        "mcq",
        "acmmm173",
        "histogram",
    ),
    "senior-describe-spread-from-a-dot-plot": (
        "ey_dot_plot_spread",
        "ey_dot_plot_spread_exam",
        "statistics-probability",
        "mcq",
        "acmem054",
        "dot_plot",
    ),
}


@functools.lru_cache(maxsize=1)
def _skills():
    _domains, skills = load_skills()
    return {s.id: s for s in skills}


@functools.lru_cache(maxsize=None)
def _q(skill_id, tier, seed):
    spec = next(s for s in _skills()[skill_id].sources if s.difficulty == tier)
    return make(spec.topic, seed=seed, difficulty=tier, **(spec.params or {}))


def _is_right(skill_id, q, text):
    kind = SKILLS[skill_id][3]
    if kind == "mcq" and not getattr(q, "parts", None):
        return grade("mcq", q, mcq_options(q).index(text)).correct
    return grade(kind, q, text).correct


def _parts_reply(q, replace=None):
    """The full labelled reply "(a) ..; (b) .." with one part's text swapped for ``replace=(label, text)``."""
    bits = []
    for p in q.parts:
        text = replace[1] if replace and replace[0] == p.label else p.answer_text
        bits.append(f"({p.label}) {text}")
    return "; ".join(bits)


# ------------------------------------------------------------------ the skill files


@pytest.mark.parametrize("skill_id", sorted(SKILLS))
def test_skill_is_senior_figure_skill_with_four_tiers(skill_id):
    topic, exam_topic, domain, kind, code, _fig = SKILLS[skill_id]
    skill = _skills()[skill_id]
    assert skill.status == "active" and skill.deferred_reason is None
    assert skill.needs_figure is True and skill.capability_gates == ["figure_stimulus"]
    assert load_capability_gates()["figure_stimulus"]["status"] == "passed"
    assert skill.domain == domain and skill.answer == kind and skill.placement_eligible is False
    assert [s.difficulty for s in skill.sources] == TIERS
    assert [s.topic for s in skill.sources] == [topic, topic, topic, exam_topic]
    assert [list(s.stages) for s in skill.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert code in skill.tags and "figure" in skill.tags
    assert len(skill.misconceptions) >= 6 and all(m.description for m in skill.misconceptions)
    assert set(skill.difficulty_progression) == set(TIERS)
    # a senior skill is not a read-aloud early-years skill
    assert not skill.read_aloud and getattr(skill, "learner_band", None) != "early"


# ------------------------------------------------------------------ figures and readers


@pytest.mark.parametrize("skill_id", sorted(SKILLS))
def test_every_tier_draws_a_valid_existing_figure_kind_that_does_not_show_the_key(skill_id):
    for tier in TIERS:
        for seed in SEEDS:
            q = _q(skill_id, tier, seed)
            assert q.figure is not None and figures.validate(q.figure) == [], (tier, seed)
            assert q.figure["kind"] == SKILLS[skill_id][5]
            texts = [p.answer_text for p in q.parts] if q.parts else [q.answer_text]
            figures.assert_no_leak(q.figure, texts)


# The exam chains of the mcq skills have no registered reader (the sweep hands a reader's text to the mcq path, which a
# labelled multi-part reply is not), so their readers are called here directly.
_EXAM_READERS = {
    "ey_graph_translation_exam": tf._read_translation_exam,
    "ey_graph_dilation_exam": tf._read_dilation_exam,
    "ey_graph_reciprocal_abs_exam": tf._read_recip_exam,
    "ey_context_line_exam": ctx._read_context_exam,
    "ey_position_tangent_exam": ctx._read_tangent_exam,
    "ey_sample_variability_exam": spread._read_samples_exam,
    "ey_dot_plot_spread_exam": spread._read_dots_exam,
}


@pytest.mark.parametrize("skill_id", sorted(SKILLS))
def test_registered_reader_re_derives_the_key(skill_id):
    topic, exam_topic = SKILLS[skill_id][0], SKILLS[skill_id][1]
    for tier in TIERS:
        reader = _EXAM_READERS[exam_topic] if tier == "exam" else figures.get_reader(topic)
        assert reader is not None
        for seed in SEEDS:
            q = _q(skill_id, tier, seed)
            assert reader(q) == q.answer_text, (tier, seed)


@pytest.mark.parametrize("skill_id", sorted(SKILLS))
def test_every_graph_value_read_sits_on_the_grid(skill_id):
    """Marked points and line anchors are whole numbers (or sit on a labelled tick); a tangent point is a lattice point."""
    for tier in TIERS:
        for seed in SEEDS:
            fig = _q(skill_id, tier, seed).figure
            if fig["kind"] == "cartesian":
                for p in fig["points"]:
                    assert all(float(v) == int(v) for v in p["at"]), (tier, seed, p)
                for ln in fig["lines"]:
                    assert all(float(v) == int(v) for pt in ln["through"] for v in pt), (tier, seed)
            elif fig["kind"] == "dot_plot":
                low, step = fig["range"][0], fig["step"]
                assert all((d["at"] - low) % step == 0 for d in fig["dots"])


# ------------------------------------------------------------------ grading


@pytest.mark.parametrize("skill_id", sorted(SKILLS))
def test_real_grader_accepts_the_key_and_rejects_every_distractor(skill_id):
    for tier in TIERS:
        for seed in SEEDS:
            q = _q(skill_id, tier, seed)
            if q.parts:
                assert _is_right(skill_id, q, q.answer_text), (tier, seed)
                assert _is_right(skill_id, q, _parts_reply(q)), (tier, seed)
                for p in q.parts:
                    assert p.distractors and p.answer_text not in p.distractors
                    for d in p.distractors:
                        reply = _parts_reply(q, replace=(p.label, d))
                        assert not _is_right(skill_id, q, reply), (tier, seed, p.label, d)
                continue
            assert _is_right(skill_id, q, q.answer_text), (tier, seed)
            assert q.distractors and q.answer_text not in q.distractors
            for d in q.distractors:
                assert not _is_right(skill_id, q, d), (tier, seed, d, q.answer_text)


@pytest.mark.parametrize("skill_id", sorted(SKILLS))
def test_exam_tier_is_a_follow_through_chain(skill_id):
    q = _q(skill_id, "exam", 0)
    assert len(q.parts) >= 4
    chained = [p for p in q.parts if p.follow_through]
    assert len(chained) >= 2, "an exam tier must chain: later parts follow through from earlier ones"
    for p in chained:
        assert any(f"part_{label}" in p.follow_through for label in "abcde")
    # a learner whose early part is wrong but whose later parts follow from it still earns those marks
    first = q.parts[0]
    wrong_first = first.distractors[0]
    result = grade(SKILLS[skill_id][3], q, _parts_reply(q, replace=(first.label, wrong_first)))
    assert not result.correct and result.marks < result.max_marks


@pytest.mark.parametrize("skill_id", sorted(SKILLS))
@pytest.mark.parametrize("family", sorted(skill_contract.FAMILIES))
def test_skill_contract(skill_id, family):
    skill_contract.FAMILIES[family](skill_id)


# ------------------------------------------------------------------ the misconceptions each distractor carries


def test_translation_distractors_include_the_sign_traps():
    for seed in SEEDS:
        q = _q("senior-graph-translation-read-shift", "medium", seed)
        assert str(-int(q.answer_text)) in q.distractors  # f(x + b) moves the graph the other way
        q = _q("senior-graph-translation-read-shift", "easy", seed)
        assert str(-int(q.answer_text)) in q.distractors


def test_dilation_distractors_include_the_inverse_factor():
    for seed in SEEDS:
        q = _q("senior-graph-dilation-read-factor", "medium", seed)
        k = tf.F(q.answer_text)
        assert tf.frac_text(1 / k) in q.distractors  # f(kx) divides x by k, so 1/k is the trap


# ------------------------------------------------------------------ bad figures are refused


def _fig_q(fig, text):
    return type("Q", (), {"figure": fig, "text": text})


def test_translation_reader_refuses_graphs_that_are_not_a_translation():
    q = _q("senior-graph-translation-read-shift", "easy", 0)
    fig = {**q.figure, "curves": [dict(c) for c in q.figure["curves"]]}
    fig["curves"][1] = tf.sample(lambda x: 0.5 * x * x - 3, label="g")
    with pytest.raises(VerificationError):
        tf._read_translation(_fig_q(fig, q.text))


def test_dilation_reader_refuses_a_graph_that_is_no_dilation():
    q = _q("senior-graph-dilation-read-factor", "medium", 0)
    fig = {**q.figure, "curves": [dict(c) for c in q.figure["curves"]]}
    fig["curves"][1] = tf.sample(lambda x: abs(x) + 3, label="g")
    with pytest.raises(VerificationError):
        tf._read_dilation(_fig_q(fig, q.text))


def test_context_reader_refuses_a_marked_point_off_the_line():
    q = _q("senior-graph-slope-intercept-meaning", "easy", 0)
    pts = [dict(p) for p in q.figure["points"]]
    pts[0]["at"] = [pts[0]["at"][0], pts[0]["at"][1] + 1]
    with pytest.raises(VerificationError):
        ctx._read_context(_fig_q({**q.figure, "points": pts}, q.text))


def test_tangent_reader_refuses_a_line_that_is_not_the_tangent():
    q = _q("senior-position-time-tangent-velocity", "easy", 0)
    (x0, y0), (x1, y1) = q.figure["lines"][0]["through"]
    bent = {"through": [[x0, y0], [x1, y0 + (y1 - y0) * 2 + 1]], "label": "tangent"}
    with pytest.raises(VerificationError):
        ctx._read_tangent(_fig_q({**q.figure, "lines": [bent]}, q.text))


def test_dot_plot_classifier_refuses_three_clusters_and_in_between_spreads():
    with pytest.raises(AssertionError):
        spread._classify({0: 2, 6: 2, 12: 2})
    with pytest.raises(VerificationError):
        spread._classify({0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1})  # 8 steps wide: neither tight nor wide


def test_histogram_reader_refuses_two_samples_that_are_equally_spread():
    q = _q("senior-simulated-samples-variability", "easy", 0)
    panels = [dict(p) for p in q.figure["panels"]]
    panels[1] = {**panels[1], "frequencies": list(panels[0]["frequencies"])}
    with pytest.raises(VerificationError):
        spread._read_samples(_fig_q({**q.figure, "panels": panels}, q.text))


def test_reciprocal_reader_refuses_a_graph_that_matches_no_option():
    q = _q("senior-graph-reciprocal-abs-compare", "easy", 0)
    fig = {**q.figure, "curves": [dict(c) for c in q.figure["curves"]]}
    fig["curves"][1] = tf.sample(lambda x: x * x - 1, label="g")
    with pytest.raises(VerificationError):
        tf._read_recip(_fig_q(fig, q.text))
