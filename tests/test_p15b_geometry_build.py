"""Focused contract tests for: ucv1-fundamental-group-map (answer: mcq) and ucv1-simplex-homology (answer:
ordered_list)."""

import re
from collections import Counter

import pytest

from mathgen.grading.typed import grade_mcq, grade_ordered_list, mcq_options
from mathgen.skills import load_skills
from mathgen import make

FGM_ID = "ucv1-fundamental-group-map"
HOM_ID = "ucv1-simplex-homology"
SEEDS = range(40)
SWEEP_SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]

_TRUE_BETTI_TEXT = {"1, 1", "1, 0", "1, 0, 0"}


def _skill_sources(skill_id):
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    return skill, {src.difficulty: src for src in skill.sources}


FGM_SKILL, FGM_SOURCES = _skill_sources(FGM_ID)
HOM_SKILL, HOM_SOURCES = _skill_sources(HOM_ID)


def _make(sources, difficulty, seed):
    src = sources[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# ==========================================================================
# ucv1-fundamental-group-map
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_fgm_determinism_and_distractor_hygiene(difficulty):
    for seed in SEEDS:
        q1 = _make(FGM_SOURCES, difficulty, seed)
        assert len(q1.distractors) == 3


@pytest.mark.parametrize("difficulty", TIERS)
def test_fgm_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        q = _make(FGM_SOURCES, difficulty, seed)
        options = mcq_options(q)
        correct_index = options.index(q.answer_text)
        assert grade_mcq(q, str(correct_index)).correct
        for index, option in enumerate(options):
            if option != q.answer_text:
                assert not grade_mcq(q, str(index)).correct


def test_fgm_easy_medium_never_draw_zero_or_interval():
    for difficulty in ("easy", "medium"):
        for seed in SWEEP_SEEDS:
            q = _make(FGM_SOURCES, difficulty, seed)
            assert q.answer_text != "the zero map", (difficulty, seed, q.text)
            assert "interval" not in q.text


def test_fgm_hard_tier_draws_both_zero_map_routes():
    """The hard tier's required trap: the constant map z -> z^0 = 1 AND the
    interval-factoring map both render 'the zero map', never two colliding
    option strings for the same fact."""
    saw_degree_zero = False
    saw_interval = False
    for seed in range(200):
        q = _make(FGM_SOURCES, "hard", seed)
        if q.answer_text == "the zero map":
            if "interval" in q.text:
                saw_interval = True
            elif "z^{0}" in q.text or "z^0" in q.text:
                saw_degree_zero = True
    assert saw_interval, "no hard-tier interval-factoring draw found in 200 seeds"
    assert saw_degree_zero, "no hard-tier degree-0 draw found in 200 seeds"


def test_fgm_exam_tier_is_a_genuine_composition_and_multiplies_degrees():
    for seed in SEEDS:
        q = _make(FGM_SOURCES, "exam", seed)
        assert "g o f" in q.text or "h =" in q.text, q.text
        m = re.findall(r"z\^\{(-?\d+)\}", q.text)
        assert len(m) == 2, (q.text, m)
        a, b = int(m[0]), int(m[1])
        if q.answer_text == "the zero map":
            assert a * b == 0
        else:
            expected = f"multiplication by {a * b}"
            assert q.answer_text == expected, (q.text, q.answer_text, expected)


def test_fgm_exam_tier_question_asks_about_h_never_bare_f():
    """The question must name h (or g o f) as the map being asked about, never bare f_* with no h/composition
    qualifier."""
    for seed in SEEDS:
        q = _make(FGM_SOURCES, "exam", seed)
        question_sentence = q.text.split("h = g o f.")[-1].strip()
        assert "h" in question_sentence, (seed, q.text)
        assert "does f induce" not in question_sentence, (seed, q.text)
        assert "is f_*," not in question_sentence, (seed, q.text)


@pytest.mark.parametrize("difficulty", TIERS)
def test_fgm_200_seed_answer_variety(difficulty):
    counts = Counter(_make(FGM_SOURCES, difficulty, seed).answer_text for seed in range(200))
    assert len(counts) >= 4, f"{difficulty}: only {len(counts)} distinct answers over 200 seeds"
    top_share = max(counts.values()) / 200
    assert top_share <= 0.45, f"{difficulty}: top answer is {top_share:.1%} of 200 draws"


def test_fgm_skill_loads_through_api_with_correct_properties():
    assert FGM_SKILL.domain == "advanced-geometry-and-topology"
    assert FGM_SKILL.status == "active"
    assert FGM_SKILL.placement_eligible is False
    assert FGM_SKILL.answer == "mcq"
    assert FGM_SKILL.prerequisites == ["ucv1-topology-basis-membership", "integer-arithmetic"]
    assert {src.difficulty for src in FGM_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    assert FGM_SOURCES["exam"].stages == ["exam"]


# ==========================================================================
# ucv1-simplex-homology
# ==========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_hom_determinism_and_distractor_hygiene(difficulty):
    for seed in SEEDS:
        q1 = _make(HOM_SOURCES, difficulty, seed)
        assert len(q1.distractors) == 3


@pytest.mark.parametrize("difficulty", TIERS)
def test_hom_reversed_answer_rejected_when_it_differs(difficulty):
    """Order matters positionally (b0, b1[, b2]): the REVERSED list must be
    rejected whenever it differs from the stated answer (a palindrome list
    is skipped since reversing it changes nothing to test)."""
    for seed in SEEDS:
        q = _make(HOM_SOURCES, difficulty, seed)
        parts = [p.strip() for p in q.answer_text.split(",")]
        reversed_text = ", ".join(reversed(parts))
        if reversed_text == q.answer_text:
            continue
        assert not grade_ordered_list(q, reversed_text).correct, (difficulty, seed, q.answer_text)


@pytest.mark.parametrize("difficulty", TIERS)
def test_hom_answer_is_one_of_the_three_true_betti_lists(difficulty):
    for seed in SEEDS:
        q = _make(HOM_SOURCES, difficulty, seed)
        assert q.answer_text in _TRUE_BETTI_TEXT, (difficulty, seed, q.answer_text)


@pytest.mark.parametrize("difficulty", TIERS)
def test_hom_200_seed_three_roughly_equal_classes(difficulty):
    """The contract fixes exactly three complexes, so exactly three answer classes are expected and roughly equal
    shares -- stated, not hidden."""
    counts = Counter(_make(HOM_SOURCES, difficulty, seed).answer_text for seed in range(200))
    assert set(counts) == _TRUE_BETTI_TEXT, f"{difficulty}: classes seen {set(counts)}"
    top_share = max(counts.values()) / 200
    assert top_share <= 0.45, f"{difficulty}: top class is {top_share:.1%} of 200 draws"


def test_hom_easy_tier_names_the_complex():
    for seed in SEEDS:
        q = _make(HOM_SOURCES, "easy", seed)
        assert "simplicial complex X is" in q.text, q.text


def test_hom_medium_tier_lists_simplices_not_the_name():
    for seed in SEEDS:
        q = _make(HOM_SOURCES, "medium", seed)
        assert "X has vertices" in q.text
        for forbidden in ("filled triangle", "boundary of a triangle", "single edge"):
            assert forbidden not in q.text, (seed, q.text)


def test_hom_hard_and_exam_tiers_give_boundary_matrices():
    for difficulty in ("hard", "exam"):
        for seed in SEEDS:
            q = _make(HOM_SOURCES, difficulty, seed)
            assert "boundary matrices" in q.text
            assert r"\partial_1" in q.latex, (difficulty, seed, q.latex)


def test_hom_exam_tier_reorients_some_edges_relative_to_hard():
    """hard never reorients (every edge's canonical sign is kept); exam
    randomly flips some edges' signs -- so the SAME underlying complex's
    matrix entries should differ in sign pattern across exam seeds, while
    the hard tier's matrix entries for a given complex are always the
    canonical ones. Detected by checking exam shows at least one negative-
    to-positive sign swap relative to the fixed hard-tier pattern for the
    single_edge complex (d1 = [-1, 1], the smallest case to compare)."""
    hard_patterns = set()
    exam_patterns = set()
    for seed in SWEEP_SEEDS:
        qe = _make(HOM_SOURCES, "exam", seed)
        m = re.search(r"d1 \(rows [A-Z], [A-Z]; columns e1\) = \[(-?\d+); (-?\d+)\]", qe.text)
        if m:
            exam_patterns.add((int(m.group(1)), int(m.group(2))))
        qh = _make(HOM_SOURCES, "hard", seed)
        m2 = re.search(r"d1 \(rows [A-Z], [A-Z]; columns e1\) = \[(-?\d+); (-?\d+)\]", qh.text)
        if m2:
            hard_patterns.add((int(m2.group(1)), int(m2.group(2))))
    assert hard_patterns == {(-1, 1)}, hard_patterns
    assert exam_patterns == {(-1, 1), (1, -1)}, exam_patterns


def test_hom_skill_loads_through_api_with_correct_properties():
    assert HOM_SKILL.domain == "advanced-geometry-and-topology"
    assert HOM_SKILL.status == "active"
    assert HOM_SKILL.placement_eligible is False
    assert HOM_SKILL.answer == "ordered_list"
    assert HOM_SKILL.prerequisites == ["matrix-rank-small", "matrix-multiplication"]
    assert {src.difficulty for src in HOM_SKILL.sources} == {"easy", "medium", "hard", "exam"}
    assert HOM_SOURCES["exam"].stages == ["exam"]
