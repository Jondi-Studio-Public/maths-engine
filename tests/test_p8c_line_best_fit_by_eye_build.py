"""Line of best fit by eye (senior-find-the-line-of-best-fit-by-eye)."""

import re
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.generators.p8c_line_best_fit_by_eye_build import _ssr, _sad

ROOT = Path(__file__).resolve().parents[1]
TOPIC = "p8c_line_best_fit_by_eye"
SEEDS = range(40)
TIERS = ["easy", "medium", "hard"]

_EQ_RE = re.compile(r"^y = (-?\d+)x ([+-]) (\d+)$")


def _parse_eq(eq):
    m = _EQ_RE.match(eq)
    assert m, f"equation {eq!r} does not match the uniform shape"
    coef = int(m.group(1))
    b = int(m.group(3)) * (1 if m.group(2) == "+" else -1)
    return coef, b


@pytest.mark.parametrize("difficulty", TIERS)
def test_deterministic_across_seeds(difficulty):
    for seed in SEEDS:
        q1 = make(TOPIC, seed=seed, difficulty=difficulty)
        q2 = make(TOPIC, seed=seed, difficulty=difficulty)
        assert q1.to_json() == q2.to_json()


@pytest.mark.parametrize("difficulty", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert len(q.distractors) == 3
        assert len(set(q.distractors)) == 3
        assert q.answer_text not in q.distractors


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_mcq_grader_accepts_answer_rejects_distractors(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        options = mcq_options(q)
        correct_idx = options.index(q.answer_text)
        assert grade_mcq(q, correct_idx).correct
        for i, option in enumerate(options):
            if option != q.answer_text:
                assert not grade_mcq(q, i).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_deliberately_wrong_answer_rejected(difficulty):
    for seed in SEEDS[:10]:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        options = mcq_options(q)
        bogus_idx = len(options) + 5  # out of range -> never correct
        assert not grade_mcq(q, bogus_idx).correct
        assert not grade_mcq(q, "not-a-number").correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_every_option_uniform_shape_and_answer_matches_smallest_ssr_and_sad(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        pts = [tuple(p["at"]) for p in q.figure["points"]]
        assert len(pts) == 5
        eqs_in_text = re.findall(r"y = -?\d+x [+-] \d+", q.text)
        assert len(eqs_in_text) == 4
        options = [q.answer_text] + list(q.distractors)
        # every option is present verbatim in the question text
        for opt in options:
            assert opt in eqs_in_text
            _parse_eq(opt)  # uniform "y = <int>x +/- <int>" shape, never raises
        ssrs = {}
        sads = {}
        for eq in eqs_in_text:
            m, b = _parse_eq(eq)
            ssrs[eq] = _ssr(pts, m, b)
            sads[eq] = _sad(pts, m, b)
        winner_ssr = min(eqs_in_text, key=lambda e: ssrs[e])
        winner_sad = min(eqs_in_text, key=lambda e: sads[e])
        assert winner_ssr == winner_sad == q.answer_text
        # margin: relative 50% on ssr, absolute 4 on sad
        ssr_sorted = sorted(ssrs.values())
        sad_sorted = sorted(sads.values())
        assert 2 * ssr_sorted[1] >= 3 * ssr_sorted[0]
        assert sad_sorted[1] - sad_sorted[0] >= 4


@pytest.mark.parametrize("difficulty", TIERS)
def test_at_least_20_distinct_questions_over_60_seeds(difficulty):
    texts = {make(TOPIC, seed=s, difficulty=difficulty).text for s in range(60)}
    assert len(texts) >= 20


def test_ssr_algebraic_identity_and_sad_alt_loop_fire_on_corrupted_value():
    """The two independent-check routes are genuine identities: corrupting
    the direct-route value must break agreement with the independently
    recomputed one."""
    pts = [(1, 2), (2, 5), (3, 9), (4, 10), (5, 14)]
    m, b = 2, 1
    ssr_direct = _ssr(pts, m, b)
    n = len(pts)
    sum_x = sum(x for x, _ in pts)
    sum_y = sum(y for _, y in pts)
    sum_x2 = sum(x * x for x, _ in pts)
    sum_y2 = sum(y * y for _, y in pts)
    sum_xy = sum(x * y for x, y in pts)
    ssr_algebraic = sum_y2 - 2 * m * sum_xy - 2 * b * sum_y + m * m * sum_x2 + 2 * m * b * sum_x + b * b * n
    assert ssr_direct == ssr_algebraic
    corrupted = ssr_direct + 7
    assert corrupted != ssr_algebraic  # the check would fire on this

    sad_direct = _sad(pts, m, b)
    sad_alt = 0
    for x, y in pts:
        sad_alt += abs(y - (m * x + b))
    assert sad_direct == sad_alt
    corrupted_sad = sad_direct + 3
    assert corrupted_sad != sad_alt  # the check would fire on this


def test_skill_loads_with_expected_shape():
    _, skills = load_skills()
    by_id = {s.id: s for s in skills}
    skill = by_id["senior-find-the-line-of-best-fit-by-eye"]
    assert skill.prerequisites == ["simple-event-probability", "senior-identify-the-dependent-and-independent-variable"]
    assert skill.answer == "mcq"
    assert skill.placement_eligible is False
    assert {s.difficulty for s in skill.sources} == {"easy", "medium", "hard", "exam"}
