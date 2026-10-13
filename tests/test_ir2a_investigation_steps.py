"""The ir2a statistical-investigation skills: wiring, exam chains, option counts and two independent oracles."""

from __future__ import annotations

import re
import statistics

import pytest

from mathgen.engine import grade, make_for_skill
from tests import exd_x_r2j_kit as kit

SKILLS = {
    "choose-sampling-method": "choose_sampling_method",
    "spot-bias": "spot_bias",
    "question-to-data-needed": "question_to_data_needed",
    "pick-display-for-question": "pick_display_for_question",
    "investigation-step-order": "investigation_step_order",
    "judge-stated-conclusion": "judge_stated_conclusion",
    "compare-groups-evidence": "compare_groups_evidence",
    "limits-of-a-bivariate-claim": "limits_of_a_bivariate_claim",
    "misleading-graph-figure": "misleading_graph_figure",
}
TIERS = ("easy", "medium", "hard")


@pytest.mark.parametrize("skill_id", SKILLS)
def test_exam_chain(skill_id):
    kit.check_wiring(skill_id, f"exd_x_ir2a_{SKILLS[skill_id]}")
    kit.check_no_defects(skill_id)
    kit.check_determinism(skill_id)
    kit.check_answer_space(skill_id, minimum=100)
    kit.check_grading(skill_id, n=60)
    for seed in range(40):
        q = kit.exam_q(skill_id, seed)
        assert 3 <= len(q.parts) <= 5
        assert any(p.follow_through for p in q.parts), q.text


@pytest.mark.parametrize("skill_id", SKILLS)
def test_tiers_use_the_matching_topic(skill_id):
    sk = kit.skill(skill_id)
    assert [s.difficulty for s in sk.sources] == ["easy", "medium", "hard", "exam"]
    assert [s.topic for s in sk.sources[:3]] == [f"ir2a_{SKILLS[skill_id]}"] * 3


@pytest.mark.parametrize("skill_id", SKILLS)
@pytest.mark.parametrize("tier", TIERS)
def test_multiple_choice_is_not_guessable(skill_id, tier):
    """Every choice part offers at least four options and is paired with a typed or list part in the same question."""
    sk = kit.skill(skill_id)
    for seed in range(25):
        q = make_for_skill(sk, seed, source=tier)
        for p in q.parts or [q]:
            if p.answer_kind == "mcq":
                assert len(p.distractors) >= 3, (seed, q.text)
                assert len(set(p.distractors) | {p.answer_text}) == len(p.distractors) + 1
        given = {p.label: p.answer_text for p in q.parts} if q.parts else q.answer_text
        assert grade(sk, q, given).correct, (seed, q.text)


def _iqr(vals: list[int]) -> float:
    vals = sorted(vals)
    n = len(vals)
    low, high = vals[: n // 2], vals[(n + 1) // 2 :]
    return statistics.median(high) - statistics.median(low)


def test_compare_groups_exam_matches_an_independent_calculation():
    skill_id = "compare-groups-evidence"
    for seed in range(120):
        q = kit.exam_q(skill_id, seed)
        groups = re.findall(r"\((\d+) values\): ([\d, ]+)\.", q.stem)
        a, b = ([int(x) for x in g[1].split(", ")] for g in groups)
        canon = kit.canon(q)
        d = abs(statistics.median(a) - statistics.median(b))
        assert float(canon["a"]) == d, q.text
        assert float(canon["b"]) == _iqr(a), q.text
        assert float(canon["c"]) == _iqr(b), q.text
        strong = d >= max(_iqr(a), _iqr(b))
        assert float(canon["d"]) == max(_iqr(a), _iqr(b)) - d, q.text
        assert canon["e"].startswith("Strong" if strong else "Weak"), (seed, canon["e"])


def test_judge_conclusion_exam_percentages_match_the_stem():
    skill_id = "judge-stated-conclusion"
    seen = 0
    for seed in range(120):
        q = kit.exam_q(skill_id, seed)
        m = re.search(r"followed (\d+) .*?\. (\d+) .*?, and (\d+) of them .*?\. (\d+) .*?, and (\d+) of them", q.stem)
        if not m:
            continue
        n1, k1, n2, k2 = int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
        canon = kit.canon(q)
        assert float(canon["a"]) == pytest.approx(100 * k1 / n1), q.text
        assert float(canon["b"]) == pytest.approx(100 * k2 / n2), q.text
        seen += 1
    assert seen >= 40


def test_misleading_graph_percentage_follows_the_two_readings():
    skill_id = "misleading-graph-figure"
    for seed in range(120):
        q = kit.exam_q(skill_id, seed)
        canon = kit.canon(q)
        first, last = float(canon["a"]), float(canon["b"])
        assert float(canon["c"]) == pytest.approx(round((last - first) / first * 100, 1)), q.text
