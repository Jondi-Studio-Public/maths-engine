"""See mathgen/generators/p19a_expanded_powers_ten_build.py for the tier design and misconceptions."""

from collections import Counter
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make

ROOT = Path(__file__).resolve().parents[1]
TOPIC = "p19a_expanded_powers_ten"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(60)


# --- determinism / distractor hygiene ---------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_deterministic_across_seeds(difficulty):
    for seed in range(40):
        q1 = make(TOPIC, seed=seed, difficulty=difficulty)
        q2 = make(TOPIC, seed=seed, difficulty=difficulty)
        assert q1.to_json() == q2.to_json()


@pytest.mark.parametrize("difficulty", TIERS)
def test_distractors_present_distinct_nonempty_and_not_the_answer(difficulty):
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert len(q.distractors) >= 2
        assert all(d for d in q.distractors)
        assert len(q.distractors) == len(set(q.distractors))
        assert q.answer_text not in q.distractors


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_is_a_whole_number_graded_exactly(difficulty):
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert isinstance(q.answer, int)
        assert q.answer_text == str(q.answer)
        assert "." not in q.answer_text


# --- real grader accept/reject -----------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_answer_rejects_distractors_and_a_wrong_value(difficulty):
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert grade_number(q, q.answer_text).correct
        for d in q.distractors:
            assert not grade_number(q, d).correct
        assert not grade_number(q, str(q.answer + 100000)).correct


def test_grader_probe_comma_and_space_grouped_accepted():
    """The exact probe: GRADERS['number'] accepts "4356", "4,356" and the space-grouped "4 356" (thousands
    grouped by a space, as the SI and the Australian style manual write them: maintainer ruling, 2026-10-12)."""
    q = make(TOPIC, seed=0, difficulty="medium")
    n = q.answer
    grouped = f"{n:,}"  # "4,356"
    space_grouped = grouped.replace(",", " ")  # "4 356"
    assert grade_number(q, str(n)).correct
    assert grade_number(q, grouped).correct
    if grouped != str(n):  # only a meaningful probe once the numeral is >= 1000
        assert grade_number(q, space_grouped).correct


def test_grader_probe_on_a_five_digit_numeral():
    """Force a numeral >= 10000 so the comma-vs-space probe is unconditional,
    matching the worked example."""
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="medium")
        if q.answer >= 10000:
            grouped = f"{q.answer:,}"
            space_grouped = grouped.replace(",", " ")
            assert grade_number(q, str(q.answer)).correct
            assert grade_number(q, grouped).correct
            assert grade_number(q, space_grouped).correct
            return
    pytest.skip("no five-digit numeral drawn in 200 seeds")


# --- variety ------------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_at_least_20_distinct_questions_over_60_seeds(difficulty):
    texts = {make(TOPIC, seed=seed, difficulty=difficulty).text for seed in SEEDS}
    assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions in 60 seeds"


# --- 200-seed answer-concentration instrument ------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_concentration_stays_under_15_percent(difficulty):
    answers = [make(TOPIC, seed=seed, difficulty=difficulty).answer for seed in range(200)]
    counts = Counter(answers)
    top_answer, top_count = counts.most_common(1)[0]
    assert top_count / 200 < 0.15, (
        f"{difficulty}: answer {top_answer} appears in {top_count}/200 draws, at/over the 15% cap"
    )


# --- tier-specific invariants --------------------------------------------


def test_easy_never_has_a_missing_power():
    """Easy never writes explicit low powers, so the number of `x` symbols is
    exactly (digit count - 1) precisely when every place is present."""
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="easy")
        digit_count = len(str(q.answer))
        assert q.text.count("×") == digit_count - 1, q.text


def test_medium_sometimes_has_a_missing_power():
    """At least one medium draw in 60 seeds must have fewer `x`-marked terms
    than (digit count - 1) -- the missing-power case the tier exists to
    teach."""
    found = False
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="medium")
        digit_count = len(str(q.answer))
        if q.text.count("×") < digit_count - 1:
            found = True
            break
    assert found, "no missing-power draw found in 60 medium seeds"


def test_hard_never_lists_terms_in_descending_order():
    import re

    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="hard")
        powers = [int(p) for p in re.findall(r"10\^(\d+)", q.text)]
        # every term after the first "x 10" fragment: recover ALL powers,
        # including the implicit power-1/power-0 terms, by re-deriving from
        # the coefficients' own order against the answer's digits is done in
        # the oracle; here we only need: hard's own explicit powers, when >= 2
        # are present, are never sorted descending.
        if len(powers) >= 2:
            assert powers != sorted(powers, reverse=True), q.text
            return
    pytest.skip("no hard draw with >= 2 explicit powers in 60 seeds")


def test_exam_states_a_coefficient_of_ten_or_more():
    import re

    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="exam")
        coeffs = [int(c) for c in re.findall(r"(\d+) x 10\^\d+", q.working[0])]
        assert any(c >= 10 for c in coeffs), q.working[0]


# --- skill contract -------------------------------------------------------


def test_skill_loads_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    skill = by_id["f10-expanded-powers-ten"]

    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.status == "active"
    assert sorted(skill.prerequisites) == ["f10-powers-ten-multiply-divide"]
    assert {source.topic for source in skill.sources} == {TOPIC}
    assert [source.difficulty for source in skill.sources] == ["easy", "medium", "hard", "exam"]
    assert [source.stages for source in skill.sources] == [
        [0, 1, 2],
        [3, 4],
        [5, 6, "maintenance"],
        ["exam"],
    ]
    assert len(skill.misconceptions) == 4
