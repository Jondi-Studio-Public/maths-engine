"""Normal distribution probabilities and quantiles.

Tests for senior-calculate-probabilities-and-quantiles-associated-with-a-given
(ACMMM170, topic p13d_normal_dist_prob_quantile).
"""

import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import stated_precision_tolerance

TOPIC = "p13d_normal_dist_prob_quantile"
SKILL_ID = "senior-calculate-probabilities-and-quantiles-associated-with-a-given"
DIFFICULTIES = ["easy", "medium", "hard", "exam"]
MODES = {"easy": "one_tail", "medium": "between", "hard": "quantile", "exam": "solve_parameter"}
DP_BY_TIER = {"easy": 4, "medium": 4, "hard": 1, "exam": 2}
SEEDS = range(40)

# --- Per-scenario plausible mu/sigma ranges, keyed
# by the scenario's own unit -- units are unique per scenario, so the unit
# stated in the rendered text identifies which scenario was drawn without
# importing anything from the generator. (mu_lo, mu_hi, sigma_lo, sigma_hi).
_SCENARIO_RANGES = {
    "g": (300, 900, 5, 25),
    "cm": (165, 185, 6, 9),
    "hours": (8, 20, 1, 3),
    "ms": (150, 300, 15, 40),
    "mm": (5, 30, 0.1, 1.0),
}


@pytest.fixture(scope="module")
def skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skills_dict, difficulty, seed):
    """Generate through the skill's own `sources` entry, params-driven, not a
    bare `make(topic, seed, difficulty)` call."""
    skill = skills_dict[SKILL_ID]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# --- Determinism -------------------------------------------------------------

# --- Distractor hygiene --------------------------------------------------

# --- Distractor gap exceeds the stated-precision tolerance, every seed -----


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_distractor_gap_exceeds_stated_precision_tolerance(skills_dict, difficulty):
    tol = stated_precision_tolerance(DP_BY_TIER[difficulty])
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        values = [Fraction(q.answer_text)] + [Fraction(d) for d in q.distractors]
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                gap = abs(values[i] - values[j])
                assert gap > tol, (difficulty, seed, q.answer_text, q.distractors, gap)


# --- Real grader accept/reject -----------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_real_grader_rejects_a_deliberately_wrong_answer(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        step = float(stated_precision_tolerance(DP_BY_TIER[difficulty])) * 4
        wrong = str(round(float(q.answer) + step, DP_BY_TIER[difficulty]))
        assert not grade_number(q, wrong).correct, (difficulty, seed, q.text, q.answer_text, wrong)


# --- Answer states exactly its tier's own precision -------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_answer_and_distractors_state_exact_precision(skills_dict, difficulty):
    dp = DP_BY_TIER[difficulty]
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        for text in [q.answer_text] + list(q.distractors):
            assert "." in text, (difficulty, seed, text)
            assert len(text.split(".")[-1].lstrip("-")) == dp, (difficulty, seed, text)


# --- Instruction phrase present in both text and latex ---------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_instruction_phrase_in_both_text_and_latex(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        assert "decimal place" in q.text.lower(), (difficulty, seed, q.text)
        assert "decimal place" in q.latex.lower(), (difficulty, seed, q.latex)


# --- Every input a learner needs (mu, sigma, and the tier's own extra
# quantity) is stated, never assumed ------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_mean_and_standard_deviation_both_stated(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        assert "mean" in q.text.lower(), (difficulty, seed, q.text)
        assert "standard deviation" in q.text.lower(), (difficulty, seed, q.text)


# --- At least 20 distinct rendered questions per tier over 60 seeds ------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_at_least_20_distinct_questions_per_tier(difficulty):
    mode = MODES[difficulty]
    seen = {make(TOPIC, seed=s, difficulty=difficulty, mode=mode).text for s in range(60)}
    assert len(seen) >= 20, (difficulty, mode, len(seen))


# --- ValueError on a nonsensical preset -----------------------------------


def test_raises_value_error_on_unknown_mode():
    import pytest as _pytest

    with _pytest.raises(ValueError):
        make(TOPIC, seed=1, mode="bogus")


# --- The skill's own YAML wires 4 sources, one per tier, exam under
# stages: [exam] ------------------------------------------------------------


def test_skill_sources_and_stages(skills_dict):
    skill = skills_dict[SKILL_ID]
    assert skill.answer == "number"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["simple-event-probability"]
    by_difficulty = {s.difficulty: s for s in skill.sources}
    assert set(by_difficulty) == {"easy", "medium", "hard", "exam"}
    assert by_difficulty["exam"].stages == ["exam"]
    assert by_difficulty["exam"].stages != []


# --- Plausibility: hard-tier quantile and exam-tier solved parameter land
# in a sane range, measured not assumed --------------------------------

# Original bound: `ok = sum(1 for x in xs if -100 < x < 3000)`. That was pointed
# at a single global constant range wide enough to admit a bolt as big as a
# house (4281 mm) alongside a plausible person's height -- it could never
# catch a scenario-specific impossibility because it had no notion of
# "scenario" at all. The replacement below is pointed at each drawn
# scenario's OWN declared range (identified by its stated unit, `_SCENARIO_RANGES`
# above -- the same table the swept test below uses), so a bolt quantile miles
# outside bolt-plausible mm is caught even though a person-height quantile in
# the same numeric neighbourhood would not be.
_HARD_STMT_RE = re.compile(
    r"mean (?P<mu>-?\d+(?:\.\d+)?) (?P<unit>\S+) and standard deviation (?P<sigma>-?\d+(?:\.\d+)?) \S+"
)


def test_hard_quantile_bounds_plausible():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="hard", mode="quantile")
        m = _HARD_STMT_RE.search(q.text)
        assert m, (seed, q.text)
        unit = m["unit"]
        mu_lo, mu_hi, sigma_lo, sigma_hi = _SCENARIO_RANGES[unit]
        x = float(q.answer)
        lo, hi = mu_lo - 5 * sigma_hi, mu_hi + 5 * sigma_hi
        assert lo <= x <= hi, (seed, unit, x, lo, hi, q.text)


# Each scenario draws its OWN plausible mu/sigma range (not one shared 50-900 draw for every unit),
# 200 seeds per tier -----


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_mu_and_sigma_stay_in_scenario_range(difficulty):
    mode = MODES[difficulty]
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=difficulty, mode=mode)
        m = _HARD_STMT_RE.search(q.text)
        assert m, (difficulty, seed, q.text)
        mu, sigma, unit = float(m["mu"]), float(m["sigma"]), m["unit"]
        assert unit in _SCENARIO_RANGES, (difficulty, seed, unit, q.text)
        mu_lo, mu_hi, sigma_lo, sigma_hi = _SCENARIO_RANGES[unit]
        assert mu_lo <= mu <= mu_hi, (difficulty, seed, unit, mu, q.text)
        assert sigma_lo <= sigma <= sigma_hi, (difficulty, seed, unit, sigma, q.text)


_EXAM_STMT_RE = re.compile(
    r"unknown (?P<unknown>mean|standard deviation) and (?P<known_label>mean|standard deviation) "
    r"(?P<known_val>-?\d+(?:\.\d+)?) (?P<unit>\w+)\."
)


def test_mu_and_sigma_stay_in_scenario_range_exam():
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="exam", mode="solve_parameter")
        m = _EXAM_STMT_RE.search(q.text)
        assert m, (seed, q.text)
        known_val, unit = float(m["known_val"]), m["unit"]
        assert unit in _SCENARIO_RANGES, (seed, unit, q.text)
        mu_lo, mu_hi, sigma_lo, sigma_hi = _SCENARIO_RANGES[unit]
        if m["known_label"] == "mean":
            assert mu_lo <= known_val <= mu_hi, (seed, unit, known_val, q.text)
        else:
            assert sigma_lo <= known_val <= sigma_hi, (seed, unit, known_val, q.text)
        solved = float(q.answer)
        if m["unknown"] == "mean":
            lo, hi = mu_lo - 5 * sigma_hi, mu_hi + 5 * sigma_hi
            assert lo <= solved <= hi, (seed, unit, solved, lo, hi, q.text)
        else:
            assert 0 < solved < sigma_hi * 3, (seed, unit, solved, q.text)


def test_exam_solved_parameter_nonzero():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="exam", mode="solve_parameter")
        assert q.answer != 0, (seed, q.text)


# --- Every question is a CONTINUOUS normal distribution problem: never a
# discrete pmf/Bernoulli word --

_FORBIDDEN_WORDS = ["bernoulli", "success", "trial", "binomial", "discrete"]


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_never_a_discrete_distribution_question(skills_dict, difficulty):
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        lower = q.text.lower()
        for word in _FORBIDDEN_WORDS:
            assert word not in lower, (difficulty, seed, word, q.text)
