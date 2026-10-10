"""Sample standard deviation from a small data set (standard-deviation-from-data, domain statistics-probability,
answer: number).

The single most important property tested here is that every distractor is
actually rejected by the real grader (`mathgen.grading.grade_number`) -- the
population standard deviation is a natural, dangerously close distractor for
a sample standard deviation question, and it is checked unconditionally,
never assumed "obviously" far enough away.

An independent oracle (`statistics.stdev`, Python's own stdlib SAMPLE
standard deviation) cross-checks every question's parsed data against the
generator's own answer, entirely outside the generator's own Fraction-based
arithmetic.
"""

import re
import statistics
from fractions import Fraction
from pathlib import Path

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
from mathgen.generators.sample_standard_deviation import _distinct_from

ROOT = Path(__file__).resolve().parents[1]
TOPIC = "standard_deviation_from_data"
SKILL_ID = "standard-deviation-from-data"
VARIANTS = ["integers", "non_integer_mean", "context", "compare_sets"]
SEEDS = range(40)

# Typed-number grading tolerance for a "to 2 decimal places" answer: half a
# unit in the last stated decimal place (mathgen/grading/typed.py's
# `_typed_number_tolerance(2)`).
_TOL = Fraction(1, 200)

_SINGLE_DATA_RE = re.compile(r": ([0-9.,\s]+)\. Treating")
_LABELLED_DATA_RE = re.compile(r"(Line A|Line B|Team 1|Team 2|Morning shift|Afternoon shift): ([0-9.,\s]+)\.")
_TARGET_LABEL_RE = re.compile(r"Treating (.+?)'s data as a sample")


def _parse_values(tok_str: str) -> list[float]:
    return [float(tok.strip()) for tok in tok_str.split(",")]


def _oracle_sample_sd(question) -> float:
    """Recompute the sample standard deviation independently of the
    generator, straight from the numbers printed in `question.text`, using
    Python's own `statistics.stdev` (sample, n-1) -- a genuinely different
    implementation from the generator's Fraction-based one."""
    if "Treating" in question.text and _TARGET_LABEL_RE.search(question.text):
        target = _TARGET_LABEL_RE.search(question.text).group(1)
        groups = dict(_LABELLED_DATA_RE.findall(question.text))
        assert target in groups, f"target label {target!r} not found in {question.text!r}"
        values = _parse_values(groups[target])
    else:
        m = _SINGLE_DATA_RE.search(question.text)
        assert m, f"could not find a data list in {question.text!r}"
        values = _parse_values(m.group(1))
    return statistics.stdev(values)


# --- Determinism --------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_generator_is_deterministic_across_seeds(variant):
    for seed in SEEDS:
        q1 = make(TOPIC, seed=seed, variant=variant)
        q2 = make(TOPIC, seed=seed, variant=variant)
        assert q1.to_json() == q2.to_json()


# --- The oracle: independent recomputation from the rendered text -------


@pytest.mark.parametrize("variant", VARIANTS)
def test_oracle_matches_generators_answer(variant):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, variant=variant)
        oracle = _oracle_sample_sd(q)
        answer = float(q.answer)
        assert abs(oracle - answer) <= float(_TOL), (
            f"{variant} seed={seed}: oracle sample sd {oracle} disagrees with "
            f"generator answer {answer} (text={q.text!r})"
        )


# --- Distractor hygiene: every distractor must clear the grader ----------


@pytest.mark.parametrize("variant", VARIANTS)
def test_distractors_distinct_nonempty_and_clear_grader_tolerance(variant):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, variant=variant)
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))
        assert len(q.distractors) >= 2
        answer_value = Fraction(q.answer_text)
        for distractor in q.distractors:
            distractor_value = Fraction(distractor)
            gap = abs(distractor_value - answer_value)
            assert gap > _TOL, (
                f"{variant} seed={seed}: distractor {distractor!r} is within "
                f"grading tolerance of answer {q.answer_text!r}"
            )


@pytest.mark.parametrize("variant", VARIANTS)
def test_real_grader_accepts_canonical_rejects_every_distractor(variant):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, variant=variant)
        assert grade_number(q, q.answer_text).correct, (
            f"{variant} seed={seed}: canonical answer {q.answer_text!r} rejected"
        )
        for distractor in q.distractors:
            assert not grade_number(q, distractor).correct, (
                f"{variant} seed={seed}: distractor {distractor!r} was ACCEPTED by the real grader (text={q.text!r})"
            )


@pytest.mark.parametrize("variant", VARIANTS)
def test_real_grader_rejects_deliberately_wrong_answer(variant):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, variant=variant)
        assert not grade_number(q, "999.99").correct, f"{variant} seed={seed}: wrong answer accepted"


# --- Answer kind / precision ---------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_answer_is_a_plain_number_to_two_decimal_places(variant):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, variant=variant)
        assert isinstance(q.answer, float)
        # answer_text carries exactly one numeric literal, at exactly 2dp --
        # the single literal `_typed_number_target` needs to trust it as the
        # grading target/precision source.
        literals = re.findall(r"-?\d+\.\d+", q.answer_text)
        assert len(literals) == 1, f"{variant} seed={seed}: answer_text {q.answer_text!r}"
        assert len(literals[0].split(".")[1]) == 2


# --- Vocabulary: "sample", never left to inference ------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_question_always_says_sample_standard_deviation(variant):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, variant=variant)
        assert "sample standard deviation" in q.text
        assert "sample" in q.text
        assert "2 decimal places" in q.text
        # Never the word "population" anywhere -- this skill never builds a
        # population-SD question, so there is nothing for the learner to
        # disambiguate against in the prompt itself.
        assert "population" not in q.text.lower()


def test_context_variant_states_a_measurement_unit():
    units = ["millimetres", "grams", "seconds", "centimetres"]
    seen = set()
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, variant="context")
        for unit in units:
            if unit in q.text:
                seen.add(unit)
    assert seen, "context variant never mentioned a measurement unit over 40 seeds"


def test_compare_sets_variant_names_both_labels_and_the_target():
    label_pairs = [("Line A", "Line B"), ("Team 1", "Team 2"), ("Morning shift", "Afternoon shift")]
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, variant="compare_sets")
        assert any(a in q.text and b in q.text for a, b in label_pairs), (
            f"seed={seed}: {q.text!r} does not name a recognised label pair"
        )
        m = _TARGET_LABEL_RE.search(q.text)
        assert m, f"seed={seed}: no 'Treating <label>'s data' clause in {q.text!r}"
        assert f"standard deviation of {m.group(1)}'s values" in q.text


def test_compare_sets_wrong_data_set_distractor_present_most_of_the_time():
    """The other set's own sample sd is included among the candidates
    whenever it clears the grading tolerance; over 40 seeds it should show up
    plenty (never REQUIRED on every seed, since a resample can legitimately
    drop it when it collides -- but its absence on every seed would mean the
    hazard this skill exists to guard against was never actually exercised)."""
    hits = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, variant="compare_sets")
        target = _TARGET_LABEL_RE.search(q.text).group(1)
        groups = dict(_LABELLED_DATA_RE.findall(q.text))
        other_label = next(lbl for lbl in groups if lbl != target)
        other_values = _parse_values(groups[other_label])
        other_sd = statistics.stdev(other_values)
        if any(abs(Fraction(d) - Fraction(f"{other_sd:.2f}")) < Fraction(1, 100) for d in q.distractors):
            hits += 1
    assert hits >= 5, f"only {hits}/40 compare_sets questions offered the other set's sd as a distractor"


@pytest.mark.parametrize("variant", VARIANTS)
def test_population_sd_hazard_is_always_present_and_always_rejected(variant):
    """The #1 hazard: the population standard deviation (divide by
    n, not n-1) is the FIRST candidate built for every question (see
    `_misconceptions`) and `_distinct_from` only drops a draw, never drops
    just this one candidate out of several -- so unlike the compare_sets
    'other data set' distractor (which can legitimately collide and be
    dropped on some draws), the population sd should appear as a distractor
    on every single question, every seed, every variant. This turns "the
    guard exists" into "the hazard is exercised on every one of 160
    questions", and every occurrence is independently re-checked against the
    real grader."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, variant=variant)
        if "Treating" in q.text and _TARGET_LABEL_RE.search(q.text):
            target = _TARGET_LABEL_RE.search(q.text).group(1)
            values = _parse_values(dict(_LABELLED_DATA_RE.findall(q.text))[target])
        else:
            values = _parse_values(_SINGLE_DATA_RE.search(q.text).group(1))
        population_sd_text = f"{statistics.pstdev(values):.2f}"
        assert population_sd_text in q.distractors, (
            f"{variant} seed={seed}: population sd {population_sd_text!r} (the "
            f"named hazard) is not among the distractors {q.distractors!r}"
        )
        assert not grade_number(q, population_sd_text).correct, (
            f"{variant} seed={seed}: the real grader ACCEPTED the population sd "
            f"{population_sd_text!r} for a sample-sd question"
        )


def test_distinct_from_unit_rejects_a_close_collision():
    """Direct unit test on the guard itself, independent of any live draw:
    two values that would DISPLAY the same (or within half a unit in the
    last stated decimal place) at 2dp are correctly flagged as not distinct;
    values clearly apart are not. Values are chosen well clear of any
    round-half-to-even boundary, so the test result cannot depend on
    floating-point rounding direction."""
    assert _distinct_from(1.00, 1.004) is False  # both display as "1.00"
    assert _distinct_from(5.29, 5.294) is False  # 5.294 also rounds to "5.29"
    assert _distinct_from(1.00, 1.02) is True  # 0.02 apart, well clear of the 0.005 tolerance
    assert _distinct_from(5.29, 4.83) is True  # a real population-sd-vs-sample-sd gap


# --- Unknown variant fails loudly, never silently defaults ----------------


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        make(TOPIC, seed=0, variant="bogus")


# --- Distinctness of the draw space ---------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_at_least_20_distinct_questions_per_variant_over_60_seeds(variant):
    texts = {make(TOPIC, seed=seed, variant=variant).text for seed in range(60)}
    assert len(texts) >= 20, f"{variant}: only {len(texts)} distinct questions/60 seeds"


# --- Bare make(topic, seed) (the default variant) ---------------------


def test_bare_make_uses_a_valid_default_variant_and_is_robust():
    for seed in range(200):
        q = make(TOPIC, seed=seed)
        assert q.text and q.answer_text
        assert "sample standard deviation" in q.text


# --- Skill contract loads with the expected shape -------------------------


def test_skill_contract_loads_with_expected_shape():
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    skill = by_id[SKILL_ID]
    assert skill.answer == "number"
    assert skill.status == "active"
    assert len(skill.sources) == 5  # the four variants plus the exam source
    seen_variants = set()
    for source in skill.sources:
        if source.difficulty == "exam":
            continue  # the exam source is covered by this skill's tests/test_exd_x_* file
        assert source.topic == TOPIC
        variant = source.params["variant"]
        seen_variants.add(variant)
        for seed in range(5):
            q = make(source.topic, seed=seed, difficulty=source.difficulty, **source.params)
            assert q.text and q.answer_text
    assert seen_variants == set(VARIANTS)


def test_placement_eligible_is_the_unset_schema_default():
    # Nothing settles placement_eligible; the YAML omits it, and this pins the resulting
    # loaded value (the schema default, `True`) so a silent future change is visible.
    _, skills = load_skills(ROOT / "skills")
    by_id = {skill.id: skill for skill in skills}
    skill = by_id[SKILL_ID]
    assert skill.placement_eligible is True
