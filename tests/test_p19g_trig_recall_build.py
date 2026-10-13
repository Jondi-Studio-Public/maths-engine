"""Tests the exact-trig-value skills, including `p19g_exact_trig_value` and unit-circle coordinates: determinism,
distractors and grading.

Focused contract tests for the exact trig-value recall / unit-circle
coordinate skills:

- ``senior-recognise-the-exact-values-of-cos-theta-sin`` (topic
  ``p19g_exact_trig_value``, `answer: mcq`)
- ``senior-understand-the-unit-circle-definition-of-cos-theta`` (topic
  ``p19g_unit_circle_coordinates``, `answer: expression`)

The independent oracles below parse ONLY `question.text`/`answer_text`
(never anything from `mathgen.generators.p19g_trig_recall_build`) and
re-derive the expected value via a fresh sympy route -- `sympy.cos`/
`sympy.sin`/`sympy.tan` evaluated directly on a theta rebuilt from the
rendered text, catching `zoo` as the undefined case, never the module's own
hand-built `_TABLE`."""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest
import sympy

from mathgen.grading.typed import GRADERS, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.generators.p19g_trig_recall_build import _TABLE, _verify_table

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]

VALUE_TOPIC = "p19g_exact_trig_value"
COORD_TOPIC = "p19g_unit_circle_coordinates"
TOPICS = [VALUE_TOPIC, COORD_TOPIC]

VALUE_ID = "senior-recognise-the-exact-values-of-cos-theta-sin"
COORD_ID = "senior-understand-the-unit-circle-definition-of-cos-theta"


# ===========================================================================
# Determinism, shape, distractor hygiene (both topics, every tier)
# ===========================================================================


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", TIERS)
def test_generators_are_deterministic_across_seeds(topic, difficulty):
    for seed in SEEDS:
        q1 = make(topic, seed=seed, difficulty=difficulty)
        q2 = make(topic, seed=seed, difficulty=difficulty)
        assert q1.to_json() == q2.to_json()


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", TIERS)
def test_distractors_distinct_non_empty_never_equal_answer(topic, difficulty):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty=difficulty)
        assert q.distractors, f"seed {seed}: no distractors"
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors)), q.distractors
        assert len(q.distractors) == 3, f"seed {seed}: {len(q.distractors)} distractors"


#: The whole standard-angle domain has only 16 theta values -- unlike a
#: continuous-range generator, ">= 20 distinct questions" is unreachable at
#: every tier here (skill 1 easy: 3 theta x 2 fn = 6 combinations total;
#: skill 2 easy: 5 theta values). Each floor below is the topic/tier's own
#: FULL combinatorial size (skill 1: theta x fn, minus the tan-at-
#: quadrantal-angle exclusions; skill 2: theta alone) -- over 60 seeds every
#: combination is overwhelmingly likely to appear at least once, so hitting
#: the floor exactly is the real bar, not a placeholder guess.
_EXPECTED_MIN_DISTINCT = {
    (VALUE_TOPIC, "easy"): 6,
    (VALUE_TOPIC, "medium"): 40,
    (VALUE_TOPIC, "hard"): 80,
    (VALUE_TOPIC, "exam"): 60,
    (COORD_TOPIC, "easy"): 5,
    (COORD_TOPIC, "medium"): 16,
    (COORD_TOPIC, "hard"): 30,
    (COORD_TOPIC, "exam"): 20,
}


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("difficulty", TIERS)
def test_distinct_questions_reach_the_tier_s_full_combinatorial_size(topic, difficulty):
    texts = {make(topic, seed=seed, difficulty=difficulty).text for seed in range(200)}
    floor = _EXPECTED_MIN_DISTINCT[(topic, difficulty)]
    assert len(texts) >= floor, f"{topic} [{difficulty}]: only {len(texts)} distinct questions (floor {floor})"


def test_table_verification_helper_is_importable_and_passes():
    """`_verify_table` already ran once at import time; re-running it here
    proves it is a real check (not a no-op) and stays green on a second
    call."""
    _verify_table()


# ===========================================================================
# Skill 1: p19g_exact_trig_value (answer: mcq)
# ===========================================================================


@pytest.mark.parametrize("difficulty", TIERS)
def test_value_answer_is_mcq_option_text(difficulty):
    for seed in SEEDS:
        q = make(VALUE_TOPIC, seed=seed, difficulty=difficulty)
        assert isinstance(q.answer, str)
        assert q.answer == q.answer_text
        options = mcq_options(q)
        assert q.answer_text in options
        assert len(options) == 4


@pytest.mark.parametrize("difficulty", TIERS)
def test_value_real_grader_accepts_correct_index_and_rejects_every_other(difficulty):
    for seed in SEEDS:
        q = make(VALUE_TOPIC, seed=seed, difficulty=difficulty)
        options = mcq_options(q)
        correct_idx = options.index(q.answer_text)
        for i in range(len(options)):
            result = GRADERS["mcq"](q, str(i))
            expected = i == correct_idx
            assert result.correct == expected, (seed, difficulty, i, options)


def test_value_expression_grader_cannot_grade_undefined_by_construction():
    """Direct probe of the load-bearing grader-kind decision (see the
    module docstring): the literal word "undefined" typed against
    the expression grader is parsed as a product of six single-letter
    symbols, never a keyword, and the sympy-native spelling "zoo" fails
    `symbolically_equal` because `simplify(zoo - zoo)` is `nan`, not `0`."""
    found_undefined = None
    for seed in range(200):
        q = make(VALUE_TOPIC, seed=seed, difficulty="hard")
        if q.answer_text == "undefined":
            found_undefined = q
            break
    assert found_undefined is not None, "no undefined-tangent draw found in 200 hard seeds"
    q = found_undefined
    result_word = GRADERS["expression"](q, "undefined")
    assert not result_word.correct
    parsed = sympy.sympify("undefined")
    assert parsed.free_symbols, "expected 'undefined' to parse as a product of free symbols"
    assert sympy.simplify(sympy.zoo - sympy.zoo) is sympy.nan


@pytest.mark.parametrize("difficulty", ["hard", "exam"])
def test_value_undefined_tangent_is_a_real_key_on_hard_and_exam(difficulty):
    seen_undefined = any(
        make(VALUE_TOPIC, seed=seed, difficulty=difficulty).answer_text == "undefined" for seed in range(200)
    )
    assert seen_undefined, f"{difficulty}: undefined never occurred as the correct key over 200 seeds"


def test_value_never_undefined_on_easy_or_medium():
    for difficulty in ["easy", "medium"]:
        for seed in range(200):
            q = make(VALUE_TOPIC, seed=seed, difficulty=difficulty)
            assert q.answer_text != "undefined", (difficulty, seed, q.text)


def test_value_undefined_hint_only_when_undefined_option_is_offered():
    """The tan prompt's 'Select "undefined"' hint must never name an option that is not among the four offered.
    200 seeds x 4 tiers."""
    tan_draws = 0
    for difficulty in TIERS:
        for seed in range(200):
            q = make(VALUE_TOPIC, seed=seed, difficulty=difficulty)
            if 'Select "undefined"' in q.text:
                assert "undefined" in mcq_options(q), (difficulty, seed, q.text)
            if "value of tan(theta)" in q.text:
                tan_draws += 1
    assert tan_draws > 100


def test_value_no_dead_distractors_undefined_for_sin_cos_or_half_for_tan():
    """Sin/cos are always defined, so never offer "undefined"; tan never equals +-1/2 at a standard angle, so
    never offer those."""
    seen = {"sincos": 0, "tan": 0}
    for difficulty in TIERS:
        for seed in range(200):
            q = make(VALUE_TOPIC, seed=seed, difficulty=difficulty)
            options = mcq_options(q)
            if "value of tan(theta)" in q.text:
                seen["tan"] += 1
                assert "1/2" not in options and "-1/2" not in options, (difficulty, seed, options)
            else:
                seen["sincos"] += 1
                assert "undefined" not in options, (difficulty, seed, q.text, options)
            assert len(set(options)) == 4, (difficulty, seed, options)
    assert seen["tan"] > 100 and seen["sincos"] > 100


# --- Answer-space contract: 200 seeds/tier, top value < 15%, cos/sin/tan split ---

_FN_RE = re.compile(r"value of (\w+)\(theta\)")


@pytest.mark.parametrize("difficulty", TIERS)
def test_value_top_answer_concentration_and_function_split(difficulty):
    """The answer-space contract ("top exact value under 15% of 200 draws per tier") is checked as
    written on medium/hard/exam, which each draw from >= 40 (theta, function) combinations. The easy tier's
    OWN progression text fixes it to exactly 3 theta values x 2 functions = 6 combinations, and
    cos(pi/4) == sin(pi/4) collapses that to only 3 distinct VALUES ("1/2", "sqrt(3)/2", "sqrt(2)/2") -- a
    true mathematical floor of 1/3 (33%) that no draw distribution can get under 15%. Reported, not silently
    enforced, for easy."""
    answers = Counter()
    fns = Counter()
    for seed in range(200):
        q = make(VALUE_TOPIC, seed=seed, difficulty=difficulty)
        answers[q.answer_text] += 1
        match = _FN_RE.search(q.text)
        assert match, q.text
        fns[match.group(1)] += 1
    top = max(answers.values()) / 200
    if difficulty == "easy":
        assert top < 0.40, (difficulty, answers.most_common(3))  # true floor is 1/3
        assert set(fns) == {"cos", "sin"}
    else:
        assert top < 0.15, (difficulty, answers.most_common(3))
        assert set(fns) >= {"cos", "sin", "tan"}
    for fn, count in fns.items():
        assert count > 0, (difficulty, fns)


# --- Independent oracle: parse the RENDERED TEXT, never the generator's own table ---


def _parse_pi_multiple(term: str) -> Fraction:
    term = term.strip()
    if term == "0":
        return Fraction(0)
    match = re.fullmatch(r"(-?)(\d*)pi(?:/(\d+))?", term)
    assert match, term
    sign, num, den = match.groups()
    coeff = Fraction(int(num) if num else 1, int(den) if den else 1)
    return -coeff if sign == "-" else coeff


def _parse_theta_multiple(theta_str: str) -> Fraction:
    """Independent parse of a rendered theta (a bare pi-multiple, a
    degrees phrase, or a compound sum/difference) into the Fraction
    coefficient of pi -- a wholly separate route from
    `p19g_trig_recall_build._theta_str`/`_compound_str`/`_degrees_str`."""
    theta_str = theta_str.strip()
    deg_match = re.fullmatch(r"(-?\d+) degrees", theta_str)
    if deg_match:
        return Fraction(int(deg_match.group(1)), 180)
    if " - " in theta_str:
        left, right = theta_str.split(" - ", 1)
        return _parse_pi_multiple(left) - _parse_pi_multiple(right)
    if " + " in theta_str:
        left, right = theta_str.split(" + ", 1)
        return _parse_pi_multiple(left) + _parse_pi_multiple(right)
    return _parse_pi_multiple(theta_str)


_THETA_RE = re.compile(r"theta = ([^,]+),", re.IGNORECASE)
_FN_ORACLE = {"cos": sympy.cos, "sin": sympy.sin, "tan": sympy.tan}


def _oracle_value_str(theta_str: str, fn: str) -> str | None:
    """Returns the option TEXT the oracle computes as correct, or the
    literal string "undefined" -- via `sympy.cos`/`sin`/`tan` evaluated
    directly on a theta rebuilt from the rendered text, never `_TABLE`."""
    coeff = _parse_theta_multiple(theta_str)
    theta = sympy.Rational(coeff.numerator, coeff.denominator) * sympy.pi
    value = _FN_ORACLE[fn](theta)
    if value == sympy.zoo or value is sympy.nan:
        return "undefined"
    return value


@pytest.mark.parametrize("difficulty", TIERS)
def test_value_oracle_matches_generator_over_300_seeds_per_tier(difficulty):
    """>= 1000 seeds total across the 4 tiers (300 x 4 = 1200), as
    an oracle requirement -- re-derives the correct mcq option via
    a wholly different sympy route than construction's hand-built table,
    and asserts the option at the correct index matches."""
    for seed in range(300):
        q = make(VALUE_TOPIC, seed=seed, difficulty=difficulty)
        theta_match = _THETA_RE.search(q.text)
        assert theta_match, q.text
        fn_match = _FN_RE.search(q.text)
        assert fn_match, q.text
        oracle_value = _oracle_value_str(theta_match.group(1), fn_match.group(1))
        if oracle_value == "undefined":
            assert q.answer_text == "undefined", (seed, difficulty, q.text)
        else:
            assert q.answer_text != "undefined", (seed, difficulty, q.text)
            assert sympy.simplify(sympy.sympify(q.answer_text) - oracle_value) == 0, (
                seed,
                difficulty,
                q.text,
                q.answer_text,
                oracle_value,
            )
        # The mcq option AT the correct index (not just the canonical
        # string) is confirmed via the real grader.
        options = mcq_options(q)
        correct_idx = options.index(q.answer_text)
        assert GRADERS["mcq"](q, str(correct_idx)).correct


# ===========================================================================
# Skill 2: p19g_unit_circle_coordinates (answer: expression, Tuple-shaped)
# ===========================================================================

_PAIR_RE = re.compile(r"^\(([^,]+), ([^,]+)\)$")


@pytest.mark.parametrize("difficulty", TIERS)
def test_coordinates_answer_is_tuple_shaped_string(difficulty):
    for seed in SEEDS:
        q = make(COORD_TOPIC, seed=seed, difficulty=difficulty)
        assert isinstance(q.answer, str)
        assert q.answer == q.answer_text
        assert _PAIR_RE.match(q.answer_text), q.answer_text
        for d in q.distractors:
            assert _PAIR_RE.match(d), d


@pytest.mark.parametrize("difficulty", TIERS)
def test_coordinates_real_grader_accepts_canonical_spelling_variants(difficulty):
    for seed in range(20):
        q = make(COORD_TOPIC, seed=seed, difficulty=difficulty)
        match = _PAIR_RE.match(q.answer_text)
        cos_part, sin_part = match.group(1), match.group(2)
        variants = [
            q.answer_text,
            f"({cos_part},{sin_part})",  # no space after comma
            f"{cos_part}, {sin_part}",  # no parentheses
        ]
        for variant in variants:
            result = GRADERS["expression"](q, variant)
            assert result.correct, (seed, difficulty, variant)


@pytest.mark.parametrize("difficulty", TIERS)
def test_coordinates_real_grader_rejects_decimal_approximation_and_distractors(difficulty):
    for seed in SEEDS:
        q = make(COORD_TOPIC, seed=seed, difficulty=difficulty)
        for i, distractor in enumerate(q.distractors):
            result = GRADERS["expression"](q, distractor)
            assert not result.correct, f"seed {seed}: distractor {i} {distractor!r} accepted"


def test_coordinates_decimal_approximation_rejected_direct_probe():
    """Direct probe: a decimal approximation like
    "(0.866, 0.5)" for "(sqrt(3)/2, 1/2)" must grade incorrect (exact-surd-
    strict), never accepted as a numeric near-match."""
    q = next(
        make(COORD_TOPIC, seed=s, difficulty="medium")
        for s in range(60)
        if make(COORD_TOPIC, seed=s, difficulty="medium").answer_text == "(sqrt(3)/2, 1/2)"
    )
    result = GRADERS["expression"](q, "(0.866, 0.5)")
    assert not result.correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_coordinates_top_answer_concentration_and_quadrant_sign_split(difficulty):
    """Regression guard for a previously fixed defect."""
    answers = Counter()
    sign_patterns = Counter()
    for seed in range(200):
        q = make(COORD_TOPIC, seed=seed, difficulty=difficulty)
        answers[q.answer_text] += 1
        match = _PAIR_RE.match(q.answer_text)
        cos_sign = "-" if match.group(1).startswith("-") else "+"
        sin_sign = "-" if match.group(2).startswith("-") else "+"
        sign_patterns[(cos_sign, sin_sign)] += 1
    top = max(answers.values()) / 200
    if difficulty == "easy":
        assert top < 0.25, (difficulty, answers.most_common(3))  # true floor is 1/5
    else:
        assert top < 0.15, (difficulty, answers.most_common(3))
    if difficulty != "easy":
        # Both quadrant-sign combinations are represented (not clustered in
        # quadrant I): at least one draw with a negative cos and at least
        # one draw with a negative sin somewhere across the sweep.
        assert any(cos == "-" for cos, _sin in sign_patterns), (difficulty, sign_patterns)
        assert any(sin == "-" for _cos, sin in sign_patterns), (difficulty, sign_patterns)


# --- Independent oracle: recompute (cos theta, sin theta) directly, never `_TABLE` ---

_COORD_THETA_RE = re.compile(r"theta = ([^,]+),", re.IGNORECASE)


@pytest.mark.parametrize("difficulty", TIERS)
def test_coordinates_oracle_matches_generator_over_60_seeds(difficulty):
    for seed in SEEDS:
        q = make(COORD_TOPIC, seed=seed, difficulty=difficulty)
        theta_match = _COORD_THETA_RE.search(q.text)
        assert theta_match, q.text
        coeff = _parse_theta_multiple(theta_match.group(1))
        theta = sympy.Rational(coeff.numerator, coeff.denominator) * sympy.pi
        oracle_cos, oracle_sin = sympy.cos(theta), sympy.sin(theta)
        match = _PAIR_RE.match(q.answer_text)
        assert sympy.simplify(sympy.sympify(match.group(1)) - oracle_cos) == 0, (seed, difficulty, q.text)
        assert sympy.simplify(sympy.sympify(match.group(2)) - oracle_sin) == 0, (seed, difficulty, q.text)


# ===========================================================================
# Boundary: skill 1 (single value) vs skill 2 (full pair) vs the shipping
# ACMMM034 sibling (derives tan from a supplied plotted point)
# ===========================================================================


def test_value_never_states_full_coordinate_pair():
    for difficulty in TIERS:
        for seed in range(30):
            q = make(VALUE_TOPIC, seed=seed, difficulty=difficulty)
            assert "," not in q.answer_text  # a Tuple-shaped pair always has a comma; a scalar never does


def test_coordinates_never_asks_for_a_single_named_value():
    for difficulty in TIERS:
        for seed in range(30):
            q = make(COORD_TOPIC, seed=seed, difficulty=difficulty)
            assert "value of" not in q.text


def test_neither_skill_gives_a_plotted_terminal_point():
    for topic in TOPICS:
        for difficulty in TIERS:
            for seed in range(30):
                q = make(topic, seed=seed, difficulty=difficulty)
                assert "terminal point" not in q.text


# ===========================================================================
# Skill contract loading tests
# ===========================================================================


def test_both_skills_load_with_correct_answer_kind_and_prerequisites():
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    value_skill = by_id[VALUE_ID]
    coord_skill = by_id[COORD_ID]
    assert value_skill.answer == "mcq"
    assert value_skill.prerequisites == [
        "senior-understand-the-unit-circle-definition-of-cos-theta-acmmm034",
        "angle-degrees-radians-pi-form",
    ]
    assert coord_skill.answer == "expression"
    assert coord_skill.prerequisites == ["angle-degrees-radians-pi-form", "f10-trig-ratios-similarity"]


def test_both_skills_are_active_placement_ineligible_and_general_band():
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    for skill_id in (VALUE_ID, COORD_ID):
        skill = by_id[skill_id]
        assert skill.status == "active"
        assert skill.placement_eligible is False
        assert getattr(skill, "learner_band", "general") == "general"


def test_both_skills_have_sources_covering_all_four_tiers():
    _, skills = load_skills(ROOT / "skills")
    by_id = {s.id: s for s in skills}
    for skill_id in (VALUE_ID, COORD_ID):
        difficulties = {src.difficulty for src in by_id[skill_id].sources}
        assert difficulties == {"easy", "medium", "hard", "exam"}
    for skill_id in (VALUE_ID, COORD_ID):
        skill = by_id[skill_id]
        exam_src = next(s for s in skill.sources if s.difficulty == "exam")
        assert exam_src.stages == ["exam"]


def test_table_verification_raises_on_a_corrupted_entry():
    """`_verify_table`'s own mechanism is a real check, not a decorative
    call: corrupt one entry via a monkeypatch-free direct construction and
    confirm `check_equivalent`/`check` would fire (proved on a copy, not by
    mutating the shared module table)."""
    from mathgen.core import check_equivalent

    with pytest.raises(VerificationError):
        check_equivalent(sympy.sympify("1/2"), sympy.cos(sympy.pi / 6), "deliberately wrong")


def test_2139_value_easy_has_at_least_40_prompts_and_5_key_texts_over_200_seeds():
    prompts, keys = set(), set()
    for seed in range(200):
        q = make(VALUE_TOPIC, seed=seed, difficulty="easy")
        prompts.add(q.text)
        keys.add(q.answer_text)
        options = mcq_options(q)
        assert len(set(options)) == 4 and q.answer_text in options
        theta = _parse_theta_multiple(_THETA_RE.search(q.text).group(1))
        assert 0 <= theta <= Fraction(1, 2), q.text  # stays first-quadrant / axis
        assert _FN_RE.search(q.text).group(1) in {"cos", "sin"}
    assert len(prompts) >= 40, len(prompts)
    assert len(keys) >= 5, keys


def _consensus_scores(q):
    """Per option, how many of its component strings also appear in the other
    options (the shared-coordinate 'consensus' signal)."""
    options = [q.answer_text, *q.distractors]
    comps = [set(_PAIR_RE.match(o).groups()) for o in options]
    scores = []
    for i, own in enumerate(comps):
        others = [c for j, c in enumerate(comps) if j != i]
        scores.append(sum(sum(1 for c in other if c in own) for other in others))
    return scores


def _attack_credit(scores, pick_max):
    """Fractional credit for the key (index 0) when the attacker picks the
    highest (or lowest) scoring option, ties shared."""
    target = max(scores) if pick_max else min(scores)
    tied = [i for i, s in enumerate(scores) if s == target]
    return (1 / len(tied)) if 0 in tied else 0.0


@pytest.mark.parametrize("difficulty", TIERS)
def test_2143_coordinates_consensus_attack_at_chance(difficulty):
    high = low = 0.0
    for seed in range(200):
        q = make(COORD_TOPIC, seed=seed, difficulty=difficulty)
        scores = _consensus_scores(q)
        high += _attack_credit(scores, True)
        low += _attack_credit(scores, False)
    assert high / 200 <= 0.40, (difficulty, "max-consensus", high / 200)
    assert low / 200 <= 0.40, (difficulty, "min-consensus", low / 200)


@pytest.mark.parametrize("difficulty", TIERS)
def test_2143_coordinates_distractors_distinct_and_not_key_by_value(difficulty):
    for seed in range(200):
        q = make(COORD_TOPIC, seed=seed, difficulty=difficulty)
        assert len(q.distractors) == 3 and len(set(q.distractors)) == 3
        key = tuple(sympy.simplify(sympy.sympify(p)) for p in _PAIR_RE.match(q.answer_text).groups())
        for d in q.distractors:
            val = tuple(sympy.simplify(sympy.sympify(p)) for p in _PAIR_RE.match(d).groups())
            assert val != key, (seed, difficulty, d)


@pytest.mark.parametrize("difficulty", TIERS)
def test_2143_coordinates_grader_accepts_key_rejects_distractors(difficulty):
    for seed in range(40):
        q = make(COORD_TOPIC, seed=seed, difficulty=difficulty)
        assert GRADERS["expression"](q, q.answer_text).correct, (seed, difficulty)
        for d in q.distractors:
            assert not GRADERS["expression"](q, d).correct, (seed, difficulty, d)


def test_2143_coordinates_easy_and_medium_are_wider():
    for difficulty, min_prompts in (("easy", 25), ("medium", 60)):
        prompts = set()
        for seed in range(300):
            q = make(COORD_TOPIC, seed=seed, difficulty=difficulty)
            assert _COORD_THETA_RE.search(q.text), q.text
            assert "value of" not in q.text
            prompts.add(q.text)
        assert len(prompts) >= min_prompts, (difficulty, len(prompts))


_2147_TABLE_PAIRS = {(c, s) for c, s, _t in _TABLE.values()}


def _2147_parts(option):
    return _PAIR_RE.match(option).groups()


@pytest.mark.parametrize("difficulty", TIERS)
def test_2147_coordinates_every_distractor_on_unit_circle_within_range(difficulty):
    for seed in range(200):
        q = make(COORD_TOPIC, seed=seed, difficulty=difficulty)
        for d in q.distractors:
            x, y = (sympy.simplify(sympy.sympify(p)) for p in _2147_parts(d))
            assert sympy.simplify(x**2 + y**2 - 1) == 0, (seed, difficulty, d)
            assert -1 <= x <= 1 and -1 <= y <= 1, (seed, difficulty, d)
            assert _2147_parts(d) in _2147_TABLE_PAIRS, (seed, difficulty, d)


@pytest.mark.parametrize("difficulty", TIERS)
def test_2147_coordinates_strict_and_fractional_consensus_attack(difficulty):
    strict_hi = strict_lo = frac_hi = frac_lo = 0.0
    for seed in range(200):
        q = make(COORD_TOPIC, seed=seed, difficulty=difficulty)
        scores = _consensus_scores(q)
        strict_hi += scores[0] == max(scores)  # first option wins ties; key is option 0
        strict_lo += scores[0] == min(scores)
        frac_hi += _attack_credit(scores, True)
        frac_lo += _attack_credit(scores, False)
    for name, value in (
        ("strict-max", strict_hi),
        ("strict-min", strict_lo),
        ("frac-max", frac_hi),
        ("frac-min", frac_lo),
    ):
        assert value / 200 <= 0.40, (difficulty, name, value / 200)


@pytest.mark.parametrize("difficulty", TIERS)
def test_2147_coordinates_distractors_do_not_cluster(difficulty):
    shared = total = 0
    for seed in range(200):
        q = make(COORD_TOPIC, seed=seed, difficulty=difficulty)
        comps = [set(_2147_parts(d)) for d in q.distractors]
        assert len({frozenset(c) for c in comps}) == 3, (seed, difficulty, q.distractors)
        for i in range(3):
            for j in range(i + 1, 3):
                total += 1
                shared += bool(comps[i] & comps[j])
    assert shared / total <= 0.45, (difficulty, shared / total)
