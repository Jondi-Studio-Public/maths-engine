"""Focused contract tests for: `ucv1-ftc-polynomial` (`answer: number`)."""

from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
import mathgen.generators.p17a_ftc_polynomial_build  # noqa: F401 -- registers the topic

SKILL_ID = "ucv1-ftc-polynomial"
TOPIC = "p17a_ftc_inverse"

SEEDS = range(60)
TIERS = ["easy", "medium", "hard", "exam"]


def _skill_sources():
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == SKILL_ID)
    return skill, {src.difficulty: src for src in skill.sources}


SKILL, SOURCES = _skill_sources()


def _make(difficulty, seed):
    src = SOURCES[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# ---------------------------------------------------------------------------
# determinism / distractor hygiene
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_distractors_distinct_nonempty_never_equal_to_answer(difficulty):
    zero_count = 0
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for seed in range(200):
        q = _make(difficulty, seed)
        n = min(len(q.distractors), 3)
        counts[n] += 1
        assert q.answer_text not in q.distractors
        assert len(set(q.distractors)) == len(q.distractors)
        if n == 0:
            zero_count += 1
    # >= 2 distractors on >= 85% of draws (default params, every tier).
    at_least_2 = counts[2] + counts[3]
    assert at_least_2 / 200 >= 0.85, (difficulty, counts)


# ---------------------------------------------------------------------------
# real grader: accept canonical, reject every distractor and a deliberately
# wrong value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        # A value far from the answer and every distractor, formatted like
        # the tier's own answer_text.
        if difficulty == "easy":
            wrong = "999/7"
        else:
            wrong = "-999.00"
        assert not grade_number(q, wrong).correct, (difficulty, seed, q.text)


# ---------------------------------------------------------------------------
# answer-kind / format-alike
# ---------------------------------------------------------------------------


def test_easy_answers_are_exact_rationals_never_decimal():
    import re

    decimal_re = re.compile(r"\d+\.\d+")
    for seed in range(200):
        q = _make("easy", seed)
        assert isinstance(q.answer, int)
        assert not decimal_re.search(q.answer_text), q.answer_text
        for d in q.distractors:
            assert not decimal_re.search(d), d


@pytest.mark.parametrize("difficulty", ["medium", "hard", "exam"])
def test_limit_tiers_display_exactly_two_decimal_places(difficulty):
    import re

    two_dp_re = re.compile(r"^-?\d+\.\d\d$")
    for seed in range(100):
        q = _make(difficulty, seed)
        assert two_dp_re.match(q.answer_text), q.answer_text
        for d in q.distractors:
            assert two_dp_re.match(d), d


@pytest.mark.parametrize("difficulty", ["medium", "hard", "exam"])
def test_precision_instruction_present_in_text_and_latex(difficulty):
    for seed in range(20):
        q = _make(difficulty, seed)
        assert "correct to 2 decimal places" in q.text.lower()
        assert "correct to 2 decimal places" in (q.latex or "").lower()


# ---------------------------------------------------------------------------
# forward-evaluation exclusion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_every_prompt_states_the_value_and_asks_for_the_unknown(difficulty):
    for seed in range(30):
        q = _make(difficulty, seed)
        assert "find" in q.text.lower() or "what is" in q.text.lower()
        assert "equals" in q.text.lower() or " is " in q.text.lower()


# ---------------------------------------------------------------------------
# in-generator check firing on a corrupted value
# ---------------------------------------------------------------------------


def test_in_generator_check_fires_on_corrupted_root():
    from mathgen.core import VerificationError, check_equivalent

    with pytest.raises(VerificationError):
        check_equivalent(Fraction(3, 1), Fraction(4, 1), "deliberately corrupted")


# ---------------------------------------------------------------------------
# skill load
# ---------------------------------------------------------------------------


def test_skill_loads_with_expected_shape():
    assert SKILL.answer == "number"
    assert SKILL.placement_eligible is False
    assert SKILL.domain == "single-variable-calculus"
    assert SKILL.prerequisites == ["integrate-polynomial"]
    assert set(SOURCES) == {"easy", "medium", "hard", "exam"}
    assert SOURCES["exam"].stages == ["exam"]


# ---------------------------------------------------------------------------
# ladder shape: hard has a nonzero lower limit and a rejected root; medium's
# lower limit is fixed at 0; exam states a condition on TWO integrals.
# ---------------------------------------------------------------------------


def test_hard_uses_a_nonzero_lower_limit():
    for seed in range(20):
        q = _make("hard", seed)
        assert "from 0 to k" not in q.text
        assert "x = 0 to x = k" not in q.text


def test_medium_lower_limit_is_zero():
    for seed in range(20):
        q = _make("medium", seed)
        assert "from 0 to k" in q.text or "x = 0 to x = k" in q.text


def test_exam_states_a_second_fully_specified_integral():
    for seed in range(20):
        q = _make("exam", seed)
        assert q.text.lower().count("integral") >= 2
        assert "times" in q.text.lower()


def test_no_forward_evaluation_wording():
    """Every prompt gives V and asks for the unknown; none ever says
    'evaluate the definite integral' the way the shipping definite-integral
    skill does."""
    for difficulty in TIERS:
        for seed in range(20):
            q = _make(difficulty, seed)
            assert "evaluate the definite integral" not in q.text.lower()


# ---------------------------------------------------------------------------
# at least 20 distinct rendered questions per tier over 60 seeds
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# no ** or bare * between a number and a variable
# ---------------------------------------------------------------------------


def test_hard_uses_a_quadratic_integrand():
    for seed in range(30):
        q = _make("hard", seed)
        assert "x^2" in q.text, q.text


@pytest.mark.parametrize("difficulty", ["medium", "hard", "exam"])
def test_at_least_two_in_domain_distractors(difficulty):
    import re

    domain_re = re.compile(r"k > (-?\d+)")
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for seed in range(200):
        q = _make(difficulty, seed)
        L = int(domain_re.search(q.text).group(1))
        in_domain = [d for d in q.distractors if float(d) > L]
        counts[min(len(in_domain), 3)] += 1
    at_least_2 = counts[2] + counts[3]
    assert at_least_2 / 200 >= 0.85, (difficulty, counts)


# "Key is the only in-domain option ending.00/.50" <= 10%, and "key ends.00/.50" within 15
# percentage points of the rate for a randomly chosen in-domain distractor.


@pytest.mark.parametrize("difficulty", ["medium", "hard", "exam"])
def test_key_form_carries_no_signal(difficulty):
    import re

    domain_re = re.compile(r"k > (-?\d+)")
    nice_re = re.compile(r"^-?\d+\.(00|50)$")
    n = 200
    key_only_nice = 0
    key_nice = 0
    distractor_total = 0
    distractor_nice = 0
    for seed in range(n):
        q = _make(difficulty, seed)
        L = int(domain_re.search(q.text).group(1))
        in_domain_distractors = [d for d in q.distractors if float(d) > L]
        is_key_nice = bool(nice_re.match(q.answer_text))
        key_nice += is_key_nice
        options_nice = [is_key_nice] + [bool(nice_re.match(d)) for d in in_domain_distractors]
        if is_key_nice and sum(options_nice) == 1:
            key_only_nice += 1
        for d in in_domain_distractors:
            distractor_total += 1
            distractor_nice += bool(nice_re.match(d))
    assert key_only_nice / n <= 0.10, (difficulty, key_only_nice)
    distractor_rate = distractor_nice / distractor_total if distractor_total else 0.0
    assert abs(key_nice / n - distractor_rate) <= 0.15, (difficulty, key_nice / n, distractor_rate)


@pytest.mark.parametrize("difficulty", TIERS)
def test_no_double_star_or_number_star_variable(difficulty):
    import re

    bad_re = re.compile(r"\*\*|\d\s*\*\s*[a-zA-Z]")
    for seed in range(30):
        q = _make(difficulty, seed)
        assert not bad_re.search(q.text), q.text
        assert not bad_re.search(q.answer_text), q.answer_text
        for d in q.distractors:
            assert not bad_re.search(d), d


@pytest.mark.parametrize("value", [Fraction(1, 8), Fraction(-1, 8), Fraction(-3, 8), Fraction(5, 8)])
def test_round_grid_rejects_a_halfway_tie_of_either_sign(value):
    # Decimal % keeps the dividend's sign, so a negative tie
    # left a remainder of -0.5 and slipped past the ambiguity guard.
    import sympy
    from mathgen.generators.p17a_ftc_polynomial_build import _round_grid

    assert _round_grid(sympy.Rational(value.numerator, value.denominator), 2) is None
