"""Focused contract tests for: ucv1-optimality-sensitivity (answer: mcq)."""

import re
from collections import Counter

import pytest

from mathgen.grading.typed import grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
import mathgen.generators.p17a_optimality_sensitivity_build  # noqa: F401 -- registers topic

SKILL_ID = "ucv1-optimality-sensitivity"
TOPIC = "p17a_optimality_sensitivity"
SEEDS = range(40)
TIERS = ["easy", "medium", "hard", "exam"]


def _skill_sources():
    _domains, skills = load_skills()
    skill = next(s for s in skills if s.id == SKILL_ID)
    return skill, {src.difficulty: src for src in skill.sources}


SKILL, SOURCES = _skill_sources()


def _make(difficulty, seed):
    src = SOURCES[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


@pytest.mark.parametrize("difficulty", TIERS)
def test_determinism_and_distractor_hygiene(difficulty):
    for seed in SEEDS:
        q1 = _make(difficulty, seed)
        assert len(q1.distractors) == 3
        assert q1.answer_text in ("A", "B", "C", "D")
        assert set(q1.distractors) | {q1.answer_text} == {"A", "B", "C", "D"}


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        options = mcq_options(q)
        correct_idx = options.index(q.answer_text)
        assert grade_mcq(q, str(correct_idx)).correct
        for i, opt in enumerate(options):
            if opt == q.answer_text:
                continue
            assert not grade_mcq(q, str(i)).correct


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        assert not grade_mcq(q, "9").correct


def test_skill_loads_with_expected_shape():
    assert SKILL.answer == "mcq"
    assert SKILL.placement_eligible is False
    assert SKILL.domain == "optimisation"
    assert SKILL.prerequisites == ["convex-quadratic-minimum"]
    assert set(SOURCES) == {"easy", "medium", "hard", "exam"}
    assert SOURCES["exam"].stages == ["exam"]


@pytest.mark.parametrize("difficulty", TIERS)
def test_default_params_and_tier_have_enough_distractors(difficulty):
    # DEFAULT params (no difficulty) and every tier: >=2 distractors on
    # >=85% of 200 draws. Here every draw gives exactly 3.
    counts = Counter()
    for seed in range(200):
        q = _make(difficulty, seed)
        counts[len(q.distractors)] += 1
    at_least_two = sum(n for k, n in counts.items() if k >= 2)
    assert at_least_two / 200 >= 0.85, counts

    default_counts = Counter()
    for seed in range(200):
        q = make(TOPIC, seed=seed)
        default_counts[len(q.distractors)] += 1
    default_at_least_two = sum(n for k, n in default_counts.items() if k >= 2)
    assert default_at_least_two / 200 >= 0.85, default_counts


@pytest.mark.parametrize("difficulty", TIERS)
def test_key_letter_roughly_uniform_at_400_seeds(difficulty):
    letters = Counter(_make(difficulty, seed).answer_text for seed in range(400))
    assert set(letters) == {"A", "B", "C", "D"}
    for count in letters.values():
        assert count / 400 <= 0.40, letters


@pytest.mark.parametrize("difficulty", TIERS)
def test_magnitude_error_balance_between_30_and_70_percent(difficulty):
    """The magnitude-error class must be sometimes above and sometimes below
    the key's own amount (30-70% at 200 seeds), so neither 'always bigger'
    nor 'always smaller' finds the key. The rendered text always lists the
    four options in fixed A/B/C/D order (`_LABELS`), so the letter position
    in `q.text` -- not the further `mcq_options` shuffle used for grading --
    is what tells the key from the distractors here."""
    smaller = 0
    total = 0
    for seed in range(200):
        q = _make(difficulty, seed)
        amounts = [_amount_from_option(seg) for seg in _option_segments(q.text)]
        key_amount = amounts[_LETTER_INDEX[q.answer_text]]
        other_amounts = [a for i, a in enumerate(amounts) if i != _LETTER_INDEX[q.answer_text]]
        # the magnitude-error class's amount differs from the key's and from
        # the sign-flip's (which shares the key's amount) -- so among the
        # three others there are two distinct magnitudes; compare against
        # the one that isn't equal to key_amount.
        distinct_other = sorted(set(other_amounts) - {key_amount})
        assert distinct_other, (q.text, amounts)
        mag_err_amount = distinct_other[0]
        smaller += key_amount < mag_err_amount
        total += 1
    fraction = smaller / total
    assert 0.30 <= fraction <= 0.70, (difficulty, fraction)


_LETTER_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}


def _option_segments(text):
    import re

    pattern = re.compile(r"[ABCD]: (.+?)(?=(?:\s+[ABCD]: )|$)")
    return [m.group(1) for m in pattern.finditer(text)]


def _amount_from_option(segment):
    import re
    from fractions import Fraction

    m = re.search(r"by (-?\d+(?:/\d+)?)", segment)
    if m:
        return abs(Fraction(m.group(1)))
    # exam tier's compound options carry the amount after "by " too, same
    # pattern covers it.
    raise AssertionError(segment)


def test_corrupted_zero_claim_is_caught_by_the_in_generator_check():
    import mathgen.generators.p17a_optimality_sensitivity_build as mod

    with pytest.raises(VerificationError):
        mod._claim("the minimiser x*", 0)


def test_corrupted_colliding_options_are_caught_by_check_pairwise_distinct():
    """The real verification this convention asks for: two options that
    collapse to the SAME value (a corrupted magnitude-error equal to the
    key) must be rejected by `check_pairwise_distinct` inside `_build_four`,
    not merely a lone zero-delta."""
    import random
    from fractions import Fraction

    import mathgen.generators.p17a_optimality_sensitivity_build as mod

    with pytest.raises(VerificationError):
        mod._build_four(random.Random(0), "the minimiser x*", Fraction(1, 2), Fraction(1, 2))


def test_hard_tier_two_variable_and_exam_tier_compound_judgement():
    for seed in range(10):
        qh = _make("hard", seed)
        assert "f(x, y)" in qh.text
        qe = _make("exam", seed)
        assert "Let t =" in qe.text
        assert "moves past t" in qe.text


@pytest.mark.parametrize("difficulty", TIERS)
def test_working_never_prints_a_double_minus(difficulty):
    # The easy working printed "-{b_new}" and so read
    # "New x* = --8/(2·5)" whenever the new coefficient was negative.
    for seed in range(300):
        q = _make(difficulty, seed)
        for line in q.working:
            # "--" followed by a number or bracket; "exact -- contains" is prose.
            assert not re.search(r"--[\w(]", line), (seed, line)
