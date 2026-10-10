"""Focused contract tests for: the unknown inside a logarithm
(senior-solve-simple-equations-involving-logarithmic-functions-algebraically-and, ACMMM157)."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from mathgen.grading.typed import grade_fraction
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError

TOPIC = "pv0b_log_equation"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(60)

_SKILL_ID = "senior-solve-simple-equations-involving-logarithmic-functions-algebraically-and"

_domains, _skills = load_skills()


# --- determinism / distractor hygiene ---------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_generators_are_deterministic_with_unique_nonempty_distractors(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        repeat = make(TOPIC, seed=seed, difficulty=difficulty)
        assert q.to_json() == repeat.to_json()
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))
        assert len(q.distractors) >= 1


@pytest.mark.parametrize("difficulty", TIERS)
def test_at_least_20_distinct_questions_per_tier_over_60_seeds(difficulty):
    texts = {make(TOPIC, seed=s, difficulty=difficulty).text for s in SEEDS}
    assert len(texts) >= 20, f"{difficulty}: only {len(texts)} distinct questions over 60 seeds"


@pytest.mark.parametrize("difficulty", TIERS)
def test_at_least_two_distractors_on_most_draws(difficulty):
    counts = [len(make(TOPIC, seed=s, difficulty=difficulty).distractors) for s in range(200)]
    at_least_two = sum(1 for c in counts if c >= 2)
    assert at_least_two / len(counts) >= 0.85, (
        f"{difficulty}: only {at_least_two}/{len(counts)} draws have >= 2 distractors"
    )


# --- answer kind / real grader -----------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_is_fraction_dict_in_lowest_terms(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert isinstance(q.answer, dict) and set(q.answer) == {"p", "q"}
        p, den = q.answer["p"], q.answer["q"]
        assert isinstance(p, int) and isinstance(den, int) and den > 0
        assert math.gcd(abs(p), den) == 1, f"{p}/{den} is not in lowest terms"
        assert q.answer_text == (str(p) if den == 1 else f"{p}/{den}")


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_canonical_and_rejects_distractors(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert grade_fraction(q, q.answer_text).correct, f"{difficulty} seed {seed}: {q.answer_text!r}"
        for d in q.distractors:
            assert not grade_fraction(q, d).correct, f"{difficulty} seed {seed}: distractor {d!r} graded correct"


@pytest.mark.parametrize("difficulty", TIERS)
def test_deliberately_wrong_answer_is_rejected(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        x = Fraction(q.answer["p"], q.answer["q"])
        wrong = x + 1000
        assert not grade_fraction(q, str(wrong)).correct


# --- grader-probe table -----


def test_grader_probe_integer_and_fraction_spellings():
    class _Q:
        def __init__(self, answer_text, answer):
            self.answer_text = answer_text
            self.answer = answer

    int_key = _Q("7", {"p": 7, "q": 1})
    assert grade_fraction(int_key, "7").correct
    assert grade_fraction(int_key, "7/1").correct
    assert not grade_fraction(int_key, "x = 7").correct
    assert grade_fraction(int_key, "7.0").correct

    frac_key = _Q("-3/2", {"p": -3, "q": 2})
    assert grade_fraction(frac_key, "-3/2").correct
    assert grade_fraction(frac_key, "-1.5").correct
    assert not grade_fraction(frac_key, "x = -3/2").correct
    assert not grade_fraction(frac_key, "(-3)/2").correct
    assert grade_fraction(frac_key, "-6/4").correct


# --- distractor shape (no lone fraction/integer among the options) ---------


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_never_lone_shape_among_options(difficulty):
    lone = 0
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        shapes = ["/" in q.answer_text] + ["/" in d for d in q.distractors]
        n_frac = sum(shapes)
        n_int = len(shapes) - n_frac
        if (shapes[0] and n_frac == 1) or (not shapes[0] and n_int == 1):
            lone += 1
    assert lone / 200 < 0.05, f"{difficulty}: answer is the lone shape on {lone}/200 draws"


# --- no distractor equals a number printed in the prompt (lesson 9) --------


_DIGIT_RUN_RE = re.compile(r"\d+")


@pytest.mark.parametrize("difficulty", TIERS)
def test_no_distractor_equals_a_printed_number(difficulty):
    """A bare-integer distractor must never equal the MAGNITUDE of any
    literal digit-run printed in the prompt, regardless of the sign context
    it was printed under (a distractor "4" beside a prompt showing "... - 4)"
    is still a printed-number collision even though the printed value is
    really -4) -- proven red just below."""
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        # Numbers printed before the trailing "Give x as ..." instruction
        # sentence, which itself carries no digits.
        prompt = q.text.split(" Give x as")[0]
        printed = {int(tok) for tok in _DIGIT_RUN_RE.findall(prompt)}
        for d in q.distractors:
            value = Fraction(d)
            if value.denominator == 1:
                assert abs(value.numerator) not in printed, (
                    f"{difficulty} seed {seed}: distractor {d!r} equals a printed number in {q.text!r}"
                )


def test_no_distractor_equals_a_printed_number_proven_red_on_unguarded_draw():
    """Proof this check is not vacuous: an unfiltered candidate CAN equal a
    printed number (easy seed 183 draws a=b=4, which collides `forgot-c`
    with the base `b` itself before the printed-number filter runs)."""
    from mathgen.generators.pv0b_log_equation_build import _core_distractors

    b, k, a, c = 4, 2, 4, -8
    value = Fraction(b) ** k
    core = _core_distractors(b, k, a, c, value)
    forgot_c = Fraction(value) / a
    assert forgot_c in core
    assert forgot_c == b  # this raw candidate DOES equal a printed number (b)


# --- per-tier structural guards (proven red first, see comments) -----------


_EASY_RE = re.compile(r"log_(\d+)\(([+-]?\d+)x ([+-]) (\d+)\) = (\d+)\b")


def test_easy_argument_never_bare_x_a_and_c_both_nontrivial():
    """Guard proven red: relaxing `_A_CHOICES`/`_NONZERO_SMALL` to include 1
    and 0 respectively lets a==1 or c==0 through, colliding with the shipping
    neighbours `senior-define-logarithms-as-indices-a-x-b-is`/
    `senior-define-the-natural-logarithm-ln-x-log-ex` (log_b(x) = k)."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="easy")
        m = _EASY_RE.search(q.text)
        assert m, q.text
        a = int(m.group(2))
        c_val = int(m.group(4)) * (1 if m.group(3) == "+" else -1)
        assert a != 1 and a != 0, q.text
        assert c_val != 0, q.text


def test_medium_k_reaches_zero_and_negative():
    """Guard proven red: an early draft only ever isolated to a positive k,
    which never exercises "b^k is 1 or a fraction" -- the medium-tier
    decision the contract names explicitly. `k` is not printed directly (the
    equation is not isolated), so the source pool itself is asserted here;
    the oracle file independently re-derives k from the printed m/n/p and
    confirms it does reach <= 0 over its own seed sweep."""
    from mathgen.generators.pv0b_log_equation_build import _MEDIUM_K

    assert min(_MEDIUM_K) < 0 and 0 in _MEDIUM_K


_LOG_ARG_RE = re.compile(r"log_\d+\(([^)]*)\)")


def test_hard_uses_a_second_purely_numeric_log_term():
    """Guard proven red: an early draft's hard tier could draw the same
    `_arg_text` helper for the second log too, which would carry an "x" --
    collapsing hard into the same shape as exam."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard")
        args = _LOG_ARG_RE.findall(q.text)
        assert len(args) == 2, q.text
        assert "x" in args[0] and "x" not in args[1], q.text


def test_exam_second_log_carries_x_and_both_arguments_positive():
    """Guard proven red: swap the tier's own `arg2_text` for a bare digit
    (hard-tier shape) and this test fails -- exam's headline decision is
    that BOTH logs' arguments contain the unknown."""
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam")
        args = _LOG_ARG_RE.findall(q.text)
        assert len(args) == 2, q.text
        assert "x" in args[0] and "x" in args[1], q.text


def test_independent_check_fires_on_corrupted_value():
    """The generator's `check()` calls raise `VerificationError` on a
    corrupted algebraic identity -- exercised directly against `mathgen.core.check`."""
    from mathgen.core import check as _check

    with pytest.raises(VerificationError):
        _check(abs(math.log(float(Fraction(2) ** 3), 2) - 5) < 1e-9, "corrupted k must fail")
    _check(abs(math.log(float(Fraction(2) ** 3), 2) - 3) < 1e-9, "correct k must not raise")


# --- skill loads through mathgen.skills.load_skills -----------------------------


def test_skill_loads_via_api_skills():
    ids = {s.id for s in _skills}
    assert _SKILL_ID in ids


def test_skill_properties():
    skill = next(s for s in _skills if s.id == _SKILL_ID)
    assert skill.status == "active"
    assert skill.placement_eligible is False
    assert skill.answer == "fraction"
    assert skill.domain == "functions-sequences"
    assert "function-notation" in skill.prerequisites
    difficulties = {src.difficulty for src in skill.sources}
    assert difficulties == {"easy", "medium", "hard", "exam"}
    exam_src = next(src for src in skill.sources if src.difficulty == "exam")
    assert exam_src.stages == ["exam"]


@pytest.mark.parametrize("difficulty", TIERS)
def test_key_denominator_at_most_12_over_200_seeds(difficulty):
    """Proven red on the pre-fix generator: medium and exam both failed this over 200 seeds before the fix
    (a=9,k=-2,b=9 gave denominator 405; the exam linear-system solve gave denominators in the thousands)."""
    bad = []
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        x = Fraction(q.answer_text)
        if x.denominator > 12:
            bad.append((seed, q.answer_text))
    assert not bad, f"{difficulty}: {len(bad)}/200 keys have denominator > 12, e.g. {bad[:5]}"


@pytest.mark.parametrize("difficulty", TIERS)
def test_no_distractor_denominator_over_24_or_magnitude_over_1000(difficulty):
    """not values a learner would ever produce. Proven red on the pre-fix
    generator: exam's `quotient-inverted`/`argument-negative` candidates
    routinely exceeded both bounds before `_finalize` dropped them."""
    bad = []
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        for d in q.distractors:
            v = Fraction(d)
            if v.denominator > 24 or abs(v) > 1000:
                bad.append((seed, d))
    assert not bad, f"{difficulty}: {len(bad)} distractors exceed the bound, e.g. {bad[:5]}"
