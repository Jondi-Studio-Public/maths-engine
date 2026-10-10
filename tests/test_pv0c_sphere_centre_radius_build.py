"""Focused contract tests for: senior-introduce-cartesian-coordinates-for-three-dimensional-space-including
(ACMSM103, ``pv0c_sphere_centre_radius``, ``answer: ordered_list``)."""

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen.grading.typed import GRADERS, grade_ordered_list
from mathgen.skills import load_skills
from mathgen import make
import mathgen.generators.pv0c_sphere_centre_radius_build as gen

SKILL_ID = "senior-introduce-cartesian-coordinates-for-three-dimensional-space-including"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(40)
SWEEP_SEEDS = range(60)

_ITEM_RE = r"-?\d+(?:/\d+)?"
_ANSWER_RE = re.compile(rf"{_ITEM_RE}(?:, {_ITEM_RE}){{3}}")

_domains, _skills = load_skills()
SKILL = next(s for s in _skills if s.id == SKILL_ID)
SOURCES = {src.difficulty: src for src in SKILL.sources}


def _make(difficulty, seed):
    src = SOURCES[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


# ---------------------------------------------------------------------------
# Determinism / distractor hygiene
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_at_least_two_distractors_on_85_percent_of_draws(difficulty):
    n_ge_2 = sum(1 for seed in range(200) if len(_make(difficulty, seed).distractors) >= 2)
    assert n_ge_2 / 200 >= 0.85, f"{difficulty}: only {n_ge_2}/200 draws had >= 2 distractors"


# ---------------------------------------------------------------------------
# Real grader accept/reject
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_accepts_canonical_and_rejects_every_distractor(difficulty):
    grader = GRADERS[SKILL.answer]
    assert grader is grade_ordered_list


@pytest.mark.parametrize("difficulty", TIERS)
def test_real_grader_rejects_a_deliberately_wrong_answer(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        a, b, c, r = (Fraction(v) for v in q.answer)
        wrong = gen._fmt_answer((a + 1000, b, c, r))
        assert not grade_ordered_list(q, wrong).correct, (
            f"seed {seed} {difficulty}: deliberate wrong {wrong!r} not rejected"
        )


# ---------------------------------------------------------------------------
# Answer kind / formatting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_answer_is_four_exact_rationals_lowest_terms(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        assert isinstance(q.answer, list) and len(q.answer) == 4
        for v in q.answer:
            assert type(v) in (int, Fraction), f"{v!r} is {type(v)}, not int/Fraction"
            f = Fraction(v)
            assert f == Fraction(f.numerator, f.denominator)
        assert _ANSWER_RE.fullmatch(q.answer_text), q.answer_text
        assert q.answer_latex and "plain:" not in q.answer_latex


@pytest.mark.parametrize("difficulty", TIERS)
def test_every_distractor_formatted_like_the_answer(difficulty):
    for seed in SEEDS:
        q = _make(difficulty, seed)
        for d in q.distractors:
            assert _ANSWER_RE.fullmatch(d), d


def test_r_is_always_rational_never_a_surd():
    """Every r must be an integer or exact half-integer (grade_ordered_list's
    numeric-item parser accepts "3/2" and "1.5" -- see the grader-spelling
    probe test below -- but never a decimal surd approximation)."""
    for difficulty in TIERS:
        for seed in SWEEP_SEEDS:
            q = _make(difficulty, seed)
            r = Fraction(q.answer[3])
            assert r.denominator in (1, 2), f"{difficulty} seed {seed}: r={r} is not a half-integer"
            assert r >= 1


# ---------------------------------------------------------------------------
# Grader-spelling probe
# ---------------------------------------------------------------------------


def test_grader_spelling_probe_table():
    class _Q:
        pass

    rows = []
    q = _Q()
    q.answer = [Fraction(2), Fraction(-3), Fraction(0), Fraction(4)]
    q.answer_text = "2, -3, 0, 4"
    for typed, expected in [
        ("2, -3, 0, 4", True),
        ("(2, -3, 0, 4)", True),
        ("2,-3,0,4", True),
        ("(2, -3, 0), 4", False),
    ]:
        result = grade_ordered_list(q, typed).correct
        rows.append((typed, result))
        assert result is expected, f"{typed!r}: expected {expected}, got {result}"

    q2 = _Q()
    q2.answer = [Fraction(3, 2), Fraction(-1), Fraction(0), Fraction(5, 2)]
    q2.answer_text = "3/2, -1, 0, 5/2"
    for typed, expected in [
        ("3/2, -1, 0, 5/2", True),
        ("1.5, -1, 0, 2.5", True),
    ]:
        result = grade_ordered_list(q2, typed).correct
        rows.append((typed, result))
        assert result is expected, f"{typed!r}: expected {expected}, got {result}"

    print("grader spelling probe:", rows)


# ---------------------------------------------------------------------------
# Distinct-question / concentration bars
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("difficulty", TIERS)
def test_200_seed_answer_concentration_under_25_percent(difficulty):
    counts = Counter(_make(difficulty, seed).answer_text for seed in range(200))
    top_share = max(counts.values()) / 200
    assert top_share <= 0.25, (
        f"{difficulty}: top answer {counts.most_common(1)[0][0]!r} is {top_share:.1%} of 200 draws"
    )
    r_counts = Counter(str(Fraction(_make(difficulty, seed).answer[3])) for seed in range(200))
    r_top_share = max(r_counts.values()) / 200
    print(f"{difficulty}: whole-tuple top share {top_share:.1%}, r-alone top share {r_top_share:.1%}")


# ---------------------------------------------------------------------------
# Tier-specific shape assertions
# ---------------------------------------------------------------------------


def test_easy_prints_centre_radius_form_directly():
    for seed in SEEDS:
        q = _make("easy", seed)
        assert "^2" in q.text
        assert "= 0" not in q.text  # never the expanded/moved-constant form


def test_medium_needs_normalising_first():
    variants_seen = set()
    for seed in SWEEP_SEEDS:
        q = _make("medium", seed)
        if "= 0" in q.text:
            variants_seen.add("moved")
        elif re.search(r"\d\(x", q.text):
            variants_seen.add("scaled")
        else:
            variants_seen.add("split")
    assert len(variants_seen) == 3, f"only saw variants {variants_seen}"


def test_hard_is_expanded_form_with_even_linear_coefficients_and_integer_answer():
    for seed in SEEDS:
        q = _make("hard", seed)
        assert "x^2 + y^2 + z^2" in q.text
        for v in q.answer:
            assert Fraction(v).denominator == 1


def test_exam_has_a_leading_coefficient_and_sometimes_a_half_integer_answer():
    saw_half = False
    saw_scale_only = False
    for seed in SWEEP_SEEDS:
        q = _make("exam", seed)
        assert re.search(r"\b[2-4]x\^2", q.text), q.text
        if any(Fraction(v).denominator == 2 for v in q.answer):
            saw_half = True
        else:
            saw_scale_only = True
    assert saw_half, "exam tier never produced a half-integer answer over 60 seeds"
    assert saw_scale_only, "exam tier never produced an all-integer answer over 60 seeds"


def test_exam_g_never_zero_and_hard_g_never_zero():
    for difficulty in ("hard", "exam"):
        for seed in SWEEP_SEEDS:
            q = _make(difficulty, seed)
            a, b, c, r = (Fraction(v) for v in q.answer)
            assert a * a + b * b + c * c - r * r != 0


# Guards, proven red first
# ---------------------------------------------------------------------------


def test_guard_expansion_mismatch_fires():
    from mathgen.core import VerificationError

    a, b, c, r = Fraction(1), Fraction(2), Fraction(3), Fraction(4)
    D, E, F, G = gen._defg(a, b, c, r)
    with pytest.raises(VerificationError):
        gen._verify_expansion(a, b, c, r, D, E, F, G + 1)  # corrupt G


def test_guard_hard_rejects_g_equals_zero():
    a = b = c = Fraction(0)
    r = Fraction(0)  # a^2+b^2+c^2 == r^2 == 0 -> G == 0
    assert gen._guard_hard(a, b, c, r) is False


def test_pick_rejects_a_distractor_equal_to_the_answer():
    """`_pick`'s own tuple-equality pre-filter (`cand == answer: continue`)
    would silently swallow an EXACT tuple duplicate before `check_distinct`
    ever runs, so this exercises `check_distinct` directly: a candidate
    passed as a `list` with the SAME values as the `answer` tuple is not
    `==` to it (Python never equates a `list` and a `tuple`), so it slips
    past the pre-filter and reaches `check_distinct`, which must still catch
    it via `pairs_equal`'s element-wise sequence comparison."""
    from mathgen.core import VerificationError

    answer = (Fraction(1), Fraction(2), Fraction(3), Fraction(4))
    same_values_as_list = [Fraction(1), Fraction(2), Fraction(3), Fraction(4)]
    with pytest.raises(VerificationError):
        gen._pick(answer, [("bad", same_values_as_list)])


# ---------------------------------------------------------------------------
# Skill load
# ---------------------------------------------------------------------------


def test_skill_loads_through_api_with_correct_properties():
    assert SKILL.domain == "vectors"
    assert SKILL.status == "active"
    assert SKILL.placement_eligible is False
    assert SKILL.answer == "ordered_list"
    assert SKILL.prerequisites == []
    assert {src.difficulty for src in SKILL.sources} == {"easy", "medium", "hard", "exam"}
    assert SOURCES["exam"].stages == ["exam"]
    assert SOURCES["hard"].stages == [6, "maintenance"]
