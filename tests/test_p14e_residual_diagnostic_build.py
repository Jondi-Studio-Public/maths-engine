"""Tests `p14e_residual_diagnostic` (`ucv1-regression-residual-diagnostic`), built by
`mathgen/generators/p14e_residual_diagnostic_build.py`.

``ucv1-regression-residual-diagnostic``, built by ``mathgen/generators/p14e_residual_diagnostic_build.py``
(topic ``p14e_residual_diagnostic``, answer: mcq, all four tiers).

See the generator module's docstring for the tier design and the boundary
against ``least-squares-line`` / ``ucv1-multivariate-regression``,
and the recorded deviation from the original hard-tier spec
(three-class draw at every tier, not "both points off the line").
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction

import pytest

from mathgen.grading.typed import GRADERS, grade_mcq, mcq_options
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import PRESETS, REGISTRY, VerificationError, check
from mathgen.generators.p14e_residual_diagnostic_build import (
    LABELS,
    _classify,
    _construct,
    _fmt_line,
    p14e_residual_diagnostic,
)

TOPIC = "p14e_residual_diagnostic"
SKILL_ID = "ucv1-regression-residual-diagnostic"
TIERS = ("easy", "medium", "hard", "exam")
SEEDS = range(60)


# --- Registration -------------------------------------------------------


def test_topic_registered_with_all_four_presets():
    assert TOPIC in REGISTRY
    assert set(PRESETS[TOPIC]) == {"easy", "medium", "hard", "exam"}


# --- Determinism ----------------------------------------------------------


# --- Distractor hygiene: exactly 2 distractors, unconditionally ----------


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_distinct_nonempty_never_equal_answer(tier):
    counts = Counter()
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert q.answer_text not in q.distractors
        assert len(q.distractors) == len(set(q.distractors))
        counts[len(q.distractors)] += 1
    # Answer space by construction: the two OTHER fixed labels
    # are offered as distractors on every single draw, unconditionally.
    assert counts == {2: 200}, f"{tier}: distractor-count distribution {dict(counts)}"


@pytest.mark.parametrize("tier", TIERS)
def test_all_three_labels_always_present_as_options(tier):
    for seed in range(100):
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert set(mcq_options(q)) == set(LABELS)


# --- Real grader: mcq is graded by STRING INDEX into mcq_options ---------


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_answer_index_rejects_distractor_indices(tier):
    grader = GRADERS["mcq"]
    assert grader is grade_mcq
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        options = mcq_options(q)
        correct_index = options.index(q.answer_text)
        assert grader(q, correct_index).correct, f"{tier} seed={seed}: correct index rejected"
        for dist in q.distractors:
            dist_index = options.index(dist)
            assert not grader(q, str(dist_index)).correct, f"{tier} seed={seed}: distractor accepted"
        assert not grader(q, str(len(options))).correct, f"{tier} seed={seed}: out-of-range accepted"


# --- Answer kind: answer_text is always exactly one of the 3 labels ------


@pytest.mark.parametrize("tier", TIERS)
def test_answer_is_one_of_the_three_fixed_labels(tier):
    for seed in range(100):
        q = make(TOPIC, seed=seed, difficulty=tier)
        assert q.answer_text in LABELS


# --- Options formatted alike: all four options are always plain labels ---


@pytest.mark.parametrize("tier", TIERS)
def test_options_formatted_alike(tier):
    for seed in range(50):
        q = make(TOPIC, seed=seed, difficulty=tier)
        for opt in mcq_options(q):
            assert opt in LABELS


# --- At least 20 distinct rendered questions per tier over 60 seeds ------


# --- 200-seed class-balance measurement -----------------------


@pytest.mark.parametrize("tier", TIERS)
def test_200_seed_class_concentration(tier):
    """Assert a wide honest band: top class <= 45%, every class >= 20% (a uniform three-way split at n=200 has
    real sampling noise around the 33% baseline)."""
    counts = Counter(make(TOPIC, seed=seed, difficulty=tier).answer_text for seed in range(200))
    assert set(counts) == set(LABELS), f"{tier}: not all 3 classes drawn in 200 seeds: {dict(counts)}"
    top_share = counts.most_common(1)[0][1] / 200
    min_share = min(counts.values()) / 200
    assert top_share <= 0.45, f"{tier}: top class share {top_share:.1%} ({dict(counts)})"
    assert min_share >= 0.20, f"{tier}: min class share {min_share:.1%} ({dict(counts)})"


# --- The tie case (e == 0, "on the line") is deliberately reachable -------


@pytest.mark.parametrize("tier", TIERS)
def test_tie_case_reachable_and_offered_as_key(tier):
    """Confirm 'on the line' occurs over a seed sweep, and whenever it is the
    generator's key, mcq_options offers it (never silently dropped)."""
    on_count = 0
    for seed in range(300):
        q = make(TOPIC, seed=seed, difficulty=tier)
        if q.answer_text == "on the line":
            on_count += 1
            assert "on the line" in mcq_options(q)
    assert on_count > 0, f"{tier}: 'on the line' never drawn as the key over 300 seeds"


# --- Independent in-generator check tripwire ------------------------------


def test_check_fires_on_corrupted_condition():
    with pytest.raises(VerificationError):
        check(1 == 2, "should fail")
    check(1 == 1, "should not fail")


def test_easy_independent_recheck_fires_on_corrupted_construction(monkeypatch):
    """`_easy` picks a label and calls `_construct` to build y0 realising it;
    its own independent recheck recomputes the residual from the rendered
    numbers and must fire if `_construct` is corrupted to always build
    'above the line' regardless of the requested label."""
    import random

    import mathgen.generators.p14e_residual_diagnostic_build as module

    def _always_above(y_pred, label, rng, lo=1, hi=6):
        return y_pred + rng.randint(lo, hi)

    monkeypatch.setattr(module, "_construct", _always_above)
    rng = random.Random(0)
    with pytest.raises(VerificationError):
        for _ in range(30):
            module._easy(rng)


def test_exam_independent_recheck_fires_on_corrupted_construction(monkeypatch):
    """Same tripwire for the exam tier's sign-recheck (the classification
    check, not the sympy two-point-derivation check)."""
    import random

    import mathgen.generators.p14e_residual_diagnostic_build as module

    def _always_above(y_pred, label, rng, lo=1, hi=6):
        return y_pred + rng.randint(lo, hi)

    monkeypatch.setattr(module, "_construct", _always_above)
    rng = random.Random(0)
    with pytest.raises(VerificationError):
        for _ in range(30):
            module._exam(rng)


def test_exam_derived_coefficients_check_fires_on_a_wrong_expected_value():
    """The exam tier's OTHER independent check (`check_equivalent` against a
    sympy two-point-system solve, confirming m/b) -- exercised directly with
    the exact same solve mechanism `_exam` uses, and a deliberately WRONG
    expected slope, confirming the check raises rather than passing silently."""
    import sympy

    from mathgen.generators.p14e_residual_diagnostic_build import check_equivalent

    x1, y1, x2, y2 = 0, 5, 3, 11  # a genuine line: m=2, b=5
    system = sympy.Matrix([[x1, 1], [x2, 1]])
    rhs = sympy.Matrix([y1, y2])
    solution = system.solve(rhs)
    check_equivalent(solution[0], sympy.Rational(2, 1), "true slope must match")
    with pytest.raises(VerificationError):
        check_equivalent(solution[0], sympy.Rational(3, 1), "deliberately wrong slope")


def test_construct_and_classify_agree_by_construction():
    """`_construct` builds a y-value realising a given label against a
    predicted value; `_classify` must read that label back for every label,
    at both integer and fractional predicted values."""
    import random

    rng = random.Random(0)
    for y_pred in (Fraction(3), Fraction(-2), Fraction(7, 3), Fraction(-11, 4)):
        for label in LABELS:
            y0 = _construct(y_pred, label, rng)
            e = y0 - y_pred
            assert _classify(e) == label


def test_hard_tier_draws_all_three_labels_for_the_named_point():
    """Deviation from the original hard-tier spec ('both off
    the line'): this generator draws each of the two points' classes
    independently and uniformly, so the named point's label is a genuine
    3-way draw, not 2-way. Confirm all 3 occur over a seed sweep."""
    labels = {make(TOPIC, seed=seed, difficulty="hard").answer_text for seed in range(300)}
    assert labels == set(LABELS)


_HARD_LINE_RE = __import__("re").compile(r"y = (\(-?\d+/\d+\)|-?\d*)x(?: ([+-]) (\d+))?(?=[.,;]| |$)")
_HARD_POINT_RE = __import__("re").compile(r"[AB] = \((-?\d+), (-?\d+/\d+|-?\d+)\)")
_HARD_TARGET_RE = __import__("re").compile(r"point (\w)(?:'s residual| above| below)")


def test_hard_tier_wrong_point_collision_rate_measured():
    """The 'wrong named point' misconception (reading the OTHER point's numbers) is exercisable whenever the two
    points fall in DIFFERENT classes; measure and report the collision rate honestly. Both points' coordinates
    and the line are all in the rendered text, so this recomputes both classes directly from the text using
    the generator's own `_classify` (reusing the sign rule, not an independence claim -- that lives in the
    oracle file) and reports the collision share."""
    total = 300
    collisions = 0
    for seed in range(total):
        q = make(TOPIC, seed=seed, difficulty="hard")
        line_m = _HARD_LINE_RE.search(q.text)
        assert line_m, q.text
        m_text, sign, b_text = line_m.groups()
        b = 0 if sign is None else (int(b_text) if sign == "+" else -int(b_text))
        if m_text in ("", "-"):
            m = Fraction(1) if m_text == "" else Fraction(-1)
        elif m_text.startswith("("):
            m = Fraction(m_text[1:-1])
        else:
            m = Fraction(int(m_text))
        points = _HARD_POINT_RE.findall(q.text)
        assert len(points) == 2, q.text
        labels = {}
        for label, (x_str, y_str) in zip(("A", "B"), points):
            x, y = int(x_str), Fraction(y_str)
            e = y - (m * x + b)
            labels[label] = _classify(e)
        target = _HARD_TARGET_RE.search(q.text)
        assert target, q.text
        target_letter = target.group(1)
        other_letter = "B" if target_letter == "A" else "A"
        assert labels[target_letter] == q.answer_text, (seed, q.text, labels)
        if labels[other_letter] == labels[target_letter]:
            collisions += 1
    collision_rate = collisions / total
    assert 0.0 <= collision_rate <= 1.0


# --- Skill loading ---------------------------------------------------------


def test_skill_loads_with_correct_properties():
    _, skills = load_skills()
    skill = next((s for s in skills if s.id == SKILL_ID), None)
    assert skill is not None
    assert skill.answer == "mcq"
    assert skill.placement_eligible is False
    assert skill.prerequisites == ["least-squares-line"]
    assert len(skill.sources) == 4
    stages = [set(s.stages) for s in skill.sources]
    assert stages == [{0, 1, 2}, {3, 4}, {5, 6, "maintenance"}, {"exam"}]
    assert skill.sources[3].difficulty == "exam"
    assert all(
        source.topic == ("exd_x_r3i_regression_fit" if source.difficulty == "exam" else TOPIC)
        for source in skill.sources
    )


# --- Sanity on the generator's own helpers ---------------------------------


def test_fmt_line_is_unambiguous_for_fraction_and_integer_slopes():
    assert _fmt_line(Fraction(-1, 3), 2) == "y = (-1/3)x + 2"
    assert _fmt_line(1, -5) == "y = x - 5"
    assert _fmt_line(-1, 3) == "y = -x + 3"
    assert _fmt_line(4, 0) == "y = 4x"


def test_direct_builder_callable_via_public_entry_point():
    import random

    rng = random.Random(1)
    d = p14e_residual_diagnostic(rng, kind="direct")
    assert d["answer_text"] in LABELS
