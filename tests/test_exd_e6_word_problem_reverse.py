"""A NEW `kind="linear_eq_reverse"` on the `word_problem` topic (`mathgen/generators/foundations_algebra.py`).

The existing `kind="linear_eq"` shape ("multiply, then add/subtract" ->
`ax + b = c`) was the ONLY shape every one of the skill's three sources used,
at every difficulty -- range-only by construction. The fix does not touch
that shape's body at all; it adds a second branch, `linear_eq_reverse`
("increased by b, then multiplied by a" -> `a(x + b) = c`), whose narrative
order matches the construction order but whose UNDO order is the reverse of
`linear_eq`'s (divide by `a` first, then subtract `b` -- backwards from
`linear_eq`'s subtract-then-divide). The HARD source is repointed to the new
kind; easy/medium keep `kind: linear_eq` untouched.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "word_problem"


def _make_reverse(seed, difficulty="hard"):
    return make(TOPIC, seed=seed, difficulty=difficulty, kind="linear_eq_reverse")


def _make_linear(seed, difficulty="hard"):
    return make(TOPIC, seed=seed, difficulty=difficulty, kind="linear_eq")


# Text quality, prompt variety, determinism
# ---------------------------------------------------------------------------


def test_reverse_no_bad_text_patterns_over_60_seeds():
    for seed in range(60):
        q = _make_reverse(seed)
        defects = find_text_defects(q)
        assert defects == [], f"seed={seed}: {defects} ({q.text!r})"


def test_reverse_prompt_variety():
    """>= 3 distinct prompt_shape outputs over 60 seeds for the new hard kind alone."""
    shapes = {prompt_shape(_make_reverse(s)) for s in range(60)}
    assert len(shapes) >= 3, shapes


def test_reverse_hard_adds_shapes_over_easy_and_medium():
    """The new hard kind's shapes are not a subset of the union of the
    UNCHANGED easy/medium (`kind: linear_eq`) tiers' shapes -- a genuine
    escalation in kind, not merely a bigger number."""
    lower_shapes = {prompt_shape(make(TOPIC, seed=s, difficulty="easy", kind="linear_eq")) for s in range(60)}
    lower_shapes |= {prompt_shape(make(TOPIC, seed=s, difficulty="medium", kind="linear_eq")) for s in range(60)}
    hard_shapes = {prompt_shape(_make_reverse(s)) for s in range(60)}
    new_shapes = hard_shapes - lower_shapes
    assert len(new_shapes) >= 1, (hard_shapes, lower_shapes)


def test_reverse_determinism_same_seed_is_byte_identical():
    for seed in (0, 1, 2, 3, 100, 12345):
        first = _make_reverse(seed).to_json()
        second = _make_reverse(seed).to_json()
        assert first == second


# ---------------------------------------------------------------------------
# Independent oracle: re-derive x0 from the rendered prompt text ALONE, with
# Fraction arithmetic, never importing mathgen.generators.foundations_algebra.
# ---------------------------------------------------------------------------

_REVERSE_RES = [
    re.compile(
        r"^A number is increased by (\d+), then the result is multiplied by (\d+), "
        r"giving (-?\d+)\. What was the number\?$"
    ),
    re.compile(
        r"^Take a number, increase it by (\d+), then multiply the result by (\d+)\. "
        r"The answer is (-?\d+)\. What was the number\?$"
    ),
    re.compile(
        r"^First, a number is increased by (\d+)\. Multiplying that result by (\d+) "
        r"gives (-?\d+)\. What was the number\?$"
    ),
]


def _match_reverse(question_text: str):
    for pattern in _REVERSE_RES:
        m = pattern.match(question_text)
        if m is not None:
            return m
    return None


def _independent_reverse_solution(question_text: str) -> Fraction:
    m = _match_reverse(question_text)
    assert m is not None, f"oracle: could not parse {question_text!r}"
    b, a, c = (int(v) for v in m.groups())
    # a(x + b) = c  =>  x = c/a - b
    x0 = Fraction(c, a) - b
    assert x0.denominator == 1, f"oracle: non-integer solution {x0} for {question_text!r}"
    return x0


def test_reverse_answer_matches_independent_solve_over_600_seeds():
    mismatches = []
    for seed in range(600):
        q = _make_reverse(seed)
        oracle_x0 = _independent_reverse_solution(q.text)
        if oracle_x0 != q.answer:
            mismatches.append((seed, q.text, oracle_x0, q.answer))
    assert mismatches == [], mismatches


def test_reverse_b_is_always_non_negative_over_600_seeds():
    """Documents a design choice: `b` is drawn
    non-negative only, a single "increased by" phrasing (not `linear_eq`'s
    bidirectional add/subtract), so the hard tier isolates the reverse-order
    trap rather than compounding it with a sign trap."""
    for seed in range(600):
        q = _make_reverse(seed)
        m = _match_reverse(q.text)
        assert m is not None, q.text
        b = int(m.group(1))
        assert b >= 1, (seed, q.text)


import mathgen.generators.foundations_algebra as _fa  # noqa: E402


def test_reverse_corrupted_solve_raises_verification_error(monkeypatch):
    original_solve = _fa.solve

    def _wrong_solve(*args, **kwargs):
        result = original_solve(*args, **kwargs)
        return [r + 1 for r in result]

    monkeypatch.setattr(_fa, "solve", _wrong_solve)
    with pytest.raises(VerificationError):
        make(TOPIC, seed=0, difficulty="hard", kind="linear_eq_reverse")


def test_reverse_corrupted_solve_is_not_a_false_positive_when_uncorrupted():
    make(TOPIC, seed=0, difficulty="hard", kind="linear_eq_reverse")  # no exception


def test_reverse_corrupted_check_equivalent_raises_verification_error(monkeypatch):
    """The second, independent route (`check_equivalent(a * (x0 + b), c)`)
    also actually fires on a corrupted construction -- not merely present in
    the source but exercised."""
    original_check_equivalent = _fa.check_equivalent

    def _wrong_check_equivalent(actual, expected, msg="answer verification failed"):
        # Corrupt by comparing against a shifted expected value instead.
        return original_check_equivalent(actual, expected + 1, msg)

    monkeypatch.setattr(_fa, "check_equivalent", _wrong_check_equivalent)
    with pytest.raises(VerificationError):
        make(TOPIC, seed=0, difficulty="hard", kind="linear_eq_reverse")


# ---------------------------------------------------------------------------
# Distractors: distinct, non-empty, never equal to the answer.
# ---------------------------------------------------------------------------


def test_reverse_distractors_non_empty_and_never_equal_answer_over_300_seeds():
    for seed in range(300):
        q = _make_reverse(seed)
        assert len(q.distractors) >= 1, (seed, q.text, q.distractors)
        for d in q.distractors:
            assert d != q.answer_text, (seed, q.text, d)


def test_reverse_distractors_distinct_within_each_question_over_300_seeds():
    for seed in range(300):
        q = _make_reverse(seed)
        assert len(set(q.distractors)) == len(q.distractors), (seed, q.text, q.distractors)


# ---------------------------------------------------------------------------
# The real grader (mathgen.grading.GRADERS["number"])
# ---------------------------------------------------------------------------

from mathgen.grading.typed import GRADERS  # noqa: E402

_NUMBER_GRADER = GRADERS["number"]


def test_reverse_grader_accepts_canonical_and_rejects_every_distractor_over_200_seeds():
    bad = []
    for seed in range(200):
        q = _make_reverse(seed)
        canonical_result = _NUMBER_GRADER(q, q.answer_text)
        if not canonical_result.correct:
            bad.append(("canonical-rejected", seed, q.answer_text))
        for d in q.distractors:
            distractor_result = _NUMBER_GRADER(q, d)
            if distractor_result.correct:
                bad.append(("distractor-accepted", seed, d, q.answer_text))
    assert bad == [], bad


def test_reverse_grader_rejects_deliberately_wrong_values():
    """The three misconception values, computed by hand against a fixed
    seed and checked they are actually wrong before asserting the grader
    rejects them (never a vacuous check).

    seed=0, difficulty=hard: b=8, a=2, c=76, x0=30 (see the render output
    subtract_first = (c - b)/a = 68/2 = 34; forgot_bracket = c - b = 68.
    """
    q = _make_reverse(0)
    x0 = q.answer
    assert x0 == 30
    wrong_candidates = ["46", "34", "68", "0", str(x0 + 1)]
    for wrong in wrong_candidates:
        wrong_val = Fraction(wrong)
        assert wrong_val != x0, (wrong, x0)
        result = _NUMBER_GRADER(q, wrong)
        assert not result.correct, (wrong, q.answer_text)


# Original kind="linear_eq" behaviour untouched
# ---------------------------------------------------------------------------


def test_original_linear_eq_kind_unaffected_over_300_seeds():
    """Confirms the ORIGINAL `kind == "linear_eq"` branch (easy/medium
    sources) still matches its own long-standing shape: "I think of a
    number, multiply it by A, then add/subtract B. The result is C."
    """
    pattern = re.compile(
        r"^I think of a number, multiply it by (\d+), then (add|subtract) (\d+)\. "
        r"The result is (-?\d+)\. What was the number\?$"
    )
    for seed in range(300):
        for difficulty in ("easy", "medium", "hard"):
            q = make(TOPIC, seed=seed, difficulty=difficulty, kind="linear_eq")
            assert pattern.match(q.text), (seed, difficulty, q.text)


# ---------------------------------------------------------------------------
# Skill loads via mathgen.skills.load_skills
# ---------------------------------------------------------------------------


def test_skill_loads_via_api_skills():
    from mathgen import skills as sm

    _domains, skill_list = sm.load_skills()
    ids = {s.id: s for s in skill_list}
    assert "solve-equation-word-problems" in ids
    skill = ids["solve-equation-word-problems"]
    hard_sources = [s for s in skill.sources if s.difficulty == "hard"]
    assert len(hard_sources) == 1
    assert hard_sources[0].params.get("kind") == "linear_eq_reverse"
