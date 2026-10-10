"""The `solve-quadratics-factorable` exam source -- non-monic `a*x^2 + b*x + c = 0`, `a` in 2..6, both roots
integer, rendered as `"Solve... = 0"` or, for roughly half the seeds, `"0 =..."` (an exam-style "incl.
written as 0 =..." requirement).

`mathgen.generators.algebra` is NOT a `_build` module; `quadratic` is not a build topic, so a generic
text-defect or prompt-variety check over build topics would not reach the `factor_nonmonic` kind on its own.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.core import Question, VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "quadratic"
KIND = "factor_nonmonic"
_SEEDS = range(300)  # generous sweep; this generator is cheap (no sympy solve loop)


def _make(seed, difficulty="exam"):
    return make(TOPIC, seed=seed, difficulty=difficulty, kind=KIND)


# -- by-hand text-quality, variety and distractor checks ----


def test_no_bad_text_patterns_over_60_seeds():
    for seed in range(60):
        q = _make(seed)
        defects = find_text_defects(q)
        assert defects == [], f"seed={seed}: {defects} ({q.text!r})"


def test_prompt_variety():
    """>= 3 distinct prompt_shape() outputs over 60 seeds, for the exam
    source alone."""
    shapes = {prompt_shape(_make(s)) for s in range(60)}
    assert len(shapes) >= 3, shapes


def test_exam_source_adds_shapes_over_hard_tier():
    """The exam source's shapes are not a subset of the (monic) hard tier's
    shapes -- a genuine escalation in kind, not merely bigger numbers."""
    hard_shapes = {prompt_shape(make(TOPIC, seed=s, difficulty="hard")) for s in range(60)}
    exam_shapes = {prompt_shape(_make(s)) for s in range(60)}
    new_shapes = exam_shapes - hard_shapes
    assert len(new_shapes) >= 3, (exam_shapes, hard_shapes)


def test_determinism_same_seed_is_byte_identical():
    for seed in (0, 1, 2, 3, 100, 12345):
        first = _make(seed).to_json()
        second = _make(seed).to_json()
        assert first == second


def test_difficulty_exam_preset_is_registered():
    """`make("quadratic", difficulty="exam")` must not raise -- `quadratic`
    has other named presets (easy/medium/hard), so `mathgen.core.make`
    requires "exam" to be a registered preset name too, even though the real
    exam-tier shape comes from the skill YAML's own `params: {kind:
    factor_nonmonic}`, not from this preset's own `root_hi`."""
    make(TOPIC, seed=0, difficulty="exam")  # kind stays default "factor"; must not raise


def test_a_is_in_2_to_6_and_roots_are_integers():
    for seed in _SEEDS:
        q = _make(seed)
        assert isinstance(q.answer, list)
        assert all(isinstance(r, int) for r in q.answer)
        a, _b, _c = _parse_abc(q.text)
        assert 2 <= a <= 6, (seed, a, q.text)


def test_split_between_flipped_and_unflipped_forms():
    """Roughly half the seeds render '0 = ...' and half 'Solve ... = 0' --
    a genuine token-order structural difference, chosen by `rng.random() <
    0.5`. Reports the actually-observed split (not merely "both occur")."""
    flipped = sum(1 for s in _SEEDS if _make(s).text.startswith("0 = "))
    unflipped = len(list(_SEEDS)) - flipped
    # Both forms must occur, and neither may dominate to the point the other
    # is a rounding artefact -- a generous band around 50/50 over 300 seeds.
    assert flipped > 0 and unflipped > 0
    assert 0.35 <= flipped / len(list(_SEEDS)) <= 0.65, (flipped, unflipped)


_VALUE_RE = re.compile(r"-?\d+(?:/\d+)?")


def _parse_root_set(text: str) -> frozenset:
    return frozenset(Fraction(v) for v in _VALUE_RE.findall(text))


def test_third_misconception_fires_on_some_seeds():
    """The non-monic-specific distractor (ignores `a` entirely) is only
    offered when that wrong quadratic has a real, rational root pair
    (perfect-square discriminant) -- conditional, like
    `_quadratic_formula`'s own `Dwrong` candidate. `len(distractors) == 3`
    is NOT a valid proxy for "the slip fired": with no slip there are
    already 3 candidates (both-flip, flip-one-a, flip-one-b) unless a
    symmetric/zero root collides one of them with the answer, so that count
    conflates two unrelated things. Instead, per seed, identify any
    distractor whose VALUE SET is not one of the three sign-flip
    possibilities -- that is the slip candidate, and only the slip."""
    fired = 0
    for seed in _SEEDS:
        q = _make(seed)
        u, v = q.answer[0], q.answer[-1]
        sign_flip_sets = {
            frozenset({-u, -v}),
            frozenset({-u, v}),
            frozenset({u, -v}),
        }
        if any(_parse_root_set(d) not in sign_flip_sets for d in q.distractors):
            fired += 1
    assert fired > 0, "the third (non-monic) misconception never fired in 300 seeds"


# -- the independent oracle (never imports the generator) -------------------
#
# Parses `question.text` alone (either "Solve <lhs> = 0" or "0 = <lhs>") and
# recomputes the roots from scratch with the quadratic formula over
# `fractions.Fraction`, then compares to `question.answer`. Shares no code,
# helper or constant with `mathgen.generators.algebra` -- this file's own
# import list (no `algebra` import here) is the proof.

_TOKEN_RE = re.compile(r"([+-]?)\s*(\d+x\^2|\d*x(?!\^)|\d+)")


def _extract_lhs(question_text: str) -> str:
    if question_text.startswith("0 = "):
        return question_text[len("0 = ") :]
    assert question_text.startswith("Solve ") and question_text.endswith(" = 0"), question_text
    return question_text[len("Solve ") : -len(" = 0")]


def _parse_abc(question_text: str) -> tuple:
    """(a, b, c) parsed from the rendered `a*x^2 + b*x + c` prompt text
    alone. Shared by the oracle below and the `a in 2..6` contract test."""
    lhs = _extract_lhs(question_text)
    a = b = c = 0
    seen_quadratic_term = False
    for sign, token in _TOKEN_RE.findall(lhs):
        mult = -1 if sign == "-" else 1
        if token.endswith("x^2"):
            digits = token[: -len("x^2")]
            a = mult * int(digits)
            seen_quadratic_term = True
        elif token.endswith("x"):
            digits = token[:-1]
            mag = int(digits) if digits else 1
            b = mult * mag
        else:
            c = mult * int(token)
    assert seen_quadratic_term, f"oracle: no x^2 term found in {lhs!r}"
    assert a != 0
    return a, b, c


def _independent_roots(question_text: str) -> list:
    """Re-derive the two roots from the rendered prompt alone -- the oracle.
    Never reads `question.answer`/`question.working`."""
    a, b, c = _parse_abc(question_text)
    disc = b * b - 4 * a * c
    assert disc >= 0, f"oracle: negative discriminant for {question_text!r}"
    sqrt_disc = math.isqrt(disc)
    assert sqrt_disc * sqrt_disc == disc, f"oracle: non-perfect-square discriminant for {question_text!r}"
    r_lo = Fraction(-b - sqrt_disc, 2 * a)
    r_hi = Fraction(-b + sqrt_disc, 2 * a)
    roots = sorted({r_lo, r_hi})
    for r in roots:
        assert r.denominator == 1, f"oracle: non-integer root {r} for {question_text!r}"
    return [int(r) for r in roots]


def test_independent_oracle_matches_answer_over_300_seeds():
    mismatches = []
    for seed in _SEEDS:
        q = _make(seed)
        oracle_roots = _independent_roots(q.text)
        if oracle_roots != q.answer:
            mismatches.append((seed, q.text, oracle_roots, q.answer))
    assert mismatches == [], mismatches


# -- the independent-verification-firing test --------------------------------
#
# This test (only) imports the generator module directly, to monkeypatch the
# expression-construction step into returning a corrupted value -- proving
# the GENERATOR's own in-body `check()` (run on every real call to `make()`)
# actually fires, not merely that this test file's oracle would catch it.

import mathgen.generators.algebra as _algebra  # noqa: E402


def test_corrupted_expression_raises_verification_error(monkeypatch):
    original_nonmonic_expr = _algebra._nonmonic_expr

    def _corrupted_nonmonic_expr(a, r1, r2):
        return original_nonmonic_expr(a, r1, r2) + 1  # wrong-by-1 constant term

    monkeypatch.setattr(_algebra, "_nonmonic_expr", _corrupted_nonmonic_expr)
    with pytest.raises(VerificationError):
        make(TOPIC, seed=1, difficulty="exam", kind=KIND)


def test_corrupted_expression_is_not_a_false_positive_when_uncorrupted():
    """Sanity check on the test above: the same seed, unpatched, must NOT
    raise -- otherwise the firing test would be vacuous."""
    make(TOPIC, seed=1, difficulty="exam", kind=KIND)  # no exception


# -- the real grader (mathgen.grading.GRADERS["list"]) ---------------------------
#
# Every canonical answer must be accepted and every distractor -- and every
# deliberately-wrong test value -- must be rejected, checked unconditionally
# over every seed, not just the ones that happen to be spot-checked.

from mathgen.grading.typed import GRADERS  # noqa: E402

_LIST_GRADER = GRADERS["list"]


def test_grader_accepts_canonical_and_rejects_every_distractor_over_300_seeds():
    bad = []
    for seed in _SEEDS:
        q = _make(seed)
        canonical_result = _LIST_GRADER(q, q.answer_text)
        if not canonical_result.correct:
            bad.append(("canonical-rejected", seed, q.answer_text))
        for d in q.distractors:
            distractor_result = _LIST_GRADER(q, d)
            if distractor_result.correct:
                bad.append(("distractor-accepted", seed, d, q.answer_text))
    assert bad == [], bad


def test_grader_rejects_deliberately_wrong_values():
    """A learner-plausible but wrong answer (roots off by one, roots
    reversed-sign, an empty answer) must never grade correct, checked
    unconditionally against a fixed real question (seed=3: roots [-4, 1] --
    none of the five candidates below coincide with that value set, verified
    directly, so this is not a vacuous check for this fixture)."""
    q = _make(3)
    assert q.answer == [-4, 1]  # pins the fixture this test's candidates were chosen against
    r1, r2 = q.answer[0], q.answer[-1]
    wrong_candidates = [
        f"x = {r1 + 1} or x = {r2}",
        f"x = {r1} or x = {r2 + 1}",
        f"x = {-r1} or x = {-r2}",
        "",
        "x = 0 or x = 0",
    ]
    for wrong in wrong_candidates:
        assert _parse_root_set(wrong) != frozenset(q.answer), (wrong, q.answer)
        result = _LIST_GRADER(q, wrong)
        assert not result.correct, (wrong, q.answer_text)


# -- Question-shaped smoke check for the grading contract -------------------
#
# Confirms the grader is exercised the way it is called in practice: "called
# with a question.answer_text-shaped string" -- built from a `mathgen.make`
# question, not a hand-rolled `Question` (the real shape a learner's session
# actually produces).


def test_question_answer_is_list_of_two_distinct_ints_or_one():
    for seed in range(30):
        q = _make(seed)
        assert isinstance(q, Question)
        assert q.answer == sorted(set(q.answer))  # matches the sorted-set construction
