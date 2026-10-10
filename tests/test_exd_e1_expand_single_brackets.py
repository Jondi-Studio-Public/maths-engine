"""A range-only fix for `expand-single-brackets`'s hard tier.

The fix: a new `kind == "single_trinomial"` branch on `expand_bracket` (`a(bx + cy + d)`, three terms, two
variables, `a` drawn negative-or-fractional), wired to the skill's `hard` source only.

Sign-masking trap: a leading sign is masked together with its digit run, so a bare negative multiplier alone
would NOT move `prompt_shape`. The new branch's variety comes from two genuinely structural sources instead:
the extra term/variable (always present, so every `single_trinomial` render already differs in SHAPE from a
two-term `kind == "single"` render), and the literal "/" a fractional `a` leaves in the masked text (a "/" is
never swallowed by `_DIGIT_RUN_RE`, unlike a bare sign) -- so `a`'s int-vs-fraction split alone gives 2+ shape
families before the b/c/d sign pattern (which also survives masking, since `join_terms` always puts a space
between a "+"/"-" operator and the next term) adds more.
"""

from __future__ import annotations

import pytest

from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape
from mathgen.grading import typed as grading

TOPIC = "expand_bracket"
KIND = "single_trinomial"
SEEDS = range(60)  # 60 seeds, matching measure_tiers.py's own sweep


def _mk(seed):
    return make(TOPIC, seed=seed, difficulty="hard", kind=KIND)


def test_determinism_same_seed_is_byte_identical():
    for seed in SEEDS:
        first = _mk(seed).to_json()
        second = _mk(seed).to_json()
        assert first == second, f"seed={seed}: not deterministic"


def test_no_bad_text_patterns_over_60_seeds():
    for seed in SEEDS:
        q = _mk(seed)
        defects = find_text_defects(q)
        assert defects == [], f"seed={seed}: {defects}"


def test_prompt_variety_hard_tier_specifically():
    shapes = {prompt_shape(_mk(seed)) for seed in SEEDS}
    assert len(shapes) >= 3, shapes


def test_hard_tier_adds_a_shape_neither_lower_tier_has():
    """Mirrors `scripts/measure_tiers.py`'s own escalation check (the
    instrument that flipped `expand-single-brackets` from range-only to
    escalates once this branch was wired to the `hard` source) -- proved
    independently here rather than trusted by eye."""
    easy_shapes = {prompt_shape(make(TOPIC, seed=s, difficulty="easy", kind="single")) for s in SEEDS}
    medium_shapes = {prompt_shape(make(TOPIC, seed=s, difficulty="medium", kind="single")) for s in SEEDS}
    hard_shapes = {prompt_shape(_mk(s)) for s in SEEDS}
    new_shapes = hard_shapes - easy_shapes - medium_shapes
    assert new_shapes, "hard tier adds no shape over easy/medium"


def test_default_kind_single_is_unaffected():
    """`expand_bracket`'s default (`kind="single"`, no params) is provably
    unchanged by this branch's addition -- same behaviour before and after this change,
    confirmed directly rather than only inferred from the additive-only
    `git diff`."""
    for seed in (0, 1, 2, 7, 42):
        q = make(TOPIC, seed=seed)
        assert q.answer_form == "expanded"
        assert "single_trinomial" not in q.text


def test_mental_block_kind_single_still_resolves():
    """The skill's `mental` block passes `kind: single` with its own `hi`
    overrides at every level and is untouched by this change -- confirm it
    still resolves correctly rather than assume it (do not
    silently "fix" it if it looks odd)."""
    for hi in (6, 9, 12):
        q = make(TOPIC, seed=3, kind="single", hi=hi)
        assert q.text and q.answer_text
        assert q.answer_form == "expanded"


# ---------------------------------------------------------------------------
# Answer shape and grading, every seed.


def test_answer_form_is_expanded_every_seed():
    for seed in SEEDS:
        assert _mk(seed).answer_form == "expanded"


def test_answer_and_every_distractor_contain_no_bracket():
    """`answer_form == "expanded"` means the canonical answer_text is a fully distributed expression -- so is
    every misconception distractor here, since each is itself an already-expanded sympy expression (never the
    question copied verbatim). A stray "(" on any option would make it identifiable by shape alone."""
    for seed in SEEDS:
        q = _mk(seed)
        assert "(" not in q.answer_text, (seed, q.answer_text)
        for d in q.distractors:
            assert "(" not in d, (seed, d)


def test_three_distractors_distinct_nonempty_and_not_the_answer():
    for seed in SEEDS:
        q = _mk(seed)
        assert len(q.distractors) == 3, (seed, q.distractors)
        assert len(set(q.distractors)) == 3, (seed, q.distractors)
        assert q.answer_text not in q.distractors, (seed, q.distractors)
        assert all(d for d in q.distractors), (seed, q.distractors)


def test_real_grader_accepts_canonical_and_rejects_every_distractor():
    """`mathgen.grading.GRADERS["expression"]`, called directly with learner-
    typed strings (`question.answer_text` for the canonical case, each
    distractor string for the misconception cases) -- not a self-consistency
    check against the generator's own `check()`/`check_equivalent()`."""
    grade_expression = grading.GRADERS["expression"]
    for seed in SEEDS:
        q = _mk(seed)
        accepted = grade_expression(q, q.answer_text)
        assert accepted.correct, (seed, q.text, q.answer_text)
        for d in q.distractors:
            rejected = grade_expression(q, d)
            assert not rejected.correct, (seed, q.text, "distractor wrongly accepted:", d)


def test_a_deliberately_wrong_answer_is_rejected():
    """A hand-corrupted value (answer + 1 on the constant term, as a raw
    string) must also be rejected -- independent of whichever misconception
    distractors the generator happens to draw for a given seed."""
    grade_expression = grading.GRADERS["expression"]
    for seed in SEEDS:
        q = _mk(seed)
        corrupted = q.answer_text + " + 1"
        result = grade_expression(q, corrupted)
        assert not result.correct, (seed, q.text, q.answer_text, corrupted)


def test_factored_question_copied_verbatim_is_rejected():
    """A learner who copies the (unexpanded) question back as their answer
    must be rejected -- the same check every expanded-form generator needs, applied here to this
    topic's `kind="single"`."""
    grade_expression = grading.GRADERS["expression"]
    for seed in SEEDS:
        q = _mk(seed)
        # `q.text` is "Expand <a>(<inner>)"; the bracketed question itself,
        # copied verbatim, is never a valid "expanded" answer.
        copied = q.text.removeprefix("Expand ")
        result = grade_expression(q, copied)
        assert not result.correct, (seed, q.text, copied)


# The independent, in-generator check firing on a corrupted value -- imports the generator module
# directly, ONLY here, to monkeypatch it; every test above treats `mathgen.make` as a black box.

import mathgen.generators.foundations_algebra as _foundations_algebra  # noqa: E402


def test_corrupted_constant_term_raises_verification_error(monkeypatch):
    """Corrupts the FIRST `expand()` call inside `expand_bracket` (the
    branch's own `ans = expand(a * (b * x + c * y + d))`) by adding 1 to its
    result, which shifts the constant term away from `a * d` -- the in-body
    `check_equivalent(ans.subs({x: 0, y: 0}), a * d, ...)` must then raise."""
    original_expand = _foundations_algebra.expand
    call_count = {"n": 0}

    def _corrupted_expand(*args, **kwargs):
        result = original_expand(*args, **kwargs)
        call_count["n"] += 1
        if call_count["n"] == 1:
            return result + 1  # corrupt only the branch's own `ans`
        return result

    monkeypatch.setattr(_foundations_algebra, "expand", _corrupted_expand)
    with pytest.raises(VerificationError):
        make(TOPIC, seed=1, difficulty="hard", kind=KIND)


def test_corrupted_constant_term_is_not_a_false_positive_when_uncorrupted():
    """Sanity check on the test above: the same seed, unpatched, must NOT
    raise -- otherwise the firing test would be vacuous."""
    make(TOPIC, seed=1, difficulty="hard", kind=KIND)  # must not raise
