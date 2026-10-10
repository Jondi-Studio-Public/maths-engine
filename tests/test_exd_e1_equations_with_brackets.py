"""Tests the `linear_equation` topic's new `brackets_leading` kind, which puts a constant outside the bracket
before the distribution step.

Two separate deliverables on the `linear_equation` topic (`mathgen/generators/algebra.py`, PINNED topic).

1. The range-only hard tier is fixed with a NEW kind, `brackets_leading`:
   `d (+/-) a(x + h) = c` -- a leading constant term `d` OUTSIDE the bracket,
   on the same side, before the distribution step. `a` may now be drawn
negative too, but
 a negative
   multiplier ALONE does not register as an escalation to
   `scripts/measure_tiers.py`: `mathgen.text_quality._mask_digits` masks a
   leading sign together with its digit run. The genuine escalation is the
   leading constant `d`, a real token-sequence change.

2. A new exam source, kind `double_brackets`: `a1(x + h1) = a2(x + h2)`,
   brackets on BOTH sides, each side its own multiplier, at least one
   negative -- exam item shape, `6(2b - 7) = -4(9 + 3b)`. This generator's
   equivalent form keeps the inner variable coefficient at 1 (`x`, not
   "or your equivalent form"), and the "b" vs "x"
   naming note.

The default `kind="standard"` output of `linear_equation` is untouched: the bare seeds 0/1/7/42 and the
"N:easy"/"N:medium"/"N:hard" presets all use the *default* `kind="standard"` (none passes a `kind=` override),
never `brackets_leading`/`double_brackets`. Both new branches, and the ORIGINAL
`kind == "brackets"` behaviour (no leading constant, one bracket, one side, `a` always positive), are
unaffected -- `git diff main -- mathgen/generators/algebra.py` is purely additive.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "linear_equation"

# ---------------------------------------------------------------------------
# 1. brackets_leading (hard tier fix)
# ---------------------------------------------------------------------------


def _make_leading(seed, difficulty="hard"):
    return make(TOPIC, seed=seed, difficulty=difficulty, kind="brackets_leading")


def test_leading_no_bad_text_patterns_over_60_seeds():
    for seed in range(60):
        q = _make_leading(seed)
        defects = find_text_defects(q)
        assert defects == [], f"seed={seed}: {defects} ({q.text!r})"


def test_leading_prompt_variety():
    """>= 3 distinct prompt_shape() outputs over 60 seeds for the new hard
    kind alone."""
    shapes = {prompt_shape(_make_leading(s)) for s in range(60)}
    assert len(shapes) >= 3, shapes


def test_leading_hard_adds_shapes_over_easy_and_medium():
    """The new hard kind's shapes are not a subset of the union of the
    UNCHANGED easy/medium (`kind: brackets`) tiers' shapes -- a genuine
    escalation in kind, not merely a bigger number. This is exactly the
    sign-masking trap check: if the only difference were `a`'s sign, the
    digit-run mask would erase it and this would fail."""
    lower_shapes = {prompt_shape(make(TOPIC, seed=s, difficulty="easy", kind="brackets")) for s in range(60)}
    lower_shapes |= {prompt_shape(make(TOPIC, seed=s, difficulty="medium", kind="brackets")) for s in range(60)}
    hard_shapes = {prompt_shape(_make_leading(s)) for s in range(60)}
    new_shapes = hard_shapes - lower_shapes
    assert len(new_shapes) >= 1, (hard_shapes, lower_shapes)


def test_leading_determinism_same_seed_is_byte_identical():
    for seed in (0, 1, 2, 3, 100, 12345):
        first = _make_leading(seed).to_json()
        second = _make_leading(seed).to_json()
        assert first == second


def test_leading_answer_matches_independent_solve_over_300_seeds():
    """Re-derive x0 from the rendered prompt text ALONE, with Fraction
    arithmetic, never importing `mathgen.generators.algebra`. The oracle."""
    mismatches = []
    for seed in range(300):
        q = _make_leading(seed)
        oracle_x0 = _independent_leading_solution(q.text)
        if oracle_x0 != q.answer:
            mismatches.append((seed, q.text, oracle_x0, q.answer))
    assert mismatches == [], mismatches


_LEADING_RE = re.compile(r"^Solve (-?\d+) ([+-]) (\d+)\(x ([+-]) (\d+)\) = (-?\d+)$")


def _independent_leading_solution(question_text: str) -> int:
    m = _LEADING_RE.match(question_text)
    assert m is not None, f"oracle: could not parse {question_text!r}"
    d, sign_a, a_mag, sign_h, h_mag, c = m.groups()
    d, a_mag, h_mag, c = int(d), int(a_mag), int(h_mag), int(c)
    a = a_mag if sign_a == "+" else -a_mag
    h = h_mag if sign_h == "+" else -h_mag
    # d + a(x + h) = c  =>  x = (c - d)/a - h
    x0 = Fraction(c - d, a) - h
    assert x0.denominator == 1, f"oracle: non-integer solution {x0} for {question_text!r}"
    return int(x0)


import mathgen.generators.algebra as _algebra  # noqa: E402


def test_leading_corrupted_solve_raises_verification_error(monkeypatch):
    original_solve = _algebra.solve

    def _wrong_solve(*args, **kwargs):
        result = original_solve(*args, **kwargs)
        return [r + 1 for r in result]

    monkeypatch.setattr(_algebra, "solve", _wrong_solve)
    with pytest.raises(VerificationError):
        make(TOPIC, seed=0, difficulty="hard", kind="brackets_leading")


def test_leading_corrupted_solve_is_not_a_false_positive_when_uncorrupted():
    make(TOPIC, seed=0, difficulty="hard", kind="brackets_leading")  # no exception


# -- the real grader (mathgen.grading.GRADERS["number"]) -------------------------

from mathgen.grading.typed import GRADERS  # noqa: E402

_NUMBER_GRADER = GRADERS["number"]


def test_leading_grader_accepts_canonical_and_rejects_every_distractor_over_200_seeds():
    bad = []
    for seed in range(200):
        q = _make_leading(seed)
        canonical_result = _NUMBER_GRADER(q, q.answer_text)
        if not canonical_result.correct:
            bad.append(("canonical-rejected", seed, q.answer_text))
        for d in q.distractors:
            distractor_result = _NUMBER_GRADER(q, d)
            if distractor_result.correct:
                bad.append(("distractor-accepted", seed, d, q.answer_text))
    assert bad == [], bad


def test_leading_grader_rejects_deliberately_wrong_values():
    """The three misconception values, computed by hand against a fixed
    seed and checked they are actually wrong before asserting the grader
    rejects them (never a vacuous check)."""
    q = _make_leading(0)
    x0 = q.answer
    wrong_candidates = [f"x = {x0 + 1}", f"x = {x0 - 1}", f"x = {-x0}", "x = 0"]
    for wrong in wrong_candidates:
        wrong_val = int(wrong.split("=")[1].strip())
        assert wrong_val != x0, (wrong, x0)
        result = _NUMBER_GRADER(q, wrong)
        assert not result.correct, (wrong, q.answer_text)


def test_difficulty_exam_preset_is_registered():
    """`make("linear_equation", difficulty="exam")` must not raise --
    `linear_equation` has other named presets (easy/medium/hard), so
    `mathgen.core.make` requires "exam" to be a registered preset name too,
    even though the real exam-tier shape comes from the skill YAML's own
    `params: {kind: double_brackets}`, not from this preset's own
    coef_hi/sol_hi/b_hi (mirrors the identical fix for
    `quadratic`)."""
    make(TOPIC, seed=0, difficulty="exam")  # kind stays default "standard"; must not raise


# ---------------------------------------------------------------------------
# 2. double_brackets (exam source)
# ---------------------------------------------------------------------------


def _make_double(seed, difficulty="exam"):
    return make(TOPIC, seed=seed, difficulty=difficulty, kind="double_brackets")


def test_double_no_bad_text_patterns_over_60_seeds():
    for seed in range(60):
        q = _make_double(seed)
        defects = find_text_defects(q)
        assert defects == [], f"seed={seed}: {defects} ({q.text!r})"


def test_double_prompt_variety():
    shapes = {prompt_shape(_make_double(s)) for s in range(60)}
    assert len(shapes) >= 3, shapes


def test_double_exam_adds_shapes_over_hard_tier():
    """The exam source's shapes are not a subset of the new hard tier's
    shapes -- brackets on both sides is a genuinely different shape from a
    leading constant with one bracket."""
    hard_shapes = {prompt_shape(_make_leading(s, "hard")) for s in range(60)}
    exam_shapes = {prompt_shape(_make_double(s)) for s in range(60)}
    new_shapes = exam_shapes - hard_shapes
    assert len(new_shapes) >= 1, (exam_shapes, hard_shapes)


def test_double_determinism_same_seed_is_byte_identical():
    for seed in (0, 1, 2, 3, 100, 12345):
        first = _make_double(seed).to_json()
        second = _make_double(seed).to_json()
        assert first == second


def test_double_at_least_one_multiplier_negative_over_600_seeds():
    for seed in range(600):
        q = _make_double(seed)
        a1, _h1, a2, _h2 = _parse_double(q.text)
        assert a1 < 0 or a2 < 0, (seed, q.text)


def test_double_no_construction_exceptions_over_600_seeds():
    """The rejection-sampling loop in `_linear_equation_double_brackets`
    must reliably find valid integer h1/h2 across the full exam seed
    range -- no seed should raise."""
    for seed in range(600):
        _make_double(seed)  # must not raise


_DOUBLE_RE = re.compile(r"^Solve (-?\d+)\(x ([+-]) (\d+)\) = (-?\d+)\(x ([+-]) (\d+)\)$")


def _parse_double(question_text: str) -> tuple:
    m = _DOUBLE_RE.match(question_text)
    assert m is not None, f"oracle: could not parse {question_text!r}"
    a1, sign_h1, h1_mag, a2, sign_h2, h2_mag = m.groups()
    a1, h1_mag, a2, h2_mag = int(a1), int(h1_mag), int(a2), int(h2_mag)
    h1 = h1_mag if sign_h1 == "+" else -h1_mag
    h2 = h2_mag if sign_h2 == "+" else -h2_mag
    return a1, h1, a2, h2


def _independent_double_solution(question_text: str) -> int:
    a1, h1, a2, h2 = _parse_double(question_text)
    assert a1 != a2, f"oracle: degenerate equation {question_text!r}"
    # a1(x + h1) = a2(x + h2)  =>  x = (a2*h2 - a1*h1) / (a1 - a2)
    x0 = Fraction(a2 * h2 - a1 * h1, a1 - a2)
    assert x0.denominator == 1, f"oracle: non-integer solution {x0} for {question_text!r}"
    return int(x0)


def test_double_answer_matches_independent_solve_over_600_seeds():
    mismatches = []
    for seed in range(600):
        q = _make_double(seed)
        oracle_x0 = _independent_double_solution(q.text)
        if oracle_x0 != q.answer:
            mismatches.append((seed, q.text, oracle_x0, q.answer))
    assert mismatches == [], mismatches


def test_double_corrupted_solve_raises_verification_error(monkeypatch):
    original_solve = _algebra.solve

    def _wrong_solve(*args, **kwargs):
        result = original_solve(*args, **kwargs)
        return [r + 1 for r in result]

    monkeypatch.setattr(_algebra, "solve", _wrong_solve)
    with pytest.raises(VerificationError):
        make(TOPIC, seed=0, difficulty="exam", kind="double_brackets")


def test_double_corrupted_solve_is_not_a_false_positive_when_uncorrupted():
    make(TOPIC, seed=0, difficulty="exam", kind="double_brackets")  # no exception


def test_double_grader_accepts_canonical_and_rejects_every_distractor_over_200_seeds():
    bad = []
    for seed in range(200):
        q = _make_double(seed)
        canonical_result = _NUMBER_GRADER(q, q.answer_text)
        if not canonical_result.correct:
            bad.append(("canonical-rejected", seed, q.answer_text))
        for d in q.distractors:
            distractor_result = _NUMBER_GRADER(q, d)
            if distractor_result.correct:
                bad.append(("distractor-accepted", seed, d, q.answer_text))
    assert bad == [], bad


def test_double_grader_rejects_deliberately_wrong_values():
    q = _make_double(0)
    x0 = q.answer
    wrong_candidates = [f"x = {x0 + 1}", f"x = {x0 - 1}", f"x = {-x0}", "x = 0"]
    for wrong in wrong_candidates:
        wrong_val = int(wrong.split("=")[1].strip())
        assert wrong_val != x0, (wrong, x0)
        result = _NUMBER_GRADER(q, wrong)
        assert not result.correct, (wrong, q.answer_text)


# ---------------------------------------------------------------------------
# 3. Original `kind == "brackets"` behaviour, and the mental block, untouched
# ---------------------------------------------------------------------------


def test_original_brackets_kind_never_draws_a_negative():
    """Confirms the ORIGINAL `kind == "brackets"` branch (easy/medium
    sources, and every mental level) is unaffected: `a` is still always
    drawn positive, no leading constant is ever rendered."""
    for seed in range(300):
        for difficulty in ("easy", "medium", "hard"):
            q = make(TOPIC, seed=seed, difficulty=difficulty, kind="brackets")
            assert re.match(r"^Solve \d+\(x [+-] \d+\) = -?\d+$", q.text), q.text


def test_mental_levels_use_original_brackets_kind_unaffected():
    """The skill YAML's `mental.levels` block uses `kind: brackets` at all
    three levels with no `difficulty:` key -- confirmed here by reading the
    YAML and reproducing its params directly, to verify (not assume)
    that the mental block is unaffected."""
    import yaml as _yaml

    yaml_path = __file__.rsplit("tests", 1)[0] + "skills/equations/equations-with-brackets.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        raw = _yaml.safe_load(f)
    for level in raw["mental"]["levels"]:
        assert level["params"]["kind"] == "brackets"
        assert "difficulty" not in level
        params = dict(level["params"])
        params.pop("kind")
        q = make(TOPIC, seed=0, kind="brackets", **params)
        assert re.match(r"^Solve \d+\(x [+-] \d+\) = -?\d+$", q.text), q.text
