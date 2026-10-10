"""The `exam` source on `polynomial-linear-combination` and `add-subtract-polynomials`.

The independent oracle below parses the RENDERED question text from scratch (never the
generator's own coefficient arrays or its `mathgen.generators.t2_polynomial_combine_build`
module) -- it extracts each labelled polynomial's own text, parses it with sympy, parses the
"Find ..." formula as its own small expression over Symbol('P')/Symbol('Q')/Symbol('R') (or
A/B/C), and substitutes the polynomials in. Both templates on the `add_subtract` exam source
(the nested `P - (Q + R)` and the flat `A + B - C`) match the SAME `L = expr. L = expr. L =
expr. Find <formula>.` shape, so one oracle regex covers both skills and both templates.
"""

from __future__ import annotations

import re

import pytest
import sympy
from sympy import Symbol, symbols
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.grading import typed as grading
from mathgen.skills import load_skills
from mathgen import make
from mathgen.core import VerificationError
from mathgen.text_quality import find_text_defects, prompt_shape

x, y = symbols("x y")
_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)
SEEDS = range(60)

_LC_TOPIC = "polynomial_linear_combination"
_AS_TOPIC = "polynomial_add_subtract"

_QUESTION_RE = re.compile(r"^(\w) = (.+?)\. (\w) = (.+?)\. (\w) = (.+?)\. Find (.+)\.$")


def _sympify(text: str, local_dict: dict | None = None):
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMATIONS, local_dict=local_dict)


def _oracle_answer(question_text: str):
    """Re-derive the answer from the RENDERED text alone, independent of the
    generator's own coefficient-array route: parse each labelled polynomial's
    own displayed text with sympy, parse the "Find ..." formula as a small
    expression over its own three letter-symbols, and substitute. Works for
    both the linear-combination (P, Q, R with drawn multipliers/signs) and
    add-subtract (P - (Q + R) nested, or A + B - C flat) exam sources -- both
    match the same rendered shape."""
    m = _QUESTION_RE.match(question_text.strip())
    assert m, f"question text does not match the expected L = expr. ... Find formula. shape: {question_text!r}"
    l1, e1, l2, e2, l3, e3, formula = m.groups()
    xy_dict = {"x": x, "y": y}
    v1, v2, v3 = _sympify(e1, xy_dict), _sympify(e2, xy_dict), _sympify(e3, xy_dict)
    letter_syms = {l1: Symbol(l1), l2: Symbol(l2), l3: Symbol(l3)}
    formula_expr = _sympify(formula, letter_syms)
    substituted = formula_expr.subs({letter_syms[l1]: v1, letter_syms[l2]: v2, letter_syms[l3]: v3})
    return sympy.expand(substituted)


def _stated_answer(question) -> sympy.Expr:
    return sympy.expand(_sympify(question.answer_text, {"x": x, "y": y}))


# ---------------------------------------------------------------------------
# The independent oracle itself: examine every skill source (not just the
# exam one) the same way tests/test_t2_polynomial_combine_oracle.py does, so
# a regression on an EXISTING (single-variable) source would also be caught
# here -- though this file's own scope is the exam source.


@pytest.mark.parametrize("skill_id", ["add-subtract-polynomials", "polynomial-linear-combination"])
def test_oracle_agrees_with_every_source_every_seed(skill_id):
    _, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    examined = 0
    challenged = []
    for source in skill.sources:
        for seed in SEEDS:
            q = make(topic=source.topic, seed=seed, difficulty=source.difficulty, **(source.params or {}))
            examined += 1
            oracle = _oracle_answer(q.text)
            stated = _stated_answer(q)
            if sympy.simplify(oracle - stated) != 0:
                challenged.append(
                    {
                        "source_difficulty": source.difficulty,
                        "seed": seed,
                        "issue": f"oracle {oracle} != stated {stated}",
                        "text": q.text,
                    }
                )
    assert not challenged, f"{len(challenged)} of {examined} challenged: {challenged}"


# ---------------------------------------------------------------------------
# Two-variable design: both exam sources are genuinely two-variable (x AND
# y), not single-variable-by-degree.


def test_linear_combination_exam_is_genuinely_two_variable():
    for seed in SEEDS:
        q = make(_LC_TOPIC, seed=seed, difficulty="exam")
        assert "xy" in q.text, (seed, q.text)
        assert re.search(r"x\^2", q.text), (seed, q.text)
        assert re.search(r"y\^2", q.text), (seed, q.text)


def test_add_subtract_exam_is_genuinely_two_variable():
    for seed in SEEDS:
        q = make(_AS_TOPIC, seed=seed, difficulty="exam")
        assert "xy" in q.text, (seed, q.text)
        assert re.search(r"x\^2", q.text), (seed, q.text)
        assert re.search(r"y\^2", q.text), (seed, q.text)


# ---------------------------------------------------------------------------
# Required behaviour: multipliers up to 5, both signs -- the
# subtracted slot varies across P/Q/R rather than a fixed pattern.


def test_linear_combination_exam_subtracted_slot_varies():
    subtracted_letters = set()
    for seed in range(200):
        q = make(_LC_TOPIC, seed=seed, difficulty="exam")
        formula = q.text.split("Find ")[1].rstrip(".")
        padded = "+" + formula
        for sign, mag, letter in re.findall(r"([+-])\s*(\d*)([PQR])", padded):
            mult = int(mag) if mag else 1
            assert 2 <= mult <= 5, (seed, formula, letter, mult)
            if sign == "-":
                subtracted_letters.add(letter)
    assert subtracted_letters == {"P", "Q", "R"}, (
        f"expected all three of P/Q/R to appear as the subtracted slot over 200 seeds, got {subtracted_letters}"
    )


# ---------------------------------------------------------------------------
# Required behaviour: both the nested-bracket "P - (Q + R)" and the
# flat "A + B - C" templates occur.


def test_add_subtract_exam_both_templates_occur():
    nested = flat = 0
    for seed in range(200):
        q = make(_AS_TOPIC, seed=seed, difficulty="exam")
        if "P - (Q + R)" in q.text:
            nested += 1
        elif "A + B - C" in q.text:
            flat += 1
        else:
            pytest.fail(f"seed={seed}: neither template matched: {q.text!r}")
    assert nested > 0 and flat > 0, f"nested={nested}, flat={flat} over 200 seeds"


# ---------------------------------------------------------------------------
# The coefficient-drop trap's two independent exponent pairs (x^2, xy, y^2
# each independently exercisable as the missing monomial).


def _missing_monomial_labels(seen: set[str], seed: int, topic: str) -> None:
    q = make(topic, seed=seed, difficulty="exam")
    # Every polynomial segment ("L = <text>.") should carry all three
    # quadratic labels UNLESS it is the one segment the generator forced
    # missing -- collect which label is absent from at least one segment.
    for segment in re.findall(r"\w = (.+?)\.", q.text):
        for label in ("x\\^2", "xy", "y\\^2"):
            if not re.search(label, segment):
                seen.add(label.replace("\\", ""))


def test_linear_combination_exam_coefficient_drop_covers_all_three_monomials():
    seen: set[str] = set()
    for seed in range(300):
        _missing_monomial_labels(seen, seed, _LC_TOPIC)
    assert seen == {"x^2", "xy", "y^2"}, (
        f"expected all three of x^2/xy/y^2 to be exercised as the missing monomial over 300 seeds, got {seen}"
    )


def test_add_subtract_exam_coefficient_drop_covers_all_three_monomials():
    seen: set[str] = set()
    for seed in range(300):
        _missing_monomial_labels(seen, seed, _AS_TOPIC)
    assert seen == {"x^2", "xy", "y^2"}, (
        f"expected all three of x^2/xy/y^2 to be exercised as the missing monomial over 300 seeds, got {seen}"
    )


@pytest.mark.parametrize("topic", [_LC_TOPIC, _AS_TOPIC])
def test_determinism_same_seed_is_byte_identical(topic):
    for seed in SEEDS:
        first = make(topic, seed=seed, difficulty="exam").to_json()
        second = make(topic, seed=seed, difficulty="exam").to_json()
        assert first == second, f"seed={seed}: not deterministic"


@pytest.mark.parametrize("topic", [_LC_TOPIC, _AS_TOPIC])
def test_no_bad_text_patterns_over_60_seeds(topic):
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty="exam")
        defects = find_text_defects(q)
        assert defects == [], f"seed={seed}: {defects}"


@pytest.mark.parametrize("topic", [_LC_TOPIC, _AS_TOPIC])
def test_prompt_variety(topic):
    shapes = {prompt_shape(make(topic, seed=s, difficulty="exam")) for s in SEEDS}
    assert len(shapes) >= 3, shapes


@pytest.mark.parametrize("topic", [_LC_TOPIC, _AS_TOPIC])
def test_exam_adds_a_shape_neither_lower_tier_has(topic):
    """Mirrors `scripts/measure_tiers.py`'s own escalation check, proved
    directly here rather than trusted by eye."""
    lower_shapes: set[str] = set()
    for difficulty in ("easy", "medium", "hard"):
        lower_shapes |= {prompt_shape(make(topic, seed=s, difficulty=difficulty)) for s in SEEDS}
    exam_shapes = {prompt_shape(make(topic, seed=s, difficulty="exam")) for s in SEEDS}
    new_shapes = exam_shapes - lower_shapes
    assert new_shapes, f"{topic} exam source adds no shape over easy/medium/hard"


@pytest.mark.parametrize("topic", [_LC_TOPIC, _AS_TOPIC])
def test_at_least_20_distinct_rendered_questions_over_60_seeds(topic):
    texts = {make(topic, seed=s, difficulty="exam").text for s in SEEDS}
    assert len(texts) >= 20, f"{topic} exam: only {len(texts)} distinct rendered questions over 60 seeds"


# ---------------------------------------------------------------------------
# Answer shape and grading, every seed.


@pytest.mark.parametrize("topic", [_LC_TOPIC, _AS_TOPIC])
def test_answer_form_is_expanded_every_seed(topic):
    for seed in SEEDS:
        assert make(topic, seed=seed, difficulty="exam").answer_form == "expanded"


@pytest.mark.parametrize("topic", [_LC_TOPIC, _AS_TOPIC])
def test_answer_and_every_distractor_formatted_alike_no_bracket(topic):
    """`answer_form == "expanded"` means the canonical answer_text is a fully distributed expression -- so is
    every misconception distractor, since each is itself an already-expanded sympy-derived value, never the
    question copied verbatim. A stray "(" on any option would make it identifiable by shape alone."""
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty="exam")
        assert "(" not in q.answer_text, (seed, q.answer_text)
        for d in q.distractors:
            assert "(" not in d, (seed, d)


@pytest.mark.parametrize("topic", [_LC_TOPIC, _AS_TOPIC])
def test_three_distractors_distinct_nonempty_and_not_the_answer(topic):
    """All three misconceptions (sign-error, the skill's second trap, and
    coefficient-drop -- forced every exam question) each survive
    `pick_distractors` as a genuinely distinct offered option, every seed --
    proof each is honestly exercisable in the two-variable setting, not just
    theoretically possible."""
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty="exam")
        assert len(q.distractors) == 3, (seed, q.distractors)
        assert len(set(q.distractors)) == 3, (seed, q.distractors)
        assert q.answer_text not in q.distractors, (seed, q.distractors)
        assert all(d for d in q.distractors), (seed, q.distractors)


@pytest.mark.parametrize("topic", [_LC_TOPIC, _AS_TOPIC])
def test_real_grader_accepts_canonical_and_rejects_every_distractor(topic):
    """`mathgen.grading.GRADERS["expression"]`, called directly with
    learner-typed strings -- not a self-consistency check against the
    generator's own `check()`/`check_equivalent()`."""
    grade_expression = grading.GRADERS["expression"]
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty="exam")
        accepted = grade_expression(q, q.answer_text)
        assert accepted.correct, (seed, q.text, q.answer_text)
        for d in q.distractors:
            rejected = grade_expression(q, d)
            assert not rejected.correct, (seed, q.text, "distractor wrongly accepted:", d)


@pytest.mark.parametrize("topic", [_LC_TOPIC, _AS_TOPIC])
def test_a_deliberately_wrong_answer_is_rejected(topic):
    """A hand-corrupted value (answer + 1, as raw text) must also be
    rejected -- independent of whichever misconception distractors the
    generator happens to draw for a given seed."""
    grade_expression = grading.GRADERS["expression"]
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty="exam")
        corrupted = q.answer_text + " + 1"
        result = grade_expression(q, corrupted)
        assert not result.correct, (seed, q.text, q.answer_text, corrupted)


@pytest.mark.parametrize("topic", [_LC_TOPIC, _AS_TOPIC])
def test_a_bracketed_reconstruction_is_rejected(topic):
    """`answer_form="expanded"` must reject a raw-text answer that still
    carries a bracket, even one built from the question's own three
    polynomials -- the known, disclosed grader gap this skill's description
    names is that an UNCOMBINED-but-unbracketed answer slips through (sympy
    auto-combines like terms at parse time); a bracketed one must not, and
    this is the proof for that half. (Not necessarily equal by VALUE to the
    canonical answer for every drawn sign/multiplier combination -- the
    bracket-presence check in `mathgen/grading/typed.py`'s "expanded" branch runs on
    the raw text before any value comparison, so this holds regardless.)"""
    grade_expression = grading.GRADERS["expression"]
    for seed in SEEDS:
        q = make(topic, seed=seed, difficulty="exam")
        m = _QUESTION_RE.match(q.text.strip())
        l1, e1, l2, e2, l3, e3, formula = m.groups()
        own_construction = f"({e1}) + ({e2}) - ({e3})"
        result = grade_expression(q, own_construction)
        assert not result.correct, (seed, q.text, own_construction)


# The independent, in-generator check firing on a corrupted value -- imports the generator module
# directly, ONLY here, to monkeypatch it; every test above treats `mathgen.make` as a black box.

import mathgen.generators.t2_polynomial_combine_build as _t2_build  # noqa: E402


def _corrupt_first_combine(monkeypatch):
    """Corrupts the FIRST `_combine_arrays` call inside either two-variable
    branch -- which, in both `_polynomial_add_subtract_two_var` and
    `_polynomial_linear_combination_two_var`, computes `correct` before any
    distractor is built -- so the in-body `check_equivalent(correct, ...)`
    must then disagree and raise."""
    original = _t2_build._combine_arrays
    call_count = {"n": 0}

    def _corrupted(arrays, signs):
        result = original(arrays, signs)
        call_count["n"] += 1
        if call_count["n"] == 1:
            result = list(result)
            result[0] += 1  # shift the x^2 coefficient away from the true value
        return result

    monkeypatch.setattr(_t2_build, "_combine_arrays", _corrupted)


@pytest.mark.parametrize("topic", [_LC_TOPIC, _AS_TOPIC])
def test_corrupted_combine_raises_verification_error(topic, monkeypatch):
    _corrupt_first_combine(monkeypatch)
    with pytest.raises(VerificationError):
        make(topic, seed=1, difficulty="exam")


@pytest.mark.parametrize("topic", [_LC_TOPIC, _AS_TOPIC])
def test_corrupted_combine_is_not_a_false_positive_when_uncorrupted(topic):
    """Sanity check on the test above: the same seed, unpatched, must NOT
    raise -- otherwise the firing test would be vacuous."""
    make(topic, seed=1, difficulty="exam")  # must not raise


# ---------------------------------------------------------------------------
# Skill loading -- the new source itself.


@pytest.mark.parametrize(
    "skill_id,topic",
    [
        ("add-subtract-polynomials", _AS_TOPIC),
        ("polynomial-linear-combination", _LC_TOPIC),
    ],
)
def test_skill_has_an_exam_source_reachable_only_from_the_exam_stage(skill_id, topic):
    _, skills = load_skills()
    skill = next(s for s in skills if s.id == skill_id)
    exam_sources = [s for s in skill.sources if s.difficulty == "exam"]
    assert len(exam_sources) == 1, (skill_id, [s.difficulty for s in skill.sources])
    exam_source = exam_sources[0]
    assert exam_source.topic == topic
    assert list(exam_source.stages) == ["exam"]
    # The three existing sources are unchanged: still easy/medium/hard, still
    # topic-matched, still with their own non-exam stages.
    other_difficulties = sorted(s.difficulty for s in skill.sources if s.difficulty != "exam")
    assert other_difficulties == ["easy", "hard", "medium"]
    for s in skill.sources:
        if s.difficulty != "exam":
            assert "exam" not in s.stages, (skill_id, s.difficulty, s.stages)
