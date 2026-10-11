"""Fault injection for the shared skill contract (the coverage-parity evidence).

Each case damages a healthy skill's generator or grader in the way a per-batch check exists to catch, and
asserts the matching contract family fails. If a family stops noticing its fault, the per-batch checks it
replaces are not safe to delete.
"""

import itertools

import pytest

from mathgen.grading import typed as grading
from scripts import skill_contract as sc

EXPR = "senior-determine-f-left-x-right-given-f-x"  # expression answer, 3 tiers
NUMBER = "angle-degrees-to-radians-decimal"  # number answer, 3 tiers


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    sc.draws.cache_clear()
    yield
    sc.draws.cache_clear()


def _patch_make(monkeypatch, fn):
    real = sc.make
    monkeypatch.setattr(sc, "make", lambda *a, **k: fn(real(*a, **k)))


def test_healthy_skills_pass_every_family():
    for sid in (EXPR, NUMBER):
        for name, check in sc.FAMILIES.items():
            check(sid)


def test_determinism_notices_a_generator_that_varies_between_calls(monkeypatch):
    calls = itertools.count()

    def vary(q):
        q.text += f" #{next(calls)}"
        return q

    _patch_make(monkeypatch, vary)
    with pytest.raises(AssertionError, match="not deterministic"):
        sc.check_determinism(EXPR)


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda q: q.distractors.append(q.answer_text), "answer among distractors"),
        (lambda q: q.distractors.append(q.distractors[0]), "duplicate distractors"),
        (lambda q: q.distractors.clear(), "no distractors"),
        (lambda q: q.distractors.__setitem__(0, "  "), "blank distractor"),
    ],
)
def test_distractors_notice_each_hygiene_fault(monkeypatch, mutate, message):
    def damage(q):
        mutate(q)
        return q

    _patch_make(monkeypatch, damage)
    with pytest.raises(AssertionError, match=message):
        sc.check_distractors(EXPR)


def test_variety_notices_a_collapsed_prompt_pool(monkeypatch):
    def collapse(q):
        q.text = "Same prompt " + str(q.seed % 3)
        return q

    _patch_make(monkeypatch, collapse)
    with pytest.raises(AssertionError, match="distinct prompts"):
        sc.check_variety(EXPR)


def test_escalation_notices_tiers_that_draw_the_same_questions(monkeypatch):
    real = sc.make
    # the exam source (its own topic, with only an "exam" preset) is left alone: the three practice tiers collapse
    monkeypatch.setattr(
        sc,
        "make",
        lambda topic, seed=None, difficulty=None, **p: real(
            topic, seed=seed, difficulty=difficulty if difficulty == "exam" else "easy", **p
        ),
    )
    with pytest.raises(AssertionError, match="draw the same questions"):
        sc.check_escalation(EXPR)


def test_text_quality_notices_a_programmer_artefact_in_the_prompt(monkeypatch):
    def leak(q):
        q.text += " Find 6*x."
        return q

    _patch_make(monkeypatch, leak)
    with pytest.raises(AssertionError, match="star_operator"):
        sc.check_options(EXPR)


def test_number_options_notice_a_distractor_with_the_wrong_shape(monkeypatch):
    def tell(q):
        q.distractors[0] = q.answer_text + ".5" if "." not in q.answer_text else q.answer_text.split(".")[0]
        return q

    _patch_make(monkeypatch, tell)
    with pytest.raises(AssertionError, match="vs distractor|differ in places"):
        sc.check_options(NUMBER)


def test_grader_notices_a_false_accept(monkeypatch):
    monkeypatch.setitem(
        grading.GRADERS, "expression", lambda q, given: grading.GradeResult(correct=True, canonical=q.answer_text)
    )
    with pytest.raises(AssertionError, match="false_accepts"):
        sc.check_grader(EXPR)


def test_grader_notices_a_false_reject(monkeypatch):
    monkeypatch.setitem(
        grading.GRADERS, "expression", lambda q, given: grading.GradeResult(correct=False, canonical=q.answer_text)
    )
    with pytest.raises(AssertionError, match="false_rejects"):
        sc.check_grader(EXPR)


def test_shape_notices_a_missing_prerequisite_and_a_duplicate_source():
    sk = sc.skill(EXPR)
    original = list(sk.prerequisites), list(sk.sources)
    try:
        sk.prerequisites.append("no-such-skill")
        with pytest.raises(AssertionError, match="prerequisite"):
            sc.check_shape(EXPR)
        sk.prerequisites[:] = original[0]
        sk.sources.append(sk.sources[0])
        with pytest.raises(AssertionError, match="duplicate source"):
            sc.check_shape(EXPR)
    finally:
        sk.prerequisites[:] = original[0]
        sk.sources[:] = original[1]
