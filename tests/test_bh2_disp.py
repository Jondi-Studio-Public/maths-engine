"""Regression tests for the round-2 display pipeline fixes (`mathgen/core.py`: `_learner_text`, `display_question`).

Three kinds of check: exact rewrites of the strings the bug hunters reported, idempotence (a second pass changes
nothing), and a property test over rendered questions that no rewrite introduces a digit or changes a key.
"""

import re
from collections import Counter

import pytest

from mathgen import engine
from mathgen.core import _learner_text, display_question

_SKILLS = None


def _skill(skill_id):
    global _SKILLS
    if _SKILLS is None:
        _SKILLS = {s.id: s for s in engine.load_skills()[1]}
    return _SKILLS[skill_id]


WORKING = [
    # D1: a collection label "A" is not an article
    ("Add up each collection: A 85c, B 85c.", "Add up each collection: A 85c, B 85c."),
    ("Add up each collection: A 85c, B $1.", "Add up each collection: A 85c, B $1."),
    ("A 85c coin", "An 85c coin"),
    ("a 85 cm rod", "an 85 cm rod"),
    ("a $18 setup fee and a $8,500 holiday", "an $18 setup fee and an $8,500 holiday"),
    ("a $180 fee and a $80,000 car", "a $180 fee and an $80,000 car"),
    ("does a octagonal pyramid have", "does an octagonal pyramid have"),
    ("price of a ice cream", "price of an ice cream"),
    # D2: the collapse is conservative
    ("(c) 10 = 10^1, which is 1 step", "(c) 10 = 10^1, which is 1 step"),
    ("(b) Using A: -10 = -10 × (1) + c, so c = 0.", "(b) Using A: -10 = -10 × 1 + c, so c = 0."),
    ("16 - 15 = 1 km, so 1/18 = 1/18.", "16 - 15 = 1 km, so 1/18."),
    ("A single recurring digit gives 5/9 = 5/9.", "A single recurring digit gives 5/9."),
    ("n + 3 = 3 = 4", "n + 3 = 3 = 4"),
    ("x = 8/1 = 8", "x = 8"),
    ("n = 8 ÷ 1 = 8", "n = 8"),
    # R1/B2/B3
    ("δy = k((-4)² - 2²) + -(-4 - 2) = -30.", "δy = k((-4)² - 2²) - (-4 - 2) = -30."),
    ("so -20 = k × -4.", "so -20 = k × (-4)."),
    # (b) typography
    ("7 symbols x 5 = 35", "7 symbols × 5 = 35"),
    ("1/2 x 12 x x", "1/2 × 12 × x"),
    ("1.5 x IQR", "1.5 × IQR"),
    ("20 x (8-3)", "20 × (8-3)"),
    ("f'(x) = L x 9.8^x", "f'(x) = L × 9.8^x"),
    ("x <= 3 and y >= 2, (x, y) -> (4x, -2y)", "x ≤ 3 and y ≥ 2, (x, y) → (4x, -2y)"),
    (
        "perimeter_mm = 27144, surface_area = 3, area_cm2 = 4",
        "perimeter (mm) = 27144, surface area = 3, area (cm²) = 4",
    ),
    ("log_{6}(x + 13)", "log_6(x + 13)"),
    ("11a^2 m^3 and 2 m^3", "11a^2 m^3 and 2 m³"),
    ("1 x t", "t"),
    ("intercept = 65.0 - 2.5×2 = 60", "intercept = 65 - 2.5 × 2 = 60"),
    # (c4) cosmetics
    ("f(x) = -x^3 + 15·x^2 - x + 0", "f(x) = -x^3 + 15x^2 - x"),
    ("Sum = 13/2 × (2 × 133 + 12 × (7))", "Sum = 13/2 × (2 × 133 + 12 × 7)"),
    ("Area = 1/2 × 16/9 × 64 = 512/9", "Area = (1/2) × (16/9) × 64 = 512/9"),
    ("63 ÷ 1/567", "63 ÷ (1/567)"),
    ("5.5/4%", "(5.5/4)%"),
    ("z-bar", "z̄"),
    (
        "A = 18,000 × 1.00875^20 = $21426.12 from a $11000 loan",
        "A = 18,000 × 1.00875^20 = $21,426.12 from an $11,000 loan",
    ),
    ("5 × 6 - 1 × 4 - 3 × (11 − 1) = -4", "5 × 6 - 1 × 4 - 3 × (11 - 1) = -4"),
    ("Cost: 870 × $2.00 = $1740.0", "Cost: 870 × $2.00 = $1,740.00"),
    ("Factorise: (x - 0)(x - 5) = 0", "Factorise: x(x - 5) = 0"),
    ("1 cents and 1 leaves", "1 cent and 1 leaf"),
    ("y = 5/2x", "y = (5/2)x"),
    ("its velocity is -1d m/s", "its velocity is -d m/s"),
    ("4.0 and 2.50", "4 and 2.50"),
    ("round 4.0 to 1 d.p.", "round 4.0 to 1 d.p."),
]


@pytest.mark.parametrize("raw, expected", WORKING)
def test_working_rewrites(raw, expected):
    assert _learner_text(raw, working=True) == expected


def test_precision_spelling_in_the_question_is_kept():
    assert _learner_text("so x = 4.0", working=True, keep={"4.0"}) == "so x = 4.0"
    assert _learner_text("so x = 4.0", working=True) == "so x = 4"


@pytest.mark.parametrize("raw, _expected", WORKING)
def test_rewrites_are_idempotent(raw, _expected):
    once = _learner_text(raw, working=True)
    assert _learner_text(once, working=True) == once


def test_stem_is_not_collapsed_or_stripped_of_precision():
    assert _learner_text("Find 2 x (x+1)") == "Find 2 x (x+1)"
    assert _learner_text("Round 4.0 kg") == "Round 4.0 kg"
    assert _learner_text("(1) and (2) are labels") == "(1) and (2) are labels"


def test_restated_lines_are_dropped_from_working_not_left_as_fragments():
    from mathgen.core import _working_lines

    assert _working_lines(["764 = 764.", "Then 5 + 5 = 10."], set()) == ["Then 5 + 5 = 10."]
    assert _working_lines(["(c) 2,000,050 = 2,000,050."], set()) == ["(c) 2,000,050."]  # all there is: kept


_SAMPLE = [
    ("f10-money-compare-collections", 0),
    ("f10-large-number-naming", 3),
    ("f10-number-represent-order-to-1000", 3),
    ("f10-logarithmic-scales", 3),
    ("gradient-from-graph", 3),
    ("recurring-decimal-to-fraction", 0),
    ("rectangle-area-perimeter-three-units", 0),
    ("identify-like-terms", 2),
    ("two-step-equations", 3),
    ("annulus-cross-section-area", 0),
    ("solve-quadratics-factorable", 0),
    ("calculate-overtime", 0),
]


def _digits(text):
    return Counter(re.findall(r"\d", text))


@pytest.mark.parametrize("skill_id, source", _SAMPLE)
def test_rendered_questions_gain_no_digit_and_keep_the_key(skill_id, source):
    skill = _skill(skill_id)
    for seed in range(40):
        q = engine.make_for_skill(skill, seed, source=source)
        shown = display_question(q)
        # display never changes the stored question, and options/key stay in step
        assert shown["answer_text"] is not None
        assert len(shown["options"]) == len(shown["distractors"]) + 1 or not q.distractors
        raw_working = " ".join(q.working or [])
        new_working = " ".join(shown["working"])
        gained = _digits(new_working) - _digits(raw_working)
        # only a money pad ("$1.5" -> "$1.50") may add a zero; nothing else is added
        assert not (set(gained) - {"0"}), (skill_id, seed, gained)
        for line in shown["working"]:
            assert not re.fullmatch(r"(?:\([a-z]\) )?[\d,./]+\.?", line) or len(shown["working"]) == 1, line
            assert not re.search(r"\bA n \d|\bAn \d+c, B", line), line
        for text in [shown["text"], *shown["options"], *shown["working"]]:
            assert "<=" not in text and ">=" not in text and "->" not in text
            assert not re.search(r"[a-z]{3,}_(?:mm|cm|m)\d?\b", text), text


def test_display_is_idempotent_over_sample():
    for skill_id, source in _SAMPLE:
        skill = _skill(skill_id)
        for seed in range(10):
            shown = display_question(engine.make_for_skill(skill, seed, source=source))
            for line in shown["working"]:
                assert _learner_text(line, working=True) == line, (skill_id, seed, line)
            for text in [shown["text"], shown["answer_text"], *shown["options"]]:
                assert _learner_text(text) == text, (skill_id, seed, text)


@pytest.mark.parametrize(
    "skill_id, source",
    [("f10-money-compare-collections", 0), ("recurring-decimal-to-fraction", 0), ("gradient-from-graph", 3)],
)
def test_displayed_key_and_distractors_grade_as_before(skill_id, source):
    skill = _skill(skill_id)
    for seed in range(20):
        q = engine.make_for_skill(skill, seed, source=source)
        shown = display_question(q)
        assert engine.grade(skill, q, q.answer_text).correct
        if not q.parts:
            assert engine.grade(skill, q, shown["answer_text"]).correct, (seed, shown["answer_text"])
            for d in shown["distractors"]:
                assert not engine.grade(skill, q, d).correct, (seed, d)


def test_a_collection_label_keeps_its_article_in_the_real_skill():
    skill = _skill("f10-money-compare-collections")
    for seed in range(10):
        shown = display_question(engine.make_for_skill(skill, seed, source=0))
        assert not any(re.search(r"\bAn \d+c?,", w) for w in shown["working"]), shown["working"]
