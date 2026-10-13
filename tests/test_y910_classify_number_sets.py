"""Proofs for the `y910_classify_number_sets` sources of `classify-real-numbers`.

The oracle re-reads the numbers from the question text and classifies each with SymPy, a different route from
the generator's own construction.
"""

from __future__ import annotations

import re

import pytest
import sympy

from mathgen import make
from mathgen.grading import grade
from mathgen.skills import load_skills

SKILL_ID = "f10-number-set-membership"
SEEDS = range(80)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
NEW = [s for s in _SKILL.sources if s.topic == "y910_classify_number_sets"]
BY_TIER = {s.difficulty: s for s in NEW}


def _make(tier, seed):
    src = BY_TIER[tier]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _value(text: str):
    """The exact SymPy value of a displayed number, or None for a non-repeating decimal described in words."""
    text = text.strip()
    if "one more zero" in text or "counting numbers" in text:
        return None
    rec = re.match(r"0\.(\d+)\.\.\. \(the repeating (?:digit|block) is (\d+)\)", text)
    if rec:
        block = rec.group(2)
        return sympy.Rational(int(block), 10 ** len(block) - 1)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return sympy.Rational(text)
    s = text.replace("−", "-").replace("×", "*").replace("÷", "/").replace("π", "pi").replace("²", "**2")
    s = re.sub(r"∛\((-?\d+)\)", r"real_root(\1, 3)", s)
    s = re.sub(r"∛(\d+)", r"real_root(\1, 3)", s)
    s = re.sub(r"√\(([^)]*)\)", r"sqrt(\1)", s)
    s = re.sub(r"√(\d+)", r"sqrt(\1)", s)
    s = re.sub(r"(\d)\(", r"\1*(", s)
    s = re.sub(r"\)\(", ")*(", s)
    s = re.sub(r"(\d)(pi)", r"\1*\2", s)
    s = re.sub(r"(?<![\w.])(\d+)/(\d+)", r"Rational(\1, \2)", s)
    return sympy.sympify(s, locals={"real_root": sympy.real_root})


def _rational(expr) -> bool:
    s = sympy.simplify(expr)
    if s.is_Rational:
        return True
    if s.has(sympy.pi):
        return False
    return sympy.degree(sympy.minimal_polynomial(s, sympy.Symbol("x"))) == 1


def _items(text: str):
    body = text.split("listed. ", 1)[1].rsplit(". ", 2)[0]
    return re.findall(r"(?:^|; )(?:[A-F]|\(\d\)): ?(.*?)(?=; (?:[A-F]|\(\d\)):|$)", body) or re.findall(
        r"\(\d\) (.*?)(?=; \(\d\)|$)", body
    )


def test_skill_sources_and_the_old_skill_is_untouched():
    assert [s.difficulty for s in _SKILL.sources] == ["easy", "medium", "hard", "exam"]
    assert _SKILL.answer == "set"
    old = next(s for s in load_skills()[1] if s.id == "classify-real-numbers")
    assert [s.topic for s in old.sources] == ["classify_real_number", "exd_x_sc5_rational_count"]


def test_easy_set_membership_matches_the_oracle():
    for seed in SEEDS:
        q = _make("easy", seed)
        shown = re.search(r"contain the number (.*?)\? Write", q.text).group(1)
        value = _value(shown)
        if value is None:
            continue
        rational = _rational(value)
        expected = {"R"} | (
            {"Q"}
            | ({"Z"} if sympy.simplify(value).is_Integer else set())
            | ({"N"} if sympy.simplify(value).is_Integer and sympy.simplify(value) > 0 else set())
            if rational
            else {"I"}
        )
        assert set(q.answer) == expected, (seed, shown, q.answer)


def test_medium_irrational_letters_match_the_oracle():
    for seed in SEEDS:
        q = _make("medium", seed)
        pairs = re.findall(r"([A-F]): (.*?)(?=; [A-F]: |\. Which of them)", q.text)
        assert len(pairs) == 6
        expected = []
        for letter, shown in pairs:
            value = _value(shown)
            if value is None or not _rational(value):
                expected.append(letter)
        assert q.answer == expected, (seed, q.text)


def test_hard_rational_letters_match_the_oracle():
    for seed in SEEDS:
        q = _make("hard", seed)
        pairs = re.findall(r"([A-H]): (.*?)(?=; [A-H]: |\. Simplify)", q.text)
        assert len(pairs) == 8
        expected = [letter for letter, shown in pairs if (v := _value(shown)) is not None and _rational(v)]
        assert q.answer == expected, (seed, q.text)
        assert 3 <= len(expected) <= 5


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_grader_accepts_key_and_rejects_every_distractor(tier):
    kind = "set"
    for seed in SEEDS:
        q = _make(tier, seed)
        assert grade(kind, q, q.answer_text).correct
        assert len(q.distractors) >= 2
        for d in q.distractors:
            assert not grade(kind, q, d).correct, (tier, seed, d)


def test_easy_is_never_a_two_way_guess_and_set_answers_vary():
    answers = set()
    for seed in SEEDS:
        q = _make("easy", seed)
        assert len(q.distractors) == 3, (seed, q.text, q.distractors)
        answers.add(q.answer_text)
    assert len(answers) >= 4


def test_medium_has_disguised_rational_numbers_and_hard_has_simplifying_expressions():
    for seed in SEEDS:
        q = _make("medium", seed)
        assert 2 <= len(q.answer) <= 3
        assert re.search(r"√\d+|∛|recurring|repeat|\d+/\d+", q.text)
    forms = " ".join(_make("hard", s).text for s in SEEDS)
    assert "(√" in forms and "÷" in forms and "×" in forms


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_determinism(tier):
    assert _make(tier, 7).to_json() == _make(tier, 7).to_json()
