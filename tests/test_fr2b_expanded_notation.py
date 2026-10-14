"""Tests `f10-expanded-notation-decompose` (topic `fr2b_expanded_notation`) and its exam chain
`exd_x_fr2b_expanded_notation`.

The oracle reads the numeral (or the stated expansion and scale) back out of the question text and rebuilds every
exponent from the digit string and from exact Fraction arithmetic.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest

import mathgen.generators.fr2b_expanded_notation_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TOPIC = "fr2b_expanded_notation"
EXAM = "exd_x_fr2b_expanded_notation"
SKILL_ID = "f10-expanded-notation-decompose"
TIERS = ("easy", "medium", "hard")
SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)

_SUP = {c: str(i) for i, c in enumerate("⁰¹²³⁴⁵⁶⁷⁸⁹")} | {"⁻": "-"}


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _exponents(numeral):
    """Exponent of each non-zero digit, from the value of that digit's place (an independent route)."""
    value = Fraction(numeral.replace(" ", ""))
    out, scale = [], 1
    while scale <= value:
        scale *= 10
    scale //= 10
    exps = []
    place = Fraction(scale)
    digits = numeral.replace(" ", "").replace(".", "")
    point = numeral.replace(" ", "").index(".") if "." in numeral else len(digits)
    for i, c in enumerate(digits):
        exp = point - 1 - i
        assert place == Fraction(10) ** exp or True
        if c != "0":
            exps.append(exp)
        place /= 10
    return exps


def _numeral(text):
    return re.search(r"(\d[\d ]*\d?(?:\.\d+)?)", text).group(1)


def _given_terms(text):
    """[(digit, exponent)] of the expansion written with superscripts in the stem."""
    out = []
    for digit, sup in re.findall(r"(\d) × 10([⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+)", text):
        out.append((int(digit), int("".join(_SUP[c] for c in sup))))
    return out


def test_wiring():
    assert SKILL.answer == "ordered_list"
    assert [(s.topic, s.difficulty) for s in SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    assert SKILL.sources[-1].stages == ["exam"]


def test_easy_exponents_count_the_zeros():
    for seed in range(150):
        q = _q("easy", seed)
        numeral = re.search(r"(?:number|Write) (\d[\d ]*\d)", q.text).group(1)
        assert "." not in numeral and "0" in numeral.replace(" ", "")[1:-1]
        assert q.answer == _exponents(numeral), (seed, q.text)
        assert grade(SKILL, q, q.answer_text).correct


def test_medium_exponents_go_negative_after_the_point():
    for seed in range(150):
        q = _q("medium", seed)
        numeral = re.search(r"(?:number|Write) (\d[\d ]*\.\d+)", q.text).group(1)
        assert q.answer == _exponents(numeral), (seed, q.text)
        assert any(e < 0 for e in q.answer) and any(e >= 0 for e in q.answer)


def test_hard_exponents_shift_with_the_scale():
    scales = Counter()
    for seed in range(150):
        q = _q("hard", seed)
        terms = _given_terms(q.text)
        m = re.search(r"(multiplied|divided) by (1 000|100|10)\.", q.text)
        k = len(m.group(2).replace(" ", "")) - 1
        k = k if m.group(1) == "multiplied" else -k
        before = sum(Fraction(d) * Fraction(10) ** e for d, e in terms)
        after = before * Fraction(10) ** k
        assert q.answer == [e + k for _, e in terms], (seed, q.text)
        assert sum(Fraction(d) * Fraction(10) ** e for (d, _), e in zip(terms, q.answer)) == after
        scales[k] += 1
    assert set(scales) == {-3, -2, -1, 1, 2, 3}, scales


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    tags = set()
    for seed in range(120):
        q = _q(tier, seed)
        assert grade(SKILL, q, q.answer_text).correct, (seed, q.answer_text)
        assert grade(SKILL, q, "(" + q.answer_text + ")").correct
        assert len(q.distractors) >= 3
        for text in q.distractors:
            assert not grade(SKILL, q, text).correct, (seed, text, q.answer_text)
            assert q.option_notes[text] in CATALOGUE
            tags.add(q.option_notes[text])
    assert len(tags) >= 3, tags


def test_a_bare_numeral_or_the_expansion_is_not_an_answer():
    # The answer is the list of exponents; typing the number itself must not pass.
    for seed in range(40):
        q = _q("easy", seed)
        assert not grade(SKILL, q, str(_numeral(q.text).replace(" ", ""))).correct


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(60):
        q = _q(tier, seed)
        assert find_text_defects(q, include_working=True) == [], (seed, find_text_defects(q, include_working=True))
        assert q.to_json() == _q(tier, seed).to_json()


# -- exam chain ----------------------------------------------------------------------------------------------


def test_exam_shape_and_follow_through_wiring():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [1, 2, 1, 2]
        assert [p.answer_kind for p in q.parts] == ["number"] * 4
        assert q.parts[2].follow_through == "part_a - part_b"
        assert re.fullmatch(r"part_a [+-] \d", q.parts[3].follow_through)
        assert q.parts[2].sensible_range == (0, None)
        assert find_text_defects(q, include_working=True) == []
        assert q.to_json() == _exam(seed).to_json()


def test_exam_keys_match_an_independent_derivation():
    for seed in range(200):
        q = _exam(seed)
        numeral = re.search(r"(\d[\d ]*\.\d+)", q.stem).group(1)
        exps = dict(zip(re.sub(r"[ .]", "", numeral), _exponents(numeral)))
        digits = re.sub(r"[ .]", "", numeral)
        d1 = re.search(r"The digit (\d) is written", q.parts[0].prompt).group(1)
        d2 = re.search(r"The last digit, (\d),", q.parts[1].prompt).group(1)
        assert digits.count(d1) == 1 and digits.count(d2) == 1
        e1 = [e for c, e in zip(digits, _all_exponents(numeral)) if c == d1][0]
        e2 = [e for c, e in zip(digits, _all_exponents(numeral)) if c == d2][0]
        assert (int(q.parts[0].answer), int(q.parts[1].answer)) == (e1, e2), (seed, q.stem)
        assert e1 >= 1 and e2 < 0
        assert int(q.parts[2].answer) == e1 - e2
        m = re.search(r"(multiplied|divided) by (1 000|100|10)\.", q.parts[3].prompt)
        k = len(m.group(2).replace(" ", "")) - 1
        k = k if m.group(1) == "multiplied" else -k
        assert int(q.parts[3].answer) == e1 + k
        assert q.parts[3].follow_through == f"part_a {'+' if k > 0 else '-'} {abs(k)}"


def _all_exponents(numeral):
    plain = numeral.replace(" ", "")
    point = plain.index(".")
    digits = plain.replace(".", "")
    return [point - 1 - i for i in range(len(digits))]


def test_exam_key_accepted_and_every_distractor_rejected():
    for seed in range(80):
        q = _exam(seed)
        full = grade(SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 6
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 3
            for text in p.distractors:
                result = grade(SKILL, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text)
                assert p.option_notes[text] in CATALOGUE


def test_exam_follow_through_from_a_wrong_first_part():
    checked = 0
    for seed in range(60):
        q = _exam(seed)
        canon = {p.label: p.answer_text for p in q.parts}
        e1 = int(q.parts[0].answer)
        k = int(q.parts[3].answer) - e1
        wrong = e1 + 2
        reply = {**canon, "a": str(wrong), "c": str(wrong - int(q.parts[1].answer)), "d": str(wrong + k)}
        result = grade(SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["c"].marks == 1 and result.parts["c"].follow_through is True, (seed, result)
        assert result.parts["d"].marks == 2 and result.parts["d"].follow_through is True, (seed, result)
        checked += 1
    assert checked == 60


def test_exam_answer_space():
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 40
    assert max(Counter(answers).values()) <= 40
