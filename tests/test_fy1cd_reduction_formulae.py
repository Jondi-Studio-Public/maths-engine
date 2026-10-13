"""Tests `ucv1-reduction-formulae` (topic `ucv1_reduction_formulae`) and its exam chain `exd_x_fy_reduction_formulae`.

The oracle reads the integrand and the limits back out of the question TEXT and evaluates the integral numerically
with mpmath, which shares nothing with the recurrences in the generator.
"""

from __future__ import annotations

import re
from collections import Counter

import mpmath as mp
import pytest

import mathgen.generators.fy1cd_reduction_formulae_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TOPIC = "ucv1_reduction_formulae"
EXAM = "exd_x_fy_reduction_formulae"
SKILL_ID = "ucv1-reduction-formulae"
TIERS = ("easy", "medium", "hard")
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _limit(text):
    return {"π/2": mp.pi / 2, "π/4": mp.pi / 4, "π": mp.pi, "e": mp.e}.get(text) or mp.mpf(text)


def _integral_value(text, n=None):
    """Read the integral asked for out of the text and evaluate it numerically; ``n`` overrides a symbolic power."""
    m = re.search(
        r"∫ (.*?) dx from x = (.*?) to x = ([^,\s]+)", text.split("Find")[0] if "I_k" in text or "K_k" in text else text
    )
    body, lo, hi = m.group(1), _limit(m.group(2)), _limit(m.group(3))
    if "(ln x)^" in body:
        k = int(re.search(r"\^(\d+)", body).group(1))
        fn = lambda x: mp.log(x) ** k  # noqa: E731
    elif "(1 - x²)^" in body:
        k = int(re.search(r"\^(\d+)", body).group(1))
        fn = lambda x: (1 - x * x) ** k  # noqa: E731
    elif body.startswith("tan^"):
        k = int(re.search(r"\^(\d+)", body).group(1))
        fn = lambda x: mp.tan(x) ** k  # noqa: E731
    elif body.startswith(("sin^", "cos^")):
        f = mp.sin if body.startswith("sin") else mp.cos
        fn = lambda x: f(x) ** n  # noqa: E731
    else:
        a = int(re.search(r"e\^\((-?\d+)x\)", body).group(1))
        fn = lambda x: x**n * mp.exp(a * x)  # noqa: E731
    with mp.workdps(25):
        return float(mp.quad(fn, [lo, hi]))


# -- wiring -------------------------------------------------------------------------------------------------------


def test_skill_wiring():
    assert _SKILL.answer == "number"
    assert [(s.topic, s.difficulty) for s in _SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    assert _SKILL.sources[-1].stages == ["exam"]
    assert all(m.id in CATALOGUE for m in _SKILL.misconceptions)


# -- single-answer tiers ------------------------------------------------------------------------------------------


def test_easy_key_is_the_integral_of_the_asked_power():
    kinds = Counter()
    for seed in range(100):
        q = _q("easy", seed)
        n = int(re.search(r"Find I_(\d+)", q.text).group(1))
        assert 3 <= n <= 10
        assert abs(_integral_value(q.text, n) - float(q.answer)) <= 0.0005 + 1e-9, (seed, q.text, q.answer_text)
        kinds[re.search(r"∫ (\w+)\^k x dx from x = 0 to x = (\S+)", q.text).groups()] += 1
    assert set(kinds) == {("sin", "π/2"), ("cos", "π/2"), ("sin", "π")}, kinds


def test_medium_key_is_the_integral_of_the_asked_power():
    for seed in range(100):
        q = _q("medium", seed)
        n = int(re.search(r"Find K_(\d+)", q.text).group(1))
        assert 2 <= n <= 5
        assert abs(_integral_value(q.text, n) - float(q.answer)) <= 0.0005 + 1e-9, (seed, q.text, q.answer_text)


def test_hard_key_is_the_integral_and_all_three_families_appear():
    families = Counter()
    for seed in range(120):
        q = _q("hard", seed)
        assert abs(_integral_value(q.text.split("Use it to evaluate ")[1]) - float(q.answer)) <= 0.0005 + 1e-9, (
            seed,
            q.text,
            q.answer_text,
        )
        families[re.search(r"for (\w)_n", q.text).group(1)] += 1
    assert set(families) == {"T", "L", "W"} and min(families.values()) >= 20, families


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in range(100):
        q = _q(tier, seed)
        assert grade(_SKILL, q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) >= 2, seed
        for text in q.distractors:
            assert not grade(_SKILL, q, text).correct, (seed, text, q.answer_text)
            assert len(text.split(".")[1]) == 3  # same number of decimal places as the answer
            assert q.option_notes[text] in CATALOGUE
            assert text in q.distractor_values


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(60):
        q = _q(tier, seed)
        assert find_text_defects(q, include_working=True) == [], (seed, find_text_defects(q, include_working=True))
        assert q.to_json() == _q(tier, seed).to_json()


def test_tiers_differ_in_shape():
    easy = [_q("easy", s).text for s in range(30)]
    medium = [_q("medium", s).text for s in range(30)]
    hard = [_q("hard", s).text for s in range(30)]
    assert all("satisfy I_k = ((k - 1)/k)" in t for t in easy)  # a homogeneous formula, given
    assert all("satisfy K_k" in t and "e^" in t for t in medium)  # a formula with a boundary term, given
    assert not any("satisfy" in t for t in hard)  # the formula has to be derived


@pytest.mark.parametrize("tier", TIERS)
def test_answer_space(tier):
    texts = [_q(tier, s).text for s in range(120)]
    answers = [_q(tier, s).answer_text for s in range(120)]
    assert len(set(texts)) >= 20
    assert len(set(answers)) >= 8


# -- exam chain ---------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["number"] * 4
        assert [p.marks for p in q.parts] == [2, 2, 2, 2]
        assert [bool(p.follow_through) for p in q.parts] == [False, True, True, True]
        assert find_text_defects(q, include_working=True) == []
        assert q.to_json() == _exam(seed).to_json()


def _index(part):
    return int(re.search(r"K_(\d)", part.prompt).group(1))


def test_exam_keys_match_independent_derivation():
    starts = Counter()
    for seed in range(100):
        q = _exam(seed)
        a = int(re.search(r"e\^\((-?\d+)x\)", q.text).group(1))
        b = int(re.search(r"to x = (\d+)", q.text).group(1))
        assert [_index(p) for p in q.parts] == list(range(_index(q.parts[0]), _index(q.parts[0]) + 4))
        starts[_index(q.parts[0])] += 1
        for j, part in enumerate(q.parts):
            i = _index(part)
            with mp.workdps(25):
                truth = float(mp.quad(lambda x: x**i * mp.exp(a * x), [0, b]))
            places = 3 if j == 3 else 4
            assert abs(truth - float(part.answer)) <= 0.5 * 10**-places + 1e-9, (seed, i)
            assert len(part.answer_text.split(".")[1]) == places
    assert set(starts) == {0, 1}, starts


def test_exam_key_accepted_and_every_distractor_rejected():
    for seed in range(80):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 8
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2
            for text in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text)
                assert p.option_notes[text] in CATALOGUE


def test_exam_follow_through_from_a_wrong_earlier_part():
    checked = 0
    for seed in range(60):
        q = _exam(seed)
        a = int(re.search(r"e\^\((-?\d+)x\)", q.text).group(1))
        b = int(re.search(r"to x = (\d+)", q.text).group(1))
        canon = {p.label: p.answer_text for p in q.parts}
        for j in (1, 2, 3):
            earlier, mine = q.parts[j - 1], q.parts[j]
            i = _index(mine)
            places = 3 if j == 3 else 4
            for wrong in earlier.distractors:
                own = b**i * mp.exp(a * b) / a - i * float(wrong) / a
                text = f"{float(own):.{places}f}"
                if text in (mine.answer_text, wrong):  # repeating the earlier reply is not following through
                    continue
                result = grade(_SKILL, q, {**canon, earlier.label: wrong, mine.label: text})
                if result.parts[mine.label].marks == 2:
                    assert result.parts[mine.label].follow_through is True
                    checked += 1
    assert checked >= 60


def test_exam_answer_space():
    answers = [_exam(s).answer_text for s in range(200)]
    assert len(set(answers)) >= 20
    assert len({_exam(s).text for s in range(200)}) >= 20
