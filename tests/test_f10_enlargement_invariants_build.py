"""`f10-enlargement-invariants`: lengths scale by the factor, areas by its square, angles not at all. Answers are
re-derived here from the question text alone, by a different route from the generator's."""

import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

from mathgen import make
from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills

ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(80)
TIERS = ["easy", "medium", "hard"]
SKILL_ID = "f10-enlargement-invariants"


def _questions(tier, seeds=SEEDS):
    return [make("enlargement_invariants", seed=s, difficulty=tier) for s in seeds]


def _skill():
    _, skills = load_skills(ROOT / "skills")
    return next(s for s in skills if s.id == SKILL_ID)


def _factor(text):
    m = re.search(r"scale factor of (\d+(?:\.\d+)?|1/\d+)", text)
    return Fraction(m.group(1))


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic_with_three_distinct_tagged_distractors(tier):
    declared = {m.id for m in _skill().misconceptions}
    for seed in SEEDS:
        q = make("enlargement_invariants", seed=seed, difficulty=tier)
        assert q.to_json() == make("enlargement_invariants", seed=seed, difficulty=tier).to_json()
        assert len(q.distractors) == 3
        assert len(set(q.distractors) | {q.answer_text}) == 4
        assert set(q.option_notes) == set(q.distractors)
        assert set(q.option_notes.values()) <= declared


@pytest.mark.parametrize("tier", TIERS)
def test_real_grader_accepts_answer_and_rejects_every_distractor_and_a_wrong_value(tier):
    for q in _questions(tier):
        assert grade_number(q, q.answer_text).correct
        for d in q.distractors:
            assert not grade_number(q, d).correct
        assert not grade_number(q, str(float(q.answer_text) + 1)).correct
        assert not grade_number(q, str(float(q.answer_text) * 2)).correct


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_share_the_answers_decimal_places(tier):
    for q in _questions(tier):
        places = len(q.answer_text.partition(".")[2])
        assert all(len(d.partition(".")[2]) == places for d in q.distractors)


def test_no_exam_tier_and_three_sources():
    sk = _skill()
    assert [s.difficulty for s in sk.sources] == TIERS
    assert sk.answer == "number" and sk.status == "active"


def test_easy_answers_rederived_from_the_stem():
    kinds = Counter()
    for q in _questions("easy"):
        k = _factor(q.text)
        if "matching angle" in q.text:
            kinds["angle"] += 1
            assert int(q.answer_text) == int(re.search(r"angle of the original is (\d+)°", q.text).group(1))
        else:
            kinds["length"] += 1
            assert int(q.answer_text) == k * int(re.search(r"side of the original is (\d+) cm", q.text).group(1))
    assert all(kinds.values())


def test_medium_answers_rederived_from_the_stem():
    kinds = Counter()
    for q in _questions("medium", range(200)):
        k = _factor(q.text)
        answer = int(q.answer_text)
        dims = re.search(r"original is (\d+) cm by (\d+) cm", q.text)
        if "area of the image" in q.text and dims:
            kinds["area"] += 1
            l, w = map(int, dims.groups())
            assert answer == (k * l) * (k * w)
        elif "perimeter of the image" in q.text:
            kinds["perimeter"] += 1
            l, w = map(int, dims.groups())
            assert answer == 2 * (k * l + k * w)
        elif "times the area" in q.text:
            kinds["factor"] += 1
            assert answer == k * k
        elif "area of" in q.text:
            kinds["area_from_area"] += 1
            area = int(re.search(r"area of (\d+) cm²", q.text).group(1))
            assert answer == k * k * area
        else:
            kinds["original"] += 1
            image = int(re.search(r"side of the image is (\d+) cm", q.text).group(1))
            assert answer * k == image
    assert len(kinds) == 5


def test_hard_answers_rederived_from_the_stem():
    kinds = Counter()
    for q in _questions("hard", range(200)):
        if "area of the image" in q.text or "image has an area" in q.text:
            if "image has an area" in q.text:
                kinds["areas"] += 1
                a, b = map(int, re.search(r"area of (\d+) cm² and the image has an area of (\d+) cm²", q.text).groups())
                side = int(re.search(r"side of the original is (\d+) cm", q.text).group(1))
                factor = next(f for f in range(1, 50) if f * f * a == b)
                assert int(q.answer_text) == factor * side
            else:
                kinds["reduction"] += 1
                area = int(re.search(r"with an area of (\d+) cm²", q.text).group(1))
                m = int(re.search(r"scale factor of 1/(\d+)", q.text).group(1))
                assert Fraction(q.answer_text) == Fraction(area, m * m)
        elif "How far apart" in q.text:
            kinds["gap"] += 1
            d = int(re.search(r"P is (\d+) cm from", q.text).group(1))
            assert int(q.answer_text) == (_factor(q.text) - 1) * d
        else:
            kinds["decimal"] += 1
            assert "Give your answer to 1 decimal place" in q.text
            length = int(re.search(r"side of the original is (\d+) cm", q.text).group(1))
            assert Fraction(q.answer_text) == _factor(q.text) * length
            assert len(q.answer_text.partition(".")[2]) == 1
            assert not grade_number(q, q.answer_text + "4").correct  # more places than asked, not rounded
    assert len(kinds) == 4


@pytest.mark.parametrize("tier", TIERS)
def test_answer_pool_is_varied_and_prompts_do_not_repeat(tier):
    qs = _questions(tier, range(120))
    assert max(Counter(q.answer_text for q in qs).values()) / len(qs) <= 0.12
    assert len({q.text for q in qs}) >= 60


def test_every_generated_question_has_a_distractor_set_across_many_seeds():
    for tier in TIERS:
        for q in _questions(tier, range(3000)):
            assert len(q.distractors) == 3
