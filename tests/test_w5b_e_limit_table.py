"""Tests `e-as-a-limit-from-a-table` (topic `w5b_e_limit_table`) and its exam chain `exd_x_w5b_e_limit_table`.

The oracle reads the base(s) and every table cell back out of the question TEXT and re-derives the key from the
natural logarithm (the limit of (a^h - 1)/h is ln a), never from the displayed cells alone.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import pytest

from mathgen import make
from mathgen.engine import grade
from mathgen.misconceptions import CATALOGUE
from fractions import Fraction

from mathgen.core import round_half_up
from tests import exd_x_r3b_support as S

TOPIC = "w5b_e_limit_table"
EXAM = "exd_x_w5b_e_limit_table"
SKILL_ID = "e-as-a-limit-from-a-table"
_SKILL = S.skill(SKILL_ID)
_ENTRY = re.compile(r"a = (\d+\.\d+) \(((?:h = [\d.]+: [\d.]+(?:, )?)+)\)")
_CELL = re.compile(r"h = ([\d.]+): ([\d.]+)")


def _entries(text):
    """{base: {h: cell}} for every `a = 2.70 (h = ...: ..., ...)` entry in the text."""
    return {float(a): {h: float(v) for h, v in _CELL.findall(cells)} for a, cells in _ENTRY.findall(text)}


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _kind(q):
    if "interpolat" in q.text:
        return "interpolate"
    if re.search(r"\(\d+(\.\d+)?\^h - 1\)/h|y = \d+(\.\d+)?\^x", q.text):
        return "scale"
    return "choose"


def test_skill_wiring():
    assert _SKILL.answer == "number"
    assert [(s.topic, s.difficulty) for s in _SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    S.check_wiring(EXAM, SKILL_ID)


def test_cells_match_an_independent_evaluation():
    for tier in ("easy", "medium", "hard"):
        for seed in range(40):
            q = _q(tier, seed)
            for base, cells in _entries(q.text).items():
                for h, cell in cells.items():
                    exact = math.expm1(float(h) * math.log(base)) / float(h)
                    assert abs(cell - exact) < 6e-5, (q.text, base, h)


@pytest.mark.parametrize("tier,rows", [("easy", 2), ("medium", 3)])
def test_choice_key_is_the_base_nearest_e(tier, rows):
    places = Counter()
    for seed in range(150):
        q = _q(tier, seed)
        entries = _entries(q.text)
        assert len(entries) in (4, 5) and all(len(c) == rows for c in entries.values()), q.text
        best = min(entries, key=lambda a: abs(math.log(a) - 1))
        assert float(q.answer_text) == best, q.text
        # the answer is never the first or last base, so position is no clue
        assert best not in (min(entries), max(entries))
        # the displayed last row agrees with ln a
        assert best == min(entries, key=lambda a: abs(entries[a]["0.001"] - 1))
        places[len(q.answer_text.split(".")[1])] += 1
    assert set(places) <= {1, 2}


def test_medium_coarse_row_points_the_wrong_way():
    for seed in range(150):
        q = _q("medium", seed)
        entries = _entries(q.text)
        coarse = min(entries, key=lambda a: abs(entries[a]["0.1"] - 1))
        assert coarse != float(q.answer_text), q.text
        assert f"{coarse:.{len(q.answer_text.split('.')[1])}f}" in q.distractors or len(q.distractors) >= 2


def test_easy_has_no_coarse_row():
    for seed in range(40):
        assert "h = 0.1:" not in _q("easy", seed).text


def test_hard_has_two_shapes_and_each_key_is_right():
    kinds = Counter()
    for seed in range(200):
        q = _q("hard", seed)
        kind = _kind(q)
        kinds[kind] += 1
        entries = _entries(q.text)
        if kind == "interpolate":
            lo, hi = sorted(entries)
            target = float(re.search(r"(?:limit is|exactly|limit equals|gradient) (\d\.?\d*)", q.text).group(1))
            assert target != 1 and math.log(lo) < target < math.log(hi)
            exact = lo + (hi - lo) * (target - math.log(lo)) / (math.log(hi) - math.log(lo))
            assert abs(float(q.answer_text) - exact) < 0.2 * (hi - lo), q.text
            a, b = entries[lo]["0.001"], entries[hi]["0.001"]
            assert abs(float(q.answer_text) - (lo + (hi - lo) * (target - a) / (b - a))) < 0.0051
            assert abs(float(q.answer_text) - math.e) > 0.1, "the answer must not be recallable as e"
        else:
            base = float(re.search(r"(?i)for a = (\d+)", q.text).group(1))
            target = float(re.search(r"(?:\(|y = )(\d+(?:\.\d+)?)\^(?:h|x)", q.text.split("estimate")[-1]).group(1))
            k = math.log(target) / math.log(base)
            assert abs(float(q.answer_text) - k * math.log(base)) < 0.0051, q.text
            assert abs(k - round(k)) < 1e-9 or abs(k - 0.5) < 1e-9
    assert set(kinds) == {"interpolate", "scale"}, kinds


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_every_distractor_is_wrong_tagged_and_the_same_length(tier):
    for seed in range(100):
        q = _q(tier, seed)
        assert grade(_SKILL, q, q.answer_text).correct, (tier, seed)
        places = len(q.answer_text.split(".")[1])
        assert len(q.distractors) >= 2
        for d in q.distractors:
            assert not grade(_SKILL, q, d).correct, (tier, seed, d)
            assert len(d.split(".")[1]) == places, (tier, seed, d)
            assert q.option_notes.get(d) in CATALOGUE


def test_hard_shape_is_new():
    from mathgen.text_quality import prompt_shape

    lower = {prompt_shape(_q(t, s)) for t in ("easy", "medium") for s in range(60)}
    hard = {prompt_shape(_q("hard", s)) for s in range(60)}
    assert hard - lower


# -- exam chain ---------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def questions():
    return S.exam_questions(EXAM, 250)


def test_exam_structure_and_shapes():
    S.check_structure(EXAM, "abcd", [1, 1, 2, 2])
    S.check_shapes(EXAM, SKILL_ID)


def test_exam_keys_match_the_logarithm(questions):
    for q in questions:
        entries = _entries(q.stem)
        p, qq = sorted(entries)
        target = float(re.search(r"limit of \(a\^h - 1\)/h is (\S+), to 2", q.parts[2].prompt).group(1))
        assert target != 1 and math.log(p) < target < math.log(qq)
        a, b, c, d = (float(part.answer_text) for part in q.parts)
        assert a == round(math.log(p), 2) and b == round(math.log(qq), 2), q.stem
        fa, fb, fp, fq = (Fraction(str(v)) for v in (a, b, p, qq))
        assert Fraction(str(c)) == round_half_up(fp + (fq - fp) * (Fraction(str(target)) - fa) / (fb - fa), 2), q.stem
        assert abs(c - math.e) > 0.1
        assert Fraction(str(d)) == round_half_up(-1 / fb, 2), q.stem


def test_exam_grading_tags_and_space():
    S.check_grading(EXAM, SKILL_ID, 120)
    S.check_tags(EXAM)
    S.check_answer_space(EXAM)
    S.check_clean_and_deterministic(EXAM)


def test_follow_through_from_wrong_limits(questions):
    checked = 0
    for q in questions[:120]:
        entries = _entries(q.stem)
        p, qq = sorted(entries)
        a, b = float(q.parts[0].distractors[0]), float(q.parts[1].distractors[0])
        fa, fb, fp, fq = (Fraction(str(v)) for v in (a, b, p, qq))
        tg = Fraction(str(re.search(r"limit of \(a\^h - 1\)/h is (\S+), to 2", q.parts[2].prompt).group(1)))
        if not (fa < tg < fb):
            continue
        raw = (fp + (fq - fp) * (tg - fa) / (fb - fa)) * 100
        if abs(raw - int(raw)) in (Fraction(1, 2),) or abs(-100 / fb - int(-100 / fb)) == Fraction(1, 2):
            continue
        c = round_half_up(fp + (fq - fp) * (tg - fa) / (fb - fa), 2)
        d = round_half_up(-1 / fb, 2)
        reply = {"a": f"{a:.2f}", "b": f"{b:.2f}", "c": f"{float(c):.2f}", "d": f"{float(d):.2f}"}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0 and result.parts["b"].marks == 0
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True, (q.stem, reply)
        assert result.parts["d"].marks == 2 and result.parts["d"].follow_through is True, (q.stem, reply)
        checked += 1
    assert checked >= 20


def test_hard_and_exam_options_never_look_like_e():
    """Only the easy and medium items are about e itself; no hard or exam option may be a recallable 2.72."""
    for seed in range(150):
        q = _q("hard", seed)
        for text in [q.answer_text, *q.distractors]:
            assert abs(float(text) - math.e) > 0.03, (seed, text, q.text)
    for seed in range(100):
        q = make(EXAM, seed=seed, difficulty="exam")
        for part in q.parts[2:3]:
            for text in [part.answer_text, *part.distractors]:
                assert abs(float(text) - math.e) > 0.03, (seed, text)
