"""Tests the skill `ucv1-proportion-z-test`: an independent oracle that re-derives every key from the question TEXT
(exact fractions, the normal tail from `math.erfc`, and numpy), the typed grader on every key and distractor, and the
four-part exam chain with follow-through."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import numpy as np
import pytest

from tests import fy1ms_support as T

SKILL_ID = "ucv1-proportion-z-test"
TOPIC = "fy1ms_z_test_proportion"
EXAM = "exd_x_fy1ms_z_test_proportion"
SK = T.load(SKILL_ID)
TIERS = ("easy", "medium", "hard")


def _upper_tail(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def _p0(text: str) -> Fraction:
    return Fraction(re.search(r"H0: p = (0\.\d+)", text).group(1))


def _sample(text: str):
    """(n, p-hat) from either 'In a random sample of N things, X ...' or the percentage form."""
    m = re.search(r"random sample of (\d+) [a-z]+, (\d+(?:\.\d)?)(%)? ", text)
    n, value, percent = int(m.group(1)), Fraction(m.group(2)), m.group(3)
    return n, (value / 100 if percent else value / n)


def _se(p0: Fraction, n: int) -> float:
    return float(np.sqrt(float(p0) * (1 - float(p0)) / n))


def oracle(q) -> str:
    text = q.text
    if "find the p-value" in text:
        z = float(re.search(r"statistic z = (-?\d+\.\d\d)", text).group(1))
        alt = re.search(r"H1: p ([≠><])", text).group(1)
        value = {"≠": 2 * _upper_tail(abs(z)), ">": _upper_tail(z), "<": _upper_tail(-z)}[alt]
        return T.rnd(Fraction(value).limit_denominator(10**15), 4)
    p0 = _p0(text)
    if "smallest value of the sample proportion" in text or "smallest number" in text:
        n = int(re.search(r"random sample of (\d+)", text).group(1))
        edge = float(p0) + 1.645 * _se(p0, n)
        if "smallest value" in text:
            return T.rnd(edge, 4)
        return str(math.floor(edge * n) + 1)
    n, phat = _sample(text)
    if "standard error" in text:
        return T.rnd(_se(p0, n), 4)
    if "difference p̂ − p0" in text:
        return T.rnd(phat - p0, 4)
    return T.rnd(float(phat - p0) / _se(p0, n), 2)


def test_wiring():
    T.assert_wiring(SK, TOPIC, [EXAM])


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_the_independent_derivation(tier):
    bad = [s for s in range(100) if oracle(T.single(TOPIC, tier, s)) != T.single(TOPIC, tier, s).answer_text]
    assert bad == []


def test_the_normal_approximation_is_always_valid():
    for tier in TIERS:
        for s in range(60):
            text = T.single(TOPIC, tier, s).text
            n = int(re.search(r"sample of (\d+)", text).group(1)) if "sample of" in text else None
            if n:
                p0 = _p0(text)
                assert n * p0 >= 10 and n * (1 - p0) >= 10, (tier, s)


def test_tiers_use_their_shapes():
    def shape(tier, s):
        text = T.single(TOPIC, tier, s).text
        if "find the p-value" in text:
            return "p-value"
        if "smallest value of the sample proportion" in text:
            return "cutoff"
        if "smallest number" in text:
            return "count"
        if "standard error" in text:
            return "se"
        if "difference p̂ − p0" in text:
            return "diff"
        return "z-percent" if "%" in text.split("The sample is large")[0].split("random sample of")[1] else "z"

    assert {shape("easy", s) for s in range(100)} == {"se", "diff"}
    assert {shape("medium", s) for s in range(100)} == {"z", "z-percent"}
    assert {shape("hard", s) for s in range(100)} == {"p-value", "cutoff", "count"}


def test_p_value_questions_cover_every_tail_and_both_signs():
    seen = set()
    for s in range(200):
        text = T.single(TOPIC, "hard", s).text
        if "find the p-value" in text:
            alt = re.search(r"H1: p ([≠><])", text).group(1)
            seen.add((alt, "-" in re.search(r"z = (-?\d)", text).group(1)))
    assert {a for a, _ in seen} == {"≠", ">", "<"}
    assert {neg for _, neg in seen} == {True, False}


@pytest.mark.parametrize("tier", TIERS)
def test_grading_and_diagnosis(tier):
    T.assert_single_grading(SK, TOPIC, tier)


@pytest.mark.parametrize("tier", TIERS)
def test_clean_text_variety_and_determinism(tier):
    T.assert_clean_variety_and_determinism(TOPIC, tier, distinct_prompts=40, distinct_answers=20)


# --- the exam chain --------------------------------------------------------------------------------------------


def exam_oracle(q):
    text = q.stem
    p0 = _p0(text)
    n, phat = _sample(text)
    se = _se(p0, n)
    z = float(phat - p0) / se
    p_value = 2 * _upper_tail(abs(z))
    verdict = (
        "Reject H0, because the p-value is less than"
        if p_value < 0.05
        else "Do not reject H0, because the p-value is greater than"
    )
    statements = re.findall(r"(\d)\. (.*?)(?= \d\. |$)", q.parts[3].prompt.split("correct statement.")[1].strip())
    right = [i for i, body in statements if body.startswith(verdict)]
    assert len(right) == 1, q.parts[3].prompt
    return dict(a=T.rnd(se, 4), b=T.rnd(z, 2), c=T.rnd(Fraction(p_value).limit_denominator(10**15), 3), d=right[0])


def test_exam_shape():
    T.assert_exam_shape(EXAM, ["a", "b", "c", "d"], [1, 2, 2, 1], [False, True, True, False])


def test_exam_keys_match_the_independent_derivation():
    for seed in range(100):
        q = T.exam(EXAM, seed)
        assert {p.label: p.answer_text for p in q.parts} == exam_oracle(q), seed


def test_exam_reaches_both_conclusions():
    verdicts = {T.exam(EXAM, s).parts[3].working[0].split()[-3] for s in range(40)}
    assert verdicts == {"less", "greater"}, verdicts


def test_exam_grading_distractors_and_notes():
    T.assert_exam_grading(SK, EXAM)


def test_follow_through_from_a_wrong_standard_error_and_a_wrong_statistic():
    T.assert_follow_through_credited(SK, EXAM, minimum=60)


def test_exam_answer_space():
    T.assert_exam_variety(EXAM, seeds=range(200), distinct=60, repeat=12)
    assert len({T.exam(EXAM, s).stem for s in range(60)}) >= 50
