"""Exam-tier tests for the two Markov chain skills (`exd_x_s2_markov_step`, `exd_x_s2_markov_stationary`).

The oracle parses the RENDERED PROMPT and recomputes the key with exact fractions; it shares no code with the
generator module.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

STEP, STAT = "exd_x_s2_markov_step", "exd_x_s2_markov_stationary"
_FRAC = r"(\d+(?:/\d+)?)"


def _q(topic, seed):
    return make(topic, seed=seed, difficulty="exam")


def _fr(s):
    return Fraction(s)


def _stat_hard(seed):
    return make("markov_two_state_stationary", seed=seed, difficulty="hard", kind="find_transition")


def _step_hard(seed):
    return make("markov_two_state_step", seed=seed, difficulty="hard", kind="find_initial")


def test_no_text_defects():
    for topic in (STEP, STAT):
        for seed in range(60):
            q = _q(topic, seed)
            assert find_text_defects(q) == [], (topic, seed, q.text)


def test_shape_variety_and_novelty_over_hard():
    for topic, hard in ((STEP, _step_hard), (STAT, _stat_hard)):
        exam = {prompt_shape(_q(topic, s)) for s in range(60)}
        hard_shapes = {prompt_shape(hard(s)) for s in range(60)}
        assert len(exam) >= 3, exam
        assert exam - hard_shapes, (exam, hard_shapes)


def test_deterministic():
    for topic in (STEP, STAT):
        for seed in (0, 1, 7, 99):
            assert _q(topic, seed).to_json() == _q(topic, seed).to_json()


def _step_oracle(text):
    p = _fr(re.search(r"0→1\)=" + _FRAC, text).group(1)) if "P(0→1)=" in text else None
    if p is None:
        p = _fr(re.search(r"idle to running with probability " + _FRAC, text).group(1))
        q = _fr(re.search(r"running to idle with probability " + _FRAC, text).group(1))
        v = _fr(re.findall(r"probability " + _FRAC, text)[-1])
    else:
        q = _fr(re.search(r"1→0\)=" + _FRAC, text).group(1))
        v = _fr(re.findall(r"\)=" + _FRAC, text)[-1])
    lam = 1 - p - q
    if re.search(r"(Find P\(X₀=1\)\.|initial probability|at the start\?)", text):
        return (v - p * (1 + lam)) / (lam * lam)
    x = v
    for _ in range(2):
        x = (1 - x) * p + x * (1 - q)
    return x


def test_step_oracle_300_seeds():
    bad = []
    for seed in range(300):
        q = _q(STEP, seed)
        got = _step_oracle(q.text)
        if got != Fraction(q.answer["p"], q.answer["q"]):
            bad.append((seed, q.text, got, q.answer_text))
    assert bad == [], bad[:3]


def _stat_oracle(text):
    p = _fr(re.search(r"P\(0→1\)=" + _FRAC, text).group(1))
    if "stationary probability of state 1 is" in text or "long-run probability" in text:
        pi1 = _fr(re.search(r"(?:is|probability) " + _FRAC + r"(?:\.| of being)", text).group(1))
        pi0 = 1 - pi1
        q = pi0 * p / pi1
    else:
        q = _fr(re.search(r"P\(1→0\)=" + _FRAC, text).group(1))
        pi1 = p / (p + q)
        pi0 = 1 - pi1
    if "different" in text or "change the state" in text or "state changes" in text:
        return pi0 * p + pi1 * q
    assert "from state 0 to state 1" in text, text
    return pi0 * p


def test_stationary_oracle_300_seeds():
    bad = []
    for seed in range(300):
        q = _q(STAT, seed)
        got = _stat_oracle(q.text)
        if got != Fraction(q.answer["p"], q.answer["q"]):
            bad.append((seed, q.text, got, q.answer_text))
    assert bad == [], bad[:3]


def test_grader_accepts_key_and_rejects_distractors():
    grade = GRADERS["fraction"]
    for topic in (STEP, STAT):
        for seed in range(150):
            q = _q(topic, seed)
            assert grade(q, q.answer_text).correct, (topic, seed)
            assert len(q.distractors) >= 2, (topic, seed, q.distractors)
            for d in q.distractors:
                assert not grade(q, d).correct, (topic, seed, d, q.answer_text)
