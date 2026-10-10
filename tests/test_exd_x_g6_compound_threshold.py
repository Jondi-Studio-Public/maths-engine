"""Tests for the exam-tier compounding source (``mathgen/generators/exd_x_g6_compound_threshold.py``)."""

from __future__ import annotations

import math
import re

from mathgen import make
from mathgen.grading.typed import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_g6_compound_threshold"
_PARSE = re.compile(
    r"(?:contains|numbers|population of) (\d+)\b.*?(grows|increases|decreases|falls|declines) by (\d+)%.*?"
    r"first (exceed|fall below) (\d+)\?"
)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def test_variety_and_new_shapes_over_hard_tier():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make("p10c_population_compounding", seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3
    assert len(exam - hard) >= 3


def test_no_text_defects_and_nothing_financial():
    for s in range(100):
        q = _exam(s)
        assert find_text_defects(q) == []
        low = (q.text + " ".join(q.working)).lower()
        assert "$" not in low and "interest" not in low and "loan" not in low


def test_deterministic():
    for s in (0, 1, 5, 42):
        assert _exam(s).to_json() == _exam(s).to_json()


def test_both_directions_occur():
    kinds = {"exceed" in _exam(s).text for s in range(60)}
    assert kinds == {True, False}


def test_oracle_by_logarithms():
    bad = []
    for s in range(300):
        q = _exam(s)
        m = _PARSE.search(q.text)
        assert m, q.text
        p0, verb, g, mode, thr = int(m[1]), m[2], int(m[3]), m[4], int(m[5])
        up = mode == "exceed"
        assert up == (verb in ("grows", "increases"))
        ratio = (100 + g if up else 100 - g) / 100
        n = math.floor(math.log(thr / p0) / math.log(ratio)) + 1
        # the threshold must sit clear of the exact values either side of the crossing
        before = p0 * ratio ** (n - 1)
        after = p0 * ratio**n
        lo, hi = sorted((before, after))
        if n != q.answer or not (lo + 0.5 < thr < hi - 0.5):
            bad.append((s, n, q.answer))
    assert bad == []


def test_grader_accepts_key_and_rejects_distractors():
    grade = GRADERS["number"]
    bad = []
    for s in range(150):
        q = _exam(s)
        if not grade(q, q.answer_text).correct or len(q.distractors) < 2:
            bad.append((s, "key"))
        bad += [(s, d) for d in q.distractors if grade(q, d).correct]
        bad += [(s, w) for w in (q.answer - 1, q.answer + 1, 0) if grade(q, str(w)).correct]
    assert bad == []
