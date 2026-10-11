"""Tests for the exam-tier assignment source (``mathgen/generators/exd_x_g6_assignment_max_forbidden.py``)."""

from __future__ import annotations

import re
from itertools import permutations

from mathgen import make
from mathgen.grading.typed import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_g6_assignment_max_forbidden"


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _parse(text):
    rows = re.findall(r"Worker (\d): ((?:Task [A-E] \$\d+(?:, )?)+)", text)
    m = [[int(v) for v in re.findall(r"\$(\d+)", body)] for _w, body in rows]
    w, t = re.search(r"Worker (\d) cannot be given Task ([A-E])", text).groups()
    return m, (int(w) - 1, "ABCDE".index(t))


def test_variety_and_new_shapes_over_hard_tier():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {
        prompt_shape(make("determine_the_optimum_assignment_s_by_inspection_for", seed=s, difficulty="hard"))
        for s in range(60)
    }
    assert len(exam) >= 3
    assert len(exam - hard) >= 3


def test_no_text_defects():
    for s in range(60):
        assert find_text_defects(_exam(s)) == []


def test_deterministic():
    for s in (0, 1, 3, 77):
        assert _exam(s).to_json() == _exam(s).to_json()


def test_oracle_matches_key_and_restriction_matters():
    bad = []
    for s in range(300):
        q = _exam(s)
        m, (bw, bt) = _parse(q.text)
        assert len(m) == 5 and all(len(r) == 5 for r in m)
        totals_all = [sum(m[i][p[i]] for i in range(5)) for p in permutations(range(5))]
        totals_ok = [sum(m[i][p[i]] for i in range(5)) for p in permutations(range(5)) if p[bw] != bt]
        if max(totals_ok) != q.answer or max(totals_all) <= q.answer:
            bad.append((s, q.answer, max(totals_ok)))
    assert bad == []


def test_grader_accepts_key_and_rejects_distractors():
    grade = GRADERS["number"]
    bad = []
    for s in range(150):
        q = _exam(s)
        if not grade(q, q.answer_text).correct or len(q.distractors) < 2:
            bad.append((s, "key"))
        bad += [(s, d) for d in q.distractors if grade(q, d).correct]
        bad += [(s, w) for w in (q.answer - 1, q.answer + 1) if grade(q, str(w)).correct]
    assert bad == []
