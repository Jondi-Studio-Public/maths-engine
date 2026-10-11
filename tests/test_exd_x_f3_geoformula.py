"""Exam tier of senior-use-the-formula-t-n-r-n-1.

The oracle re-derives the key from the rendered prompt alone with exact fractions and never imports the
generator.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.grading.typed import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f3_geoformula"
_GRADER = GRADERS["number"]


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def test_prompt_variety_and_new_shapes_over_hard():
    shapes = {prompt_shape(_q(s)) for s in range(60)}
    hard = {
        prompt_shape(make("geometric_nth_term_formula", seed=s, difficulty="hard", variant="find_n")) for s in range(60)
    }
    assert len(shapes) >= 3, shapes
    assert shapes - hard


def test_no_text_defects_and_deterministic():
    for s in range(60):
        assert find_text_defects(_q(s)) == [], s
    assert _q(8).to_json() == _q(8).to_json()


_HEAD = re.compile(r"with (positive|negative) common ratio, t(\d+) = (-?\d+) and t(\d+) = (-?\d+)\. ")
_TERM = re.compile(r"to find t(\d+)\.")
_POS = re.compile(r"Which term of the sequence equals (-?\d+)\?")


def _oracle(text):
    sign, p, tp, q, tq = _HEAD.search(text).groups()
    p, tp, q, tq = int(p), int(tp), int(q), int(tq)
    ratio = Fraction(tq, tp)
    rs = [Fraction(c) for c in range(-50, 51) if c and (c < 0) == (sign == "negative") and c ** (q - p) == ratio]
    assert len(rs) == 1, (text, rs)
    r = rs[0]
    t1 = tp / r ** (p - 1)
    m = _TERM.search(text)
    if m:
        return int(t1 * r ** (int(m.group(1)) - 1))
    target = int(_POS.search(text).group(1))
    hits = [n for n in range(1, 60) if t1 * r ** (n - 1) == target]
    assert len(hits) == 1, (text, hits)
    return hits[0]


def test_key_matches_independent_oracle_over_400_seeds():
    bad = [(s, _q(s).text) for s in range(400) if _oracle(_q(s).text) != _q(s).answer]
    assert bad == [], bad[:3]


def test_both_forms_occur():
    texts = [_q(s).text for s in range(40)]
    assert any("to find t" in t for t in texts) and any("Which term" in t for t in texts)


def test_grader_accepts_key_and_rejects_every_distractor_over_200_seeds():
    bad = []
    for s in range(200):
        q = _q(s)
        assert len(q.distractors) >= 2
        assert _GRADER(q, q.answer_text).correct, s
        bad += [(s, d) for d in q.distractors if _GRADER(q, d).correct]
    assert bad == []


def test_grader_rejects_fixed_wrong_values():
    q = _q(1)
    k = q.answer
    for wrong in (str(k + 1), str(k - 1), str(-k), "0"):
        assert not _GRADER(q, wrong).correct, wrong
