"""Tests for the second exam source of `binomial-point-probability` (`exd_x_r3a_binomial_point_wide`, n 12 to 25),
with the answers re-derived from the rendered stem."""

from __future__ import annotations

import math
import re
from fractions import Fraction

from tests import r3a_support as sup

SKILL = "binomial-point-probability"
EXAM = "exd_x_r3a_binomial_point_wide"
OLD_EXAM = "exd_x_h2_binomial_point_probability"


def _n_p(text):
    n = int(re.search(r"\d+", text).group())
    p = Fraction(re.search(r"probability (0\.\d+)", text).group(1))
    return n, p


def test_wired_beside_the_old_exam_source_which_is_untouched():
    sources = sup.skill(SKILL).sources
    assert [(s.topic, s.difficulty) for s in sources] == [
        ("binomial_probability", "easy"),
        (OLD_EXAM, "exam"),
        (EXAM, "exam"),
    ]
    assert dict(sources[2].params) == {"tier": "exam"} and list(sources[2].stages) == ["exam"]
    assert sources[2].key == "wide-exam" and sources[1].key != sources[2].key


def test_uses_n_beyond_the_old_exam_tier():
    ns = {_n_p(q.text)[0] for q in sup.questions(EXAM, 200)}
    assert min(ns) >= 12 and max(ns) <= 25 and max(ns) > 9 and len(ns) >= 8


def test_exam_shape():
    sup.check_marks(EXAM, [1, 1, 2, 3], 40)
    assert all(
        [p.follow_through is not None for p in x.parts] == [False, True, True, False] for x in sup.questions(EXAM, 40)
    )
    sup.check_clean_and_deterministic(EXAM)


def test_exam_keys_match_independent_derivation_over_300_seeds():
    bad = []
    for q in sup.questions(EXAM):
        n, p = _n_p(q.text)
        masses = [Fraction(math.comb(n, k)) * p**k * (1 - p) ** (n - k) for k in range(n + 1)]
        mode = masses.index(max(masses))
        assert masses.count(max(masses)) == 1
        got = sup.canonical(q)
        want = {
            "a": f"{float(n * p):.2f}",
            "b": f"{math.sqrt(float(n * p * (1 - p))):.2f}",
            "c": str(mode),
            "d": f"{float(masses[mode]):.4f}",
        }
        if got != want:
            bad.append((q.text, got, want))
    assert bad == [], bad[:2]


def test_answer_space_grading_and_tags():
    sup.check_answer_space(EXAM)
    sup.check_grading(SKILL, EXAM)
    sup.check_tagged(EXAM)


def test_follow_through_from_a_wrong_mean():
    def replies(q):
        n, p = _n_p(q.text)
        wrong = q.parts[0].distractors[0]
        sd = math.sqrt(float(Fraction(wrong) * (1 - p)))
        mode = math.floor(float(Fraction(wrong) + p))
        return {"a": wrong, "b": f"{sd:.2f}", "c": str(mode), "d": q.parts[3].answer_text}, {"b", "c", "d"}, {"a"}

    sup.check_follow_through(SKILL, EXAM, replies, at_least=10)
