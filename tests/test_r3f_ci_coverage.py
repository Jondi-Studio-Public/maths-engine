"""Tests for `confidence-interval-simulation-coverage`: tiers (`r3f_ci_coverage`) and the exam chain
(`exd_x_r3f_ci_coverage`), with every count and probability re-derived from the rendered stem."""

from __future__ import annotations

import math
import re

from tests import r3f_support as sup

SKILL = "confidence-interval-simulation-coverage"
TOPIC = "r3f_ci_coverage"
EXAM = "exd_x_r3f_ci_coverage"


def _table(text):
    return [(float(lo), float(hi)) for lo, hi in re.findall(r"\d+: (\d+(?:\.\d+)?) to (\d+(?:\.\d+)?)", text)]


def test_wired():
    sup.check_tier_wiring(SKILL, TOPIC, EXAM)
    sup.check_wired(SKILL, EXAM)


def test_tiers_grade_and_format():
    sup.check_tier_grading(SKILL, TOPIC)


def test_tiers_differ_in_shape():
    sup.check_tier_shapes(TOPIC)
    sup.check_tier_determinism(TOPIC)


def test_easy_keys():
    bad = []
    for q in sup.tier_questions(TOPIC, "easy", 200):
        t = q.text
        numbers = [int(v) for v in re.findall(r"\b(\d+)\b", t.split("Using z")[0].replace("Year 10", ""))]
        n, x = numbers[0], numbers[1]
        z = float(re.search(r"z = (\d+(?:\.\d+)?)", t).group(1))
        p = x / n
        me = 100 * z * math.sqrt(p * (1 - p) / n)
        want = me if "margin of error" in t.split("Using z")[1] and "upper end" not in t else 100 * p + me
        if abs(float(q.answer_text) - want) > 0.0501:
            bad.append((t, q.answer_text, want))
    assert bad == [], bad[:2]


def test_medium_keys():
    bad, seen = [], set()
    for q in sup.tier_questions(TOPIC, "medium", 200):
        t = q.text
        rows = _table(t)
        mu = int(re.search(r"μ = (\d+)", t).group(1))
        count = int(re.search(r"takes (\d+) independent", t).group(1))
        assert len(rows) == count
        if "do not contain" in t:
            seen.add("miss")
            want = sum(1 for lo, hi in rows if not lo < mu < hi)
        elif "contain μ?" in t:
            seen.add("contain")
            want = sum(1 for lo, hi in rows if lo < mu < hi)
        else:
            seen.add("above")
            want = sum(1 for lo, hi in rows if lo > mu)
        if int(q.answer_text) != want:
            bad.append((t, q.answer_text, want))
    assert seen == {"miss", "contain", "above"}
    assert bad == [], bad[:2]


def test_hard_keys():
    bad, seen = [], set()
    for q in sup.tier_questions(TOPIC, "hard", 250):
        t = q.text
        if "how many of the" in t.lower() and "expect NOT" in t:
            seen.add("expected")
            count, level = (int(v) for v in re.search(r"builds (\d+) independent (\d+)%", t).groups())
            want = count * (100 - level) / 100
            tol = 0.0501
        elif "at least one" in t:
            seen.add("some")
            count, level = (int(v) for v in re.search(r"builds (\d+) independent (\d+)%", t).groups())
            want = 1 - (level / 100) ** count
            tol = 0.00006
        elif "all " in t:
            seen.add("all")
            count, level = (int(v) for v in re.search(r"builds (\d+) independent (\d+)%", t).groups())
            want = (level / 100) ** count
            tol = 0.00006
        else:
            seen.add("wide")
            n = int(re.search(r"n = (\d+)", t).group(1))
            s = int(re.search(r"s = (\d+)", t).group(1))
            want = 2 * 1.96 * s - 2 * 1.96 * s / math.sqrt(n)
            tol = 0.0051
        if abs(float(q.answer_text) - want) > tol:
            bad.append((t, q.answer_text, want))
    assert seen == {"expected", "some", "all", "wide"}
    assert bad == [], bad[:2]


def test_exam_shape_and_clean():
    sup.check_marks(EXAM, [2, 3, 1, 2], 40)
    q = sup.questions(EXAM, 40)
    assert all([p.follow_through is not None for p in z.parts] == [False, False, True, False] for z in q)
    assert all(z.parts[3].answer_kind == "mcq" for z in q)
    sup.check_clean_and_deterministic(EXAM)


def test_exam_keys_match_independent_derivation():
    bad = []
    for q in sup.questions(EXAM, 250):
        t = q.text
        mu = int(re.search(r"μ = (\d+)", t).group(1))
        sigma = int(re.search(r"σ = (\d+)", t).group(1))
        n = int(re.search(r"samples of size (\d+)", t).group(1))
        rows = _table(t)
        me = 1.96 * sigma / math.sqrt(n)
        miss = sum(1 for lo, hi in rows if not lo < mu < hi)
        got = sup.canonical(q)
        if (
            abs(float(got["a"]) - me) > 0.0051
            or int(got["b"]) != miss
            or abs(float(got["c"]) - miss / len(rows)) > 0.0051
        ):
            bad.append((t, got))
        # the table really is centre +/- the margin of error
        if any(abs((hi - lo) / 2 - me) > 0.0051 for lo, hi in rows) or len(rows) not in (20, 40, 80):
            bad.append((t, "table"))
        if f"a count of {miss} " not in got["d"]:
            bad.append((t, got["d"]))
    assert bad == [], bad[:2]


def test_exam_answer_space_grading_and_tags():
    sup.check_answer_space(EXAM)
    sup.check_grading(SKILL, EXAM)
    sup.check_tagged(EXAM)


def test_follow_through_from_a_wrong_count():
    def replies(q):
        wrong = q.parts[1].distractors[0]
        return (
            {
                "a": q.parts[0].answer_text,
                "b": wrong,
                "c": f"{int(wrong) / int(re.search(r'takes (\d+)', q.text).group(1)):.4f}",
                "d": q.parts[3].answer_text,
            },
            {"a", "c", "d"},
            {"b"},
        )

    sup.check_follow_through(SKILL, EXAM, replies, n=100, at_least=20)
