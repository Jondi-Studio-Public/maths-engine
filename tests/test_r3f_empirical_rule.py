"""Tests for `normal-curve-empirical-rule`: tiers (`r3f_empirical_rule`) and the exam chain
(`exd_x_r3f_empirical_rule`), with the answers re-derived from the rendered stem and the exact normal distribution."""

from __future__ import annotations

import math
import re

from tests import r3f_support as sup

SKILL = "normal-curve-empirical-rule"
TOPIC = "r3f_empirical_rule"
EXAM = "exd_x_r3f_empirical_rule"


def _phi(z):
    return 0.5 * math.erfc(-z / math.sqrt(2))


def _share(text, mu, sd):
    """Exact percentage in the region a stem names, to compare with the rule's rounded answer."""
    between = re.search(r"between (-?\d+) and (-?\d+)", text)
    if between:
        lo, hi = float(between.group(1)), float(between.group(2))
        return 100 * (_phi((hi - mu) / sd) - _phi((lo - mu) / sd))
    below = re.search(r"are below (-?\d+)", text)
    if below:
        return 100 * _phi((float(below.group(1)) - mu) / sd)
    above = re.search(r"are above (-?\d+)", text)
    return 100 * (1 - _phi((float(above.group(1)) - mu) / sd))


def _mu_sd(text):
    mu = float(re.search(r"mean (-?\d+)", text).group(1))
    sd = float(re.search(r"standard deviation (\d+)", text).group(1))
    return mu, sd


def test_wired():
    sup.check_tier_wiring(SKILL, TOPIC, EXAM)
    sup.check_wired(SKILL, EXAM)


def test_tiers_grade_and_format():
    sup.check_tier_grading(SKILL, TOPIC)


def test_tiers_differ_in_shape():
    sup.check_tier_shapes(TOPIC)
    sup.check_tier_determinism(TOPIC)


def test_easy_keys_agree_with_the_normal_distribution():
    bad = []
    for q in sup.tier_questions(TOPIC, "easy", 150):
        mu, sd = _mu_sd(q.text)
        exact = _share(q.text, mu, sd)
        if abs(float(q.answer_text) - exact) > 0.5:
            bad.append((q.text, q.answer_text, exact))
    assert bad == [], bad[:2]


def test_medium_keys():
    bad = []
    seen = set()
    for q in sup.tier_questions(TOPIC, "medium", 200):
        t = q.text
        if "expected to have values" in t:
            seen.add("count")
            mu, sd = _mu_sd(t)
            n = int(re.search(r"sample of (\d+)", t).group(1))
            exact = n * _share(t, mu, sd) / 100
            if abs(float(q.answer_text) - exact) > 0.005 * n:
                bad.append((t, q.answer_text, exact))
        elif "points of inflection at" in t and "Curve" not in t:
            seen.add("inflection")
            a, b = (int(x) for x in re.search(r"inflection at (\d+) and (\d+)", t).groups())
            want = (b - a) / 2 if "standard deviation" in t.split("inflection")[1] else (a + b) / 2
            if float(q.answer_text) != want:
                bad.append((t, q.answer_text, want))
        elif "About 95%" in t:
            seen.add("interval95")
            a, b = (int(x) for x in re.search(r"between (\d+) and (\d+)", t).groups())
            if float(q.answer_text) != (b - a) / 4:
                bad.append((t, q.answer_text))
        else:
            seen.add("compare")
            found = re.findall(r"inflection at (\d+) and (\d+)", t)
            (a1, a2), (b1, b2) = ((int(x), int(y)) for x, y in found)
            if abs(float(q.answer_text) - (b2 - b1) / (a2 - a1)) > 0.0501:
                bad.append((t, q.answer_text))
    assert seen == {"count", "inflection", "interval95", "compare"}
    assert bad == [], bad[:2]


def test_hard_keys():
    bad = []
    seen = set()
    for q in sup.tier_questions(TOPIC, "hard", 200):
        t = q.text
        if t.startswith("Population A"):
            seen.add("second")
            ma, sa, mb, sb = (
                int(x)
                for x in re.findall(r"mean (\d+) and standard deviation (\d+)", t)[0]
                + re.findall(r"mean (\d+) and standard deviation (\d+)", t)[1]
            )
            lo, hi = (int(x) for x in re.search(r"between (\d+) and (\d+)\. What", t).groups())
            exact = 100 * (_phi((hi - mb) / sb) - _phi((lo - mb) / sb))
            if abs(float(q.answer_text) - exact) > 0.5:
                bad.append((t, q.answer_text, exact))
        else:
            seen.add("tails")
            pct = {"16": 1, "2.5": 2, "0.15": 3}
            up = re.search(r"(\d+(?:\.\d+)?)% of the values exceed (\d+)", t)
            dn = re.search(r"(\d+(?:\.\d+)?)% fall below (\d+)", t)
            k1, k2 = pct[up.group(1)], pct[dn.group(1)]
            top, bottom = int(up.group(2)), int(dn.group(2))
            sd = (top - bottom) / (k1 + k2)
            mean = top - k1 * sd
            want = sd if "standard deviation" in t.split("below")[1] else mean
            if float(q.answer_text) != want:
                bad.append((t, q.answer_text, want))
    assert seen == {"second", "tails"}
    assert bad == [], bad[:2]


def test_exam_shape_and_clean():
    sup.check_marks(EXAM, [1, 2, 1, 2], 40)
    q = sup.questions(EXAM, 40)
    assert all([p.follow_through is not None for p in x.parts] == [False, False, True, True] for x in q)
    sup.check_clean_and_deterministic(EXAM)


def test_exam_keys_match_independent_derivation():
    bad = []
    for q in sup.questions(EXAM, 200):
        mu = float(re.search(r"mean (\d+)", q.text).group(1))
        a, b = (int(x) for x in re.search(r"inflection at (\d+) and (\d+)", q.text).groups())
        sd = (b - a) / 2
        lo, hi = (int(x) for x in re.search(r"between (\d+) and (\d+)\?", q.text).groups())
        exact = 100 * (_phi((hi - mu) / sd) - _phi((lo - mu) / sd))
        n = int(re.search(r"sample of (\d+)", q.text).group(1))
        got = sup.canonical(q)
        if float(got["a"]) != sd or abs(float(got["b"]) - exact) > 0.5:
            bad.append((q.text, got))
        if (
            abs(float(got["c"]) - n * float(got["b"]) / 100) > 1e-6
            or float(got["d"]) != mu + 2 * sd
            or mu != (a + b) / 2
        ):
            bad.append((q.text, got))
    assert bad == [], bad[:2]


def test_exam_answer_space_grading_and_tags():
    sup.check_answer_space(EXAM)
    sup.check_grading(SKILL, EXAM)
    sup.check_tagged(EXAM)


def test_follow_through_from_a_wrong_percentage_and_sd():
    def replies(q):
        n = int(re.search(r"sample of (\d+)", q.text).group(1))
        wrong_b = q.parts[1].distractors[0]
        wrong_a = q.parts[0].distractors[0]
        mu = float(re.search(r"mean (\d+)", q.text).group(1))
        c = f"{float(wrong_b) * n / 100:.0f}"
        d = f"{mu + 2 * float(wrong_a):.0f}"
        return (
            {"a": wrong_a, "b": wrong_b, "c": c, "d": d},
            {"c", "d"},
            {"a", "b"},
        )

    sup.check_follow_through(SKILL, EXAM, replies)
