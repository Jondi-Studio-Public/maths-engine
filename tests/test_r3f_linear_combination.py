"""Tests for `linear-combination-random-variables`: tiers (`r3f_linear_combination`) and the exam chain
(`exd_x_r3f_linear_combination`), with the answers re-derived from the rendered stem."""

from __future__ import annotations

import math
import re

from tests import r3f_support as sup

SKILL = "linear-combination-random-variables"
TOPIC = "r3f_linear_combination"
EXAM = "exd_x_r3f_linear_combination"


def _phi(z):
    return 0.5 * math.erfc(-z / math.sqrt(2))


def _coeffs(text):
    """(a, b) from 'aX + bY' or 'aX - bY' in a prompt."""
    m = re.search(r"\((-?\d*)X ([+-]) (\d*)Y\)|of (-?\d*)X ([+-]) (\d*)Y\.", text)
    groups = [g for g in m.groups() if g is not None]
    a = -1 if groups[0] == "-" else int(groups[0] or 1)
    b = int(groups[2] or 1) * (1 if groups[1] == "+" else -1)
    return a, b


def test_wired():
    sup.check_tier_wiring(SKILL, TOPIC, EXAM)
    sup.check_wired(SKILL, EXAM)


def test_tiers_grade_and_format():
    sup.check_tier_grading(SKILL, TOPIC)


def test_tiers_differ_in_shape():
    sup.check_tier_shapes(TOPIC)
    sup.check_tier_determinism(TOPIC)


def test_easy_keys():
    bad, seen = [], set()
    for q in sup.tier_questions(TOPIC, "easy", 200):
        t = q.text
        a, b = _coeffs(t)
        mx, my = (int(v) for v in re.search(r"E\(X\) = (\d+).*E\(Y\) = (\d+)", t).groups())
        if "Find E(" in t:
            seen.add("mean")
            want = a * mx + b * my
        elif "Find Var(" in t:
            seen.add("var")
            vx, vy = (int(v) for v in re.search(r"Var\(X\) = (\d+).*Var\(Y\) = (\d+)", t).groups())
            want = a * a * vx + b * b * vy
        else:
            seen.add("sd")
            sx, sy = (int(v) for v in re.search(r"SD\(X\) = (\d+).*SD\(Y\) = (\d+)", t).groups())
            want = math.sqrt(a * a * sx * sx + b * b * sy * sy)
        if abs(float(q.answer_text) - want) > 1e-9:
            bad.append((t, q.answer_text, want))
    assert seen == {"mean", "var", "sd"}
    assert bad == [], bad[:2]


def test_medium_keys():
    bad, seen = [], set()
    for q in sup.tier_questions(TOPIC, "medium", 250):
        t = q.text
        if "tiles" in t or "bricks" in t:
            n = int(re.search(r"(\d+) (?:tiles|bricks)", t).group(1))
            sigma = int(re.search(r"standard deviation (\d+) g", t).group(1))
            if "multiplied by" in t:
                seen.add("multiple")
                want = n * sigma
            else:
                seen.add("copies")
                want = math.sqrt(n * sigma * sigma)
        else:
            sx, sy = (int(v) for v in re.findall(r"standard deviation (\d+)", t))
            mx, my = (int(v) for v in re.findall(r"mean (\d+)", t))
            if "standard deviation of" in t.split("distributed")[-1]:
                seen.add("sd")
                want = math.sqrt(sx * sx + sy * sy)
            elif "Find the mean of how much longer" in t:
                seen.add("diff_mean")
                want = my - mx
            else:
                seen.add("total_mean")
                want = mx + my
        if abs(float(q.answer_text) - want) > 1e-9:
            bad.append((t, q.answer_text, want))
    assert {"multiple", "copies", "sd", "diff_mean"} <= seen
    assert bad == [], bad[:2]


def test_hard_keys():
    bad, seen = [], set()
    for q in sup.tier_questions(TOPIC, "hard", 250):
        t = q.text
        if "total mass of the" in t:
            seen.add("total")
            mu, sigma = (int(v) for v in re.search(r"mean (\d+) g and standard deviation (\d+) g", t).groups())
            n = int(re.search(r"A batch has (\d+)", t).group(1))
            c = int(re.search(r"more than (\d+) g", t).group(1))
            want = 1 - _phi((c - n * mu) / (sigma * math.sqrt(n)))
        elif "differ by less than" in t:
            seen.add("within")
            (mx, my), (sx, sy) = (
                (int(v) for v in re.findall(r"mean (\d+) mL", t)),
                (int(v) for v in re.findall(r"standard deviation (\d+) mL", t)),
            )
            d = int(re.search(r"less than (\d+) mL", t).group(1))
            sd = math.sqrt(sx * sx + sy * sy)
            want = _phi((d - (mx - my)) / sd) - _phi((-d - (mx - my)) / sd)
        else:
            seen.add("greater")
            means = [int(v) for v in re.findall(r"mean (\d+)", t)]
            sds = [int(v) for v in re.findall(r"standard deviation (\d+)", t)]
            want = 1 - _phi((0 - (means[0] - means[1])) / math.sqrt(sds[0] ** 2 + sds[1] ** 2))
        if abs(float(q.answer_text) - want) > 0.00006:
            bad.append((t, q.answer_text, want))
    assert seen == {"total", "within", "greater"}
    assert bad == [], bad[:2]


def test_exam_shape_and_clean():
    sup.check_marks(EXAM, [1, 2, 3, 4], 40)
    q = sup.questions(EXAM, 40)
    assert all([p.follow_through is not None for p in z.parts] == [False, False, True, True] for z in q)
    sup.check_clean_and_deterministic(EXAM)


def test_exam_keys_match_independent_derivation():
    bad = []
    for q in sup.questions(EXAM, 250):
        t = q.text
        n = int(re.search(r"holds (\d+) packets", t).group(1))
        carton = int(re.search(r"weighs (\d+) g when empty", t).group(1))
        mu, sigma = (int(v) for v in re.search(r"mean (\d+) g and standard deviation (\d+) g", t).groups())
        prompts = " ".join(p.prompt for p in q.parts)
        limit = int(re.search(r"weighs more than (\d+) g", prompts).group(1))
        mu_b, sd_b = (
            int(v)
            for v in re.search(
                r"mean (\d+) g and standard deviation (\d+) g, independently of the first", prompts
            ).groups()
        )
        tm, tsd = n * mu + carton, sigma * math.sqrt(n)
        got = sup.canonical(q)
        want = (tm, tsd, 1 - _phi((limit - tm) / tsd), 1 - _phi((mu_b - tm) / math.sqrt(tsd**2 + sd_b**2)))
        for key, w in zip("abcd", want):
            if abs(float(got[key]) - w) > (0.00006 if key in "cd" else 1e-9):
                bad.append((t, key, got[key], w))
    assert bad == [], bad[:2]


def test_exam_answer_space_grading_and_tags():
    sup.check_answer_space(EXAM)
    sup.check_grading(SKILL, EXAM)
    sup.check_tagged(EXAM)


def test_follow_through_from_wrong_mean_and_sd():
    def replies(q):
        t = q.text
        prompts = " ".join(p.prompt for p in q.parts)
        limit = int(re.search(r"weighs more than (\d+) g", prompts).group(1))
        mu_b, sd_b = (
            int(v)
            for v in re.search(
                r"mean (\d+) g and standard deviation (\d+) g, independently of the first", prompts
            ).groups()
        )
        wrong_a = float(q.parts[0].distractors[0])
        wrong_b = float(q.parts[1].distractors[0])
        c = 1 - _phi((limit - wrong_a) / wrong_b)
        d = _phi((wrong_a - mu_b) / math.sqrt(wrong_b**2 + sd_b**2))
        if not (0.001 < c < 0.999 and 0.001 < d < 0.999):
            return None
        return (
            {"a": q.parts[0].distractors[0], "b": q.parts[1].distractors[0], "c": f"{c:.4f}", "d": f"{d:.4f}"},
            {"c", "d"},
            {"a", "b"},
        )

    sup.check_follow_through(SKILL, EXAM, replies, n=150, at_least=10)
