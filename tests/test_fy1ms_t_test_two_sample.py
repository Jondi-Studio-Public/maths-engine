"""Tests the skill `ucv1-two-sample-and-paired-t-test`: an independent oracle that re-derives every key from the
question TEXT (numpy and mpmath, sharing no code with the generator), the printed critical-value table checked against
the t distribution, the typed grader on every key and distractor, and the four-part pooled-test exam chain with
follow-through."""

from __future__ import annotations

import re

import mpmath
import numpy as np
import pytest

from mathgen.generators.fy1ms_t_test_two_sample_build import T_TWO_SIDED_5, verify_critical_values
from tests import fy1ms_support as T

SKILL_ID = "ucv1-two-sample-and-paired-t-test"
TOPIC = "fy1ms_t_test_two_sample"
EXAM = "exd_x_fy1ms_t_test_two_sample"
SK = T.load(SKILL_ID)
TIERS = ("easy", "medium", "hard")

_SAMPLES = re.compile(
    r"random sample of (\d+) .*?, the mean .*? was (-?[\d.]+) \w+ and the sample standard deviation was ([\d.]+) \w+\. "
    r"In an independent random sample of (\d+) .*?, the mean .*? was (-?[\d.]+) \w+ and the sample standard deviation "
    r"was ([\d.]+) \w+\."
)


def _samples(text: str):
    m = _SAMPLES.search(text)
    assert m, text
    n1, m1, s1, n2, m2, s2 = m.groups()
    mp = mpmath.mpf
    return int(n1), mp(m1), mp(s1), int(n2), mp(m2), mp(s2)


def _pooled(n1, s1, n2, s2):
    return ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)


def _t_pooled(n1, m1, s1, n2, m2, s2):
    return (m1 - m2) / mpmath.sqrt(_pooled(n1, s1, n2, s2) * (mpmath.mpf(1) / n1 + mpmath.mpf(1) / n2))


def oracle(q) -> str:
    text = q.text
    with mpmath.workdps(40):
        if "paired t statistic" in text:
            m = re.search(r"The (\d+) differences have mean (-?[\d.]+) and sample standard deviation ([\d.]+)\.", text)
            if m:
                n, mean, sd = int(m.group(1)), mpmath.mpf(m.group(2)), mpmath.mpf(m.group(3))
                return T.rnd(mean / (sd / mpmath.sqrt(n)), 2)
            diffs = [int(v) for v in re.search(r"is: (.*?)\. Let", text).group(1).split(", ")]
            d = np.array(diffs, dtype=float)
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
            return T.rnd(t, 2)
        n1, m1, s1, n2, m2, s2 = _samples(text)
        if "standard error of the difference" in text:
            return T.rnd(mpmath.sqrt(s1**2 / n1 + s2**2 / n2), 3)
        if "pooled sample variance" in text:
            return T.rnd(_pooled(n1, s1, n2, s2), 2)
        if "Welch" in text:
            return T.rnd((m1 - m2) / mpmath.sqrt(s1**2 / n1 + s2**2 / n2), 2)
        if "pooled two-sample t statistic" in text:
            return T.rnd(_t_pooled(n1, m1, s1, n2, m2, s2), 2)
        m = re.search(
            r"critical value of t for (\d+) degrees of freedom is ([\d.]+) .*?Find the (upper|lower) limit", text
        )
        assert m, text
        assert int(m.group(1)) == n1 + n2 - 2
        se = (m1 - m2) / _t_pooled(n1, m1, s1, n2, m2, s2)
        margin = mpmath.mpf(m.group(2)) * abs(se)
        return T.rnd((m1 - m2) + (margin if m.group(3) == "upper" else -margin), 2)


def test_wiring():
    T.assert_wiring(SK, TOPIC, [EXAM])


def test_printed_critical_values_match_the_t_distribution():
    verify_critical_values()
    assert min(T_TWO_SIDED_5) <= 4 and max(T_TWO_SIDED_5) >= 40


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_the_independent_derivation(tier):
    bad = [s for s in range(80) if oracle(T.single(TOPIC, tier, s)) != T.single(TOPIC, tier, s).answer_text]
    assert bad == []


def test_critical_value_in_a_confidence_interval_question_is_the_table_entry():
    seen = 0
    for s in range(120):
        text = T.single(TOPIC, "hard", s).text
        m = re.search(r"critical value of t for (\d+) degrees of freedom is ([\d.]+)", text)
        if m:
            seen += 1
            assert m.group(2) == f"{T_TWO_SIDED_5[int(m.group(1))]:.3f}"
    assert seen >= 15


def test_tiers_use_their_shapes():
    def shape(tier, s):
        text = T.single(TOPIC, tier, s).text
        if "standard error of the difference" in text:
            return "se"
        if "Welch" in text:
            return "welch"
        if "pooled sample variance" in text:
            return "pooled-variance"
        if "pooled two-sample" in text:
            return "pooled-t"
        if "confidence interval" in text:
            return "ci"
        return "paired-list" if "is: " in text else "paired-summary"

    assert {shape("easy", s) for s in range(80)} == {"se"}
    assert {shape("medium", s) for s in range(120)} == {"welch", "pooled-variance", "paired-summary"}
    assert {shape("hard", s) for s in range(120)} == {"pooled-t", "paired-list", "ci"}


@pytest.mark.parametrize("tier", TIERS)
def test_grading_and_diagnosis(tier):
    T.assert_single_grading(SK, TOPIC, tier)


@pytest.mark.parametrize("tier", TIERS)
def test_clean_text_variety_and_determinism(tier):
    T.assert_clean_variety_and_determinism(TOPIC, tier, distinct_prompts=40, distinct_answers=25)


# --- the exam chain --------------------------------------------------------------------------------------------


def exam_oracle(q):
    n1, m1, s1, n2, m2, s2 = _samples(q.stem)
    with mpmath.workdps(40):
        sp2 = _pooled(n1, s1, n2, s2)
        se = mpmath.sqrt(sp2 * (mpmath.mpf(1) / n1 + mpmath.mpf(1) / n2))
        t = (m1 - m2) / se
    df = n1 + n2 - 2
    crit = {int(k): float(v) for k, v in re.findall(r"(\d+) degrees of freedom: (\d\.\d{3})", q.stem)}
    assert set(crit) == {df - 1, df, df + 1}
    assert all(crit[k] == T_TWO_SIDED_5[k] for k in crit)
    verdict = (
        "Reject H0, because |t| is greater than" if abs(t) > crit[df] else "Do not reject H0, because |t| is less than"
    )
    prompt = q.parts[3].prompt
    statements = re.findall(r"(\d)\. (.*?)(?= \d\. |$)", prompt.split("correct statement.")[1].strip())
    right = [i for i, text in statements if text.startswith(verdict)]
    assert len(right) == 1, prompt
    return dict(a=T.rnd(sp2, 2), b=T.rnd(se, 3), c=T.rnd(t, 2), d=right[0])


def test_exam_shape():
    T.assert_exam_shape(EXAM, ["a", "b", "c", "d"], [2, 1, 2, 1], [False, True, True, False])


def test_exam_keys_match_the_independent_derivation():
    for seed in range(80):
        q = T.exam(EXAM, seed)
        assert {p.label: p.answer_text for p in q.parts} == exam_oracle(q), seed


def test_exam_reaches_both_conclusions():
    verdicts = {T.exam(EXAM, s).parts[3].working[0].split(":")[1].split(",")[0] for s in range(40)}
    assert len(verdicts) == 2, verdicts


def test_exam_grading_distractors_and_notes():
    T.assert_exam_grading(SK, EXAM)


def test_follow_through_from_a_wrong_variance_and_a_wrong_standard_error():
    T.assert_follow_through_credited(SK, EXAM, minimum=100)


def test_exam_answer_space():
    T.assert_exam_variety(EXAM, seeds=range(200), distinct=80, repeat=10)
    assert len({T.exam(EXAM, s).stem for s in range(60)}) >= 55
