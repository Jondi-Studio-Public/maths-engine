"""Tests the skill `ucv1-one-sample-t-test`: the hard-coded t tables against mpmath, an independent oracle that
re-derives every key from the question TEXT, the typed grader on every key and distractor, and the four-part exam
chain with follow-through."""

from __future__ import annotations

import math
import re
import statistics
from decimal import ROUND_HALF_UP, Decimal

import mpmath
import pytest

import mathgen.generators.ucv1_t_test_build as module
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects

TOPIC = "ucv1_one_sample_t_test"
EXAM_TOPIC = "exd_x_uc_t_test"
SKILL_ID = "ucv1-one-sample-t-test"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
TIERS = ("easy", "medium", "hard")
SEEDS = range(60)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM_TOPIC, seed=seed, difficulty="exam")


def _r(x: float, dp: int) -> str:
    return str(Decimal(repr(x)).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP))


def _dp(text: str) -> int:
    return int(re.search(r"correct to (\d) decimal place", text).group(1))


# --- the tables -------------------------------------------------------------------------------------------------


def _t_quantile(p: float, nu: int) -> float:
    with mpmath.workdps(30):
        cdf = lambda t: 1 - mpmath.betainc(nu / 2, mpmath.mpf(1) / 2, 0, nu / (nu + t * t), regularized=True) / 2  # noqa: E731
        return float(mpmath.findroot(lambda t: cdf(t) - p, 2))


def test_critical_value_tables_match_the_t_distribution():
    module.verify_critical_values()  # the module's own check, all three tables, df 4..30


@pytest.mark.parametrize("df", [4, 9, 16, 24, 30])
def test_a_few_critical_values_against_an_independent_quantile(df):
    assert abs(_t_quantile(0.975, df) - module.T_TWO_SIDED_5[df]) < 0.00051
    assert abs(_t_quantile(0.995, df) - module.T_TWO_SIDED_1[df]) < 0.00051
    assert abs(_t_quantile(0.95, df) - module.T_ONE_SIDED_5[df]) < 0.00051


def test_tables_cover_df_4_to_30_and_decrease():
    for table in (module.T_TWO_SIDED_5, module.T_TWO_SIDED_1, module.T_ONE_SIDED_5):
        assert sorted(table) == list(range(4, 31))
        values = [table[d] for d in range(4, 31)]
        assert values == sorted(values, reverse=True)


# --- wiring -----------------------------------------------------------------------------------------------------


def test_sources_are_wired_in_tier_order_with_the_exam_last():
    sources = _SKILL.sources
    assert [(s.topic, s.difficulty) for s in sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM_TOPIC, "exam"),
    ]
    assert [s.stages for s in sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert _SKILL.status == "active" and _SKILL.answer == "number"


# --- the independent oracle -------------------------------------------------------------------------------------


def _summary(text):
    n = int(re.search(r"sample of (\d+)", text).group(1))
    xbar = float(re.search(r"has a mean [a-z ]+ of ([\d.]+) ", text).group(1))
    sd = re.search(r"standard deviation of ([\d.]+)", text)
    var = re.search(r"variance of ([\d.]+)", text)
    s = float(sd.group(1)) if sd else math.sqrt(float(var.group(1)))
    return n, xbar, s


def oracle(tier, q):
    text = q.text
    if tier == "easy":
        n, _xbar, s = _summary(text)
        return _r(s / math.sqrt(n), 3)
    if tier == "medium":
        n, xbar, s = _summary(text)
        mu0 = float(re.search(r"H0: μ = ([\d.]+)", text).group(1))
        return _r((xbar - mu0) / (s / math.sqrt(n)), _dp(text))
    if "confidence interval for the true mean" in text:
        n, xbar, s = _summary(text)
        tstar = float(re.search(r"t\* = (\d\.\d+)", text).group(1))
        margin = tstar * s / math.sqrt(n)
        return _r(xbar + margin if "upper limit" in text else xbar - margin, _dp(text))
    if "are:" in text:
        diffs = [int(v) for v in re.search(r"are: ([-\d, ]+)\.", text).group(1).split(",")]
        return _r(statistics.mean(diffs) / (statistics.stdev(diffs) / math.sqrt(len(diffs))), _dp(text))
    assert "gave t =" in text
    n = int(re.search(r"random sample of (\d+)", text).group(1))
    s = float(re.search(r"standard deviation of ([\d.]+)", text).group(1))
    mu0 = float(re.search(r"H0: μ = ([\d.]+)", text).group(1))
    t = float(re.search(r"gave t = (-?\d\.\d+)", text).group(1))
    return _r(mu0 + t * s / math.sqrt(n), _dp(text))


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_the_independent_derivation(tier):
    bad = [s for s in range(200) if oracle(tier, _q(tier, s)) != _q(tier, s).answer_text]
    assert bad == []


def test_hard_tier_uses_all_three_shapes():
    kinds = set()
    for seed in range(60):
        text = _q("hard", seed).text
        kinds.add("ci" if "confidence interval" in text else "paired" if "are:" in text else "inverse")
    assert kinds == {"ci", "paired", "inverse"}


def test_medium_tier_uses_variance_and_sd_stems():
    stems = {"variance" if "variance" in _q("medium", s).text else "sd" for s in range(60)}
    assert stems == {"variance", "sd"}


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        assert grade(_SKILL, q, q.answer_text).correct, (tier, seed)
        assert len(q.distractors) >= 2
        assert all(q.option_notes.get(d) for d in q.distractors)
        for d in q.distractors:
            assert not grade(_SKILL, q, d).correct, (tier, seed, d)
        # a unit in the last place away is wrong
        dp = len(q.answer_text.split(".")[1])
        off = float(q.answer_text) + 2 * 10**-dp
        assert not grade(_SKILL, q, f"{off:.{dp}f}").correct


@pytest.mark.parametrize("tier", TIERS)
def test_each_distractor_is_diagnosed_as_its_own_error(tier):
    for seed in range(30):
        q = _q(tier, seed)
        for d in q.distractors:
            result = grade(_SKILL, q, d)
            assert result.misconception == q.option_notes[d], (tier, seed, d)


@pytest.mark.parametrize("tier", TIERS)
def test_variety_and_no_dominant_answer(tier):
    answers = [_q(tier, s).answer_text for s in SEEDS]
    assert len(set(answers)) >= 12
    assert max(answers.count(a) for a in set(answers)) <= 4
    prompts = {_q(tier, s).text for s in SEEDS}
    assert len(prompts) >= 40


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic_and_clean_text(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        assert q.to_json() == _q(tier, seed).to_json()
        assert find_text_defects(q) == []


def test_degrees_of_freedom_is_n_minus_1_in_every_printed_critical_value():
    for seed in range(60):
        text = _q("hard", seed).text
        if "t* =" not in text:
            continue
        n = int(re.search(r"sample of (\d+)", text).group(1))
        df = int(re.search(r"with (\d+) degrees of freedom", text).group(1))
        tstar = float(re.search(r"t\* = (\d\.\d+)", text).group(1))
        assert df == n - 1
        assert abs(_t_quantile(0.975, df) - tstar) < 0.00051


# --- the exam chain ---------------------------------------------------------------------------------------------


def _givens(q):
    stem = q.text.split(" (a)")[0]
    n, xbar, s = _summary(stem)
    mu0 = float(re.search(r"H0: μ = ([\d.]+)", stem).group(1))
    rel = re.search(r"H1: μ (.) ", stem).group(1)
    level = re.search(r"at the (\d)% level", stem).group(1)
    tail = re.search(r"Critical values of t \((one|two)-sided", stem).group(1)
    rows = {int(a): float(b) for a, b in re.findall(r"df = (\d+): (\d\.\d+)", stem)}
    return n, xbar, s, mu0, rel, level, tail, rows


def exam_oracle(q):
    n, xbar, s, mu0, rel, level, tail, rows = _givens(q)
    p = {("two", "5"): 0.975, ("two", "1"): 0.995, ("one", "5"): 0.95}[(tail, level)]
    crit_exact = _t_quantile(p, n - 1)
    assert abs(crit_exact - rows[n - 1]) < 0.00051  # the printed row for df = n - 1 is the real critical value
    crit = rows[n - 1]
    se = s / math.sqrt(n)
    t = (xbar - mu0) / se
    edge = mu0 - crit * se if rel == "<" else mu0 + crit * se
    reject = t < -crit if rel == "<" else (t > crit if rel == ">" else abs(t) > crit)
    return dict(a=_r(se, 3), b=_r(t, 2), c=_r(edge, 2)), reject


def _correct_statement(q):
    prompt = q.parts[3].prompt
    return {int(i): kind for i, kind in re.findall(r"(\d)\. (Reject|Do not reject) H0", prompt)}


def test_source_wired_last():
    src = _SKILL.sources[-1]
    assert (src.topic, src.difficulty, src.stages, src.params) == (EXAM_TOPIC, "exam", ["exam"], {"tier": "exam"})


def test_shape_marks_and_follow_through():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [1, 2, 2, 1]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, False]
        assert find_text_defects(q) == []


def test_exam_keys_match_the_independent_derivation():
    for seed in range(300):
        q = _exam(seed)
        keys, reject = exam_oracle(q)
        got = {p.label: p.answer_text for p in q.parts[:3]}
        assert got == keys, seed
        statements = _correct_statement(q)
        assert statements[int(q.parts[3].answer_text)] == ("Reject" if reject else "Do not reject"), seed


def test_exam_conclusions_are_balanced_and_positions_vary():
    rejects = sum(exam_oracle(_exam(s))[1] for s in range(200))
    assert 60 <= rejects <= 140
    positions = {_exam(s).parts[3].answer_text for s in range(60)}
    assert positions == {"1", "2", "3"}


def test_all_three_test_types_and_directions_appear():
    seen = {_givens(_exam(s))[4:7] for s in range(200)}
    assert {("≠", "5", "two"), ("≠", "1", "two"), (">", "5", "one"), ("<", "5", "one")} <= seen


def test_key_accepted_and_every_distractor_rejected():
    for seed in range(120):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 6
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2
            for text in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text)


def test_follow_through_from_a_wrong_standard_error():
    checked = 0
    for seed in range(120):
        q = _exam(seed)
        n, xbar, s, mu0, rel, level, tail, rows = _givens(q)
        crit = rows[n - 1]
        wrong_se = float(q.parts[0].distractors[0])
        t_ft = (xbar - mu0) / wrong_se
        edge_ft = mu0 - crit * wrong_se if rel == "<" else mu0 + crit * wrong_se
        reply = {
            "a": f"{wrong_se:.3f}",
            "b": _r(t_ft, 2),
            "c": _r(edge_ft, 2),
            "d": q.parts[3].answer_text,
        }
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        if abs(t_ft) < 0.2:
            continue
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through is True, (seed, result.parts["b"])
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True, (seed, result.parts["c"])
        checked += 1
    assert checked >= 80


def test_the_rounded_standard_error_can_be_used_downstream():
    for seed in range(80):
        q = _exam(seed)
        n, xbar, s, mu0, rel, level, tail, rows = _givens(q)
        crit = rows[n - 1]
        se3 = float(q.parts[0].answer_text)
        edge = mu0 - crit * se3 if rel == "<" else mu0 + crit * se3
        reply = {
            "a": q.parts[0].answer_text,
            "b": _r((xbar - mu0) / se3, 2),
            "c": _r(edge, 2),
            "d": q.parts[3].answer_text,
        }
        assert grade(_SKILL, q, reply).marks == 6, seed


def test_a_wrong_critical_row_is_not_credited():
    # the boundary mean from the df = n row differs from the df = n - 1 one: the marks need the right row
    wrong = 0
    for seed in range(120):
        q = _exam(seed)
        n, xbar, s, mu0, rel, level, tail, rows = _givens(q)
        crit = rows[n]
        se = s / math.sqrt(n)
        edge = mu0 - crit * se if rel == "<" else mu0 + crit * se
        if _r(edge, 2) != q.parts[2].answer_text:
            wrong += 1
            canon = {p.label: p.answer_text for p in q.parts}
            assert grade(_SKILL, q, {**canon, "c": _r(edge, 2)}).parts["c"].marks == 0
    assert wrong >= 30


def test_exam_answer_space_and_determinism():
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 6
    assert _exam(7).to_json() == _exam(7).to_json()
