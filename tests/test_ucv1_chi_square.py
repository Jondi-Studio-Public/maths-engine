"""Tests the skill `ucv1-chi-square-test`: the printed chi-square tables against mpmath, an independent oracle that
re-derives every key from the question TEXT (tables parsed back out of the prose), the typed grader on every key
and distractor, and the four-part exam chain with follow-through."""

from __future__ import annotations

import math
import re
from decimal import ROUND_HALF_UP, Decimal

import mpmath
import pytest

import mathgen.generators.ucv1_chi_square_build as module
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects

TOPIC = "ucv1_chi_square_test"
EXAM_TOPIC = "exd_x_uc_chi_square"
SKILL_ID = "ucv1-chi-square-test"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
TIERS = ("easy", "medium", "hard")
SEEDS = range(60)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM_TOPIC, seed=seed, difficulty="exam")


def _r(x: float, dp: int) -> str:
    return str(Decimal(repr(x)).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP))


def _chi_quantile(tail: float, df: int) -> float:
    with mpmath.workdps(30):
        f = lambda x: mpmath.gammainc(mpmath.mpf(df) / 2, x / 2, mpmath.inf, regularized=True) - tail  # noqa: E731
        return float(mpmath.findroot(f, df + 2))


# --- the tables -------------------------------------------------------------------------------------------------


def test_chi_square_tables_match_the_distribution():
    module.verify_critical_values()


@pytest.mark.parametrize("df", [1, 2, 3, 5, 8, 12])
def test_a_few_critical_values_against_an_independent_quantile(df):
    assert abs(_chi_quantile(0.05, df) - module.CHI_5[df]) < 0.00051
    assert abs(_chi_quantile(0.01, df) - module.CHI_1[df]) < 0.00051


# --- wiring -----------------------------------------------------------------------------------------------------


def test_sources_are_wired_in_tier_order_with_the_exam_last():
    assert [(s.topic, s.difficulty) for s in _SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM_TOPIC, "exam"),
    ]
    assert [s.stages for s in _SKILL.sources] == [[0, 1, 2], [3, 4], [5, 6, "maintenance"], ["exam"]]
    assert _SKILL.status == "active" and _SKILL.answer == "number"
    assert _SKILL.sources[-1].params == {"tier": "exam"}


# --- parsing the question text back into numbers ---------------------------------------------------------------


def parse_grid(text):
    """(header, [(row name, [values])]) from the 'Table (...): ...;' sentence."""
    match = re.search(r"Table \((.*?)\): (.*?)\.(?: |$)", text)
    header = match.group(1).split(" — ")
    rows = []
    for chunk in match.group(2).split("; "):
        name, *values = chunk.split(" — ")
        rows.append((name, [int(v) for v in values]))
    return header, rows


def table_expected(rows):
    values = [r[1] for r in rows]
    total = sum(map(sum, values))
    rt = [sum(r) for r in values]
    ct = [sum(c) for c in zip(*values)]
    return [[rt[i] * ct[j] / total for j in range(len(ct))] for i in range(len(rt))], values


def chi(obs, exp):
    return sum((o - e) ** 2 / e for o, e in zip(obs, exp))


def shares_in(text):
    return [(label, int(p)) for label, p in re.findall(r"([A-Z][A-Za-z ]*?) (\d+)%[,.]", text)]


def gof_expected(text, obs):
    n = sum(obs)
    ratio = re.search(r"ratio (\d(?: : \d)+)", text)
    if ratio:
        w = [int(v) for v in ratio.group(1).split(" : ")]
        return [n * v / sum(w) for v in w]
    shares = shares_in(text)
    if shares:
        return [n * p / 100 for _label, p in shares]
    return [n / len(obs)] * len(obs)


def oracle_easy(text):
    if "Table (" in text:
        header, rows = parse_grid(text)
        exp, _values = table_expected(rows)
        row_name, col_name = re.search(r"cell \((.*), (.*)\)", text).groups()
        i = [r[0] for r in rows].index(row_name)
        j = header[1:].index(col_name)
        return _r(exp[i][j], 2)
    if "%" in text:
        shares = shares_in(text)
        n = int(re.search(r"This year (\d+) ", text).group(1))
        label = re.search(r"in the (.*?) category", text).group(1)
        p = dict(shares)[label]
        return _r(n * p / 100, 2)
    n = int(re.search(r"(?:rolled|records|spun|tub of) (\d+)", text).group(1))
    k = {"six-sided": 6, "weekdays": 5}.get(next((w for w in ("six-sided", "weekdays") if w in text), ""), None) or int(
        re.search(r"(?:has|its) (\d) (?:sectors|colours)", text).group(1)
    )
    return _r(n / k, 2)


def oracle_stat(text):
    header, rows = parse_grid(text)
    if rows[0][0] == "Observed":
        obs = rows[0][1]
        return _r(chi(obs, gof_expected(text, obs)), 2)
    exp, values = table_expected(rows)
    return _r(chi([v for r in values for v in r], [e for r in exp for e in r]), 2)


def oracle_df(text):
    table = re.search(r"with (\d) categories of .* and (\d) categories of", text)
    if table:
        return str((int(table.group(1)) - 1) * (int(table.group(2)) - 1))
    k = int(re.search(r"grouped into (\d+)", text).group(1))
    m = 2 if "mean and standard deviation both" in text else 1
    return str(k - 1 - m)


def oracle(tier, q):
    text = q.text
    if tier == "easy":
        return oracle_easy(text)
    if "degrees of freedom" in text:
        return oracle_df(text)
    return oracle_stat(text)


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_the_independent_derivation(tier):
    bad = [s for s in range(200) if oracle(tier, _q(tier, s)) != _q(tier, s).answer_text]
    assert bad == []


def test_each_shape_appears():
    easy = {
        ("cell" if "Table (" in _q("easy", s).text else "share" if "%" in _q("easy", s).text else "equal")
        for s in range(60)
    }
    assert easy == {"cell", "share", "equal"}
    medium = {
        "df"
        if "degrees of freedom" in _q("medium", s).text
        else "gof"
        if "Observed —" in _q("medium", s).text
        else "2x2"
        for s in range(60)
    }
    assert medium == {"df", "gof", "2x2"}
    hard = {"ratio" if "ratio" in _q("hard", s).text else "2x3" for s in range(60)}
    assert hard == {"ratio", "2x3"}
    # df questions cover tables and fitted models
    kinds = {
        "fitted" if "grouped into" in _q("medium", s).text else "table"
        for s in range(120)
        if "degrees of freedom" in _q("medium", s).text
    }
    assert kinds == {"fitted", "table"}


@pytest.mark.parametrize("tier", ("medium", "hard"))
def test_every_expected_count_in_a_statistic_question_is_at_least_5(tier):
    for seed in range(120):
        text = _q(tier, seed).text
        if "degrees of freedom" in text:
            continue
        header, rows = parse_grid(text)
        if rows[0][0] == "Observed":
            obs = rows[0][1]
            assert min(gof_expected(text, obs)) >= 5, (tier, seed)
        else:
            exp, _ = table_expected(rows)
            assert min(min(r) for r in exp) >= 5, (tier, seed)


def test_tables_are_2x2_in_medium_and_2x3_in_hard():
    for seed in range(60):
        t = _q("medium", seed).text
        if "Observed counts by" in t:
            assert all(len(r[1]) == 2 for r in parse_grid(t)[1])
        t = _q("hard", seed).text
        if "Observed counts by" in t:
            assert all(len(r[1]) == 3 for r in parse_grid(t)[1])


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        assert grade(_SKILL, q, q.answer_text).correct, (tier, seed)
        assert len(q.distractors) >= 2
        assert all(q.option_notes.get(d) for d in q.distractors)
        for d in q.distractors:
            assert not grade(_SKILL, q, d).correct, (tier, seed, d)
        value = float(q.answer_text)
        dp = len(q.answer_text.split(".")[1]) if "." in q.answer_text else 0
        wrong = value + (2 * 10**-dp if dp else 1)
        assert not grade(_SKILL, q, f"{wrong:.{dp}f}").correct


@pytest.mark.parametrize("tier", TIERS)
def test_each_distractor_is_diagnosed_as_its_own_error(tier):
    for seed in range(30):
        q = _q(tier, seed)
        for d in q.distractors:
            assert grade(_SKILL, q, d).misconception == q.option_notes[d], (tier, seed, d)


@pytest.mark.parametrize("tier", TIERS)
def test_variety_and_no_dominant_answer(tier):
    answers = [_q(tier, s).answer_text for s in SEEDS]
    assert len(set(answers)) >= 12
    assert max(answers.count(a) for a in set(answers)) <= 6
    assert len({_q(tier, s).text for s in SEEDS}) >= 40


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic_and_clean_text(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        assert q.to_json() == _q(tier, seed).to_json()
        assert find_text_defects(q) == []


# --- the exam chain ---------------------------------------------------------------------------------------------


def exam_oracle(q):
    stem = q.text.split(" (a)")[0]
    prompt_a = q.parts[0].prompt
    level = re.search(r"at the (\d)% level", stem).group(1)
    tail = 0.05 if level == "5" else 0.01
    rows_crit = {int(a): float(b) for a, b in re.findall(r"df = (\d+): (\d+\.\d+)", stem)}
    header, rows = parse_grid(stem)
    if rows[0][0] == "Observed":
        obs = rows[0][1]
        exp = gof_expected(stem, obs)
        label = re.search(r"expected number of (.*?) under H0", prompt_a).group(1)
        idx = header[1:].index(label)
        df = len(obs) - 1
    else:
        exp_t, values = table_expected(rows)
        obs = [v for r in values for v in r]
        exp = [e for r in exp_t for e in r]
        row_name, col_name = re.search(r"cell \((.*), (.*)\)", prompt_a).groups()
        i = [r[0] for r in rows].index(row_name)
        j = header[1:].index(col_name)
        idx = i * len(values[0]) + j
        df = (len(values) - 1) * (len(values[0]) - 1)
    stat = chi(obs, exp)
    crit = rows_crit[df]
    assert abs(_chi_quantile(tail, df) - crit) < 0.00051  # the printed row for this df is the real critical value
    return dict(a=_r(exp[idx], 2), b=_r(stat, 2), c=str(df)), stat > crit, obs, exp, idx, stat


def _statements(q):
    return {int(i): kind for i, kind in re.findall(r"(\d)\. (Reject|Do not reject) H0", q.parts[3].prompt)}


def test_shape_marks_and_follow_through():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [1, 3, 1, 1]
        assert [p.follow_through is not None for p in q.parts] == [False, True, False, False]
        assert find_text_defects(q) == []


def test_exam_keys_match_the_independent_derivation():
    for seed in range(300):
        q = _exam(seed)
        keys, reject, *_ = exam_oracle(q)
        assert {p.label: p.answer_text for p in q.parts[:3]} == keys, seed
        assert _statements(q)[int(q.parts[3].answer_text)] == ("Reject" if reject else "Do not reject"), seed


def test_exam_conclusions_are_balanced_and_positions_vary():
    rejects = sum(exam_oracle(_exam(s))[1] for s in range(200))
    assert 70 <= rejects <= 130
    assert {_exam(s).parts[3].answer_text for s in range(60)} == {"1", "2", "3"}


def test_exam_shapes_and_levels_appear():
    shapes, dfs, levels = set(), set(), set()
    for s in range(200):
        q = _exam(s)
        stem = q.text
        shapes.add("gof" if "goodness-of-fit" in stem else "table")
        dfs.add(q.parts[2].answer_text)
        levels.add(re.search(r"at the (\d)% level", stem).group(1))
    assert shapes == {"gof", "table"} and {"1", "2"} <= dfs and levels == {"1", "5"}


def test_key_accepted_and_every_distractor_rejected():
    for seed in range(120):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 6
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2
            for text in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: text}).parts[p.label].marks == 0, (seed, p.label, text)


def test_follow_through_from_a_wrong_expected_count():
    checked = 0
    for seed in range(150):
        q = _exam(seed)
        _keys, _rej, obs, exp, idx, stat = exam_oracle(q)
        wrong = float(q.parts[0].distractors[0])
        rest = stat - (obs[idx] - exp[idx]) ** 2 / exp[idx]
        ft = rest + (obs[idx] - wrong) ** 2 / wrong
        reply = {"a": f"{wrong:.2f}", "b": _r(ft, 2), "c": q.parts[2].answer_text, "d": q.parts[3].answer_text}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 3 and result.parts["b"].follow_through is True, (seed, result.parts["b"])
        assert result.parts["c"].marks == 1 and result.parts["d"].marks == 1
        checked += 1
    assert checked == 150


def test_the_rounded_expected_count_can_be_used_downstream():
    for seed in range(80):
        q = _exam(seed)
        _keys, _rej, obs, exp, idx, stat = exam_oracle(q)
        e2 = float(q.parts[0].answer_text)
        rest = stat - (obs[idx] - exp[idx]) ** 2 / exp[idx]
        reply = {
            "a": q.parts[0].answer_text,
            "b": _r(rest + (obs[idx] - e2) ** 2 / e2, 2),
            "c": q.parts[2].answer_text,
            "d": q.parts[3].answer_text,
        }
        assert grade(_SKILL, q, reply).marks == 6, seed


def test_exam_answer_space_and_determinism():
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 6
    assert _exam(7).to_json() == _exam(7).to_json()
