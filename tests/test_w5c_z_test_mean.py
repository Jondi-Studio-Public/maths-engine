"""Tests the skill `ucv1-z-test-mean` (topic `w5c_z_test_mean`) and its exam chain `exd_x_w5c_z_test_mean`.

The oracle reads the numbers back out of the question TEXT (sample size, means, sigma or variance, critical value, the
printed table rows) and re-derives each key with `math.erfc`, never with the generator's own table helpers."""

from __future__ import annotations

import math
import re
from decimal import ROUND_HALF_UP, Decimal

import pytest

import mathgen.generators.w5c_z_test_mean_build as module
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects

TOPIC = "w5c_z_test_mean"
EXAM_TOPIC = "exd_x_w5c_z_test_mean"
SKILL_ID = "ucv1-z-test-mean"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
TIERS = ("easy", "medium", "hard")
SEEDS = range(60)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM_TOPIC, seed=seed, difficulty="exam")


def _r(x: float, dp: int) -> str:
    return str(Decimal(repr(round(x, 12))).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP))


def phi(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2))


def table(z: float) -> float:
    """Phi(|z|) as a 4-place table prints it."""
    return float(_r(phi(abs(z)), 4))


def p_value(z: float, tail: str) -> float:
    t = table(z)
    if tail == "two":
        return round(2 * (1 - t), 4)
    if tail == "upper":
        return round(1 - t if z >= 0 else t, 4)
    return round(1 - t if z <= 0 else t, 4)


# --- the printed table -------------------------------------------------------------------------------------------


def test_the_table_agrees_with_mpmath_at_30_digits():
    import mpmath
    from fractions import Fraction

    with mpmath.workdps(30):
        for k in range(0, 400):
            exact = mpmath.ncdf(mpmath.mpf(k) / 100)
            assert module.phi_table(Fraction(k, 100)) == Fraction(mpmath.nstr(mpmath.nint(exact * 10000) / 10000, 6)), k


def test_z_and_p_agree_with_mpmath_over_many_seeds():
    import mpmath

    for seed in range(40):
        q = _q("easy", seed)
        n, xbar, mu0, sigma = _facts(q.text)
        with mpmath.workdps(40):
            z = (mpmath.mpf(str(xbar)) - mpmath.mpf(str(mu0))) / (mpmath.mpf(str(sigma)) / mpmath.sqrt(n))
        assert abs(float(q.answer_text) - float(z)) < 0.0051


@pytest.mark.parametrize("z100", [0, 17, 64, 100, 164, 196, 233, 258, 299])
def test_table_values_against_erfc(z100):
    from fractions import Fraction

    assert float(module.phi_table(Fraction(z100, 100))) == float(_r(phi(z100 / 100), 4))


def test_printed_excerpts_are_real_table_rows():
    for tier in ("medium", "hard"):
        for seed in SEEDS:
            text = _q(tier, seed).text
            rows = re.findall(r"Φ\((\d\.\d\d)\) = (\d\.\d{4})", text)
            if not rows:
                continue
            assert len(rows) == 5, (tier, seed)
            for z, value in rows:
                assert float(value) == float(_r(phi(float(z)), 4)), (tier, seed, z, value)
            zs = [float(z) for z, _ in rows]
            assert zs == sorted(zs) and all(abs(b - a - 0.01) < 1e-9 for a, b in zip(zs, zs[1:]))


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


def _facts(text):
    n = int(re.search(r"(?:sample of|takes|Using a sample of|on) (\d+) ", text).group(1))
    m = re.search(r"random sample of (\d+) \w+ has a mean [a-z ]+? of ([\d.]+) ", text)
    xbar = float(m.group(2)) if m else None
    m0 = re.search(r"H0: μ = ([\d.]+)", text) or re.search(
        r"true mean (?:is greater than|is less than|differs from) ([\d.]+) ", text
    )
    mu0 = float(m0.group(1))
    var = re.search(r"variance[^\d]*([\d.]+)", text)
    if var:
        sigma = math.sqrt(float(var.group(1)))
    else:
        sigma = float(re.search(r"standard deviation[^\d]*([\d.]+)", text).group(1))
    return n, xbar, mu0, sigma


def _dp_of(text_number: str) -> int:
    return len(text_number.split(".")[1]) if "." in text_number else 0


def oracle_easy(q):
    n, xbar, mu0, sigma = _facts(q.text)
    return _r((xbar - mu0) / (sigma / math.sqrt(n)), 2)


def _tail_of(text):
    h1 = re.search(r"H1: μ (.) ", text).group(1)
    return {"≠": "two", ">": "upper", "<": "lower"}[h1]


def oracle_medium(q):
    n, xbar, mu0, sigma = _facts(q.text)
    z = float(_r((xbar - mu0) / (sigma / math.sqrt(n)), 2))
    return f"{p_value(z, _tail_of(q.text)):.4f}"


def _kind(q):
    t = q.text
    if "smallest" in t or "least number" in t or "at a minimum" in t:
        return "min_n"
    if "type II" in t or "fails to reject" in t or "not rejected" in t:
        return "type2"
    return "reverse"


def _zstar(text):
    m = re.search(r"z [<>] (?:−)?(\d+(?:\.\d+)?)|\|z\| > (\d+(?:\.\d+)?)", text)
    return float(m.group(1) or m.group(2))


def oracle_hard(q):
    t = q.text
    kind = _kind(q)
    if kind == "min_n":
        m = re.search(r"(?:mean of|to be|mean turns out to be) ([\d.]+) ", t.split("rejects H0")[-1]) or re.search(
            r"mean (?:of|is expected to be|turns out to be) ([\d.]+) ", t.split("rejected")[-1]
        )
        mu0 = float(re.search(r"H0: μ = ([\d.]+)", t).group(1))
        sigma = float(re.search(r"standard deviation[^\d]*([\d.]+)", t).group(1))
        xbar = float(re.search(r"(?:gives a mean of|turns out to be|expected to be) ([\d.]+)", t).group(1))
        zs = _zstar(t)
        n = next(n for n in range(2, 1000) if abs(xbar - mu0) / (sigma / math.sqrt(n)) > zs)
        return str(n)
    if kind == "type2":
        mu0 = float(re.search(r"H0: μ = ([\d.]+)", t).group(1))
        sigma = float(re.search(r"standard deviation[^\d]*([\d.]+)", t).group(1))
        n = int(re.search(r"(?:sample of|takes|Using a sample of) (\d+) ", t).group(1))
        mu1 = float(re.search(r"(?:actually|in fact|true mean is) ([\d.]+)", t).group(1))
        zs = _zstar(t)
        se = sigma / math.sqrt(n)
        z_beta = float(_r(zs - abs(mu1 - mu0) / se, 2))
        beta = phi(z_beta)
        beta_table = table(z_beta) if z_beta >= 0 else 1 - table(z_beta)
        assert abs(beta - beta_table) < 6e-5
        return f"{beta_table:.4f}"
    mu0 = float(re.search(r"H0: μ = ([\d.]+)", t).group(1))
    sigma = float(re.search(r"standard deviation[^\d]*([\d.]+)", t).group(1))
    n = (
        int(re.search(r"(?:of|on) (\d+) \w+ (?:gave|had|reported)|sample of (\d+)", t).group(1) or 0)
        if False
        else int(re.search(r"(\d+) (?:cartons|trips|batteries|loaves|swims|students)", t).group(1))
    )
    p = float(re.search(r"p(?:-value of| =) (0\.\d{4})", t).group(1))
    above = " above " in t
    z = min((k / 100 for k in range(0, 400)), key=lambda zz: abs(2 * (1 - table(zz)) - p))
    dp = int(re.search(r"correct to (\d) decimal place", t).group(1))
    return _r(mu0 + (1 if above else -1) * z * sigma / math.sqrt(n), dp)


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_the_independent_derivation(tier):
    fn = {"easy": oracle_easy, "medium": oracle_medium, "hard": oracle_hard}[tier]
    bad = [s for s in range(200) if fn(_q(tier, s)) != _q(tier, s).answer_text]
    assert bad == []


def test_hard_tier_uses_all_three_shapes():
    kinds = {_kind(_q("hard", s)) for s in range(60)}
    assert kinds == {"min_n", "type2", "reverse"}


def test_medium_tier_uses_every_tail_and_both_sides_of_the_null():
    seen = set()
    for s in range(120):
        q = _q("medium", s)
        n, xbar, mu0, _ = _facts(q.text)
        seen.add((_tail_of(q.text), xbar > mu0))
    assert {("upper", True), ("upper", False), ("lower", True), ("lower", False), ("two", True), ("two", False)} <= seen


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
        if "." in q.answer_text:
            dp = len(q.answer_text.split(".")[1])
            off = float(q.answer_text) + 2 * 10**-dp
            assert not grade(_SKILL, q, f"{off:.{dp}f}").correct


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


def test_tiers_differ_in_shape():
    easy = {_q("easy", s).answer_text for s in range(20)}
    assert all("Φ" not in _q("easy", s).text for s in range(20))
    assert all("Φ" in _q("medium", s).text for s in range(20))
    assert all(re.fullmatch(r"-?\d\.\d\d", a) for a in easy)


def test_critical_values_are_printed_in_the_hard_stem():
    for seed in range(60):
        q = _q("hard", seed)
        if _kind(q) != "reverse":
            assert re.search(r"z [<>] |\|z\| >", q.text)


# --- the exam chain ---------------------------------------------------------------------------------------------


def _exam_facts(q):
    stem = q.text.split(" (a)")[0]
    n, xbar, mu0, sigma = _facts(stem)
    tail = _tail_of(stem)
    zstar = float(re.search(r"z\* = (\d+(?:\.\d+)?)", stem).group(1))
    level = re.search(r"at the (\d+)% level", stem).group(1)
    return n, xbar, mu0, sigma, tail, zstar, int(level) / 100


def exam_oracle(q):
    n, xbar, mu0, sigma, tail, zstar, alpha = _exam_facts(q)
    se = sigma / math.sqrt(n)
    z = float(_r((xbar - mu0) / se, 2))
    p = p_value(z, tail)
    edge = mu0 + (1 if xbar > mu0 else -1) * zstar * se
    dp = _dp_of(re.search(r"H0: μ = ([\d.]+)", q.text).group(1))
    return dict(a=f"{z:.2f}", b=f"{p:.4f}", c=_r(edge, 2)), p < alpha


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
        assert [p.marks for p in q.parts] == [2, 2, 2, 1]
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
    assert {_exam(s).parts[3].answer_text for s in range(60)} == {"1", "2", "3"}


def test_every_tail_and_level_appears():
    seen = {_exam_facts(_exam(s))[4] for s in range(100)}
    assert seen == {"upper", "lower", "two"}


def test_key_accepted_and_every_distractor_rejected_in_the_exam():
    for seed in range(120):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 7
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2
            for text in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text)


def test_follow_through_from_a_wrong_z():
    checked = 0
    for seed in range(150):
        q = _exam(seed)
        n, xbar, mu0, sigma, tail, zstar, alpha = _exam_facts(q)
        wrong_z = float(q.parts[0].distractors[0])
        if abs(wrong_z) < 0.2:
            continue
        p_ft = p_value(wrong_z, tail)
        if not 0 < p_ft < 1:
            continue
        # the boundary recovers a wrong standard error from the wrong z
        edge_ft = mu0 + zstar * (xbar - mu0) / abs(wrong_z)
        reply = {"a": f"{wrong_z:.2f}", "b": f"{p_ft:.4f}", "c": _r(edge_ft, 2), "d": q.parts[3].answer_text}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through is True, (seed, result.parts["b"])
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True, (seed, result.parts["c"])
        checked += 1
    assert checked >= 80


def test_a_wrong_critical_value_is_not_credited_in_part_c():
    for seed in range(60):
        q = _exam(seed)
        n, xbar, mu0, sigma, tail, zstar, alpha = _exam_facts(q)
        se = sigma / math.sqrt(n)
        other = 1.960 if abs(zstar - 1.960) > 1e-9 else 1.645
        edge = mu0 + (1 if xbar > mu0 else -1) * other * se
        canon = {p.label: p.answer_text for p in q.parts}
        if _r(edge, 2) != canon["c"]:
            assert grade(_SKILL, q, {**canon, "c": _r(edge, 2)}).parts["c"].marks == 0
