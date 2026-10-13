"""Tests the skill `ucv1-proof-by-induction`: an independent oracle that re-parses the statement from the question
TEXT with SymPy and re-derives each step (brute-force sums, f(k + 1) - a f(k)), the typed expression grader on every
key and distractor, the "factored" form, and the exam chains with follow-through."""

from __future__ import annotations

import re

import pytest
import sympy
from sympy import Integer, Rational, Symbol

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects

TOPIC = "ucv1_proof_by_induction"
EXAM_TOPIC = "exd_x_uc_induction"
SKILL_ID = "ucv1-proof-by-induction"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
TIERS = ("easy", "medium", "hard")
SEEDS = range(60)
n, k, q = Symbol("n"), Symbol("k"), Symbol("q")


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM_TOPIC, seed=seed, difficulty="exam")


# --- reading the maths back out of the text -------------------------------------------------------------------


def parse(text: str):
    """A learner-facing expression (× ² ^ and juxtaposition) as a SymPy expression in n, k or q."""
    s = text.replace("×", "*").replace("²", "**2").replace("³", "**3").replace("^", "**").replace("−", "-")
    s = re.sub(r"\)(?=[\w(])", ")*", s)  # ")(" ")n" ")2"
    s = re.sub(r"(?<=\d)(?=[a-zA-Z(])", "*", s)  # "2n" "2("
    s = re.sub(r"(?<=[nkq])(?=\()", "*", s)  # "n(" "k("
    s = re.sub(r"(?<=\*\*)\s*(?=\()", "", s)
    return sympy.sympify(s, locals={"n": n, "k": k, "q": q})


def parse_statement(text: str):
    """(general term as a function of n, right-hand side in n) from 'P(n): a + b + c + … + GENERAL = RHS'."""
    body = re.search(
        r"(?:statement P\(n\): |statement )([^,]*?), for (?:all )?integers n ≥ 1|statement ([^,]*?) is to be proved",
        text,
    )
    stmt = next(g for g in body.groups() if g)
    lhs, rhs = stmt.split(" = ")
    general = lhs.split(" + … + ")[1]
    return parse(general), parse(rhs)


def brute(term, upto):
    return sum(term.subs(n, i) for i in range(1, upto + 1))


def assert_true_statement(term, rhs):
    for m in range(1, 9):
        assert sympy.nsimplify(brute(term, m) - rhs.subs(n, m)) == 0


def num_text(value) -> str:
    value = Rational(value)
    return str(int(value)) if value.q == 1 else str(value)


def same(a, b) -> bool:
    return sympy.simplify(a - b) == 0


# --- oracle ------------------------------------------------------------------------------------------------------


def oracle(tier, q_):
    text = q_.text
    if "f(n) = " in text:
        return oracle_div(text)
    if tier == "easy":
        term, rhs = parse_statement(text)
        assert_true_statement(term, rhs)
        if "common value" in text:
            assert term.subs(n, 1) == rhs.subs(n, 1)
            return ("number", Rational(term.subs(n, 1)))
        return ("expr", term.subs(n, k + 1))
    if tier == "medium":
        term, rhs = parse_statement(text)
        assert_true_statement(term, rhs)
        return ("expr", rhs.subs(n, k + 1))
    # hard, sums: hypothesis in expanded form
    stmt = re.search(r"The sum (.*?) is being evaluated", text).group(1)
    general = parse(stmt.split(" + … + ")[1])
    hyp = parse(re.search(r"first k terms is (.*?)\. Add", text).group(1))
    for m in range(1, 9):  # the hypothesis is the real sum
        assert sympy.nsimplify(brute(general, m) - hyp.subs(k, m)) == 0
    return ("expr", hyp + general.subs(n, k + 1))


def oracle_div(text):
    cubic = re.search(r"f\(n\) = (n³ [-+] \d?n) is divisible", text)
    if cubic:
        f = parse(cubic.group(1)).subs(n, k)
        return ("expr", f.subs(k, k + 1) - f)
    power = re.search(r"f\(n\) = (\d+)\^n - (\d+)\^n is divisible", text)
    if power:
        a, b = int(power.group(1)), int(power.group(2))
        return (
            "expr",
            sympy.expand(Integer(a) ** (k + 1) - Integer(b) ** (k + 1) - a * (Integer(a) ** k - Integer(b) ** k)),
        )
    lin = re.search(r"f\(n\) = (\d+)\^n ([-+]) (\d+) is divisible by (\d+)", text)
    a, sign, b, d = int(lin.group(1)), lin.group(2), int(lin.group(3)), int(lin.group(4))
    b = b if sign == "+" else -b
    assert (a + b) % d == 0
    if "Find the value of f(1)" in text:
        return ("number", Rational(a + b))
    mult = int(re.search(r"f\(k \+ 1\) = (\d+)f\(k\) \+ m", text).group(1))
    assert mult == a
    # m from two evaluations of f(k + 1) - a f(k): constant in k
    values = {(a ** (j + 1) + b) - a * (a**j + b) for j in range(1, 6)}
    assert len(values) == 1
    return ("number", Rational(values.pop()))


def key_matches(tier, q_) -> bool:
    kind, expected = oracle(tier, q_)
    if kind == "number":
        return Rational(parse(q_.answer_text)) == expected
    return same(parse(q_.answer_text), expected)


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_the_independent_derivation(tier):
    bad = [s for s in range(150) if not key_matches(tier, _q(tier, s))]
    assert bad == []


def test_every_shape_appears():
    easy = {
        "base" if "common value" in _q("easy", s).text else "f1" if "value of f(1)" in _q("easy", s).text else "term"
        for s in range(60)
    }
    assert easy == {"base", "f1", "term"}
    hard = set()
    for s in range(80):
        text = _q("hard", s).text
        hard.add(
            "cubic"
            if "n³" in text
            else "power"
            if "^n - " in text
            else "number"
            if "where m is a number" in text
            else "factored"
        )
    assert hard == {"cubic", "power", "number", "factored"}


def test_sum_families_both_polynomial_and_not_appear_before_hard():
    easy_text = " ".join(_q("easy", s).text for s in range(80))
    assert "2^n" in easy_text and "1/(" in easy_text and "n²" in easy_text


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in SEEDS:
        q_ = _q(tier, seed)
        assert grade(_SKILL, q_, q_.answer_text).correct, (tier, seed, q_.answer_text)
        assert len(q_.distractors) >= 2
        assert all(q_.option_notes.get(d) for d in q_.distractors)
        for d in q_.distractors:
            assert not grade(_SKILL, q_, d).correct, (tier, seed, d)


@pytest.mark.parametrize("tier", TIERS)
def test_each_distractor_is_diagnosed_as_its_own_error(tier):
    for seed in range(30):
        q_ = _q(tier, seed)
        for d in q_.distractors:
            assert grade(_SKILL, q_, d).misconception == q_.option_notes[d], (tier, seed, d)


def test_an_equivalent_form_is_accepted_for_the_target_expression():
    for seed in range(60):
        q_ = _q("medium", seed)
        expanded = str(sympy.expand(parse(q_.answer_text))).replace("**", "^")
        assert grade(_SKILL, q_, expanded).correct, (seed, q_.answer_text, expanded)
        assert not grade(_SKILL, q_, str(sympy.expand(parse(q_.answer_text) + 1)).replace("**", "^")).correct


def test_factored_form_is_required_when_asked():
    checked = 0
    for seed in range(100):
        q_ = _q("hard", seed)
        if q_.answer_form != "factored":
            continue
        expanded = str(sympy.expand(parse(q_.answer_text))).replace("**", "^")
        assert not grade(_SKILL, q_, expanded).correct, (seed, expanded)
        checked += 1
    assert checked >= 20


@pytest.mark.parametrize("tier", TIERS)
def test_variety_and_no_dominant_answer(tier):
    answers = [_q(tier, s).answer_text for s in SEEDS]
    assert len(set(answers)) >= 12
    assert max(answers.count(a) for a in set(answers)) <= 12
    assert len({_q(tier, s).text for s in SEEDS}) >= 30


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic_and_clean_text(tier):
    for seed in SEEDS:
        q_ = _q(tier, seed)
        assert q_.to_json() == _q(tier, seed).to_json()
        assert find_text_defects(q_) == []


def test_no_multiple_choice_anywhere():
    for tier in TIERS:
        for seed in range(30):
            assert _q(tier, seed).answer_kind != "mcq"
    for seed in range(30):
        assert {p.answer_kind for p in _exam(seed).parts} == {"expression"}


# --- the exam chains --------------------------------------------------------------------------------------------


def is_div(q_):
    return "divisible by" in q_.text


def exam_oracle_sum(q_):
    term, rhs = parse_statement(q_.text.replace("Let P(n) be the statement ", "statement P(n): "))
    assert_true_statement(term, rhs)
    base = term.subs(n, 1)
    t_next = term.subs(n, k + 1)
    hyp = parse(re.search(r"first k terms is (.*?)\. Add", q_.parts[2].prompt).group(1))
    for m in range(1, 9):
        assert sympy.nsimplify(brute(term, m) - hyp.subs(k, m)) == 0
    s_next = hyp + t_next
    assert same(s_next, rhs.subs(n, k + 1))
    d_prompt = q_.parts[3].prompt
    if "least number of terms" in d_prompt:
        thr = int(re.search(r"exceeds (\d+)", d_prompt).group(1))
        d = next(m for m in range(1, 200) if brute(term, m) > thr)
    else:
        big = int(re.search(r"first (\d+) terms", d_prompt).group(1))
        d = brute(term, big)
    return Rational(base), t_next, s_next, Rational(d), term, hyp


def exam_oracle_div(q_):
    a, sign, b, d = re.search(r"f\(n\) = (\d+)\^n ([-+]) (\d+) is divisible by (\d+)", q_.text).groups()
    a, d = int(a), int(d)
    b = int(b) if sign == "+" else -int(b)
    f = lambda e: a**e + b  # noqa: E731
    big = int(re.search(r"f\((\d)\) is divisible", q_.parts[3].prompt).group(1))
    m = f(2) - a * f(1)
    assert all(f(j + 1) - a * f(j) == m for j in range(1, 7))
    return Rational(f(1)), Rational(m), a * q + Rational(m, d), Rational(f(big), d), a, d


def test_shape_marks_and_follow_through():
    kinds = set()
    for seed in range(100):
        q_ = _exam(seed)
        assert [p.label for p in q_.parts] == ["a", "b", "c", "d"]
        assert find_text_defects(q_) == []
        if is_div(q_):
            kinds.add("div")
            assert [p.marks for p in q_.parts] == [1, 2, 2, 1]
            assert [p.follow_through is not None for p in q_.parts] == [False, False, True, False]
        else:
            kinds.add("sum")
            assert [p.marks for p in q_.parts][:3] == [1, 1, 3]
            assert q_.parts[2].follow_through is not None and q_.parts[2].answer_form == "factored"
            assert q_.parts[0].follow_through is None and q_.parts[1].follow_through is None
    assert kinds == {"sum", "div"}


def test_exam_keys_match_the_independent_derivation():
    for seed in range(300):
        q_ = _exam(seed)
        parts = {p.label: p for p in q_.parts}
        if is_div(q_):
            f1, m, quotient, final, *_ = exam_oracle_div(q_)
            assert Rational(parse(parts["a"].answer_text)) == f1
            assert Rational(parse(parts["b"].answer_text)) == m
            assert same(parse(parts["c"].answer_text), quotient)
            assert Rational(parse(parts["d"].answer_text)) == final
        else:
            base, t_next, s_next, d, *_ = exam_oracle_sum(q_)
            assert Rational(parse(parts["a"].answer_text)) == base
            assert same(parse(parts["b"].answer_text), t_next)
            assert same(parse(parts["c"].answer_text), s_next)
            assert Rational(parse(parts["d"].answer_text)) == d


def test_both_uses_of_the_closed_form_appear():
    uses = {
        "least" if "least number" in _exam(s).parts[3].prompt else "value" for s in range(100) if not is_div(_exam(s))
    }
    assert uses == {"least", "value"}


def test_key_accepted_and_every_distractor_rejected():
    for seed in range(100):
        q_ = _exam(seed)
        full = grade(_SKILL, q_, q_.answer_text)
        assert full.correct and full.marks == full.max_marks, (seed, full)
        canon = {p.label: p.answer_text for p in q_.parts}
        for p in q_.parts:
            assert len(p.distractors) >= 2
            for text in p.distractors:
                assert grade(_SKILL, q_, {**canon, p.label: text}).parts[p.label].marks == 0, (seed, p.label, text)


def test_follow_through_from_a_wrong_next_term_into_the_factored_sum():
    checked = 0
    for seed in range(150):
        q_ = _exam(seed)
        if is_div(q_):
            continue
        _base, _t, _s, _d, term, hyp = exam_oracle_sum(q_)
        wrong_b = q_.parts[1].distractors[0]
        ft = sympy.factor(hyp + parse(wrong_b))
        if not (ft.is_Mul or ft.is_Pow):
            continue  # an irreducible value cannot be typed in factored form
        reply = {
            "a": q_.parts[0].answer_text,
            "b": wrong_b,
            "c": str(ft).replace("**", "^"),
            "d": q_.parts[3].answer_text,
        }
        result = grade(_SKILL, q_, reply)
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == 3 and result.parts["c"].follow_through is True, (
            seed,
            wrong_b,
            ft,
            result.parts["c"],
        )
        checked += 1
    assert checked >= 50


def test_follow_through_from_a_wrong_sum_into_its_use():
    checked = 0
    for seed in range(150):
        q_ = _exam(seed)
        if is_div(q_) or "first" not in q_.parts[3].prompt or "least" in q_.parts[3].prompt:
            continue
        big = int(re.search(r"first (\d+) terms", q_.parts[3].prompt).group(1))
        wrong_c = q_.parts[2].distractors[0]
        value = parse(wrong_c).subs(k, big - 1)
        reply = {"a": q_.parts[0].answer_text, "b": q_.parts[1].answer_text, "c": wrong_c, "d": str(value)}
        result = grade(_SKILL, q_, reply)
        assert result.parts["c"].marks == 0
        assert result.parts["d"].marks == 2 and result.parts["d"].follow_through is True, (seed, result.parts["d"])
        checked += 1
    assert checked >= 30


def test_follow_through_from_a_wrong_remainder_in_the_divisibility_chain():
    checked = 0
    for seed in range(200):
        q_ = _exam(seed)
        if not is_div(q_):
            continue
        _f1, _m, _quot, _final, a, d = exam_oracle_div(q_)
        wrong_m = int(q_.parts[1].distractors[0])
        reply = {
            "a": q_.parts[0].answer_text,
            "b": str(wrong_m),
            "c": str(a * q + Rational(wrong_m, d)).replace("**", "^"),
            "d": q_.parts[3].answer_text,
        }
        result = grade(_SKILL, q_, reply)
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True, (seed, result.parts["c"])
        checked += 1
    assert checked >= 15


def test_exam_answer_space_and_determinism():
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 10
    assert _exam(7).to_json() == _exam(7).to_json()
    assert _SKILL.sources[-1].params == {"tier": "exam"} and _SKILL.sources[-1].stages == ["exam"]
