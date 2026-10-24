"""Tests `implication-converse-contrapositive-counts`: every key is re-derived from the printed stem.

The oracle reads the two clauses and the universe from the question text with its own small parser, builds the truth
table of "if P then Q" at every n (or reads the explicit sets), and evaluates the statement the question names.
"""

from __future__ import annotations

import math
import re

from r3d_support import (
    assert_deterministic_and_varied,
    assert_exam_rules,
    assert_follow_through_credit,
    assert_number_distractors_match_places,
    assert_single_answer_rules,
    questions,
    skill,
    sources,
)

from mathgen import make

SK = skill("implication-converse-contrapositive-counts")
SEEDS = range(60)


def clause(text):
    """A function n -> bool for the printed clause."""
    t = text.strip()
    if m := re.fullmatch(r"n is a multiple of (\d+)", t):
        k = int(m.group(1))
        return lambda n: n % k == 0
    if m := re.fullmatch(r"n is a factor of (\d+)", t):
        k = int(m.group(1))
        return lambda n: k % n == 0
    if t == "n is odd":
        return lambda n: n % 2 == 1
    if t == "n is even":
        return lambda n: n % 2 == 0
    if t == "n is prime":
        return lambda n: n > 1 and all(n % d for d in range(2, n))
    if t == "n is a perfect square":
        return lambda n: math.isqrt(n) ** 2 == n
    if t == "n is a perfect cube":
        return lambda n: round(n ** (1 / 3)) ** 3 == n
    if m := re.fullmatch(r"n ends in the digit (\d)", t):
        d = int(m.group(1))
        return lambda n: n % 10 == d
    if m := re.fullmatch(r"n leaves a remainder of ([\d or]+) when divided by (\d+)", t):
        rs = {int(x) for x in re.findall(r"\d+", m.group(1))}
        k = int(m.group(2))
        return lambda n: n % k in rs
    raise AssertionError(t)


def statement_clauses(text):
    m = re.search(r"S: “if (.*?), then (.*?)”", text)
    return clause(m.group(1)), clause(m.group(2))


def sets_and_universe(text):
    if text.startswith("The universal set"):
        top = int(re.search(r"\{1, 2, \.\.\., (\d+)\}", text).group(1))
        a, b = (
            {int(x) for x in re.findall(r"\d+", g)}
            for g in re.search(r"A = \{([\d, ]+)\} and B = \{([\d, ]+)\}", text).groups()
        )
        return a, b, top
    top = int(re.search(r"from 1 to (\d+)", text).group(1))
    p, q = statement_clauses(text)
    return {n for n in range(1, top + 1) if p(n)}, {n for n in range(1, top + 1) if q(n)}, top


def truth(form, p, q):
    return {
        "statement": (not p) or q,
        "converse": (not q) or p,
        "inverse": p or (not q),
        "contrapositive": q or (not p),
        "iff": p == q,
    }[form]


def oracle(text):
    a, b, top = sets_and_universe(text)
    ns = range(1, top + 1)

    def fails(form):
        return [n for n in ns if not truth(form, n in a, n in b)]

    def holds(form):
        return [n for n in ns if truth(form, n in a, n in b)]

    if "smallest value of n that is a counterexample to the converse" in text:
        return min(fails("converse"))
    if "smallest value of n that is a counterexample to S" in text:
        return min(fails("statement"))
    if "counterexamples to the converse?" in text:
        return len(fails("converse"))
    if "counterexamples to the contrapositive of S" in text:
        return len(fails("contrapositive"))
    if "negation of the converse" in text:
        return len([n for n in ns if n in b and n not in a])
    if "negation of S true" in text:
        return len([n for n in ns if n in a and n not in b])
    if "is the statement S true" in text:
        return len(holds("statement"))
    if "the converse true" in text:
        return len(holds("converse"))
    if "the inverse is true" in text or "is the inverse true" in text:
        return len(holds("inverse"))
    if "if and only if Q” true" in text or "if and only if Q” true?" in text:
        return len(holds("iff"))
    raise AssertionError(text)


def check_tier(tier, seeds=range(250)):
    for q in questions(SK, tier, seeds):
        assert_number_distractors_match_places(q)
        assert int(q.answer_text) == oracle(q.text), (q.text, q.answer_text)


def test_easy():
    assert_single_answer_rules(SK, "easy", SEEDS)
    check_tier("easy")


def test_medium():
    assert_single_answer_rules(SK, "medium", SEEDS)
    check_tier("medium")


def test_hard():
    assert_single_answer_rules(SK, "hard", SEEDS)
    check_tier("hard")


def test_tiers_use_different_questions():
    def asked(tier):
        return {
            re.sub(r"^.*?S[.?] ", "", q.text).split(" For how many")[0][:40] for q in questions(SK, tier, range(80))
        }

    text = {t: " ".join(q.text for q in questions(SK, t, range(120))) for t in ("easy", "medium", "hard")}
    assert "smallest value" in text["easy"] and "smallest value" not in text["medium"]
    assert "contrapositive" in text["medium"] and "negation of S true" in text["medium"]
    assert "if and only if" in text["hard"] and "negation of the converse" in text["hard"]
    assert "universal set" in text["hard"] and "universal set" not in text["medium"]
    assert asked("easy")


def test_exam_chain_and_keys():
    assert_exam_rules(SK, range(40), labels="abcde", marks=[1, 1, 2, 2, 2])
    topic = sources(SK)["exam"]
    for seed in range(150):
        q = make(topic, seed=seed, difficulty="exam")
        p, c = statement_clauses(q.text)
        top = int(re.search(r"from 1 to (\d+)", q.text).group(1))
        a = {n for n in range(1, top + 1) if p(n)}
        b = {n for n in range(1, top + 1) if c(n)}
        got = [int(part.answer_text) for part in q.parts]
        want_c = len([n for n in range(1, top + 1) if not truth("contrapositive", n in a, n in b)])
        want_d = len([n for n in range(1, top + 1) if not truth("converse", n in a, n in b)])
        want_e = len([n for n in range(1, top + 1) if not truth("iff", n in a, n in b)])
        assert got == [len(a), len(b), want_c, want_d, want_e], (q.text, got)
        assert got[3] == got[1] - got[0] + got[2] and got[4] == got[2] + got[3]
        for part in q.parts:
            assert_number_distractors_match_places(part)


def test_exam_follow_through():
    assert_follow_through_credit(SK, sources(SK)["exam"], range(40), min_checked=10)


def test_deterministic_varied_and_clean():
    for tier, distinct in (("easy", 25), ("medium", 25), ("hard", 25)):
        assert_deterministic_and_varied("sc_conditional_counts", tier, distinct=distinct)
        assert all("**" not in q.text for q in questions(SK, tier, range(30)))
    assert_deterministic_and_varied("sc_conditional_counts_exam", "exam", distinct=25)


def test_easy_smallest_counterexample_is_not_just_the_first_value_satisfying_the_hypothesis():
    hits = total = 0
    for seed in range(400):
        q = make("sc_conditional_counts", seed=seed, difficulty="easy")
        if "smallest value" not in q.text:
            continue
        total += 1
        first = int(re.search(r"start ([\d, .]+); the first", q.working[1])[1].split(",")[0])
        hits += first == int(q.answer)
    assert total >= 50 and hits / total < 0.25, (hits, total)
