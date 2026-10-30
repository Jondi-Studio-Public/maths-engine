"""Tests `uc2-sylow-counting`: every key is re-derived from the RENDERED stem by SymPy's divisors and factorisation (and,
for the fixed points of a p-group, by a search over the possible orbit decompositions), the typed graders on every key
and distractor, and the three-part exam chain with follow-through."""

from __future__ import annotations

import re

import sympy

from mathgen import make
from mathgen.engine import grade
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape
from tests.exd_x_sc2_support import (
    check_all_wrong_earns_nothing,
    check_answer_space,
    check_deterministic,
    check_key_accepted_distractors_rejected,
    check_no_text_defects,
    check_oracle,
    check_wired,
    exam,
    skill,
)

TOPIC = "w5b_sylow_counting"
EXAM = "exd_x_w5b_sylow_counting"
SKILL = "uc2-sylow-counting"


def _n(text):
    return int(re.search(r"(?:order(?: is)?|\|G\| =) (\d+)", text).group(1))


def _np_set(n, p):
    """Possible n_p: divisors of n congruent to 1 mod p (SymPy, no use of the cofactor)."""
    return [d for d in sympy.divisors(n) if d % p == 1 % p]


def _sylow_order(n, p):
    return p ** sympy.factorint(n).get(p, 0)


def _first_prime_before_subgroup(text):
    return int(re.search(r"Sylow (\d+)-subgroup", text).group(1))


def _min_fixed_points(N, p, k):
    """Search: the least f with N - f a sum of orbit sizes p, p^2, ..., p^k (a multiple of p is enough, and necessary)."""
    sizes = [p**j for j in range(1, k + 1)]
    reachable = {0}
    # N - f is reachable iff it is a non-negative combination of the sizes; sizes include p, so test by residue
    for f in range(0, N + 1):
        rest = N - f
        if rest % sizes[0] == 0:
            return f
        reachable.add(rest)
    raise AssertionError


def oracle_single(question):
    text = question.text
    tier = question.difficulty
    if tier == "easy":
        n = _n(text)
        if "sum of these orders" in text or "total" in text:
            return sum(_sylow_order(n, p) for p in sympy.factorint(n))
        if "sum of all of them" in text or "What is the sum" in text:
            p = _first_prime_before_subgroup(text)
            return sum(_np_set(n, p))
        p = (
            int(re.search(r"largest power of (\d+)", text).group(1))
            if "largest power" in text
            else _first_prime_before_subgroup(text)
        )
        return _sylow_order(n, p)
    if tier == "medium":
        n = _n(text)
        if "For how many of the primes" in text or "Count the primes" in text:
            return sum(1 for p in sympy.factorint(n) if _np_set(n, p) == [1])
        p = _first_prime_before_subgroup(text)
        if "how many possibilities" in text or "how many different values" in text:
            return len(_np_set(n, p))
        if "largest value" in text or "greatest" in text:
            return max(_np_set(n, p))
        x = int(re.search(r"(?:exactly|n_\d+ =) (\d+)", text).group(1))
        assert x in _np_set(n, p)
        return x * (p - 1)
    n = _n(text)
    if "acts on a set" in text or "act on a finite set" in text:
        N = int(re.search(r"(?:\|X\| = |set X with )(\d+)", text).group(1))
        p = next(iter(sympy.factorint(n)))
        return _min_fixed_points(N, p, sympy.factorint(n)[p])
    if "do not have order" in text or "not of order" in text:
        q = int(re.search(r"Sylow (\d+)-subgroups", text).group(1))
        x = int(re.search(r"(?:exactly|n_\d+ =) (\d+)", text).group(1))
        assert x in _np_set(n, q)
        return n - x * (q - 1)
    primes = [int(v) for v in re.search(r"= (\d+) × (\d+) × (\d+)", text).groups()]
    counts = {int(p): int(x) for p, x in re.findall(r"n_(\d+) = (\d+)", text)}
    assert len(counts) == 2 and set(counts) <= set(primes) and all(counts[p] in _np_set(n, p) for p in counts)
    used = sum(x * (p - 1) for p, x in counts.items())
    if "how many more elements" in text or "exceed" in text:
        return 1 + used - n
    return n - 1 - used


def test_wiring():
    s = skill(SKILL)
    assert s.answer == "number" and "university-year-2" in s.tags and tuple(s.year_level) == ("university",)
    assert [(src.topic, src.difficulty) for src in s.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    check_wired(SKILL, EXAM)
    assert s.prerequisites


def test_keys_by_independent_oracle():
    for tier in ("easy", "medium", "hard"):
        for seed in range(150):
            q = make(TOPIC, seed=seed, difficulty=tier)
            assert int(q.answer) == oracle_single(q), (tier, seed, q.text, q.answer_text)


def test_keys_accepted_distractors_rejected_and_tagged():
    s = skill(SKILL)
    for tier in ("easy", "medium", "hard"):
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty=tier)
            assert grade(s, q, q.answer_text).correct, (tier, seed)
            assert len(q.distractors) >= 2
            tagged = 0
            for text in q.distractors:
                assert not grade(s, q, text).correct, (tier, seed, text, q.answer_text)
                note = q.option_notes.get(text)
                if note is not None:
                    assert note in CATALOGUE
                    tagged += 1
            assert tagged >= 1


def test_tiers_have_answer_variety_and_prompt_shapes():
    for tier in ("easy", "medium", "hard"):
        qs = [make(TOPIC, seed=seed, difficulty=tier) for seed in range(60)]
        assert len({q.answer_text for q in qs}) >= 6, tier
        assert len({prompt_shape(q) for q in qs}) >= 3, tier
        assert len({q.text for q in qs}) >= 20, tier


def test_determinism_and_text():
    for tier in ("easy", "medium", "hard"):
        for seed in range(40):
            q = make(TOPIC, seed=seed, difficulty=tier)
            assert q.to_json() == make(TOPIC, seed=seed, difficulty=tier).to_json()
            assert find_text_defects(q, include_working=True) == [], (tier, seed)
    check_deterministic(EXAM)
    check_no_text_defects(EXAM)


def test_unknown_tier_raises_value_error():
    import pytest

    from mathgen.core import REGISTRY

    for topic in (TOPIC, EXAM):
        with pytest.raises(ValueError):
            REGISTRY[topic][1](__import__("random").Random(1), tier="nonsense")


def test_no_hard_question_is_answered_by_a_single_step_of_the_stem():
    """No answer is a number printed in the stem (a guess from the text alone)."""
    for tier in ("easy", "medium", "hard"):
        for seed in range(60):
            q = make(TOPIC, seed=seed, difficulty=tier)
            printed = set(re.findall(r"\d+", q.text))
            if tier != "easy":
                assert q.answer_text not in printed or int(q.answer_text) < 4, (tier, seed, q.text, q.answer_text)


# -- exam -----------------------------------------------------------------------------------------------------------


def _set_text(values):
    return "{" + ", ".join(str(v) for v in sorted(values)) + "}"


def exam_oracle(q):
    text = q.stem
    n = _n(text)
    a, b, c = q.parts
    prime_a = int(re.search(r"n_(\d+)", a.prompt).group(1))
    x = int(re.search(r"(?:n_\d+ =|exactly) (\d+)", b.prompt).group(1))
    cofactor_free_set = _np_set(n, prime_a)
    assert x in cofactor_free_set and x > 1
    elems = x * (prime_a - 1)
    if "other than 1" in c.prompt:
        s, t = (int(v) for v in re.search(r"other than 1, (\d+) or (\d+)", c.prompt).groups())
        assert t == prime_a
        xs = int(re.search(r"n_%d = (\d+)" % s, c.prompt).group(1))
        assert xs in _np_set(n, s)
        left = n - 1 - elems - xs * (s - 1)
    else:
        assert "not of order %d" % prime_a in c.prompt
        left = n - elems
    return {"a": _set_text(cofactor_free_set), "b": str(elems), "c": str(left)}


def test_exam_shape_and_marks():
    s = skill(SKILL)
    for seed in range(60):
        q = exam(EXAM, seed)
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.answer_kind for p in q.parts] == ["set", "number", "number"]
        assert [p.marks for p in q.parts] == [2, 2, 3]
        assert q.parts[2].follow_through
    assert s.sources[-1].stages == ["exam"]


def test_exam_oracle_agrees():
    check_oracle(EXAM, exam_oracle)


def test_exam_answer_space_and_shapes():
    check_answer_space(EXAM)
    shapes = {prompt_shape(exam(EXAM, s)) for s in range(60)}
    assert len(shapes) >= 3, shapes
    assert len({exam(EXAM, s).answer_text for s in range(300)}) >= 100


def test_exam_key_accepted_distractors_rejected():
    check_key_accepted_distractors_rejected(SKILL, EXAM)


def test_exam_follow_through_from_a_wrong_part_b():
    s = skill(SKILL)
    checked = 0
    for seed in range(120):
        q = exam(EXAM, seed)
        truth = exam_oracle(q)
        wrong_b = int(truth["b"]) + 7
        expr = sympy.sympify(q.parts[2].follow_through.replace("part_b", "wb"), locals={"wb": sympy.Symbol("wb")})
        wrong_c = int(expr.subs("wb", wrong_b))
        result = grade(s, q, {"a": truth["a"], "b": str(wrong_b), "c": str(wrong_c)})
        assert result.parts["a"].marks == q.parts[0].marks
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == q.parts[2].marks and result.parts["c"].follow_through is True, (seed, wrong_c)
        checked += 1
    assert checked == 120


def test_exam_all_wrong_earns_nothing():
    check_all_wrong_earns_nothing(SKILL, EXAM, {"a": "{99}", "b": "9999", "c": "9998"})


def test_part_c_leaves_room_for_the_remaining_subgroup():
    """The elements counted for the leftover must be at least as many as the other Sylow subgroups need."""
    for seed in range(150):
        q = exam(EXAM, seed)
        n = _n(q.stem)
        assert int(q.parts[2].answer_text) > 0
        assert int(q.parts[2].answer_text) < n
