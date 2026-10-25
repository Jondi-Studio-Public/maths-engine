"""Tests for the multi-part exam source `exd_x_pc1_contradiction_chain` (senior-use-proof-by-contradiction).

The oracle re-derives every key from the rendered stem and part prompts alone, with plain integer arithmetic, and
never imports the generator module.
"""

from __future__ import annotations

import collections
import math
from functools import reduce
import re

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_pc1_contradiction_chain"
SKILL_ID = "senior-use-proof-by-contradiction"
SEEDS = range(300)

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _is_prime(n):
    return n > 1 and all(n % d for d in range(2, math.isqrt(n) + 1))


def family(q):
    if "primes" in q.stem:
        return "euclid"
    if re.search(r"\d+m \+ \d+n = \d+", q.stem):
        return "linear"
    return "residue"


def oracle(q) -> dict[str, str]:
    stem = q.stem
    kind = family(q)
    if kind == "euclid":
        listed = [int(x) for x in re.search(r"(\d+(?:, \d+)+)", stem).group(1).split(", ")]
        assert all(_is_prime(p) for p in listed)
        m = math.prod(listed) + 1
        low = next(d for d in range(2, m + 1) if m % d == 0)
        return {"a": str(m), "b": str(low), "c": str(m // low)}
    if kind == "linear":
        a_coef, b_coef, const = (int(x) for x in re.search(r"(\d+)m \+ (\d+)n = (\d+)", stem).groups())
        # Brute force: the values A m + B n take, their common divisor, and which numbers near C are values at all.
        values = {a_coef * m + b_coef * n for m in range(-300, 301) for n in range(-300, 301)}
        d = reduce(math.gcd, (v for v in values if v > 0))
        count = sum(1 for t in range(1, const + 1) if t in values)
        above = min(v for v in values if v > const)
        return {"a": str(d), "b": str(const % d), "c": str(count), "d": str(above)}
    c = int(re.search(r"n² \+ (\d+)", stem).group(1))
    mod = int(re.search(r"divided by (\d+)", q.parts[0].prompt).group(1))
    squares = {(n * n) % mod for n in range(2 * mod)}
    needed = next(r for r in range(mod) if (r + c) % mod == 0)
    ks = [k for k in range(1, mod + 1) if any((n * n + k) % mod == 0 for n in range(mod))]
    return {"a": str(needed), "b": str(len(squares)), "c": str(mod - len(squares)), "d": str(ks[0] + ks[1])}


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_source_is_wired_into_the_skill():
    last = _SKILL.sources[-1]
    assert last.topic == TOPIC and last.difficulty == "exam" and list(last.stages) == ["exam"]
    assert last.params == {"tier": "exam"}
    assert [s.difficulty for s in _SKILL.sources] == ["easy", "medium", "hard", "exam"]


def test_unknown_tier_is_rejected():
    with pytest.raises(ValueError):
        make(TOPIC, seed=1, tier="easy")


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] in (list("abc"), list("abcd"))
        assert q.parts[0].follow_through is None and any(p.follow_through for p in q.parts[1:])
        assert all(p.answer_kind == "number" for p in q.parts)


def test_oracle_agrees_300_seeds(questions):
    bad = [(s, q.text) for s, q in zip(SEEDS, questions) if oracle(q) != _canonical(q)]
    assert bad == []


def test_all_three_families_occur(questions):
    assert {family(q) for q in questions} == {"euclid", "linear", "residue"}


def test_answer_space_and_prompt_shapes(questions):
    answers = collections.Counter(q.answer_text for q in questions)
    assert len(answers) >= 100 and max(answers.values()) <= 0.1 * len(questions)
    assert len({prompt_shape(q) for q in questions[:60]}) >= 3


def test_no_part_answer_is_guessable_by_a_constant(questions):
    for label in "abcd":
        counts = collections.Counter(
            q.parts[ord(label) - 97].answer_text for q in questions if len(q.parts) > ord(label) - 97
        )
        assert max(counts.values()) / sum(counts.values()) <= 0.35, (label, counts.most_common(2))


def test_key_accepted_and_distractors_rejected(questions):
    for q in questions[:120]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) == 3 and len(set(p.distractors)) == 3
            for d in p.distractors:
                assert d != p.answer_text
                assert grade(_SKILL, q, {**canon, p.label: d}).parts[p.label].marks == 0, (q.text, p.label, d)
                assert p.option_notes.get(d) is None or p.option_notes[d] in CATALOGUE


def test_distractors_are_tagged_where_an_error_exists(questions):
    tagged = [d for q in questions[:100] for p in q.parts for d in p.distractors if d in p.option_notes]
    assert len(tagged) >= 300


def test_follow_through_from_a_wrong_earlier_part(questions):
    checked = 0
    for q in questions:
        kind = family(q)
        canon = _canonical(q)
        if kind == "euclid":
            m = int(canon["a"])
            wrong_b = next(p for p in range(2, m) if m % p == 0 and str(p) != canon["b"])
            reply = {"b": str(wrong_b), "c": str(m // wrong_b)}
            target = "c"
        elif kind == "residue":
            s = int(canon["b"]) + 1
            mod = int(canon["b"]) + int(canon["c"])
            reply = {"b": str(s), "c": str(mod - s)}
            target = "c"
        else:
            const = int(re.search(r"= (\d+)", q.stem).group(1))
            d = int(canon["a"]) * 2
            if const % d == 0:
                continue
            reply = {"a": str(d), "c": str(const // d), "d": str((const // d + 1) * d)}
            target = "c"
        replies = {**canon, **reply}
        earlier = {replies[p.label] for p in q.parts if p.label < target}
        # a value that is a listed wrong option, or repeats an earlier reply, is not marked as following through
        if reply[target] == canon[target] or reply[target] in q.parts[ord(target) - 97].distractors:
            continue
        if reply[target] in earlier:
            continue
        result = grade(_SKILL, q, replies)
        assert result.parts[target].marks == result.parts[target].max_marks, (q.text, reply)
        assert result.parts[target].follow_through is True
        checked += 1
    assert checked >= 150


def test_all_wrong_earns_zero(questions):
    for q in questions[:60]:
        result = grade(_SKILL, q, {p.label: "999999" for p in q.parts})
        assert result.marks == 0 and not result.correct


def test_text_is_clean_and_deterministic(questions):
    for q in questions[:100]:
        assert find_text_defects(q) == [], q.text
    assert _exam(7).to_json() == _exam(7).to_json()


def test_no_wrong_option_is_impossible_for_what_the_part_asks(questions):
    for q in questions:
        kind = family(q)
        if kind == "linear":
            d = int(q.parts[0].answer_text)
            assert all(int(x) < d for x in q.parts[1].distractors), (q.text, q.parts[1].distractors)
        if kind == "residue":
            m = int(q.parts[0].prompt.split("divided by ")[1].rstrip("?"))
            for p in (q.parts[0], q.parts[1], q.parts[2]):
                assert all(int(x) < m for x in p.distractors), (q.text, p.label, p.distractors)
        if kind == "euclid":
            listed = [int(x) for x in re.search(r"(\d+(?:, \d+)+)", q.stem).group(1).split(", ")]
            assert all(int(x) > 1 for x in q.parts[2].distractors)
            assert all(str(p) != q.parts[1].answer_text for p in listed)
            # part (b) says the answer is a prime outside the list, so no option may be a listed prime or a non-prime
            for x in q.parts[1].distractors:
                assert _is_prime(int(x)) and int(x) not in listed, (q.text, q.parts[1].distractors)
