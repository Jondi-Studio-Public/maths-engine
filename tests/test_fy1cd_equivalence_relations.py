"""Tests `ucv1-equivalence-relations-classes` (topic `ucv1_equivalence_relations`) and its exam chain.

The oracle reads the set and the relation (or the rule) back out of the question TEXT and re-derives each key by
building the relation as a set of ordered pairs and testing it directly.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from math import gcd

import pytest

import mathgen.generators.fy1cd_equivalence_relations_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TOPIC = "ucv1_equivalence_relations"
EXAM = "exd_x_fy_equivalence_relations"
SKILL_ID = "ucv1-equivalence-relations-classes"
TIERS = ("easy", "medium", "hard")
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _range(text):
    lo, hi = re.search(r"\{(\d+), \d+, \.\.\., (\d+)\}", text).groups()
    return list(range(int(lo), int(hi) + 1))


def _classes(elements, pairs):
    """Equivalence classes of a relation given as a set of pairs, by testing reflexivity, symmetry, transitivity."""
    assert all((e, e) in pairs for e in elements)
    assert all((b, a) in pairs for a, b in pairs)
    assert all((a, d) in pairs for a, b in pairs for c, d in pairs if b == c)
    seen, out = set(), []
    for e in elements:
        if e not in seen:
            cls = {x for x in elements if (e, x) in pairs}
            seen |= cls
            out.append(cls)
    return out


def _closure(elements, pairs):
    """Smallest reflexive, symmetric and transitive relation containing `pairs`, by saturating."""
    rel = set(pairs) | {(e, e) for e in elements}
    while True:
        new = rel | {(b, a) for a, b in rel} | {(a, d) for a, b in rel for c, d in rel if b == c}
        if new == rel:
            return rel
        rel = new


# -- wiring -------------------------------------------------------------------------------------------------------


def test_skill_wiring():
    assert _SKILL.answer == "number"
    assert [(s.topic, s.difficulty) for s in _SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    assert _SKILL.sources[-1].stages == ["exam"]
    assert all(m.id in CATALOGUE for m in _SKILL.misconceptions)


# -- single-answer tiers ------------------------------------------------------------------------------------------


def test_easy_key_is_a_class_size():
    asked = Counter()
    for seed in range(150):
        q = _q("easy", seed)
        elements = _range(q.text)
        n = int(re.search(r"divisible by (\d+)", q.text).group(1))
        pairs = {(a, b) for a in elements for b in elements if (a - b) % n == 0}
        classes = _classes(elements, pairs)
        if "largest" in q.text:
            assert int(q.answer) == max(map(len, classes)), (seed, q.text)
            asked["largest"] += 1
        else:
            m = int(re.search(r"class of (\d+)", q.text).group(1))
            assert int(q.answer) == len(next(c for c in classes if m in c)), (seed, q.text)
            asked["of"] += 1
    assert min(asked.values()) >= 40, asked


def test_medium_key_is_the_number_of_pairs_to_add():
    for seed in range(150):
        q = _q("medium", seed)
        elements = ast.literal_eval(re.search(r"S = (\{[^}]*\})", q.text).group(1).replace("{", "[").replace("}", "]"))
        pairs = set(ast.literal_eval("[" + re.search(r"R = \{(.*)\}\. What", q.text).group(1) + "]"))
        closure = _closure(elements, pairs)
        assert int(q.answer) == len(closure) - len(pairs), (seed, q.text)
        # the three one-sided counts all differ from the answer, so every shortcut is detectably wrong
        assert len({int(q.answer), *map(int, q.distractors)}) == len(q.distractors) + 1


def test_hard_key_is_the_number_of_classes():
    asked = Counter()
    for seed in range(150):
        q = _q("hard", seed)
        elements = _range(q.text)
        if "a² ≡ b²" in q.text:
            n = int(re.search(r"\(mod (\d+)\)", q.text).group(1))
            pairs = {(a, b) for a in elements for b in elements if (a * a - b * b) % n == 0}
            asked["square"] += 1
        elif "gcd" in q.text:
            n = int(re.search(r"gcd\(a, (\d+)\)", q.text).group(1))
            pairs = {(a, b) for a in elements for b in elements if gcd(a, n) == gcd(b, n)}
            asked["gcd"] += 1
        else:
            m, k = map(int, re.findall(r"\(mod (\d+)\)", q.text))
            pairs = {(a, b) for a in elements for b in elements if (a - b) % m == 0 and (a - b) % k == 0}
            asked["two"] += 1
        assert int(q.answer) == len(_classes(elements, pairs)), (seed, q.text)
    assert min(asked.values()) >= 20, asked


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in range(100):
        q = _q(tier, seed)
        assert grade(_SKILL, q, q.answer_text).correct, (seed, q.answer_text)
        assert len(q.distractors) >= 2, seed
        for text in q.distractors:
            assert not grade(_SKILL, q, text).correct, (seed, text, q.answer_text)
            assert q.option_notes[text] in CATALOGUE
            assert text in q.distractor_values


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(60):
        q = _q(tier, seed)
        assert find_text_defects(q, include_working=True) == [], (seed, find_text_defects(q, include_working=True))
        assert q.to_json() == _q(tier, seed).to_json()


def test_tiers_differ_in_shape():
    easy = [_q("easy", s).text for s in range(30)]
    medium = [_q("medium", s).text for s in range(30)]
    hard = [_q("hard", s).text for s in range(30)]
    assert all("divisible by" in t for t in easy)  # a plain congruence
    assert all("(1, 1)" in t or "(2, 2)" in t or "R = {(" in t for t in medium)  # a relation listed pair by pair
    assert not any("divisible by" in t for t in hard)  # a rule that is not a plain congruence


@pytest.mark.parametrize("tier", TIERS)
def test_answer_space(tier):
    texts = [_q(tier, s).text for s in range(80)]
    assert len(set(texts)) >= (30 if tier != "medium" else 70)
    assert len({_q(tier, s).answer_text for s in range(80)}) >= 3


# -- exam chain ---------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["number"] * 4
        assert [p.marks for p in q.parts] == [2, 2, 2, 2]
        assert [bool(p.follow_through) for p in q.parts] == [False, True, True, True]
        assert find_text_defects(q, include_working=True) == []
        assert q.to_json() == _exam(seed).to_json()


def test_exam_keys_match_independent_derivation():
    for seed in range(150):
        q = _exam(seed)
        elements = _range(q.text)
        m = int(re.search(r"\(mod (\d+)\)", q.text).group(1))
        pairs = {(a, b) for a in elements for b in elements if (a - b) % m == 0}
        classes = _classes(elements, pairs)
        assert int(q.parts[0].answer) == len(classes)
        assert int(q.parts[1].answer) == max(map(len, classes))
        assert int(q.parts[2].answer) == len(pairs)
        unordered = {frozenset((a, b)) for a, b in pairs if a != b}
        assert int(q.parts[3].answer) == len(unordered)
        assert len(elements) % m != 0


def test_exam_key_accepted_and_every_distractor_rejected():
    for seed in range(80):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 8
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2
            for text in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: text})
                assert result.parts[p.label].marks == 0, (seed, p.label, text)
                assert p.option_notes[text] in CATALOGUE


def test_exam_follow_through_from_a_wrong_class_count():
    checked = 0
    for seed in range(80):
        q = _exam(seed)
        n_elements = len(_range(q.text))
        canon = {p.label: p.answer_text for p in q.parts}
        for wrong in q.parts[0].distractors:
            own_b = -(-n_elements // int(wrong))
            if own_b in (int(q.parts[1].answer), int(wrong)):  # repeating the earlier reply is not following through
                continue
            result = grade(_SKILL, q, {**canon, "a": wrong, "b": str(own_b)})
            assert result.parts["a"].marks == 0
            assert result.parts["b"].marks == 2 and result.parts["b"].follow_through is True, (seed, wrong)
            checked += 1
    assert checked >= 50


def test_exam_answer_space():
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 40
    assert max(Counter(answers).values()) <= 30
