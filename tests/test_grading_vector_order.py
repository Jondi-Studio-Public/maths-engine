"""vector-addition and vector-cross-product grade the components in order.

Both skills used `answer: list`, which compares the values as an unordered collection, so the
reversed vector (3, 2, 1) was accepted for (1, 2, 3). They now use `ordered_list`.
"""

import itertools

import pytest

from mathgen import make
from mathgen.core import PRESETS
from mathgen.grading.typed import GRADERS
from mathgen.skills import load_skills

SKILL_IDS = ("vector-addition", "vector-cross-product")
SEEDS = range(25)


def _skills():
    _domains, skills = load_skills()
    by_id = {s.id: s for s in skills}
    return [by_id[i] for i in SKILL_IDS]


def _questions(skill):
    for src in skill.sources:
        params = {k: (tuple(v) if isinstance(v, list) else v) for k, v in (src.params or {}).items()}
        tiers = (
            [src.difficulty] if src.difficulty else [d for d in ("easy", "medium", "hard") if d in PRESETS[src.topic]]
        )
        for difficulty in tiers:
            for seed in SEEDS:
                yield make(src.topic, seed=seed, difficulty=difficulty, **params)


@pytest.mark.parametrize("skill", _skills(), ids=SKILL_IDS)
def test_skill_uses_an_order_sensitive_answer_kind(skill):
    assert skill.answer == "ordered_list"


@pytest.mark.parametrize("skill", _skills(), ids=SKILL_IDS)
def test_canonical_and_bracket_spellings_pass(skill):
    grade = GRADERS[skill.answer]
    for q in _questions(skill):
        parts = [str(v) for v in q.answer]
        for given in (
            q.answer_text,
            f"({', '.join(parts)})",
            ", ".join(parts),
            " ".join(parts),
            f"[{', '.join(parts)}]",
        ):
            assert grade(q, given).correct, (q.text, given)


@pytest.mark.parametrize("skill", _skills(), ids=SKILL_IDS)
def test_every_reordering_of_the_components_fails(skill):
    grade = GRADERS[skill.answer]
    checked = 0
    for q in _questions(skill):
        for perm in set(itertools.permutations(q.answer)):
            if list(perm) == list(q.answer):
                continue
            given = f"({', '.join(str(v) for v in perm)})"
            assert not grade(q, given).correct, (q.text, q.answer_text, given)
            checked += 1
    assert checked > 0


def test_the_reversed_vector_is_the_motivating_case():
    skill = _skills()[0]
    q = next(q for q in _questions(skill) if len(set(q.answer)) == 3)
    reversed_text = f"({', '.join(str(v) for v in reversed(q.answer))})"
    assert not GRADERS["ordered_list"](q, reversed_text).correct
    assert GRADERS["list"](q, reversed_text).correct  # what the old kind wrongly accepted
