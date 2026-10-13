"""Shared checks for the `test_exd_x_pr6_*` files: the primary-school exam chains.

Nothing here imports a generator module. Each test file supplies an oracle that re-derives every part's answer
from the whole numbers printed in the question, and the checks below run it against the rendered text.
"""

from __future__ import annotations

import re
from collections import Counter

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

_DIGITS = re.compile(r"\d+")


def skill(skill_id: str):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def numbers(q) -> list[int]:
    """Every whole number printed in the question, in reading order."""
    return [int(n) for n in _DIGITS.findall(q.text)]


def exam(topic: str, seed: int):
    return make(topic, seed=seed, difficulty="exam")


def canonical(q) -> dict[str, str]:
    return {p.label: p.answer_text for p in q.parts}


def check_wired(topic: str, skill_id: str) -> None:
    sources = [s for s in skill(skill_id).sources if s.difficulty == "exam"]
    assert [s.topic for s in sources] == [topic]
    assert sources[0].stages == ["exam"]


def check_variety(topic: str, skill_id: str, n: int = 60) -> None:
    hard = next(s for s in skill(skill_id).sources if s.difficulty == "hard")
    params = {k: (tuple(v) if isinstance(v, list) else v) for k, v in (hard.params or {}).items()}
    hard_shapes = {prompt_shape(make(hard.topic, seed=s, difficulty="hard", **params)) for s in range(n)}
    exam_shapes = {prompt_shape(exam(topic, s)) for s in range(n)}
    assert len(exam_shapes) >= 3, exam_shapes
    assert exam_shapes - hard_shapes


def check_oracle(topic: str, oracle, n: int = 300) -> None:
    bad = []
    for seed in range(n):
        q = exam(topic, seed)
        if oracle(numbers(q)) != canonical(q):
            bad.append((seed, q.text, canonical(q)))
    assert bad == [], bad[:3]


def check_answer_space(topic: str, n: int = 300) -> None:
    counts = Counter(exam(topic, s).answer_text for s in range(n))
    assert len(counts) >= 100, len(counts)
    assert max(counts.values()) <= 0.05 * n, counts.most_common(2)


def check_grading(topic: str, skill_id: str, n: int = 120) -> None:
    sk = skill(skill_id)
    bad = []
    for seed in range(n):
        q = exam(topic, seed)
        full = grade(sk, q, q.answer_text)
        if not (full.correct and full.marks == full.max_marks == sum(p.marks for p in q.parts)):
            bad.append(("key rejected", seed))
        canon = canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label)
            for d in p.distractors:
                result = grade(sk, q, {**canon, p.label: d})
                if result.parts[p.label].marks != 0:
                    bad.append(("distractor accepted", seed, p.label, d))
    assert bad == [], bad[:3]


def check_tagged(topic: str, n: int = 60) -> None:
    for seed in range(n):
        q = exam(topic, seed)
        for p in q.parts:
            assert p.answer_kind == "number"
            assert set(p.distractors) == set(p.option_notes), (seed, p.label)
            assert set(p.distractors) == set(p.distractor_values), (seed, p.label)


def check_follow_through(topic: str, skill_id: str, follow, n: int = 120) -> None:
    """A wrong (a) used correctly in the later parts earns those parts' marks.

    `follow(nums, a_wrong)` returns `{label: value}` for every part that depends on (a), computed from the
    printed numbers and the wrong (a) alone. `wrong_a` is the canonical (a) plus `bump(nums)`.
    """
    sk = skill(skill_id)
    checked = 0
    for seed in range(n):
        q = exam(topic, seed)
        nums = numbers(q)
        canon = canonical(q)
        a_wrong, dependent = follow(nums, int(canon["a"]))
        assert a_wrong != int(canon["a"])
        if a_wrong in dependent.values():
            continue  # a reply that only repeats the earlier value earns no follow-through, by design
        checked += 1
        reply = {**canon, "a": str(a_wrong), **{k: str(v) for k, v in dependent.items()}}
        result = grade(sk, q, reply)
        assert result.parts["a"].marks == 0, (seed, q.text)
        for label in dependent:
            part = result.parts[label]
            assert part.marks == part.max_marks, (seed, label, reply)
            assert part.follow_through is True
        later = [p.label for p in q.parts if p.follow_through]
        assert sorted(dependent) == sorted(x for x in later if x in dependent)
        assert result.marks < result.max_marks and not result.correct
        # the canonical later parts still earn their marks beside the wrong (a)
        keep = grade(sk, q, {**canon, "a": str(a_wrong)})
        assert keep.parts["a"].marks == 0
        assert all(keep.parts[p.label].marks == p.marks for p in q.parts if p.label != "a")
    assert checked >= 0.8 * n, checked
    assert grade(sk, exam(topic, 0), {"a": "99999", "b": "1", "c": "2", "d": "3"}).marks == 0


def check_deterministic_and_clean(topic: str, n: int = 40) -> None:
    from mathgen.text_quality import find_text_defects

    for seed in range(n):
        q = exam(topic, seed)
        assert find_text_defects(q) == [], (seed, q.text)
        assert q.to_json() == exam(topic, seed).to_json()
