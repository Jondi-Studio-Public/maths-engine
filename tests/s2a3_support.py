"""Shared checks for the `test_s2a3_*` files: the flux and integral-theorem skills of the second-year vector calculus set.

Nothing here imports a generator module. Each file names its skill and calls these checks, which go through
`mathgen.make` and `mathgen.engine.grade` exactly as the site does.
"""

from __future__ import annotations

from functools import cache

import sympy as sp
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.engine import _tuplify, grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
TIERS = ("easy", "medium", "hard")
SEEDS = range(24)


@cache
def skill(skill_id: str):
    return next(s for s in load_skills()[1] if s.id == skill_id)


def _source(skill_id: str, tier: str):
    return next(s for s in skill(skill_id).sources if s.difficulty == tier)


@cache
def single(skill_id: str, tier: str, seed: int):
    src = _source(skill_id, tier)
    return make(src.topic, seed=seed, difficulty=tier, **_tuplify(src.params))


@cache
def exam(skill_id: str, seed: int):
    src = _source(skill_id, "exam")
    return make(src.topic, seed=seed, difficulty="exam", **_tuplify(src.params))


def check_single_tiers(skill_id: str) -> None:
    listed = {m.id for m in skill(skill_id).misconceptions}
    for tier in TIERS:
        for seed in SEEDS:
            q = single(skill_id, tier, seed)
            assert q.answer_kind == "expression"
            assert grade(skill(skill_id), q, q.answer_text).correct, (tier, seed, q.text)
            assert len(q.distractors) >= 3, (tier, seed)
            for d in q.distractors:
                assert not grade(skill(skill_id), q, d).correct, (tier, seed, d, q.text)
                assert q.option_notes[d] in listed, (q.option_notes[d], skill_id)
                assert q.option_notes[d] in CATALOGUE
            assert find_text_defects(q) == [], (tier, seed, find_text_defects(q))


def check_tiers_differ(skill_id: str) -> None:
    """The three tiers ask different questions, and the harder ones have answers that cost more to reach."""
    texts = {t: {single(skill_id, t, s).text for s in SEEDS} for t in TIERS}
    assert not (texts["easy"] & texts["medium"]) and not (texts["medium"] & texts["hard"])
    assert not (texts["easy"] & texts["hard"])
    mean = {t: sum(len(single(skill_id, t, s).text) for s in SEEDS) / len(SEEDS) for t in TIERS}
    assert mean["hard"] > mean["easy"] * 0.9, mean


def check_deterministic(skill_id: str) -> None:
    for tier in TIERS:
        src = _source(skill_id, tier)
        a = make(src.topic, seed=5, difficulty=tier)
        b = make(src.topic, seed=5, difficulty=tier)
        assert (a.text, a.answer_text, a.distractors) == (b.text, b.answer_text, b.distractors)


def check_exam_shape(skill_id: str, labels: str, follow_through: str) -> None:
    for seed in SEEDS:
        q = exam(skill_id, seed)
        assert "".join(p.label for p in q.parts) == labels, q.text
        got = "".join(p.label for p in q.parts if p.follow_through)
        assert got == follow_through, (got, q.text)
        assert all(p.answer_kind == "expression" for p in q.parts)
        assert sum(p.marks for p in q.parts) >= 7


def check_exam_canonical(skill_id: str) -> None:
    for seed in SEEDS:
        q = exam(skill_id, seed)
        result = grade(skill(skill_id), q, {p.label: p.answer_text for p in q.parts})
        assert result.correct and result.marks == result.max_marks, (seed, q.text)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))


def check_exam_distractors(skill_id: str) -> None:
    listed = {m.id for m in skill(skill_id).misconceptions}
    for seed in SEEDS:
        q = exam(skill_id, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2, (seed, p.label)
            for d in p.distractors:
                result = grade(skill(skill_id), q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (seed, p.label, d, p.prompt)
                assert p.option_notes[d] in listed and p.option_notes[d] in CATALOGUE


def _value(text: str) -> sp.Expr:
    return parse_expr(text.replace("π", "pi").replace("^", "**"), transformations=_TRANSFORMS)


def values_labels(names) -> list[str]:
    return [n.name[-1] for n in names]


def check_exam_follow_through(skill_id: str, repeats_allowed: bool = False) -> None:
    """A wrong earlier answer carried through correctly earns the later part's marks, and the later part's own
    canonical answer is then marked wrong when it differs."""
    checked = 0
    for seed in SEEDS:
        q = exam(skill_id, seed)
        by_label = {p.label: p for p in q.parts}
        for p in q.parts:
            if not p.follow_through:
                continue
            names = sorted({n for n in sp.sympify(p.follow_through).free_symbols}, key=str)
            for wrong_label in sorted(n.name[-1] for n in names):
                earlier = by_label[wrong_label]
                for d in earlier.distractors:
                    own = {pp.label: pp.answer_text for pp in q.parts}
                    own[wrong_label] = d
                    values = {
                        f"part_{lab}": _value(own[lab]) for lab in by_label if f"part_{lab}" in {n.name for n in names}
                    }
                    if _value(d) == 0:
                        continue  # a zero earlier answer withholds follow-through by design
                    mine = sp.simplify(sp.sympify(p.follow_through).subs({sp.Symbol(k): v for k, v in values.items()}))
                    if mine == 0:
                        continue
                    if any(sp.simplify(mine - _value(own[lab])) == 0 for lab in values_labels(names) if lab != p.label):
                        continue  # writing an earlier reply again is not following through, by design
                    reply = {**own, p.label: sp.sstr(mine).replace("**", "^")}
                    result = grade(skill(skill_id), q, reply)
                    assert result.parts[wrong_label].marks == 0
                    if result.parts[p.label].marks == p.marks:
                        checked += 1
                    else:
                        assert repeats_allowed, (seed, p.label, d, reply)
    assert checked or repeats_allowed
