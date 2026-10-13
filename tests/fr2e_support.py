"""Shared assertions for the `fr2e_*` multi-step modelling and algorithm skill tests (not a test module itself)."""

from __future__ import annotations

from pathlib import Path

import yaml

from mathgen import make
from mathgen.core import mcq_options
from mathgen.engine import grade
from mathgen.misconceptions import CATALOGUE
from mathgen.skills import load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

ROOT = Path(__file__).resolve().parent.parent
TIERS = ("easy", "medium", "hard")
SEEDS = range(60)


def skill_yaml(skill_id: str) -> dict:
    (path,) = list((ROOT / "skills").rglob(f"{skill_id}.yaml"))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def skill_object(skill_id: str):
    _, skills = load_skills()
    return next(s for s in skills if s.id == skill_id)


def single(topic: str, tier: str, seed: int):
    return make(topic, seed=seed, difficulty=tier)


def exam(topic: str, seed: int):
    return make(topic, seed=seed, difficulty="exam")


def reply(sk, q, text: str) -> str:
    """What a learner sends for option `text`: the option's 0-based index for an mcq skill (digit-only text is an index)."""
    return str(mcq_options(q).index(text)) if sk.answer == "mcq" else text


def assert_registered(skill_id: str, topic: str, exam_topic: str) -> None:
    sources = skill_yaml(skill_id)["sources"]
    assert [(s["topic"], s["difficulty"]) for s in sources] == [
        (topic, "easy"),
        (topic, "medium"),
        (topic, "hard"),
        (exam_topic, "exam"),
    ]
    assert sources[3]["stages"] == ["exam"]


def assert_clean_and_deterministic(topic: str, tiers=TIERS) -> None:
    for tier in tiers:
        for seed in SEEDS:
            q = single(topic, tier, seed) if tier != "exam" else exam(topic, seed)
            assert find_text_defects(q) == [], (tier, seed, q.text)
            again = single(topic, tier, seed) if tier != "exam" else exam(topic, seed)
            assert q.to_json() == again.to_json()


def assert_single_grading(skill_id: str, topic: str, tiers=TIERS, seeds=SEEDS) -> None:
    """The key earns the mark, every distractor and a junk reply earn nothing, and every distractor is tagged."""
    sk = skill_object(skill_id)
    for tier in tiers:
        for seed in seeds:
            q = single(topic, tier, seed)
            assert not q.parts
            assert grade(sk, q, reply(sk, q, q.answer_text)).correct, (tier, seed, q.answer_text)
            assert len(q.distractors) >= 3, (tier, seed)
            for d in q.distractors:
                assert not grade(sk, q, reply(sk, q, d)).correct, (tier, seed, d, q.answer_text)
                assert q.option_notes.get(d) in CATALOGUE, (tier, seed, d)
            if sk.answer == "mcq":
                options = mcq_options(q)
                assert options.count(q.answer_text) == 1 and len(set(options)) == len(options)
                assert len(options) >= 4, (tier, seed)


def assert_tiers_differ(topic: str, tiers=TIERS) -> None:
    pools = {t: {q.text for q in (single(topic, t, s) for s in SEEDS)} for t in tiers}
    for a in tiers:
        for b in tiers:
            if a < b:
                assert not (pools[a] & pools[b]), (a, b)
        assert len(pools[a]) >= 20, (a, len(pools[a]))


def assert_chain_shape(topic: str, kinds, marks, follow_through) -> None:
    for seed in range(40):
        q = exam(topic, seed)
        assert [p.answer_kind for p in q.parts] == list(kinds)
        assert [p.marks for p in q.parts] == list(marks)
        assert [bool(p.follow_through) for p in q.parts] == list(follow_through)


def assert_exam_grading(skill_id: str, topic: str, seeds=range(80)) -> None:
    """The composed key earns every mark, every distractor of every part earns nothing for its part, junk earns nothing,
    and every distractor is tagged."""
    sk = skill_object(skill_id)
    for seed in seeds:
        q = exam(topic, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        full = grade(sk, q, canon)
        assert full.correct and full.marks == full.max_marks, (seed, q.answer_text)
        for p in q.parts:
            assert len(p.distractors) >= 3, (seed, p.label)
            for d in p.distractors:
                assert p.option_notes.get(d) in CATALOGUE, (seed, p.label, d)
                result = grade(sk, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (seed, p.label, d, p.answer_text)
        junk = grade(sk, q, {p.label: f"{7919 * (i + 1)}.5" for i, p in enumerate(q.parts)})
        assert junk.marks == 0, (seed, junk.marks)


def assert_follow_through(skill_id: str, topic: str, label: str, wrong_label: str, compute, seeds=range(60), need=20):
    """Reply with a wrong answer to part `wrong_label` and the value `compute(q, wrong_text)` to part `label`: the
    later part earns its marks as follow-through when `compute` is not the canonical answer."""
    sk = skill_object(skill_id)
    checked = 0
    for seed in seeds:
        q = exam(topic, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        part = next(p for p in q.parts if p.label == label)
        wrong_part = next(p for p in q.parts if p.label == wrong_label)
        for d in wrong_part.distractors:
            value = compute(q, canon, d)
            if value is None or value == canon[label]:
                continue
            result = grade(sk, q, {**canon, wrong_label: d, label: value})
            if result.parts[label].marks == part.marks:
                assert result.parts[label].follow_through, (seed, d, value)
                checked += 1
    assert checked >= need, checked
