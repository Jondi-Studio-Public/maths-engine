"""Shared assertions for the `r4e_*` test files (not a test module itself): tiered single-answer questions.

Multi-part exam chains reuse `exd_x_r2b_support`, which this module re-exports as `sup`.
"""

from __future__ import annotations

from collections import Counter

from fractions import Fraction
import exd_x_r2b_support as sup
from mathgen import make
from mathgen.engine import grade
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

__all__ = ["sup", "source", "question", "assert_tier", "assert_tiers_differ", "assert_answer_space"]


def source(skill_id: str, tier: str) -> dict:
    (src,) = [s for s in sup.skill_yaml(skill_id)["sources"] if s["difficulty"] == tier]
    return src


def question(skill_id: str, tier: str, seed: int):
    src = source(skill_id, tier)
    params = {k: (tuple(v) if isinstance(v, list) else v) for k, v in (src.get("params") or {}).items()}
    return make(src["topic"], seed=seed, difficulty=tier, **params)


def assert_tier(skill_id: str, tier: str, seeds=range(80), oracle=None) -> None:
    """Across seeds: clean text, deterministic, the key grades correct, every distractor grades wrong and is tagged
    with a catalogue id, and `oracle(q)` (when given) re-derives the stated answer from the rendered text."""
    sk = sup.skill_object(skill_id)
    for seed in seeds:
        q = question(skill_id, tier, seed)
        assert find_text_defects(q) == [], (tier, seed, q.text)
        assert q.to_json() == question(skill_id, tier, seed).to_json(), (tier, seed)
        assert grade(sk, q, q.answer_text).correct, (tier, seed, q.text, q.answer_text)
        assert len(q.distractors) >= 2, (tier, seed)
        for d in q.distractors:
            assert not grade(sk, q, d).correct, (tier, seed, q.text, d, q.answer_text)
            assert q.option_notes.get(d) in CATALOGUE, (tier, seed, d)
        if oracle is not None:
            want = oracle(q)
            try:
                same = Fraction(str(want)) == Fraction(q.answer_text)
            except (ValueError, ZeroDivisionError):
                same = want == q.answer_text
            assert same, (tier, seed, q.text, want, q.answer_text)


def assert_tiers_differ(skill_id: str, tiers=("easy", "medium", "hard"), seeds=range(60)) -> None:
    shapes = {t: {prompt_shape(question(skill_id, t, s)) for s in seeds} for t in tiers}
    for earlier, later in zip(tiers, tiers[1:]):
        assert shapes[later] - shapes[earlier], f"{later} adds no prompt shape beyond {earlier}"


def assert_answer_space(skill_id: str, tier: str, seeds=range(200), distinct=40, top=0.08) -> None:
    counts = Counter(question(skill_id, tier, s).answer_text for s in seeds)
    assert len(counts) >= distinct, (tier, len(counts))
    value, n = counts.most_common(1)[0]
    assert n / len(seeds) <= top, (tier, value, n)
