"""Shared pin for the exam chains whose follow-through used to be copyable.

A follow-through part's formula must not be identically an earlier part (a step that only restates an earlier answer
is not a step), no two parts of a chain may share a canonical answer, and a junk reply typed into every box (or one
part right and the rest 1) must earn nothing on any part whose reply is not its own canonical answer.
"""

from __future__ import annotations

import sympy

from mathgen import make
from mathgen.engine import grade
from mathgen.skills import load_skills

JUNK = ("0", "1", "2", "-1")


def _norm(text: str) -> str:
    return text.replace(" ", "")


def _restates(formula: str, labels: list[str]) -> list[str]:
    """The labels whose part the formula equals for every positive value of the earlier parts (so Abs, x*1 and
    100*x/100 all count as a restatement)."""
    names = {f"part_{lab}": sympy.Symbol(f"part_{lab}", positive=True) for lab in labels}
    try:
        expr = sympy.sympify(formula.replace("^", "**"), locals={**names, "pi": sympy.pi, "I": sympy.I})
    except (sympy.SympifyError, SyntaxError, TypeError):
        return []
    return [lab for lab in labels if sympy.simplify(expr - names[f"part_{lab}"]) == 0]


def assert_chain_is_not_copyable(topic: str, skill_id: str, seeds=range(60)) -> None:
    skill = next(s for s in load_skills()[1] if s.id == skill_id)
    for seed in seeds:
        q = make(topic, seed=seed, difficulty="exam")
        labels = [p.label for p in q.parts]
        texts = [_norm(p.answer_text) for p in q.parts]
        assert len(set(texts)) == len(texts), (seed, texts)
        for p in q.parts:
            if p.follow_through:
                assert not _restates(p.follow_through, labels), (seed, p.label, p.follow_through)
        replies = [{lab: v for lab in labels} for v in JUNK]
        for i, lab in enumerate(labels):
            replies.append({x: (q.parts[i].answer_text if x == lab else "1") for x in labels})
        for reply in replies:
            result = grade(skill, q, reply)
            for p in q.parts:
                if reply[p.label] == p.answer_text:
                    continue
                assert result.parts[p.label].marks == 0, (seed, reply, p.label)


def assert_answer_spread(
    topic: str, label: str, *, max_share: float = 0.45, seeds: int = 300, at_least: int = 2
) -> None:
    """Part `label` of the exam chain is not constant and no single answer covers more than `max_share` of the seeds."""
    from collections import Counter

    counts = Counter(
        next(p.answer_text for p in make(topic, seed=s, difficulty="exam").parts if p.label == label)
        for s in range(seeds)
    )
    assert len(counts) >= at_least, counts
    answer, n = counts.most_common(1)[0]
    assert n / seeds <= max_share, (answer, n / seeds, counts)
