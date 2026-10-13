"""Build tests for the short written-answer generators (`wa_*_build.py`).

Each generator returns a prompt plus a per-item marking dict, never a value, because the written answer is marked
against a rubric. These tests pin the shape a marker depends on, that the prompt is clean and rendered both ways,
that the tiers are separate pools of questions with real variety, and that the marking dict does not leak into the
prompt. The numbers each prompt states are re-derived in `test_written_answer_oracle.py`.
"""

from __future__ import annotations

import random
import re

import pytest

from mathgen import make
from mathgen.core import REGISTRY
from mathgen.generators import _written_answer_kit as kit
from mathgen.text_quality import find_text_defects

TOPICS = tuple(sorted(t for t in REGISTRY if t.startswith("wa_")))
TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = 60


def _balanced(tex: str) -> bool:
    depth = 0
    for i, ch in enumerate(tex):
        if i and tex[i - 1] == "\\":
            continue
        depth += (ch == "{") - (ch == "}")
        if depth < 0:
            return False
    return depth == 0


@pytest.fixture(scope="module")
def drawn():
    return {(t, d): [make(t, seed=s, difficulty=d) for s in range(N_SEEDS)] for t in TOPICS for d in TIERS}


def test_all_twenty_two_topics_are_registered():
    assert len(TOPICS) == 22


@pytest.mark.parametrize("topic", TOPICS)
def test_determinism(topic):
    for tier in TIERS:
        for seed in (0, 1, 7, 42):
            assert (
                make(topic, seed=seed, difficulty=tier).to_dict() == make(topic, seed=seed, difficulty=tier).to_dict()
            )


@pytest.mark.parametrize("topic", TOPICS)
def test_unknown_variant_is_refused(topic):
    _level, fn = REGISTRY[topic]
    with pytest.raises(ValueError):
        fn(random.Random(0), variant="impossible")


@pytest.mark.parametrize("topic", TOPICS)
def test_marking_dict_shape(drawn, topic):
    for tier in TIERS:
        for q in drawn[(topic, tier)]:
            a = q.answer
            assert isinstance(a, dict)
            assert set(a) == {"start", "target", "reference_lines", "key_steps", "traps", "checks"}
            assert a["start"] is None and a["target"] is None and a["checks"] == []
            assert len(a["reference_lines"]) >= 2
            assert all(isinstance(line, str) and line.strip() == line and line for line in a["reference_lines"])
            ids = [s["criterion_id"] for s in a["key_steps"]]
            assert set(ids) <= set(kit.CRITERIA)
            assert {"correct_idea", "use_of_context", "completeness"} <= set(ids)
            assert len(ids) >= 4
            assert all(s["description"].strip() for s in a["key_steps"])
            names = [t["name"] for t in a["traps"]]
            assert len(names) >= 2 and len(set(names)) == len(names)
            assert all("?" in t["question"] for t in a["traps"])


@pytest.mark.parametrize("topic", TOPICS)
def test_prompt_is_clean_and_rendered_both_ways(drawn, topic):
    for tier in TIERS:
        for q in drawn[(topic, tier)]:
            assert not find_text_defects(q), (q.text, find_text_defects(q))
            assert q.text.strip() == q.text and q.answer_text.strip()
            assert q.latex and _balanced(q.latex), q.latex
            assert "  " not in q.text
            assert not re.search(r"[?!]\.|(?<!\.)\.\.(?!\.)|, ,|\( ", q.text), q.text
            # an instruction, never a bare statement or a one-word recall prompt
            assert len(q.text.split()) >= 18, q.text
            assert re.search(
                r"\b(?:[Ee]xplain|[Dd]escribe|[Ee]valuate|[Ii]dentify|[Ww]hy|[Ww]hat|[Ww]hich|[Dd]ecide|[Ii]nterpret|"
                r"[Rr]ewrite|[Rr]ecommend|[Gg]ive|[Ss]uggest|[Ss]ay|[Ii]s |[Ww]ould|[Uu]se|[Ss]tate|[Ff]ind|[Nn]ame)\b",
                q.text,
            ), q.text


@pytest.mark.parametrize("topic", TOPICS)
def test_no_figure_is_called_for(drawn, topic):
    """The prompts are text only: nothing refers to a diagram, picture or graph that is not on the page."""
    for tier in TIERS:
        for q in drawn[(topic, tier)]:
            assert not re.search(r"\b(?:diagram|figure below|shown below|picture|graph shown|image)\b", q.text), q.text


@pytest.mark.parametrize("topic", TOPICS)
def test_marking_is_not_printed_in_the_prompt(drawn, topic):
    for tier in TIERS:
        for q in drawn[(topic, tier)]:
            for line in q.answer["reference_lines"]:
                if len(line) >= 45:
                    assert line not in q.text, (q.text, line)
            for step in q.answer["key_steps"]:
                assert step["description"] not in q.text
            for trap in q.answer["traps"]:
                assert trap["question"] not in q.text and trap["name"] not in q.text
            assert q.answer_text not in q.text or len(q.answer_text) < 40
            assert "reference" not in q.text.lower() and "key point" not in q.text.lower()


@pytest.mark.parametrize("topic", TOPICS)
def test_each_tier_is_its_own_pool_and_has_variety(drawn, topic):
    pools = {t: {q.text for q in drawn[(topic, t)]} for t in TIERS}
    for tier, pool in pools.items():
        assert len(pool) >= 20, (topic, tier, len(pool))
    for i, a in enumerate(TIERS):
        for b in TIERS[i + 1 :]:
            assert pools[a] != pools[b]
            assert len(pools[a] & pools[b]) <= len(pools[a]) // 4


@pytest.mark.parametrize("topic", TOPICS)
def test_tiers_escalate(drawn, topic):
    """The harder tiers ask for more: the exam tier asks about at least two claims, cases or tasks."""
    for q in drawn[(topic, "exam")]:
        assert len(q.answer["key_steps"]) >= 4
    easy = sum(len(q.text) for q in drawn[(topic, "easy")]) / N_SEEDS
    exam = sum(len(q.text) for q in drawn[(topic, "exam")]) / N_SEEDS
    assert exam > easy, (topic, easy, exam)


@pytest.mark.parametrize("topic", TOPICS)
def test_marking_text_has_no_unfilled_template(drawn, topic):
    for tier in TIERS:
        for q in drawn[(topic, tier)]:
            blob = " ".join(
                [q.text, q.answer_text, *q.answer["reference_lines"]]
                + [s["description"] for s in q.answer["key_steps"]]
                + [t["question"] for t in q.answer["traps"]]
            )
            assert not re.search(r"None|\bFalse\b|\bTrue\b|\{[a-z_0-9]+\}|\+ -|- -|\(\)|nan\b|inf\b", blob), blob


def test_marking_helper_refuses_a_weak_item():
    ok = [("correct_idea", "a"), ("use_of_context", "b"), ("completeness", "c"), ("communication", "d")]
    wrong = [("a", "q?"), ("b", "q?")]
    kit.marking(reference=["x", "y"], points=ok, wrong=wrong)
    with pytest.raises(Exception):
        kit.marking(reference=["only one line"], points=ok, wrong=wrong)
    with pytest.raises(Exception):
        kit.marking(reference=["x", "y"], points=ok[:3], wrong=wrong)
    with pytest.raises(Exception):
        kit.marking(reference=["x", "y"], points=ok, wrong=wrong[:1])
    with pytest.raises(Exception):
        kit.marking(reference=["x", "y"], points=ok, wrong=[("a", "q?"), ("a", "q?")])
    with pytest.raises(Exception):
        kit.marking(reference=["x", "y"], points=ok, wrong=[("a", "q?"), ("b", "no question")])
    with pytest.raises(Exception):
        kit.marking(reference=["x", "y"], points=[("elegance", "z")] + ok, wrong=wrong)


def test_tidy_removes_machine_spellings():
    assert kit.tidy("a  b ,c .") == "a b,c."
    assert kit.tidy("Is it? .") == "Is it?"
    assert kit.pct(175, 400) == "43.8%"
    assert kit.pct(1, 4) == "25%"
    assert kit.join_and(["a", "b", "c"]) == "a, b and c"
