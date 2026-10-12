"""Build tests for the free-form proof generators (`proof_*_build.py`).

Each generator returns a prompt plus a per-item marking dict, never a value, because the written proof is
marked against a rubric. These tests pin the shape that a marker depends on, that the prompt is clean and
rendered both ways, that the tiers are separate pools of questions, and that the marking dict does not leak into
the prompt. The statements themselves are re-derived in `test_free_proof_oracle.py`.
"""

from __future__ import annotations

import random
import re

import pytest

from mathgen import make
from mathgen.core import REGISTRY
from mathgen.generators import _free_proof_kit as kit
from mathgen.text_quality import find_text_defects

TOPICS = (
    "proof_f10_deductive_shapes",
    "proof_derivative_exp",
    "proof_derivative_ln",
    "proof_integral_exp",
    "proof_integral_recip_pos",
    "proof_integral_recip_neg",
    "proof_irrationality",
    "proof_simple_number_results",
    "proof_pigeonhole",
    "proof_independence",
    "proof_circle_theorems",
)
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
            assert {"correctness", "logical_completeness"} <= set(ids)
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
            assert any(q.text.endswith(tail) for tail in kit.TAILS), q.text
            assert "  " not in q.text
            # the question is a question about a proof, never a bare statement
            assert re.search(r"\b(?:[Pp]rove|[Ss]how|[Ff]ind|[Ee]xplain|[Ss]olve)\b", q.text), q.text


@pytest.mark.parametrize("topic", TOPICS)
def test_reference_proof_is_not_printed_in_the_prompt(drawn, topic):
    for tier in TIERS:
        for q in drawn[(topic, tier)]:
            for line in q.answer["reference_lines"]:
                if len(line) >= 45:
                    assert line not in q.text, (q.text, line)


@pytest.mark.parametrize("topic", TOPICS)
def test_each_tier_is_its_own_pool_and_has_variety(drawn, topic):
    pools = {t: {q.text for q in drawn[(topic, t)]} for t in TIERS}
    for tier, pool in pools.items():
        assert len(pool) >= 20, (topic, tier, len(pool))
    for i, a in enumerate(TIERS):
        for b in TIERS[i + 1 :]:
            assert pools[a] != pools[b]
            # a harder tier is a different set of tasks, not the easier set relabelled
            assert len(pools[a] & pools[b]) <= len(pools[a]) // 4


@pytest.mark.parametrize("topic", TOPICS)
def test_marking_text_has_no_unfilled_template(drawn, topic):
    for tier in TIERS:
        for q in drawn[(topic, tier)]:
            blob = " ".join(
                [q.text, q.answer_text, *q.answer["reference_lines"]]
                + [s["description"] for s in q.answer["key_steps"]]
                + [t["question"] for t in q.answer["traps"]]
            )
            assert not re.search(r"None|\bFalse\b|\bTrue\b|\+ -|- -|\(\)|\(-1[a-z]|\b1[xy]\b", blob), blob


def test_marking_helper_refuses_a_weak_item():
    with pytest.raises(Exception):
        kit.marking(
            reference=["only one line"],
            steps=[("correctness", "x"), ("logical_completeness", "y")],
            traps=[("a", "q?"), ("b", "q?")],
        )
    with pytest.raises(Exception):
        kit.marking(reference=["a", "b"], steps=[("correctness", "x")], traps=[("a", "q?"), ("b", "q?")])
    with pytest.raises(Exception):
        kit.marking(
            reference=["a", "b"], steps=[("correctness", "x"), ("logical_completeness", "y")], traps=[("a", "q?")]
        )
    with pytest.raises(Exception):
        kit.marking(
            reference=["a", "b"],
            steps=[("elegance", "x"), ("correctness", "x"), ("logical_completeness", "y")],
            traps=[("a", "q?"), ("b", "q?")],
        )


def test_tidy_removes_machine_spellings():
    assert kit.tidy("(1/-2) ln|x|") == "(-1/2) ln|x|"
    assert kit.tidy("dy/dx = -1y") == "dy/dx = -y"
    assert kit.tidy("x^2 + 1x + c") == "x^2 + x + c"
    assert kit.tidy("(x - -1) + -3 = 0 and e^(-1x)") == "(x + 1) - 3 = 0 and e^(-x)"
    assert kit.tidy("y' = (1)e^(1x) + 1e^(-1x)") == "y' = e^x + e^(-x)"
    assert kit.tidy("11x and 21y and e^(1x)") == "11x and 21y and e^x"
