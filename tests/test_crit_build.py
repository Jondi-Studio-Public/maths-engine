"""Build tests for the critique, audit, proof and analysis generators (`crit_*`, `ucv1_*`, `f10_*` builds).

Eleven of the twelve return a prompt plus a per-item marking dict, never a value, because the written answer is
marked against a rubric. These tests pin the shape that a marker depends on, that the prompt is clean and rendered
both ways, that the tiers are separate pools of questions, and that the marking dict does not leak into the
prompt. The twelfth, `float_error_budget`, is a typed number. The facts themselves are re-derived in
`test_crit_oracle.py`.
"""

from __future__ import annotations

import random
import re

import pytest

from mathgen import make
from mathgen.core import REGISTRY
from mathgen.generators import _crit_kit as kit
from mathgen.text_quality import find_text_defects

PROOF_TOPICS = (
    "proof_direct_contrapositive",
    "proof_prove_or_refute",
    "riemann_interchange",
    "uniform_convergence",
    "sequence_limit",
)
SHORT_TOPICS = (
    "model_assumption_audit",
    "model_limit_validation",
    "proof_exposition",
    "source_provenance",
    "data_media_critique",
    "sampling_techniques",
)
RUBRIC_TOPICS = PROOF_TOPICS + SHORT_TOPICS
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
    topics = RUBRIC_TOPICS + ("float_error_budget",)
    return {(t, d): [make(t, seed=s, difficulty=d) for s in range(N_SEEDS)] for t in topics for d in TIERS}


def _criteria(topic: str):
    return kit.PROOF_CRITERIA if topic in PROOF_TOPICS else kit.SHORT_CRITERIA


def _required(topic: str):
    return kit.PROOF_REQUIRED if topic in PROOF_TOPICS else kit.SHORT_REQUIRED


@pytest.mark.parametrize("topic", RUBRIC_TOPICS + ("float_error_budget",))
def test_determinism(topic):
    for tier in TIERS:
        for seed in (0, 1, 7, 42):
            assert (
                make(topic, seed=seed, difficulty=tier).to_dict() == make(topic, seed=seed, difficulty=tier).to_dict()
            )


@pytest.mark.parametrize("topic", RUBRIC_TOPICS + ("float_error_budget",))
def test_unknown_variant_is_refused(topic):
    _level, fn = REGISTRY[topic]
    with pytest.raises(ValueError):
        fn(random.Random(0), variant="impossible")


@pytest.mark.parametrize("topic", RUBRIC_TOPICS)
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
            assert set(ids) <= set(_criteria(topic))
            assert set(_required(topic)) <= set(ids)
            assert all(s["description"].strip() for s in a["key_steps"])
            names = [t["name"] for t in a["traps"]]
            assert len(names) >= 2 and len(set(names)) == len(names)
            assert all("?" in t["question"] for t in a["traps"])


@pytest.mark.parametrize("topic", RUBRIC_TOPICS + ("float_error_budget",))
def test_prompt_is_clean_and_rendered_both_ways(drawn, topic):
    for tier in TIERS:
        for q in drawn[(topic, tier)]:
            assert not find_text_defects(q), (q.text, find_text_defects(q))
            assert q.text.strip() == q.text and q.answer_text.strip()
            assert q.latex and _balanced(q.latex), q.latex
            assert "  " not in q.text
            last = re.split(r"(?<=[.?]) ", q.text)[-1]
            assert last.endswith((".", "?")), q.text
            if topic != "float_error_budget":
                # the closing sentence tells the learner what the answer is marked for
                assert re.search(
                    r"reason|step|justify|explain|show|proof|bound|calculation|counterexample|argument|why|check|index|point|sentence|answer|recommend|variable|ethic",
                    last,
                    re.I,
                ), q.text


@pytest.mark.parametrize("topic", RUBRIC_TOPICS)
def test_reference_answer_is_not_printed_in_the_prompt(drawn, topic):
    for tier in TIERS:
        for q in drawn[(topic, tier)]:
            for line in q.answer["reference_lines"]:
                if len(line) >= 45:
                    assert line not in q.text, (q.text, line)


@pytest.mark.parametrize("topic", RUBRIC_TOPICS + ("float_error_budget",))
def test_each_tier_is_its_own_pool_and_has_variety(drawn, topic):
    pools = {t: {q.text for q in drawn[(topic, t)]} for t in TIERS}
    for tier, pool in pools.items():
        assert len(pool) >= 20, (topic, tier, len(pool))
    for i, a in enumerate(TIERS):
        for b in TIERS[i + 1 :]:
            assert pools[a] != pools[b]
            # a harder tier is a different set of tasks, not the easier set relabelled
            assert len(pools[a] & pools[b]) <= len(pools[a]) // 4


@pytest.mark.parametrize("topic", RUBRIC_TOPICS)
def test_marking_text_has_no_unfilled_template(drawn, topic):
    for tier in TIERS:
        for q in drawn[(topic, tier)]:
            blob = " ".join(
                [q.text, q.answer_text, *q.answer["reference_lines"]]
                + [s["description"] for s in q.answer["key_steps"]]
                + [t["question"] for t in q.answer["traps"]]
            )
            assert not re.search(
                r"\bNone\b(?! (?:of|is|does|are|works|exists))|\bFalse\b|\+ -|- -|\(\)|\{[A-Za-z_]+\}", blob
            ), blob


def test_marking_helper_refuses_a_weak_item():
    ok_steps = [("correctness", "x"), ("logical_completeness", "y")]
    ok_traps = [("a", "q?"), ("b", "q?")]
    with pytest.raises(Exception):
        kit.marking(reference=["only one line"], steps=ok_steps, traps=ok_traps)
    with pytest.raises(Exception):
        kit.marking(reference=["a", "b"], steps=[("correctness", "x")], traps=ok_traps)
    with pytest.raises(Exception):
        kit.marking(reference=["a", "b"], steps=ok_steps, traps=[("a", "q?")])
    with pytest.raises(Exception):
        kit.marking(reference=["a", "b"], steps=ok_steps, traps=[("a", "q?"), ("a", "q?")])
    with pytest.raises(Exception):
        kit.marking(reference=["a", "b"], steps=ok_steps, traps=[("a", "no question mark"), ("b", "q?")])
    with pytest.raises(Exception):
        kit.marking(reference=["a", "b"], steps=ok_steps + [("not_a_criterion", "z")], traps=ok_traps)
    with pytest.raises(Exception):
        # a short-answer item must cover its own required criteria, not the proof ones
        kit.short_marking(reference=["a", "b"], steps=ok_steps, traps=ok_traps)
    good = kit.short_marking(reference=["a", "b"], steps=[("correct_idea", "x"), ("completeness", "y")], traps=ok_traps)
    assert good["checks"] == []


def test_float_error_budget_is_a_typed_number(drawn):
    for tier in TIERS:
        for q in drawn[("float_error_budget", tier)]:
            assert isinstance(q.answer, (int, float)) and not isinstance(q.answer, bool)
            assert float(q.answer_text) == q.answer
            assert q.answer != 0
            assert len(q.distractors) >= 2 and len(set(q.distractors)) == len(q.distractors)
            # a wrong value reads like the answer: the same number of decimal places
            places = len(q.answer_text.split(".")[1]) if "." in q.answer_text else 0
            assert all((len(d.split(".")[1]) if "." in d else 0) == places for d in q.distractors), q.distractors
            assert q.answer_text not in q.distractors
            # the answer is stated to the precision it has: a terminating decimal (division by 2, 4 or 8)
            assert re.fullmatch(r"-?\d+(?:\.\d{1,3})?", q.answer_text), q.answer_text


def test_float_error_budget_tiers_escalate(drawn):
    # more operations at each step up; the exam tier adds a different task shape (summation order, Horner, long trace)
    def n_ops(q):
        return len(re.findall(r", then ", q.text)) + 1

    means = {t: sum(n_ops(q) for q in drawn[("float_error_budget", t)]) / N_SEEDS for t in TIERS}
    assert means["easy"] < means["medium"] <= means["hard"] + 1.5
    assert any("Horner" in q.text for q in drawn[("float_error_budget", "exam")])
    assert any("added in two ways" in q.text for q in drawn[("float_error_budget", "exam")])
