"""Build tests for the nine `cp1_*_proof` topics (open-response proofs written as a chain of equal expressions).

Generator-level only: determinism, the shape of the marking dict, prompt cleanliness, variety, tier escalation,
the LaTeX shape and the in-generator check. The independent truth check is `test_cp1_chain_proofs_oracle.py`.
"""

from __future__ import annotations

import random
import re
from collections import Counter

import pytest

import mathgen
import mathgen.core
from mathgen.generators import (
    _chain_proof_kit as kit,
    cp1_arithmetic_sum_build,
    cp1_cartesian_path_build,
    cp1_de_moivre_build,
    cp1_divisibility_build,
    cp1_exponential_laws_build,
    cp1_factor_remainder_build,
    cp1_geometric_sum_build,
    cp1_pascal_identities_build,
    cp1_sum_induction_build,
)

MODULES = {
    "cp1_sum_induction_proof": cp1_sum_induction_build,
    "cp1_divisibility_proof": cp1_divisibility_build,
    "cp1_pascal_identity_proof": cp1_pascal_identities_build,
    "cp1_cartesian_path_proof": cp1_cartesian_path_build,
    "cp1_exponential_laws_proof": cp1_exponential_laws_build,
    "cp1_arithmetic_sum_proof": cp1_arithmetic_sum_build,
    "cp1_geometric_sum_proof": cp1_geometric_sum_build,
    "cp1_factor_remainder_proof": cp1_factor_remainder_build,
    "cp1_de_moivre_proof": cp1_de_moivre_build,
}
TOPICS = tuple(MODULES)
TIERS = ("easy", "medium", "hard", "exam")
CRITERIA = {"correctness", "logical_completeness", "definitions_assumptions", "communication"}
EXPECTED = {"authoritative_failure", "non_authoritative_failure", "passes_chain"}


@pytest.fixture(scope="module")
def questions():
    return {(t, d): [mathgen.make(t, seed=s, difficulty=d) for s in range(200)] for t in TOPICS for d in TIERS}


@pytest.mark.parametrize("topic", TOPICS)
def test_determinism(topic):
    for tier in TIERS:
        for seed in (0, 1, 5, 17, 42):
            assert (
                mathgen.make(topic, seed=seed, difficulty=tier).to_dict()
                == mathgen.make(topic, seed=seed, difficulty=tier).to_dict()
            )


@pytest.mark.parametrize("topic", TOPICS)
def test_marking_dict_shape_and_prompt_cleanliness(questions, topic):
    for tier in TIERS:
        for q in questions[(topic, tier)]:
            a = q.answer
            assert a["start"] and a["target"] and isinstance(a["symbols"], dict)
            assert len(a["checks"]) == 1
            check = a["checks"][0]
            assert check["type"] == "equivalence_chain" and check["authoritative"] is True
            assert (check["start"], check["target"], check["symbols"]) == (a["start"], a["target"], a["symbols"])
            assert a["reference_lines"][0] == a["start"] and a["reference_lines"][-1] == a["target"]
            ids = {s["criterion_id"] for s in a["key_steps"]}
            assert ids <= CRITERIA and {"correctness", "logical_completeness", "communication"} <= ids
            for step in a["key_steps"]:
                assert step["description"].strip()
            assert "*" not in q.text and "*" not in q.answer_text
            assert not re.search(r"\b[0-9]+[a-z]\^?\(?1[a-z]", q.text)
            assert q.text.strip().endswith(".")
            for trap in a["traps"]:
                assert trap["expected_check"] in EXPECTED and trap["example_lines"]
                assert trap["name"] and trap["question"]
            names = [t["name"] for t in a["traps"]]
            assert len(names) == len(set(names))
            assert any(t["expected_check"] == "non_authoritative_failure" for t in a["traps"])
            assert any(t["expected_check"] == "passes_chain" for t in a["traps"])
            if a["symbols"]:
                assert any(t["expected_check"] == "authoritative_failure" for t in a["traps"])


@pytest.mark.parametrize("topic", TOPICS)
def test_every_distinct_reference_chain_holds_numerically(questions, topic):
    """Independent of the generator's own check: evaluate each reference line with sympy at fixed points."""
    import sympy as sp

    for tier in TIERS:
        seen = {}
        for q in questions[(topic, tier)]:
            seen.setdefault(q.answer_text, q)
        assert len(seen) >= 6, (topic, tier, len(seen))
        for q in seen.values():
            a = q.answer
            names = sorted(a["symbols"])
            for p in range(3):
                env = {sp.Symbol(n): sp.Rational(3 + 2 * p + i, 2) for i, n in enumerate(names)}
                if any("!" in line or "factorial" in line for line in a["reference_lines"]):
                    env = {sp.Symbol(n): 3 + p + i for i, n in enumerate(names)}
                values = []
                for line in a["reference_lines"]:
                    try:
                        values.append(complex(sp.N(sp.sympify(line).subs(env), 25)))
                    except (TypeError, ZeroDivisionError):
                        values.append(None)
                clean = [v for v in values if v is not None]
                for v in clean:
                    assert abs(v - clean[0]) <= 1e-8 * max(1.0, abs(clean[0])), (topic, tier, a["reference_lines"])


@pytest.mark.parametrize("topic", TOPICS)
def test_at_least_20_distinct_prompts_per_tier(questions, topic):
    for tier in TIERS:
        assert len({q.text for q in questions[(topic, tier)][:60]}) >= 20, (topic, tier)


@pytest.mark.parametrize("topic", TOPICS)
def test_no_answer_dominates(questions, topic):
    for tier in TIERS:
        counts = Counter(q.answer_text for q in questions[(topic, tier)])
        assert counts.most_common(1)[0][1] / 200 <= 0.25, (topic, tier, counts.most_common(2))


@pytest.mark.parametrize("topic", TOPICS)
def test_tier_prompt_shapes_escalate(questions, topic):
    from mathgen.text_quality import prompt_shape

    shapes = {tier: {prompt_shape(q) for q in questions[(topic, tier)][:60]} for tier in TIERS}
    for tier in TIERS:
        assert len(shapes[tier]) >= 2, (topic, tier, shapes[tier])
    assert shapes["exam"] - (shapes["easy"] | shapes["medium"]) or shapes["hard"] - (
        shapes["easy"] | shapes["medium"]
    ), topic


@pytest.mark.parametrize("topic", TOPICS)
def test_hard_and_exam_chains_are_longer_than_easy(questions, topic):
    def mean_len(tier):
        qs = questions[(topic, tier)]
        return sum(len(q.answer["reference_lines"]) + len(q.answer["target"]) / 40 for q in qs) / len(qs)

    assert mean_len("exam") > mean_len("easy") * 0.9, topic
    assert max(mean_len("hard"), mean_len("exam")) >= mean_len("easy"), topic


_ESCAPE_MACRO_RE = re.compile(r"\\text(?:asciicircum|asciitilde|backslash)\{\}")
_TEXT_GROUP_RE = re.compile(r"\\text\{[^{}]*\}")
_TRIG_NAME_RE = re.compile(r"(?<!\\)\b(sin|cos|tan|sec|csc|cot)\b")


@pytest.mark.parametrize("topic", TOPICS)
def test_latex_is_authored_and_is_the_same_prompt(questions, topic):
    for tier in TIERS:
        for q in questions[(topic, tier)][:40]:
            latex = q.latex
            assert latex and r"\verb" not in latex and "**" not in latex, (tier, latex)
            stripped = _ESCAPE_MACRO_RE.sub("", latex)
            spans = [m.span() for m in _TEXT_GROUP_RE.finditer(stripped)]

            def inside(pos, spans=spans):
                return any(a <= pos < b for a, b in spans)

            for m in re.finditer(r"\^", stripped):
                assert not inside(m.start()), (tier, "bare ^ inside \\text{}", latex)
            for m in _TRIG_NAME_RE.finditer(stripped):
                assert inside(m.start()), (tier, f"unescaped {m.group()} outside \\text{{}}", latex)
            flat = re.sub(r"\\[A-Za-z]+|[{}\\]", " ", latex)
            for word in [w for w in re.sub(r"[^A-Za-z ]", " ", q.text).split() if len(w) > 2][:3]:
                assert word in flat, (topic, tier, word, latex[:120])


@pytest.mark.parametrize("topic", TOPICS)
def test_in_generator_check_fires_on_a_corrupted_reference_line(topic):
    pools = MODULES[topic].pools()
    fam = dict(next(f for f in pools["medium"] if f.get("symbols")))
    ref = list(fam["ref"])
    ref[1] = "999"
    fam["ref"] = ref
    with pytest.raises(mathgen.core.VerificationError):
        kit.prove_item(random.Random("corrupt"), fam)


@pytest.mark.parametrize("topic", TOPICS)
def test_every_tier_is_a_preset_and_unknown_tiers_are_refused(topic):
    for tier in TIERS:
        assert mathgen.make(topic, seed=1, difficulty=tier).topic == topic
    gen = getattr(MODULES[topic], topic)
    with pytest.raises(ValueError):
        gen(random.Random(0), variant="nonsense")
