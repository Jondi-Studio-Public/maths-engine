"""PR1A task 1964: senior-prove-and-apply-the-identities-for-products-of
(ACMSM047). Fast checks against the generator's own tables -- the marking
dict shape, the pool sizes, the domain statements and the in-generator
checks firing on corrupted entries. (The rubric-marking checks of each
reference chain and trap run in the site that hosts the rubric marker.) `tests/test_pr1a_product_to_sum_oracle.py` is the
independent oracle that never imports this module.
"""

from __future__ import annotations

import re

import pytest
import sympy

from mathgen.grading.typed import _parse_expression
from mathgen.core import make
from mathgen.generators import pr1a_product_to_sum_build as gen

SKILL_ID = "senior-prove-and-apply-the-identities-for-products-of"
TOPIC = "pr1a_product_to_sum_proof"
TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = 200

RUBRIC_IDS = {"correctness", "justified_steps", "communication"}
_NO_STAR_RE = re.compile(r"\*\*|\*")
_SAFE_SYMBOLS = {"x", "t", "u", "v", "A", "B", "P", "R", "theta", "alpha", "phi"}


def _questions(tier: str, n: int = N_SEEDS):
    return [make(TOPIC, seed=s, difficulty=tier) for s in range(n)]


# --- determinism -----------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_determinism(tier):
    for s in range(20):
        q1 = make(TOPIC, seed=s, difficulty=tier)
        q2 = make(TOPIC, seed=s, difficulty=tier)
        assert q1.text == q2.text
        assert q1.answer == q2.answer
        assert q1.answer_text == q2.answer_text


# --- text/answer_text free of ** and * --------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_no_sympy_syntax_in_learner_facing_text(tier):
    for q in _questions(tier, 60):
        assert not _NO_STAR_RE.search(q.text), q.text
        assert not _NO_STAR_RE.search(q.answer_text), q.answer_text
        assert q.distractors == []


# --- marking dict shape ------------------------------------------------------


def _all_marking_blocks(tier: str, n: int = N_SEEDS):
    seen = {}
    for s in range(n):
        q = make(TOPIC, seed=s, difficulty=tier)
        marking = q.answer
        key = (marking["start"], marking["target"], tuple(marking["reference_lines"]))
        seen.setdefault(key, marking)
    return seen


@pytest.mark.parametrize("tier", TIERS)
def test_marking_dict_shape(tier):
    blocks = _all_marking_blocks(tier)
    assert blocks, f"no marking blocks collected for {tier}"
    for marking in blocks.values():
        assert isinstance(marking["start"], str) and isinstance(marking["target"], str)
        assert isinstance(marking["symbols"], dict)
        checks = marking["checks"]
        assert len(checks) == 1
        check0 = checks[0]
        assert check0["type"] == "equivalence_chain"
        assert check0["start"] == marking["start"]
        assert check0["target"] == marking["target"]
        assert check0["symbols"] == marking["symbols"]
        assert check0["authoritative"] is True
        assert all(k["criterion_id"] in RUBRIC_IDS for k in marking["key_steps"])
        assert any(k["criterion_id"] == "justified_steps" for k in marking["key_steps"])
        assert any(k["criterion_id"] == "communication" for k in marking["key_steps"])
        for trap in marking["traps"]:
            assert trap["expected_check"] in (
                "authoritative_failure",
                "non_authoritative_failure",
                "passes_chain",
            )
            assert trap["example_lines"]
        expected_checks = {t["expected_check"] for t in marking["traps"]}
        assert "passes_chain" in expected_checks
        assert "non_authoritative_failure" in expected_checks
        if marking["symbols"]:  # symbolic (prove/apply) item: must also carry an authoritative trap
            assert "authoritative_failure" in expected_checks, (tier, marking["start"])


# --- reference chains and traps against the REAL checker --------------------


# --- distinct pool size (rule 9 / the ladder) -------------------------------


def test_easy_tier_pool_size():
    # 4 product-to-sum identities x 5 letter pairs.
    assert len(_all_marking_blocks("easy", 200)) == 20


def test_medium_tier_pool_size():
    # 3 families (sincos, coscos, sinsin) x 4 (m, n) pairs each -- only pairs
    # whose reference chain was confirmed against `symbolically_equal` are
    # kept.
    assert len(_all_marking_blocks("medium", 200)) == 12


def test_hard_tier_pool_size():
    # (i) 4 sum-to-product identities x 5 letter pairs = 20, plus
    # (ii) 4 families x 3 numeric (m, n) pairs = 12. Total 32.
    assert len(_all_marking_blocks("hard", 400)) == 32


def test_exam_tier_pool_size():
    # (a) 6 special-angle apply items, plus (b) 2 composite tan identities.
    assert len(_all_marking_blocks("exam", 200)) == 8


# --- decide() short-circuit on an authoritative failure ---------------------

_HIGH_CONFIDENCE_ASSESSMENT = {
    "criteria": [
        {"id": "correctness", "points": 4, "evidence": "matches"},
        {"id": "justified_steps", "points": 3, "evidence": "cites the given"},
        {"id": "communication", "points": 1, "evidence": "clear"},
    ],
    "confidence": "high",
    "needs_review": False,
}
_CLEAN_ADVERSARIAL = {"issues": [], "confidence": "high", "needs_review": False}
_RUBRIC = {
    "id": "trig-identity-proof-v1",
    "pass_threshold": 0.8,
    "criteria": [
        {"id": "correctness", "label": "Correctness", "max_points": 4},
        {"id": "justified_steps", "label": "Justified steps", "max_points": 3},
        {"id": "communication", "label": "Communication", "max_points": 1},
    ],
}


# --- in-generator check fires on a corrupted reference line -----------------


def test_verify_symbolic_chain_fires_on_corruption():
    good = [
        "(sin(A + B) + sin(A - B))/2",
        "(sin(A)*cos(B) + cos(A)*sin(B) + sin(A)*cos(B) - cos(A)*sin(B))/2",
        "(2*sin(A)*cos(B))/2",
        "sin(A)*cos(B)",
    ]
    gen._verify_symbolic_chain(good, "sanity")
    corrupted = list(good)
    corrupted[-1] = "cos(A)*cos(B)"  # no longer equal to line 3
    with pytest.raises(Exception):
        gen._verify_symbolic_chain(corrupted, "corrupted")


def test_verify_single_symbol_chain_fires_on_corruption():
    entry = gen._MEDIUM_POOL[0]
    gen._verify_single_symbol_chain(entry["reference_lines"], "sanity")
    corrupted = list(entry["reference_lines"])
    corrupted[-1] = "0"
    with pytest.raises(Exception):
        gen._verify_single_symbol_chain(corrupted, "corrupted")


def test_verify_numeric_chain_fires_on_corruption():
    entry = gen._EXAM_APPLY_POOL[0]
    gen._verify_numeric_chain(entry["reference_lines"], "sanity")
    corrupted = list(entry["reference_lines"])
    corrupted[-1] = "0"
    with pytest.raises(Exception):
        gen._verify_numeric_chain(corrupted, "corrupted")


# --- symbol names round-trip sympify and _parse_expression ------------------


def test_symbols_used_are_in_the_safe_list():
    used = set()
    for tier in TIERS:
        for marking in _all_marking_blocks(tier, 60).values():
            used |= set(marking["symbols"])
            for line in marking["reference_lines"]:
                for name in re.findall(r"[A-Za-z_]\w*", line):
                    if name not in ("sin", "cos", "tan", "pi", "sqrt"):
                        used.add(name)
    assert used <= _SAFE_SYMBOLS, used - _SAFE_SYMBOLS
    for name in used:
        sympy.Symbol(name)  # sympify-safe by construction
        parsed = _parse_expression(name)
        assert parsed is not None


# --- skill loads correctly ---------------------------------------------------


# --- rule 12: the rendered start/end line parses to the marking dict's ---
# --- start/target, extracted from `text` by regex here, never a second ---
# --- copy kept in the generator. Reuses the oracle's own regex patterns ---
# --- (test-side only; the oracle itself never imports this module). ------


# --- rule 13: never claim "all real" for an identity not defined -----------
# --- everywhere (tan or a division), and the exam composite states its ---
# --- domain instead. --------------------------------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_no_false_all_real_claim(tier):
    for s in range(60):
        q = make(TOPIC, seed=s, difficulty=tier)
        text_low = q.text.lower()
        if "all real" in text_low or "every real" in text_low:
            marking = q.answer
            for expr in (marking["start"], marking["target"]):
                assert not re.search(r"\btan\(|\bsec\(|\bcsc\(|\bcot\(", expr), (tier, s, expr)
                assert "/" not in expr, (tier, s, expr)


def test_exam_tan_composite_states_domain_condition():
    checked = 0
    for s in range(200):
        q = make(TOPIC, seed=s, difficulty="exam")
        if "tan(" not in q.answer["target"]:
            continue
        text_low = q.text.lower()
        assert "all real" not in text_low and "every real" not in text_low, q.text
        assert "nonzero" in text_low, q.text
        checked += 1
    assert checked > 0


if __name__ == "__main__":
    # Marking dump helper -- not collected by
    # pytest as a test.
    for tier in TIERS:
        print(f"=== {tier} ===")
        for s in range(3):
            q = make(TOPIC, seed=s, difficulty=tier)
            print(q.text)
            print(q.answer)
            print()
