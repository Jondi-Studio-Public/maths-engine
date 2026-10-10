"""Independent oracle for the `p9a_quantifiers_contradiction` questions: it re-derives each answer from the
rendered question text.

Parses ONLY the rendered question text (`question.text`, `question.answer_text`) and re-derives the expected
answer from scratch; imports nothing from `mathgen.generators.p9a_quantifiers_contradiction_build`.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

from mathgen import make

QUANT_TOPIC = "quantifier_forall_exists_evaluate"
CONTRA_TOPIC = "proof_by_contradiction_identify"


# ===========================================================================
# quantifier_forall_exists_evaluate -- hard tier ("select_quantifier")
# ===========================================================================

_QUANT_HARD_RE = re.compile(
    r"for the set \{([\d, ]+)\}: '__ n in \{[\d, ]+\}, n (.+?)'\? "
    r"Choose from for-all, there-exists, both, or neither\."
)

# Re-implemented fresh, from the six predicate phrases the question can state
# -- NOT imported from the generator's own `_PREDICATES`.
_QUANT_PREDICATE_FNS = {
    "is even": lambda n: n % 2 == 0,
    "is odd": lambda n: n % 2 == 1,
    "is a multiple of 3": lambda n: n % 3 == 0,
    "is a multiple of 5": lambda n: n % 5 == 0,
    "is greater than 20": lambda n: n > 20,
    "is a perfect square": lambda n: n >= 0 and math.isqrt(n) ** 2 == n,
}

_QUANT_OPTION_FOR_PAIR = {
    (True, True): "Both 'for all' and 'there exists'",
    (False, True): "'there exists' only",
    (False, False): "Neither",
    # (True, False) is structurally unreachable: a non-empty set with every
    # element passing also has at least one element passing.
}


def _parse_quant_hard(text: str) -> tuple[list[int], str]:
    m = _QUANT_HARD_RE.search(text)
    if not m:
        raise AssertionError(f"unparseable hard-tier quantifier question: {text!r}")
    values = [int(tok.strip()) for tok in m.group(1).split(",")]
    predicate = m.group(2)
    if predicate not in _QUANT_PREDICATE_FNS:
        raise AssertionError(f"unknown predicate phrase: {predicate!r} in {text!r}")
    return values, predicate


def test_quantifier_hard_tier_oracle_enumerates_every_predicate():
    """Sweeps 1200 independently re-drawn (predicate, target_class, D)
    combinations (measured: >=189 draws per predicate, comfortably over the
    >=100 requirement) and re-derives the classification from the
    rendered text alone."""
    by_predicate: dict[str, list[int]] = defaultdict(list)
    checked = 0
    for seed in range(1200):
        q = make(QUANT_TOPIC, seed=seed, difficulty="hard", variant="select_quantifier")
        values, predicate = _parse_quant_hard(q.text)
        fn = _QUANT_PREDICATE_FNS[predicate]
        all_true = all(fn(v) for v in values)
        any_true = any(fn(v) for v in values)
        expected = _QUANT_OPTION_FOR_PAIR.get((all_true, any_true))
        assert expected is not None, (
            f"seed {seed}: unreachable pair (all_true, any_true)=({all_true}, {any_true}) "
            f"for predicate {predicate!r}, values {values} -- text {q.text!r}"
        )
        assert expected == q.answer_text, (
            f"seed {seed}: predicate {predicate!r}, values {values} -> "
            f"oracle expected {expected!r}, generator gave {q.answer_text!r}"
        )
        by_predicate[predicate].append(seed)
        checked += 1
    assert checked == 1200
    for predicate in _QUANT_PREDICATE_FNS:
        n = len(by_predicate[predicate])
        assert n >= 100, f"predicate {predicate!r} only independently re-drawn {n}/1200 times"


# ===========================================================================
# proof_by_contradiction_identify -- hard tier ("compound_claim")
# ===========================================================================

_COMPOUND_RE = re.compile(
    r"^Claim: (\d+) is (.+) and \1 is (.+)\. Assume this compound claim for contradiction\. "
    r"Which clause produces the contradiction\?$"
)

# Re-implemented fresh, from the ten adjective phrases the question can state
# -- NOT imported from the generator's own `_PAIRS`.
_ADJ_FAMILY = {
    "even": "parity",
    "odd": "parity",
    "positive": "sign",
    "negative": "sign",
    "a multiple of 3": "mult3",
    "not a multiple of 3": "mult3",
    "a multiple of 4": "mult4",
    "not a multiple of 4": "mult4",
    "a perfect square": "square",
    "not a perfect square": "square",
}


def _oracle_adj_truth(n: int, adj: str) -> bool:
    if adj == "even":
        return n % 2 == 0
    if adj == "odd":
        return n % 2 == 1
    if adj == "positive":
        return n > 0
    if adj == "negative":
        return n < 0
    if adj == "a multiple of 3":
        return n % 3 == 0
    if adj == "not a multiple of 3":
        return n % 3 != 0
    if adj == "a multiple of 4":
        return n % 4 == 0
    if adj == "not a multiple of 4":
        return n % 4 != 0
    if adj == "a perfect square":
        return n >= 0 and math.isqrt(n) ** 2 == n
    if adj == "not a perfect square":
        return not (n >= 0 and math.isqrt(n) ** 2 == n)
    raise AssertionError(f"unknown adjective phrase: {adj!r}")


def _parse_compound(text: str) -> tuple[int, str, str]:
    m = _COMPOUND_RE.match(text)
    if not m:
        raise AssertionError(f"unparseable compound-claim question: {text!r}")
    return int(m.group(1)), m.group(2), m.group(3)


def _expected_family_pairs():
    families = sorted(set(_ADJ_FAMILY.values()))
    pairs = set()
    for i in range(len(families)):
        for j in range(i + 1, len(families)):
            for orientation in ("first", "second"):
                pairs.add((frozenset((families[i], families[j])), orientation))
    return pairs


def test_compound_claim_oracle_enumerates_every_pair_and_orientation():
    """5 choose 2 = 10 unordered family pairs x 2 false_clause orientations =
    20 rows (measured: fully saturated within 4000 seeds). For every row,
    confirms from the rendered text alone that exactly one stated clause is
    false at n and it matches the claimed answer."""
    expected_keys = _expected_family_pairs()
    assert len(expected_keys) == 20

    seen_keys = set()
    checked = 0
    for seed in range(4000):
        q = make(CONTRA_TOPIC, seed=seed, difficulty="hard", variant="compound_claim")
        n, adj1, adj2 = _parse_compound(q.text)
        assert adj1 in _ADJ_FAMILY, f"seed {seed}: unknown clause-1 adjective {adj1!r}"
        assert adj2 in _ADJ_FAMILY, f"seed {seed}: unknown clause-2 adjective {adj2!r}"
        family1, family2 = _ADJ_FAMILY[adj1], _ADJ_FAMILY[adj2]
        assert family1 != family2, f"seed {seed}: both clauses drawn from the same family {family1!r}"

        truth1 = _oracle_adj_truth(n, adj1)
        truth2 = _oracle_adj_truth(n, adj2)
        false_count = (not truth1) + (not truth2)
        assert false_count == 1, (
            f"seed {seed}: expected exactly one false clause at n={n}, "
            f"got truth1={truth1} truth2={truth2} -- text {q.text!r}"
        )

        expected_answer = (
            f"The first clause ('{n} is {adj1}')" if not truth1 else f"The second clause ('{n} is {adj2}')"
        )
        assert expected_answer == q.answer_text, (
            f"seed {seed}: oracle expected {expected_answer!r}, generator gave {q.answer_text!r}"
        )

        orientation = "first" if not truth1 else "second"
        seen_keys.add((frozenset((family1, family2)), orientation))
        checked += 1

    assert checked == 4000
    missing = expected_keys - seen_keys
    assert not missing, f"draw space not saturated over 4000 seeds, missing: {missing}"
