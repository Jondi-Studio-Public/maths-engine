"""Independent oracle for the `p9a_examples_counterexamples` questions: it re-derives each answer from the
rendered question text.

Parses ONLY the rendered question text (``question.text``, ``question.answer_text``) and re-derives the
expected answer from scratch, by re-implementing each predicate's boolean test fresh in this file. Imports
nothing from ``mathgen.generators.p9a_examples_counterexamples_build`` -- not `_PREDICATES`, not a helper, not
a table.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

from mathgen import make

TOPIC = "counterexample_identify"

# Re-implemented fresh, from the six predicate phrases the question can state.
_PREDICATE_FNS = {
    "is even": lambda n: n % 2 == 0,
    "is odd": lambda n: n % 2 == 1,
    "is a multiple of 3": lambda n: n % 3 == 0,
    "is a multiple of 5": lambda n: n % 5 == 0,
    "is greater than 20": lambda n: n > 20,
    "is a perfect square": lambda n: n >= 0 and math.isqrt(n) ** 2 == n,
}

_EASY_PHRASES = {"is even", "is odd"}
_MEDIUM_PHRASES = {"is a multiple of 3", "is a multiple of 5", "is greater than 20", "is a perfect square"}

_ALL_PHRASES_SORTED = sorted(_PREDICATE_FNS, key=len, reverse=True)
_PHRASE_ALTERNATION = "|".join(re.escape(p) for p in _ALL_PHRASES_SORTED)

_CLASSIFY_RE = re.compile(
    r"^Claim: for every n in \{([\d, ]+)\}, n (" + _PHRASE_ALTERNATION + r")\. "
    r"Is n = (-?\d+) an example or a counterexample to this claim\?$"
)

_FIND_RE = re.compile(
    r"^Claim: every one of the values ([\d, ]+) (" + _PHRASE_ALTERNATION + r")\. "
    r"Which of these values disproves the claim\?$"
)


def _parse_classify(text: str) -> tuple[list[int], str, int]:
    m = _CLASSIFY_RE.match(text)
    if not m:
        raise AssertionError(f"unparseable classify question: {text!r}")
    values = [int(tok.strip()) for tok in m.group(1).split(",")]
    predicate = m.group(2)
    c = int(m.group(3))
    return values, predicate, c


def _parse_find(text: str) -> tuple[list[int], str]:
    m = _FIND_RE.match(text)
    if not m:
        raise AssertionError(f"unparseable find_counterexample question: {text!r}")
    values = [int(tok.strip()) for tok in m.group(1).split(",")]
    predicate = m.group(2)
    return values, predicate


# ===========================================================================
# classify variant (easy/medium): >=100 independently-drawn triples per tier,
# covering both tiers' predicate slices.
# ===========================================================================


def _run_classify_oracle(difficulty, params, expected_phrases, min_seeds):
    by_predicate: dict[str, int] = defaultdict(int)
    checked = 0
    for seed in range(min_seeds):
        q = make(TOPIC, seed=seed, difficulty=difficulty, **params)
        values, predicate, c = _parse_classify(q.text)
        assert predicate in expected_phrases, (
            f"seed {seed}: predicate {predicate!r} not in the {difficulty} slice {expected_phrases}"
        )
        assert c in values, f"seed {seed}: c={c} missing from displayed set {values}"
        assert len(values) == 5, f"seed {seed}: displayed set is not exactly 5 values: {values}"
        assert len(set(values)) == 5, f"seed {seed}: displayed set has duplicates: {values}"

        fn = _PREDICATE_FNS[predicate]
        expected = "Example" if fn(c) else "Counterexample"
        assert expected == q.answer_text, (
            f"seed {seed}: predicate {predicate!r}, c={c} -> oracle expected {expected!r}, "
            f"generator gave {q.answer_text!r} -- text {q.text!r}"
        )
        by_predicate[predicate] += 1
        checked += 1
    assert checked == min_seeds
    for phrase in expected_phrases:
        assert by_predicate[phrase] >= 100, (
            f"{difficulty}: predicate {phrase!r} only independently re-drawn "
            f"{by_predicate[phrase]}/{min_seeds} times (need >=100)"
        )
    return checked, dict(by_predicate)


def test_classify_easy_oracle_enumerates_100_plus_per_predicate():
    """Requirement: >=100 independently-drawn triples PER predicate, not
    just per tier -- easy has 2 predicates, so 400 seeds (measured: ~200 each,
    comfortably over 100)."""
    checked, by_predicate = _run_classify_oracle("easy", dict(variant="classify", pool="easy"), _EASY_PHRASES, 400)
    assert checked >= 400
    print(f"easy per-predicate counts over {checked} seeds: {by_predicate}")


def test_classify_medium_oracle_enumerates_100_plus_per_predicate():
    """Requirement: >=100 independently-drawn triples PER predicate --
    medium has 4 predicates, so 800 seeds (measured: ~185-220 each)."""
    checked, by_predicate = _run_classify_oracle(
        "medium", dict(variant="classify", pool="medium"), _MEDIUM_PHRASES, 800
    )
    assert checked >= 800
    print(f"medium per-predicate counts over {checked} seeds: {by_predicate}")


# ===========================================================================
# find_counterexample variant (hard): all 6 predicates x >=20 independently-
# drawn seeds each.
# ===========================================================================


def test_find_counterexample_oracle_enumerates_all_6_predicates_over_20_seeds_each():
    by_predicate: dict[str, int] = defaultdict(int)
    checked = 0
    for seed in range(400):
        q = make(TOPIC, seed=seed, difficulty="hard", variant="find_counterexample")
        values, predicate = _parse_find(q.text)
        assert predicate in _PREDICATE_FNS, f"seed {seed}: unknown predicate phrase {predicate!r}"
        assert len(values) == 4, f"seed {seed}: expected 4 listed values, got {values}"

        fn = _PREDICATE_FNS[predicate]
        failing = [v for v in values if not fn(v)]
        passing = [v for v in values if fn(v)]
        assert len(failing) == 1, (
            f"seed {seed}: expected exactly one failing value among {values} for {predicate!r}, "
            f"got failing={failing} passing={passing} -- text {q.text!r}"
        )
        assert str(failing[0]) == q.answer_text, (
            f"seed {seed}: oracle expected {failing[0]!r}, generator gave {q.answer_text!r}"
        )
        by_predicate[predicate] += 1
        checked += 1
    assert checked == 400
    for phrase in _PREDICATE_FNS:
        n = by_predicate[phrase]
        assert n >= 20, f"predicate {phrase!r} only independently re-drawn {n}/400 times"
