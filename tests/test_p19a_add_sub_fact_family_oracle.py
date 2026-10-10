"""Independent oracle for `p19a_add_sub_fact_family`.

Every stated addition fact appears as either "a + b = s" (add-first) or
"s = a + b" (sum-first); the final subtraction question always appears as a
bare "m - n" (the only "-" in the whole text, since no addition fact ever
uses a minus sign). This oracle:

1. collects every stated fact (normalised to (a, b, s) triples) from BOTH
   forms;
2. finds the (minuend, subtrahend) pair from the "m - n" expression;
3. confirms the subtraction is genuinely RELATED to a stated fact -- the
   minuend equals that fact's sum and the subtrahend is one of its two
   addends (never just "any arithmetic that happens to work out"); and
4. recomputes the expected key as `minuend - subtrahend` directly by
   subtraction, a different route from however the generator picked "the
   other addend".

Seed count is `N_SEEDS` (default 250 per tier, >= 750 questions over the
three tiers), overridable by `P19A_ORACLE_SEEDS`.
"""

from __future__ import annotations

import os
import re

import pytest

from mathgen import make

TOPIC = "p19a_add_sub_fact_family"
TIERS = ("easy", "medium", "hard")
N_SEEDS = int(os.environ.get("P19A_ORACLE_SEEDS", "250"))

_ADD_FIRST_RE = re.compile(r"(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)")
_SUM_FIRST_RE = re.compile(r"(\d+)\s*=\s*(\d+)\s*\+\s*(\d+)")
_SUBTRACTION_RE = re.compile(r"(\d+)\s*-\s*(\d+)")


def _oracle_facts(text: str) -> list[tuple[int, int, int]]:
    """Every stated fact in the text, normalised to (a, b, s) triples,
    regardless of which of the two written forms was used."""
    facts = []
    for m in _ADD_FIRST_RE.finditer(text):
        a, b, s = int(m[1]), int(m[2]), int(m[3])
        facts.append((a, b, s))
    for m in _SUM_FIRST_RE.finditer(text):
        s, a, b = int(m[1]), int(m[2]), int(m[3])
        facts.append((a, b, s))
    return facts


def _oracle_subtraction(text: str) -> tuple[int, int]:
    m = _SUBTRACTION_RE.search(text)
    assert m is not None, f"oracle: no subtraction expression found in {text!r}"
    return int(m[1]), int(m[2])


def _oracle_key(text: str) -> int:
    """Re-derives the expected key from `question.text` alone: finds the
    subtraction's (minuend, subtrahend), confirms it is related to a stated
    fact (minuend == that fact's sum, subtrahend in that fact's two
    addends), and returns minuend - subtrahend computed directly."""
    facts = _oracle_facts(text)
    assert facts, f"oracle: no stated addition fact found in {text!r}"
    minuend, subtrahend = _oracle_subtraction(text)
    related = [f for f in facts if f[2] == minuend and subtrahend in (f[0], f[1])]
    assert related, (
        f"oracle: subtraction {minuend} - {subtrahend} is not related to any stated fact {facts} in {text!r}"
    )
    return minuend - subtrahend


def _all(tier, seeds=None):
    seeds = seeds if seeds is not None else range(N_SEEDS)
    return [make(TOPIC, seed=s, difficulty=tier) for s in seeds]


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_matches_answer_text(tier):
    mismatches = []
    for q in _all(tier):
        expected = _oracle_key(q.text)
        if expected != int(q.answer_text):
            mismatches.append((q.seed, q.text, expected, q.answer_text))
    assert mismatches == [], mismatches


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_every_distractor_disagrees_with_the_oracle(tier):
    for q in _all(tier, seeds=range(60)):
        expected = _oracle_key(q.text)
        for d in q.distractors:
            assert int(d) != expected, f"{tier} seed {q.seed}: distractor {d!r} matches the oracle {expected}"


def test_oracle_fires_on_a_deliberately_corrupted_key():
    """Red-first proof: an oracle that parsed the wrong fact/side would not
    notice a corrupted key. Confirm the oracle's relatedness check itself
    rejects an unrelated subtraction (the whole point of step 3)."""
    with pytest.raises(AssertionError):
        _oracle_key("8 + 5 = 13. What is 13 - 2?")  # 2 is not an addend of 8+5=13


def test_hard_tier_subtrahend_always_comes_from_the_second_stated_fact():
    """The oracle's relatedness check must select the SECOND fact for hard
    tier's subtraction (never the first), matching the generator's own
    construction rule -- verified independently here by checking which fact
    the subtrahend actually belongs to."""
    for q in _all("hard", seeds=range(60)):
        facts = _oracle_facts(q.text)
        assert len(facts) == 2
        minuend, subtrahend = _oracle_subtraction(q.text)
        first, second = facts
        assert subtrahend in (second[0], second[1]), (
            f"hard seed {q.seed}: subtrahend {subtrahend} not drawn from the second fact {second} in {q.text!r}"
        )
