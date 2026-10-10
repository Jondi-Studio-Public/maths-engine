"""Independent oracle for the `p18a_sharing_grouping` questions: it re-derives each answer from the rendered
question text.

Every prompt marks its reading with one of two fixed substrings (duplicated
here, not imported): "between <D> <container>" always means the stated
number is a CONTAINER COUNT (partitive: sharing), and "hold(s) <D> <item>"
always means the stated number is a PER-CONTAINER SIZE (quotitive: grouping).
The exam tier's step 1 is always the partitive scene-setter ("shares ...
equally between <g> <container>"); its step 2 marks the SAME two readings
again, with "among <s2> <container2>" for a further share and
"(grouped|packed) into <container2> of <s2> each" for a further group.

Also enumerates context template x reading, marking each coherent: every
draw this oracle parses either matches the partitive or the quotitive answer
shape, never neither and never both.

Seed count is `N_SEEDS` (default 250 per tier, >= 1000 questions over the
four tiers), overridable by `P18A_ORACLE_SEEDS`.
"""

from __future__ import annotations

import os
import re

import pytest

from mathgen import make

TOPIC = "p18a_sharing_grouping"
TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = int(os.environ.get("P18A_ORACLE_SEEDS", "250"))

_ITEM_PL = r"(?:apples|stickers|marbles|counters)"
_CONTAINER = r"(?:bags|boxes|baskets|cups|bag|box|basket|cup)"

_TOTAL_RE = re.compile(r"(\d+)\s+(" + _ITEM_PL + r")\b")
_BETWEEN_RE = re.compile(r"between\s+(\d+)\s+(" + _CONTAINER + r")\b")
_HOLD_RE = re.compile(r"holds?\s+(\d+)\s+(" + _ITEM_PL + r")\b")
_AMONG_RE = re.compile(r"among\s+(\d+)\s+(" + _CONTAINER + r")\b")
_GROUPED_INTO_RE = re.compile(r"(?:grouped|packed) into (" + _CONTAINER + r") of (\d+) each\b")

_PARTITIVE_ANS_RE = re.compile(r"^(\d+) [a-z]+ in each [a-z]+(?:, (\d+) [a-z]+ left over)?$")
_QUOTITIVE_REM_ANS_RE = re.compile(r"^(\d+) full [a-z]+, (\d+) [a-z]+ left over$")
_QUOTITIVE_PLAIN_ANS_RE = re.compile(r"^(\d+) [a-z]+$")


def _parse_answer(text: str):
    """Returns (kind, number, remainder) parsed from an option's display
    text -- 'share' for a partitive ('N x in each y[, r left over]') answer,
    'group' for a quotitive ('N y'/'N full y, r x left over') answer."""
    m = _PARTITIVE_ANS_RE.match(text)
    if m:
        return "share", int(m.group(1)), int(m.group(2)) if m.group(2) else 0
    m = _QUOTITIVE_REM_ANS_RE.match(text)
    if m:
        return "group", int(m.group(1)), int(m.group(2))
    m = _QUOTITIVE_PLAIN_ANS_RE.match(text)
    if m:
        return "group", int(m.group(1)), 0
    raise AssertionError(f"oracle: could not parse option text {text!r}")


def _oracle_simple(text: str):
    """easy/medium/hard: exactly one of the two markers is present. Returns
    (situation, total, divisor, quotient, remainder)."""
    total_m = _TOTAL_RE.search(text)
    assert total_m is not None, f"oracle: no total found in {text!r}"
    total = int(total_m.group(1))

    between_m = _BETWEEN_RE.search(text)
    hold_m = _HOLD_RE.search(text)
    assert (between_m is not None) != (hold_m is not None), (
        f"oracle: expected exactly one of between/hold markers in {text!r}"
    )
    if between_m:
        situation = "share"
        divisor = int(between_m.group(1))
    else:
        situation = "group"
        divisor = int(hold_m.group(1))

    quotient, remainder = divmod(total, divisor)
    return situation, total, divisor, quotient, remainder


def _oracle_exam(text: str):
    """exam: step 1 is always the partitive scene-setter ('between g
    container'); step 2 marks the final situation via 'among s2 container2'
    (share-final) or '(grouped|packed) into container2 of s2 each'
    (group-final). Returns (situation, c) -- the FINAL result only."""
    total_m = _TOTAL_RE.search(text)
    assert total_m is not None, f"oracle: no total found in {text!r}"
    total = int(total_m.group(1))

    step1_m = _BETWEEN_RE.search(text)
    assert step1_m is not None, f"oracle: no step-1 'between' marker in {text!r}"
    g = int(step1_m.group(1))
    assert total % g == 0, f"oracle: step-1 total {total} not divisible by g {g}"
    e = total // g

    among_m = _AMONG_RE.search(text)
    grouped_m = _GROUPED_INTO_RE.search(text)
    assert (among_m is not None) != (grouped_m is not None), (
        f"oracle: expected exactly one of among/grouped-into markers in {text!r}"
    )
    if among_m:
        situation = "share"
        s2 = int(among_m.group(1))
    else:
        situation = "group"
        s2 = int(grouped_m.group(2))

    assert e % s2 == 0, f"oracle: step-2 share {e} not divisible by s2 {s2}"
    c = e // s2
    return situation, c


def _all(tier, seeds=None):
    seeds = seeds if seeds is not None else range(N_SEEDS)
    return [make(TOPIC, seed=s, difficulty=tier) for s in seeds]


@pytest.mark.parametrize("tier", ("easy", "medium", "hard"))
def test_oracle_matches_answer_text_simple_tiers(tier):
    mismatches = []
    for q in _all(tier):
        situation, total, divisor, quotient, remainder = _oracle_simple(q.text)
        kind, number, rem = _parse_answer(q.answer_text)
        if (kind, number, rem) != (situation, quotient, remainder):
            mismatches.append((q.seed, q.text, (situation, quotient, remainder), q.answer_text))
    assert mismatches == [], mismatches


def test_oracle_matches_answer_text_exam_tier():
    mismatches = []
    for q in _all("exam"):
        situation, c = _oracle_exam(q.text)
        kind, number, rem = _parse_answer(q.answer_text)
        if (kind, number, rem) != (situation, c, 0):
            mismatches.append((q.seed, q.text, (situation, c), q.answer_text))
    assert mismatches == [], mismatches


@pytest.mark.parametrize("tier", ("easy", "medium", "hard"))
def test_oracle_every_distractor_disagrees_with_the_oracle(tier):
    for q in _all(tier, seeds=range(60)):
        situation, total, divisor, quotient, remainder = _oracle_simple(q.text)
        oracle_value = (situation, quotient, remainder)
        for d in q.distractors:
            assert _parse_answer(d) != oracle_value, f"{tier} seed {q.seed}: distractor {d!r} matches the oracle"


def test_oracle_every_distractor_disagrees_with_the_oracle_exam():
    for q in _all("exam", seeds=range(60)):
        situation, c = _oracle_exam(q.text)
        oracle_value = (situation, c, 0)
        for d in q.distractors:
            assert _parse_answer(d) != oracle_value, f"exam seed {q.seed}: distractor {d!r} matches the oracle"


@pytest.mark.parametrize("tier", ("easy", "medium", "hard"))
def test_oracle_hard_tier_remainder_matches(tier):
    for q in _all(tier, seeds=range(60)):
        situation, total, divisor, quotient, remainder = _oracle_simple(q.text)
        if tier == "hard":
            assert remainder != 0, f"hard seed {q.seed}: oracle found zero remainder in {q.text!r}"
        else:
            assert remainder == 0, f"{tier} seed {q.seed}: oracle found nonzero remainder in {q.text!r}"


def test_context_template_x_reading_all_coherent():
    """Every context (template shape) x reading combination that this
    generator can draw produces a question this oracle can parse and agree
    with -- i.e. no template/reading pairing is silently incoherent."""
    seen_pairs = set()
    for tier in TIERS:
        for q in _all(tier, seeds=range(60)):
            if tier == "exam":
                situation, c = _oracle_exam(q.text)
                kind, number, rem = _parse_answer(q.answer_text)
                assert (kind, number, rem) == (situation, c, 0), q.text
            else:
                situation, total, divisor, quotient, remainder = _oracle_simple(q.text)
                kind, number, rem = _parse_answer(q.answer_text)
                assert (kind, number, rem) == (situation, quotient, remainder), q.text
            seen_pairs.add((tier, situation))
    # both readings appear at every tier.
    for tier in TIERS:
        assert (tier, "share") in seen_pairs, f"{tier}: sharing reading never seen in 60 seeds"
        assert (tier, "group") in seen_pairs, f"{tier}: grouping reading never seen in 60 seeds"
