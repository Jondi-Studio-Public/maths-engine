"""Independent oracle for `p19a_part_part_whole`.

Three mutually exclusive text shapes, identified by fixed keyword markers
duplicated here (not imported):

- "Whole:" present -- the explicit-label shape. The whole and the given part
  are read directly off their own labels, in whichever line order they
  appear (the label decides, never position).
- "make" present (and no "Whole:") -- the combining-language shape. The
  whole is the number immediately followed by "make"; the given part is the
  other number in the text (there are always exactly two).
- neither marker present -- the easy/medium shape, where the whole is always
  the FIRST number stated and the given part is the SECOND (both templates
  in this generator's easy/medium tiers state the whole before the part).

Every shape re-derives the missing part by direct subtraction (whole - part)
and checks `2 <= whole <= 10`.

Seed count is `N_SEEDS` (default 250 per tier, >= 750 questions over the
three tiers), overridable by `P19A_ORACLE_SEEDS`.
"""

from __future__ import annotations

import os
import re

import pytest

from mathgen import make

TOPIC = "p19a_part_part_whole"
TIERS = ("easy", "medium", "hard")
N_SEEDS = int(os.environ.get("P19A_ORACLE_SEEDS", "250"))

_WHOLE_LABEL_RE = re.compile(r"Whole:\s*(\d+)")
_PART_LABEL_RE = re.compile(r"One part:\s*(\d+)")
_MAKE_RE = re.compile(r"make\s+(\d+)")
_ALL_NUMS_RE = re.compile(r"\d+")


def _oracle_whole_part(text: str) -> tuple[int, int]:
    """Returns (whole, part) identified by the text's own keyword shape."""
    if "Whole:" in text:
        whole_m = _WHOLE_LABEL_RE.search(text)
        part_m = _PART_LABEL_RE.search(text)
        assert whole_m is not None and part_m is not None, f"oracle: labelled shape missing a label in {text!r}"
        return int(whole_m[1]), int(part_m[1])
    if "make" in text:
        whole_m = _MAKE_RE.search(text)
        assert whole_m is not None, f"oracle: 'make' present but no number follows it in {text!r}"
        whole = int(whole_m[1])
        nums = [int(x) for x in _ALL_NUMS_RE.findall(text)]
        others = [n for n in nums if n != whole]
        assert len(others) == 1, f"oracle: expected exactly one other number in {text!r}, got {others}"
        return whole, others[0]
    nums = [int(x) for x in _ALL_NUMS_RE.findall(text)]
    assert len(nums) >= 2, f"oracle: expected at least 2 numbers in {text!r}"
    return nums[0], nums[1]


def _oracle_key(text: str) -> int:
    whole, part = _oracle_whole_part(text)
    assert 2 <= whole <= 10, f"oracle: whole {whole} out of range 2-10 in {text!r}"
    return whole - part


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


def test_oracle_fires_on_a_deliberately_corrupted_whole():
    """Red-first proof: the oracle's own range check must reject a whole
    outside 2-10 rather than silently accepting it."""
    with pytest.raises(AssertionError):
        _oracle_key("Whole: 11. One part: 4. Other part: ?")


def test_context_template_x_shape_all_coherent():
    """Every (tier, shape) combination this generator can draw produces a
    question this oracle can parse and agree with."""
    seen = set()
    for tier in TIERS:
        for q in _all(tier, seeds=range(60)):
            expected = _oracle_key(q.text)
            assert expected == int(q.answer_text), q.text
            if "Whole:" in q.text:
                shape = "labelled"
            elif "make" in q.text:
                shape = "combine"
            else:
                shape = "plain"
            seen.add((tier, shape))
    for tier in ("easy", "medium"):
        assert (tier, "plain") in seen, f"{tier}: plain shape never seen in 60 seeds"
    assert ("hard", "combine") in seen, "hard: combine shape never seen in 60 seeds"
    assert ("hard", "labelled") in seen, "hard: labelled shape never seen in 60 seeds"
