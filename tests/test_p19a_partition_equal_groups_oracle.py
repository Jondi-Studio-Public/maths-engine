"""- easy: parses "{total} {item} into {groups} equal groups" and computes
  total // groups directly.
- medium: parses "{bundles} bundles of ten and {loose} loose {item}" and
  "{groups} groups", and computes (bundles // groups) * 10 +
  (loose // groups) -- the tens and ones partitioned separately, never
  dividing the combined total.
- hard: parses the same bundle/loose/groups triple. The generator's prompt
  defect fix), so the oracle computes it BOTH ways from the parsed numbers
  and picks the one that applies: if `bundles % groups == 0`, expected is
  `(bundles // groups) * 10 + (loose // groups)` (no unbundling needed,
  exactly medium's formula); if `bundles % groups == 1`, expected is
  `((bundles - 1) // groups) * 10 + (10 + loose) // groups` (one bundle
  unbundled). Any other remainder is a generator defect (the oracle asserts
  this never happens).

Seed count is `N_SEEDS` (default 250 per tier, >= 1000 questions over the
three tiers at the override), overridable by `P19A_ORACLE_SEEDS`.
"""

from __future__ import annotations

import os
import re

import pytest

from mathgen import make

TOPIC = "p19a_partition_equal_groups"
TIERS = ("easy", "medium", "hard")
N_SEEDS = int(os.environ.get("P19A_ORACLE_SEEDS", "250"))

_TOTAL_RE = re.compile(r"(\d+)\s+[a-z]+")
_EASY_GROUPS_RE = re.compile(r"into\s+(\d+)\s+(?:equal\s+)?groups")
_BUNDLES_LOOSE_RE = re.compile(r"(\d+)\s+bundles of ten.*?(\d+)\s+loose\s+[a-z]+", re.DOTALL)
_GROUPS_RE = re.compile(r"into\s+(\d+)\s+(?:equal\s+)?groups")


def _parse_easy(text: str):
    total_m = _TOTAL_RE.search(text)
    assert total_m is not None, f"oracle: no total match in {text!r}"
    groups_m = _EASY_GROUPS_RE.search(text)
    assert groups_m is not None, f"oracle: no groups match in {text!r}"
    return int(total_m.group(1)), int(groups_m.group(1))


def _parse_bundle_loose(text: str):
    bl = _BUNDLES_LOOSE_RE.search(text)
    assert bl is not None, f"oracle: no bundles/loose match in {text!r}"
    bundles, loose = int(bl.group(1)), int(bl.group(2))
    g = _GROUPS_RE.search(text)
    assert g is not None, f"oracle: no groups match in {text!r}"
    groups = int(g.group(1))
    return bundles, loose, groups


def _all(tier, seeds=None):
    seeds = seeds if seeds is not None else range(N_SEEDS)
    return [make(TOPIC, seed=s, difficulty=tier) for s in seeds]


def test_oracle_matches_answer_text_easy():
    mismatches = []
    for q in _all("easy"):
        total, groups = _parse_easy(q.text)
        expected = total // groups
        if str(expected) != q.answer_text:
            mismatches.append((q.seed, q.text, expected, q.answer_text))
    assert mismatches == [], mismatches


def test_oracle_matches_answer_text_medium():
    mismatches = []
    for q in _all("medium"):
        bundles, loose, groups = _parse_bundle_loose(q.text)
        expected = (bundles // groups) * 10 + (loose // groups)
        if str(expected) != q.answer_text:
            mismatches.append((q.seed, q.text, expected, q.answer_text))
    assert mismatches == [], mismatches


def _expected_hard(bundles, loose, groups):
    r = bundles % groups
    if r == 0:
        return (bundles // groups) * 10 + (loose // groups)
    if r == 1:
        return ((bundles - 1) // groups) * 10 + (10 + loose) // groups
    raise AssertionError(f"bundles {bundles} % groups {groups} == {r}, expected 0 or 1")


def test_oracle_matches_answer_text_hard():
    mismatches = []
    for q in _all("hard"):
        bundles, loose, groups = _parse_bundle_loose(q.text)
        expected = _expected_hard(bundles, loose, groups)
        if str(expected) != q.answer_text:
            mismatches.append((q.seed, q.text, expected, q.answer_text))
    assert mismatches == [], mismatches


def test_hard_mixes_evenly_dividing_and_leftover_draws():
    """Cross-check against the build test of the same name: recomputed here
    independently via the oracle's own parse, so the two never quietly
    disagree about what fraction of hard draws need unbundling."""
    even = 0
    seeds = _all("hard", seeds=range(200))
    for q in seeds:
        bundles, loose, groups = _parse_bundle_loose(q.text)
        if bundles % groups == 0:
            even += 1
    share = even / len(seeds)
    assert 0.20 <= share <= 0.45, f"hard evenly-dividing share {share} ({even}/{len(seeds)})"


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_every_distractor_disagrees_with_the_oracle(tier):
    for q in _all(tier, seeds=range(60)):
        if tier == "easy":
            total, groups = _parse_easy(q.text)
            expected = total // groups
        else:
            bundles, loose, groups = _parse_bundle_loose(q.text)
            if tier == "medium":
                expected = (bundles // groups) * 10 + (loose // groups)
            else:
                expected = _expected_hard(bundles, loose, groups)
        for d in q.distractors:
            assert int(d) != expected, f"{tier} seed {q.seed}: distractor {d!r} matches the oracle"
