"""It does NOT import `mathgen.generators.p18a_line_graph_claims_build` -- confirmed by the import list below
(only `os`, `re`, `collections`, `mathgen.make`) -- and reads only `question.text`, `question.answer_text` and
`question.distractors`, exactly as a learner sees them.

Comparing option TRUTH structurally (direction + numeric magnitude / time
labels), rather than re-deriving the generator's exact phrase and string-
matching it, is deliberate: the generator formats a delta with its own unit
($, °C, "N people", ...) that this oracle does not need to reproduce
byte-for-byte to prove the CLAIM is right or wrong.

`N_SEEDS` (per tier) is overridable by `P18A_ORACLE_SEEDS` so a contributor can
sweep harder without editing the file.
"""

from __future__ import annotations

import os
import re
from collections import Counter

import pytest

from mathgen import make

TOPIC = "p18a_line_graph_claims"
TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = int(os.environ.get("P18A_ORACLE_SEEDS", "250"))

_CLAIM_RE = re.compile(r'A claim is made: "([^"]+)"')
_NUM_RE = re.compile(r"-?\d+")

_VALUE_CLAIM_RE = re.compile(r"^In (?P<t>.+?), .+ was \$?(?P<v>-?\d+)")
_CHANGE_CLAIM_RE = re.compile(r"^Between (?P<t1>.+?) and (?P<t2>.+?), ")
_OVERALL_CLAIM_RE = re.compile(r"^Overall, from (?P<t1>.+?) to (?P<t2>.+?), ")
_MONOTONIC_RE = re.compile(
    r"^.+ (?P<dir>rose every \S+|fell every \S+|stayed the same throughout),? from (?P<t1>.+?) to (?P<t2>.+?)\.$"
)


def _parse_data(text: str) -> dict[str, int]:
    """`{time_label: value}` parsed straight from the "(label: number unit)"
    list, in appearance order. `[^():]+` for the label stops at the first
    colon/paren, so it never crosses into a neighbouring pair. `\\$?` skips
    the optional currency sign so the number itself is captured cleanly."""
    data: dict[str, int] = {}
    for m in re.finditer(r"\(([^():]+):\s*\$?(-?\d+)", text):
        data[m.group(1).strip()] = int(m.group(2))
    return data


def _extract_delta_from_claim(claim: str) -> int:
    """ "... rose by $46 ..." -> 46, "... fell by 46 members ..." -> -46,
    "... stayed the same ..." -> 0. Searches anywhere in the claim rather
    than anchoring to end-of-string, so a trailing unit word or the
    "even though ..." exam clause never breaks the match."""
    if "stayed the same" in claim:
        return 0
    m = re.search(r"(rose|fell) by \$?(-?\d+)", claim)
    assert m, claim
    mag = int(m.group(2))
    return mag if m.group(1) == "rose" else -mag


def _parse_claim(claim: str, data: dict[str, int]):
    """Returns (claim_class, info) where `info` is a small dict the
    corresponding `_option_truth_*` function needs to judge any option."""
    m = _VALUE_CLAIM_RE.match(claim)
    if m:
        return "value", dict(t=m.group("t"), claimed=int(m.group("v")))

    m = _CHANGE_CLAIM_RE.match(claim) or _OVERALL_CLAIM_RE.match(claim)
    if m:
        return "change", dict(t1=m.group("t1"), t2=m.group("t2"), claimed_delta=_extract_delta_from_claim(claim))

    m = _MONOTONIC_RE.match(claim)
    if m:
        dir_text = m.group("dir")
        if dir_text.startswith("rose"):
            direction = "rose"
        elif dir_text.startswith("fell"):
            direction = "fell"
        else:
            direction = "same"
        return "monotonic", dict(direction=direction, t1=m.group("t1"), t2=m.group("t2"))

    raise AssertionError(f"claim did not match any of the 3 kernel claim classes: {claim!r}")


def _time_sort_key(label: str) -> int:
    """The exam tier displays the (label: value) pairs in the
    order they were filed, not in time order, so the monotonic branch below
    -- which needs the TRUE time-adjacent pairs -- must re-sort the parsed
    labels by their own time semantics rather than trust dict-insertion
    (i.e. appearance) order. A "Week N" label sorts on N; a bare year label
    (e.g. "2019") sorts on its own numeric value."""
    m = re.match(r"Week (\d+)$", label)
    if m:
        return int(m.group(1))
    return int(label)


def _diff_word(d: int) -> str:
    if d > 0:
        return "rose"
    if d < 0:
        return "fell"
    return "same"


def _option_truth(option: str, cls: str, info: dict, data: dict[str, int]) -> bool:
    labels = list(data.keys())

    if cls == "value":
        actual = data[info["t"]]
        if option == "Supported":
            return info["claimed"] == actual
        assert option.startswith("Not supported: it was actually ")
        stated = int(_NUM_RE.search(option[len("Not supported: it was actually ") :]).group())
        return stated == actual

    if cls == "change":
        actual_delta = data[info["t2"]] - data[info["t1"]]
        if option == "Supported":
            return info["claimed_delta"] == actual_delta
        assert option.startswith("Not supported: it ")
        option_delta = _extract_delta_from_claim(option)
        return option_delta == actual_delta

    # monotonic -- re-sort by TIME, never by the order the pairs
    # happen to appear in the text (the exam tier displays them out of
    # order; see `_time_sort_key`).
    labels = sorted(labels, key=_time_sort_key)
    diffs = [data[labels[i + 1]] - data[labels[i]] for i in range(len(labels) - 1)]
    if option == "Supported":
        return all(_diff_word(d) == info["direction"] for d in diffs)
    assert option.startswith("Not supported: it ")
    body = option[len("Not supported: it ") :]
    m = re.match(r"(rose|fell|stayed the same) between (.+) and (.+)$", body)
    assert m, body
    word = "same" if m.group(1) == "stayed the same" else m.group(1)
    a, b = m.group(2), m.group(3)
    idx = labels.index(a)
    assert labels[idx + 1] == b, (a, b, labels)
    return _diff_word(diffs[idx]) == word


def _all_questions():
    out = []
    for tier in TIERS:
        for seed in range(N_SEEDS):
            out.append((tier, make(TOPIC, seed=seed, difficulty=tier)))
    return out


QUESTIONS = _all_questions()


def test_swept_enough_questions():
    assert len(QUESTIONS) >= 1000
    by_tier = Counter(tier for tier, _ in QUESTIONS)
    for tier in TIERS:
        assert by_tier[tier] >= 250, by_tier


def test_answer_text_is_the_unique_true_option():
    failures = []
    for tier, q in QUESTIONS:
        data = _parse_data(q.text)
        claim = _CLAIM_RE.search(q.text)
        assert claim, (tier, q.text)
        cls, info = _parse_claim(claim.group(1), data)
        options = [q.answer_text, *q.distractors]
        truth = [_option_truth(o, cls, info, data) for o in options]
        if sum(truth) != 1 or not truth[0]:
            failures.append((tier, q.text, options, truth))
    assert not failures, f"{len(failures)} failure(s), first: {failures[:3]}"


def test_claim_class_x_key_class_matrix_report():
    """Regression guard for a previously fixed defect."""
    combos = Counter()
    for tier, q in QUESTIONS:
        data = _parse_data(q.text)
        claim = _CLAIM_RE.search(q.text).group(1)
        cls, _info = _parse_claim(claim, data)
        key_class = "Supported" if q.answer_text == "Supported" else "Not supported"
        combos[(cls, key_class)] += 1
    print(f"\nclaim-class x key-class combinations: {dict(sorted(combos.items()))}")
    for cls in ("value", "change", "monotonic"):
        assert (cls, "Supported") in combos, f"{cls}/Supported never seen"
        assert (cls, "Not supported") in combos, f"{cls}/Not supported never seen"


@pytest.mark.parametrize("tier", TIERS)
def test_each_tier_reaches_its_own_claim_classes(tier):
    seen = set()
    for t, q in QUESTIONS:
        if t != tier:
            continue
        data = _parse_data(q.text)
        claim = _CLAIM_RE.search(q.text).group(1)
        cls, _info = _parse_claim(claim, data)
        seen.add(cls)
    expected = {
        "easy": {"value"},
        "medium": {"change"},
        "hard": {"monotonic"},
        "exam": {"change", "monotonic"},
    }[tier]
    assert seen == expected, (tier, seen, expected)
