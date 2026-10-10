"""Re-derives the correct answer from `question.text` alone via this test's own rules (hand-written marker lists
for the quantity and for the reporting context that settles km/h vs m/s) -- does NOT import the generator's
pools, subject tables or scenarios; only `_KEYS` is imported, purely to validate the answer space, never to
drive the oracle's own lookup."""

from __future__ import annotations

import re

from mathgen import make
from mathgen.generators.p19h_rate_usage_build import _KEYS

TOPIC = "p19h_rate_usage_classification"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(300)

# Public-facing / road / TV / gym / news reporting contexts -> km/h.
_KMH_MARKERS = (
    "speedometer",
    "speed graphic",
    "on-screen",
    "timing graphic",
    "scoreboard",
    "newspaper",
    "general public",
    "general readers",
    "commentator",
    "police media release",
    "speed-limit sign",
    "electronic speed sign",
    "information board",
    "timetable",
    "onboard speed display",
    "on-board trip display",
    "bike computer",
    "console",
    "display panel",
    "summary screen",
    "weather bureau",
    "local news",
    "storm warning",
    "brochure",
    "documentary",
    "narration",
    "zoo's",
    "magazine",
    "sports website",
)
# SI-unit scientific / engineering contexts -> m/s.
_MS_MARKERS_RE = re.compile(
    r"\bSI\b|physics|journal|engineering|wind-tunnel|meteorology research|\blab\b|lab's|zoology"
)


def _speed_key(text: str) -> str:
    km = [m for m in _KMH_MARKERS if m in text]
    ms = _MS_MARKERS_RE.search(text)
    assert bool(km) != bool(ms), f"speed prompt must name exactly one convention: {text!r} km={km} ms={ms}"
    return "km/h" if km else "m/s"


def _classify(clause: str) -> str:
    """Key for ONE clause / phrase."""
    if "speed" in clause or "was moving" in clause or "radar gun" in clause:
        return _speed_key(clause)
    if "respirat" in clause or "breath" in clause:
        return "breaths/minute"
    if "heart" in clause or "pulse" in clause:
        return "beats/minute"
    if "flow" in clause or "water" in clause or "bucket" in clause:
        return "L/minute"
    raise AssertionError(f"unclassifiable clause: {clause!r}")


def _oracle_single_clause(text: str) -> str:
    m = re.fullmatch(r"Which unit is conventionally used to report (.+)\?", text)
    assert m, text
    return _classify(m.group(1))


def _oracle_hard(text: str) -> str:
    m = re.search(
        r"is reported in two ways: (.+?), and separately (.+?)\. "
        r"Considering only the (first|second) of these two",
        text,
    )
    assert m, text
    clause_a, clause_b, which = m.groups()
    return _classify(clause_a if which == "first" else clause_b)


# Regexes for the three exam question templates, each capturing the NAMED
# source ({device}) the item asks about.
_EXAM_DEVICE_PATTERNS = (
    r"Considering only what (.+?) reports, which unit",
    r"Focusing only on the reading given by (.+?), which unit",
    r"Taking only the figure (.+?) gives, which unit",
)


def _oracle_exam(text: str) -> str:
    device = None
    for pattern in _EXAM_DEVICE_PATTERNS:
        m = re.search(pattern, text)
        if m:
            device = m.group(1)
            break
    assert device is not None, text

    # Body = "<intro> is written up with three figures: c1; c2; c3". Neither
    # the intro nor any clause contains a period or a semicolon.
    body, _question = text.split(". ", 1)
    _intro, clauses_str = body.split(": ", 1)
    clauses = [c.strip() for c in clauses_str.split("; ")]
    assert len(clauses) == 3, (clauses, text)

    matches = [c for c in clauses if device in c]
    assert len(matches) == 1, (device, clauses, text)
    # The three clauses of an exam item are one of each of: km/h, m/s, and a
    # non-speed key (the oracle checks that shape independently).
    kinds = sorted(_classify(c) for c in clauses)
    assert "km/h" in kinds and "m/s" in kinds, (kinds, text)
    assert sum(1 for k in kinds if k in ("km/h", "m/s")) == 2, (kinds, text)
    return _classify(matches[0])


def _oracle(tier: str, text: str) -> str:
    if tier == "hard":
        return _oracle_hard(text)
    if tier == "exam":
        return _oracle_exam(text)
    return _oracle_single_clause(text)


def test_oracle_easy_medium():
    checked = 0
    for tier in ("easy", "medium"):
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
            expected = _oracle_single_clause(q.text)
            assert expected == q.answer_text, (tier, seed, q.text, q.answer_text, expected)
            checked += 1
    assert checked == 2 * len(SEEDS)


def test_oracle_hard():
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="hard", tier="hard")
        expected = _oracle_hard(q.text)
        assert expected == q.answer_text, (seed, q.text, q.answer_text, expected)
        checked += 1
    assert checked == len(SEEDS)


def test_oracle_exam():
    """exam: three named clauses, resolved by matching the named source in
    the question to its own clause -- never an ordinal position."""
    checked = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty="exam", tier="exam")
        expected = _oracle_exam(q.text)
        assert expected == q.answer_text, (seed, q.text, q.answer_text, expected)
        checked += 1
    assert checked == len(SEEDS)


def test_oracle_1000_draws_per_tier_all_speed_prompts_name_one_convention():
    """1000 seeds per tier; the oracle's exactly-one-context assertion means every speed prompt names a reporting
    context that settles the unit (no key rides on cue words alone)."""
    for tier in TIERS:
        for seed in range(1000):
            q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
            assert _oracle(tier, q.text) == q.answer_text, (tier, seed, q.text)


def test_enumerates_every_activity_option_set_combination_coherent():
    total_checked = 0
    seen_shapes = set()
    for tier in TIERS:
        for seed in SEEDS:
            q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
            expected = _oracle(tier, q.text)
            assert expected == q.answer_text, (tier, seed, q.text, q.answer_text)
            assert expected in _KEYS
            options = set(q.distractors) | {q.answer_text}
            assert 3 <= len(options) <= 5, (tier, seed, options)
            seen_shapes.add(tier)
            total_checked += 1
    assert total_checked >= 1000
    assert seen_shapes == set(TIERS)


# ===========================================================================
# Negative control: the oracle is not vacuously true -- a prompt with its
# context marker swapped flips the oracle's answer, and a corrupted answer is
# rejected.
# ===========================================================================


def test_oracle_rejects_a_corrupted_answer_to_a_different_key():
    checked = 0
    for tier in TIERS:
        for seed in range(20):
            q = make(TOPIC, seed=seed, difficulty=tier, tier=tier)
            expected = _oracle(tier, q.text)
            assert expected == q.answer_text, (tier, seed, q.text)
            corrupted_answer = q.distractors[0]
            assert corrupted_answer != expected, (tier, seed, q.text, q.distractors)
            checked += 1
    assert checked == 4 * 20


def test_oracle_context_swap_flips_the_speed_key():
    assert _classify("the speed of a cricket ball on the television broadcast's speed graphic") == "km/h"
    assert _classify("the speed of a cricket ball in a physics-class problem, using SI units") == "m/s"
