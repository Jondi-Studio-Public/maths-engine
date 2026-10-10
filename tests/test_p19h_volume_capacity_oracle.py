"""Re-derives the answer from the numbers stated in `question.text` alone, via a units table hand-written here --
does NOT import `mathgen.generators.p19h_volume_capacity_build` (no catalogue, no helper, no constant shared
with construction)."""

from __future__ import annotations

import re

import pytest

from mathgen.grading.typed import grade_number
from mathgen import make

VC1_TOPIC = "p19h_volcap_general_relationship"
VC2_TOPIC = "p19h_volcap_named_equivalence"
SEEDS = range(40)


# --- skill 1: general relationship, mL/L only, forward only -----------------

_VC1_CM3_RE = re.compile(r"(\d+) cm³")
_VC1_M3_RE = re.compile(r"(\d+) m³")
_VC1_HARD_DIMS_RE = re.compile(r"measures (\d+) cm by (\d+) cm by (\d+) cm")
_VC1_EXAM_DIMS_RE = re.compile(r"measures (\d+) m by (\d+) m by (\d+) m|is (\d+) m long, (\d+) m wide and (\d+) m deep")


def test_oracle_vc1_easy_cm3_to_mL_or_L():
    """Easy asks for mL (1 cm3 = 1 mL, numeral unchanged) or for L (1000 cm3 = 1 L); the target unit is read from
    the prompt."""
    checked = 0
    for seed in range(200):
        q = make(VC1_TOPIC, seed=seed, difficulty="easy")
        m = _VC1_CM3_RE.search(q.text)
        assert m, f"easy/{seed}: unexpected text: {q.text!r}"
        volume_cm3 = int(m.group(1))
        expected = volume_cm3 if "millilitres (mL)" in q.text else volume_cm3 // 1000
        assert q.answer_text == str(expected), (
            f"easy/{seed}: {q.text!r} -> {q.answer_text!r}, oracle expected {expected!r}"
        )
        checked += 1
    assert checked == 200


def test_oracle_vc1_medium_m3_to_L_derives_x1000():
    checked = 0
    for seed in SEEDS:
        q = make(VC1_TOPIC, seed=seed, difficulty="medium")
        m = _VC1_M3_RE.search(q.text)
        assert m, f"medium/{seed}: unexpected text: {q.text!r}"
        volume_m3 = int(m.group(1))
        # Derive independently via the linear ladder: 1 m^3 = 100^3 cm^3 =
        # 1,000,000 cm^3 = 1,000,000 mL, then the ordinary mL->L ladder
        # (/1000) -- a different route from the module's own cm3-pivot chain.
        volume_cm3 = volume_m3 * (100**3)
        volume_mL = volume_cm3  # 1 cm^3 = 1 mL
        expected = volume_mL // 1000
        assert q.answer_text == str(expected), (
            f"medium/{seed}: {q.text!r} -> {q.answer_text!r}, oracle expected {expected!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_oracle_vc1_hard_computes_volume_then_1to1():
    checked = 0
    for seed in SEEDS:
        q = make(VC1_TOPIC, seed=seed, difficulty="hard")
        m = _VC1_HARD_DIMS_RE.search(q.text)
        assert m, f"hard/{seed}: unexpected text: {q.text!r}"
        a, b, c = (int(g) for g in m.groups())
        expected = a * b * c
        assert q.answer_text == str(expected), (
            f"hard/{seed}: {q.text!r} -> {q.answer_text!r}, oracle expected {expected!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_oracle_vc1_exam_computes_volume_then_derives_x1000():
    checked = 0
    for seed in SEEDS:
        q = make(VC1_TOPIC, seed=seed, difficulty="exam")
        m = _VC1_EXAM_DIMS_RE.search(q.text)
        assert m, f"exam/{seed}: unexpected text: {q.text!r}"
        groups = [g for g in m.groups() if g is not None]
        assert len(groups) == 3, groups
        l, w, h = (int(g) for g in groups)
        volume_m3 = l * w * h
        volume_cm3 = volume_m3 * (100**3)
        expected = volume_cm3 // 1000  # 1 cm^3 = 1 mL, then mL -> L (/1000)
        assert q.answer_text == str(expected), (
            f"exam/{seed}: {q.text!r} -> {q.answer_text!r}, oracle expected {expected!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


# --- skill 2: named equivalence, mL/kL only, either direction ---------------

_VC2_EASY_RE = re.compile(r"(\d+) cm³")
# medium: TWO separately-stated quantities (one m³, one kL),
# summed -- forward states the m³ figure first, then the kL figure; reverse
# states the kL figure first, then the m³ figure.
_VC2_MEDIUM_FORWARD_RE = re.compile(r"(\d+) m³.*?(\d+) kL.*capacity in kilolitres")
_VC2_MEDIUM_REVERSE_RE = re.compile(r"(\d+) kL.*?(\d+) m³.*volume in cubic metres")
_VC2_HARD_FORWARD_RE = re.compile(r"holds (\d+) cm³.*capacity in kilolitres")
_VC2_HARD_REVERSE_RE = re.compile(r"capacity of (\d+) kL.*volume in cubic centimetres")
_VC2_HARD_TO_ML_RE = re.compile(r"holds (\d+) m³.*capacity in millilitres")
_VC2_EXAM_RE = re.compile(
    r"Container A holds (\d+) cm³.*Container B holds (\d+) m³.*combined capacity in (millilitres|kilolitres)"
)


def test_oracle_vc2_easy_named_equivalence_numeral_unchanged():
    """Easy is cm3 -> mL or m3 -> kL (both exact 1:1, fact stated)."""
    checked = 0
    for seed in range(200):
        q = make(VC2_TOPIC, seed=seed, difficulty="easy")
        m = _VC2_EASY_RE.search(q.text) or re.search(r"(\d+) m³", q.text)
        assert m, f"easy/{seed}: unexpected text: {q.text!r}"
        value = int(m.group(1))
        assert q.answer_text == str(value), f"easy/{seed}: {q.text!r} -> {q.answer_text!r}, oracle expected {value!r}"
        checked += 1
    assert checked == 200


def test_oracle_vc2_medium_m3_kL_either_direction_both_represented():
    """Medium states TWO separately-given quantities, one in m^3 and one in kL, and the answer is their sum (1
    m^3 = 1 kL is numeral-preserving, but two GIVEN quantities can never both equal their own sum, so this is
    no longer 200/200 numeral-preserving) -- this oracle also confirms both directions are actually reachable
    over the seed range."""
    forward = reverse = 0
    checked = 0
    for seed in SEEDS:
        q = make(VC2_TOPIC, seed=seed, difficulty="medium")
        fm = _VC2_MEDIUM_FORWARD_RE.search(q.text)
        rm = _VC2_MEDIUM_REVERSE_RE.search(q.text)
        assert fm or rm, f"medium/{seed}: unexpected text: {q.text!r}"
        v1, v2 = (int(g) for g in (fm or rm).groups())
        expected = v1 + v2
        assert q.answer_text == str(expected), (
            f"medium/{seed}: {q.text!r} -> {q.answer_text!r}, oracle expected {expected!r}"
        )
        forward += bool(fm)
        reverse += bool(rm)
        checked += 1
    assert checked == len(SEEDS)
    assert forward > 0 and reverse > 0, f"forward={forward}, reverse={reverse}"


def test_oracle_vc2_hard_chains_both_named_facts_and_the_metric_ladder():
    forward = reverse = 0
    checked = 0
    for seed in SEEDS:
        q = make(VC2_TOPIC, seed=seed, difficulty="hard")
        fm = _VC2_HARD_FORWARD_RE.search(q.text)
        rm = _VC2_HARD_REVERSE_RE.search(q.text)
        mm = _VC2_HARD_TO_ML_RE.search(q.text)
        assert fm or rm or mm, f"hard/{seed}: unexpected text: {q.text!r}"
        if mm:
            # m3 -> kL (given fact) -> mL (the ordinary ladder, x1,000,000)
            expected = int(mm.group(1)) * 1_000_000
            forward += 1
        elif fm:
            cm3_value = int(fm.group(1))
            # cm3 -> mL (1 cm3 = 1 mL) -> kL (the ordinary ladder, /1,000,000)
            expected = cm3_value // 1_000_000
            forward += 1
        else:
            kl_value = int(rm.group(1))
            # kL -> m3 (1 m3 = 1 kL) -> cm3 (the ordinary ladder, x1,000,000)
            expected = kl_value * 1_000_000
            reverse += 1
        assert q.answer_text == str(expected), (
            f"hard/{seed}: {q.text!r} -> {q.answer_text!r}, oracle expected {expected!r}"
        )
        checked += 1
    assert checked == len(SEEDS)
    assert forward > 0 and reverse > 0, f"forward={forward}, reverse={reverse}"


def test_oracle_vc2_exam_two_containers_summed_to_target():
    ml_target = kl_target = 0
    checked = 0
    for seed in SEEDS:
        q = make(VC2_TOPIC, seed=seed, difficulty="exam")
        m = _VC2_EXAM_RE.search(q.text)
        assert m, f"exam/{seed}: unexpected text: {q.text!r}"
        a, b, target = m.group(1), m.group(2), m.group(3)
        a, b = int(a), int(b)
        if target == "millilitres":
            expected = a + b * 1_000_000  # A: 1cm3=1mL; B: 1m3=1kL=1,000,000mL
            ml_target += 1
        else:
            expected = (a // 1_000_000) + b  # A: 1cm3=1mL, 1kL=1,000,000mL; B: 1m3=1kL
            kl_target += 1
        assert q.answer_text == str(expected), (
            f"exam/{seed}: {q.text!r} -> {q.answer_text!r}, oracle expected {expected!r}"
        )
        checked += 1
    assert checked == len(SEEDS)
    assert ml_target > 0 and kl_target > 0, f"mL={ml_target}, kL={kl_target}"


# --- negative control: the OLD (pre-fix) defect must now be rejected -------


def test_oracle_vc2_medium_negative_control_rejects_old_numeral_preserving_answer():
    """Corrupt a real generated medium-tier question's answer to that OLD wrong value (the first stated quantity
    alone) and confirm both the independent oracle recompute and the real grader reject it -- proving the fix,
    and this oracle, actually discriminate the old defect from the new construction."""
    checked = 0
    for seed in SEEDS:
        q = make(VC2_TOPIC, seed=seed, difficulty="medium")
        fm = _VC2_MEDIUM_FORWARD_RE.search(q.text)
        rm = _VC2_MEDIUM_REVERSE_RE.search(q.text)
        assert fm or rm, f"medium/{seed}: unexpected text: {q.text!r}"
        v1, _v2 = (int(g) for g in (fm or rm).groups())
        old_defective_answer = str(v1)  # what the old generator would have returned

        # The independent oracle's own recompute must not agree with the old value.
        assert q.answer_text != old_defective_answer, (
            f"medium/{seed}: fix did not change the answer away from the old "
            f"numeral-preserving value {old_defective_answer!r}"
        )
        # And the real grader must reject the old value as a submitted answer.
        assert not grade_number(q, old_defective_answer).correct, (
            f"medium/{seed}: real grader incorrectly accepted the old "
            f"numeral-preserving answer {old_defective_answer!r}"
        )
        checked += 1
    assert checked == len(SEEDS)
