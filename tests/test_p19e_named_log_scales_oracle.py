"""Independent oracle for the `p19e_named_log_scales` questions: it re-derives each answer from the rendered
question text.

Parses the scale, its inputs and the stated precision straight from `question.text`, recomputes via
`math.log(x)/math.log(base)` (never `math.log10`/`math.log2`, the generator's own route), and compares to
`answer_text`.

`N_SEEDS` (per tier) overridable by `P19E_ORACLE_SEEDS`, default 250 so
>= 1000 questions are checked over the four tiers.
"""

import math
import os
import re

from mathgen.grading.typed import grade_number
from mathgen.skills import load_skills
from mathgen import make
import mathgen.generators.p19e_named_log_scales_build  # noqa: F401 -- registers the topic

N_SEEDS = int(os.environ.get("P19E_ORACLE_SEEDS", "250"))
TIERS = ["easy", "medium", "hard", "exam"]
SKILL_ID = "senior-interpret-and-use-logarithmic-scales-such-as-decibels"

_domains, _skills = load_skills()
_SKILL = next(s for s in _skills if s.id == SKILL_ID)
SOURCES = {src.difficulty: src for src in _SKILL.sources}


def _make(difficulty, seed):
    src = SOURCES[difficulty]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _oracle_log(x, base):
    return math.log(x) / math.log(base)


def _dp_from_text(text: str) -> int:
    if "2 decimal places" in text:
        return 2
    if "1 decimal place" in text:
        return 1
    return 0


_SCI_RE = r"(-?\d+(?:\.\d+)?) × 10\^(-?\d+)"


def _parse_sci(s: str) -> float:
    m = re.match(_SCI_RE, s)
    mantissa, exponent = float(m.group(1)), int(m.group(2))
    return mantissa * 10**exponent


def _oracle_value(text: str):
    """Return the oracle's independently recomputed scale value, decided by
    which formula's own name/symbol appears in the text."""
    if "L = 10 log10(I / I0)" in text or "decibel" in text:
        i0 = 1e-12
        i_matches = re.findall(rf"I(\d)? = {_SCI_RE}", text)
        by_label = {label: (float(m), int(e)) for (label, m, e) in i_matches}
        if "1" in by_label and "2" in by_label:
            i1 = by_label["1"][0] * 10 ** by_label["1"][1]
            i2 = by_label["2"][0] * 10 ** by_label["2"][1]
            return 10 * (_oracle_log(i2, 10) - _oracle_log(i1, 10))
        i_match = re.search(rf"I = {_SCI_RE}", text)
        if i_match:
            i = float(i_match.group(1)) * 10 ** int(i_match.group(2))
            return 10 * (_oracle_log(i, 10) - _oracle_log(i0, 10))
        mult_match = re.search(r"is (\d+) times the threshold intensity", text)
        if mult_match:
            return 10 * _oracle_log(int(mult_match.group(1)), 10)
        raise AssertionError(f"could not parse dB inputs from: {text!r}")

    if "M = log10(A / A0)" in text or "Richter" in text:
        a_matches = re.findall(r"A(\d)? = ([\d.]+)", text)
        by_label = {label: float(v) for (label, v) in a_matches}
        if "1" in by_label and "2" in by_label:
            a1, a2 = by_label["1"], by_label["2"]
            return _oracle_log(a2, 10) - _oracle_log(a1, 10)
        mm_match = re.findall(r"A = ([\d.]+) mm", text)
        a0_match = re.search(r"A0 = ([\d.]+) mm", text)
        if mm_match and a0_match:
            return _oracle_log(float(mm_match[0]) / float(a0_match.group(1)), 10)
        mult_match = re.search(r"A (?:=|is) (\d+) times (?:the reference amplitude|A0)", text) or re.search(
            r"A = (\d+) × A0", text
        )
        if mult_match:
            return _oracle_log(int(mult_match.group(1)), 10)
        raise AssertionError(f"could not parse Richter inputs from: {text!r}")

    if "n = log2(f / f0)" in text or "octave" in text:
        f0_match = re.search(r"f0 = (\d+) Hz", text)
        f_matches = re.findall(r"f(\d)? = ([\d.]+) Hz", text)
        by_label = {label: float(v) for (label, v) in f_matches if label}
        if "1" in by_label and "2" in by_label:
            return _oracle_log(by_label["2"], 2) - _oracle_log(by_label["1"], 2)
        unlabelled = [float(v) for (label, v) in f_matches if not label]
        if f0_match and unlabelled:
            return _oracle_log(unlabelled[0] / float(f0_match.group(1)), 2)
        ratio_match = re.search(r"is (\d+)/(\d+) times (?:a|the) reference frequency", text)
        if ratio_match:
            return _oracle_log(int(ratio_match.group(1)) / int(ratio_match.group(2)), 2)
        raise AssertionError(f"could not parse octave inputs from: {text!r}")

    if "pH = -log10[H+]" in text or "pH" in text:
        h_matches = re.findall(rf"\[H\+\](\d)? = {_SCI_RE}", text)
        by_label = {label: (float(m), int(e)) for (label, m, e) in h_matches}
        if "1" in by_label and "2" in by_label:
            m1, e1 = by_label["1"]
            m2, e2 = by_label["2"]
            h1 = m1 * 10**e1
            h2 = m2 * 10**e2
            return -_oracle_log(h2, 10) - (-_oracle_log(h1, 10))
        h_match = re.search(rf"\[H\+\] = {_SCI_RE}", text)
        if h_match:
            h = float(h_match.group(1)) * 10 ** int(h_match.group(2))
            return -_oracle_log(h, 10)
        mult_match = re.search(r"is (\d+) times that of a reference solution with pH = (\d+)", text)
        if mult_match:
            mult, ref_ph = int(mult_match.group(1)), int(mult_match.group(2))
            return ref_ph - _oracle_log(mult, 10)
        raise AssertionError(f"could not parse pH inputs from: {text!r}")

    raise AssertionError(f"could not identify a named scale in: {text!r}")


def test_oracle_red_first_on_corrupted_key():
    q = _make("easy", 0)
    dp = _dp_from_text(q.text)
    value = _oracle_value(q.text)
    expected = f"{value:.{dp}f}" if dp else str(round(value))
    assert expected == q.answer_text
    corrupted = float(q.answer_text) + 5.0
    assert abs(corrupted - value) > 0.5


def test_oracle_re_derives_the_key_over_many_seeds():
    checked = 0
    for tier in TIERS:
        for seed in range(N_SEEDS):
            q = _make(tier, seed)
            dp = _dp_from_text(q.text)
            value = _oracle_value(q.text)
            expected = f"{value:.{dp}f}" if dp else str(round(value))
            got = float(q.answer_text)
            assert abs(got - value) < 0.5 * 10**-dp + 1e-6, (tier, seed, q.text, got, value)
            checked += 1
    assert checked >= 1000


def test_oracle_grader_agrees():
    for tier in TIERS:
        for seed in range(60):
            q = _make(tier, seed)
            dp = _dp_from_text(q.text)
            value = _oracle_value(q.text)
            typed = f"{value:.{dp}f}" if dp else str(round(value))
            assert grade_number(q, typed).correct
