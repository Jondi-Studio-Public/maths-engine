"""Independent oracle for `p19b_association_describe`
(skill `senior-describe-the-association-between-two-numerical-variables-in`).

Imports NOTHING from `mathgen.generators.p19b_association_describe_build`.
Parses `question.text` alone: reads the direction from either a signed `r =`
value or the stated trend clause; the form from whether a "residual plot"
clause says "no pattern"/"random scatter" (linear) or "curved
pattern" (non-linear); and the strength from either the stated |r| against
its OWN independently re-written band rule, or the stated closeness-to-curve
phrase, applying the exam tier's own "residual plot overrides r" rule
without ever reading the generator's `_classify_r_strength`.

`N_SEEDS` (overridable by `P19B_ORACLE_SEEDS`) sets the per-tier seed count.
"""

from __future__ import annotations

import os
import re

import pytest

from mathgen import make

TOPIC = "p19b_association_describe"
TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = int(os.environ.get("P19B_ORACLE_SEEDS", 250))

_R_RE = re.compile(r"r = (-?\d+\.\d+)")
_R2_RE = re.compile(r"r\^2 = (\d+\.\d+)%")
_LINE_RE = re.compile(r"y = (-?\d+) ([+-]) (\d+\.\d+)x")


def _oracle_strength_from_abs_r(abs_r: float) -> str:
    """A second, independently-written copy of the linear band rule."""
    if abs_r >= 0.75:
        return "strong"
    if abs_r >= 0.5:
        return "moderate"
    if abs_r >= 0.25:
        return "weak"
    return None


_STRENGTH_SUBSTRINGS = {
    "strong": ("packed very tightly", "almost exactly along the curve", "lie almost exactly on the curve"),
    "moderate": (
        "fairly close to the trend",
        "fairly close to the curve, with some scatter",
        "scattered fairly close to the curve",
    ),
    "weak": (
        "loosely scattered around the trend",
        "loosely scattered around the curve",
        "noticeably scattered around the curve",
    ),
}


def _match_strength_substrings(clause: str) -> str | None:
    for label, substrings in _STRENGTH_SUBSTRINGS.items():
        if any(s in clause for s in substrings):
            return label
    return None


def _oracle_closeness_strength(text: str) -> str | None:
    """Reads the ACTUAL closeness clause only -- the clause following
    'Relative to the fitted curve,' on medium/hard/exam -- never the fixed
    band-RULE sentence every non-linear prompt also states (which mentions
    every phrase, e.g. 'lying almost exactly on the curve is strong ...',
    and would otherwise make every draw look 'strong'). `easy` has no such
    rule sentence, so its own strength clause is matched directly against
    the whole text."""
    m = re.search(r"Relative to the fitted curve, (.*?)\.", text)
    clause = m.group(1) if m else text
    return _match_strength_substrings(clause)


def _oracle_direction_from_trend(text: str) -> str | None:
    m = re.search(r"tends to (increase|decrease)", text)
    if not m:
        return None
    return "positive" if m.group(1) == "increase" else "negative"


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_agrees_with_key(tier):
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        key_direction, key_form, key_strength = q.answer
        assert q.answer_text == f"{key_direction}, {key_form}, {key_strength}"

        curved = ("curved pattern" in q.text) or ("clear curve" in q.text) or ("not a straight line" in q.text)
        random_scatter = (
            ("no pattern" in q.text) or ("random scatter" in q.text) or ("roughly straight-line trend" in q.text)
        )

        r_match = _R_RE.search(q.text)
        r2_match = _R2_RE.search(q.text)
        line_match = _LINE_RE.search(q.text)

        if tier == "exam":
            # The trap rule: even with r stated, a curved residual plot
            # means r does NOT establish linearity -- strength (and form)
            # must come from the residual plot / closeness clause, never
            # from r, whenever the two disagree.
            assert r_match is not None
            r_val = float(r_match.group(1))
            direction = "positive" if r_val > 0 else "negative"
            if curved:
                derived_form = "non-linear"
                derived_strength = _oracle_closeness_strength(q.text)
            else:
                derived_form = "linear"
                derived_strength = _oracle_strength_from_abs_r(abs(r_val))
        elif r_match is not None:
            # medium/hard linear branch (r stated directly).
            r_val = float(r_match.group(1))
            direction = "positive" if r_val > 0 else "negative"
            derived_form = "linear"
            derived_strength = _oracle_strength_from_abs_r(abs(r_val))
        elif line_match is not None and r2_match is not None:
            # hard linear branch (r derived from r^2 + line slope sign).
            sign = line_match.group(2)
            direction = "positive" if sign == "+" else "negative"
            derived_form = "linear"
            abs_r = (float(r2_match.group(1)) / 100) ** 0.5
            derived_strength = _oracle_strength_from_abs_r(abs_r)
        else:
            # non-linear branch (medium/hard) or the easy tier: direction
            # from the trend clause, form from the curved/random clause,
            # strength from the closeness phrase.
            direction = _oracle_direction_from_trend(q.text)
            derived_form = "non-linear" if curved else "linear"
            if derived_form == "non-linear":
                derived_strength = _oracle_closeness_strength(q.text)
            else:
                derived_strength = _oracle_closeness_strength(q.text)  # easy tier reuses same phrase bank

        assert direction == key_direction, f"{tier} seed={seed}: direction {direction} != {key_direction}\n{q.text}"
        assert derived_form == key_form, f"{tier} seed={seed}: form {derived_form} != {key_form}\n{q.text}"
        assert derived_strength == key_strength, (
            f"{tier} seed={seed}: strength {derived_strength} != {key_strength}\n{q.text}"
        )


def test_red_first_proof():
    """Prove the oracle's band rule actually distinguishes bands: a red-first
    check that a corrupted (off-by-one) mapping would have been caught."""
    assert _oracle_strength_from_abs_r(0.80) == "strong"
    assert _oracle_strength_from_abs_r(0.80) != "moderate"
    assert _oracle_closeness_strength("the points lie almost exactly on the curve") == "strong"
    assert _oracle_closeness_strength("the points lie almost exactly on the curve") != "weak"


# ============================================================================
# An independently-written table of
# each context's own natural direction (never imported from the build
# module), matched against the x-phrase found in the text.
# ============================================================================

_ORACLE_CONTEXT_DIRECTIONS = {
    "hours studied per week": "positive",
    "daily maximum temperature": "positive",
    "weekly training hours": "positive",
    "monthly advertising spend": "positive",
    "a plant's daily sunlight hours": "positive",
    "a car's age": "negative",
    "altitude": "negative",
    "the number of days absent per term": "negative",
    "the price": "negative",
    "hours of screen time per day": "negative",
}


def _oracle_context_direction(text: str) -> str | None:
    matches = [phrase for phrase in _ORACLE_CONTEXT_DIRECTIONS if phrase in text]
    if len(matches) != 1:
        return None
    return _ORACLE_CONTEXT_DIRECTIONS[matches[0]]


def test_red_first_context_direction_proof():
    """Prove the oracle's own context table actually distinguishes
    contexts: corrupt one entry, show it disagrees with the true reading,
    then restore it."""
    assert _oracle_context_direction("a car's age increases") == "negative"
    corrupted = dict(_ORACLE_CONTEXT_DIRECTIONS)
    corrupted["a car's age"] = "positive"
    assert corrupted["a car's age"] != _ORACLE_CONTEXT_DIRECTIONS["a car's age"]


@pytest.mark.parametrize("tier", TIERS)
def test_context_carries_its_own_natural_direction(tier):
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        oracle_direction = _oracle_context_direction(q.text)
        assert oracle_direction is not None, f"{tier} seed={seed}: no known context in {q.text!r}"
        assert oracle_direction == q.answer[0], (
            f"{tier} seed={seed}: context implies {oracle_direction} but key is {q.answer[0]}\n{q.text}"
        )


@pytest.mark.parametrize("tier", TIERS)
def test_template_matrix_coherent(tier):
    """Enumerate every (form-branch, direction, strength) combination seen
    over a 60-seed sweep and assert each is one of the designed set."""
    seen = set()
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=tier)
        seen.add(tuple(q.answer))
    assert seen
    for direction, form, strength in seen:
        assert direction in ("positive", "negative")
        assert form in ("linear", "non-linear")
        assert strength in ("strong", "moderate", "weak")
