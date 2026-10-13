"""`sequence_term_from_rule`: easy/medium/hard source of `generate-sequence-from-rule`.

Each answer is recomputed here from the rendered prompt alone (explicit formula, or iteration of the recursion),
never from the generator's own helpers.
"""

from __future__ import annotations

import re

import pytest
from sympy import Symbol, sympify

from mathgen import make
from mathgen.grading import grade
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects

TOPIC = "sequence_term_from_rule"
TIERS = ("easy", "medium", "hard")
SEEDS = range(80)

_EXPLICIT = re.compile(
    r"A sequence has the rule t\(n\) = (.+?), where n = 1 gives the first term\. "
    r"Find (?:t\((\d+)\)|the (\d+)(?:st|nd|rd|th) term)\."
)
_RECURSIVE = re.compile(
    r"A sequence is defined by t\(1\) = (-?\d+) and t\(n \+ 1\) = (.+?) for every n ≥ 1\. Find t\((\d+)\)\."
)
n, T = Symbol("n"), Symbol("T")


def _expr(rule: str, var: str):
    rule = rule.replace("^", "**")
    rule = re.sub(r"(\d)(n|t\(n\))", r"\1*\2", rule)
    return sympify(rule.replace("t(n)", "T"), locals={"n": n, "T": T}) if var == "T" else sympify(rule, locals={"n": n})


def derive(text: str) -> tuple[int, int, str]:
    """(answer, number of evaluation steps the learner chains, kind) from the prompt alone."""
    m = _EXPLICIT.fullmatch(text)
    if m:
        k = int(m.group(2) or m.group(3))
        return int(_expr(m.group(1), "n").subs(n, k)), 1, "explicit"
    m = _RECURSIVE.fullmatch(text)
    assert m, text
    value, rule, k = int(m.group(1)), _expr(m.group(2), "T"), int(m.group(3))
    for _ in range(k - 1):
        value = int(rule.subs(T, value))
    return value, k - 1, "recursive"


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


@pytest.mark.parametrize("tier", TIERS)
def test_deterministic_per_seed(tier):
    for seed in SEEDS:
        a, b = _q(tier, seed), _q(tier, seed)
        assert (a.text, a.answer_text, a.distractors, a.working) == (b.text, b.answer_text, b.distractors, b.working)


@pytest.mark.parametrize("tier", TIERS)
def test_answer_recomputed_from_the_prompt(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        value, _, kind = derive(q.text)
        assert q.answer == value and q.answer_text == str(value), (tier, seed, q.text)
        assert kind == ("recursive" if tier == "hard" else "explicit")
        assert q.answer_kind == "number"
        assert q.working and q.working[-1].endswith(str(value))


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_never_equal_the_answer_and_grade_wrong(tier):
    for seed in SEEDS:
        q = _q(tier, seed)
        assert grade("number", q, q.answer_text).correct
        assert len(q.distractors) >= 2, (tier, seed)
        assert len(set(q.distractors)) == len(q.distractors)
        for d in q.distractors:
            assert d != q.answer_text and int(d) != q.answer
            assert not grade("number", q, d).correct, (tier, seed, d, q.answer_text)


@pytest.mark.parametrize("tier", TIERS)
def test_text_passes_text_quality(tier):
    for seed in SEEDS:
        assert find_text_defects(_q(tier, seed), include_working=True) == []


def test_tags_are_catalogue_ids_and_misconception_values_match():
    from mathgen.answers import tagged_distractors

    for tier in TIERS:
        for seed in SEEDS:
            q = _q(tier, seed)
            tagged = tagged_distractors(q)
            assert tagged, (tier, seed)
            for d in tagged:
                assert d.misconception is None or d.misconception in CATALOGUE, d
                assert d.value != q.answer
            # A reply spelled differently from the option still diagnoses the tagged error.
            for d in tagged:
                if d.misconception:
                    assert grade("number", q, f"{d.value}.0").misconception == d.misconception


def test_hard_needs_more_steps_than_easy_and_medium():
    steps = {t: [derive(_q(t, s).text)[1] for s in SEEDS] for t in TIERS}
    assert set(steps["easy"]) == {1}
    assert min(steps["hard"]) >= 3 and set(steps["hard"]) <= {3, 4} and len(set(steps["hard"])) == 2
    assert min(steps["hard"]) > max(steps["easy"])


def test_medium_needs_order_of_operations():
    rules = [_EXPLICIT.fullmatch(_q("medium", s).text).group(1) for s in SEEDS]
    assert any("n^2" in r for r in rules) and any(re.fullmatch(r"-?\d+ - \d*n", r) for r in rules)
    assert all("n^2" in r or " - " in r for r in rules)


def test_easy_is_linear_and_hard_uses_varied_multipliers():
    for seed in SEEDS:
        assert "^" not in _EXPLICIT.fullmatch(_q("easy", seed).text).group(1)
    multipliers = {re.search(r"= (-?\d*)t\(n\)", _q("hard", s).text).group(1) for s in SEEDS}
    assert len(multipliers) >= 4


def test_easy_shows_positive_and_negative_constants():
    rules = [_EXPLICIT.fullmatch(_q("easy", seed).text).group(1) for seed in SEEDS]
    assert any(" + " in r for r in rules) and any(" - " in r for r in rules)


def test_phrasing_varies_in_medium():
    texts = [_q("medium", s).text for s in SEEDS]
    assert any("term." in t and "Find the" in t for t in texts) and any("Find t(" in t for t in texts)
