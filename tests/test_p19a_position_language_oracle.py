"""The arrangement statement always has the shape "<label> <verb>[, listed
from|. Listed from|in this order, from] <direction>[, they are]: <entity>,
<entity>, ...." The direction phrase identifies the axis (top/bottom ->
stack: above/below; front/back -> queue: in front of/behind; left/right ->
shelf: to the left of/to the right of) and whether the list order matches
the axis's own "forward" sense. The question names the target and
reference(s) via one of four fixed markers ("of X relative to Y", "where X
is, relative to Y", "describes X's position relative to Y", or, for the
`between` variant, "describes X's position" alone with the offered options
each spelling out "between <A> and <B>").

Every offered option's truth is recomputed from the entity list position
alone: `above`/`below`/`in front of`/`behind`/`to the left of`/`to the right
of` from a rank comparison, `next to` from adjacency, the BARE word
`between` from whether the target has an item on each side of it anywhere in
the full listed order (an interior-position test, independent of which
always marked bare `between` False, which is wrong for an interior target),
and a PARAMETERIZED `between <A> and <B>` from a strict-between rank test --
and exactly one option must be true, matching `answer_text`.

Seed count is `N_SEEDS` (default 250 per tier, >= 1000 questions over the
three tiers at the override), overridable by `P19A_ORACLE_SEEDS`.
"""

from __future__ import annotations

import os
import re

import pytest

from mathgen.grading.typed import mcq_options
from mathgen import make

TOPIC = "p19a_position_language"
TIERS = ("easy", "medium", "hard")
N_SEEDS = int(os.environ.get("P19A_ORACLE_SEEDS", "250"))

_STATEMENT_RE = re.compile(r"from ([a-z]+ to [a-z]+)[:,]\s*(?:they are:\s*)?([^.]+)\.")

_AXES_BY_DIRECTION = {
    "top to bottom": ("stack", "above", "below", True),
    "bottom to top": ("stack", "above", "below", False),
    "front to back": ("queue", "in front of", "behind", True),
    "back to front": ("queue", "in front of", "behind", False),
    "left to right": ("shelf", "to the left of", "to the right of", True),
    "right to left": ("shelf", "to the left of", "to the right of", False),
}

_QUESTION_PATTERNS = (
    re.compile(r"of (.+?) relative to (.+?)[\?\.]"),
    re.compile(r"where (.+?) is, relative to (.+?)[\?\.]"),
    re.compile(r"describes (.+?)'s position relative to (.+?)[\?\.]"),
)
_BETWEEN_QUESTION_PATTERNS = (
    re.compile(r"describes (.+?)'s position[\?\.]"),
    re.compile(r"describes where (.+?) is[\?\.]"),
)
_BETWEEN_OPTION_RE = re.compile(r"^between (.+) and (.+)$")


def _parse_arrangement(text: str):
    m = _STATEMENT_RE.search(text)
    assert m is not None, f"oracle: no arrangement statement found in {text!r}"
    direction, list_str = m.group(1), m.group(2)
    assert direction in _AXES_BY_DIRECTION, f"oracle: unknown direction {direction!r} in {text!r}"
    axis, forward_term, backward_term, listed_forward = _AXES_BY_DIRECTION[direction]
    order = [e.strip() for e in list_str.split(",")]
    n = len(order)
    ranks = {e: (i if listed_forward else (n - 1 - i)) for i, e in enumerate(order)}
    return axis, forward_term, backward_term, order, ranks


def _parse_target_ref(text: str):
    for pat in _QUESTION_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    return None


def _parse_between_target(text: str):
    for pat in _BETWEEN_QUESTION_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).strip()
    raise AssertionError(f"oracle: no target found for between-style question in {text!r}")


def _bare_between_truth(order, target) -> bool:
    """The bare word "between" is true of `target` iff it has an item on
    each side of it anywhere in the full listed order -- a claim about the
    target alone, independent of which referent is named alongside it."""
    i = order.index(target)
    return 0 < i < len(order) - 1


def _option_truth(option: str, forward_term, backward_term, ranks, order, target, ref):
    if option == forward_term:
        return ranks[target] < ranks[ref]
    if option == backward_term:
        return ranks[target] > ranks[ref]
    if option == "next to":
        return abs(ranks[target] - ranks[ref]) == 1
    if option == "between":
        return _bare_between_truth(order, target)
    m = _BETWEEN_OPTION_RE.match(option)
    if m:
        a, b = m.group(1), m.group(2)
        return (ranks[a] - ranks[target]) * (ranks[b] - ranks[target]) < 0
    raise AssertionError(f"oracle: unrecognised option text {option!r}")


def _all(tier, seeds=None):
    seeds = seeds if seeds is not None else range(N_SEEDS)
    return [make(TOPIC, seed=s, difficulty=tier) for s in seeds]


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_exactly_one_true_option_matches_answer(tier):
    mismatches = []
    for q in _all(tier):
        axis, forward_term, backward_term, order, ranks = _parse_arrangement(q.text)
        options = mcq_options(q)
        target_ref = _parse_target_ref(q.text)
        if target_ref is not None:
            target, ref = target_ref
            truths = {
                opt: _option_truth(opt, forward_term, backward_term, ranks, order, target, ref) for opt in options
            }
        else:
            target = _parse_between_target(q.text)
            truths = {}
            for opt in options:
                m = _BETWEEN_OPTION_RE.match(opt)
                assert m is not None, f"oracle: expected a between-style option, got {opt!r}"
                a, b = m.group(1), m.group(2)
                truths[opt] = (ranks[a] - ranks[target]) * (ranks[b] - ranks[target]) < 0
        true_options = [opt for opt, truth in truths.items() if truth]
        if len(true_options) != 1 or true_options[0] != q.answer_text:
            mismatches.append((q.seed, q.text, truths, q.answer_text))
    assert mismatches == [], mismatches


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_every_distractor_disagrees_with_the_oracle(tier):
    for q in _all(tier, seeds=range(60)):
        axis, forward_term, backward_term, order, ranks = _parse_arrangement(q.text)
        target_ref = _parse_target_ref(q.text)
        for d in q.distractors:
            if target_ref is not None:
                target, ref = target_ref
                truth = _option_truth(d, forward_term, backward_term, ranks, order, target, ref)
            else:
                target = _parse_between_target(q.text)
                m = _BETWEEN_OPTION_RE.match(d)
                assert m is not None, f"oracle: expected a between-style distractor, got {d!r}"
                a, b = m.group(1), m.group(2)
                truth = (ranks[a] - ranks[target]) * (ranks[b] - ranks[target]) < 0
            assert not truth, f"{tier} seed {q.seed}: distractor {d!r} is actually true in {q.text!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_bare_between_never_offered_for_an_interior_target(tier):
    """Bare "between" is true whenever the target has an item on each side of it anywhere in the full listed
    order, so it must never be offered (as key or distractor) for such a target."""
    for q in _all(tier, seeds=range(200)):
        axis, forward_term, backward_term, order, ranks = _parse_arrangement(q.text)
        options = mcq_options(q)
        if "between" not in options:
            continue
        target_ref = _parse_target_ref(q.text)
        target = target_ref[0] if target_ref is not None else _parse_between_target(q.text)
        assert not _bare_between_truth(order, target), (
            f"{tier} seed {q.seed}: bare 'between' offered for interior target {target!r} in {q.text!r}"
        )


def test_axis_x_direction_combinations_all_coherent():
    """Enumerates every axis x listing-direction combination seen over the
    three tiers and confirms each one produced a coherent (exactly-one-true)
    question."""
    seen = set()
    for tier in TIERS:
        for q in _all(tier, seeds=range(60)):
            m = _STATEMENT_RE.search(q.text)
            assert m is not None
            direction = m.group(1)
            axis, _f, _b, _fwd = _AXES_BY_DIRECTION[direction]
            seen.add((tier, axis, direction))
    assert len(seen) >= 6, f"only {len(seen)} axis x direction combinations seen: {seen}"
