"""Independent oracle for f10-polygon-angle-sums.
Deliberately does NOT import anything from
``mathgen.generators.p13a_polygon_angle_sums_build`` -- it parses
``question.text`` with its own regexes, using its OWN independent
name -> n lookup table and its own independent count * value expansion, and
recomputes the missing angle as ``(n - 2) * 180 - sum(known_angles)`` from
those literal parsed values alone, then compares against
``question.answer_text``.
"""

import re
from types import SimpleNamespace

import pytest

from mathgen import make

TOPIC = "p13a_polygon_angle_sum"
SEEDS = range(60)

# Independent name -> n table: written fresh here, not copied from the
# generator's own `_NAME_TO_N`/`_CHECK_NAME_TO_N` dicts.
_ORACLE_NAME_TO_N = {
    "triangle": 3,
    "quadrilateral": 4,
    "pentagon": 5,
    "hexagon": 6,
    "heptagon": 7,
    "octagon": 8,
}

_NUM_RE = re.compile(r"\d+")


def _oracle_easy(text: str) -> int:
    m = re.search(r"A polygon has (\d+) sides\. (\d+) of its interior angles measure (.+?) degrees\.", text)
    assert m, f"could not parse easy-shape text: {text!r}"
    n, stated_count = int(m.group(1)), int(m.group(2))
    knowns = [int(x) for x in _NUM_RE.findall(m.group(3))]
    assert stated_count == len(knowns) == n - 1, (
        f"easy: stated count {stated_count} != listed {len(knowns)} != n-1 {n - 1}"
    )
    return (n - 2) * 180 - sum(knowns)


def _oracle_medium(text: str) -> int:
    m = re.search(
        r"A polygon has (\d+) sides\. (\d+) of the angles measure (\d+) degrees each, "
        r"and the rest measure (.+?) degrees\.",
        text,
    )
    assert m, f"could not parse medium-shape text: {text!r}"
    n, g, v = int(m.group(1)), int(m.group(2)), int(m.group(3))
    rest = [int(x) for x in _NUM_RE.findall(m.group(4))]
    assert len(rest) == n - 1 - g, f"medium: rest count {len(rest)} != n-1-g {n - 1 - g}"
    # Independent count * value expansion: sum the grouped value g separate times.
    group_total = 0
    for _ in range(g):
        group_total += v
    return (n - 2) * 180 - group_total - sum(rest)


def _oracle_hard(text: str) -> int:
    m = re.search(r"An? (\w+) has (\d+) known interior angles: (.+?) degrees, with one angle still unknown\.", text)
    assert m, f"could not parse hard-shape text: {text!r}"
    name, stated_count = m.group(1), int(m.group(2))
    assert name in _ORACLE_NAME_TO_N, f"unrecognised polygon name {name!r}"
    n = _ORACLE_NAME_TO_N[name]
    knowns = [int(x) for x in _NUM_RE.findall(m.group(3))]
    assert stated_count == len(knowns) == n - 1, (
        f"hard: stated count {stated_count} != listed {len(knowns)} != n-1 {n - 1}"
    )
    return (n - 2) * 180 - sum(knowns)


def _oracle_exam(text: str) -> int:
    m = re.search(
        r"An? (\w+) has (\d+) angles measuring (\d+) degrees each and the rest measuring (.+?)\.",
        text,
    )
    assert m, f"could not parse exam-shape text: {text!r}"
    name = m.group(1)
    assert name in _ORACLE_NAME_TO_N, f"unrecognised polygon name {name!r}"
    n = _ORACLE_NAME_TO_N[name]
    g, v = int(m.group(2)), int(m.group(3))
    rest = [int(x) for x in _NUM_RE.findall(m.group(4))]
    assert len(rest) == n - 1 - g, f"exam: rest count {len(rest)} != n-1-g {n - 1 - g}"
    group_total = 0
    for _ in range(g):
        group_total += v
    return (n - 2) * 180 - group_total - sum(rest)


_ORACLES = {"easy": _oracle_easy, "medium": _oracle_medium, "hard": _oracle_hard, "exam": _oracle_exam}


def _challenge(oracle, question) -> bool:
    """The ONE comparison every test below shares: does the oracle's own
    re-derivation from ``question.text`` disagree with the shipped
    ``question.answer_text``? Returns True on a challenge (mismatch)."""
    return oracle(question.text) != int(question.answer_text)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "exam"])
def test_oracle_agrees_with_the_shipped_answer(difficulty):
    oracle = _ORACLES[difficulty]
    checked = 0
    challenged = []
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        checked += 1
        if _challenge(oracle, q):
            challenged.append((seed, q.text, q.answer_text, oracle(q.text)))
    assert checked == len(SEEDS)
    assert not challenged, (
        f"{difficulty}: {len(challenged)}/{checked} questions disagree with the independent oracle: {challenged[:5]}"
    )


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "exam"])
def test_oracle_fires_on_a_corrupted_answer(difficulty):
    """Exercise the oracle's sensitivity through the exact comparison
    ``test_oracle_agrees_with_the_shipped_answer`` uses (``_challenge``), on a
    real, un-modified question TEXT paired with a hand-corrupted
    ``answer_text`` -- the shape of the real defect this instrument exists to
    catch (a generator whose formatter disagrees with its own construction).
    ``SimpleNamespace`` stands in for ``Question``: only ``.text``/
    ``.answer_text`` are read by ``_challenge``."""
    oracle = _ORACLES[difficulty]
    real = make(TOPIC, seed=0, difficulty=difficulty)
    correct = SimpleNamespace(text=real.text, answer_text=real.answer_text)
    corrupted = SimpleNamespace(text=real.text, answer_text=str(int(real.answer_text) + 9))
    assert not _challenge(oracle, correct), "the real, uncorrupted answer must not be challenged"
    assert _challenge(oracle, corrupted), "a hand-corrupted answer_text must be challenged"


def test_oracle_disagreement_would_be_visible_on_a_deliberately_broken_question():
    """Prove the oracle actually discriminates: build a text string with a
    swapped known-angle value (as a real generator regression might ship) and
    confirm the oracle computes a DIFFERENT missing angle than the original,
    undamaged question -- i.e. the oracle is sensitive to the defect it
    exists to catch, not a tautology that always agrees."""
    q = make(TOPIC, seed=1, difficulty="easy")
    original = _oracle_easy(q.text)
    m = re.search(r"measure (.+?) degrees\.", q.text)
    segment = m.group(1)
    nums = _NUM_RE.findall(segment)
    broken_segment = segment.replace(nums[0], str(int(nums[0]) + 20), 1)
    broken_text = q.text.replace(f"measure {segment} degrees.", f"measure {broken_segment} degrees.", 1)
    assert _oracle_easy(broken_text) != original
