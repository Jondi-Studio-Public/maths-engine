"""Re-derives each rendered question's mcq key from `question.text` alone, via hand-written parsing and
predicates -- never importing anything from `mathgen.generators.p15a_definition_quantifier_build`.

Total re-derived: 4 tiers x 300 seeds = 1200.
"""

from __future__ import annotations

import re

import pytest

from mathgen.grading.typed import mcq_options
from mathgen import make

TOPIC = "p15a_definition_quantifier_check"
N_SEEDS = 300


def _parse_ab(text: str):
    m = re.search(r"Let A = \{([^}]*)\} and B = \{([^}]*)\}", text)
    A = [int(v.strip()) for v in m.group(1).split(",")]
    B = [v.strip() for v in m.group(2).split(",")]
    return A, B


def _parse_relation_list(text: str):
    m = re.search(r"let R = \{(.*?)\} be a relation", text)
    pairs_str = m.group(1)
    pairs = re.findall(r"\((\d+), (\w)\)", pairs_str)
    return [(int(a), b) for a, b in pairs]


def _parse_relation_grid(text: str, A, B):
    """Grid format: '     b1  b2  b3 |  a1: X  -  - |  a2: - X  - | ...'."""
    m = re.search(r"grid below \(X marks a pair \(a, b\) in R, row a, column b\): (.*?)\. ", text)
    grid_text = m.group(1)
    row_sections = grid_text.split(" | ")[1:]  # skip header
    R = []
    for a, row in zip(A, row_sections):
        marks = row.split(":", 1)[1].strip().split()
        for b, mark in zip(B, marks):
            if mark == "X":
                R.append((a, b))
    return R


def _split_options(text: str) -> dict:
    opts_section = text[text.index("A:") :]
    markers = list(re.finditer(r"(?:^|\s)([ABCD]): ", opts_section))
    opts = {}
    for i, m in enumerate(markers):
        letter = m.group(1)
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(opts_section)
        opts[letter] = opts_section[start:end].strip()
    return opts


def _get_relation(text: str):
    A, B = _parse_ab(text)
    if "grid below" in text:
        R = _parse_relation_grid(text, A, B)
    else:
        R = _parse_relation_list(text)
    return A, B, R


def _oracle_domain(text: str) -> str:
    # The domain of a relation is the set of first coordinates that appear in R (Ruling by Jondi, 2026-10-26, round 4 C5).
    A, B, R = _get_relation(text)
    active = sorted({a for a, _ in R})
    opts = _split_options(text)
    for letter, opt_text in opts.items():
        vals = re.findall(r"[\w]+", opt_text)
        if vals and all(re.match(r"^\d+$", v) for v in vals) and sorted(int(v) for v in vals) == active:
            return letter
    raise AssertionError(f"could not find domain option matching {active} in {opts}")


def _oracle_codomain(text: str) -> str:
    A, B, R = _get_relation(text)
    opts = _split_options(text)
    for letter, opt_text in opts.items():
        vals = re.findall(r"[a-z]+", opt_text)
        if sorted(vals) == sorted(B):
            return letter
    raise AssertionError(f"could not find codomain option matching B={B} in {opts}")


def _oracle_is_function(text: str) -> str:
    A, B, R = _get_relation(text)
    degree = {a: sum(1 for x, _ in R if x == a) for a in A}
    opts = _split_options(text)
    if all(d == 1 for d in degree.values()):
        target_phrase = "It IS a function"
    elif any(d >= 2 for d in degree.values()):
        culprit = next(a for a in A if degree[a] >= 2)
        target_phrase = f"NOT a function: {culprit} has more than one image"
    else:
        culprit = next(a for a in A if degree[a] == 0)
        target_phrase = f"NOT a function: {culprit} has no image"
    for letter, opt_text in opts.items():
        if opt_text.startswith(target_phrase):
            return letter
    raise AssertionError(f"could not find is-function option for {target_phrase} in {opts}")


def _oracle_quantifier_scope(text: str) -> str:
    A, B, R = _get_relation(text)
    R_set = set(R)
    i_true = all(any((a, b) in R_set for b in B) for a in A)
    ii_true = any(all((a, b) in R_set for a in A) for b in B)
    assert not (ii_true and not i_true), "quantifier-scope theorem violated -- oracle itself is wrong"
    if i_true and ii_true:
        target_phrase = "Both statements are true."
    elif i_true:
        target_phrase = "Only statement (i) is true."
    else:
        target_phrase = "Neither statement is true."
    opts = _split_options(text)
    for letter, opt_text in opts.items():
        if opt_text == target_phrase:
            return letter
    raise AssertionError(f"could not find quantifier-scope option for {target_phrase} in {opts}")


def _oracle(text: str) -> str:
    if "What is the domain of R" in text:
        return _oracle_domain(text)
    if "What is the codomain of R" in text:
        return _oracle_codomain(text)
    if "Is R a function" in text:
        return _oracle_is_function(text)
    if "Which is correct?" in text:
        return _oracle_quantifier_scope(text)
    raise AssertionError(f"could not classify aspect of {text!r}")


@pytest.mark.parametrize("tier", ["easy", "medium", "hard", "exam"])
def test_oracle_matches(tier):
    checked = 0
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        expected = _oracle(q.text)
        assert expected == q.answer_text, (tier, seed, q.text, q.answer_text)
        checked += 1
    assert checked == N_SEEDS


def test_oracle_is_red_on_a_corrupted_answer():
    q = make(TOPIC, seed=0, difficulty="easy")
    expected = _oracle(q.text)
    corrupted = next(o for o in mcq_options(q) if o != expected)
    assert corrupted != expected
