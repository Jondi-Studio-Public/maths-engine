"""Independent oracle for the `p8c_normality_approx` questions: it re-derives each answer from the rendered
question text.

Parses each rendered question's text, independently recomputes n*p-hat and n*(1-p-hat) and the resulting 4-way
classification from scratch, and compares against `question.answer_text` -- must NOT import anything from
`mathgen.generators.p8c_normality_approx_build`.

Classification skill: this is a case where the answer IS derivable from
the parsed question text (n and p-hat, or n and x), so the oracle both
re-derives the answer AND confirms, over the combined sweep, that all four
categories are observed somewhere.
"""

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen import make

TOPIC = "p8c_normality_approx_phat"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(40)

_VALID = "Valid -- both n×p-hat and n×(1-p-hat) are at least 10"
_P_FAILS = "Not valid -- n×p-hat is less than 10"
_Q_FAILS = "Not valid -- n×(1-p-hat) is less than 10"
_BOTH_FAIL = "Not valid -- both n×p-hat and n×(1-p-hat) are less than 10"
_ALL = {_VALID, _P_FAILS, _Q_FAILS, _BOTH_FAIL}

_N_RE = re.compile(r"n = (\d+)")
_PHAT_DIRECT_RE = re.compile(r"sample proportion of p-hat = (\d+\.\d+)")
_X_RE = re.compile(r"x = (\d+) success")


def _oracle_category(text: str) -> str:
    n = int(_N_RE.search(text).group(1))
    direct = _PHAT_DIRECT_RE.search(text)
    if direct:
        phat = Fraction(direct.group(1))
    else:
        x = int(_X_RE.search(text).group(1))
        phat = Fraction(x, n)
    prod_p = n * phat
    prod_q = n * (1 - phat)
    p_ok = prod_p >= 10
    q_ok = prod_q >= 10
    if p_ok and q_ok:
        return _VALID
    if p_ok:
        return _Q_FAILS
    if q_ok:
        return _P_FAILS
    return _BOTH_FAIL


@pytest.mark.parametrize("difficulty", TIERS)
def test_oracle_classification_matches_answer_text(difficulty):
    challenged = []
    n_checked = 0
    for seed in SEEDS:
        question = make(TOPIC, seed=seed, difficulty=difficulty)
        oracle = _oracle_category(question.text)
        n_checked += 1
        if oracle != question.answer_text:
            challenged.append((seed, question.text, question.answer_text, oracle))
    assert not challenged, f"oracle challenges: {challenged}"
    assert n_checked == len(SEEDS)


def test_oracle_confirms_all_four_categories_occur_across_tiers_combined():
    counts = Counter()
    for difficulty in TIERS:
        for seed in SEEDS:
            question = make(TOPIC, seed=seed, difficulty=difficulty)
            counts[_oracle_category(question.text)] += 1
    assert set(counts) == _ALL, f"only saw: {dict(counts)}"
    print(f"per-category counts over {len(TIERS) * len(SEEDS)} questions: {dict(counts)}")
