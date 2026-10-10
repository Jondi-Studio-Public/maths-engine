"""Derives the correct answer from the rendered ``question.text`` alone, with
its own regex and its own copy of the four fixed option strings for each
topic. Never imports ``mathgen.generators.p10b_checking_results_build`` --
this generator's answer space is a small closed set of templates x classes,
so the oracle recomputes from the parsed numbers rather than re-deriving a
formula, which is exactly the same route this pairing's classification takes.
"""

import re
from collections import Counter
from fractions import Fraction

from mathgen import make

SEEDS = 400
TIERS = ("easy", "medium", "hard")

# --- reasonableness_verdict (ACMEM003), re-declared independently ----------

_OPT_REASONABLE = "Reasonable"
_OPT_TOO_LARGE = "Not reasonable -- the stated answer is too large"
_OPT_TOO_SMALL = "Not reasonable -- the stated answer is too small"
_OPT_WRONG_SIGN = "Not reasonable -- the stated answer has the wrong sign"
_OPTIONS_A = {_OPT_REASONABLE, _OPT_TOO_LARGE, _OPT_TOO_SMALL, _OPT_WRONG_SIGN}

_RATIO_TO_OPTION = {
    1: _OPT_REASONABLE,
    10: _OPT_TOO_LARGE,
    Fraction(1, 10): _OPT_TOO_SMALL,
    -1: _OPT_WRONG_SIGN,
}

_A_PATTERN = re.compile(r"states: (.+?) = (-?\d+)\.")


def _oracle_answer_a(text: str) -> str:
    expr, stated = _A_PATTERN.search(text).groups()
    true = eval(expr.replace("×", "*"), {"__builtins__": {}}, {})
    stated = int(stated)
    ratio = Fraction(stated, true)
    return _RATIO_TO_OPTION[ratio]


def test_reasonableness_verdict_oracle_matches_every_rendered_answer():
    total = 0
    distribution = {tier: Counter() for tier in TIERS}
    for tier in TIERS:
        for seed in range(SEEDS):
            question = make("reasonableness_verdict", seed=seed, difficulty=tier)
            total += 1
            oracle_answer = _oracle_answer_a(question.text)
            assert oracle_answer in _OPTIONS_A
            assert oracle_answer == question.answer_text, (
                f"reasonableness_verdict seed={seed} tier={tier}: oracle {oracle_answer!r} "
                f"!= rendered answer {question.answer_text!r}\ntext: {question.text}"
            )
            distribution[tier][question.answer_text] += 1
    print(f"\nreasonableness_verdict oracle: {total} questions re-derived")
    for tier in TIERS:
        print(f"  {tier}: {dict(distribution[tier])}")
    assert total == SEEDS * len(TIERS)


# --- locate_calculation_error (ACMEM006), re-declared independently --------

_OPT_STEP1 = "The error is in Step 1"
_OPT_STEP2 = "The error is in Step 2"
_OPT_STEP3 = "The error is in Step 3"
_OPT_NO_ERROR = "There is no error -- every step is correct"
_OPTIONS_B = {_OPT_STEP1, _OPT_STEP2, _OPT_STEP3, _OPT_NO_ERROR}
_STEP_TO_OPTION = {1: _OPT_STEP1, 2: _OPT_STEP2, 3: _OPT_STEP3, None: _OPT_NO_ERROR}

_STEP_PATTERN = re.compile(r"Step (\d): (\d+) (\+|-|×) (\d+) = (\d+)\.")
_APPLY = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "×": lambda a, b: a * b}


def _oracle_answer_b(text: str) -> str:
    steps = _STEP_PATTERN.findall(text)
    assert len(steps) == 3, f"expected 3 steps, found {len(steps)} in: {text}"
    mismatched_index = None
    for idx, a, op, b, stated in steps:
        true = _APPLY[op](int(a), int(b))
        if true != int(stated):
            assert mismatched_index is None, f"more than one step mismatches: {text}"
            mismatched_index = int(idx)
    return _STEP_TO_OPTION[mismatched_index]


def test_locate_calculation_error_oracle_matches_every_rendered_answer():
    total = 0
    distribution = {tier: Counter() for tier in TIERS}
    for tier in TIERS:
        for seed in range(SEEDS):
            question = make("locate_calculation_error", seed=seed, difficulty=tier)
            total += 1
            oracle_answer = _oracle_answer_b(question.text)
            assert oracle_answer in _OPTIONS_B
            assert oracle_answer == question.answer_text, (
                f"locate_calculation_error seed={seed} tier={tier}: oracle {oracle_answer!r} "
                f"!= rendered answer {question.answer_text!r}\ntext: {question.text}"
            )
            distribution[tier][question.answer_text] += 1
    print(f"\nlocate_calculation_error oracle: {total} questions re-derived")
    for tier in TIERS:
        print(f"  {tier}: {dict(distribution[tier])}")
    assert total == SEEDS * len(TIERS)
