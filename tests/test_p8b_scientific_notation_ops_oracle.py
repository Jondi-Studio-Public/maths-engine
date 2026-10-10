"""Independent oracle for `scientific_notation_operate`, recomputing the renormalised coefficient and exponent.

Independent oracle for `scientific_notation_operate` topic.

Parses the two operands and the operation out of `question.text` with a
regular expression, then re-derives the renormalised (coefficient, exponent)
answer from scratch in plain Python (Fraction arithmetic; never imported from
`mathgen.generators.p8b_scientific_notation_ops_build` -- not a helper, not a
constant, not a table) and asserts it equals `question.answer_text` exactly.

**Exact text template relied on** (both operators are ASCII "+"/"-" for
add/subtract and the two Unicode symbols "×"/"÷" for multiply/divide, and
every scientific-notation number -- operand or option -- is formatted
`f"{coeff:g} × 10^{exp}"`, matching the existing `scientific_notation`
topic's own format string exactly):

    Calculate ({c1:g} × 10^{e1}) {op} ({c2:g} × 10^{e2}). Give your answer in scientific notation.

**String match is fine for the answer comparison** because both the
generator and this oracle format the final (coefficient, exponent) pair with
the same `:g` convention -- stated explicitly here rather than assumed. For a genuinely irrational-looking quotient
(divide's coefficient), `:g`'s 6-significant-figure rounding is itself part
of what's being reproduced: the oracle recomputes the exact quotient as a
Fraction and applies the identical `float(...)`+`:g` formatting step, which
is deterministic given the same input, so the two sides converge on the same
rounded string even though the underlying quotient does not terminate.

**Independence note on the add/subtract common-exponent convention**: the generator always displays the
larger-exponent operand first and rescales the other one to it (see the generator module's docstring). This
oracle does not assume "operand 1 has the larger exponent" as an implementation detail -- it independently
computes `common = max(e1, e2)` and rescales whichever PARSED operand has the smaller exponent, so it would
still get the right answer even if that "operand order" detail ever changed.
"""

import re
from fractions import Fraction

from mathgen import make

SEEDS = range(60)
OPS = ("add", "subtract", "multiply", "divide")

_TEXT_RE = re.compile(
    r"^Calculate \((?P<c1>\d+(?:\.\d+)?) × 10\^(?P<e1>-?\d+)\) (?P<op>[+\-×÷]) "
    r"\((?P<c2>\d+(?:\.\d+)?) × 10\^(?P<e2>-?\d+)\)\. "
    r"Give your answer in scientific notation\.$"
)

_OPTION_RE = re.compile(r"^(?P<coeff>-?\d+(?:\.\d+)?) × 10\^(?P<exp>-?\d+)$")

_SYMBOL_TO_KIND = {"+": "add", "-": "subtract", "×": "multiply", "÷": "divide"}


def _parse_question(text):
    m = _TEXT_RE.match(text)
    assert m, f"could not parse question text: {text!r}"
    return (
        Fraction(m["c1"]),
        int(m["e1"]),
        _SYMBOL_TO_KIND[m["op"]],
        Fraction(m["c2"]),
        int(m["e2"]),
    )


def _parse_option(option_text):
    m = _OPTION_RE.match(option_text)
    assert m, f"could not parse option text: {option_text!r}"
    return Fraction(m["coeff"]), int(m["exp"])


def _renormalise(coeff, exponent):
    """Shift `coeff` into [1, 10) by exact powers of ten -- written from
    scratch here, not imported from the generator."""
    while coeff >= 10:
        coeff, exponent = coeff / 10, exponent + 1
    while coeff < 1:
        coeff, exponent = coeff * 10, exponent - 1
    return coeff, exponent


def _fmt(coeff, exponent):
    return f"{float(coeff):g} × 10^{exponent}"


def _oracle_raw_and_answer(c1, e1, kind, c2, e2):
    """The un-renormalised (raw coefficient, raw exponent) pair and the
    renormalised (answer coefficient, answer exponent) pair, computed from
    scratch."""
    if kind in ("add", "subtract"):
        common = max(e1, e2)
        rc1 = c1 * Fraction(10) ** (e1 - common)
        rc2 = c2 * Fraction(10) ** (e2 - common)
        raw = rc1 + rc2 if kind == "add" else rc1 - rc2
        raw_exp = common
    elif kind == "multiply":
        raw, raw_exp = c1 * c2, e1 + e2
    else:
        raw, raw_exp = c1 / c2, e1 - e2
    return (raw, raw_exp), _renormalise(raw, raw_exp)


def test_oracle_answer_matches_question_answer_text_for_every_seed_and_op():
    checked = 0
    for op in OPS:
        for seed in SEEDS:
            q = make("scientific_notation_operate", seed=seed, ops=(op,))
            c1, e1, kind, c2, e2 = _parse_question(q.text)
            assert kind == op

            (_raw, _raw_exp), (answer_coeff, answer_exp) = _oracle_raw_and_answer(c1, e1, kind, c2, e2)
            expected_answer_text = _fmt(answer_coeff, answer_exp)
            assert expected_answer_text == q.answer_text, (
                q.text,
                q.answer_text,
                expected_answer_text,
            )
            # The oracle's own re-derivation must itself be a valid
            # scientific-notation coefficient.
            assert 1 <= answer_coeff < 10
            checked += 1
    assert checked == len(OPS) * len(SEEDS)


def test_un_renormalised_distractor_present_and_equal_to_the_true_raw_result():
    """The un-renormalised form must (a) be present among the distractors,
    formatted the same `:g` way as every other option, and (b) be
    numerically equal to the TRUE (unrounded) pre-renormalisation result,
    recomputed independently here from the parsed operands -- never taken
    from the generator's own intermediate values."""
    checked = 0
    for op in OPS:
        for seed in SEEDS:
            q = make("scientific_notation_operate", seed=seed, ops=(op,))
            c1, e1, kind, c2, e2 = _parse_question(q.text)

            (raw, raw_exp), (answer_coeff, answer_exp) = _oracle_raw_and_answer(c1, e1, kind, c2, e2)
            expected_raw_text = _fmt(raw, raw_exp)
            assert expected_raw_text in q.distractors, (
                q.text,
                q.distractors,
                expected_raw_text,
            )

            # Numerically equal to the true (unrounded) result: compare the
            # exact Fraction values, not the (necessarily :g-rounded, for a
            # non-terminating divide quotient) display strings.
            true_value = raw * Fraction(10) ** raw_exp
            answer_value = answer_coeff * Fraction(10) ** answer_exp
            assert true_value == answer_value

            # The distractor's own displayed (coefficient, exponent) pair
            # must differ from the answer's -- renormalisation was genuinely
            # needed.
            distractor_pair = _parse_option(expected_raw_text)
            answer_pair = _parse_option(q.answer_text)
            assert distractor_pair != answer_pair
            checked += 1
    assert checked == len(OPS) * len(SEEDS)


def test_oracle_recognises_all_four_operator_symbols_across_the_sweep():
    seen = set()
    for op in OPS:
        q = make("scientific_notation_operate", seed=0, ops=(op,))
        _c1, _e1, kind, _c2, _e2 = _parse_question(q.text)
        seen.add(kind)
    assert seen == set(OPS)
