"""Independent oracle for the `pv0b_break_even` questions: it re-derives each answer from the rendered question
text.

Imports nothing from `mathgen.generators.pv0b_break_even_build`. Parses `F`,
`v`, `p` (or the two-points/percentage-fee equivalents) straight out of the
RENDERED `question.text` and re-derives the break-even quantity from scratch
via the plain cost-equals-revenue identity, solved however each tier's own
prompt states. Run across >= 1000 questions total, spread over all four
tiers (>= 250 each).
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from mathgen import make

TOPIC = "pv0b_break_even"
SEEDS_PER_TIER = 260  # 4 tiers x 260 = 1040 >= 1000


def _stated(text: str) -> dict:
    F = int(re.search(r"fixed costs of \$(\d+)", text).group(1))
    v = int(re.search(r"costs? \$(\d+) to \w+", text).group(1))
    p = int(re.search(r"sells for \$(\d+)(?:,| but| for)?", text).group(1))
    return dict(F=F, v=v, p=p)


def _two_points(text: str) -> dict:
    m = re.search(
        r"(\d+) [\w' -]+? costs \$(\d+) in total, and \w+ (\d+) [\w' -]+? costs \$(\d+) in total\. "
        r"Each [\w' -]+? sells for \$(\d+)",
        text,
    )
    q1, cost1, q2, cost2, p = (int(g) for g in m.groups())
    return dict(q1=q1, cost1=cost1, q2=q2, cost2=cost2, p=p)


def _percentage(text: str) -> dict:
    fields = _stated(text)
    r = int(re.search(r"(\d+)% of the selling price", text).group(1))
    fields["r"] = r
    return fields


def test_oracle_stated_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="easy", mode="stated")
        fields = _stated(q.text)
        F, v, p = fields["F"], fields["v"], fields["p"]
        q_exact = Fraction(F, p - v)
        assert q_exact.denominator == 1, (seed, q.text, q_exact)
        assert str(int(q_exact)) == q.answer_text, (seed, q.text, q.answer_text, q_exact)
        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_round_up_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="medium", mode="round_up")
        fields = _stated(q.text)
        F, v, p = fields["F"], fields["v"], fields["p"]
        q_exact = Fraction(F, p - v)
        assert q_exact.denominator != 1, (seed, q.text, q_exact)
        assert str(math.ceil(q_exact)) == q.answer_text, (seed, q.text, q.answer_text, q_exact)
        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_two_points_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="hard", mode="two_points")
        fields = _two_points(q.text)
        q1, cost1, q2, cost2, p = (
            fields["q1"],
            fields["cost1"],
            fields["q2"],
            fields["cost2"],
            fields["p"],
        )
        v = Fraction(cost2 - cost1, q2 - q1)
        F = Fraction(cost1) - v * q1
        q_exact = F / (p - v)
        assert q_exact.denominator == 1, (seed, q.text, q_exact)
        assert str(int(q_exact)) == q.answer_text, (seed, q.text, q.answer_text, q_exact)
        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_percentage_fee_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="exam", mode="percentage_fee")
        fields = _percentage(q.text)
        F, v, p, r = fields["F"], fields["v"], fields["p"], fields["r"]
        margin = Fraction(p) * (100 - r) / 100 - v
        q_exact = Fraction(F) / margin
        assert q_exact.denominator != 1, (seed, q.text, q_exact)
        assert str(math.ceil(q_exact)) == q.answer_text, (seed, q.text, q.answer_text, q_exact)
        checked += 1
    assert checked == SEEDS_PER_TIER
