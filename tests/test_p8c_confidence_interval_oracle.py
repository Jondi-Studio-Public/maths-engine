"""Independent oracle for the `p8c_confidence_interval` questions: it re-derives each answer from the rendered
question text.

Parses each rendered question's text and re-derives (lower, upper) from scratch, comparing against
`question.answer_text` -- must NOT import anything from `mathgen.generators.p8c_confidence_interval_build`.
"""

import math
import re
from fractions import Fraction

import pytest

from mathgen import make

MEAN_TOPIC = "p8c_confidence_interval_mean"
PROP_TOPIC = "p8c_confidence_interval_proportion"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(40)
TOLERANCE = Fraction(51, 1000)

_TABLE_ROW_RE = re.compile(r"a (\d+)% confidence level uses z = ([\d.]+)")
_PAIR_RE = re.compile(r"lower = (-?[\d.]+), upper = (-?[\d.]+)")


def _z_from_text(text: str) -> float:
    """Either a directly stated `(z = ...)`, or a lookup-table row matching
    the stated confidence level."""
    m = re.search(r"confidence level \(z = ([\d.]+)\)", text)
    if m:
        return float(m.group(1))
    table = dict(_TABLE_ROW_RE.findall(text))
    m = re.search(r"at the (\d+)% confidence level", text)
    assert m, f"no confidence level found: {text!r}"
    return float(table[m.group(1)])


def _answer_pair(text: str) -> tuple[Fraction, Fraction]:
    m = _PAIR_RE.match(text)
    assert m, f"unparseable answer_text: {text!r}"
    return Fraction(m.group(1)), Fraction(m.group(2))


def _oracle_mean(question):
    text = question.text
    n = int(re.search(r"sample of (\d+) parcels", text).group(1))
    x_bar = int(re.search(r"sample mean delivery time of (\d+) minutes", text).group(1))
    s_match = re.search(r"sample standard deviation of (\d+) minutes", text)
    if s_match:
        s = int(s_match.group(1))
    else:
        variance = int(re.search(r"sample variance of (\d+) square minutes", text).group(1))
        s = math.sqrt(variance)
    z = _z_from_text(text)
    margin = z * s / math.sqrt(n)
    return round(x_bar - margin, 1), round(x_bar + margin, 1)


def _oracle_proportion(question):
    text = question.text
    n = int(re.search(r"survey of (\d+) voters", text).group(1))
    phat_match = re.search(r"phat, is ([\d.]+)\.", text)
    if phat_match:
        phat = float(phat_match.group(1))
    else:
        x = int(re.search(r"survey of \d+ voters, (\d+) said they intend to vote yes", text).group(1))
        phat = x / n
    z = _z_from_text(text)
    margin_p = z * math.sqrt(phat * (1 - phat) / n)
    lower_pct = round((phat - margin_p) * 100, 1)
    upper_pct = round((phat + margin_p) * 100, 1)
    return lower_pct, upper_pct


@pytest.mark.parametrize("difficulty", TIERS)
def test_oracle_mean_interval_matches_answer_text(difficulty):
    challenged = []
    n_checked = 0
    for seed in SEEDS:
        question = make(MEAN_TOPIC, seed=seed, difficulty=difficulty)
        oracle_lower, oracle_upper = _oracle_mean(question)
        stated_lower, stated_upper = _answer_pair(question.answer_text)
        n_checked += 1
        if abs(oracle_lower - float(stated_lower)) > 0.051 or abs(oracle_upper - float(stated_upper)) > 0.051:
            challenged.append((seed, question.text, question.answer_text, (oracle_lower, oracle_upper)))
    assert not challenged, f"oracle challenges: {challenged}"
    assert n_checked == len(SEEDS)


@pytest.mark.parametrize("difficulty", TIERS)
def test_oracle_proportion_interval_matches_answer_text(difficulty):
    challenged = []
    n_checked = 0
    for seed in SEEDS:
        question = make(PROP_TOPIC, seed=seed, difficulty=difficulty)
        oracle_lower, oracle_upper = _oracle_proportion(question)
        stated_lower, stated_upper = _answer_pair(question.answer_text)
        n_checked += 1
        if abs(oracle_lower - float(stated_lower)) > 0.051 or abs(oracle_upper - float(stated_upper)) > 0.051:
            challenged.append((seed, question.text, question.answer_text, (oracle_lower, oracle_upper)))
    assert not challenged, f"oracle challenges: {challenged}"
    assert n_checked == len(SEEDS)
