"""Deliberately does NOT import anything from
``mathgen.generators.p19i_ratio_build`` -- it parses ``question.text`` with
its own regexes and recomputes the answer from scratch using its own
``fractions.Fraction``/``math.gcd`` arithmetic, then compares against
``question.answer``.
"""

import math
import re
from fractions import Fraction
from types import SimpleNamespace

import pytest

from mathgen import make

TOPIC_A = "p19i_ratio_fraction_share"
TOPIC_B = "p19i_ratio_simplest_form"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(60)


# =========================================================================
# Skill A oracle: fraction of the whole
# =========================================================================

_STATED_RATIO_RE = re.compile(r"ratio of ([a-zA-Z ]+?) to ([a-zA-Z ]+?) (?:is|of) (\d+):(\d+)")
_TWO_COUNT_RE = re.compile(r"(\d+) ([a-zA-Z ]+?) and (\d+) ([a-zA-Z ]+?)[.,]")
_TOTAL_ONE_RE = re.compile(r"(\d+) [a-zA-Z ]+? (?:in total|altogether)\. (\d+) of them are")
_EXAM_ITEM_RE = re.compile(r"(\d+) ([a-zA-Z ]+?), (\d+) ([a-zA-Z ]+?) and (\d+) ([a-zA-Z ]+?)[.,]")
_EXAM_RATIO_RE = re.compile(
    r"(\d+) ([a-zA-Z ]+?)\. The ratio of ([a-zA-Z ]+?) to \2 is (\d+):(\d+), and there are (\d+) ([a-zA-Z ]+?)\."
)


def _asked_term(text, x_label, y_label):
    return x_label if text.rfind(x_label) > text.rfind(y_label) else y_label


def _oracle_a_ratio_stated(text):
    """easy / medium: 'the ratio of X to Y is/of a:b ... is/are <asked>?'."""
    m = _STATED_RATIO_RE.search(text)
    x_label, y_label, a, b = m.group(1).strip(), m.group(2).strip(), int(m.group(3)), int(m.group(4))
    asked = _asked_term(text, x_label, y_label)
    numerator = a if asked == x_label else b
    return Fraction(numerator, a + b)


def _oracle_a_hard(text):
    """hard: a total and ONE group's count; the asked group is 'the rest'
    (total - given), so the numerator is derived by subtraction."""
    m = _TOTAL_ONE_RE.search(text)
    assert m, text
    total, given = int(m.group(1)), int(m.group(2))
    return Fraction(total - given, total)


def _oracle_a_exam(text):
    """exam: three stated numbers, either a third irrelevant item or a
    redundant (but consistent) total; only the first two named counts are
    ever relevant."""
    # The total is never stated; the whole is three groups, given either as three counts or as one
    # count + an x:y ratio + a third count.
    r = _EXAM_RATIO_RE.search(text)
    if r:
        cy, y_label, x_label, a, b, cz, z_label = (
            int(r.group(1)),
            r.group(2).strip(),
            r.group(3).strip(),
            int(r.group(4)),
            int(r.group(5)),
            int(r.group(6)),
            r.group(7).strip(),
        )
        groups = {x_label: cy // b * a, y_label: cy, z_label: cz}
    else:
        m = _EXAM_ITEM_RE.search(text)
        assert m, text
        groups = {
            m.group(2).strip(): int(m.group(1)),
            m.group(4).strip(): int(m.group(3)),
            m.group(6).strip(): int(m.group(5)),
        }
    asked = text.rsplit(" are ", 1)[1].rstrip("?.").strip()
    return Fraction(groups[asked], sum(groups.values()))


_ORACLES_A = {
    "easy": _oracle_a_ratio_stated,
    "medium": _oracle_a_ratio_stated,
    "hard": _oracle_a_hard,
    "exam": _oracle_a_exam,
}


def _value(q):
    return Fraction(q.answer["p"], q.answer["q"])


def _challenge_a(oracle, question):
    return oracle(question.text) != _value(question)


@pytest.mark.parametrize("difficulty", TIERS)
def test_skill_a_oracle_agrees_with_the_shipped_answer(difficulty):
    oracle = _ORACLES_A[difficulty]
    checked, challenged = 0, []
    for seed in SEEDS:
        q = make(TOPIC_A, seed=seed, difficulty=difficulty)
        checked += 1
        if _challenge_a(oracle, q):
            challenged.append((seed, q.text, q.answer_text, oracle(q.text)))
    assert checked == len(SEEDS)
    assert not challenged, (
        f"skill A {difficulty}: {len(challenged)}/{checked} disagree with the oracle: {challenged[:5]}"
    )


@pytest.mark.parametrize("difficulty", TIERS)
def test_skill_a_oracle_fires_on_a_corrupted_answer(difficulty):
    oracle = _ORACLES_A[difficulty]
    real = make(TOPIC_A, seed=0, difficulty=difficulty)
    correct = SimpleNamespace(text=real.text, answer=dict(real.answer))
    corrupted = SimpleNamespace(text=real.text, answer={"p": real.answer["p"] + 1, "q": real.answer["q"]})
    assert not _challenge_a(oracle, correct)
    assert _challenge_a(oracle, corrupted)


# =========================================================================
# Skill B oracle: simplest whole-number ratio
# =========================================================================

_STUDENTS_RE = re.compile(r"(\d+) students.*?(\d+)/(\d+)", re.DOTALL)
_DECIMAL_RE = re.compile(r"(\d+\.\d)")


def _oracle_b_two_count(text):
    """easy: exactly two named counts, joined directly by 'and'."""
    m = _TWO_COUNT_RE.search(text)
    x_count, y_count = int(m.group(1)), int(m.group(3))
    g = math.gcd(x_count, y_count)
    return (x_count // g, y_count // g)


def _oracle_b_three_count(text):
    """medium: three named counts; only the first two are relevant."""
    m = _EXAM_ITEM_RE.search(text)
    x_count, y_count = int(m.group(1)), int(m.group(3))
    g = math.gcd(x_count, y_count)
    return (x_count // g, y_count // g)


def _oracle_b_hard(text):
    """hard: a total and a stated fraction of it are girls; the rest boys."""
    m = _STUDENTS_RE.search(text)
    total, a_frac, b_frac = int(m.group(1)), int(m.group(2)), int(m.group(3))
    girls = total * a_frac // b_frac
    boys = total - girls
    g = math.gcd(girls, boys)
    return (girls // g, boys // g)


def _oracle_b_exam(text):
    """exam: two one-decimal-place measurements; clear by *10, then reduce."""
    nums = _DECIMAL_RE.findall(text)
    d1, d2 = Fraction(nums[0]), Fraction(nums[1])
    w1, w2 = int(d1 * 10), int(d2 * 10)
    g = math.gcd(w1, w2)
    return (w1 // g, w2 // g)


_ORACLES_B = {
    "easy": _oracle_b_two_count,
    "medium": _oracle_b_three_count,
    "hard": _oracle_b_hard,
    "exam": _oracle_b_exam,
}


def _pair_b(q):
    return (q.answer["p"], q.answer["q"])


def _challenge_b(oracle, question):
    return oracle(question.text) != _pair_b(question)


@pytest.mark.parametrize("difficulty", TIERS)
def test_skill_b_oracle_agrees_with_the_shipped_answer(difficulty):
    oracle = _ORACLES_B[difficulty]
    checked, challenged = 0, []
    for seed in SEEDS:
        q = make(TOPIC_B, seed=seed, difficulty=difficulty)
        checked += 1
        if _challenge_b(oracle, q):
            challenged.append((seed, q.text, q.answer_text, oracle(q.text)))
    assert checked == len(SEEDS)
    assert not challenged, (
        f"skill B {difficulty}: {len(challenged)}/{checked} disagree with the oracle: {challenged[:5]}"
    )


@pytest.mark.parametrize("difficulty", TIERS)
def test_skill_b_oracle_fires_on_a_corrupted_answer(difficulty):
    oracle = _ORACLES_B[difficulty]
    real = make(TOPIC_B, seed=0, difficulty=difficulty)
    p, q_ = real.answer["p"], real.answer["q"]
    correct = SimpleNamespace(text=real.text, answer={"p": p, "q": q_})
    corrupted = SimpleNamespace(text=real.text, answer={"p": p + 1, "q": q_})
    assert not _challenge_b(oracle, correct)
    assert _challenge_b(oracle, corrupted)


def test_skill_a_oracle_disagreement_is_visible_on_a_deliberately_broken_question():
    q = make(TOPIC_A, seed=1, difficulty="easy")
    original = _oracle_a_ratio_stated(q.text)
    m = _STATED_RATIO_RE.search(q.text)
    a = m.group(3)
    broken_text = q.text.replace(f"{a}:", f"{int(a) + 5}:", 1)
    assert _oracle_a_ratio_stated(broken_text) != original


def test_skill_b_oracle_disagreement_is_visible_on_a_deliberately_broken_question():
    q = make(TOPIC_B, seed=1, difficulty="easy")
    original = _oracle_b_two_count(q.text)
    m = _TWO_COUNT_RE.search(q.text)
    x_count = m.group(1)
    broken_text = q.text.replace(f"{x_count} ", f"{int(x_count) + 40} ", 1)
    assert _oracle_b_two_count(broken_text) != original
