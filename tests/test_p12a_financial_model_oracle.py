"""Independent oracle for the `p12a_financial_model` questions: it re-derives each answer from the rendered
question text.

Parses the RENDERED question text only (`question.text`/`question.answer_text`)
and imports NOTHING from `mathgen.generators.p12a_financial_model_build`. Every
numeric route below is re-derived from the stated text with `Fraction`
arithmetic, never by reusing the generator's own construction values.

Both skills answer in exact integer cents (or, for the ratio-rate exam tier, an
exact litres/kg value to 2dp); the oracle recomputes each in that same exact
arithmetic and compares against `question.answer_text` as a `Fraction`.
"""

import functools
import re
from fractions import Fraction

import pytest

from mathgen.skills import load_skills
from mathgen import make

PM_SKILL = "f10-rational-percentage-financial-model"
RR_SKILL = "f10-ratio-rate-financial-model"
DIFFICULTIES = ["easy", "medium", "hard", "exam"]
SEEDS = range(60)


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skill_id, difficulty, seed):
    skill = _skills_dict()[skill_id]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    topic, params = src.topic, src.params or {}
    if topic == "exd_x_r3e_fee_discount_gst":
        # The exam source moved to a multi-part chain; these tests pin the earlier single-answer tier.
        topic, params = "rational_percentage_financial_model", {}
    return make(topic, seed=seed, difficulty=src.difficulty, **params)


def _frac_value(text: str) -> Fraction:
    m = re.fullmatch(r"(\d+)/(\d+)", text)
    return Fraction(int(m.group(1)), int(m.group(2)))


# =============================================================================
# Percentage-model oracle
# =============================================================================

_PM_MONEY_RE = re.compile(r"\$(\d+)\b")
_PM_FRAC_RE = re.compile(r"\b(\d+/\d+)\b")
_PM_PCT_RE = re.compile(r"\b(\d+)%")
_PM_F_RE = re.compile(r"\$(\d+(?:\.\d+)?)")

_PM_EASY_FRACTION_TAILS = ("save", "donate to charity", "spend on rent")


def _pm_easy_oracle(text: str) -> Fraction:
    M = int(_PM_MONEY_RE.search(text).group(1))
    frac = _frac_value(_PM_FRAC_RE.search(text).group(1))
    pct = Fraction(int(_PM_PCT_RE.search(text).group(1)), 100)
    tail = text.rstrip("?").rsplit(".", 1)[-1].strip()
    asked_fraction = tail.endswith(_PM_EASY_FRACTION_TAILS)
    value = frac if asked_fraction else pct
    return Fraction(M) * value


def _pm_medium_oracle(text: str) -> Fraction:
    M = int(_PM_MONEY_RE.search(text).group(1))
    tokens = re.findall(r"\d+/\d+|\d+%", text)
    assert len(tokens) == 2, text
    values = []
    for tok in tokens:
        values.append(_frac_value(tok) if "/" in tok else Fraction(int(tok.rstrip("%")), 100))
    # multiplication commutes: the chain's value does not depend on which
    # operation the text names first.
    return Fraction(M) * values[0] * values[1]


def _pm_hard_oracle(text: str) -> Fraction:
    tokens = re.findall(r"\d+/\d+|\d+%", text)
    assert len(tokens) == 2, text
    values = []
    for tok in tokens:
        values.append(_frac_value(tok) if "/" in tok else Fraction(int(tok.rstrip("%")), 100))
    F = Fraction(_PM_F_RE.search(text).group(1))
    coeff = (1 - values[0]) * (1 - values[1])
    return F / coeff


def _pm_exam_oracle(text: str) -> Fraction:
    frac = _frac_value(re.search(r"(\d+/\d+) of the ORIGINAL amount", text).group(1))
    pct2 = Fraction(int(re.search(r"further (\d+)% of what THEN remained", text).group(1)), 100)
    pct3 = Fraction(int(re.search(r"flat (\d+)% service fee", text).group(1)), 100)
    F = Fraction(re.search(r"has \$(\d+(?:\.\d+)?) left", text).group(1))
    coeff = (1 - frac) * (1 - pct2) - pct3
    return F / coeff


_PM_ORACLES = {
    "easy": _pm_easy_oracle,
    "medium": _pm_medium_oracle,
    "hard": _pm_hard_oracle,
    "exam": _pm_exam_oracle,
}


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_pm_oracle_matches_answer_text(difficulty):
    checked = 0
    for seed in SEEDS:
        q = _make_from_sources(PM_SKILL, difficulty, seed)
        expected = _PM_ORACLES[difficulty](q.text)
        actual = Fraction(q.answer_text)
        assert expected == actual, (
            f"seed {seed} {difficulty}: expected {expected}, got {actual} -- {q.text!r} -> {q.answer_text!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


def test_pm_hard_forward_simulation_reaches_the_stated_final_amount():
    """The "prove it the other way" discipline: forward-
    simulate the STATED model from the recovered starting amount and
    confirm it lands on the stated final amount, for the reverse tiers."""
    for seed in range(30):
        q = _make_from_sources(PM_SKILL, "hard", seed)
        tokens = re.findall(r"\d+/\d+|\d+%", q.text)
        values = [_frac_value(t) if "/" in t else Fraction(int(t.rstrip("%")), 100) for t in tokens]
        F = Fraction(_PM_F_RE.search(q.text).group(1))
        M = Fraction(q.answer_text)
        forward = M * (1 - values[0]) * (1 - values[1])
        assert forward == F, (seed, q.text, q.answer_text)


def test_pm_exam_forward_simulation_reaches_the_stated_final_amount():
    for seed in range(30):
        q = _make_from_sources(PM_SKILL, "exam", seed)
        frac = _frac_value(re.search(r"(\d+/\d+) of the ORIGINAL amount", q.text).group(1))
        pct2 = Fraction(int(re.search(r"further (\d+)% of what THEN remained", q.text).group(1)), 100)
        pct3 = Fraction(int(re.search(r"flat (\d+)% service fee", q.text).group(1)), 100)
        F = Fraction(re.search(r"has \$(\d+(?:\.\d+)?) left", q.text).group(1))
        M = Fraction(q.answer_text)
        after1 = M * (1 - frac)
        after2 = after1 * (1 - pct2)
        fee = M * pct3
        forward = after2 - fee
        assert forward == F, (seed, q.text, q.answer_text)


def test_pm_oracle_hand_worked_example():
    text = (
        "After spending 1/5 of Mia's allowance, then a further 20% of what remained, "
        "Mia has $96.00 left. How much did Mia start with?"
    )
    expected = _pm_hard_oracle(text)
    assert expected == Fraction("150.00")


def test_pm_oracle_fires_on_a_corrupted_answer():
    text = (
        "After spending 1/5 of Mia's allowance, then a further 20% of what remained, "
        "Mia has $96.00 left. How much did Mia start with?"
    )
    expected = _pm_hard_oracle(text)
    corrupted = expected + 1
    assert corrupted != expected


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_pm_oracle_never_matches_a_distractor(difficulty):
    for seed in SEEDS:
        q = _make_from_sources(PM_SKILL, difficulty, seed)
        expected = _PM_ORACLES[difficulty](q.text)
        for d in q.distractors:
            assert Fraction(d) != expected, (difficulty, seed, q.text, d)


# =============================================================================
# Ratio-rate-model oracle
# =============================================================================

_RR_T_RE = re.compile(r"\$(\d+)\b")
_RR_RATIO2_RE = re.compile(r"ratio (\d+):(\d+)")
_RR_RATIO3_RE = re.compile(r"ratio (\d+):(\d+):(\d+)")
_RR_RATE_RE = re.compile(r"\$(\d+(?:\.\d+)?) per \w+")
_RR_QTY_RE = re.compile(r"buys? (?:for )?(\d+) \w+")


def _rr_easy_oracle(q):
    text = q.text
    ratio_m = _RR_RATIO2_RE.search(text)
    if ratio_m and "split" in text and "per" not in text:
        T = int(_RR_T_RE.search(text).group(1))
        p, qv = int(ratio_m.group(1)), int(ratio_m.group(2))
        who = re.search(r"does (\w+) (?:receive|pay|keep)\?$", text).group(1)
        n1 = re.search(r"between (\w+) and \w+", text).group(1)
        share_p = Fraction(T * p, p + qv)
        share_q = Fraction(T * qv, p + qv)
        return share_p if who == n1 else share_q
    # rate branch
    rate = Fraction(_RR_RATE_RE.search(text).group(1))
    qty_m = re.search(r"buys (\d+) \w+|billed for (\d+) \w+", text)
    qty = int(qty_m.group(1) or qty_m.group(2))
    return rate * qty


def _rr_medium_oracle(q):
    text = q.text
    ratio_m = _RR_RATIO3_RE.search(text)
    if ratio_m:
        T = int(_RR_T_RE.search(text).group(1))
        p, qv, r = (int(x) for x in ratio_m.groups())
        n1, n2 = re.findall(r"between (\w+), (\w+) and \w+", text)[0]
        return Fraction(T * (p + qv), p + qv + r)
    rate = Fraction(_RR_RATE_RE.search(text).group(1))
    qty_m = re.search(r"buys (\d+) \w+|(\d+) \w+, a", text)
    qty = int(qty_m.group(1) or qty_m.group(2))
    base = rate * qty
    adjust = Fraction(re.search(r"\$(\d+(?:\.\d+)?) flat (discount|surcharge)", text).group(1))
    direction = re.search(r"\$(?:\d+(?:\.\d+)?) flat (discount|surcharge)", text).group(1)
    return base - adjust if direction == "discount" else base + adjust


def _rr_hard_oracle(q):
    text = q.text
    if "materials fee" in text:
        T = int(_RR_T_RE.search(text).group(1))
        p, qv = (int(x) for x in _RR_RATIO2_RE.search(text).group(1, 2))
        flat = Fraction(re.search(r"flat \$(\d+(?:\.\d+)?) materials fee", text).group(1))
        return (T - flat) * Fraction(p, p + qv)
    rate = Fraction(_RR_RATE_RE.search(text).group(1))
    free_qty = int(re.search(r"first (\d+) \w+ are free", text).group(1))
    qty = int(re.search(r"buys (\d+) \w+ in total", text).group(1))
    return rate * (qty - free_qty)


def _rr_exam_oracle(q):
    text = q.text
    T = int(_RR_T_RE.search(text).group(1))
    p, qv = (int(x) for x in _RR_RATIO2_RE.search(text).group(1, 2))
    rate = Fraction(_RR_RATE_RE.search(text).group(1))
    larger = max(p, qv)
    share = Fraction(T * larger, p + qv)
    return share / rate


_RR_ORACLES = {
    "easy": _rr_easy_oracle,
    "medium": _rr_medium_oracle,
    "hard": _rr_hard_oracle,
    "exam": _rr_exam_oracle,
}


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_rr_oracle_matches_answer_text(difficulty):
    checked = 0
    for seed in SEEDS:
        q = _make_from_sources(RR_SKILL, difficulty, seed)
        expected = _RR_ORACLES[difficulty](q)
        actual = Fraction(q.answer_text)
        assert expected == actual, (
            f"seed {seed} {difficulty}: expected {expected}, got {actual} -- {q.text!r} -> {q.answer_text!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_rr_oracle_never_matches_a_distractor(difficulty):
    for seed in SEEDS:
        q = _make_from_sources(RR_SKILL, difficulty, seed)
        expected = _RR_ORACLES[difficulty](q)
        for d in q.distractors:
            assert Fraction(d) != expected, (difficulty, seed, q.text, d)


def test_rr_oracle_hand_worked_example():
    text = "A $200 bill is split between Amir and Priya in the ratio 3:5. How much does Priya pay?"

    class _Q:
        pass

    fake = _Q()
    fake.text = text
    expected = _rr_easy_oracle(fake)
    assert expected == Fraction(125)


def test_rr_oracle_fires_on_a_corrupted_answer():
    text = "A $200 bill is split between Amir and Priya in the ratio 3:5. How much does Priya pay?"

    class _Q:
        pass

    fake = _Q()
    fake.text = text
    expected = _rr_easy_oracle(fake)
    corrupted = expected + 1
    assert corrupted != expected
