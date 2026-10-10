"""Independent oracle for the currency-exchange skill, recomputing the AUD value from the rendered text.

Independent oracle for currency-exchange skill.

Parses the rendered question text WITHOUT importing anything from
`mathgen.generators.p8e_currency_exchange_build` and recomputes the expected
AUD value from first principles, determining the rate's DIRECTION from the
text itself (which currency name immediately follows "1 =") rather than from
any parameter the generator happens to pass around.

Generates through the skill's own `sources` (params-driven `make` call),
never `make(topic, seed, difficulty=tier)` alone.
"""

import functools
import re
from fractions import Fraction

import pytest

from mathgen.skills import load_skills
from mathgen import make

TOPIC = "currency_exchange_conversion"
SKILL_ID = "senior-use-currency-exchange-rates-to-determine-the-cost"
SEEDS = range(60)

_SYMBOLS = ("US$", "€", "£")

# "The exchange rate is <lhs>1 = <rhs><rate>. What is the .../.of <amount_sym><amount>?"
_TEXT_RE = re.compile(
    r"The exchange rate is (A\$|US\$|€|£)1 = (A\$|US\$|€|£)(\d+\.\d{2})\. "
    r"What is the (?:cost, in Australian dollars, of buying|value, in Australian dollars, of) "
    r"(US\$|€|£)(\d+)\?"
)


@functools.lru_cache(maxsize=None)
def _skills_dict():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def _make_from_sources(skills_dict, difficulty, seed):
    skill = skills_dict[SKILL_ID]
    src = next(s for s in skill.sources if s.difficulty == difficulty)
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _oracle_aud_value(text: str) -> Fraction:
    """Recompute the AUD value from the rendered text alone. The direction
    is read from which currency's "1" the rate sentence names FIRST -- "A$1
    = ..." means the rate divides a foreign amount; "<foreign>1 = A$..."
    means it multiplies -- never from a parameter passed by the caller."""
    m = _TEXT_RE.search(text)
    if not m:
        raise AssertionError(f"unparseable currency-exchange question: {text!r}")
    lhs_sym, rhs_sym, rate_str, amount_sym, amount_str = m.groups()
    rate = Fraction(rate_str)
    amount = Fraction(int(amount_str))
    if lhs_sym == "A$":
        # "A$1 = <foreign>rate" -- one Australian dollar buys `rate` units of
        # the foreign currency, so an amount of the foreign currency is worth
        # fewer Australian dollars: divide.
        assert rhs_sym in _SYMBOLS, text
        assert amount_sym == rhs_sym, text
        return amount / rate
    # "<foreign>1 = A$rate" -- one foreign-currency unit is worth `rate`
    # Australian dollars: multiply.
    assert lhs_sym in _SYMBOLS, text
    assert rhs_sym == "A$", text
    assert amount_sym == lhs_sym, text
    return amount * rate


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_oracle_matches_answer_text(difficulty):
    skills_dict = _skills_dict()
    checked = 0
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        expected = _oracle_aud_value(q.text)
        actual = Fraction(q.answer_text.removeprefix("A$"))
        assert expected == actual, (
            f"seed {seed} {difficulty}: expected {expected}, got {actual} -- {q.text!r} -> {q.answer_text!r}"
        )
        checked += 1
    assert checked == len(SEEDS)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_oracle_answer_never_matches_a_distractor(difficulty):
    skills_dict = _skills_dict()
    for seed in SEEDS:
        q = _make_from_sources(skills_dict, difficulty, seed)
        expected = _oracle_aud_value(q.text)
        for d in q.distractors:
            given = Fraction(d.removeprefix("A$"))
            assert expected != given, (
                f"seed {seed} {difficulty}: oracle answer {expected} matches distractor {d!r} -- {q.text!r}"
            )


def test_oracle_direction_detection_is_purely_textual():
    """A hand-built AUD-based and a hand-built foreign-based sentence, fed
    straight to the oracle with no generator involvement at all, to confirm
    the direction comes from the text and nothing else."""
    aud_based = "The exchange rate is A$1 = US$0.65. What is the value, in Australian dollars, of US$65?"
    assert _oracle_aud_value(aud_based) == Fraction(100)  # 65 / 0.65 = 100
    foreign_based = "The exchange rate is US$1 = A$1.35. What is the value, in Australian dollars, of US$100?"
    assert _oracle_aud_value(foreign_based) == Fraction(135)  # 100 * 1.35 = 135
