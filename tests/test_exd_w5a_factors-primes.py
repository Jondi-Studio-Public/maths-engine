"""Hard tier of the factors / primes skills shows the number in
prime-factorised form (form="factored"); easy/medium are unchanged."""

import math
import re
from pathlib import Path

import pytest
import yaml

from mathgen import make

SEEDS = range(300)
ROOT = Path(__file__).resolve().parent.parent
_SUP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}


def _parse_factored(text: str) -> dict[int, int]:
    out = {}
    for tok in text.split(" × "):
        m = re.fullmatch(r"(\d+)([⁰¹²³⁴⁵⁶⁷⁸⁹]*)", tok)
        assert m, tok
        out[int(m.group(1))] = int("".join(_SUP[c] for c in m.group(2))) if m.group(2) else 1
    return out


def _value(fac):
    return math.prod(p**e for p, e in fac.items())


@pytest.mark.parametrize("seed", SEEDS)
def test_divisibility_factored_key_and_distractor(seed):
    q = make("factors_multiples_divisibility", seed=seed, difficulty="hard", form="factored")
    m = re.fullmatch(r"Does (\d+) divide (.+) exactly \(with no remainder\)\?", q.text)
    assert m, q.text
    a = int(m.group(1))
    b_fac = _parse_factored(m.group(2))
    # independent: exponent comparison on a's factorisation
    a_fac, n, d = {}, a, 2
    while n > 1:
        while n % d == 0:
            a_fac[d] = a_fac.get(d, 0) + 1
            n //= d
        d += 1
    expected = "yes" if all(b_fac.get(p, 0) >= e for p, e in a_fac.items()) else "no"
    assert q.answer == expected
    assert (_value(b_fac) % a == 0) == (expected == "yes")
    assert len(q.distractors) == 1 and q.distractors[0] != q.answer
    assert {q.answer, q.distractors[0]} == {"yes", "no"}


@pytest.mark.parametrize("seed", SEEDS)
def test_square_factored_key_and_distractor(seed):
    q = make("prime_composite_square_properties", seed=seed, difficulty="hard", form="factored")
    m = re.fullmatch(r"Is (.+) a perfect square\?", q.text)
    assert m, q.text
    fac = _parse_factored(m.group(1))
    expected = "yes" if all(e % 2 == 0 for e in fac.values()) else "no"
    assert q.answer == expected
    n = _value(fac)
    assert (math.isqrt(n) ** 2 == n) == (expected == "yes")
    assert len(q.distractors) == 1 and q.distractors[0] != q.answer


def test_both_answers_occur():
    for topic in ("factors_multiples_divisibility", "prime_composite_square_properties"):
        answers = {make(topic, seed=s, difficulty="hard", form="factored").answer for s in range(40)}
        assert answers == {"yes", "no"}


@pytest.mark.parametrize("topic", ["factors_multiples_divisibility", "prime_composite_square_properties"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_default_form_output_unchanged_by_new_param(topic, difficulty):
    for seed in range(60):
        a = make(topic, seed=seed, difficulty=difficulty)
        b = make(topic, seed=seed, difficulty=difficulty, form="number")
        assert (a.text, a.answer, a.distractors) == (b.text, b.answer, b.distractors)


def test_yaml_hard_sources_use_factored_and_easy_medium_untouched():
    for sid in ("f10-factors-multiples-divisibility", "f10-prime-composite-square-properties"):
        d = yaml.safe_load((ROOT / "skills" / "secondary-number" / f"{sid}.yaml").read_text(encoding="utf-8"))
        by = {s["difficulty"]: s["params"] for s in d["sources"]}
        assert by["hard"] == {"form": "factored", "choose": True}  # w2: "choose" gives these skills 4 options
        assert by["easy"] == {} and by["medium"] == {}
