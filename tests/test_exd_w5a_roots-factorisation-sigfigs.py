"""Hard-tier change in kind for square roots and prime factorisation.

Keys are recomputed here from the rendered prompt, not from generator internals.
"""

import math
import random
import re

import pytest
from sympy import factorint

from mathgen.generators.acara_gap_build import prime_factorisation_exponent
from mathgen.generators.p3a_secondary_number_powers_build import square_roots_perfect_squares

SUP = {ord(c): str(i) for i, c in enumerate("⁰¹²³⁴⁵⁶⁷⁸⁹")}
SEEDS = range(300)


def _term_value(term: str) -> int:
    m = re.fullmatch(r"(\d+)([⁰¹²³⁴⁵⁶⁷⁸⁹]*)", term.strip())
    base = int(m.group(1))
    exp = int(m.group(2).translate(SUP) or "1")
    return base**exp


def _parse_product(text: str) -> int:
    expr = re.search(r"(?:Is|Write) (.+?) (?:a perfect|as a product)", text).group(1)
    return math.prod(_term_value(t) for t in expr.split("×"))


def _terms(text: str):
    expr = re.search(r"(?:Is|Write) (.+?) (?:a perfect|as a product)", text).group(1)
    return [t.strip() for t in expr.split("×")]


@pytest.mark.parametrize("seed", SEEDS)
def test_square_power_form_key_and_distractors(seed):
    q = square_roots_perfect_squares(random.Random(seed), power_form=True)
    n = _parse_product(q["text"])
    root = math.isqrt(n)
    expected = str(root) if root * root == n else "not a perfect square"
    assert q["answer"] == expected
    assert len(q["distractors"]) == 3
    assert len(set(q["distractors"])) == 3
    assert q["answer"] not in q["distractors"]
    # A distractor that is a numeric root must not be the real root.
    for d in q["distractors"]:
        if d != "not a perfect square" and expected != "not a perfect square":
            assert int(d) != root


def test_square_power_form_covers_both_outcomes():
    answers = {
        square_roots_perfect_squares(random.Random(s), power_form=True)["answer"] == "not a perfect square"
        for s in SEEDS
    }
    assert answers == {True, False}


@pytest.mark.parametrize("seed", SEEDS)
def test_square_default_path_unchanged(seed):
    a = square_roots_perfect_squares(random.Random(seed), root_range=(1, 20))
    b = square_roots_perfect_squares(random.Random(seed), root_range=(1, 20), power_form=False)
    assert a == b
    assert "×" not in a["text"]


@pytest.mark.parametrize("seed", SEEDS)
def test_prime_composite_form_key_and_distractors(seed):
    q = prime_factorisation_exponent(
        random.Random(seed), primes=(2, 3, 5, 7), term_hi=3, exponent_hi=3, composite_form=True
    )
    terms = _terms(q["text"])
    assert any(int(re.match(r"\d+", t).group()) not in (2, 3, 5, 7) for t in terms)
    n = _parse_product(q["text"])
    f = factorint(n)
    expected = " × ".join(
        str(p) if e == 1 else f"{p}{str(e).translate(str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'))}"
        for p, e in sorted(f.items())
    )
    assert q["answer"] == expected
    assert q["answer"] not in q["distractors"]
    assert len(set(q["distractors"])) == len(q["distractors"]) >= 2


@pytest.mark.parametrize("seed", SEEDS)
def test_prime_default_path_unchanged(seed):
    kw = dict(primes=(2, 3, 5, 7, 11), term_hi=4, exponent_hi=4)
    assert prime_factorisation_exponent(random.Random(seed), **kw) == prime_factorisation_exponent(
        random.Random(seed), composite_form=False, **kw
    )
