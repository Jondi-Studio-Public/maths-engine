"""Tests `number-bases-primes-and-divisors` (topic `w7b_number_systems_primes`) and its exam chain.

The oracles read the numerals, numbers and factorisations back out of the question TEXT and re-derive the key with
`int(numeral, base)`, `bin`/`oct`, SymPy's number theory functions and `math`, never from the generator's helpers.
"""

from __future__ import annotations

import math
import re

import pytest
import sympy

import mathgen.generators.w7b_number_systems_primes_build  # noqa: F401  (registers the topics)
from tests import w7b_support as sup

TOPIC = "w7b_number_systems_primes"
EXAM = "exd_x_w7b_number_systems_primes"
SKILL = sup.skill_of("number-bases-primes-and-divisors")

_SUP = {c: str(i) for i, c in enumerate("⁰¹²³⁴⁵⁶⁷⁸⁹")}


def _factored(expr: str) -> int:
    n = 1
    for term in expr.split(" × "):
        base = re.match(r"\d+", term).group(0)
        exp = "".join(_SUP[c] for c in term[len(base) :]) or "1"
        n *= int(base) ** int(exp)
    return n


def _square_divisors(n: int) -> int:
    return sum(1 for d in range(1, n + 1) if n % d == 0 and math.isqrt(d) ** 2 == d)


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM, "number")
    assert "nsp-gcd-lcm-swapped" in {m.id for m in SKILL.misconceptions}
    steps = [SKILL.difficulty_progression[t].steps for t in ("easy", "medium", "hard", "exam")]
    assert steps == sorted(steps) and len(set(steps)) == 4


@pytest.mark.parametrize("tier", sup.TIERS)
def test_key_accepted_and_distractors_rejected(tier):
    sup.check_single_tier(SKILL, TOPIC, tier)


def test_pools_and_shapes():
    sup.check_pools(TOPIC)


def test_no_answer_or_option_starts_with_zero_or_has_a_letter():
    for tier in sup.TIERS:
        for seed in range(80):
            q = sup.single(TOPIC, tier, seed)
            for text in [q.answer_text, *q.distractors]:
                assert re.fullmatch(r"[1-9]\d*", text), (tier, seed, text)


def test_easy_keys_by_independent_conversion():
    seen = set()
    for seed in range(200):
        q = sup.single(TOPIC, "easy", seed)
        t = q.text
        if m := re.search(r"binary (?:number|numeral) ([01]+)", t):
            assert int(q.answer) == int(m.group(1), 2), (seed, t)
            seen.add("bin-to-dec")
        elif m := re.search(r"(?:decimal number (\d+) in binary|Convert (\d+) to binary|is (\d+) in base 2)", t):
            n = int(next(g for g in m.groups() if g))
            assert q.answer_text == bin(n)[2:], (seed, t)
            seen.add("dec-to-bin")
        else:
            m = re.search(r"(octal|hexadecimal) number ([0-9A-F]+)", t)
            base = {"octal": 8, "hexadecimal": 16}[m.group(1)]
            assert int(q.answer) == int(m.group(2), base), (seed, t)
            seen.add(m.group(1))
    assert seen == {"bin-to-dec", "dec-to-bin", "octal", "hexadecimal"}


def test_medium_keys():
    seen = set()
    for seed in range(300):
        q = sup.single(TOPIC, "medium", seed)
        t = q.text
        if "to octal" in t or "in base 8" in t:
            b = re.search(r"binary number ([01]+)", t).group(1)
            assert q.answer_text == oct(int(b, 2))[2:], (seed, t)
            seen.add("octal")
        elif "to binary" in t or ("hexadecimal number" in t and "in binary" in t):
            h = re.search(r"hexadecimal number ([0-9A-F]+)", t).group(1)
            assert q.answer_text == bin(int(h, 16))[2:], (seed, t)
            seen.add("hex-to-bin")
        elif "Add the binary" in t or "in binary arithmetic" in t:
            a, b = re.findall(r"\b([01]{4,})\b", t)[:2]
            assert q.answer_text == bin(int(a, 2) + int(b, 2))[2:], (seed, t)
            seen.add("sum")
        elif "smallest prime factor" in t:
            n = (
                int(re.match(r"(?:By trial division.*?of )?(\d+)", t).group(1))
                if t[0].isdigit()
                else int(re.search(r"factor of (\d+)", t).group(1))
            )
            assert int(q.answer) == min(sympy.factorint(n)), (seed, t)
            seen.add("smallest")
        else:
            n = int(re.search(r"whether (\d+) is prime", t).group(1))
            root = math.isqrt(n)
            if "largest prime" in t:
                assert int(q.answer) == sympy.prevprime(root + 1), (seed, t)
                seen.add("largest")
            else:
                assert int(q.answer) == sympy.primepi(root), (seed, t)
                seen.add("count")
    assert seen == {"octal", "hex-to-bin", "sum", "smallest", "largest", "count"}


def test_hard_keys():
    seen = set()
    for seed in range(300):
        q = sup.single(TOPIC, "hard", seed)
        t = q.text
        if t.startswith("Let N"):
            expr = re.match(r"Let N = ([^.]+?)\. ", t).group(1)
            n = int(expr) if expr.isdigit() else _factored(expr)
            if "perfect squares" in t:
                assert int(q.answer) == _square_divisors(n), (seed, t)
                seen.add("squares")
            elif "sum of all" in t:
                assert int(q.answer) == sympy.divisor_sigma(n), (seed, t)
                seen.add("sigma")
            else:
                assert int(q.answer) == sympy.divisor_count(n), (seed, t)
                seen.add("count")
        elif "greatest common divisor" in t or "lowest common multiple" in t:
            a, b = (int(x) for x in re.findall(r"(\d+) and (\d+)", t)[0])
            want = math.gcd(a, b) if "greatest" in t else math.lcm(a, b)
            assert int(q.answer) == want, (seed, t)
            seen.add("gcd" if "greatest" in t else "lcm")
        elif "product of two primes" in t:
            n = int(re.match(r"(\d+) is", t).group(1))
            smallest = min(sympy.factorint(n))
            assert int(q.answer) == sympy.primepi(smallest), (seed, t)
            seen.add("trial")
        else:
            h = re.search(r"hexadecimal number ([0-9A-F]+)", t).group(1)
            assert int(q.answer) == sympy.divisor_count(int(h, 16)), (seed, t)
            seen.add("hex-divisors")
    assert seen == {"squares", "sigma", "count", "gcd", "lcm", "trial", "hex-divisors"}


def test_exam_chain_shape():
    sup.check_exam_chain(SKILL, EXAM, [1, 2, 2, 2], ["number"] * 4)


def test_exam_keys_by_independent_arithmetic():
    for seed in range(100):
        q = sup.exam(EXAM, seed)
        h = re.search(r"hexadecimal number ([0-9A-F]+)", q.stem).group(1)
        n = int(h, 16)
        a, b, c, d = q.parts
        assert int(a.answer) == n
        assert int(b.answer) == sympy.primepi(math.isqrt(n))
        assert int(c.answer) == sympy.divisor_count(n)
        m = int(re.search(r"and (\d+)\.", d.prompt).group(1))
        assert int(d.answer) == math.gcd(n, m)
        assert math.gcd(n, m) not in (1, m, n)


def test_answers_are_not_guessable():
    for tier in sup.TIERS:
        spread = sup.answer_spread(TOPIC, tier, 120)
        assert len(spread) >= 25, (tier, len(spread))
        assert max(spread.values()) <= 0.2 * 120, (tier, spread.most_common(3))
