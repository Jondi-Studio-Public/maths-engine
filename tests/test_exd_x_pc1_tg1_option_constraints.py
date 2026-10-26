"""Every wrong option of every part of `exd_x_pc1_contradiction_chain` and `exd_x_tg1_extended_domain_chain` satisfies the
constraints the stem puts on an answer, over many seeds, family by family.

An option the stem rules out (a remainder as big as the divisor, an integer "greater than C" that is not, a prime that is
in the listed set, a count the figure cannot have) gives the key away by elimination. The constraints audited:

pc1, Euclid:   (a) a whole number above 1; (b) a prime, not in the listed set (the stem says so); (c) a whole number from
               2 to M/2 (M = p x q with p a prime), and not above M.
pc1, linear:   (a) from 1 to the smaller coefficient (it divides A m + B n at m = 1, n = 0, and the same for B);
               (b) from 1 to g - 1, a remainder on division by (a)'s key g; (c) from 1 to C/2 (the divisor is at least 2);
               (d) strictly greater than C ("the smallest integer greater than C").
pc1, residue:  (a) from 1 to m - 1 ("from 0 to m - 1", and not 0 since the claim is true); (b) from 2 to m - 1 (0 and 1 are
               always square remainders and the needed remainder never is); (c) from 1 to m - 2 (same reason);
               (d) from 3 to 2m - 1 (a sum of two different positive integers k, each at most m).
tg1:           every option of (a)-(d) is a positive whole number; sine/cosine (c) is below (b)'s key (turning points
               alternate, so some are minima); tangent (c) is not (b)'s key (pieces = asymptotes + 1); (e) is between n*lo and n*hi (n = the
               count from (c) or (d), lo and hi the domain ends in half-pi or quarter-pi), has the sign of a one-sided
               domain (0 is a legitimate option), and is in the key's class mod 4 (the maxima of sine and cosine sit at 4n or 4n + 1 half-pi, the tan x = 1 points at
               1 + 4n quarter-pi).
All: no option repeats the key, all options are whole numbers.
"""

from __future__ import annotations

import math
import re

import pytest

from mathgen import make

SEEDS = range(900)


def _prime(n):
    return n > 1 and all(n % d for d in range(2, math.isqrt(n) + 1))


def _pi(text):
    """'-3π/2' -> Fraction(-3, 2); '0' -> 0."""
    from fractions import Fraction

    if text == "0":
        return Fraction(0)
    m = re.fullmatch(r"(-?)(\d*)π(?:/(\d+))?", text)
    return (-1 if m.group(1) else 1) * Fraction(int(m.group(2) or 1), int(m.group(3) or 1))


def _opts(part):
    return [int(x) for x in part.distractors]


def _family(q):
    if "primes" in q.stem:
        return "euclid"
    if re.search(r"\d+m \+ \d+n = \d+", q.stem):
        return "linear"
    return "residue"


@pytest.fixture(scope="module")
def pc1():
    return [make("exd_x_pc1_contradiction_chain", seed=s, difficulty="exam") for s in SEEDS]


@pytest.fixture(scope="module")
def tg1():
    return [make("exd_x_tg1_extended_domain_chain", seed=s, difficulty="exam") for s in SEEDS]


def test_every_pc1_option_meets_the_stem_constraints(pc1):
    seen = set()
    for q in pc1:
        kind = _family(q)
        seen.add(kind)
        for p in q.parts:
            assert p.answer_text not in p.distractors and len(set(p.distractors)) == len(p.distractors)
        if kind == "euclid":
            listed = [int(x) for x in re.search(r"(\d+(?:, \d+)+)", q.stem).group(1).split(", ")]
            m = int(q.parts[0].answer_text)
            assert all(x > 1 for x in _opts(q.parts[0])), q.text
            assert all(_prime(x) and x not in listed for x in _opts(q.parts[1])), q.text
            low = int(q.parts[1].answer_text)
            # q = M / p: at least p (all its prime factors are), at most M / 2, odd (M is odd) and coprime to the list
            assert all(low <= x <= m // 2 for x in _opts(q.parts[2])), q.text
            assert all(x % 2 == 1 and all(x % p for p in listed) for x in _opts(q.parts[2])), q.text
            assert all(x * x <= m for x in _opts(q.parts[1])), q.text  # the smallest prime factor is at most sqrt(M)
        elif kind == "linear":
            a_coef, b_coef, const = (int(x) for x in re.search(r"(\d+)m \+ (\d+)n = (\d+)", q.stem).groups())
            g = int(q.parts[0].answer_text)
            assert all(1 <= x <= min(a_coef, b_coef) for x in _opts(q.parts[0])), q.text
            assert all(1 <= x <= g - 1 for x in _opts(q.parts[1])), q.text
            assert all(1 <= x <= const // 2 for x in _opts(q.parts[2])), q.text
            assert all(x > const for x in _opts(q.parts[3])), (q.text, q.parts[3].distractors)
        else:
            m = int(re.search(r"divided by (\d+)", q.parts[0].prompt).group(1))
            assert all(1 <= x <= m - 1 for x in _opts(q.parts[0])), q.text
            assert all(2 <= x <= m - 1 for x in _opts(q.parts[1])), q.text
            assert all(1 <= x <= m - 2 for x in _opts(q.parts[2])), q.text
            assert all(3 <= x <= 2 * m - 1 for x in _opts(q.parts[3])), q.text
    assert seen == {"euclid", "linear", "residue"}


def test_every_tg1_option_meets_the_stem_constraints(tg1):
    names = set()
    for q in tg1:
        stem = q.stem
        name = (
            "sin"
            if "y = sin x" in stem or "greatest value of 1" in stem
            else "cos"
            if "y = cos x" in stem or "crosses the y-axis" in stem
            else "tan"
        )
        names.add(name)
        a, b, c, d, e = q.parts
        for p in q.parts:
            assert p.answer_text not in p.distractors
        for p in (a, b, c, d):
            assert all(x >= 1 for x in _opts(p)), q.text
        if name == "tan":
            assert int(b.answer_text) not in _opts(c), q.text
        else:
            assert all(x < int(b.answer_text) for x in _opts(c)), q.text
        key = int(e.answer_text)
        assert all(x % 4 == key % 4 for x in _opts(e)), (q.text, key, e.distractors)
        # (e) is a sum of n x-coordinates, each between the ends of the domain, in units of pi/2 (sine, cosine, n = the
        # maxima counted in (c)) or pi/4 (tangent, n = the points counted in (d)): so n*lo <= sum <= n*hi
        lo_t, hi_t = re.search(r"(0|-?\d*π(?:/\d+)?) ≤ x ≤ (0|-?\d*π(?:/\d+)?)", stem).groups()
        unit, n = (4, int(d.answer_text)) if name == "tan" else (2, int(c.answer_text))
        lo_u, hi_u = _pi(lo_t) * unit, _pi(hi_t) * unit
        assert lo_u.denominator == hi_u.denominator == 1
        assert all(n * lo_u <= x <= n * hi_u for x in [key, *_opts(e)]), (q.text, key, e.distractors)
        if lo_u >= 0:  # a one-sided domain: every x-coordinate, so the sum, has that sign (0 is a legitimate option)
            assert all(x >= 0 for x in _opts(e)), (q.text, e.distractors)
        if hi_u <= 0:
            assert all(x <= 0 for x in _opts(e)), (q.text, e.distractors)
    assert names == {"sin", "cos", "tan"}


def test_the_one_sided_tangent_domain_of_seed_1347_offers_no_negative_sum():
    """Reviewer case: tan on 0 <= x <= 6π (key 66) once offered -66."""
    q = make("exd_x_tg1_extended_domain_chain", seed=1347, difficulty="exam")
    if "tan x" in q.stem and re.search(r"0 ≤ x ≤", q.stem):
        assert all(int(x) >= 0 for x in q.parts[4].distractors)
    # and the same on every one-sided tangent domain among many seeds
    checked = 0
    for seed in range(1500):
        q = make("exd_x_tg1_extended_domain_chain", seed=seed, difficulty="exam")
        m = re.search(r"(0|-?\d*π(?:/\d+)?) ≤ x ≤ (0|-?\d*π(?:/\d+)?)", q.stem)
        if "asymptote at x" in q.stem or "y = tan x" in q.stem:
            if _pi(m.group(1)) >= 0:
                checked += 1
                assert all(int(x) >= 0 for x in q.parts[4].distractors), (seed, q.parts[4].distractors)
    assert checked >= 20
