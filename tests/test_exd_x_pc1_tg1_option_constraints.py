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
               alternate, so some are minima); tangent (c) is not (b)'s key (pieces = asymptotes + 1); (e) is in the
               key's class mod 4 (the maxima of sine and cosine sit at 4n or 4n + 1 half-pi, the tan x = 1 points at
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
            assert all(2 <= x <= m // 2 for x in _opts(q.parts[2])), q.text
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
    assert names == {"sin", "cos", "tan"}
