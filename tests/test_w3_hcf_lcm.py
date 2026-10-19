"""Tests `hcf-lcm-word-problems` (topic `w3_hcf_lcm_words` and the exam chain `exd_x_w3_hcf_lcm_words`).

The oracles read the numbers back out of the question TEXT and re-derive each key with `math.gcd` and `math.lcm`, or by
brute-force search where the question asks for a divisor with remainders or a count of meetings.
"""

from __future__ import annotations

import re
from math import gcd, lcm

import mathgen.generators.w3_hcf_lcm_build as mod
from tests import w3_support as w3
from tests import w7b_support as sup

T, X = "w3_hcf_lcm_words", "exd_x_w3_hcf_lcm_words"
SK = sup.skill_of("hcf-lcm-word-problems")


def test_catalogue_has_every_module_misconception():
    w3.check_catalogue(mod)


class TestShared(w3.shared_checks(SK, T, X)):
    pass


def nums(text):
    return [int(n) for n in re.findall(r"\d+", text)]


def test_easy_keys():
    seen = set()
    for seed in range(200):
        q = sup.single(T, "easy", seed)
        t, ans = q.text, int(q.answer_text)
        if "How many pens go in each bag" in t or "How many pencils go in each bag" in t:
            pens, pencils = nums(t)[:2]
            g = gcd(pens, pencils)
            assert ans == (pens if "How many pens" in t else pencils) // g, (seed, t)
            seen.add("share")
        elif "next" in t:
            a, b = (int(x) for x in re.findall(r"(\d+) (?:minutes|seconds)", t)[:2])
            assert ans == lcm(a, b), (seed, t)
            seen.add("lcm")
        else:
            a, b = nums(t)[:2]
            assert ans == gcd(a, b), (seed, t)
            seen.add("hcf")
    assert seen == {"share", "lcm", "hcf"}


def test_medium_keys():
    seen = set()
    for seed in range(250):
        q = sup.single(T, "medium", seed)
        t, ans = q.text, int(q.answer_text)
        if "timber lengths" in t:
            a, b, c = nums(t)[:3]
            assert ans == gcd(a, b, c), (seed, t)
            seen.add("hcf3")
        elif "lighthouses" in t:
            a, b, c = nums(t)[:3]
            assert ans == lcm(a, b, c), (seed, t)
            seen.add("lcm3")
        elif "gears" in t:
            a, b = nums(t)[:2]
            assert ans == lcm(a, b) // min(a, b), (seed, t)
            seen.add("gears")
        elif "How many trips" in t:
            a, b = (int(x) for x in re.findall(r"every (\d+) minutes", t)[:2])
            who = re.search(r"does Bus ([AB]) make", t).group(1)
            assert ans == lcm(a, b) // (a if who == "A" else b), (seed, t)
            seen.add("trips")
        else:
            seen.add(("tiles", t))
            assert "tiles" in t
            a, b = nums(t)[:2]
            assert ans > 0
    assert {"hcf3", "lcm3", "gears", "trips"} <= seen
    assert any(isinstance(s, tuple) for s in seen)


def test_medium_tiles_keys():
    done = 0
    for seed in range(250):
        q = sup.single(T, "medium", seed)
        t = q.text
        if "tiles" not in t:
            continue
        done += 1
        a, b = nums(t)[:2]
        ans = int(q.answer_text)
        # a rectangular floor is covered by square tiles of the greatest side that fits both sides exactly
        side = gcd(a, b)
        assert ans in {(a // side) * (b // side), (a * b) // (side * side)}, (seed, t)
    assert done >= 20


def test_hard_keys():
    seen = set()
    for seed in range(300):
        q = sup.single(T, "hard", seed)
        t, ans = q.text, int(q.answer_text)
        if "Rectangular tiles" in t:
            a, b = nums(t)[:2]
            L = lcm(a, b)
            assert ans == (L // a) * (L // b), (seed, t)
            seen.add("square")
        elif "is divided by a certain whole number n" in t:
            a, r1, b, r2 = nums(t)[:4]
            best = max(n for n in range(1, a + 1) if a % n == r1 and b % n == r2)
            assert ans == best, (seed, t)
            seen.add("rem")
        elif "smallest whole number greater than" in t:
            r, _, x, y, z = nums(t)[:5]
            n = r + 1
            while not (n % x == r and n % y == r and n % z == r):
                n += 1
            assert ans == n, (seed, t)
            seen.add("smallest")
        elif "sound together" in t:
            a, b = nums(t)[:2]
            hours = int(re.search(r"until (\d+):00 pm", t).group(1)) if "pm" in t else 12
            minutes = hours * 60
            count = sum(1 for m in range(0, minutes + 1) if m % a == 0 and m % b == 0)
            assert ans == count, (seed, t)
            seen.add("meetings")
        else:
            h, L, known = nums(t)[:3]
            other = ans
            assert gcd(known, other) == h and lcm(known, other) == L, (seed, t)
            seen.add("other")
    assert seen == {"square", "rem", "smallest", "meetings", "other"}


def test_exam_keys_and_shape():
    sup.check_exam_chain(SK, X, [2, 1, 1, 2], ["number"] * 4)
    for seed in range(100):
        q = sup.exam(X, seed)
        a, b = nums(q.stem)[:2]
        pa, pb, pc, pd = q.parts
        L = lcm(a, b)
        assert int(pa.answer_text) == L
        assert int(pb.answer_text) == L // a
        assert int(pc.answer_text) == L // b
        seconds = 60 * int(re.search(r"for (\d+) minutes", pd.prompt).group(1))
        assert int(pd.answer_text) == sum(1 for s in range(seconds + 1) if s % a == 0 and s % b == 0)


def test_exam_follow_through():
    def wrong(q, canon):
        # a wrong but consistent first answer: twice the true meeting time
        bad = 2 * int(q.parts[0].answer_text)
        a, b = nums(q.stem)[:2]
        return {"a": str(bad), "b": str(bad // a), "c": str(bad // b)}

    assert sup.check_follow_through(SK, X, "b", wrong) >= 30
