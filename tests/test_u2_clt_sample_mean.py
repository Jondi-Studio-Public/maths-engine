"""Tests `ucv1-clt-sample-mean-probability` (topics `u2_clt_sample_mean` and its exam chain).

Oracles read the numbers back out of the question TEXT and recompute the answer with the standard library's
`statistics.NormalDist` (a different implementation from the generator's `math.erf` and mpmath routes).
"""

from __future__ import annotations

import math
import re
from statistics import NormalDist

from mathgen import make
from tests import u2_support as su

TOPIC = "u2_clt_sample_mean"
EXAM = "exd_x_u2_clt_sample_mean"
SK = su.skill("ucv1-clt-sample-mean-probability")
N = NormalDist()


def _nums(text):
    mu, sigma = re.search(r"mean (\d+(?:\.\d+)?) \S+ and standard deviation (\d+)", text).groups()
    n = int(re.search(r"sample of (\d+)", text).group(1))
    return float(mu), float(sigma), n


def _four(p):
    return f"{p:.4f}"


def test_wiring():
    su.check_wiring(SK, TOPIC, EXAM, "number")


def test_key_and_distractors_each_tier():
    for tier in su.TIERS:
        su.check_tier(SK, TOPIC, tier, seeds=60)


def test_easy_key_is_the_tail_of_the_sample_mean():
    sides = set()
    for seed in range(120):
        q = make(TOPIC, seed=seed, difficulty="easy")
        mu, sigma, n = _nums(q.text)
        side, c = re.search(r"sample mean \S+(?: \S+)? is (more|less) than (\d+(?:\.\d+)?)", q.text).groups()
        z = (float(c) - mu) / (sigma / math.sqrt(n))
        p = 1 - N.cdf(z) if side == "more" else N.cdf(z)
        assert q.answer_text == _four(p), (seed, q.text, q.answer_text, _four(p))
        sides.add(side)
    assert sides == {"more", "less"}


def test_medium_keys_total_and_within():
    kinds = set()
    for seed in range(160):
        q = make(TOPIC, seed=seed, difficulty="medium")
        mu, sigma, n = _nums(q.text)
        if "total" in q.text:
            kinds.add("total")
            word, t = re.search(r"(exceeds|is less than) (\d+(?:\.\d+)?)", q.text).groups()
            z = (float(t) - n * mu) / (sigma * math.sqrt(n))
            p = 1 - N.cdf(z) if word == "exceeds" else N.cdf(z)
        else:
            kinds.add("within")
            d = float(re.search(r"within (\d+(?:\.\d+)?)", q.text).group(1))
            z = d / (sigma / math.sqrt(n))
            p = N.cdf(z) - N.cdf(-z)
        assert q.answer_text == _four(p), (seed, q.text, q.answer_text, _four(p))
    assert kinds == {"total", "within"}


def test_hard_key_is_the_least_sufficient_n():
    levels = set()
    for seed in range(160):
        q = make(TOPIC, seed=seed, difficulty="hard")
        mu, sigma, _ = (
            *_nums(q.text.replace("Use the central", "A random sample of 1 is taken. Use the central"))[:2],
            0,
        )
        d = float(re.search(r"(?:within|by more than) (\d+(?:\.\d+)?)", q.text).group(1))
        if "within" in q.text:
            conf = float(re.search(r"at least (0\.\d+)", q.text).group(1))
            z = N.inv_cdf((1 + conf) / 2)
        else:
            alpha = float(re.search(r"at most (0\.\d+)", q.text).group(1))
            z = N.inv_cdf(1 - alpha)
        levels.add(round(z, 1))
        n = int(q.answer_text)
        # With the table z-value rounded to the book's decimals, n suffices and n - 1 does not.
        assert d * math.sqrt(n) / sigma >= z - 0.004, (seed, q.text, n)
        assert d * math.sqrt(n - 1) / sigma < z + 0.004, (seed, q.text, n)
    assert len(levels) >= 4


def test_exam_chain():
    su.check_exam(SK, EXAM, "abcd", (2, 3, 3, 3), (False, True, True, True), seeds=20)
    for seed in range(30):
        q = make(EXAM, seed=seed, difficulty="exam")
        mu, sigma, n = (
            float(v) for v in re.search(r"mean (\d+) g and standard deviation (\d+) g.*holds (\d+)", q.stem).groups()
        )
        se = sigma / math.sqrt(n)
        assert q.parts[0].answer_text == f"{se:.2f}"
        c = float(re.search(r"less than (\d+(?:\.\d+)?) g", q.parts[1].prompt).group(1))
        assert q.parts[1].answer_text == _four(N.cdf((c - mu) / se))
        t = float(re.search(r"exceeds (\d+(?:\.\d+)?) g", q.parts[2].prompt).group(1))
        assert q.parts[2].answer_text == _four(1 - N.cdf((t - n * mu) / (sigma * math.sqrt(n))))
        d = float(re.search(r"within (\d+) g", q.parts[3].prompt).group(1))
        assert int(q.parts[3].answer_text) == math.ceil((1.96 * sigma / d) ** 2)


def test_exam_follow_through_chain():
    # A wrong standard error still earns (b) when (b) is right for it.
    def b_value(q, wrong):
        mu, _, _ = (
            float(v) for v in re.search(r"mean (\d+) g and standard deviation (\d+) g.*holds (\d+)", q.stem).groups()
        )
        c = float(re.search(r"less than (\d+(?:\.\d+)?) g", q.parts[1].prompt).group(1))
        return _four(N.cdf((c - mu) / float(wrong)))

    su.check_follow_through(
        SK, EXAM, "a", "b", lambda q: f"{float(q.parts[0].answer_text) + 1.5:.2f}", b_value, seeds=15
    )

    def d_value(q, wrong):
        n = float(re.search(r"holds (\d+)", q.stem).group(1))
        d = float(re.search(r"within (\d+) g", q.parts[3].prompt).group(1))
        return str(math.ceil((1.96 * float(wrong) * math.sqrt(n) / d) ** 2))

    su.check_follow_through(
        SK, EXAM, "a", "d", lambda q: f"{float(q.parts[0].answer_text) + 1.5:.2f}", d_value, seeds=15
    )


def test_variety():
    for tier, minimum in (("easy", 30), ("medium", 30), ("hard", 30)):
        su.distinct_stems(TOPIC, tier, seeds=60, minimum=minimum)
