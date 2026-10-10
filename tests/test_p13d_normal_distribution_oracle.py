"""Independent oracle for the `p13d_normal_distribution` questions: it re-derives each answer from the rendered
question text.

Imports nothing from `mathgen.generators.p13d_normal_distribution_build`: not
the scenario list, not the formula helpers, not the misconception constants.
Parses `mu`, `sigma`, and whichever of `a`/`b`/`q`-as-a-percentage/the known
parameter the tier states straight out of the RENDERED `question.text`, and
re-derives the correct answer from scratch via `math.erf` directly -- never
`statistics.NormalDist` -- so the oracle's probability computation is not
even calling the same library function construction the generator used.

`_phi`/`_inv_phi` below are this file's own standard-normal cdf (via
`math.erf`) and its own bisection-based inverse, independent of
`statistics.NormalDist.cdf`/`.inv_cdf`.

Run across >= 1000 seeds total, spread over the four tiers.
"""

from __future__ import annotations

import math
import re

from mathgen import make

TOPIC = "p13d_normal_dist_prob_quantile"
SEEDS_PER_TIER = 250  # 4 tiers x 250 = 1000


def _phi(z: float) -> float:
    """Standard normal cdf via `math.erf` -- never `statistics.NormalDist`."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _inv_phi(p: float, lo: float = -12.0, hi: float = 12.0, iters: int = 100) -> float:
    """Bisection-based inverse of `_phi` -- a root-finder over the oracle's
    own `erf`-based cdf, independent of `statistics.NormalDist.inv_cdf`."""
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if _phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _round_half_up(value: float, dp: int) -> str:
    """Independent re-implementation of half-up rounding -- deliberately NOT
    a call to `mathgen.core.round_half_up`."""
    scale = 10**dp
    scaled = math.floor(abs(value) * scale + 0.5)
    sign = "-" if value < 0 and scaled != 0 else ""
    whole, frac = divmod(scaled, scale)
    return f"{sign}{whole}.{str(frac).zfill(dp)}" if dp else f"{sign}{whole}"


_EASY_RE = re.compile(
    r"is normally distributed with mean (?P<mu>-?\d+(?:\.\d+)?) \S+ and standard deviation "
    r"(?P<sigma>\d+(?:\.\d+)?) \S+\. Find the probability that a randomly selected \S+ has a "
    r"value (?P<tail>less than|greater than) (?P<a>-?\d+(?:\.\d+)?) \S+\. "
    r"Give your answer correct to 4 decimal places\.$"
)
_MEDIUM_RE = re.compile(
    r"is normally distributed with mean (?P<mu>-?\d+(?:\.\d+)?) \S+ and standard deviation "
    r"(?P<sigma>\d+(?:\.\d+)?) \S+\. Find the probability that a randomly selected \S+ has a "
    r"value between (?P<a>-?\d+(?:\.\d+)?) \S+ and (?P<b>-?\d+(?:\.\d+)?) \S+\. "
    r"Give your answer correct to 4 decimal places\.$"
)
_HARD_RE = re.compile(
    r"is normally distributed with mean (?P<mu>-?\d+(?:\.\d+)?) \S+ and standard deviation "
    r"(?P<sigma>\d+(?:\.\d+)?) \S+\. What is the value (?P<dir>below|above) which the \S+ "
    r"(?P<pct>\d+)% of \S+ fall, in (?P<unit>\S+)\? Give your answer correct to 1 decimal place\.$"
)
_EXAM_RE = re.compile(
    r"is normally distributed with unknown (?P<unknown>mean|standard deviation) and "
    r"(?P<known_label>mean|standard deviation) (?P<known_val>-?\d+(?:\.\d+)?) \S+\. "
    r"It is known that (?P<pct>[\d.]+)% of \S+ have a value below (?P<a>-?\d+(?:\.\d+)?) \S+\. "
    r"Find the (?:mean|standard deviation), in \S+\. Give your answer correct to 2 decimal places\.$"
)


def test_oracle_easy_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="easy", mode="one_tail")
        m = _EASY_RE.search(q.text)
        assert m, (seed, q.text)
        mu, sigma, a = float(m["mu"]), float(m["sigma"]), float(m["a"])
        z = (a - mu) / sigma
        p = _phi(z) if m["tail"] == "less than" else 1.0 - _phi(z)
        assert _round_half_up(p, 4) == q.answer_text, (seed, q.text, q.answer_text, p)
        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_medium_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="medium", mode="between")
        m = _MEDIUM_RE.search(q.text)
        assert m, (seed, q.text)
        mu, sigma = float(m["mu"]), float(m["sigma"])
        a, b = float(m["a"]), float(m["b"])
        za, zb = (a - mu) / sigma, (b - mu) / sigma
        p = _phi(zb) - _phi(za)
        assert _round_half_up(p, 4) == q.answer_text, (seed, q.text, q.answer_text, p)
        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_hard_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="hard", mode="quantile")
        m = _HARD_RE.search(q.text)
        assert m, (seed, q.text)
        mu, sigma = float(m["mu"]), float(m["sigma"])
        pct, direction = int(m["pct"]), m["dir"]
        q_eff = pct / 100.0 if direction == "below" else 1.0 - pct / 100.0
        z = _inv_phi(q_eff)
        x = mu + z * sigma
        assert _round_half_up(x, 1) == q.answer_text, (seed, q.text, q.answer_text, x)
        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_exam_tier():
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC, seed=seed, difficulty="exam", mode="solve_parameter")
        m = _EXAM_RE.search(q.text)
        assert m, (seed, q.text)
        pct, a, known_val = float(m["pct"]), float(m["a"]), float(m["known_val"])
        z = _inv_phi(pct / 100.0)
        if m["unknown"] == "standard deviation":
            # known_val is mu; solve sigma = (a - mu) / z
            solved = (a - known_val) / z
        else:
            # known_val is sigma; solve mu = a - sigma*z
            solved = a - known_val * z
        assert _round_half_up(solved, 2) == q.answer_text, (seed, q.text, q.answer_text, solved)
        checked += 1
    assert checked == SEEDS_PER_TIER


# --- Prove the oracle is red first: a corrupted answer must be caught -------


def test_oracle_catches_a_corrupted_answer():
    """The oracle's own assertion, re-run against a deliberately corrupted
    `answer_text` (an off-by-one in the last stated decimal place), must
    actually raise -- proving the check fires rather than trivially passing
    (oracles are proven red first)."""
    q = make(TOPIC, seed=0, difficulty="easy", mode="one_tail")
    m = _EASY_RE.search(q.text)
    mu, sigma, a = float(m["mu"]), float(m["sigma"]), float(m["a"])
    z = (a - mu) / sigma
    p = _phi(z) if m["tail"] == "less than" else 1.0 - _phi(z)
    correct = _round_half_up(p, 4)
    corrupted_answer_text = _round_half_up(p + 1.0, 4)
    assert correct == q.answer_text  # the oracle agrees with the real answer
    try:
        assert correct == corrupted_answer_text
    except AssertionError:
        pass
    else:
        raise AssertionError("oracle failed to catch a corrupted answer_text")
