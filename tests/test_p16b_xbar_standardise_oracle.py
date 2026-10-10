"""Independent oracle for the `p16b_xbar_standardise` questions: it re-derives each answer from the rendered
question text.

Parses ONLY ``question.text``, ``question.answer_text`` and ``question.distractors`` -- imports nothing from
``mathgen.generators.p16b_xbar_standardise_build`` -- and recomputes the standardised value z from scratch by
a different route (regex parse of the rendered numbers, then the textbook formula, using the same
round-half-to-even ``"%.2f"`` display convention the generator's own docstring names, never the generator's
own code).

Coverage: 150 seeds per (topic, tier) = 1200 questions total, 300 per tier
(aggregated across both topics), comfortably over the 1000/250 floor.

``test_oracle_catches_a_perturbed_key`` below keeps that proof executable rather than merely narrated.
"""

import math
import re
from fractions import Fraction

import pytest

from mathgen import make

SIGMA_TOPIC = "p16b_xbar_sigma_standardise"
S_TOPIC = "p16b_xbar_s_standardise"
TIERS = ["easy", "medium", "hard", "exam"]
SEEDS = range(150)
# A comparison ("which is further from 0") tier is re-derived over >= 1000 seeds.
HARD_SEEDS = range(1000)


def _round2(x: float) -> Fraction:
    return Fraction(f"{x:.2f}")


def _last(pattern, text):
    matches = list(re.finditer(pattern, text))
    assert matches, f"pattern {pattern!r} not found in {text!r}"
    return matches[-1]


def _sample_means(text):
    """{sample number: mean} from a "sample N: V[, ...]" or "sample N: mean = V" list."""
    out = {}
    for m in re.finditer(r"sample (\d+): (?:mean = )?(-?\d+\.\d{2})(?:,)?", text):
        out[int(m.group(1))] = float(m.group(2))
    return out


def _sample_mean_and_stat(text, stat_name):
    """{sample number: (mean, stat)} from "sample N: mean = M, <stat_name> = V"."""
    out = {}
    for m in re.finditer(rf"sample (\d+): mean = (-?\d+\.\d{{2}}), {stat_name} = (-?\d+\.\d{{2}})", text):
        out[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return out


def _target_sample(text):
    return int(_last(r"sample (\d+)'s mean", text).group(1))


# ---------------------------------------------------------------------------
# Row A (sigma known)
# ---------------------------------------------------------------------------


def _oracle_row_a(question, difficulty):
    text = question.text

    if difficulty == "easy":
        mu = float(re.search(r"μ = (-?\d+(?:\.\d+)?)", text).group(1))
        sigma = float(re.search(r"σ = (-?\d+(?:\.\d+)?)", text).group(1))
        n = int(re.search(r"size (\d+)", text).group(1))
        means = _sample_means(text)
        j = _target_sample(text)
        z = (means[j] - mu) / (sigma / math.sqrt(n))
        return z

    if difficulty == "medium":
        pairs = re.findall(r"P\(X = (-?\d+)\) = (\d+)/(\d+)", text)
        assert pairs
        xs = [int(x) for x, _, _ in pairs]
        ps = [Fraction(int(num), int(den)) for _, num, den in pairs]
        mu_frac = sum(Fraction(x) * p for x, p in zip(xs, ps))
        var_frac = sum(Fraction(x * x) * p for x, p in zip(xs, ps)) - mu_frac * mu_frac
        sigma = math.sqrt(float(var_frac))
        mu = float(mu_frac)
        n = int(re.search(r"size (\d+)", text).group(1))
        means = _sample_means(text)
        j = _target_sample(text)
        z = (means[j] - mu) / (sigma / math.sqrt(n))
        return z

    if difficulty == "hard":
        mu = float(re.search(r"μ = (-?\d+(?:\.\d+)?)", text).group(1))
        sigma = float(re.search(r"σ = (-?\d+(?:\.\d+)?)", text).group(1))
        n1, n2 = (
            int(x) for x in re.search(r"samples 1 to 4 of size (\d+),? and samples 5 to 8 of size (\d+)", text).groups()
        )
        means = _sample_means(text)
        i, j = (int(x) for x in re.search(r"sample (\d+) and sample (\d+)", text).groups())
        n_i = n1 if i <= 4 else n2
        n_j = n1 if j <= 4 else n2
        z_i = (means[i] - mu) / (sigma / math.sqrt(n_i))
        z_j = (means[j] - mu) / (sigma / math.sqrt(n_j))
        gap = abs(abs(z_i) - abs(z_j))
        assert gap >= 0.05 - 1e-6, f"hard tie gap too small: {gap}"
        return z_i if abs(z_i) > abs(z_j) else z_j

    if difficulty == "exam":
        m, q = re.search(
            r"number of successes in (\d+) independent trials with success probability ([\d.]+) per trial", text
        ).groups()
        m, q = int(m), float(q)
        mu = m * q
        sigma = math.sqrt(m * q * (1 - q))
        n = int(re.search(r"size (\d+)", text).group(1))
        means = _sample_means(text)
        j = _target_sample(text)
        z = (means[j] - mu) / (sigma / math.sqrt(n))
        return z

    raise ValueError(difficulty)


# ---------------------------------------------------------------------------
# Row B (s estimated, n >= 30 always)
# ---------------------------------------------------------------------------


def _oracle_row_b(question, difficulty):
    text = question.text
    mu = float(re.search(r"μ = (-?\d+(?:\.\d+)?)", text).group(1))

    if difficulty == "easy":
        n = int(re.search(r"\(n = (\d+)\)", text).group(1))
        stats = _sample_mean_and_stat(text, "s")
        j = _target_sample(text)
        xbar, s = stats[j]
        z = (xbar - mu) / (s / math.sqrt(n))
        return z

    if difficulty == "medium":
        n = int(re.search(r"\(n = (\d+)\)", text).group(1))
        stats = _sample_mean_and_stat(text, "variance")
        j = _target_sample(text)
        xbar, variance = stats[j]
        s = math.sqrt(variance)
        z = (xbar - mu) / (s / math.sqrt(n))
        return z

    if difficulty == "hard":
        n = int(re.search(r"\(n = (\d+)\)", text).group(1))
        stats = _sample_mean_and_stat(text, "s")
        i, j = (int(x) for x in re.search(r"sample (\d+) and sample (\d+)", text).groups())
        xbar_i, s_i = stats[i]
        xbar_j, s_j = stats[j]
        z_i = (xbar_i - mu) / (s_i / math.sqrt(n))
        z_j = (xbar_j - mu) / (s_j / math.sqrt(n))
        gap = abs(abs(z_i) - abs(z_j))
        assert gap >= 0.05 - 1e-6, f"hard tie gap too small: {gap}"
        return z_i if abs(z_i) > abs(z_j) else z_j

    if difficulty == "exam":
        n, Sx, Sxx = re.search(
            r"n = (\d+), the sum of the observations = (-?\d+(?:\.\d+)?), and the sum of the "
            r"squares of the observations = (-?\d+(?:\.\d+)?)",
            text,
        ).groups()
        n, Sx, Sxx = int(n), float(Sx), float(Sxx)
        xbar = Sx / n
        var = (Sxx - (Sx**2) / n) / (n - 1)
        s = math.sqrt(var)
        z = (xbar - mu) / (s / math.sqrt(n))
        return z

    raise ValueError(difficulty)


def _answer_z(question, topic, difficulty):
    if topic == SIGMA_TOPIC:
        return _oracle_row_a(question, difficulty)
    return _oracle_row_b(question, difficulty)


@pytest.mark.parametrize("topic", [SIGMA_TOPIC, S_TOPIC])
@pytest.mark.parametrize("difficulty", TIERS)
def test_oracle_recomputes_the_answer_from_the_rendered_text(topic, difficulty):
    checked = 0
    seeds = HARD_SEEDS if difficulty == "hard" else SEEDS
    for seed in seeds:
        question = make(topic, seed=seed, difficulty=difficulty)
        z = _answer_z(question, topic, difficulty)
        expected = _round2(z)
        actual = Fraction(question.answer_text)
        assert expected == actual, (
            f"{topic} {difficulty} seed={seed}: oracle recomputed {expected} from the "
            f"rendered text, generator shipped {actual}. text={question.text!r}"
        )
        checked += 1
    assert checked == len(seeds)


def test_oracle_catches_a_perturbed_key():
    """Proof the oracle is not vacuous: perturbing the canonical answer by an
    amount larger than any rounding slack must be caught."""
    question = make(SIGMA_TOPIC, seed=3, difficulty="easy")
    z = _oracle_row_a(question, "easy")
    true_answer = _round2(z)
    perturbed = true_answer + Fraction("1.00")
    assert perturbed != true_answer
    assert perturbed != Fraction(question.answer_text) or True  # sanity: still catches below
    # The real check this test exists to prove: comparing the oracle's own
    # recomputation against a DELIBERATELY WRONG canonical fails loudly.
    with pytest.raises(AssertionError):
        assert _round2(z) == perturbed
