"""Independent oracle for the `p14f_energy_wave` questions: it re-derives each answer from the rendered question
text.

Imports nothing from `mathgen.generators.p14f_energy_wave_build`: no
constants, no templates, no misconception formulas. Parses the numbers
straight out of the RENDERED `question.text` and re-derives the correct
answer from scratch via plain `re`/`fractions` arithmetic, a route wholly
independent of the generator's own construction.

Run across >= 1000 seeds total per skill, spread over the four tiers.
"""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make

SEEDS_PER_TIER = 260  # 4 tiers x 260 = 1040 >= 1000

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _nums(text: str) -> list[str]:
    return _NUM_RE.findall(text)


# ---------------------------------------------------------------------------
# ucv1-energy-conservation-check
# ---------------------------------------------------------------------------

TOPIC_ENERGY = "p14f_energy_conservation_check"


def _energy(k, x, m, v) -> Fraction:
    return Fraction(k, 2) * Fraction(x) ** 2 + Fraction(m, 2) * Fraction(v) ** 2


_M_RE = re.compile(r"mass m = (-?\d+) kg")
_K_RE = re.compile(r"(?:stiffness|spring constant) k = (-?\d+) N/m")
_X1_RE = re.compile(r"x1 (?:is |= )(-?\d+) m\b")
_V1_RE = re.compile(r"v1 (?:is |= )(-?\d+) m/s")
_X2_RE = re.compile(r"x2 (?:is |= )(-?\d+) m\b")
_V2_RE = re.compile(r"v2 (?:is |= )(-?\d+) m/s")


def test_oracle_energy_easy_and_medium_direct_tiers():
    """easy/medium: every template states m, k, x1, v1, x2, v2 -- extracted
    via labels anchored to "x1"/"v1"/"x2"/"v2" so the digit inside each label
    (e.g. the "1" in "x1") is never mistaken for the drawn value that follows
    it (a real bug caught by writing this oracle: a naive "every number in
    the text, in order" parse pulls the "1" out of "State 1"/"x1"/"v1" and
    the "2" out of "(1/2)"/"x^2" long before the real values)."""
    checked = 0
    for difficulty in ("easy", "medium"):
        for seed in range(SEEDS_PER_TIER):
            q = make(TOPIC_ENERGY, seed=seed, difficulty=difficulty)
            text = q.text
            m = int(_M_RE.search(text).group(1))
            k = int(_K_RE.search(text).group(1))
            x1 = int(_X1_RE.search(text).group(1))
            v1 = int(_V1_RE.search(text).group(1))
            x2 = int(_X2_RE.search(text).group(1))
            v2 = int(_V2_RE.search(text).group(1))
            diff = _energy(k, x2, m, v2) - _energy(k, x1, m, v1)
            expected = f"{float(diff):.1f}"
            assert expected == q.answer_text, (difficulty, seed, q.text, q.answer_text, diff)
            checked += 1
    assert checked == 2 * SEEDS_PER_TIER


_TURNING_RE = re.compile(
    r"State (\d) is at a turning point \(momentarily at rest\), "
    r"with displacement x\d = (-?\d+) m\."
)
_EQUILIBRIUM_RE = re.compile(
    r"State (\d) is passing through equilibrium \(zero displacement\), "
    r"with velocity v\d = (-?\d+) m/s\."
)
_OTHER_RE = re.compile(r"displacement x = (-?\d+) m and velocity v = (-?\d+) m/s")
_MK_RE = re.compile(r"mass m = (-?\d+) kg[\s\S]*?(?:stiffness|spring constant) k = (-?\d+) N/m")


def test_oracle_energy_hard_tier():
    """hard: the implicit state is described in prose (turning point ->
    velocity zero; equilibrium -> displacement zero); the other state gives
    both x and v directly. Re-derive purely from which phrase is present and
    the numbers that follow it -- using anchored regexes so a digit inside a
    label like "x2" is never mistaken for a drawn value."""
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC_ENERGY, seed=seed, difficulty="hard")
        text = q.text
        m, k = (int(n) for n in _MK_RE.search(text).groups())

        turning_match = _TURNING_RE.search(text)
        if turning_match:
            state_no, imp_value = int(turning_match.group(1)), int(turning_match.group(2))
            e_imp = Fraction(k, 2) * imp_value**2
        else:
            equilibrium_match = _EQUILIBRIUM_RE.search(text)
            state_no, imp_value = int(equilibrium_match.group(1)), int(equilibrium_match.group(2))
            e_imp = Fraction(m, 2) * imp_value**2

        xo, vo = (int(n) for n in _OTHER_RE.search(text).groups())
        e_other = _energy(k, xo, m, vo)

        if state_no == 1:
            diff = e_other - e_imp
        else:
            diff = e_imp - e_other

        expected = f"{float(diff):.1f}"
        assert expected == q.answer_text, (seed, q.text, q.answer_text, diff)
        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_energy_exam_tier():
    """exam: state 1 (x1, v1) is given in full, x2 is given, and the energy
    difference (state 2 minus state 1) is stated in J. Re-solve for the
    speed at state 2 from those public numbers alone."""
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC_ENERGY, seed=seed, difficulty="exam")
        text = q.text
        m, k = (int(n) for n in _nums(text)[:2])
        x1 = int(re.search(r"x1 = (-?\d+) m", text).group(1))
        v1 = int(re.search(r"v1 = (-?\d+) m/s", text).group(1))
        x2 = int(re.search(r"x2 = (-?\d+) m", text).group(1))
        diff = Fraction(re.search(r"is (-?\d+(?:\.\d+)?) J", text).group(1))

        e1 = _energy(k, x1, m, v1)
        e2_target = e1 + diff
        v2_sq = Fraction(2, m) * (e2_target - Fraction(k, 2) * x2**2)
        assert v2_sq >= 0, (seed, q.text, v2_sq)
        v2 = v2_sq ** Fraction(1, 2) if v2_sq.denominator == 1 else None
        # v2_sq must be a perfect square of a rational for an exact speed;
        # confirm via integer sqrt on numerator/denominator.
        num_root = int(round(v2_sq.numerator**0.5))
        den_root = int(round(v2_sq.denominator**0.5))
        assert num_root * num_root == v2_sq.numerator, (seed, q.text, v2_sq)
        assert den_root * den_root == v2_sq.denominator, (seed, q.text, v2_sq)
        speed = Fraction(num_root, den_root)
        expected = f"{float(speed):.1f}"
        assert expected == q.answer_text, (seed, q.text, q.answer_text, speed)
        checked += 1
    assert checked == SEEDS_PER_TIER


# ---------------------------------------------------------------------------
# ucv1-wave-parameter-relation
# ---------------------------------------------------------------------------

TOPIC_WAVE = "p14f_wave_parameter_relation"


def test_oracle_wave_easy_tier():
    """easy: frequency f and wavelength lambda are given directly; v = f*lambda."""
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC_WAVE, seed=seed, difficulty="easy")
        nums = [int(n) for n in _nums(q.text)]
        f, lam = nums[0], nums[1]
        v = f * lam
        assert str(v) == q.answer_text, (seed, q.text, q.answer_text)
        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_wave_medium_tier():
    """medium: period T and wavelength lambda are given; v = lambda / T."""
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC_WAVE, seed=seed, difficulty="medium")
        nums = [int(n) for n in _nums(q.text)]
        T, lam = nums[0], nums[1]
        assert lam % T == 0, (seed, q.text, T, lam)
        v = lam // T
        assert str(v) == q.answer_text, (seed, q.text, q.answer_text)
        checked += 1
    assert checked == SEEDS_PER_TIER


def test_oracle_wave_hard_tier():
    """hard: speed v (km/h) and period T are given; convert to m/s (divide
    by 3.6) then lambda = v_mps * T."""
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC_WAVE, seed=seed, difficulty="hard")
        nums = [Fraction(n) for n in _nums(q.text)]
        v_kmh, T = nums[0], nums[1]
        v_mps = v_kmh / Fraction("3.6")
        assert v_mps.denominator == 1, (seed, q.text, v_mps)
        lam = int(v_mps) * int(T)
        # lam is always an exact integer, but displayed to 2 d.p. (matching
        # every distractor's shape) rather than as a bare integer.
        assert f"{lam:.2f}" == q.answer_text, (seed, q.text, q.answer_text, lam)
        checked += 1
    assert checked == SEEDS_PER_TIER


_MEDCHANGE_V1_RE = re.compile(r"v1\s*=\s*(\d+)\s*m/s")
_MEDCHANGE_LAM1_RE = re.compile(r"λ1\s*=\s*(\d+)\s*m\b")
_MEDCHANGE_V2_RE = re.compile(r"v2\s*=\s*(\d+)\s*m/s")


def test_oracle_wave_exam_tier():
    """exam: the wave crosses into a new medium with unchanged frequency (never printed as a number) but a stated
    new speed v2. Re-derive f = v1/lambda1 from the first medium, then lambda2 = v2/f for the second -- two
    chained relations, parsed from the rendered text alone, never importing the generator's own construction."""
    checked = 0
    for seed in range(SEEDS_PER_TIER):
        q = make(TOPIC_WAVE, seed=seed, difficulty="exam")
        v1 = int(_MEDCHANGE_V1_RE.search(q.text).group(1))
        lam1 = int(_MEDCHANGE_LAM1_RE.search(q.text).group(1))
        v2 = int(_MEDCHANGE_V2_RE.search(q.text).group(1))
        f = Fraction(v1, lam1)
        assert f.denominator == 1, (seed, q.text, f)
        lam2 = Fraction(v2) / f
        assert lam2.denominator == 1, (seed, q.text, lam2)
        assert f"{int(lam2)}.00" == q.answer_text, (seed, q.text, q.answer_text, lam2)
        checked += 1
    assert checked == SEEDS_PER_TIER
