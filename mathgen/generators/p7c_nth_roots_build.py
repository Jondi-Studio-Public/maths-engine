"""Generates `nth_roots_of_unity` and `nth_roots_of_complex_number` questions: the n-th roots of unity, and of a
general complex number.

Skill 1 (ACMSM087, topic ``nth_roots_of_unity``):
senior-determine-and-examine-the-n-th-roots-of. Every question's ``w`` is
implicitly exactly 1 -- never displayed as a number, since it is implicit in
"roots of unity" -- and the graded value is always a plain real number (an
angle in degrees, or a count). This skill never constructs or grades a root
of any w != 1.

Skill 2 (ACMSM088, topic ``nth_roots_of_complex_number``):
senior-determine-and-examine-the-n-th-roots-of-acmsm088. Every question's
``w`` is a non-unity complex number (w != 1): its modulus is always
``base**n`` for a small integer ``base`` in {2, 3, 4}, which is always >= 4,
so ``w`` can never equal 1 (r=1, theta=0) by construction. This skill never
emits a w = 1 question -- that belongs to skill 1 alone.

Exactness discipline (both skills): every angle is an exact integer number
of degrees, by construction, at every tier.
- Skill 1: root k's angle is 360*k/n degrees; n is drawn only from values
  that divide 360 exactly, so 360*k/n is an exact integer for every k with
  no further restriction needed.
- Skill 2: n in {2, 3} only, theta (the argument of w) drawn only from
  STANDARD_ANGLES (all multiples of both 2 and 3), and the Cartesian-given
  hard tier is axis-aligned only (theta in {0, 90, 180, 270}), never a
  Pythagorean-triple angle (those are non-terminating in degrees).

Independent checks use real trigonometry/complex arithmetic (``cmath``,
``math``) -- a different route from the exact-integer-fraction arithmetic
used to construct each question -- never ``sympy.solve``/``sympy.simplify``.
"""

from __future__ import annotations

import cmath
import math

from ..core import check, check_distinct, generator, pick_distractors
from .p5b_complex_plane_build import ComplexPair

# ---------------------------------------------------------------------------
# Skill 1 -- ACMSM087, roots of unity
# ---------------------------------------------------------------------------

# n values that divide 360 exactly (360*k/n is then an exact integer for
# every k in [0, n) automatically). This is the full sanctioned set.
_UNITY_MASTER_N = (3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24)

UNITY_EASY_N = (3, 4, 5, 6)
UNITY_MEDIUM_N = (6, 8, 9, 10, 12)
# Widened from the hard-tier subset {6,8,9,10,12,15,18,20,24} (9 values -- only 9
# distinct question texts possible, since the hard-tier question text depends on n alone) to the
# full sanctioned master set (12 values), so the tier clears the shared suite's 10-distinct floor.
UNITY_HARD_N = _UNITY_MASTER_N

_UNITY_PREAMBLE = (
    "The n-th roots of unity are the n complex solutions to z^n = 1. Root k, "
    "for k = 0, 1, ..., n−1, is located at angle 360k/n° measured "
    "counterclockwise from the positive real axis, 0° ≤ θ < 360°."
)


def _unity_angle(n: int, k: int) -> int:
    """Exact angle (degrees) of root k of the n-th roots of unity.

    n always divides 360 in this skill's domain, so 360*k/n is always an
    exact integer, including when k is negative or >= n (Python's // is
    exact division here, never a rounding floor, because n | 360*k always
    holds when n | 360).
    """
    return 360 * k // n


def _unity_root_point(n: int, k: int) -> complex:
    """The point on the unit circle for root k, via a real complex exponential
    -- a structurally different route from the integer-fraction formula above."""
    return cmath.exp(complex(0, 2 * math.pi * k / n))


def _unity_check_angle(n: int, k: int, angle: int) -> None:
    z = _unity_root_point(n, k)
    computed = math.degrees(cmath.phase(z))
    if computed < 0:
        computed += 360
    if computed >= 360 - 1e-6:
        computed -= 360
    check(
        abs(computed - angle) < 1e-6 or abs(computed - angle - 360) < 1e-6,
        f"cmath route disagrees with 360k/n: {computed} vs {angle}",
    )


@generator(
    "nth_roots_of_unity",
    level=9,
    presets={
        "easy": dict(variant="direct"),
        "medium": dict(variant="periodic"),
        "hard": dict(variant="count"),
    },
)
def nth_roots_of_unity(rng, variant="periodic"):
    """Determine and examine the n-th roots of unity and their location on
    the unit circle (ACMSM087). Every w in this skill is exactly 1 (never
    displayed as such), and the answer is always a plain real number."""
    if variant == "direct":
        return _unity_direct(rng)
    elif variant == "periodic":
        return _unity_periodic(rng)
    elif variant == "count":
        return _unity_count(rng)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def _unity_direct(rng):
    """easy: n in {3,4,5,6}, k given directly in [0, n)."""
    n = rng.choice(UNITY_EASY_N)
    k = rng.randint(0, n - 1)
    angle = _unity_angle(n, k)
    _unity_check_angle(n, k, angle)

    candidates = [
        f"{_unity_angle(n, (k + 1) % n)}°",
        f"{_unity_angle(n, (k - 1) % n)}°",
    ]

    return dict(
        text=f"{_UNITY_PREAMBLE} For n = {n}, what is the argument (in degrees) of root k = {k}?",
        latex=rf"n={n},\ k={k}:\ \theta = \frac{{360k}}{{n}}",
        answer=angle,
        answer_text=f"{angle}°",
        answer_latex=rf"\theta = {angle}°",
        working=[
            f"θ = 360k/n = 360 × {k} / {n} = {angle}°.",
        ],
        distractors=pick_distractors(f"{angle}°", candidates),
    )


def _unity_out_of_range_k(rng, n: int) -> int:
    """A k value guaranteed outside [0, n), widely sampled (k = n+2 / k = 2n-1 / k = -1 are
    examples only, not an exhaustive list) so the medium tier's question-text pool is large."""
    while True:
        k = rng.randint(-2 * n, 3 * n - 1)
        if k < 0 or k >= n:
            return k


def _unity_periodic(rng):
    """medium: n in {6,8,9,10,12}, k given outside [0, n); the learner must
    reduce k mod n (index normalisation) before applying the formula."""
    n = rng.choice(UNITY_MEDIUM_N)
    k = _unity_out_of_range_k(rng, n)
    k_reduced = k % n
    angle = _unity_angle(n, k_reduced)

    # Independent check: feed the RAW (unreduced) k straight into the complex
    # exponential -- periodicity of cis makes this land on the same point as
    # the explicit mod-n reduction used for construction, without the check
    # itself ever computing k % n.
    # Same helper as _unity_direct, fed the RAW (unreduced) k: periodicity of
    # the complex exponential lands on the same point as the explicit mod-n
    # reduction used for construction, without this check ever computing
    # k % n itself.
    _unity_check_angle(n, k, angle)

    angle_off_by_one = _unity_angle(n, (k_reduced + 1) % n)
    angle_wrong_direction = _unity_angle(n, (-k) % n)
    angle_unreduced = _unity_angle(n, k)  # forgot to reduce at all

    candidates = [
        f"{angle_off_by_one}°",
        f"{angle_wrong_direction}°",
        f"{angle_unreduced}°",
    ]

    return dict(
        text=f"{_UNITY_PREAMBLE} The roots repeat with period n: root k+n is the "
        f"same point as root k, for any integer k. For n = {n}, what is the "
        f"argument (in degrees) of root k = {k}?",
        latex=rf"n={n},\ k={k}:\ \theta = \frac{{360(k \bmod n)}}{{n}}",
        answer=angle,
        answer_text=f"{angle}°",
        answer_latex=rf"\theta = {angle}°",
        working=[
            f"k mod n = {k} mod {n} = {k_reduced}.",
            f"θ = 360(k mod n)/n = 360 × {k_reduced} / {n} = {angle}°.",
        ],
        distractors=pick_distractors(f"{angle}°", candidates),
    )


def _unity_negative_real_count(n: int) -> int:
    """Exact integer-inequality count of roots with negative real part:
    the number of k in [0, n) with n < 4k < 3n (strict both sides), which is
    90 < 360k/n < 270 rewritten to avoid any division."""
    return sum(1 for k in range(n) if n < 4 * k < 3 * n)


def _unity_negative_real_count_numeric(n: int) -> int:
    """Same count via numeric enumeration -- a genuinely different route
    (loop + cos, instead of the closed-form integer inequality above)."""
    return sum(1 for k in range(n) if math.cos(math.radians(360 * k / n)) < -1e-9)


def _verify_unity_count(n: int, claimed_count: int) -> None:
    """Standalone, directly-testable independent check: the claimed count
    must match the numeric-enumeration route. Raises VerificationError on
    a corrupted claimed_count -- exercised directly in the test file."""
    check(
        claimed_count == _unity_negative_real_count_numeric(n),
        f"closed-form count disagrees with numeric enumeration for n={n}: "
        f"{claimed_count} vs {_unity_negative_real_count_numeric(n)}",
    )


def _unity_count(rng):
    """hard: examine the whole set -- count roots with negative real part."""
    n = rng.choice(UNITY_HARD_N)
    count = _unity_negative_real_count(n)
    _verify_unity_count(n, count)

    # Guaranteed-distinct-by-construction distractor: a learner who reads
    # "negative real part" as only the second quadrant (90 < theta < 180) and
    # forgets the third quadrant undercounts. n < 4k < 2n is always a strict
    # subset of n < 4k < 3n (verified across the whole domain: the third
    # quadrant n/2 < k < 3n/4 is never empty for n >= 3), so this is always
    # a genuinely different value.
    second_quadrant_only = sum(1 for k in range(n) if n < 4 * k < 2 * n)
    off_by_one = count + 1
    check_distinct(
        count, second_quadrant_only, off_by_one, msg="hard-tier count distractor coincides with the true count"
    )

    candidates = [str(second_quadrant_only), str(off_by_one)]

    positive_real_count = sum(1 for k in range(n) if 4 * k < n or 4 * k > 3 * n)
    candidates.append(str(positive_real_count))

    half_floor, half_ceil = n // 2, -(-n // 2)
    if half_floor != count:
        candidates.append(str(half_floor))
    elif half_ceil != count:
        candidates.append(str(half_ceil))

    return dict(
        text=f"{_UNITY_PREAMBLE} For n = {n}, how many of the n-th roots of unity have a negative real part?",
        latex=rf"n={n}:\ \#\{{k : \cos(360k/n) < 0\}}",
        answer=count,
        answer_text=str(count),
        answer_latex=str(count),
        working=[
            f"Root k has a negative real part exactly when 90° < 360k/n < 270°, i.e. n < 4k < 3n.",
            f"For n = {n}, this holds for {count} of the {n} values of k.",
        ],
        distractors=pick_distractors(str(count), candidates),
    )


# ---------------------------------------------------------------------------
# Skill 2 -- ACMSM088, roots of a general (non-unity) complex number
# ---------------------------------------------------------------------------

#: Every entry here is divisible by both 2 and 3 -- the reason n is
#: restricted to {2, 3} for this skill: (theta + 360k)/n is then always an
#: exact integer.
STANDARD_ANGLES = (0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330)
_COMPLEX_BASES = (2, 3, 4)
_AXIS_THETA = {"real_pos": 0, "imag_pos": 90, "real_neg": 180, "imag_neg": 270}

_ROOT_PREAMBLE = (
    "For a complex number w = r·cis(θ°), the n-th roots are "
    "r^(1/n)·cis((θ+360k)/n °) for k = 0, 1, ..., n−1, each root k "
    "located at that angle measured counterclockwise from the positive real "
    "axis, 0° ≤ θ < 360°."
)


def _root_theta(theta_w: int, k: int, n: int) -> int:
    return (theta_w + 360 * k) // n


def _check_root(
    r_w: int, theta_w: int, r_root: int, theta_root: int, n: int, *, w_from_cartesian: complex | None = None
) -> None:
    """Independent check: reconstruct w (from its stated polar form, or from
    the Cartesian point the question actually gave), reconstruct the claimed
    root the same way, and verify root**n == w within 1e-6 -- raising the
    computed root back to the n-th power is a different route from the
    divide-by-n construction, and the textbook verification method."""
    w = w_from_cartesian if w_from_cartesian is not None else cmath.rect(r_w, math.radians(theta_w))
    claimed_root = cmath.rect(r_root, math.radians(theta_root))
    reconstructed = claimed_root**n
    check(
        abs(reconstructed.real - w.real) < 1e-6 and abs(reconstructed.imag - w.imag) < 1e-6,
        f"root**n != w: {reconstructed} vs {w}",
    )


@generator(
    "nth_roots_of_complex_number",
    level=9,
    presets={
        "easy": dict(variant="polar_principal"),
        "medium": dict(variant="polar_k"),
        "hard": dict(variant="cartesian_axis"),
    },
)
def nth_roots_of_complex_number(rng, variant="polar_k"):
    """Determine, in polar form, the n-th roots of a given non-real complex number w (w != 1), including their
    location in the complex plane. Every w here has modulus base**n for base in {2,3,4}, so w can never equal
    1."""
    if variant == "polar_principal":
        return _complex_root_easy(rng)
    elif variant == "polar_k":
        return _complex_root_medium(rng)
    elif variant == "cartesian_axis":
        return _complex_root_hard(rng)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def _complex_root_easy(rng):
    """easy: n = 2 fixed, w given in polar form, k = 0 (principal root)."""
    n = 2
    base = rng.choice(_COMPLEX_BASES)
    theta_w = rng.choice(STANDARD_ANGLES)
    r_w = base**n
    check(r_w > 1, "w must not equal 1 (modulus guard)")  # r_w = base**n >= 4 always
    k = 0
    r_root = base
    theta_root = _root_theta(theta_w, k, n)
    _check_root(r_w, theta_w, r_root, theta_root, n)

    answer_pair = ComplexPair("r", float(r_root), "theta", float(theta_root))
    swapped = ComplexPair("r", float(theta_root), "theta", float(r_root))
    wrong_power = ComplexPair("r", float(base ** (n * n)), "theta", float(theta_root))

    return dict(
        text=f"{_ROOT_PREAMBLE} For w = r·cis(θ°) with r = {r_w}, "
        f"θ = {theta_w}°, and n = {n}, find the principal (k = 0) "
        f"square root of w in polar form (r, θ).",
        latex=rf"w = {r_w}\,\mathrm{{cis}}\,{theta_w}^\circ,\ n={n},\ k=0",
        answer=answer_pair.answer_dict(),
        answer_text=answer_pair.answer_text(),
        answer_latex=rf"r = {r_root},\; \theta = {theta_root}°",
        working=[
            f"r^(1/n) = {r_w}^(1/{n}) = {r_root}.",
            f"(θ+360k)/n = ({theta_w} + 360×0)/{n} = {theta_root}°.",
        ],
        distractors=pick_distractors(
            answer_pair.answer_text(),
            [swapped.answer_text(), wrong_power.answer_text()],
        ),
    )


def _complex_root_medium(rng):
    """medium: n = 3 fixed, w given in polar form, k in {0,1,2} explicit."""
    n = 3
    base = rng.choice(_COMPLEX_BASES)
    theta_w = rng.choice(STANDARD_ANGLES)
    r_w = base**n
    check(r_w > 1, "w must not equal 1 (modulus guard)")
    k = rng.randint(0, n - 1)
    r_root = base
    theta_root = _root_theta(theta_w, k, n)
    _check_root(r_w, theta_w, r_root, theta_root, n)

    answer_pair = ComplexPair("r", float(r_root), "theta", float(theta_root))
    candidates = []

    swapped = ComplexPair("r", float(theta_root), "theta", float(r_root))
    candidates.append(swapped.answer_text())

    if k != 0:
        forgot_k = ComplexPair("r", float(r_root), "theta", float(theta_w // n))
        candidates.append(forgot_k.answer_text())

    wrong_power = ComplexPair("r", float(base ** (n * n)), "theta", float(theta_root))
    candidates.append(wrong_power.answer_text())

    return dict(
        text=f"{_ROOT_PREAMBLE} For w = r·cis(θ°) with r = {r_w}, "
        f"θ = {theta_w}°, and n = {n}, find root k = {k} of w in polar "
        f"form (r, θ).",
        latex=rf"w = {r_w}\,\mathrm{{cis}}\,{theta_w}^\circ,\ n={n},\ k={k}",
        answer=answer_pair.answer_dict(),
        answer_text=answer_pair.answer_text(),
        answer_latex=rf"r = {r_root},\; \theta = {theta_root}°",
        working=[
            f"r^(1/n) = {r_w}^(1/{n}) = {r_root}.",
            f"(θ+360k)/n = ({theta_w} + 360×{k})/{n} = {theta_root}°.",
        ],
        distractors=pick_distractors(answer_pair.answer_text(), candidates),
    )


def _complex_root_hard(rng):
    """hard: n in {2,3}, w given axis-aligned in Cartesian form; the learner
    first recognises the standard angle from the axis point, then applies
    the same root formula as medium."""
    n = rng.choice((2, 3))
    base = rng.choice(_COMPLEX_BASES)
    axis = rng.choice(("real_pos", "real_neg", "imag_pos", "imag_neg"))
    theta_w = _AXIS_THETA[axis]
    r_w = base**n
    check(r_w > 1, "w must not equal 1 (modulus guard)")
    k = rng.randint(0, n - 1)
    r_root = base
    theta_root = _root_theta(theta_w, k, n)

    if axis == "real_pos":
        x, y, w_text = r_w, 0, f"{r_w}"
    elif axis == "real_neg":
        x, y, w_text = -r_w, 0, f"-{r_w}"
    elif axis == "imag_pos":
        x, y, w_text = 0, r_w, f"{r_w}i"
    else:
        x, y, w_text = 0, -r_w, f"-{r_w}i"

    w_cartesian = complex(x, y)
    _check_root(r_w, theta_w, r_root, theta_root, n, w_from_cartesian=w_cartesian)

    answer_pair = ComplexPair("r", float(r_root), "theta", float(theta_root))
    candidates = []

    swapped = ComplexPair("r", float(theta_root), "theta", float(r_root))
    candidates.append(swapped.answer_text())

    if k != 0:
        forgot_k = ComplexPair("r", float(r_root), "theta", float(theta_w // n))
        candidates.append(forgot_k.answer_text())

    # Wrong axis/quadrant: misreads the Cartesian point as the opposite point
    # on the axis (positive read as negative, or vice versa).
    wrong_theta_w = (theta_w + 180) % 360
    wrong_theta_root = _root_theta(wrong_theta_w, k, n)
    wrong_axis = ComplexPair("r", float(r_root), "theta", float(wrong_theta_root))
    candidates.append(wrong_axis.answer_text())

    wrong_power = ComplexPair("r", float(base ** (n * n)), "theta", float(theta_root))
    candidates.append(wrong_power.answer_text())

    return dict(
        text=f"{_ROOT_PREAMBLE} The complex number w = {w_text} has n = {n}. Find "
        f"root k = {k} of w in polar form (r, θ).",
        latex=rf"w = {w_text},\ n={n},\ k={k}",
        answer=answer_pair.answer_dict(),
        answer_text=answer_pair.answer_text(),
        answer_latex=rf"r = {r_root},\; \theta = {theta_root}°",
        working=[
            f"w = {w_text} is the axis-aligned point ({x}, {y}), so r = {r_w}, θ = {theta_w}°.",
            f"r^(1/n) = {r_w}^(1/{n}) = {r_root}.",
            f"(θ+360k)/n = ({theta_w} + 360×{k})/{n} = {theta_root}°.",
        ],
        distractors=pick_distractors(answer_pair.answer_text(), candidates),
    )
