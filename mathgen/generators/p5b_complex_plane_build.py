"""Complex number skills: Cartesian plane representation and polar conversion.

Skill 1: senior-consider-complex-numbers-as-points-in-a-plane (ACMSM071)
- Three difficulty tiers: direct (a+bi), degenerate (pure real/imaginary), sum (z1+z2)
- All produce ordered coordinate pairs (x, y) via answer: pair grader
- Independent check via Python's complex type, structurally different from construction

Skill 2: senior-convert-between-cartesian-and-polar-form (ACMSM081)
- Four difficulty tiers: axis (0/90/180/270 degrees), triple (Pythagorean, arbitrary angle),
  reverse (polar to Cartesian), exam (surd r in the correct quadrant, or an obtuse-theta reverse)
- All produce coordinate pairs (r, theta) or (x, y) via answer: pair grader
- easy/medium/hard: no irrational r values; theta always 0..360 degrees, never radians.
  exam is the one deliberate exception -- see _polar_exam_surd, whose r is a genuine surd.
- Independent check via cmath.polar() and cmath.rect(), structurally different from construction
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import sympy

from ..core import check, generator, phrase, pick_distractors


@dataclass
class ComplexPair:
    """Represents a complex number component pair for answer formatting."""

    label1: str
    value1: float
    label2: str
    value2: float
    #: Decimal places to render. ``None`` uses ":g", which strips a trailing
    #: zero -- right for exact integer coordinates, wrong for a variant whose
    #: prompt asks for a stated precision, where it rendered "y = 2" under
    #: "rounded to 1 decimal place" in 15 of 60 hard-tier questions.
    decimals: int | None = None

    def answer_dict(self) -> dict:
        """Return as a dict with labels as keys, in definition order."""
        return {self.label1: self.value1, self.label2: self.value2}

    def answer_text(self) -> str:
        """Return as human-readable text like 'x = 3, y = 5'.

        With ``decimals`` set, every component carries exactly that many
        decimal places, so the answer honours the precision the prompt states.
        """
        if self.decimals is None:
            return f"{self.label1} = {self.value1:g}, {self.label2} = {self.value2:g}"
        p = self.decimals
        return f"{self.label1} = {self.value1:.{p}f}, {self.label2} = {self.value2:.{p}f}"

    def matches_within_tolerance(self, other: ComplexPair, tolerance: float = 0.051) -> bool:
        """Check if two pairs match within tolerance (positional)."""
        return abs(self.value1 - other.value1) <= tolerance and abs(self.value2 - other.value2) <= tolerance


@generator(
    "complex_number_as_point",
    level=8,
    presets={
        "easy": dict(variant="direct"),
        "medium": dict(variant="degenerate"),
        "hard": dict(variant="sum"),
    },
)
def complex_number_as_point(rng, variant="sum"):
    """Represent complex numbers as points on a plane with Cartesian coordinates.

    Boundary: skills/senior-mathematics/complex-real-part.yaml and complex-imaginary-part.yaml
    extract a *single* number (real or imaginary component alone). This skill always
    produces an *ordered coordinate pair* (x, y) representing the full complex number's
    point on the plane, never a single component in isolation.

    Variants:
    - direct: given a+bi form (integers, a,b each from -9..9, b!=0)
    - degenerate: purely real or purely imaginary (b=0 or a=0, implicit zero shown)
    - sum: two complex numbers, compute z1+z2 (or z1-z2), return as coordinate pair
    """
    if variant == "direct":
        return _complex_direct(rng)
    elif variant == "degenerate":
        return _complex_degenerate(rng)
    elif variant == "sum":
        return _complex_sum(rng)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def _complex_direct(rng):
    """Directly given complex number in a+bi form."""
    a = rng.randint(-9, 9)
    b = rng.randint(-9, 9)
    # Ensure b != 0 (would fall into degenerate case)
    while b == 0:
        b = rng.randint(-9, 9)

    # Answer is the coordinate pair (a, b)
    answer_pair = ComplexPair("x", float(a), "y", float(b))

    # Independent check via complex type
    z = complex(a, b)
    check(z.real == a and z.imag == b, "complex type gives wrong components")

    # Format the complex number in question text
    if b > 0:
        complex_text = f"{a} + {b}i" if a != 0 else f"{b}i"
    else:
        complex_text = f"{a} - {-b}i" if a != 0 else f"{-b}i"

    # Misconception distractors
    candidates = []

    # swaps-real-and-imaginary: reports (b, a) instead of (a, b)
    if a != b:  # Only offer when they're different
        swapped = ComplexPair("x", float(b), "y", float(a))
        candidates.append(swapped.answer_text())

    # drops-the-sign: reports (a, abs(b)) instead of (a, b)
    if b < 0:  # Only offer when b is negative
        dropped_sign = ComplexPair("x", float(a), "y", float(abs(b)))
        candidates.append(dropped_sign.answer_text())

    return dict(
        text=f"The complex number {complex_text} is represented by a point in the plane. "
        f"State its Cartesian coordinates as an ordered pair (x, y).",
        latex=rf"\text{{Complex number: }}{complex_text}\text{{ as coordinates}}",
        answer=answer_pair.answer_dict(),
        answer_text=answer_pair.answer_text(),
        answer_latex=rf"({a}, {b})",
        working=[
            f"A complex number a + bi corresponds to the point (a, b) in the plane.",
            f"For {complex_text}, we have a = {a}, b = {b}.",
            f"Therefore, the point is ({a}, {b}).",
        ],
        distractors=pick_distractors(answer_pair.answer_text(), candidates),
    )


def _complex_degenerate(rng):
    """Purely real or purely imaginary complex number (one component implicit zero)."""
    # Randomly choose between purely real (b=0) and purely imaginary (a=0)
    use_real = rng.random() < 0.5

    if use_real:
        # Purely real: b = 0, a sampled from -9..9
        a = rng.randint(-9, 9)
        b = 0
        # Displayed as just the number, e.g., "7" or "-4", never "7 + 0i"
        complex_text = str(a)
    else:
        # Purely imaginary: a = 0, b sampled from -9..9, b != 0
        a = 0
        b = rng.randint(-9, 9)
        while b == 0:
            b = rng.randint(-9, 9)
        # Displayed as "bi" or "-bi", e.g., "6i" or "-2i"
        if b > 0:
            complex_text = f"{b}i"
        else:
            complex_text = f"{b}i"

    answer_pair = ComplexPair("x", float(a), "y", float(b))

    # Independent check via complex type
    z = complex(a, b)
    check(z.real == a and z.imag == b, "complex type gives wrong components")

    # Misconception distractors
    candidates = []

    # swaps-real-and-imaginary: reports (b, a) instead of (a, b)
    if a != b:
        swapped = ComplexPair("x", float(b), "y", float(a))
        candidates.append(swapped.answer_text())

    # treats-i-as-a-value (purely-imaginary only): reports (b, 0) instead of (0, b)
    if b != 0:  # Only when imaginary component is nonzero
        wrong_placement = ComplexPair("x", float(b), "y", float(0))
        candidates.append(wrong_placement.answer_text())

    # sign slip on either component, then (for the zero complex number, where
    # every reflection and swap collapses onto the answer) a unit-point slip
    for sx, sy in ((-1, -1), (1, -1), (-1, 1)):
        candidates.append(ComplexPair("x", float(sx * a), "y", float(sy * b)).answer_text())
    candidates.append(ComplexPair("x", 1.0, "y", 0.0).answer_text())
    candidates.append(ComplexPair("x", 0.0, "y", 1.0).answer_text())

    return dict(
        text=f"The complex number {complex_text} is represented by a point in the plane. "
        f"State its Cartesian coordinates as an ordered pair (x, y).",
        latex=rf"\text{{Complex number: }}{complex_text}\text{{ as coordinates}}",
        answer=answer_pair.answer_dict(),
        answer_text=answer_pair.answer_text(),
        answer_latex=rf"({a}, {b})",
        working=[
            f"A complex number a + bi corresponds to the point (a, b) in the plane.",
            f"The complex number {complex_text} is purely {'real' if b == 0 else 'imaginary'}.",
            f"We interpret this as {complex_text} = {a} {'+' if b >= 0 else '-'} {abs(b)}i.",
            f"Therefore, the point is ({a}, {b}).",
        ],
        distractors=pick_distractors(answer_pair.answer_text(), candidates),
    )


def _complex_sum(rng):
    """Two complex numbers z1, z2; compute z1+z2 or z1-z2 and express as coordinate pair."""
    # Sample z1 = a1 + b1*i
    a1 = rng.randint(-9, 9)
    b1 = rng.randint(-9, 9)
    while b1 == 0 and a1 == 0:  # Ensure z1 is nonzero
        a1 = rng.randint(-9, 9)
        b1 = rng.randint(-9, 9)

    # Sample z2 = a2 + b2*i, ensuring at least one of b1, b2 is nonzero
    a2 = rng.randint(-9, 9)
    b2 = rng.randint(-9, 9)
    while b2 == 0 and (a1 != 0 or b1 != 0):
        # Just keep b2 resampled until it's nonzero, or until both z1 and z2 have at least one nonzero
        if b1 == 0:
            # If b1=0 (z1 is real), we need b2 != 0 to have at least one imaginary
            b2 = rng.randint(-9, 9)
            while b2 == 0:
                b2 = rng.randint(-9, 9)
            break
        else:
            # If b1 != 0, we can allow b2 to be anything
            break

    # Randomly choose addition or subtraction
    use_add = rng.random() < 0.5

    if use_add:
        result_a = a1 + a2
        result_b = b1 + b2
        operation_text = "+"
        operation_symbol = "+"
    else:
        result_a = a1 - a2
        result_b = b1 - b2
        operation_text = "−"
        operation_symbol = "-"

    answer_pair = ComplexPair("x", float(result_a), "y", float(result_b))

    # Independent check via complex type and native addition/subtraction
    z1 = complex(a1, b1)
    z2 = complex(a2, b2)
    if use_add:
        total = z1 + z2
    else:
        total = z1 - z2
    check(total.real == result_a and total.imag == result_b, "complex arithmetic gives wrong result")

    # Format the complex numbers
    def format_complex(a, b):
        if b > 0:
            return f"{a} + {b}i" if a != 0 else f"{b}i"
        else:
            return f"{a} - {-b}i" if a != 0 else f"{-b}i"

    z1_text = format_complex(a1, b1)
    z2_text = format_complex(a2, b2)

    # Misconception distractors
    candidates = []

    # swaps-real-and-imaginary: reports (result_b, result_a)
    if result_a != result_b:
        swapped = ComplexPair("x", float(result_b), "y", float(result_a))
        candidates.append(swapped.answer_text())

    # adds-instead-of-subtracts (or vice versa): offer the other operation
    if not use_add:
        # If we did subtraction, offer addition as wrong distractor
        alt_a = a1 + a2
        alt_b = b1 + b2
    else:
        # If we did addition, offer subtraction as wrong distractor
        alt_a = a1 - a2
        alt_b = b1 - b2

    # Only offer if z2 is not zero (otherwise the operations are the same)
    if (a2, b2) != (0, 0):
        wrong_op = ComplexPair("x", float(alt_a), "y", float(alt_b))
        candidates.append(wrong_op.answer_text())

    # Format a2 and b2 for display, wrapping negatives in parentheses to avoid "+-" substring
    a2_display = f"({a2})" if a2 < 0 else str(a2)
    b2_display = f"({b2})" if b2 < 0 else str(b2)

    # Format result_b for display, wrapping negatives in parentheses to avoid "+-" substring
    result_b_display = f"({result_b})" if result_b < 0 else str(result_b)

    return dict(
        text=f"Given the complex numbers z₁ = {z1_text} and z₂ = {z2_text}, "
        f"state the Cartesian coordinates of z₁ {operation_text} z₂ as an ordered pair (x, y).",
        latex=rf"z_1 = {z1_text}, z_2 = {z2_text}; \text{{ find }} z_1 {operation_symbol} z_2",
        answer=answer_pair.answer_dict(),
        answer_text=answer_pair.answer_text(),
        answer_latex=rf"({result_a}, {result_b})",
        working=[
            f"z₁ {operation_text} z₂ = ({a1} {operation_symbol} {a2_display}) + ({b1} {operation_symbol} {b2_display})i",
            f"z₁ {operation_text} z₂ = {result_a} + {result_b_display}i",
            f"As a coordinate pair: ({result_a}, {result_b}).",
        ],
        distractors=pick_distractors(answer_pair.answer_text(), candidates),
    )


# Pythagorean triples for the polar conversion skill
_PYTHAGOREAN_TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25)]


@generator(
    "cartesian_polar_convert",
    level=8,
    presets={
        "easy": dict(variant="axis"),
        "medium": dict(variant="triple"),
        "hard": dict(variant="reverse"),
        "exam": dict(variant="exam"),
    },
)
def cartesian_polar_convert(rng, variant="triple"):
    """Convert between Cartesian and polar form for complex numbers.

    All angles are in degrees, 0 <= theta < 360, never radians.
    easy/medium/hard: no irrational r values, r is always an exact integer.

    Variants:
    - axis: Cartesian point is axis-aligned (a,0), (-a,0), (0,b), (0,-b); r and theta exact
    - triple: Cartesian point uses Pythagorean triple; theta rounded to nearest degree
    - reverse: Given polar (r, theta), compute Cartesian (x, y) rounded to 1 decimal place
    - exam: genuine escalation over "reverse" in kind, not just size -- randomly one of
      two shapes: a Cartesian->polar conversion where r is a genuine surd (the point is
      NOT drawn from a Pythagorean triple), r reported to 2 d.p. and theta to 1 d.p. in
      the correct quadrant; or a polar->Cartesian conversion with an obtuse theta
      (90 < theta < 180), x and y each to 1 d.p. See _polar_exam_surd/
      _polar_exam_reverse_obtuse.
    """
    if variant == "axis":
        return _polar_axis(rng)
    elif variant == "triple":
        return _polar_triple(rng)
    elif variant == "reverse":
        return _polar_reverse(rng)
    elif variant == "exam":
        return _polar_exam(rng)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def _polar_axis(rng):
    """Axis-aligned points: (a, 0), (-a, 0), (0, b), (0, -b)."""
    value = rng.randint(1, 12)
    quadrant = rng.randint(0, 3)

    if quadrant == 0:  # (a, 0) -- positive x-axis
        x, y = value, 0
        theta = 0
    elif quadrant == 1:  # (0, b) -- positive y-axis
        x, y = 0, value
        theta = 90
    elif quadrant == 2:  # (-a, 0) -- negative x-axis
        x, y = -value, 0
        theta = 180
    else:  # (0, -b) -- negative y-axis
        x, y = 0, -value
        theta = 270

    r = abs(value)

    # Independent check via cmath
    z = complex(x, y)
    r_check, theta_rad_check = cmath.polar(z)
    theta_check = math.degrees(theta_rad_check)
    # Normalise to 0..360
    if theta_check < 0:
        theta_check += 360

    check(abs(r_check - r) < 1e-6, f"cmath.polar r mismatch: {r_check} vs {r}")
    check(abs(theta_check - theta) < 1e-6, f"cmath.polar theta mismatch: {theta_check} vs {theta}")

    answer_pair = ComplexPair("r", float(r), "theta", float(theta))

    # Misconception distractors
    candidates = []

    # swaps-r-and-theta: reports (theta, r)
    if r != theta:
        swapped = ComplexPair("r", float(theta), "theta", float(r))
        candidates.append(swapped.answer_text())

    # Format x and y for display, wrapping negatives in parentheses to avoid "+-" substring
    x_display = f"({x})" if x < 0 else str(x)
    y_display = f"({y})" if y < 0 else str(y)

    return dict(
        text=f"Convert the Cartesian coordinates ({x}, {y}) to polar form. "
        f"Give r as the exact distance from the origin, and theta in degrees, 0 to 360.",
        latex=rf"({x}, {y}) \to (r, \theta) \ "
        r"\text{Give } r \text{ as the exact distance from the origin, and }"
        r"\theta \text{ in degrees, 0 to 360.}",
        answer=answer_pair.answer_dict(),
        answer_text=answer_pair.answer_text(),
        answer_latex=rf"r = {r}, \theta = {theta}°",
        working=[
            f"r = √(x² + y²) = √({x_display}² + {y_display}²) = √{x * x + y * y} = {r}",
            f"θ = arctan(y/x) = {theta}° (axis-aligned point)",
        ],
        distractors=pick_distractors(answer_pair.answer_text(), candidates),
    )


def _polar_triple(rng):
    """Pythagorean triple leg pairs, placed in any quadrant."""
    triple = rng.choice(_PYTHAGOREAN_TRIPLES)
    leg_a, leg_b, r = triple

    # Randomise which legs go to x/y
    if rng.random() < 0.5:
        leg_x, leg_y = leg_a, leg_b
    else:
        leg_x, leg_y = leg_b, leg_a

    # Randomise signs (quadrant)
    if rng.random() < 0.5:
        leg_x = -leg_x
    if rng.random() < 0.5:
        leg_y = -leg_y

    # Compute theta using atan2
    theta_rad = math.atan2(leg_y, leg_x)
    theta_exact = math.degrees(theta_rad)
    # Normalise to 0..360
    if theta_exact < 0:
        theta_exact += 360

    theta_rounded = round(theta_exact)

    # Independent check via cmath
    z = complex(leg_x, leg_y)
    r_check, theta_rad_check = cmath.polar(z)
    theta_check = math.degrees(theta_rad_check)
    if theta_check < 0:
        theta_check += 360

    check(abs(r_check - r) < 1e-6, f"cmath r mismatch: {r_check} vs {r}")
    check(abs(theta_check - theta_exact) < 1e-6, f"unrounded theta mismatch: {theta_check} vs {theta_exact}")
    # Check rounding is correct
    theta_correctly_rounded = round(theta_exact)
    check(theta_rounded == theta_correctly_rounded, "theta rounding incorrect")

    answer_pair = ComplexPair("r", float(r), "theta", float(theta_rounded))

    # Misconception distractors
    candidates = []

    # swaps-r-and-theta: reports (theta, r)
    if r != theta_rounded:
        swapped = ComplexPair("r", float(theta_rounded), "theta", float(r))
        candidates.append(swapped.answer_text())

    # wrong-quadrant: uses atan2(abs(y), abs(x)) ignoring signs (first quadrant angle)
    if leg_x != 0 and leg_y != 0 and not (leg_x > 0 and leg_y > 0):
        # Only offer if not already in first quadrant
        first_quadrant_rad = math.atan2(abs(leg_y), abs(leg_x))
        first_quadrant_deg = math.degrees(first_quadrant_rad)
        first_quadrant_rounded = round(first_quadrant_deg)
        if abs(first_quadrant_rounded - theta_rounded) > 0.051:
            wrong_quad = ComplexPair("r", float(r), "theta", float(first_quadrant_rounded))
            candidates.append(wrong_quad.answer_text())

    # degrees-radians-confusion: reports theta as raw radian value
    if abs(theta_exact) > 0.1 and abs(theta_exact - 2 * math.pi) > 0.1:
        # Avoid near-zero case where radian and degree values might be close
        radians_confused = ComplexPair("r", float(r), "theta", theta_exact)
        candidates.append(radians_confused.answer_text())

    # Format leg_x and leg_y for display, wrapping negatives in parentheses to avoid "+-" substring
    leg_x_display = f"({leg_x})" if leg_x < 0 else str(leg_x)
    leg_y_display = f"({leg_y})" if leg_y < 0 else str(leg_y)

    return dict(
        text=f"Convert the Cartesian coordinates ({leg_x}, {leg_y}) to polar form. "
        f"Give r as the exact distance from the origin, and theta in degrees (nearest degree), 0 to 360.",
        latex=rf"({leg_x}, {leg_y}) \to (r, \theta) \ "
        r"\text{Give } r \text{ as the exact distance from the origin, and }"
        r"\theta \text{ in degrees (nearest degree), 0 to 360.}",
        answer=answer_pair.answer_dict(),
        answer_text=answer_pair.answer_text(),
        answer_latex=rf"r = {r}, \theta = {theta_rounded}°",
        working=[
            f"r = √(x² + y²) = √({leg_x_display}² + {leg_y_display}²) = √{leg_x * leg_x + leg_y * leg_y} = {r}",
            f"θ = arctan(y/x) = arctan({leg_y}/{leg_x}) ≈ {theta_exact:.1f}° ≈ {theta_rounded}° (nearest degree)",
        ],
        distractors=pick_distractors(answer_pair.answer_text(), candidates),
    )


def _polar_reverse(rng):
    """Given polar form (r, theta), compute Cartesian (x, y) rounded to 1 decimal place."""
    r = rng.randint(2, 15)
    # Exclude axis points and 45-degree band
    theta = rng.randint(0, 359)

    # Exclude axis points (0, 90, 180, 270) and near 45, 135, 225, 315 degrees (where sin ≈ cos)
    excluded_bands = [(45 - 3, 45 + 3), (135 - 3, 135 + 3), (225 - 3, 225 + 3), (315 - 3, 315 + 3)]
    axis_points = [0, 90, 180, 270]

    while theta in axis_points or any(band[0] <= theta <= band[1] for band in excluded_bands):
        theta = rng.randint(0, 359)

    # Compute Cartesian
    theta_rad = math.radians(theta)
    x_exact = r * math.cos(theta_rad)
    y_exact = r * math.sin(theta_rad)

    x_rounded = round(x_exact, 1)
    y_rounded = round(y_exact, 1)

    # Independent check via cmath.rect
    z = cmath.rect(r, theta_rad)
    x_check = z.real
    y_check = z.imag

    check(abs(x_check - x_exact) < 1e-6, f"cmath x mismatch: {x_check} vs {x_exact}")
    check(abs(y_check - y_exact) < 1e-6, f"cmath y mismatch: {y_check} vs {y_exact}")

    # Verify rounding
    x_correctly_rounded = round(x_exact, 1)
    y_correctly_rounded = round(y_exact, 1)
    check(x_rounded == x_correctly_rounded and y_rounded == y_correctly_rounded, "rounding incorrect")

    answer_pair = ComplexPair("x", x_rounded, "y", y_rounded, decimals=1)

    # Misconception distractors
    candidates = []

    # swaps-sin-and-cos: computes (r*sin(theta), r*cos(theta))
    if abs(math.sin(theta_rad) - math.cos(theta_rad)) > 0.01:
        # Only offer if sin and cos are sufficiently different
        x_swapped = round(r * math.sin(theta_rad), 1)
        y_swapped = round(r * math.cos(theta_rad), 1)
        if abs(x_swapped - x_rounded) > 0.051 or abs(y_swapped - y_rounded) > 0.051:
            swapped = ComplexPair("x", x_swapped, "y", y_swapped, decimals=1)
            candidates.append(swapped.answer_text())

    return dict(
        text=f"Convert the polar coordinates r = {r}, θ = {theta}° to Cartesian form (x, y). "
        f"Give x and y rounded to 1 decimal place.",
        latex=rf"(r, \theta) = ({r}, {theta}°) \to (x, y) \ \text{{(rounded to 1 decimal place)}}",
        answer=answer_pair.answer_dict(),
        answer_text=answer_pair.answer_text(),
        answer_latex=rf"x = {x_rounded}, y = {y_rounded}",
        working=[
            f"x = r cos(θ) = {r} cos({theta}°) ≈ {x_exact:.3f} ≈ {x_rounded}",
            f"y = r sin(θ) = {r} sin({theta}°) ≈ {y_exact:.3f} ≈ {y_rounded}",
        ],
        distractors=pick_distractors(answer_pair.answer_text(), candidates),
    )


def _polar_exam(rng):
    """Exam-tier dispatch: a genuine escalation in KIND over `reverse`, not merely bigger numbers.

    Randomly one of two shapes, so this single `exam` source reaches both:
    - `_polar_exam_surd`: Cartesian -> polar where r is a genuine surd (the
      point is NOT drawn from a Pythagorean triple), r to 2 d.p. and theta to
      1 d.p., quadrant-aware.
    - `_polar_exam_reverse_obtuse`: polar -> Cartesian with an obtuse theta
      (90 < theta < 180), x and y to 1 d.p.
    """
    if rng.random() < 0.5:
        return _polar_exam_surd(rng)
    return _polar_exam_reverse_obtuse(rng)


#: Text patterns for the surd-r Cartesian->polar exam shape. Two genuinely
#: different wordings (not synonym swaps) so `prompt_shape()` -- which masks
#: digits, not prose -- sees them as distinct shapes; see `_polar_exam`.
_EXAM_SURD_TEXT_PATTERNS = [
    "Convert the Cartesian coordinates ({x}, {y}) to polar form. Give r rounded "
    "to 2 decimal places, and theta in degrees rounded to 1 decimal place, "
    "0 to 360.",
    "Express the point ({x}, {y}) in polar form. Round r to 2 decimal places "
    "and theta to 1 decimal place, with theta measured in degrees between "
    "0 and 360.",
]


def _polar_exam_surd(rng):
    """Cartesian -> polar where r is a genuine surd.

    (x, y) is drawn so x^2 + y^2 is NOT a perfect square -- this is exactly
    the condition that excludes every Pythagorean triple and every scaled
    multiple of one (a triple's legs always sum-of-squares to a perfect
    square by definition), so it needs no separate triple-membership check.
    r is reported to 2 d.p., theta to 1 d.p., quadrant-aware -- reuses the
    SAME atan2 + 0..360 normalisation `_polar_triple` uses, not a second
    hand-rolled one.
    """
    nonzero = [v for v in range(-12, 13) if v != 0]
    while True:
        x = rng.choice(nonzero)
        y = rng.choice(nonzero)
        s = x * x + y * y
        if math.isqrt(s) ** 2 != s:
            break

    r_exact_sym = sympy.sqrt(s)
    r_float = float(r_exact_sym)
    r_rounded = round(r_float, 2)

    # theta via atan2, normalised to 0..360 -- same pattern as _polar_triple
    theta_rad = math.atan2(y, x)
    theta_exact = math.degrees(theta_rad)
    if theta_exact < 0:
        theta_exact += 360
    theta_rounded = round(theta_exact, 1)

    # Independent check via cmath, structurally different from construction
    z = complex(x, y)
    r_check, theta_rad_check = cmath.polar(z)
    theta_check = math.degrees(theta_rad_check)
    if theta_check < 0:
        theta_check += 360

    check(abs(r_check - r_float) < 1e-9, f"cmath r mismatch: {r_check} vs {r_float}")
    check(abs(theta_check - theta_exact) < 1e-6, f"cmath theta mismatch: {theta_check} vs {theta_exact}")
    check(round(r_check, 2) == r_rounded, "r rounding incorrect")
    check(round(theta_check, 1) == theta_rounded, "theta rounding incorrect")

    answer = {"r": r_rounded, "theta": theta_rounded}
    answer_text = f"r = {r_rounded:.2f}, theta = {theta_rounded:.1f}"

    # Misconception distractors -- every candidate below is only offered when
    # it differs from the true answer by MORE than the grader's own 0.051
    # tolerance (mathgen/grading/typed.py::_PAIR_TOLERANCE), not the finer 0.005 the
    # stated 2 d.p. would imply but the grader does not actually enforce.
    candidates = []

    # swaps-r-and-theta: reports (theta, r) instead of (r, theta)
    if abs(r_rounded - theta_rounded) > 0.051:
        candidates.append(f"r = {theta_rounded:.2f}, theta = {r_rounded:.1f}")

    # wrong-quadrant: uses atan2(abs(y), abs(x)) ignoring signs (first-quadrant angle)
    if x != 0 and y != 0 and not (x > 0 and y > 0):
        naive_deg = round(math.degrees(math.atan2(abs(y), abs(x))), 1)
        if abs(naive_deg - theta_rounded) > 0.051:
            candidates.append(f"r = {r_rounded:.2f}, theta = {naive_deg:.1f}")

    # degrees-radians-confusion: reports theta as the raw radian value
    if abs(theta_exact) > 0.1 and abs(theta_exact - 360) > 0.1:
        radian_confused = round(theta_rad, 1)
        if abs(radian_confused - theta_rounded) > 0.051:
            candidates.append(f"r = {r_rounded:.2f}, theta = {radian_confused:.1f}")

    x_display = f"({x})" if x < 0 else str(x)
    y_display = f"({y})" if y < 0 else str(y)
    surd_display = str(r_exact_sym).replace("sqrt", "√").replace("*", "")

    text = phrase(rng, _EXAM_SURD_TEXT_PATTERNS, x=x, y=y)

    return dict(
        text=text,
        latex=rf"({x}, {y}) \to (r, \theta) \ "
        r"\text{Give } r \text{ rounded to 2 decimal places, and }"
        r"\theta \text{ rounded to 1 decimal place, in degrees, 0 to 360.}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=rf"r = {r_rounded:.2f}, \theta = {theta_rounded:.1f}°",
        working=[
            f"r = √(x² + y²) = √({x_display}² + {y_display}²) = √{s} = {surd_display} ≈ {r_rounded:.2f}",
            f"θ = atan2(y, x) = atan2({y_display}, {x_display}) ≈ {theta_exact:.4f}° "
            f"≈ {theta_rounded:.1f}° (quadrant-aware)",
        ],
        distractors=pick_distractors(answer_text, candidates),
    )


#: Text patterns for the obtuse-theta polar->Cartesian exam shape. Two
#: genuinely different wordings; see `_polar_exam`.
_EXAM_REVERSE_TEXT_PATTERNS = [
    "Convert the polar coordinates r = {r}, θ = {theta}° (θ is obtuse) to "
    "Cartesian form (x, y). Give x and y rounded to 1 decimal place.",
    "A point has polar coordinates r = {r}, θ = {theta}°, an obtuse angle. "
    "Find its Cartesian coordinates (x, y), each rounded to 1 decimal place.",
]


def _polar_exam_reverse_obtuse(rng):
    """Polar -> Cartesian with an obtuse theta (90 < theta < 180): cos(theta)
    is negative and sin(theta) positive, so x < 0 and y > 0 -- the trap this
    tier targets is forgetting cos is negative past 90 degrees. x and y are
    each rounded to 1 d.p., the same convention `_polar_reverse` uses.

    theta is drawn from 95..175 (not the full 91..179) and the 132..138 band
    is excluded (sin ~ cos there, same discipline `_polar_reverse` uses for
    its own excluded bands): with r >= 2, this keeps both x and y away from
    the rounds-to-zero boundary at either end of the obtuse range.
    """
    r = rng.randint(2, 20)
    theta = rng.randint(95, 175)
    while 132 <= theta <= 138:
        theta = rng.randint(95, 175)

    theta_rad = math.radians(theta)
    x_exact = r * math.cos(theta_rad)
    y_exact = r * math.sin(theta_rad)
    x_rounded = round(x_exact, 1)
    y_rounded = round(y_exact, 1)

    # Independent check via cmath.rect, structurally different from construction
    z = cmath.rect(r, theta_rad)
    x_check = z.real
    y_check = z.imag
    check(abs(x_check - x_exact) < 1e-9, f"cmath x mismatch: {x_check} vs {x_exact}")
    check(abs(y_check - y_exact) < 1e-9, f"cmath y mismatch: {y_check} vs {y_exact}")
    check(round(x_check, 1) == x_rounded and round(y_check, 1) == y_rounded, "rounding incorrect")
    check(x_rounded < 0 and y_rounded > 0, "obtuse theta must place the point in the second quadrant")

    answer = {"x": x_rounded, "y": y_rounded}
    answer_text = f"x = {x_rounded:.1f}, y = {y_rounded:.1f}"

    candidates = []

    # swaps-sin-and-cos: computes (r*sin(theta), r*cos(theta))
    if abs(math.sin(theta_rad) - math.cos(theta_rad)) > 0.01:
        x_swapped = round(r * math.sin(theta_rad), 1)
        y_swapped = round(r * math.cos(theta_rad), 1)
        if abs(x_swapped - x_rounded) > 0.051 or abs(y_swapped - y_rounded) > 0.051:
            candidates.append(f"x = {x_swapped:.1f}, y = {y_swapped:.1f}")

    # ignores-obtuse-sign: reports x as positive (forgets cos is negative past 90 deg)
    x_wrong_sign = round(abs(x_exact), 1)
    if abs(x_wrong_sign - x_rounded) > 0.051:
        candidates.append(f"x = {x_wrong_sign:.1f}, y = {y_rounded:.1f}")

    text = phrase(rng, _EXAM_REVERSE_TEXT_PATTERNS, r=r, theta=theta)

    return dict(
        text=text,
        latex=rf"(r, \theta) = ({r}, {theta}°,\ \theta\ \text{{obtuse}}) \to (x, y) \ "
        r"\text{(rounded to 1 decimal place)}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=rf"x = {x_rounded:.1f}, y = {y_rounded:.1f}",
        working=[
            f"x = r cos(θ) = {r} cos({theta}°) ≈ {x_exact:.3f} ≈ {x_rounded:.1f} (cos is negative for an obtuse angle)",
            f"y = r sin(θ) = {r} sin({theta}°) ≈ {y_exact:.3f} ≈ {y_rounded:.1f}",
        ],
        distractors=pick_distractors(answer_text, candidates),
    )
