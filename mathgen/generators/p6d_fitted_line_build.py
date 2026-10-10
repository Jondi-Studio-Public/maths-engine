"""Generates `use_the_equation_of_a_fitted_line_to` questions: make a prediction from the equation of a fitted
line.

Senior fitted-line prediction.

- ``use_the_equation_of_a_fitted_line_to``
  (``senior-use-the-equation-of-a-fitted-line-to``,
  ACMGM061, domain statistics-probability, answer: number)
  -- use the equation of a fitted line to make predictions.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import check, check_distinct, check_equivalent, generator, pick_distractors, GRADER_TOLERANCE


# Realistic context templates: (output_quantity_long, input_quantity_long, output_var, input_var)
# Each tuple gives the full phrase to use in the "predicts ... from ..." template, plus the
# short variable names for display in equations and question sentences.
CONTEXTS = [
    ("the weight (in kg) of a puppy", "its age (in weeks)", "weight", "age"),
    ("the cost (in $) of a taxi ride", "the distance travelled (in km)", "cost", "distance"),
    ("the estimated house price (in $100k)", "the age of the house (in years)", "price", "age"),
    ("the height (in cm) of a plant", "the number of days since planting", "height", "days"),
]


@generator(
    "use_the_equation_of_a_fitted_line_to",
    level=12,
    presets={
        "easy": dict(variant="easy"),
        "medium": dict(variant="medium"),
        "hard": dict(variant="hard"),
    },
)
def use_the_equation_of_a_fitted_line_to(rng, variant="easy"):
    """Use the equation of a fitted line to make predictions."""

    context = rng.choice(CONTEXTS)
    output_long, input_long, output_var, input_var = context

    if variant == "easy":
        # Easy: integer m (2-9), integer c (1-20), integer x (1-15)
        # Answer: y = m*x + c (always exact integer)
        # Distractor constraints:
        # 1. forgets-intercept (m*x): always distinct since c >= 1
        # 2. swaps-slope-intercept (c*x + m): distinct iff m != c and x != 1
        # 3. adds-instead (m + x + c): distinct iff m*x != m + x, i.e., m*(x-1) != x
        #    For integer x, this fails when x = m/(m-1), which happens only for m=2, x=2
        # 4. off-by-one (m*(x+1) + c): always distinct (differs by m)
        for attempt in range(500):
            m = rng.randint(2, 10)
            c = rng.randint(1, 21)
            x = rng.randint(1, 16)
            # Exclude: m==c, x==1, (m==2 and x==2)
            if m != c and x != 1 and not (m == 2 and x == 2):
                break
        else:
            raise RuntimeError("Could not sample easy tier avoiding m==c, x==1, and (m==2 and x==2)")

        y = m * x + c
        answer_text = str(y)

        # Format the fitted line equation: use literal 'x' for multiplication, no decimal point
        equation_text = f"{output_var} = {m} x {input_var} + {c}"

        question_text = (
            f"A fitted line predicts {output_long} from {input_long}: {equation_text}. "
            f"Using this fitted line, what {output_var} does it predict when {input_var} is {x}?"
        )

        working = [
            f"Substitute {input_var} = {x} into the equation.",
            f"{output_var} = {m} × {x} + {c} = {answer_text}.",
        ]

        # Distractors
        distractor_forgets_intercept = str(m * x)
        distractor_swaps = str(c * x + m)
        distractor_adds = str(m + x + c)
        distractor_off_by_one = str(m * (x + 1) + c)

        # Check all distractors are distinct
        check_distinct(
            y,
            int(distractor_forgets_intercept),
            int(distractor_swaps),
            int(distractor_adds),
            int(distractor_off_by_one),
            msg="easy: a distractor equals the answer",
        )

        candidates = [distractor_forgets_intercept, distractor_swaps, distractor_adds, distractor_off_by_one]
        distractors = pick_distractors(answer_text, candidates, n=3, tolerance=GRADER_TOLERANCE)

        # Independent check: solve backward using inverse operation (Fraction arithmetic for exactness)
        y_frac = Fraction(y)
        m_frac = Fraction(m)
        c_frac = Fraction(c)
        x_frac = Fraction(x)
        check_equivalent((y_frac - c_frac) / m_frac, x_frac, "easy: inverse operation should recover x")

    elif variant == "medium":
        # Medium: m and c as exact 2dp values (Fractions), integer x
        # Answer: y = m*x + c rounded to 1dp with rounding-boundary exclusion
        # Distractor constraints:
        # 1. forgets-intercept (m*x): distinct iff c != 0 (always true since c >= 1.00)
        # 2. swaps-slope-intercept (c*x + m): distinct iff (c-m)*(x-1) != 0, i.e., m != c and x != 1
        # 3. adds-instead (m + x + c): gap is |x(m-1) - m|, always large for m >= 0.10
        # Draw ranges: m ∈ [10/100, 99/100], c ∈ [100/100, 999/100]
        # This ensures |m - c| >= 0.01 structurally, and we resample if m == c or x == 1
        for attempt in range(500):
            m_num = rng.randint(10, 100)  # 0.10 to 0.99
            c_num = rng.randint(100, 1000)  # 1.00 to 9.99
            m = Fraction(m_num, 100)
            c = Fraction(c_num, 100)
            x = rng.randint(1, 16)

            # Exclude m == c (structurally unlikely but possible if ranges adjusted)
            # and x == 1 (makes swaps-slope-intercept distractor collide)
            if m == c or x == 1:
                continue

            # Compute y exactly as Fraction
            y_exact = m * x + c
            y_rounded = round(float(y_exact), 1)

            # Rounding-boundary exclusion: ensure distance to nearest 0.05 boundary is > 0.05
            # The hundredths digit must not be exactly 5 (or within tolerance of 5)
            y_fractional_part = y_exact - int(y_exact)
            hundredths_digit = (y_fractional_part * 100) % 10
            # Check if hundredths digit is far from 5
            margin = 0.05
            if abs(float(hundredths_digit) - 5) > margin:
                break
        else:
            raise RuntimeError("Could not sample medium tier with unambiguous rounding after 500 attempts")

        answer_text = f"{y_rounded:.1f}"

        # Format equation: convert Fraction to decimal string, strip trailing zeros
        m_float = float(m)
        c_float = float(c)
        m_str = f"{m_float:.2f}".rstrip("0").rstrip(".")
        c_str = f"{c_float:.2f}".rstrip("0").rstrip(".")
        equation_text = f"{output_var} = {m_str} x {input_var} + {c_str}"

        question_text = (
            f"A fitted line predicts {output_long} from {input_long}: {equation_text}. "
            f"Using this fitted line, what {output_var} does it predict when {input_var} is {x}? "
            f"Round to 1 decimal place."
        )

        working = [
            f"Substitute {input_var} = {x} into the equation.",
            f"{output_var} = {m_str} × {x} + {c_str} = {answer_text}.",
        ]

        # Distractors for medium
        # 1. forgets-intercept: m*x (no +c)
        d1_exact = m * x
        d1_rounded = round(float(d1_exact), 1)
        d1_str = f"{d1_rounded:.1f}"

        # 2. swaps-slope-intercept: c*x + m
        d2_exact = c * x + m
        d2_rounded = round(float(d2_exact), 1)
        d2_str = f"{d2_rounded:.1f}"

        # 3. adds-instead-of-multiplies: m + x + c
        d3_exact = m + x + c
        d3_rounded = round(float(d3_exact), 1)
        d3_str = f"{d3_rounded:.1f}"

        # Check all distractors are distinct at Fraction level
        check_distinct(
            y_exact, d1_exact, d2_exact, d3_exact, msg="medium: a distractor equals the answer at Fraction level"
        )

        # Also verify each distractor's gap from answer exceeds 0.05 after rounding to 1dp
        for distractor_val, distractor_name in [
            (d1_exact, "forgets-intercept"),
            (d2_exact, "swaps-slope-intercept"),
            (d3_exact, "adds-instead"),
        ]:
            gap = abs(distractor_val - y_exact)
            check(gap > Fraction(5, 100), f"medium {distractor_name}: gap={gap} not > 0.05")

        candidates = [d1_str, d2_str, d3_str]
        distractors = pick_distractors(answer_text, candidates, n=3, tolerance=GRADER_TOLERANCE)

        # Independent check: inverse operation at Fraction level
        x_frac = Fraction(x)
        check_equivalent((y_exact - c) / m, x_frac, "medium: inverse should recover x")

    elif variant == "hard":
        # Hard: same m/c as medium, but given target y (rounded to 1dp) and solve for x
        # x0 is picked first (integer, 2-15, avoiding 1), then y0_exact = m*x0 + c computed
        # Constraint: y0_exact's hundredths digit must be 0, so y0_display == y0_exact exactly
        # Distractor constraints:
        # 1. forgets-to-subtract: y0/m; distinct iff c != 0 (always true)
        # 2. swaps-operation-order: (y0-m)/c; distinct iff x0 != 1 and m != c
        # 3. sign-error: (y0+c)/m; distinct iff 2c != 0 (always true)
        for attempt in range(500):
            m_num = rng.randint(10, 100)
            c_num = rng.randint(100, 1000)
            m = Fraction(m_num, 100)
            c = Fraction(c_num, 100)
            x0 = rng.randint(2, 16)  # Start from 2, not 1, to avoid swaps-operation collision

            # Exclude m == c
            if m == c:
                continue

            # Compute y0_exact
            y0_exact = m * x0 + c

            # Hard constraint: hundredths digit must be 0 so display == exact
            hundredths = (y0_exact * 100) % 10
            if hundredths == 0:
                y0_rounded = round(float(y0_exact), 1)
                # Verify display == exact
                if Fraction(str(y0_rounded)) == y0_exact:
                    break
        else:
            raise RuntimeError("Could not sample hard tier with y0_exact having tenths-only precision")

        y0_display_str = f"{y0_rounded:.1f}"
        answer_text = str(x0)

        # Format equation: convert Fraction to decimal string, strip trailing zeros
        m_float = float(m)
        c_float = float(c)
        m_str = f"{m_float:.2f}".rstrip("0").rstrip(".")
        c_str = f"{c_float:.2f}".rstrip("0").rstrip(".")
        equation_text = f"{output_var} = {m_str} x {input_var} + {c_str}"

        question_text = (
            f"A fitted line predicts {output_long} from {input_long}: {equation_text}. "
            f"What value of {input_var} does this fitted line predict will give a {output_var} of {y0_display_str}?"
        )

        working = [
            f"Solve for {input_var}: {input_var} = ({output_var} - {c_str}) / {m_str}.",
            f"{input_var} = ({y0_display_str} - {c_str}) / {m_str} = {answer_text}.",
        ]

        # Distractors for hard: solving equation errors
        # 1. forgets-to-subtract-intercept: y0 / m
        d1_exact = y0_exact / m

        # 2. swaps-operation-order: (y0 - m) / c
        # This should always be distinct from x0 since x0 >= 2 and m != c (by construction)
        d2_exact = (y0_exact - m) / c

        # 3. sign-error: (y0 + c) / m
        d3_exact = (y0_exact + c) / m

        # Check distractors are distinct at Fraction level
        x0_frac = Fraction(x0)
        check_distinct(x0_frac, d1_exact, d2_exact, d3_exact, msg="hard: a distractor equals the answer")

        # Format distractors as integers (since hard answers are integers)
        d1_str = str(int(round(float(d1_exact))))
        d2_str = str(int(round(float(d2_exact))))
        d3_str = str(int(round(float(d3_exact))))

        candidates = [d1_str, d2_str, d3_str]
        distractors = pick_distractors(answer_text, candidates, n=3, tolerance=GRADER_TOLERANCE)

        # Independent check: forward substitution should recover y0_exact
        check_equivalent(m * x0_frac + c, y0_exact, "hard: forward substitution m*x0+c should recover y0_exact")

    return {
        "text": question_text,
        "latex": f"plain:{question_text}",
        "answer": answer_text,
        "answer_text": answer_text,
        "answer_latex": answer_text,
        "distractors": distractors,
        "working": working,
    }
