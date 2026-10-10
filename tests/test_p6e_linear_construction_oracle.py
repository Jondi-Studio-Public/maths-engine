"""Independent oracle for the straight-line equation and word-description formula questions of p6e.

Linear construction oracle."""

import re
from collections import defaultdict
from fractions import Fraction

from sympy import Rational, symbols, parse_expr, expand, Symbol
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application

from mathgen.skills import load_skills
from mathgen import make


class TestStraightLineEquation:
    """Computational oracle for senior-find-the-equation-of-a-straight-line-given."""

    def test_straight_line_oracle(self):
        """Verify straight line equations computed from question text alone."""
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "senior-find-the-equation-of-a-straight-line-given"), None)
        assert skill is not None, "Skill not found"

        challenged = []
        examined = 0

        for source in skill.sources:
            for seed in range(40):
                q = make(topic=source.topic, seed=seed, difficulty=source.difficulty, **source.params)
                examined += 1

                # Extract the question text and answer
                text = q.text
                answer_text = q.answer_text

                # Try to extract question structure based on variant
                variant = source.params.get("variant")

                if variant == "two_points":
                    # Format: "A straight line passes through (x1, y1) and (x2, y2). Give the equation of the line, as y in terms of x."
                    points_match = re.findall(r"\((-?\d+),\s*(-?\d+)\)", text)
                    if len(points_match) != 2:
                        challenged.append(
                            {
                                "seed": seed,
                                "variant": variant,
                                "issue": f"Expected 2 points, found {len(points_match)}",
                                "text": text,
                                "answer_text": answer_text,
                            }
                        )
                        continue

                    try:
                        x1, y1 = int(points_match[0][0]), int(points_match[0][1])
                        x2, y2 = int(points_match[1][0]), int(points_match[1][1])

                        if x1 == x2:
                            challenged.append(
                                {
                                    "seed": seed,
                                    "variant": variant,
                                    "issue": "Vertical line (x1 == x2)",
                                    "text": text,
                                    "answer_text": answer_text,
                                }
                            )
                            continue

                        # Compute gradient and intercept
                        m = Rational(y2 - y1, x2 - x1)
                        c = Rational(y1) - m * Rational(x1)

                        # Build expected equation: y = mx + c
                        x = Symbol("x")
                        expected_rhs = m * x + c
                        expected_rhs_expanded = expand(expected_rhs)

                        # Parse answer_text to extract RHS
                        # Answer format: "y = <expr>"
                        ans_match = re.match(r"y\s*=\s*(.+)", answer_text)
                        if not ans_match:
                            challenged.append(
                                {
                                    "seed": seed,
                                    "variant": variant,
                                    "issue": f"Cannot parse answer_text format",
                                    "text": text,
                                    "answer_text": answer_text,
                                }
                            )
                            continue

                        answer_expr_str = ans_match.group(1)
                        try:
                            answer_rhs = parse_expr(
                                answer_expr_str,
                                local_dict={"x": Symbol("x")},
                                transformations=(standard_transformations + (implicit_multiplication_application,)),
                            )
                            answer_rhs_expanded = expand(answer_rhs)

                            # Check symbolic equivalence
                            diff = expand(expected_rhs_expanded - answer_rhs_expanded)
                            if diff != 0:
                                challenged.append(
                                    {
                                        "seed": seed,
                                        "variant": variant,
                                        "issue": f"Answer mismatch: computed y = {expected_rhs_expanded}, got y = {answer_rhs_expanded}",
                                        "text": text,
                                        "answer_text": answer_text,
                                        "computed": f"y = {expected_rhs_expanded}",
                                    }
                                )
                        except Exception as e:
                            challenged.append(
                                {
                                    "seed": seed,
                                    "variant": variant,
                                    "issue": f"Cannot parse answer expression: {e}",
                                    "text": text,
                                    "answer_text": answer_text,
                                }
                            )

                    except Exception as e:
                        challenged.append(
                            {
                                "seed": seed,
                                "variant": variant,
                                "issue": f"Error computing line: {e}",
                                "text": text,
                                "answer_text": answer_text,
                            }
                        )

                elif variant == "parallel":
                    # Format: "A line is parallel to y = <expr> and passes through the point (x0, y0). Give the equation of this line, as y in terms of x."
                    # Extract reference line
                    ref_match = re.search(r"parallel to\s+y\s*=\s*([^\s]+(?:\s*[+-]\s*[^\s]+)*)", text)
                    point_match = re.search(r"through the point\s+\((-?\d+),\s*(-?\d+)\)", text)

                    if not ref_match or not point_match:
                        challenged.append(
                            {
                                "seed": seed,
                                "variant": variant,
                                "issue": "Cannot parse reference line or point",
                                "text": text,
                                "answer_text": answer_text,
                            }
                        )
                        continue

                    try:
                        # Extract reference line and gradient
                        ref_line_str = ref_match.group(1)
                        x0, y0 = int(point_match.group(1)), int(point_match.group(2))

                        x = Symbol("x")
                        ref_expr = parse_expr(
                            ref_line_str,
                            local_dict={"x": x},
                            transformations=(standard_transformations + (implicit_multiplication_application,)),
                        )

                        # Extract coefficient of x (gradient)
                        m = ref_expr.as_coefficients_dict()[x]
                        c = ref_expr - m * x

                        # For the new line through (x0, y0): y0 = m*x0 + c_new
                        c_new = Rational(y0) - m * Rational(x0)

                        # Build expected equation
                        expected_rhs = m * x + c_new
                        expected_rhs_expanded = expand(expected_rhs)

                        # Parse answer
                        ans_match = re.match(r"y\s*=\s*(.+)", answer_text)
                        if not ans_match:
                            challenged.append(
                                {
                                    "seed": seed,
                                    "variant": variant,
                                    "issue": f"Cannot parse answer_text format",
                                    "text": text,
                                    "answer_text": answer_text,
                                }
                            )
                            continue

                        answer_expr_str = ans_match.group(1)
                        answer_rhs = parse_expr(
                            answer_expr_str,
                            local_dict={"x": x},
                            transformations=(standard_transformations + (implicit_multiplication_application,)),
                        )
                        answer_rhs_expanded = expand(answer_rhs)

                        diff = expand(expected_rhs_expanded - answer_rhs_expanded)
                        if diff != 0:
                            challenged.append(
                                {
                                    "seed": seed,
                                    "variant": variant,
                                    "issue": f"Answer mismatch: computed y = {expected_rhs_expanded}, got y = {answer_rhs_expanded}",
                                    "text": text,
                                    "answer_text": answer_text,
                                    "computed": f"y = {expected_rhs_expanded}",
                                }
                            )

                    except Exception as e:
                        challenged.append(
                            {
                                "seed": seed,
                                "variant": variant,
                                "issue": f"Error computing parallel line: {e}",
                                "text": text,
                                "answer_text": answer_text,
                            }
                        )

                elif variant == "perpendicular":
                    # Format: "A line is perpendicular to y = <expr> and passes through the point (x0, y0). Give the equation of this line, as y in terms of x."
                    ref_match = re.search(r"perpendicular to\s+y\s*=\s*([^\s]+(?:\s*[+-]\s*[^\s]+)*)", text)
                    point_match = re.search(r"through the point\s+\((-?\d+),\s*(-?\d+)\)", text)

                    if not ref_match or not point_match:
                        challenged.append(
                            {
                                "seed": seed,
                                "variant": variant,
                                "issue": "Cannot parse reference line or point",
                                "text": text,
                                "answer_text": answer_text,
                            }
                        )
                        continue

                    try:
                        ref_line_str = ref_match.group(1)
                        x0, y0 = int(point_match.group(1)), int(point_match.group(2))

                        x = Symbol("x")
                        ref_expr = parse_expr(
                            ref_line_str,
                            local_dict={"x": x},
                            transformations=(standard_transformations + (implicit_multiplication_application,)),
                        )

                        # Extract coefficient of x (gradient)
                        m_ref = ref_expr.as_coefficients_dict()[x]

                        # Perpendicular gradient: -1/m_ref
                        if m_ref == 0:
                            challenged.append(
                                {
                                    "seed": seed,
                                    "variant": variant,
                                    "issue": "Cannot compute perpendicular to horizontal line",
                                    "text": text,
                                    "answer_text": answer_text,
                                }
                            )
                            continue

                        m = -1 / m_ref
                        c_new = Rational(y0) - m * Rational(x0)

                        # Build expected equation
                        expected_rhs = m * x + c_new
                        expected_rhs_expanded = expand(expected_rhs)

                        # Parse answer
                        ans_match = re.match(r"y\s*=\s*(.+)", answer_text)
                        if not ans_match:
                            challenged.append(
                                {
                                    "seed": seed,
                                    "variant": variant,
                                    "issue": f"Cannot parse answer_text format",
                                    "text": text,
                                    "answer_text": answer_text,
                                }
                            )
                            continue

                        answer_expr_str = ans_match.group(1)
                        answer_rhs = parse_expr(
                            answer_expr_str,
                            local_dict={"x": x},
                            transformations=(standard_transformations + (implicit_multiplication_application,)),
                        )
                        answer_rhs_expanded = expand(answer_rhs)

                        diff = expand(expected_rhs_expanded - answer_rhs_expanded)
                        if diff != 0:
                            challenged.append(
                                {
                                    "seed": seed,
                                    "variant": variant,
                                    "issue": f"Answer mismatch: computed y = {expected_rhs_expanded}, got y = {answer_rhs_expanded}",
                                    "text": text,
                                    "answer_text": answer_text,
                                    "computed": f"y = {expected_rhs_expanded}",
                                }
                            )

                    except Exception as e:
                        challenged.append(
                            {
                                "seed": seed,
                                "variant": variant,
                                "issue": f"Error computing perpendicular line: {e}",
                                "text": text,
                                "answer_text": answer_text,
                            }
                        )

        assert not challenged, f"Found {len(challenged)} issues in {examined} questions: {challenged}"
        print(f"senior-find-the-equation-of-a-straight-line-given: examined {examined}, challenged 0")


class TestWordDescriptionFormula:
    """Computational oracle for senior-develop-a-linear-formula-from-a-word-description."""

    def test_word_description_oracle(self):
        """Verify linear formulas from word descriptions, derived from text alone."""
        _, skills = load_skills()
        skill = next((s for s in skills if s.id == "senior-develop-a-linear-formula-from-a-word-description"), None)
        assert skill is not None, "Skill not found"

        challenged = []
        examined = 0
        template_signs = defaultdict(set)  # template -> {+1 or -1}

        for source in skill.sources:
            for seed in range(40):
                q = make(topic=source.topic, seed=seed, difficulty=source.difficulty, **source.params)
                examined += 1

                text = q.text
                answer_text = q.answer_text
                variant = source.params.get("variant")

                try:
                    # Extract the variable letters and formula structure
                    # Look for patterns like: "Write a formula for the volume $V$ (in litres) after $t$ minutes"
                    # or "Write a formula for the total cost $C$ (in dollars) after $w$ weeks"
                    var_match = re.search(r"formula for the ([\w\s]+?)\s+\$(\w)\$.*after\s+\$(\w)\$\s+(\w+)", text)
                    if not var_match:
                        challenged.append(
                            {
                                "seed": seed,
                                "variant": variant,
                                "issue": "Cannot extract variable letters",
                                "text": text,
                                "answer_text": answer_text,
                            }
                        )
                        continue

                    description, dep_var, indep_var, time_unit = var_match.groups()
                    description = description.strip()

                    # Try to identify the template (tank, gym, car, etc.)
                    template = None
                    if "tank" in text.lower() and "fill" in text.lower():
                        template = "tank_fill"
                    elif "tank" in text.lower() and "drain" in text.lower():
                        template = "tank_drain"
                    elif "gym" in text.lower():
                        template = "gym_membership"
                    elif "car" in text.lower() and ("depreciate" in text.lower() or "loses value" in text.lower()):
                        template = "car_depreciate"
                    else:
                        template = f"unknown_{description}"

                    # Determine expected sign based on variant and wording
                    if variant == "increasing":
                        expected_sign = 1
                        direction_keywords = ["fill", "charges", "gain", "earn", "add"]
                    elif variant == "decreasing":
                        expected_sign = -1
                        direction_keywords = ["drain", "lose", "depreciate", "cool", "charge then add"]
                    else:  # back_solved_intercept
                        expected_sign = -1
                        direction_keywords = ["drain", "lose", "depreciate"]

                    # Extract initial value and rate
                    if variant in ["increasing", "decreasing"]:
                        initial = None
                        rate = None

                        # Pattern 1: "A tank starts with X ... and is filled at Y litres per"
                        if "is filled at" in text or "is charged at" in text or "charges" in text:
                            # Tank filling or gym membership
                            starts_match = re.search(r"starts with\s+(\d+)", text)
                            if starts_match:
                                initial = int(starts_match.group(1))
                            # For gym: "charges a $X joining fee, then $Y per week"
                            gym_match = re.search(r"charges a \$(\d+).*then \$(\d+) per", text)
                            if gym_match:
                                initial = int(gym_match.group(1))
                                rate = int(gym_match.group(2))
                            # For tank fill: "is filled at X litres per"
                            if rate is None:
                                fill_match = re.search(r"is filled at (\d+)", text)
                                if fill_match:
                                    rate = int(fill_match.group(1))

                        # Pattern 2: "A tank drains at X ... and starts with Y" or "loses value at $X ... Initially it is worth $Y"
                        else:
                            drain_match = re.search(r"(?:drains|loses value) at \$?(\d+)", text)
                            if drain_match:
                                rate = int(drain_match.group(1))
                            initial_match = re.search(r"(?:starts with|Initially, it is worth)\s+\$?(\d+)", text)
                            if initial_match:
                                initial = int(initial_match.group(1))

                        if initial is None or rate is None:
                            challenged.append(
                                {
                                    "seed": seed,
                                    "variant": variant,
                                    "issue": f"Cannot extract initial ({initial}) or rate ({rate})",
                                    "text": text,
                                    "answer_text": answer_text,
                                }
                            )
                            continue

                        # Build expected formula
                        t = Symbol(indep_var)
                        if expected_sign == 1:
                            expected_rhs = rate * t + initial
                        else:
                            expected_rhs = initial - rate * t

                        expected_rhs_expanded = expand(expected_rhs)

                    else:  # back_solved_intercept
                        # Format: "After N time, a tank/car that drains/depreciates at X per <unit> contains/is worth Y"
                        after_match = re.search(
                            r"After\s+(\d+)\s+(\w+),.*(?:drains?|depreciates?)\s+at\s+\$?(\d+).*(?:contains|is worth)\s+\$?(\d+)",
                            text,
                        )
                        if not after_match:
                            challenged.append(
                                {
                                    "seed": seed,
                                    "variant": variant,
                                    "issue": "Cannot parse back-solved intercept format",
                                    "text": text,
                                    "answer_text": answer_text,
                                }
                            )
                            continue

                        elapsed, time_unit_in_text, rate, value_at_time = (
                            int(after_match.group(1)),
                            after_match.group(2),
                            int(after_match.group(3)),
                            int(after_match.group(4)),
                        )

                        # Back-solve for initial value: value_at_time = initial - rate * elapsed
                        initial = value_at_time + rate * elapsed

                        t = Symbol(indep_var)
                        expected_rhs = initial - rate * t
                        expected_rhs_expanded = expand(expected_rhs)

                    # Parse answer_text to extract RHS
                    # Answer format: "<var> = <expr>"
                    ans_match = re.match(rf"{dep_var}\s*=\s*(.+)", answer_text)
                    if not ans_match:
                        challenged.append(
                            {
                                "seed": seed,
                                "variant": variant,
                                "issue": f"Cannot parse answer_text format",
                                "text": text,
                                "answer_text": answer_text,
                            }
                        )
                        continue

                    answer_expr_str = ans_match.group(1)
                    try:
                        answer_rhs = parse_expr(
                            answer_expr_str,
                            local_dict={indep_var: Symbol(indep_var)},
                            transformations=(standard_transformations + (implicit_multiplication_application,)),
                        )
                        answer_rhs_expanded = expand(answer_rhs)

                        # Check symbolic equivalence
                        diff = expand(expected_rhs_expanded - answer_rhs_expanded)
                        if diff != 0:
                            challenged.append(
                                {
                                    "seed": seed,
                                    "variant": variant,
                                    "issue": f"Answer mismatch: computed {dep_var} = {expected_rhs_expanded}, got {dep_var} = {answer_rhs_expanded}",
                                    "text": text,
                                    "answer_text": answer_text,
                                    "computed": f"{dep_var} = {expected_rhs_expanded}",
                                }
                            )
                        else:
                            # Extract the coefficient of the independent variable to check sign
                            coeffs = answer_rhs_expanded.as_coefficients_dict()
                            rate_coeff = coeffs.get(Symbol(indep_var), 0)
                            observed_sign = 1 if rate_coeff > 0 else (-1 if rate_coeff < 0 else 0)
                            template_signs[template].add(observed_sign)

                    except Exception as e:
                        challenged.append(
                            {
                                "seed": seed,
                                "variant": variant,
                                "issue": f"Cannot parse answer expression: {e}",
                                "text": text,
                                "answer_text": answer_text,
                            }
                        )

                except Exception as e:
                    challenged.append(
                        {
                            "seed": seed,
                            "variant": variant,
                            "issue": f"Error processing question: {e}",
                            "text": text,
                            "answer_text": answer_text,
                        }
                    )

        # Report template/sign consistency
        print("\nTemplate sign consistency check:")
        for template in sorted(template_signs.keys()):
            signs = template_signs[template]
            print(f"  {template}: signs = {signs}")
            if len(signs) > 1:
                challenged.append(
                    {
                        "template": template,
                        "issue": f"Template has inconsistent signs: {signs}",
                    }
                )

        assert not challenged, f"Found {len(challenged)} issues in {examined} questions: {challenged}"
        print(f"senior-develop-a-linear-formula-from-a-word-description: examined {examined}, challenged 0")
