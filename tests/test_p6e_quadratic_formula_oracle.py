"""Independent oracle for the quadratic-formula questions of p6e.

Quadratic formula oracle."""

import re
from sympy import Rational, sqrt, simplify, Symbol, symbols, expand, Poly, isprime
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application, parse_expr

from mathgen.skills import load_skills
from mathgen import make


def is_perfect_square(n):
    """Check if n is a perfect square."""
    if n <= 0:
        return False
    sqrt_n = int(n**0.5)
    return sqrt_n * sqrt_n == n


def parse_quadratic(text):
    r"""
    Parse quadratic equation from question text.

    Regex pattern explanation:
    - (\d*)x\^2 : leading coefficient (optional, default 1)
    - ([+-]) : sign before x coefficient
    - (\d*)x : x coefficient (optional, default 1)
    - ([+-]) : sign before constant
    - (\d+) : constant term

    Examples:
    - "x^2 - 7x + 11" -> a=1, b=-7, c=11
    - "2x^2 + 2x - 1" -> a=2, b=2, c=-1
    - "x^2 + x - 1" -> a=1, b=1, c=-1
    """
    pattern = r"(\d*)x\^2\s*([+-])\s*(\d*)x\s*([+-])\s*(\d+)"
    match = re.search(pattern, text)

    if not match:
        return None

    a_str, sign_b, b_str, sign_c, c_str = match.groups()

    # Parse leading coefficient
    a = Rational(int(a_str)) if a_str else Rational(1)

    # Parse x coefficient
    b_mag = Rational(int(b_str)) if b_str else Rational(1)
    b = b_mag if sign_b == "+" else -b_mag

    # Parse constant
    c_mag = Rational(int(c_str))
    c = c_mag if sign_c == "+" else -c_mag

    return a, b, c


def compute_larger_root(a, b, c):
    """
    Compute the larger root of ax^2 + bx + c = 0 using the quadratic formula.
    Returns symbolic expression.
    """
    discriminant = b**2 - 4 * a * c

    # Verify discriminant is positive and not a perfect square
    disc_val = float(discriminant)
    if disc_val <= 0:
        return None, discriminant, "discriminant not positive"

    # Check if it's a perfect square
    if is_perfect_square(int(discriminant)):
        return None, discriminant, "discriminant is perfect square"

    # Quadratic formula: x = (-b ± sqrt(b^2 - 4ac)) / (2a)
    sqrt_disc = sqrt(discriminant)

    # Larger root: use + when a > 0, use - when a < 0
    if a > 0:
        root = (-b + sqrt_disc) / (2 * a)
    else:
        root = (-b - sqrt_disc) / (2 * a)

    # Simplify
    root = simplify(root)

    return root, discriminant, "ok"


class TestQuadraticFormulaOracle:
    """Independent oracle for senior-solve-quadratic-equations-using-the-quadratic-formula-and."""

    def test_quadratic_formula_oracle(self):
        """Verify quadratic formula roots computed from question text alone."""
        _, skills = load_skills()
        skill = next(
            (s for s in skills if s.id == "senior-solve-quadratic-equations-using-the-quadratic-formula-and"), None
        )
        assert skill is not None, "Skill not found"

        challenged = []
        examined = 0
        perfect_square_issues = []

        for source in skill.sources:
            variant = source.params.get("variant", "unknown")

            for seed in range(60):
                q = make(topic=source.topic, seed=seed, difficulty=source.difficulty, **source.params)
                examined += 1

                text = q.text
                answer_text = q.answer_text

                # Parse coefficients from question text
                parsed = parse_quadratic(text)
                if parsed is None:
                    challenged.append(
                        {
                            "seed": seed,
                            "variant": variant,
                            "issue": "Cannot parse quadratic equation",
                            "text": text,
                            "answer_text": answer_text,
                        }
                    )
                    continue

                a, b, c = parsed

                # Compute the larger root
                root, discriminant, status = compute_larger_root(a, b, c)

                if status != "ok":
                    if "perfect square" in status:
                        perfect_square_issues.append(
                            {
                                "seed": seed,
                                "variant": variant,
                                "text": text,
                                "a": a,
                                "b": b,
                                "c": c,
                                "discriminant": discriminant,
                            }
                        )
                    else:
                        challenged.append(
                            {
                                "seed": seed,
                                "variant": variant,
                                "issue": status,
                                "text": text,
                                "a": a,
                                "b": b,
                                "c": c,
                                "answer_text": answer_text,
                            }
                        )
                    continue

                # Parse the answer_text to extract the root expression
                # Answer format: "x = <expr>"
                ans_match = re.match(r"x\s*=\s*(.+)", answer_text)
                if not ans_match:
                    challenged.append(
                        {
                            "seed": seed,
                            "variant": variant,
                            "issue": "Cannot parse answer format",
                            "text": text,
                            "answer_text": answer_text,
                        }
                    )
                    continue

                answer_expr_str = ans_match.group(1)
                try:
                    x = Symbol("x")
                    answer_expr = parse_expr(
                        answer_expr_str,
                        local_dict={"x": x},
                        transformations=(standard_transformations + (implicit_multiplication_application,)),
                    )
                    answer_simplified = simplify(answer_expr)

                    # Check symbolic equivalence
                    diff = simplify(root - answer_simplified)
                    if diff != 0:
                        challenged.append(
                            {
                                "seed": seed,
                                "variant": variant,
                                "issue": f"Answer mismatch",
                                "text": text,
                                "parsed_a": str(a),
                                "parsed_b": str(b),
                                "parsed_c": str(c),
                                "computed_root": str(root),
                                "answer_text": answer_text,
                                "answer_parsed": str(answer_simplified),
                                "difference": str(diff),
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

        # Report perfect square discriminants if any (these are errors - they should be factorable)
        if perfect_square_issues:
            print(f"\nWARNING: Found {len(perfect_square_issues)} perfect-square discriminants:")
            for issue in perfect_square_issues:
                print(f"  Seed {issue['seed']} ({issue['variant']}): {issue['text']}")
                print(f"    a={issue['a']}, b={issue['b']}, c={issue['c']}, disc={issue['discriminant']}")

        assert not challenged, f"Found {len(challenged)} issues in {examined} questions"
        print(f"senior-solve-quadratic-equations-using-the-quadratic-formula-and: examined {examined}, challenged 0")
