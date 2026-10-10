"""Generates `apply_the_product_quotient_and_chain_rule_to` questions: differentiate a function using the
product, quotient and chain rules.

Product, quotient and chain rule.

- ``apply_the_product_quotient_and_chain_rule_to``
  (``senior-apply-the-product-quotient-and-chain-rule-to``,
  ACMMM106, domain differentiation, answer: expression) -- given bounded, fully
  specified inputs, apply the product, quotient and chain rule to differentiate
  functions. Return the resulting value, expression, relation, or classification.
"""

from __future__ import annotations

from sympy import symbols, exp, sin, cos, diff, simplify

from ..core import check, generator, latex_answer, pick_distractors, symbolically_equal


# Display helpers (^/* convention): convert ** to ^ for typed, ** to ^ and * to · for prose
def _typed(expr) -> str:
    """Convert expression to learner-facing typed form: ** -> ^."""
    return str(expr).replace("**", "^")


def _prose(expr) -> str:
    """Convert expression to prose form: ** -> ^, * -> ·."""
    typed_str = str(expr).replace("**", "^")
    return typed_str.replace("*", "·")


# Module-scope symbol
x = symbols("x")


@generator(
    "apply_the_product_quotient_and_chain_rule_to",
    level=12,
    presets={
        "easy": dict(difficulty="easy"),
        "medium": dict(difficulty="medium"),
        "hard": dict(difficulty="hard"),
    },
)
def apply_the_product_quotient_and_chain_rule_to(rng, difficulty="easy"):
    """
    Apply product, quotient, and chain rules to differentiate transcendental functions.

    easy: product rule with x * exp(k*x)
    medium: quotient rule with sin(k*x)/cos(k*x) or 1/x^n
    hard: chain rule with sin(ax+b), exp(ax+b), or exp(-k*x)*sin(m*x)
    """

    if difficulty == "easy":
        # Product rule: u * v where v is transcendental, u is affine polynomial
        shape = rng.choice(["exp", "sin", "cos"])
        k = rng.randint(2, 20)
        d = rng.randint(-12, 12)
        u = x + d
        if shape == "exp":
            v, v_prime = exp(k * x), k * exp(k * x)
        elif shape == "sin":
            v, v_prime = sin(k * x), k * cos(k * x)
        else:
            v, v_prime = cos(k * x), -k * sin(k * x)
        u_prime = 1
        ans = v + u * v_prime

        # Independent check: sympy's own differentiation
        check(symbolically_equal(diff(u * v, x) - ans, 0))

        # Verify distractors are distinct from ans
        mult_derivs = u_prime * v_prime
        only_uprime_v = v
        only_u_vprime = u * v_prime

        check(not symbolically_equal(ans - mult_derivs, 0))
        check(not symbolically_equal(ans - only_uprime_v, 0))
        check(not symbolically_equal(ans - only_u_vprime, 0))

        candidates = [
            _typed(mult_derivs),
            _typed(only_uprime_v),
            _typed(only_u_vprime),
        ]
        distractors = pick_distractors(_typed(ans), candidates)

        return {
            "text": f"Differentiate {_prose(u * v)} with respect to x.",
            "latex": f"Differentiate {_prose(u * v)} with respect to x.",
            "answer": str(ans),
            "answer_text": _typed(ans),
            "answer_latex": latex_answer(ans),
            "distractors": distractors,
        }

    elif difficulty == "medium":
        kind = rng.choice(["quotient_trig", "quotient_power"])

        if kind == "quotient_trig":
            # quotient rule: sin(a*x+b) / cos(a*x+b)
            a = rng.randint(1, 7)
            b = rng.randint(-9, 9)
            u, v = sin(a * x + b), cos(a * x + b)
            u_prime, v_prime = a * cos(a * x + b), -a * sin(a * x + b)
            ans = simplify((u_prime * v - u * v_prime) / v**2)

            # Independent check
            check(symbolically_equal(ans - diff(u / v, x), 0))

            # Verify distractors are distinct
            swapped = (u * v_prime - u_prime * v) / v**2
            no_square = (u_prime * v - u * v_prime) / v
            prod_instead = u_prime * v_prime

            check(not symbolically_equal(ans - swapped, 0))
            check(not symbolically_equal(ans - no_square, 0))
            check(not symbolically_equal(ans - prod_instead, 0))

            candidates = [
                _typed(swapped),
                _typed(no_square),
                _typed(prod_instead),
            ]
            distractors = pick_distractors(_typed(ans), candidates)

            return {
                "text": f"Differentiate {_prose(u)}/{_prose(v)} with respect to x.",
                "latex": f"Differentiate {_prose(u)}/{_prose(v)} with respect to x.",
                "answer": str(ans),
                "answer_text": _typed(ans),
                "answer_latex": latex_answer(ans),
                "distractors": distractors,
            }

        else:  # quotient_power
            # quotient rule: 1 / (a*x+b)^n
            a = rng.randint(1, 5)
            b = rng.randint(-9, 9)
            n = rng.randint(2, 6)
            v = (a * x + b) ** n
            ans = diff(1 / v, x)
            hand = simplify((0 * v - 1 * (a * n * (a * x + b) ** (n - 1))) / v**2)

            # Independent check
            check(symbolically_equal(ans - hand, 0))

            # Verify distractors are distinct
            forgot_sign = a * n * (a * x + b) ** (-(n + 1))
            no_square = simplify((0 * v - 1 * (a * n * (a * x + b) ** (n - 1))) / v)
            power_rule_wrong = a * n * (a * x + b) ** (n - 1)

            check(not symbolically_equal(ans - forgot_sign, 0))
            check(not symbolically_equal(ans - no_square, 0))
            check(not symbolically_equal(ans - power_rule_wrong, 0))

            candidates = [
                _typed(forgot_sign),
                _typed(no_square),
                _typed(power_rule_wrong),
            ]
            distractors = pick_distractors(_typed(ans), candidates)

            return {
                "text": f"Differentiate 1/({_prose(a * x + b)})^{n} with respect to x.",
                "latex": f"Differentiate 1/({_prose(a * x + b)})^{n} with respect to x.",
                "answer": str(hand),
                "answer_text": _typed(hand),
                "answer_latex": latex_answer(hand),
                "distractors": distractors,
            }

    elif difficulty == "hard":
        kind = rng.choice(["chain_sin", "chain_exp", "product_chain"])

        if kind == "chain_sin":
            # chain rule: sin(a*x + b)
            a = rng.randint(2, 8)
            b = rng.randint(-9, 9)
            u = a * x + b
            f = sin(u)
            ans = a * cos(u)

            # Independent check
            check(symbolically_equal(diff(f, x) - ans, 0))

            # Verify distractors are distinct
            forgot_inner = cos(u)
            swapped_sign = -a * sin(u)
            wrong_trig = a * sin(u)

            check(not symbolically_equal(ans - forgot_inner, 0))
            check(not symbolically_equal(ans - swapped_sign, 0))
            check(not symbolically_equal(ans - wrong_trig, 0))

            candidates = [
                _typed(forgot_inner),
                _typed(swapped_sign),
                _typed(wrong_trig),
            ]
            distractors = pick_distractors(_typed(ans), candidates)

            return {
                "text": f"Differentiate sin({_prose(u)}) with respect to x.",
                "latex": f"Differentiate sin({_prose(u)}) with respect to x.",
                "answer": str(ans),
                "answer_text": _typed(ans),
                "answer_latex": latex_answer(ans),
                "distractors": distractors,
            }

        elif kind == "chain_exp":
            # chain rule: exp(a*x + b)
            a = rng.randint(2, 8)
            b = rng.randint(-9, 9)
            u = a * x + b
            f = exp(u)
            ans = a * exp(u)

            # Independent check
            check(symbolically_equal(diff(f, x) - ans, 0))

            # Verify distractors are distinct
            forgot_inner = exp(u)
            confused_with_product = u * exp(u)
            both_factors = a * u * exp(u)

            check(not symbolically_equal(ans - forgot_inner, 0))
            check(not symbolically_equal(ans - confused_with_product, 0))
            check(not symbolically_equal(ans - both_factors, 0))

            candidates = [
                _typed(forgot_inner),
                _typed(confused_with_product),
                _typed(both_factors),
            ]
            distractors = pick_distractors(_typed(ans), candidates)

            return {
                "text": f"Differentiate exp({_prose(u)}) with respect to x.",
                "latex": f"Differentiate exp({_prose(u)}) with respect to x.",
                "answer": str(ans),
                "answer_text": _typed(ans),
                "answer_latex": latex_answer(ans),
                "distractors": distractors,
            }

        else:  # product_chain
            # product + chain rule: exp(-k*x) * sin(m*x)
            # k and m must both be in [2, 3]
            k = rng.randrange(2, 4)  # [2, 3]
            m = rng.randrange(2, 4)  # [2, 3]
            f = exp(-k * x) * sin(m * x)
            u, v = exp(-k * x), sin(m * x)
            u_prime, v_prime = -k * exp(-k * x), m * cos(m * x)
            ans = u_prime * v + u * v_prime

            # Independent check
            check(symbolically_equal(ans - diff(f, x), 0))

            # Verify distractors are distinct
            forgot_k = simplify(-1 * exp(-k * x) * sin(m * x) + exp(-k * x) * m * cos(m * x))
            forgot_m = simplify(-k * exp(-k * x) * sin(m * x) + exp(-k * x) * cos(m * x))
            mult_derivs = simplify(u_prime * v_prime)

            check(not symbolically_equal(ans - forgot_k, 0))
            check(not symbolically_equal(ans - forgot_m, 0))
            check(not symbolically_equal(ans - mult_derivs, 0))

            candidates = [
                _typed(forgot_k),
                _typed(forgot_m),
                _typed(mult_derivs),
            ]
            distractors = pick_distractors(_typed(ans), candidates)

            return {
                "text": f"Differentiate {_prose(f)} with respect to x.",
                "latex": f"Differentiate {_prose(f)} with respect to x.",
                "answer": str(ans),
                "answer_text": _typed(ans),
                "answer_latex": latex_answer(ans),
                "distractors": distractors,
            }
