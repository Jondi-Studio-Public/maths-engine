"""Senior one-to-one and conjugate roots generators (ACMSM094, ACMSM090).

Skill 1 (ACMSM094): Determine if a function is one-to-one by finding the
critical/turning-point value beyond which a supplied quadratic or
absolute-value function stops being one-to-one (the axis of symmetry or kink
point). Three difficulty tiers: parabola (x - h)^2, parabola with vertical
shift (x - h)^2 + k, absolute-value kink |x - h| + k. All answers are
real-valued integers (the x-coordinate of the critical point).

Skill 2 (ACMSM090): Consider conjugate roots for polynomials with real
coefficients. Given monic quadratics with complex-conjugate roots, state the
sum of roots (-b), the product of roots (c), or (hard tier) build a monic
quadratic from a stated conjugate root pair and find its x-coefficient (-2p).
All answers are real-valued integers. Complex arithmetic appears only in
construction/checking (via sympy.I), never in learner-typed answers.
"""

from __future__ import annotations

import sympy
from sympy import I, Symbol, expand, simplify

from ..core import check, generator, join_terms, pick_distractors

x = Symbol("x")


@generator(
    "determine_if_one_to_one",
    level=11,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def determine_if_one_to_one(rng, tier="medium"):
    """Determine if a function is one-to-one by finding its critical point."""

    if tier == "easy":
        # f(x) = (x - h)^2; no vertical shift at this tier
        # Domain: all x, range: [0, infinity)
        # One-to-one on (-infinity, h]
        # Answer: h (the axis of symmetry)
        h = rng.randint(-8, 8)

        # Build the function
        f_expr = (x - h) ** 2

        # Independent check: verify symmetry by evaluating at h-1, h+1 and h-2, h+2
        f_at_h_minus_1 = f_expr.subs(x, h - 1)
        f_at_h_plus_1 = f_expr.subs(x, h + 1)
        f_at_h_minus_2 = f_expr.subs(x, h - 2)
        f_at_h_plus_2 = f_expr.subs(x, h + 2)

        check(
            f_at_h_minus_1 == f_at_h_plus_1,
            f"Symmetry check failed at h±1: f({h - 1}) = {f_at_h_minus_1}, f({h + 1}) = {f_at_h_plus_1}",
        )
        check(
            f_at_h_minus_2 == f_at_h_plus_2,
            f"Symmetry check failed at h±2: f({h - 2}) = {f_at_h_minus_2}, f({h + 2}) = {f_at_h_plus_2}",
        )

        # Answer text
        answer_text = str(h)

        # Format prompt
        if h >= 0:
            prompt = f"(x - {h})"
        else:
            prompt = f"(x + {-h})"

        # Misconceptions
        # 1. reports-negative-of-h: states -h instead of h (guard: if h==0, then -h=0 which collides)
        neg_h_val = -h
        if neg_h_val == h:
            distractor_neg_h = str(neg_h_val + 1)
        else:
            distractor_neg_h = str(neg_h_val)

        # 2. reports y-intercept: f(0) = h^2 (guard against collision when h^2 == h, e.g., h=0 or h=1)
        y_intercept_val = h * h
        if y_intercept_val == h:
            distractor_y_intercept = str(y_intercept_val + 1)
        else:
            distractor_y_intercept = str(y_intercept_val)

        # 3. reports 0 (guard against collision when h == 0)
        if h == 0:
            distractor_zero = "2"
        else:
            distractor_zero = "0"

        candidates = [distractor_neg_h, distractor_y_intercept, distractor_zero]

        # Verify distractors are distinct from answer
        check(int(distractor_neg_h) != h, f"Distractor {distractor_neg_h} equals answer {h}")
        check(int(distractor_y_intercept) != h, f"Distractor {distractor_y_intercept} equals answer {h}")
        check(int(distractor_zero) != h, f"Distractor {distractor_zero} equals answer {h}")

        return dict(
            text=f"State the largest value a such that f(x) = {prompt}² is one-to-one on (-∞, a].",
            latex=rf"f(x) = {prompt}^2",
            answer=h,
            answer_text=answer_text,
            working=[
                f"The function f(x) = {prompt}² is a parabola with vertex at ({h}, 0).",
                f"A parabola is one-to-one on the interval from its vertex to one side.",
                f"The axis of symmetry (critical point) is at x = {h}.",
                f"So f is one-to-one on (-∞, {h}].",
                f"Answer: {answer_text}",
            ],
            distractors=pick_distractors(answer_text, candidates),
        )

    elif tier == "medium":
        # f(x) = (x - h)^2 + k; both parameters present
        # Domain: all x, range: [k, infinity)
        # One-to-one on (-infinity, h]
        # Answer: h (NOT k, a common trap)
        h = rng.randint(-8, 8)
        k = rng.randint(-8, 8)

        # Exclude k = 0 to make vertical shift visible
        while k == 0:
            k = rng.randint(-8, 8)

        # Build the function
        f_expr = (x - h) ** 2 + k

        # Independent check: verify symmetry
        f_at_h_minus_1 = f_expr.subs(x, h - 1)
        f_at_h_plus_1 = f_expr.subs(x, h + 1)
        f_at_h_minus_2 = f_expr.subs(x, h - 2)
        f_at_h_plus_2 = f_expr.subs(x, h + 2)

        check(f_at_h_minus_1 == f_at_h_plus_1, f"Symmetry check failed at h±1")
        check(f_at_h_minus_2 == f_at_h_plus_2, f"Symmetry check failed at h±2")

        # Answer text
        answer_text = str(h)

        # Format prompt
        if h >= 0:
            h_part = f"(x - {h})"
        else:
            h_part = f"(x + {-h})"

        if k >= 0:
            k_part = f" + {k}"
        else:
            k_part = f" - {-k}"

        # Misconceptions
        # 1. reports-k-instead-of-h: confuses the vertical shift with the critical point (guard: k != h by resampling)
        while k == h:
            k = rng.randint(-8, 8)
        distractor_k = str(k)

        # 2. sign-error-on-h: reports -h instead of h (guard against collision when h == 0)
        neg_h_val = -h
        if neg_h_val == h:
            distractor_neg_h = str(neg_h_val + 1)
        else:
            distractor_neg_h = str(neg_h_val)

        # 3. reports y-intercept: f(0) = h^2 + k (guard against collision)
        y_intercept_val = h * h + k
        if y_intercept_val == h:
            distractor_y_intercept = str(y_intercept_val + 1)
        else:
            distractor_y_intercept = str(y_intercept_val)

        candidates = [distractor_k, distractor_neg_h, distractor_y_intercept]

        # Verify distractors are distinct from answer
        check(int(distractor_k) != h, f"Distractor k={distractor_k} equals answer h={answer_text}")
        check(int(distractor_neg_h) != h, f"Distractor -h={distractor_neg_h} equals answer h={answer_text}")
        check(
            int(distractor_y_intercept) != h,
            f"Distractor y-intercept={distractor_y_intercept} equals answer h={answer_text}",
        )

        return dict(
            text=f"State the largest value a such that f(x) = {h_part}²{k_part} is one-to-one on (-∞, a].",
            latex=rf"f(x) = {h_part}^2 {k_part}",
            answer=h,
            answer_text=answer_text,
            working=[
                f"The function f(x) = {h_part}²{k_part} is a parabola with vertex at ({h}, {k}).",
                f"The vertical shift (parameter k = {k}) does not change the axis of symmetry.",
                f"The axis of symmetry (critical point) is still at x = {h}.",
                f"So f is one-to-one on (-∞, {h}].",
                f"Answer: {answer_text}",
            ],
            distractors=pick_distractors(answer_text, candidates),
        )

    elif tier == "hard":
        # f(x) = |x - h| + k; absolute-value function (V-shaped)
        # Domain: all x, range: [k, infinity)
        # One-to-one on (-infinity, h] OR on [h, infinity)
        # Answer: h (the kink point, where the slope changes)
        h = rng.randint(-8, 8)
        k = rng.randint(-8, 8)

        # Exclude k = 0
        while k == 0:
            k = rng.randint(-8, 8)

        # Build the function
        f_expr = sympy.Abs(x - h) + k

        # Independent check: verify symmetry by direct evaluation
        # For absolute value, we need numerical evaluation
        f_at_h_minus_1 = f_expr.subs(x, h - 1)
        f_at_h_plus_1 = f_expr.subs(x, h + 1)
        f_at_h_minus_2 = f_expr.subs(x, h - 2)
        f_at_h_plus_2 = f_expr.subs(x, h + 2)

        check(f_at_h_minus_1 == f_at_h_plus_1, f"Symmetry check failed at h±1 for absolute value")
        check(f_at_h_minus_2 == f_at_h_plus_2, f"Symmetry check failed at h±2 for absolute value")

        # Answer text
        answer_text = str(h)

        # Format prompt
        if h >= 0:
            h_part = f"|x - {h}|"
        else:
            h_part = f"|x + {-h}|"

        if k >= 0:
            k_part = f" + {k}"
        else:
            k_part = f" - {-k}"

        # Misconceptions
        # 1. reports-k-instead-of-h: confuses the vertical shift with the critical point (guard: k != h by resampling)
        while k == h:
            k = rng.randint(-8, 8)
        distractor_k = str(k)

        # 2. sign-error-on-h: reports -h instead of h (guard against collision when h == 0)
        neg_h_val = -h
        if neg_h_val == h:
            distractor_neg_h = str(neg_h_val + 1)
        else:
            distractor_neg_h = str(neg_h_val)

        # 3. reports y-intercept: f(0) = |0 - h| + k = |h| + k (guard against collision)
        y_intercept_val = abs(h) + k
        if y_intercept_val == h:
            distractor_y_intercept = str(y_intercept_val + 1)
        else:
            distractor_y_intercept = str(y_intercept_val)

        candidates = [distractor_k, distractor_neg_h, distractor_y_intercept]

        # Verify distractors are distinct from answer
        check(int(distractor_k) != h, f"Distractor k={distractor_k} equals answer h={answer_text}")
        check(int(distractor_neg_h) != h, f"Distractor -h={distractor_neg_h} equals answer h={answer_text}")
        check(
            int(distractor_y_intercept) != h,
            f"Distractor y-intercept={distractor_y_intercept} equals answer h={answer_text}",
        )

        return dict(
            text=f"State the value a such that f(x) = {h_part}{k_part} is one-to-one on (-∞, a] (including the kink point).",
            latex=rf"f(x) = {h_part} {k_part}",
            answer=h,
            answer_text=answer_text,
            working=[
                f"The function f(x) = {h_part}{k_part} is a V-shaped absolute-value function.",
                f"The vertical shift (parameter k = {k}) does not change the kink point.",
                f"The kink point (critical point) is at x = {h}.",
                f"On (-∞, {h}], the function decreases (one-to-one); on [{h}, ∞), it increases (one-to-one).",
                f"So f is one-to-one on (-∞, {h}] (or on [{h}, ∞)).",
                f"Answer: {answer_text}",
            ],
            distractors=pick_distractors(answer_text, candidates),
        )

    else:
        raise ValueError(f"Unknown tier: {tier}")


@generator(
    "conjugate_roots_real_coefficients",
    level=11,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def conjugate_roots_real_coefficients(rng, tier="medium"):
    """Work with conjugate roots of monic quadratics with real coefficients."""

    if tier == "easy":
        # Monic quadratic: x^2 + b*x + c
        # Complex-conjugate roots exist when discriminant b^2 - 4*c < 0
        # For a monic quadratic, sum of roots = -b (by Vieta's formulas)
        # Answer: -b (always real)

        b = rng.randint(-6, 6)
        # Sample c to ensure discriminant < 0
        # c = b^2 // 4 + k, where k in [1, 5]
        k = rng.randint(1, 5)
        c = (b * b) // 4 + k

        # Verify discriminant is negative
        discriminant = b * b - 4 * c
        check(discriminant < 0, f"Discriminant {discriminant} is not negative for b={b}, c={c}")

        # Independent check: use sympy to solve the quadratic
        roots = sympy.solve(x**2 + b * x + c, x)
        check(len(roots) == 2, f"Expected 2 roots, got {len(roots)}")

        # Sum the roots using sympy
        root_sum = sympy.simplify(roots[0] + roots[1])
        # Verify sum is real and equals -b
        check(sympy.im(root_sum) == 0, f"Root sum is not real: {root_sum}")
        check(sympy.re(root_sum) == -b, f"Root sum {sympy.re(root_sum)} does not equal -b={-b}")

        # Answer: -b
        answer_value = -b
        answer_text = str(answer_value)

        # Misconceptions with collision guards
        # Build distractors ensuring they're distinct from answer and each other
        used = {answer_value}

        # 1. sign-flip: states b instead of -b
        distractor_b = b
        while distractor_b in used:
            distractor_b = b + 1 if distractor_b == b else distractor_b + 1
        used.add(distractor_b)

        # 2. confuses-with-product: states c instead of -b
        distractor_c = c
        while distractor_c in used:
            distractor_c = c + 1 if distractor_c == c else distractor_c + 1
        used.add(distractor_c)

        # 3. miscalculates as 0
        distractor_other = 0
        while distractor_other in used:
            distractor_other += 1
        used.add(distractor_other)

        candidates = [str(distractor_b), str(distractor_c), str(distractor_other)]

        # Verify all are distinct
        check(distractor_b != answer_value, f"Distractor b={distractor_b} equals answer {answer_text}")
        check(distractor_c != answer_value, f"Distractor c={distractor_c} equals answer {answer_text}")
        check(distractor_other != answer_value, f"Distractor other={distractor_other} equals answer {answer_text}")
        check(len(set(candidates)) == 3, f"Duplicate distractors: {candidates}")

        # Build the quadratic expression text (avoid "1x" or "-1x")
        if b >= 0:
            b_sign = "+"
            b_val = b
        else:
            b_sign = "-"
            b_val = -b

        # Format coefficient of x (omit 1, show -1 as -)
        if b_val == 1:
            b_coeff_str = f"{b_sign} x"
        elif b_val == -1:
            b_coeff_str = "- x"
        else:
            b_coeff_str = f"{b_sign} {b_val}x"

        # Format latex coefficient (avoid 1x)
        if b_val == 1:
            b_latex_str = f"{b_sign} x"
        elif b_val == -1:
            b_latex_str = "- x"
        else:
            b_latex_str = f"{b_sign} {b_val}x"

        return dict(
            text=f"For the quadratic x² {b_coeff_str} + {c}, which has complex roots, state the sum of the roots.",
            latex=rf"x^2 {b_latex_str} + {c}",
            answer=answer_value,
            answer_text=answer_text,
            working=[
                f"The quadratic is x² {b_coeff_str} + {c}.",
                f"For a monic quadratic x² + px + q, the sum of roots equals -p (Vieta's formula).",
                f"Here, the coefficient of x is {b}, so the sum of roots = -({b}) = {answer_value}.",
                f"Answer: {answer_text}",
            ],
            distractors=pick_distractors(answer_text, candidates),
        )

    elif tier == "medium":
        # Monic quadratic: x^2 + b*x + c
        # For a monic quadratic, product of roots = c (by Vieta's formulas)
        # Answer: c (always real, but different relation from easy)

        b = rng.randint(-6, 6)
        # Sample c to ensure discriminant < 0
        k = rng.randint(1, 5)
        c = (b * b) // 4 + k

        # Verify discriminant is negative
        discriminant = b * b - 4 * c
        check(discriminant < 0, f"Discriminant {discriminant} is not negative for b={b}, c={c}")

        # Independent check: use sympy to solve the quadratic
        roots = sympy.solve(x**2 + b * x + c, x)
        check(len(roots) == 2, f"Expected 2 roots, got {len(roots)}")

        # Product the roots using sympy
        root_product = sympy.simplify(roots[0] * roots[1])
        # Verify product is real and equals c
        check(sympy.im(root_product) == 0, f"Root product is not real: {root_product}")
        check(sympy.re(root_product) == c, f"Root product {sympy.re(root_product)} does not equal c={c}")

        # Answer: c
        answer_value = c
        answer_text = str(answer_value)

        # Misconceptions with collision guards
        used = {answer_value}

        # 1. sign-error: states -c instead of c
        distractor_neg_c = -c
        while distractor_neg_c in used:
            distractor_neg_c = -c + 1 if distractor_neg_c == -c else distractor_neg_c + 1
        used.add(distractor_neg_c)

        # 2. confuses-with-sum: states -b instead of c
        distractor_sum = -b
        while distractor_sum in used:
            distractor_sum = -b + 1 if distractor_sum == -b else distractor_sum + 1
        used.add(distractor_sum)

        # 3. Another error: states b
        distractor_b = b
        while distractor_b in used:
            distractor_b = b + 1 if distractor_b == b else distractor_b + 1
        used.add(distractor_b)

        candidates = [str(distractor_neg_c), str(distractor_sum), str(distractor_b)]

        # Verify all are distinct
        check(distractor_neg_c != answer_value, f"Distractor -c={distractor_neg_c} equals answer {answer_text}")
        check(distractor_sum != answer_value, f"Distractor sum={distractor_sum} equals answer {answer_text}")
        check(distractor_b != answer_value, f"Distractor b={distractor_b} equals answer {answer_text}")
        check(len(set(candidates)) == 3, f"Duplicate distractors: {candidates}")

        # Build the quadratic expression text (avoid "1x" or "-1x")
        if b >= 0:
            b_sign = "+"
            b_val = b
        else:
            b_sign = "-"
            b_val = -b

        # Format coefficient of x (omit 1, show -1 as -)
        if b_val == 1:
            b_coeff_str = f"{b_sign} x"
        elif b_val == -1:
            b_coeff_str = "- x"
        else:
            b_coeff_str = f"{b_sign} {b_val}x"

        # Format latex coefficient (avoid 1x)
        if b_val == 1:
            b_latex_str = f"{b_sign} x"
        elif b_val == -1:
            b_latex_str = "- x"
        else:
            b_latex_str = f"{b_sign} {b_val}x"

        return dict(
            text=f"For the quadratic x² {b_coeff_str} + {c}, which has complex roots, state the product of the roots.",
            latex=rf"x^2 {b_latex_str} + {c}",
            answer=answer_value,
            answer_text=answer_text,
            working=[
                f"The quadratic is x² {b_coeff_str} + {c}.",
                f"For a monic quadratic x² + px + q, the product of roots equals q (Vieta's formula).",
                f"Here, the constant term is {c}, so the product of roots = {answer_value}.",
                f"Answer: {answer_text}",
            ],
            distractors=pick_distractors(answer_text, candidates),
        )

    elif tier == "hard":
        # Reverse direction: given a conjugate root pair p ± qi
        # Find the coefficient of x in the monic quadratic with these roots
        # The quadratic is (x - (p + qi))(x - (p - qi)) = x^2 - 2p*x + (p^2 + q^2)
        # Coefficient of x = -2p
        # Answer: -2p (always real, an integer)

        p = rng.randint(-6, 6)
        q = rng.randint(1, 6)  # q must be nonzero and positive

        # Exclude p = 0 to make the problem meaningful
        while p == 0:
            p = rng.randint(-6, 6)

        # Build the conjugate roots symbolically
        root1 = p + q * I
        root2 = p - q * I

        # Expand the quadratic using sympy
        quadratic = expand((x - root1) * (x - root2))

        # Independent check: verify the quadratic has real coefficients
        # Extract coefficients
        from sympy import Poly

        poly = Poly(quadratic, x)
        coeffs = poly.all_coeffs()

        for coeff in coeffs:
            check(sympy.im(coeff) == 0, f"Quadratic has non-real coefficient: {coeff}")

        # Verify the x-coefficient is -2p
        x_coeff = coeffs[1]  # coefficient of x in descending order (x^2, x, constant)
        check(x_coeff == -2 * p, f"Coefficient of x is {x_coeff}, expected -2*{p} = {-2 * p}")

        # Answer: -2p
        answer_value = -2 * p
        answer_text = str(answer_value)

        # Misconceptions with collision guards
        used = {answer_value}

        # 1. forgot-to-double: states -p instead of -2p
        distractor_forgot_double = -p
        while distractor_forgot_double in used:
            distractor_forgot_double = -p + 1 if distractor_forgot_double == -p else distractor_forgot_double + 1
        used.add(distractor_forgot_double)

        # 2. sign-error: states 2p instead of -2p
        distractor_sign_error = 2 * p
        while distractor_sign_error in used:
            distractor_sign_error = 2 * p + 1 if distractor_sign_error == 2 * p else distractor_sign_error + 1
        used.add(distractor_sign_error)

        # 3. Another error: states -2q instead of -2p
        distractor_2q = -2 * q
        while distractor_2q in used:
            distractor_2q = -2 * q + 1 if distractor_2q == -2 * q else distractor_2q + 1
        used.add(distractor_2q)

        candidates = [str(distractor_forgot_double), str(distractor_sign_error), str(distractor_2q)]

        # Verify all are distinct
        check(
            distractor_forgot_double != answer_value,
            f"Distractor -p={distractor_forgot_double} equals answer {answer_text}",
        )
        check(
            distractor_sign_error != answer_value, f"Distractor 2p={distractor_sign_error} equals answer {answer_text}"
        )
        check(distractor_2q != answer_value, f"Distractor -2q={distractor_2q} equals answer {answer_text}")
        check(len(set(candidates)) == 3, f"Duplicate distractors: {candidates}")

        # Format the root pair display (p ± qi)
        root_pos_text = join_terms([(p, ""), (q, "i")])
        root_neg_text = join_terms([(p, ""), (-q, "i")])

        return dict(
            text=f"A quadratic with real coefficients has complex-conjugate roots {root_pos_text} and {root_neg_text}. State the coefficient of x in the monic quadratic with these roots.",
            latex=rf"({root_pos_text}) \text{{ and }} ({root_neg_text})",
            answer=answer_value,
            answer_text=answer_text,
            working=[
                f"The roots are {root_pos_text} and {root_neg_text} (a conjugate pair).",
                f"For a monic quadratic, the coefficient of x equals the negative of the sum of roots.",
                f"Sum of roots = ({root_pos_text}) + ({root_neg_text}) = {2 * p}.",
                f"Coefficient of x = -({2 * p}) = {answer_value}.",
                f"Answer: {answer_text}",
            ],
            distractors=pick_distractors(answer_text, candidates),
        )

    else:
        raise ValueError(f"Unknown tier: {tier}")
