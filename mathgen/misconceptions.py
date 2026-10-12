"""The shared misconception catalogue: one id per error a learner makes, whichever skill it turns up in.

A generator tags each distractor with an id from here (`mathgen.answers.Distractor`). The id is the engine's, not a
skill's: one generator serves several skills, so it cannot know a skill's own ids, and an id that means the same
thing everywhere is what lets a learner's sign errors be counted across skills. A skill YAML `misconceptions` or
`near_misses` entry with the same id overrides the wording shown for that skill; grading looks there first.

`kind` sets the feedback tone, as the two YAML lists do: a `misconception` is wrong thinking, a `near_miss` is right
thinking that answered a different question (the change instead of the percentage change).

Add an entry when a generator needs one: a short kebab-case id that names the error, not the topic it was first seen
in, and a description a learner could read. Reuse an existing id whenever it says the same thing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CatalogueEntry:
    id: str
    description: str
    kind: str = "misconception"  # or "near_miss"
    area: str = "general"


_ENTRIES = (
    # -- equations ------------------------------------------------------------------------------------------------
    CatalogueEntry(
        "sign-error-moving-term",
        "Moving a term to the other side of the equals sign without changing its sign: adding it where it should be "
        "subtracted, or the reverse.",
        area="equations",
    ),
    CatalogueEntry(
        "forgot-to-divide-by-coefficient",
        "Stopping once the x term is alone, without dividing by its coefficient.",
        area="equations",
    ),
    CatalogueEntry(
        "ignored-constant-term",
        "Leaving out the constant term and dividing the other side by the coefficient straight away.",
        area="equations",
    ),
    # -- quadratics -----------------------------------------------------------------------------------------------
    CatalogueEntry(
        "sign-flip-both-roots",
        "Reading the roots straight off the factors' numbers, so both roots have the wrong sign: (x - 2)(x + 3) = 0 "
        "read as x = -2 or x = 3.",
        area="quadratics",
    ),
    CatalogueEntry(
        "sign-flip-one-root",
        "Getting the sign of one root wrong when reading the roots from the factors.",
        area="quadratics",
    ),
    CatalogueEntry(
        "missed-negative-root",
        "Taking only the positive square root, so x^2 = 9 gives x = 3 and loses x = -3.",
        area="quadratics",
    ),
    # -- substitution and functions ----------------------------------------------------------------------------------
    CatalogueEntry(
        "sign-on-power",
        "Ignoring the sign of a negative number when it is squared: reading (-3)^2 as -9.",
        area="functions",
    ),
    CatalogueEntry(
        "squared-as-doubled",
        "Doubling a number where it should be squared: reading 5^2 as 10.",
        area="functions",
    ),
    CatalogueEntry(
        "sign-error-linear-term",
        "Losing the sign of a negative input in the x term: reading 4(-2) as 8.",
        area="functions",
    ),
    CatalogueEntry(
        "dropped-constant-term",
        "Leaving out the constant term when evaluating the expression.",
        area="functions",
    ),
    # -- limits ---------------------------------------------------------------------------------------------------
    CatalogueEntry(
        "stopped-at-undefined",
        "Substituting, reaching 0/0, and stopping there instead of simplifying first.",
        area="limits",
    ),
    CatalogueEntry("answered-zero", "Seeing the zero numerator and answering zero.", area="limits"),
    CatalogueEntry(
        "sign-error-cancelling",
        "Getting the sign wrong in the factor left over after cancelling.",
        area="limits",
    ),
    CatalogueEntry(
        "no-limit-at-hole",
        "Believing a limit cannot exist where the function itself is undefined.",
        area="limits",
    ),
    CatalogueEntry(
        "assumed-infinite",
        "Assuming the limit itself is infinite because x tends to infinity.",
        area="limits",
    ),
    CatalogueEntry(
        "denominator-grows-so-zero",
        "Deciding the limit is 0 because the denominator grows without bound, though the numerator grows as fast.",
        area="limits",
    ),
    CatalogueEntry("inverted-ratio", "Inverting the ratio: dividing the wrong way round.", area="general"),
    CatalogueEntry(
        "ignored-degree-mismatch",
        "Applying the leading-coefficient ratio when the degrees of the numerator and denominator differ.",
        area="limits",
    ),
    CatalogueEntry(
        "inverted-degree-comparison",
        "Mixing up which way the degree comparison goes: a higher-degree numerator gives an infinite limit, not 0.",
        area="limits",
    ),
    CatalogueEntry("wrong-infinite-sign", "Getting the sign of an infinite limit wrong.", area="limits"),
    CatalogueEntry(
        "no-limit-at-infinity",
        "Believing a limit at infinity cannot exist because x never reaches infinity.",
        area="limits",
    ),
    CatalogueEntry(
        "ignored-side",
        "Ignoring which side the limit is approached from and reading the other piece.",
        area="limits",
    ),
    CatalogueEntry(
        "no-one-sided-limit-at-jump",
        "Believing a one-sided limit cannot exist where the function jumps; only the two-sided limit fails there.",
        area="limits",
    ),
    CatalogueEntry(
        "used-first-listed-piece",
        "Substituting into whichever piece of the function is written first, whatever side is asked for.",
        area="limits",
    ),
    CatalogueEntry(
        "substitution-sign-error",
        "Getting a sign wrong when substituting a negative value into a term.",
        area="general",
    ),
    # -- percentages ----------------------------------------------------------------------------------------------
    CatalogueEntry(
        "gave-change-not-percent",
        "Giving the amount of the change where the percentage change was asked for.",
        kind="near_miss",
        area="percentages",
    ),
    CatalogueEntry(
        "new-as-percent-of-original",
        "Giving the new value as a percentage of the original (120%) instead of the change as a percentage (20%).",
        area="percentages",
    ),
    CatalogueEntry(
        "divides-by-final",
        "Dividing the change by the final value instead of the original value.",
        area="percentages",
    ),
    CatalogueEntry(
        "subtracts-percent",
        "Undoing a percentage increase by taking the same percentage off the final value, though that percentage was "
        "of the original.",
        area="percentages",
    ),
    CatalogueEntry(
        "percent-as-absolute",
        "Treating a percentage as a fixed amount: undoing a 20% rise by subtracting 20.",
        area="percentages",
    ),
    # -- sc8: probability forms and lines of best fit ---------------------------------------------------------------
    CatalogueEntry(
        "found-complement-not-event",
        "Answering with the probability (or count) of everything else instead of the event asked about.",
        area="probability",
    ),
    CatalogueEntry(
        "decimal-percent-no-scaling",
        "Reading a decimal probability as a percentage without multiplying by 100: 0.28 taken as 0.28%.",
        area="probability",
    ),
    CatalogueEntry(
        "fraction-numerator-as-percent",
        "Reading a fraction as a percentage by using its numerator: 3/20 taken as 3%.",
        area="probability",
    ),
    CatalogueEntry(
        "percent-not-divided-by-100",
        "Finding a percentage of a quantity by multiplying by the percentage figure without dividing by 100.",
        area="percentages",
    ),
    CatalogueEntry(
        "used-whole-group-as-denominator",
        "Finding a share of a sub-group as a share of the whole group.",
        area="probability",
    ),
    CatalogueEntry(
        "used-wrong-subgroup-as-denominator",
        "Dividing by the size of the wrong sub-group when finding a share of a group.",
        area="probability",
    ),
    CatalogueEntry(
        "gradient-sign-from-point-order",
        "Subtracting the y values in one order and the x values in the other, so the gradient has the wrong sign.",
        area="linear",
    ),
    CatalogueEntry(
        "gradient-rise-only",
        "Giving the change in y between two points as the gradient, without dividing by the change in x.",
        area="linear",
    ),
    CatalogueEntry(
        "intercept-read-as-point-height",
        "Taking the y value of a point on the line as the y-intercept, without allowing for the gradient.",
        area="linear",
    ),
    CatalogueEntry(
        "intercept-sign-error",
        "Finding c from y = mx + c by adding mx to y instead of subtracting it.",
        area="linear",
    ),
    CatalogueEntry(
        "swapped-x-and-y",
        "Putting a value in the wrong variable: substituting a y value for x, or solving for x when y was wanted.",
        area="linear",
    ),
    CatalogueEntry(
        "gradient-and-intercept-swapped",
        "Using the intercept as the gradient and the gradient as the intercept in y = mx + c.",
        area="linear",
    ),
    CatalogueEntry(
        "substituted-without-multiplying",
        "Adding the gradient and the intercept instead of multiplying the gradient by x first.",
        area="linear",
    ),
    CatalogueEntry(
        "residual-sign-reversed",
        "Finding a residual as predicted minus observed instead of observed minus predicted.",
        area="bivariate data",
    ),
    CatalogueEntry(
        "gave-prediction-not-residual",
        "Giving the value the line predicts when the residual (the distance from the line) was asked for.",
        area="bivariate data",
        kind="near_miss",
    ),
    CatalogueEntry(
        "gave-observed-not-residual",
        "Giving the observed value of a data point when the residual was asked for.",
        area="bivariate data",
        kind="near_miss",
    ),
    # -- sc7: counterexamples, implication counts and quantifier thresholds ------------------------------------------
    CatalogueEntry(
        "example-confused-with-counterexample",
        "Offering a value that satisfies the claim as the counterexample that breaks it.",
        area="logic",
    ),
    CatalogueEntry(
        "reports-value-not-input",
        "Giving the value of the expression at the counterexample when the counterexample itself (the input) was asked for.",
        area="logic",
    ),
    CatalogueEntry(
        "skips-to-a-later-counterexample",
        "Naming a counterexample that does break the claim but is not the smallest, by not testing the values in order.",
        area="logic",
    ),
    CatalogueEntry(
        "evaluates-at-the-wrong-input",
        "Substituting a neighbouring value of the input instead of the one the earlier part found.",
        area="logic",
    ),
    CatalogueEntry(
        "takes-the-square-root-of-the-modulus",
        "Guessing that the smallest n with m dividing n^2 is about the square root of m, without testing it.",
        area="logic",
    ),
    CatalogueEntry(
        "stops-before-the-final-division",
        "Quoting the square n^2 where the quotient n^2 divided by m was asked, stopping before the last step.",
        area="logic",
    ),
    CatalogueEntry(
        "repeats-the-previous-answer",
        "Writing the answer to the previous part again instead of working out what this part asks.",
        area="general",
    ),
    CatalogueEntry(
        "ignores-one-of-the-two-divisors",
        "Choosing a value that is divisible by only one of the two divisors, as if one were enough.",
        area="logic",
    ),
    CatalogueEntry(
        "forgets-to-exclude-the-non-counterexamples",
        "Counting every multiple of the lowest common multiple as a counterexample, though some are multiples of the product too.",
        area="logic",
    ),
    CatalogueEntry(
        "counts-the-examples-not-the-counterexamples",
        "Counting the values that satisfy the claim when the values that break it were asked for.",
        area="logic",
    ),
    CatalogueEntry(
        "ignores-the-conclusion",
        "Counting every case where the hypothesis holds, without checking whether the conclusion fails.",
        area="logic",
    ),
    CatalogueEntry(
        "swaps-hypothesis-and-conclusion",
        "Answering for the converse when the original implication was asked, or the reverse.",
        area="logic",
    ),
    CatalogueEntry(
        "treats-implication-as-and",
        "Counting the cases where both statements are true, as if an implication meant both at once.",
        area="logic",
    ),
    CatalogueEntry(
        "counts-true-cases-not-false",
        "Counting the cases where the statement is true when the cases where it is false were asked for.",
        area="logic",
    ),
    CatalogueEntry(
        "checks-only-one-direction",
        "Checking one direction of an if-and-only-if and ignoring the other direction.",
        area="logic",
    ),
    CatalogueEntry(
        "treats-both-true-as-failing",
        "Counting the cases where both statements are true as cases where an if-and-only-if fails.",
        area="logic",
    ),
    CatalogueEntry(
        "sign-error-vertex-formula",
        "Using b/(2a) for the x-value of the vertex of a quadratic, losing the minus sign.",
        area="quadratics",
    ),
    CatalogueEntry(
        "forgot-the-two-in-the-vertex-formula",
        "Using -b/a, not -b/(2a), for the x-value of the vertex of a quadratic.",
        area="quadratics",
    ),
    CatalogueEntry(
        "ignored-the-leading-coefficient",
        "Leaving the coefficient of x^2 out of a calculation with a quadratic, as if it were 1.",
        area="quadratics",
    ),
    CatalogueEntry(
        "sign-error-minimum",
        "Reporting the least value of a quadratic with the wrong sign, or using it with the wrong sign in the next step.",
        area="quadratics",
    ),
    CatalogueEntry(
        "forgot-to-square",
        "Using x where x^2 is needed when substituting into a quadratic.",
        area="quadratics",
    ),
    CatalogueEntry(
        "includes-the-boundary-value",
        "Taking the boundary value of a parameter as allowed when the inequality is strict, so the statement just fails there.",
        area="logic",
    ),
    CatalogueEntry(
        "reverses-the-inequality-direction",
        "Giving the largest allowed integer where the smallest was asked, or the reverse, by reading the inequality the wrong way round.",
        area="logic",
    ),
    CatalogueEntry(
        "treats-inclusive-bound-as-strict",
        "Moving past the boundary value of a parameter when the inequality includes it.",
        area="logic",
    ),
    # -- sc5: units of mass and capacity, Jacobians, roots, logarithm equations and factor counts -------------------
    CatalogueEntry(
        "conversion-power-of-ten-slip",
        "Moving the decimal point the wrong number of places when converting between metric units (dividing by 100 or 10 000 instead of 1000).",
        area="units",
    ),
    CatalogueEntry(
        "converted-wrong-direction",
        "Multiplying where a unit conversion needs dividing, or the reverse (turning grams into kilograms by multiplying by 1000).",
        area="units",
    ),
    CatalogueEntry(
        "forgot-to-convert-units",
        "Using a quantity in its original unit where the answer needed another (grams left as grams, cm³ left as cm³).",
        area="units",
    ),
    CatalogueEntry(
        "mixed-units-in-arithmetic",
        "Subtracting or adding two quantities still in different units (tonnes with kilograms, litres with millilitres).",
        area="units",
    ),
    CatalogueEntry(
        "gave-amount-for-one-item",
        "Giving the amount for one item where the total for all of them was asked.",
        area="units",
    ),
    CatalogueEntry(
        "ignored-fill-depth",
        "Using the whole container (its full capacity) where only the part filled to a stated depth was meant.",
        area="units",
    ),
    CatalogueEntry(
        "used-the-empty-part",
        "Finding the amount for the empty part of a container, the depth not filled, instead of the part that is filled.",
        area="units",
    ),
    CatalogueEntry(
        "inverted-the-fraction",
        "Multiplying by the fraction the wrong way up (height over depth where depth over height was needed).",
        area="general",
    ),
    CatalogueEntry(
        "added-instead-of-subtracted",
        "Adding two amounts where the difference was asked (the load plus the limit, not the limit less the load).",
        area="general",
    ),
    CatalogueEntry(
        "subtracted-in-wrong-order",
        "Subtracting in the wrong order, so a spare amount comes out negative.",
        area="general",
    ),
    CatalogueEntry(
        "multiplied-instead-of-divided",
        "Multiplying where the count needs a division (batch mass times tablet strength instead of divided by it).",
        area="general",
    ),
    CatalogueEntry(
        "off-by-one-endpoint",
        "Being one out at an end of a count or range: including or leaving out a boundary value.",
        area="general",
    ),
    CatalogueEntry(
        "determinant-formula-misremembered",
        "Using the wrong formula for a 2 by 2 determinant, such as only the leading diagonal or the sum of the diagonal entries.",
        area="multivariable",
    ),
    CatalogueEntry(
        "determinant-added-cross-terms",
        "Adding the product of the off-diagonal entries to the product of the diagonal instead of subtracting it.",
        area="multivariable",
    ),
    CatalogueEntry(
        "substituted-origin-for-point",
        "Differentiating correctly but substituting the origin instead of the stated point.",
        area="multivariable",
    ),
    CatalogueEntry(
        "forgot-to-take-absolute-value",
        "Reporting a signed (negative) area scale factor as an area, instead of taking the absolute value of the determinant.",
        area="multivariable",
    ),
    CatalogueEntry(
        "forgot-area-scale-factor",
        "Giving the original area (or the determinant alone) as the area of the image, without multiplying by the scale factor.",
        area="multivariable",
    ),
    CatalogueEntry(
        "divided-by-scale-factor",
        "Dividing the area by the determinant where the image area is the product of the two.",
        area="multivariable",
    ),
    CatalogueEntry(
        "left-out-the-base-value",
        "Giving only the change from the linear terms and leaving out the value of the function at the starting point.",
        area="multivariable",
    ),
    CatalogueEntry(
        "dropped-one-variable-term",
        "Using the partial derivative for one variable only in a linear estimate that needs both.",
        area="multivariable",
    ),
    CatalogueEntry(
        "used-wrong-component-row",
        "Using the Jacobian row of the other output function when estimating a component.",
        area="multivariable",
    ),
    CatalogueEntry(
        "forgot-the-increments",
        "Adding the partial derivatives without multiplying each by the change in its variable.",
        area="multivariable",
    ),
    CatalogueEntry(
        "ignored-multiplier-under-root",
        "Testing the numbers k themselves for being perfect squares, ignoring the multiplier inside the root.",
        area="square-roots",
    ),
    CatalogueEntry(
        "counted-every-square-up-to-the-largest",
        "Counting every perfect square up to the largest number under the root, not only those that the multiplier produces.",
        area="square-roots",
    ),
    CatalogueEntry(
        "divided-instead-of-testing-perfect-power",
        "Dividing the length of the list by the multiplier instead of testing which products are perfect powers.",
        area="square-roots",
    ),
    CatalogueEntry(
        "counted-the-wrong-class",
        "Counting the other class (rational for irrational, even for odd, or all of them) than the one asked for.",
        area="general",
    ),
    CatalogueEntry(
        "assumed-every-root-irrational",
        "Assuming every root of a whole number is irrational.",
        area="square-roots",
    ),
    CatalogueEntry(
        "gave-radicand-not-root",
        "Giving the number under the root (or the index k) instead of the value of the root.",
        area="square-roots",
    ),
    CatalogueEntry(
        "took-root-of-the-original-number",
        "Taking the root of the given number, or of the last number in the list, instead of the value asked for.",
        area="square-roots",
    ),
    CatalogueEntry(
        "gave-root-not-asked-value",
        "Giving the square root where the number to multiply by, to add, to set aside, or the square itself was asked for.",
        area="square-roots",
    ),
    CatalogueEntry(
        "used-wrong-neighbouring-square",
        "Measuring from the wrong perfect square next to the number (the one below when adding, the one above when removing).",
        area="square-roots",
    ),
    CatalogueEntry(
        "multiplied-by-wrong-factor",
        "Multiplying by the number itself, or by the square of the missing factor, instead of the missing factor.",
        area="square-roots",
    ),
    CatalogueEntry(
        "gave-the-original-number",
        "Repeating the number in the question where the changed number was asked for.",
        area="general",
    ),
    CatalogueEntry(
        "forgot-to-exponentiate",
        "Writing the exponent of the logarithm equation as the argument, leaving out the step of raising the base to it.",
        area="logarithms",
    ),
    CatalogueEntry(
        "wrong-exponential-conversion",
        "Rewriting log_b(y) = n as y = bn, y = n^b or y = b + n instead of y = b^n.",
        area="logarithms",
    ),
    CatalogueEntry(
        "log-sum-as-log-of-sum",
        "Treating log(a) + log(b) as log(a + b), so adding the arguments instead of multiplying them.",
        area="logarithms",
    ),
    CatalogueEntry(
        "kept-the-invalid-root",
        "Keeping a root of the quadratic that makes the argument of a logarithm negative.",
        area="logarithms",
    ),
    CatalogueEntry(
        "slip-solving-the-quadratic",
        "Making an arithmetic slip while solving the quadratic, reaching a value close to the root.",
        area="logarithms",
    ),
    CatalogueEntry(
        "evaluated-wrong-log-expression",
        "Giving the sum of the logarithms, the other bracket's logarithm, or the argument itself where the logarithm of one bracket was asked.",
        area="logarithms",
    ),
    CatalogueEntry(
        "multiplied-exponents-without-adding-one",
        "Multiplying the exponents of a prime factorisation to count factors, instead of multiplying each exponent plus one.",
        area="factors",
    ),
    CatalogueEntry(
        "added-instead-of-multiplied",
        "Adding two quantities where they should be multiplied (the exponent-plus-one terms when counting factors, an area and its scale factor).",
        area="factors",
    ),
    CatalogueEntry(
        "counted-primes-not-factors",
        "Counting the prime factors (or the distinct primes) instead of all the factors.",
        area="factors",
    ),
    CatalogueEntry(
        "left-in-factors-of-two",
        "Leaving some factors of 2 in when asked for the largest odd factor.",
        area="factors",
    ),
    CatalogueEntry(
        "gave-largest-prime-factor",
        "Giving the largest prime factor where the largest odd factor was asked for.",
        area="factors",
    ),
    CatalogueEntry(
        "left-out-one-and-the-number",
        "Leaving out the factors 1 and the number itself when counting factors.",
        area="factors",
    ),
    # -- sc1: remainders, checking arithmetic, equilibria, linear equations and collecting terms ------------------
    CatalogueEntry(
        "remainder-theorem-wrong-sign",
        "Using the opposite sign for the root of the divisor: evaluating P(-c) where the divisor is x - c.",
        area="polynomials",
    ),
    CatalogueEntry(
        "remainder-ignored-given",
        "Setting the remainder to zero, as if the divisor were a factor, when the question states a non-zero remainder.",
        area="polynomials",
    ),
    CatalogueEntry(
        "lost-a-power-when-collecting",
        "Dividing by c where the collected coefficient is c squared, so a power of the divisor's root is lost.",
        area="polynomials",
    ),
    CatalogueEntry(
        "synthetic-division-no-multiply",
        "Adding the next coefficient in synthetic division without first multiplying the previous result by the root.",
        area="polynomials",
    ),
    CatalogueEntry(
        "wrong-quotient-coefficient",
        "Giving a different coefficient of the quotient from the one asked for, such as the x term when the constant term was asked.",
        kind="near_miss",
        area="polynomials",
    ),
    CatalogueEntry(
        "quotient-confused-with-remainder",
        "Giving the remainder where a coefficient of the quotient was asked for.",
        kind="near_miss",
        area="polynomials",
    ),
    CatalogueEntry(
        "quotient-evaluated-for-remainder",
        "Evaluating the quotient at the root instead of the original polynomial when asked for a remainder.",
        area="polynomials",
    ),
    CatalogueEntry(
        "blamed-first-step",
        "Naming the first step as the error without recomputing each step from its own stated inputs.",
        area="arithmetic",
    ),
    CatalogueEntry(
        "blamed-step-after-slip",
        "Naming the step after the slip, because it was recomputed from the value the previous step should have had rather than from the value it stated.",
        area="arithmetic",
    ),
    CatalogueEntry(
        "blamed-last-step",
        "Naming the last step because the final result looks wrong, instead of recomputing each step.",
        area="arithmetic",
    ),
    CatalogueEntry(
        "accepted-the-stated-result",
        "Giving the stated, incorrect result again instead of recomputing it.",
        area="arithmetic",
    ),
    CatalogueEntry(
        "swapped-the-operation",
        "Using a different operation from the one in the step, such as adding where the step multiplies.",
        area="arithmetic",
    ),
    CatalogueEntry(
        "transposed-digits",
        "Writing the right digits of a result in the wrong order.",
        area="arithmetic",
    ),
    CatalogueEntry(
        "stopped-after-the-repair",
        "Giving the corrected step as the final result without carrying it through the remaining steps.",
        area="arithmetic",
    ),
    CatalogueEntry(
        "carried-the-repair-one-step-only",
        "Carrying the corrected value through only the next step and not the rest of the calculation.",
        area="arithmetic",
    ),
    CatalogueEntry(
        "chose-the-other-equilibrium",
        "Giving the other equilibrium of the model when the larger one was asked for.",
        kind="near_miss",
        area="differential-equations",
    ),
    CatalogueEntry(
        "took-the-vertex-as-the-equilibrium",
        "Giving the position of the vertex of the quadratic, halfway between the roots, as an equilibrium.",
        area="differential-equations",
    ),
    CatalogueEntry(
        "sum-of-roots-as-root",
        "Giving the sum of the two roots, p over q, as one of the roots.",
        area="differential-equations",
    ),
    CatalogueEntry(
        "product-of-roots-as-root",
        "Giving the product of the two roots, h over q, as one of the roots.",
        area="differential-equations",
    ),
    CatalogueEntry(
        "derivative-lost-factor-of-power",
        "Differentiating the squared term without its factor of 2, so p - qP is used for f'(P).",
        area="differential-equations",
    ),
    CatalogueEntry(
        "derivative-at-other-equilibrium",
        "Giving f' at the other equilibrium, which has the opposite sign, so a stable point looks unstable.",
        kind="near_miss",
        area="differential-equations",
    ),
    CatalogueEntry(
        "evaluated-function-not-derivative",
        "Evaluating f at the equilibrium (zero) when f' was asked for.",
        area="differential-equations",
    ),
    CatalogueEntry(
        "derivative-dropped-linear-term",
        "Dropping the linear term when differentiating f, so only the derivative of the squared term is used.",
        area="differential-equations",
    ),
    CatalogueEntry(
        "gap-between-equilibria-as-root",
        "Giving the distance between the two equilibria as the smaller equilibrium.",
        area="differential-equations",
    ),
    CatalogueEntry(
        "wrong-inverse-operation",
        "Undoing a product with a product, or a quotient with a quotient, instead of using the inverse operation.",
        area="equations",
    ),
    CatalogueEntry(
        "wrong-operation-family",
        "Adding or subtracting where the equation is a product or quotient, such as reading 8p = 224 as 224 - 8.",
        area="equations",
    ),
    CatalogueEntry(
        "inverted-the-division",
        "Dividing the wrong way round, so the coefficient is divided by the total instead of the total by the coefficient.",
        area="equations",
    ),
    CatalogueEntry(
        "used-the-wrong-earlier-answer",
        "Using the answer to a different earlier part, or a number from the story, in an equation that needed the "
        "answer to the part just before.",
        area="equations",
    ),
    CatalogueEntry(
        "divided-before-subtracting",
        "Dividing the total by the coefficient first and then subtracting the constant, instead of subtracting first.",
        area="equations",
    ),
    CatalogueEntry(
        "ignored-sign",
        "Ignoring the minus signs on terms and adding their sizes when collecting like terms.",
        area="algebra",
    ),
    CatalogueEntry(
        "combined-unlike-terms",
        "Adding the coefficients of terms that are not like terms, such as an x term and a y term.",
        area="algebra",
    ),
    CatalogueEntry(
        "dropped-a-term",
        "Leaving one term out when collecting like terms.",
        area="algebra",
    ),
    CatalogueEntry(
        "swapped-the-variables",
        "Substituting the value of x for y and the value of y for x.",
        area="algebra",
    ),
    CatalogueEntry(
        "coefficient-added-not-multiplied",
        "Reading 5x as 5 + x when substituting, adding the coefficient to the value instead of multiplying.",
        area="algebra",
    ),
    # -- formulas, expressions, tables, logic counts and estimation ----------------------------------------
    CatalogueEntry(
        "forgot-to-multiply-by-time",
        "Giving the amount for one period (one year of interest) where the whole term was asked.",
        area="formulas",
    ),
    CatalogueEntry(
        "percent-used-as-whole-number",
        "Using a percentage rate as a whole number in a formula, without dividing by 100.",
        area="formulas",
    ),
    CatalogueEntry(
        "interest-not-total",
        "Giving the interest earned where the total value (principal plus interest) was asked.",
        area="formulas",
        kind="near_miss",
    ),
    CatalogueEntry(
        "compound-for-simple",
        "Using the compound interest formula where simple interest was stated.",
        area="formulas",
    ),
    CatalogueEntry(
        "treated-total-as-annual",
        "Treating an amount over several periods as if it were the amount for one period.",
        area="formulas",
    ),
    CatalogueEntry(
        "ignored-rate-in-interest",
        "Leaving the rate out of an interest calculation, so principal and interest are compared directly.",
        area="formulas",
    ),
    CatalogueEntry(
        "cents-read-as-dollars",
        "Leaving a price in cents when dollars were asked, by not dividing by 100.",
        area="formulas",
    ),
    CatalogueEntry(
        "gave-energy-not-cost",
        "Giving the energy used where the cost of that energy was asked.",
        area="formulas",
        kind="near_miss",
    ),
    CatalogueEntry(
        "divided-instead-of-multiplied",
        "Dividing where a formula multiplies, such as speed divided by time for distance.",
        area="formulas",
    ),
    CatalogueEntry(
        "ignored-included-allowance",
        "Charging a rate for every unit when a fixed fee already covers the first few.",
        area="expressions",
    ),
    CatalogueEntry(
        "allowance-not-multiplied-by-rate",
        "Subtracting the allowance as a number of units but not as a cost, so the rate is never applied to it.",
        area="expressions",
    ),
    CatalogueEntry(
        "left-out-fixed-fee",
        "Writing only the charge for the extra units and leaving out the fixed fee.",
        area="expressions",
    ),
    CatalogueEntry(
        "fee-multiplied-by-extra-units",
        "Multiplying the fixed fee as well as the rate by the number of extra units.",
        area="expressions",
    ),
    CatalogueEntry(
        "kept-discount-not-balance",
        "Taking the discount itself as the new price instead of what remains after it.",
        area="expressions",
    ),
    CatalogueEntry(
        "discount-on-variable-part-only",
        "Applying a discount to the per-unit charge only, not to the whole bill.",
        area="expressions",
    ),
    CatalogueEntry(
        "ignored-discount",
        "Giving the full price where the discounted price was asked.",
        area="expressions",
        kind="near_miss",
    ),
    CatalogueEntry(
        "change-not-divided-by-step",
        "Giving the change in output between two table entries as the rate, without dividing by the change in input.",
        area="tables",
    ),
    CatalogueEntry(
        "output-over-input-as-rate",
        "Dividing an output by its input to get the rate, which only works when the rule passes through the origin.",
        area="tables",
    ),
    CatalogueEntry(
        "first-output-as-intercept",
        "Taking the first output in the table as the output at input 0.",
        area="tables",
    ),
    CatalogueEntry(
        "sign-slip-in-intercept",
        "Adding the rate times the input where it should be subtracted when finding the starting value, or the reverse.",
        area="tables",
    ),
    CatalogueEntry(
        "assumed-first-input-is-one",
        "Treating the first input in the table as 1 when working back to input 0.",
        area="tables",
    ),
    CatalogueEntry(
        "continued-pattern-by-one",
        "Extending a table by adding 1 to the output for each step of 1 in the input, instead of the rule's rate.",
        area="tables",
    ),
    CatalogueEntry(
        "counted-from-wrong-total",
        "Dividing a count by the wrong total of cases, such as 7 instead of 8 equally likely assignments.",
        area="logic",
    ),
    CatalogueEntry(
        "odds-for-probability",
        "Giving the odds (favourable to unfavourable) where a probability was asked.",
        area="probability",
    ),
    CatalogueEntry(
        "counted-one-formula-only",
        "Counting the rows for one of two formulas where a comparison of both was asked.",
        area="logic",
        kind="near_miss",
    ),
    CatalogueEntry(
        "rounded-down-when-covering",
        "Rounding down the number of whole units needed to cover a length, which leaves it short.",
        area="estimation",
    ),
    CatalogueEntry(
        "rounded-to-nearest-when-covering",
        "Rounding to the nearest whole unit when the units must cover the whole length.",
        area="estimation",
    ),
    CatalogueEntry(
        "rounded-up-when-capacity",
        "Rounding up the number of animals a paddock can carry, which overstocks it.",
        area="estimation",
    ),
    CatalogueEntry(
        "priced-per-metre-not-per-unit",
        "Multiplying a length in metres by the price of a whole unit.",
        area="estimation",
    ),
    CatalogueEntry(
        "paces-not-converted",
        "Using the number of paces as the length without multiplying by the length of a pace.",
        area="estimation",
    ),
    CatalogueEntry(
        "linear-factor-on-area",
        "Multiplying an area by the pace length once, instead of applying it to each side.",
        area="estimation",
    ),
    CatalogueEntry(
        "perimeter-for-area",
        "Giving the distance round the edge where the area was asked.",
        area="estimation",
    ),
    CatalogueEntry(
        "left-count-unrounded",
        "Leaving the number of whole units needed as a decimal instead of rounding it to a whole number.",
        area="estimation",
    ),
    CatalogueEntry(
        "ignored-the-given-condition",
        "Answering a conditional probability as if the condition had not been given, using all the cases.",
        area="probability",
    ),
    CatalogueEntry(
        "reversed-conditional",
        "Dividing by the cases where the event holds instead of the cases where the condition holds, so the conditional is reversed.",
        area="probability",
    ),
    CatalogueEntry(
        "used-opposite-condition-value",
        "Counting the cases with the other value of the given variable than the one the question names.",
        area="logic",
    ),
    CatalogueEntry(
        "counted-both-directions",
        "Counting the cases where two formulas differ in either direction when only one direction was asked.",
        area="logic",
    ),
    CatalogueEntry(
        "counted-wrong-direction",
        "Counting the cases where the second formula is true and the first false when the reverse was asked.",
        area="logic",
    ),
    # -- sc2: wages with loading, network closures, parameter PDEs, similar triangles, enlargements, constructions, grid hikes and parallel-line triangles --
    CatalogueEntry(
        "overtime-at-ordinary-rate",
        "Paying the extra hours at the ordinary rate, ignoring the overtime loading.",
        area="consumer",
    ),
    CatalogueEntry(
        "loading-applied-to-all-hours",
        "Applying the overtime loading (time and a half, double time) to every hour worked, not only the hours after the ordinary block.",
        area="consumer",
    ),
    CatalogueEntry(
        "loading-only-for-extra-hours",
        "Paying the ordinary hours, then only the extra half (or the extra whole) of the rate for the overtime hours, instead of the full loaded rate.",
        area="consumer",
    ),
    CatalogueEntry(
        "omitted-overtime-hours",
        "Leaving the overtime hours out of the total.",
        area="consumer",
    ),
    CatalogueEntry(
        "gave-remainder-not-percentage-amount",
        "Giving what is left after a percentage is taken off where the amount taken off was asked.",
        area="consumer",
    ),
    CatalogueEntry(
        "tax-on-ordinary-pay-only",
        "Finding the percentage of the ordinary pay alone, leaving the loaded pay out of the amount it applies to.",
        area="consumer",
    ),
    CatalogueEntry(
        "averaged-over-ordinary-hours-only",
        "Dividing the whole week's pay by the ordinary hours instead of all the hours worked.",
        area="consumer",
    ),
    CatalogueEntry(
        "averaged-net-not-gross",
        "Averaging the pay after tax where the average of the gross pay was asked.",
        area="consumer",
    ),
    CatalogueEntry(
        "gave-ordinary-rate-as-average",
        "Giving the ordinary hourly rate where the average pay per hour over the whole week was asked.",
        area="consumer",
    ),
    CatalogueEntry(
        "fewest-edges-route",
        "Choosing the route with the fewest roads without comparing the total lengths.",
        area="networks",
    ),
    CatalogueEntry(
        "nearest-neighbour-route",
        "Always taking the shortest road out of the current vertex, which can lead to a longer route overall.",
        area="networks",
    ),
    CatalogueEntry(
        "ignored-closure",
        "Giving the original answer when a road on the route has been closed.",
        area="networks",
    ),
    CatalogueEntry(
        "removed-closed-road-length",
        "Taking the closed road's length off the route total instead of finding a new route.",
        area="networks",
    ),
    CatalogueEntry(
        "gave-new-total-not-difference",
        "Giving the new total where the difference between the new and the old total was asked.",
        area="networks",
    ),
    CatalogueEntry(
        "gave-closed-road-length",
        "Giving the length of the closed road where the extra distance of the detour was asked.",
        area="networks",
    ),
    CatalogueEntry(
        "forgot-square-root",
        "Stopping at k squared, so the square of the answer is given instead of the answer.",
        area="pde",
    ),
    CatalogueEntry(
        "dropped-factor-4-in-discriminant",
        "Using B squared minus AC in place of B squared minus 4AC.",
        area="pde",
    ),
    CatalogueEntry(
        "included-the-parabolic-value",
        "Counting the boundary value as elliptic, so the largest whole number below a boundary is given as the boundary itself.",
        area="pde",
    ),
    CatalogueEntry(
        "gave-boundary-value-not-integer",
        "Giving the boundary value itself where the largest whole number below it was asked.",
        area="pde",
    ),
    CatalogueEntry(
        "reversed-discriminant-sign",
        "Computing 4AC minus B squared, which reverses the sign of the discriminant.",
        area="pde",
    ),
    CatalogueEntry(
        "forgot-parameter-in-discriminant",
        "Using the coefficient of the mixed term without its parameter when substituting into the discriminant.",
        area="pde",
    ),
    CatalogueEntry(
        "forgot-to-square-coefficient",
        "Squaring the parameter but not the number multiplying it in the mixed term's coefficient.",
        area="pde",
    ),
    CatalogueEntry(
        "additive-scaling",
        "Scaling a length by adding the difference of a matching pair instead of multiplying by their ratio.",
        area="similarity",
    ),
    CatalogueEntry(
        "assumed-congruent-copy",
        "Copying the matching side of the first triangle as if the two were congruent, ignoring the scale factor.",
        area="similarity",
    ),
    CatalogueEntry(
        "ratio-inverted",
        "Using the ratio of matching sides upside down, so the length is scaled the wrong way.",
        area="similarity",
    ),
    CatalogueEntry(
        "used-unscaled-side-in-perimeter",
        "Adding a side of the first triangle into the perimeter of the second without scaling it.",
        area="similarity",
    ),
    CatalogueEntry(
        "gave-perimeter-of-original",
        "Giving the perimeter of the original figure where the perimeter of the similar figure was asked.",
        area="similarity",
    ),
    CatalogueEntry(
        "left-out-a-side-of-perimeter",
        "Adding only two sides of a triangle when finding its perimeter.",
        area="similarity",
    ),
    CatalogueEntry(
        "adds-instead-of-multiplies",
        "Adding the scale factor to a measurement instead of multiplying by it.",
        area="geometry",
    ),
    CatalogueEntry(
        "uses-area-ratio-as-scale-factor",
        "Using the ratio of the two areas as the scale factor instead of its square root.",
        area="geometry",
    ),
    CatalogueEntry(
        "halves-the-area-ratio",
        "Halving the ratio of the areas to find the scale factor, as if area scaled with twice the factor.",
        area="geometry",
    ),
    CatalogueEntry(
        "perimeter-scaled-by-k-squared",
        "Multiplying a perimeter by the square of the scale factor.",
        area="geometry",
    ),
    CatalogueEntry(
        "ignored-the-enlargement",
        "Giving the original measurement where the measurement of the image was asked.",
        area="geometry",
    ),
    CatalogueEntry(
        "finds-area-instead-of-perimeter",
        "Giving the area where the perimeter was asked.",
        area="geometry",
    ),
    CatalogueEntry(
        "gives-distance-from-centre-not-gap",
        "Giving the image's distance from the centre of enlargement where the gap between a point and its image was asked.",
        area="geometry",
    ),
    CatalogueEntry(
        "puts-image-on-opposite-side-of-centre",
        "Adding the two distances from the centre of enlargement, as if the image lay on the opposite side of the centre.",
        area="geometry",
    ),
    CatalogueEntry(
        "reports-the-compass-setting",
        "Giving the compass setting itself as the length asked for, not the side of the right triangle it creates.",
        area="geometry",
    ),
    CatalogueEntry(
        "ignores-the-refinement",
        "Answering for the original construction, as if the refined step had not changed.",
        area="geometry",
    ),
    CatalogueEntry(
        "subtracts-instead-of-pythagoras",
        "Subtracting two lengths where Pythagoras' theorem was needed.",
        area="geometry",
    ),
    CatalogueEntry(
        "reports-one-leg",
        "Giving one leg of the right triangle where the hypotenuse or the other leg was asked.",
        area="geometry",
    ),
    CatalogueEntry(
        "reports-the-given-length",
        "Giving a stated length where a length found from it was asked.",
        area="geometry",
    ),
    CatalogueEntry(
        "adds-instead-of-subtracting-segment",
        "Adding a shorter segment to the whole where the remaining part was asked.",
        area="geometry",
    ),
    CatalogueEntry(
        "reports-wrong-triangle-hypotenuse",
        "Giving the hypotenuse of an earlier right triangle for the next one in the figure.",
        area="geometry",
    ),
    CatalogueEntry(
        "adds-the-two-legs",
        "Adding the two perpendicular lengths instead of finding the hypotenuse.",
        area="geometry",
    ),
    CatalogueEntry(
        "uses-the-wrong-leg",
        "Using a segment that is not a leg of the right triangle asked about in Pythagoras' theorem.",
        area="geometry",
    ),
    CatalogueEntry(
        "horizontal-direction-reversed",
        "Moving left for right (or west for east) when changing column.",
        area="geometry",
    ),
    CatalogueEntry(
        "vertical-direction-reversed",
        "Moving the wrong way along the rows, for example treating north as increasing the row number when row 1 is in the north.",
        area="geometry",
    ),
    CatalogueEntry(
        "letter-numbered-from-zero",
        "Numbering the columns from A = 0 instead of A = 1.",
        area="geometry",
    ),
    CatalogueEntry(
        "swapped-column-row",
        "Reporting the row number as the column and the column number as the row.",
        area="geometry",
    ),
    CatalogueEntry(
        "squares-not-converted-to-km",
        "Giving a count of grid squares where a distance in kilometres was asked.",
        area="geometry",
    ),
    CatalogueEntry(
        "gave-distance-walked-not-way-back",
        "Giving the total distance already walked where the shortest way back was asked.",
        area="geometry",
    ),
    CatalogueEntry(
        "took-the-longer-side-only",
        "Counting only the larger of the east-west and north-south differences on a grid route.",
        area="geometry",
    ),
    CatalogueEntry(
        "applies-equal-to-co-interior",
        "Treating a co-interior pair as equal (alternate or corresponding) instead of summing to 180 degrees.",
        area="geometry",
    ),
    CatalogueEntry(
        "applies-supplementary-to-equal-type",
        "Making an alternate or corresponding pair sum to 180 degrees instead of being equal.",
        area="geometry",
    ),
    CatalogueEntry(
        "reports-x-not-angle",
        "Solving for x correctly but reporting x itself instead of the angle it gives.",
        area="geometry",
    ),
    CatalogueEntry(
        "reports-the-given-partner-angle",
        "Giving the other angle of the stated pair where the angle asked about was wanted.",
        area="geometry",
    ),
    CatalogueEntry(
        "ignored-the-given-angle",
        "Leaving a stated angle out of an angle-sum calculation.",
        area="geometry",
    ),
    CatalogueEntry(
        "ignored-the-found-angle",
        "Leaving an angle found earlier out of an angle-sum calculation.",
        area="geometry",
    ),
    CatalogueEntry(
        "forgot-to-subtract-from-180",
        "Adding the known angles and stopping, without taking the total from 180 degrees.",
        area="geometry",
    ),
    # -- exam chains: quadrilaterals, symmetry, 3D coordinates, formulas, notation, coefficients (sc3) ----------
    CatalogueEntry(
        "adds-instead-of-multiplying",
        "Adding the numbers in a term such as 3k to the value of k instead of multiplying them.",
        area="algebra",
    ),
    CatalogueEntry(
        "adds-instead-of-subtracting",
        "Adding two results where their difference is asked.",
        area="general",
    ),
    CatalogueEntry(
        "adds-instead-of-using-pythagoras",
        "Adding the two shorter sides of a right-angled triangle (or the components of a displacement) instead of using Pythagoras' theorem.",
        area="geometry",
    ),
    CatalogueEntry(
        "applies-the-formula-to-the-difference",
        "Finding the change between two results by putting the difference of the inputs into the formula, instead of subtracting the two results.",
        area="functions",
    ),
    CatalogueEntry(
        "confuses-square-of-sum-with-sum-of-squares",
        "Treating (a + b)^2 and a^2 + b^2 as the same, in either direction.",
        area="algebra",
    ),
    CatalogueEntry(
        "copies-the-given-angle",
        "Reporting the angle that was given (or its neighbour) instead of working out the one asked for.",
        area="geometry",
    ),
    CatalogueEntry(
        "counts-only-the-line-through-the-first-dot",
        "Counting one line of symmetry for a figure with a repeating pattern of dots, the line through the first dot only.",
        area="geometry",
    ),
    CatalogueEntry(
        "counts-only-the-new-teams",
        "Giving the number of extra teams where the number of extra games is asked.",
        area="functions",
    ),
    CatalogueEntry(
        "diagonal-not-halved-or-doubled",
        "Giving the whole diagonal where the half (to the crossing point) is asked, or the half where the whole diagonal is asked.",
        area="geometry",
    ),
    CatalogueEntry(
        "drops-the-division-by-two",
        "Leaving out the division by two (or the half) in a formula.",
        area="functions",
    ),
    CatalogueEntry(
        "drops-the-factor-of-two",
        "Dropping the 2 from a result such as 2ab.",
        area="algebra",
    ),
    CatalogueEntry(
        "drops-the-number-from-the-constant-term",
        "Giving only the part of a constant term that depends on the unknown, and leaving out the number on its own.",
        area="algebra",
    ),
    CatalogueEntry(
        "drops-the-sign-of-the-change",
        "Giving the size of a change and losing its sign.",
        area="functions",
    ),
    CatalogueEntry(
        "equates-consecutive-angles",
        "Setting the expressions for consecutive angles of a parallelogram equal, as if they were opposite angles, instead of adding them to 180 degrees.",
        area="geometry",
    ),
    CatalogueEntry(
        "forgets-the-square-root",
        "Stopping at the sum of the squares of the components without taking the square root for the distance.",
        area="geometry",
    ),
    CatalogueEntry(
        "forgets-to-add-two-to-the-sides",
        "Dividing the sum of the interior angles by 180 degrees and giving that as the number of sides, without adding the 2.",
        area="geometry",
    ),
    CatalogueEntry(
        "forgets-to-negate",
        "Solving -k = n as k = n, losing the sign when dividing by a negative coefficient.",
        area="equations",
    ),
    CatalogueEntry(
        "half-turn-used-for-sides",
        "Dividing 180 degrees, not 360 degrees, by the exterior angle to find the number of sides.",
        area="geometry",
    ),
    CatalogueEntry(
        "halves-the-given-angle",
        "Halving the angle at the given vertex where the property being used is a different one.",
        area="geometry",
    ),
    CatalogueEntry(
        "ignores-first-leg",
        "Reporting the second move only, and leaving out the first move of a two-leg journey.",
        area="geometry",
    ),
    CatalogueEntry(
        "ignores-second-leg",
        "Reporting the first move only, and leaving out the second move of a two-leg journey.",
        area="geometry",
    ),
    CatalogueEntry(
        "ignores-the-constant-in-the-term",
        "Quoting only the number in front of the unknown in a term such as 3k, as though k were not there.",
        area="algebra",
    ),
    CatalogueEntry(
        "ignores-the-marked-vertices",
        "Answering for the plain polygon, ignoring that only some vertices are marked.",
        area="geometry",
    ),
    CatalogueEntry(
        "ignores-the-vertical-component",
        "Finding the distance from the horizontal components only, leaving out the change in height.",
        area="geometry",
    ),
    CatalogueEntry(
        "ignores-the-weekly-increase",
        "Treating a plan that grows each week as a fixed amount each week.",
        area="functions",
    ),
    CatalogueEntry(
        "juxtaposition-as-addition",
        "Reading the product 3ab as 3 + a + b.",
        area="algebra",
    ),
    CatalogueEntry(
        "lines-doubled",
        "Doubling the number of lines of symmetry, counting each line once for each end.",
        area="geometry",
    ),
    CatalogueEntry(
        "misreads-direction-word",
        "Taking a move west, south or down as positive (or east, north or up as negative).",
        area="geometry",
    ),
    CatalogueEntry(
        "misreads-the-perimeter",
        "Dividing a perimeter by the wrong amount (or not at all) to find one side.",
        area="geometry",
    ),
    CatalogueEntry(
        "multiplies-instead-of-dividing",
        "Multiplying by the number in front of the unknown instead of dividing by it.",
        area="equations",
    ),
    CatalogueEntry(
        "multiplies-only-the-first-term-in-a-bracket",
        "Applying the factor outside a bracket to the first term only.",
        area="algebra",
    ),
    CatalogueEntry(
        "omits-the-final-halving",
        "Finding the angle the diagonal bisects, but not halving it.",
        area="geometry",
    ),
    CatalogueEntry(
        "only-the-full-turn",
        "Giving 360 degrees as the smallest turn that fits the figure onto itself, counting only the full turn.",
        area="geometry",
    ),
    CatalogueEntry(
        "operate-before-substituting",
        "Applying operations before substitution is complete, such as reading 20t - 5t^2 as (20 - 5)t^2.",
        area="functions",
    ),
    CatalogueEntry(
        "ratio-cancelling-slip",
        "Cancelling the factors of two powers wrongly when asked how many times larger one is than the other.",
        area="algebra",
    ),
    CatalogueEntry(
        "reads-n-minus-one-as-n",
        "Using n where the formula has n - 1.",
        area="functions",
    ),
    CatalogueEntry(
        "reports-position-not-displacement",
        "Giving the coordinate of the end point where the change in that coordinate is asked.",
        area="geometry",
    ),
    CatalogueEntry(
        "reports-the-gap-as-the-count",
        "Giving the gap between marked vertices as the number of marked vertices (or the reverse).",
        area="geometry",
    ),
    CatalogueEntry(
        "reports-the-given-value-as-the-unknown",
        "Giving the stated value of a coefficient as the unknown itself, without solving.",
        area="equations",
    ),
    CatalogueEntry(
        "reports-the-hypotenuse",
        "Giving the hypotenuse where a shorter side of the right-angled triangle is asked.",
        area="geometry",
    ),
    CatalogueEntry(
        "reports-the-neighbouring-angle",
        "Giving the consecutive angle (180 degrees minus the answer) instead of the angle asked.",
        area="geometry",
    ),
    CatalogueEntry(
        "reports-the-unknown-not-the-answer",
        "Solving for the unknown (x or k) and reporting it, instead of substituting back for the quantity asked.",
        area="equations",
    ),
    CatalogueEntry(
        "sides-from-angle-slip",
        "Finding the number of sides of a regular polygon from an angle with an off-by-one slip.",
        area="geometry",
    ),
    CatalogueEntry(
        "sign-error-in-a-bracket",
        "Getting a sign wrong inside a bracket, such as reading n(n - 1) as n(n + 1) or c - k as c + k.",
        area="algebra",
    ),
    CatalogueEntry(
        "squares-each-term-separately",
        "Squaring each term of a bracket on its own, so (k - a)^2 becomes k^2 - a^2.",
        area="algebra",
    ),
    CatalogueEntry(
        "squares-more-than-the-letter",
        "Squaring a whole product or bracket where only the letter is squared, so 3ab^2 is read as (3ab)^2.",
        area="algebra",
    ),
    CatalogueEntry(
        "squares-only-part-of-the-bracket",
        "Squaring only some of what is inside a bracket, so (3ab)^2 is read as 3ab^2.",
        area="algebra",
    ),
    CatalogueEntry(
        "squares-the-coefficient",
        "Squaring the coefficient with the letter, reading 5t^2 as (5t)^2.",
        area="algebra",
    ),
    CatalogueEntry(
        "subtracts-in-the-wrong-order",
        "Subtracting the first result from the second where the second is taken from the first (or the reverse), so the sign is wrong.",
        area="general",
    ),
    CatalogueEntry(
        "subtracts-instead-of-dividing",
        "Subtracting the number in front of the unknown instead of dividing by it.",
        area="equations",
    ),
    CatalogueEntry(
        "subtracts-instead-of-using-pythagoras",
        "Subtracting lengths where a right-angled triangle needs Pythagoras' theorem.",
        area="geometry",
    ),
    CatalogueEntry(
        "uses-360-for-two-angles",
        "Using 360 degrees, the total of all four angles, for two consecutive angles of a parallelogram.",
        area="geometry",
    ),
    CatalogueEntry(
        "uses-a-half-turn-as-the-full-turn",
        "Dividing 180 degrees instead of 360 degrees by the order of rotational symmetry.",
        area="geometry",
    ),
    CatalogueEntry(
        "uses-subtraction-for-how-many-times",
        "Subtracting two results where a 'how many times as large' ratio is asked.",
        area="general",
    ),
    CatalogueEntry(
        "uses-the-other-diagonal",
        "Using the length of the other diagonal instead of the one the question names.",
        area="geometry",
    ),
    CatalogueEntry(
        "uses-the-wrong-triangle-total",
        "Using 90 degrees minus the angle where the diagonal bisects the angle, or another wrong total in the triangle.",
        area="geometry",
    ),
    CatalogueEntry(
        "uses-whole-diagonals",
        "Using the whole diagonal where only the half belongs in the Pythagoras step.",
        area="geometry",
    ),
    CatalogueEntry(
        "wrong-direction",
        "Reporting the negated change, end point minus start point taken the wrong way round.",
        area="geometry",
    ),
    # -- sc6: exam-tier chains (primes, scientific notation, time zones, similarity, circle converses) -----------
    CatalogueEntry(
        "adds-coefficients-when-multiplying",
        "Adding the coefficients of two numbers in scientific notation when they should be multiplied.",
        area="indices",
    ),
    CatalogueEntry(
        "adds-instead-of-intersects",
        "Adding the lengths of two time windows instead of finding the length of the time they share.",
        area="time zones",
    ),
    CatalogueEntry(
        "adds-longitudes-same-side",
        "Adding the two longitudes to find the angle between them when both places are on the same side of the prime meridian.",
        area="time zones",
    ),
    CatalogueEntry(
        "angle-at-centre-halved",
        "Halving the angle at the circumference to get the angle at the centre, instead of doubling it.",
        area="circle geometry",
    ),
    CatalogueEntry(
        "applies-daylight-saving-to-wrong-place",
        "Applying the daylight-saving hour to the place that does not observe it, or leaving it off the place that does.",
        area="time zones",
    ),
    CatalogueEntry(
        "area-scaled-by-linear-factor",
        "Multiplying an area by the linear scale factor instead of its square.",
        area="similarity",
    ),
    CatalogueEntry(
        "cross-multiplies-wrongly",
        "Setting up a proportion with a pair of sides on the wrong sides of the equals sign, so the unknown comes out as a product over the wrong side.",
        area="similarity",
    ),
    CatalogueEntry(
        "decimal-point-misplaced",
        "Writing the product with its decimal point one place too far left or right.",
        area="indices",
    ),
    CatalogueEntry(
        "difference-not-ratio",
        "Finding a scale factor by subtracting a pair of corresponding lengths and comparing the difference, instead of dividing.",
        area="similarity",
    ),
    CatalogueEntry(
        "equates-instead-of-supplementary",
        "Setting opposite angles of a quadrilateral equal, as if they were angles in the same segment, instead of adding them to 180 degrees.",
        area="circle geometry",
    ),
    CatalogueEntry(
        "exponent-not-renormalised",
        "Leaving the power of ten as the sum of the exponents when the coefficients multiply to 10 or more.",
        area="indices",
    ),
    CatalogueEntry(
        "exponent-sign-reversed",
        "Giving the power of ten with the wrong sign, as if a small number had a positive exponent.",
        area="indices",
    ),
    CatalogueEntry(
        "forgets-call-length",
        "Giving the closing time of the shared window as the latest start for a call that has to finish inside it.",
        area="time zones",
    ),
    CatalogueEntry(
        "forgets-to-divide-by-15",
        "Giving the angle between two meridians as the time difference in hours, without dividing by 15.",
        area="time zones",
    ),
    CatalogueEntry(
        "forgets-zone-difference",
        "Giving the arrival time as the departure time plus the flight time, without moving the clock by the time difference.",
        area="time zones",
    ),
    CatalogueEntry(
        "ignores-daylight-saving",
        "Using standard offsets for places that are currently observing daylight saving.",
        area="time zones",
    ),
    CatalogueEntry(
        "ignores-inner-multiplicity",
        "Counting each base once for every outer power, without allowing for how many times the prime divides the base.",
        area="primes",
    ),
    CatalogueEntry(
        "ignores-outer-exponent",
        "Using how many times a prime divides a base but forgetting to multiply by the outer exponent.",
        area="primes",
    ),
    CatalogueEntry(
        "ignores-time-difference",
        "Giving the length of one place's window as the overlap, as if the two places shared a clock.",
        area="time zones",
    ),
    CatalogueEntry(
        "leaves-an-angle-out-of-the-sum",
        "Subtracting only one of the two known angles from 180 degrees to find the third angle of a triangle.",
        area="angles",
    ),
    CatalogueEntry(
        "multiplies-exponents-not-adds",
        "Multiplying the exponents a prime gets from different factors instead of adding them.",
        area="primes",
    ),
    CatalogueEntry(
        "multiplies-exponents-of-ten",
        "Multiplying the exponents of the powers of ten when multiplying numbers in scientific notation, instead of adding them.",
        area="indices",
    ),
    CatalogueEntry(
        "off-by-one-exponent",
        "Giving an exponent one more or one less than the correct one.",
        area="indices",
    ),
    CatalogueEntry(
        "off-by-one-hour",
        "Reporting a time difference or clock time one hour too large or too small.",
        area="time zones",
    ),
    CatalogueEntry(
        "omits-day-adjustment",
        "Reading two clock times as if they were on the same day when the arrival is on a different calendar day.",
        area="time zones",
    ),
    CatalogueEntry(
        "reports-duration-not-offset",
        "Giving the length of the flight as the time difference between the two places.",
        area="time zones",
    ),
    CatalogueEntry(
        "subtracts-daylight-saving",
        "Taking the daylight-saving hour off a place's offset instead of adding it.",
        area="time zones",
    ),
    CatalogueEntry(
        "subtracts-exponents-of-ten",
        "Subtracting the exponents of the powers of ten when multiplying numbers in scientific notation, instead of adding them.",
        area="indices",
    ),
    CatalogueEntry(
        "subtracts-longitudes-opposite-sides",
        "Subtracting the two longitudes to find the angle between them when the places are on opposite sides of the prime meridian.",
        area="time zones",
    ),
    CatalogueEntry(
        "sum-to-360-not-180",
        "Setting opposite angles of a cyclic quadrilateral to add to 360 degrees instead of 180 degrees.",
        area="circle geometry",
    ),
    CatalogueEntry(
        "uses-180-for-semicircle",
        "Setting the angle in a semicircle to 180 degrees instead of 90 degrees.",
        area="circle geometry",
    ),
    CatalogueEntry(
        "uses-wrong-corresponding-side",
        "Using a side that does not correspond to the unknown side when scaling.",
        area="similarity",
    ),
    CatalogueEntry(
        "uses-one-base-only",
        "Taking the exponent of a prime from only the first factor that contains it and ignoring the others.",
        area="primes",
    ),
    CatalogueEntry(
        "wrong-zone-direction",
        "Moving the clock the wrong way by the time difference: adding it where it should be subtracted, or the reverse.",
        area="time zones",
    ),
    CatalogueEntry(
        "used-exact-values-not-rounded",
        "Working with the exact measurements instead of rounding them first as the question asked, so the answer is the exact value rather than the estimate.",
        area="estimation",
    ),
)

#: Every entry, by id.
CATALOGUE: dict[str, CatalogueEntry] = {entry.id: entry for entry in _ENTRIES}

if len(CATALOGUE) != len(_ENTRIES):  # pragma: no cover - caught by the import itself
    raise ValueError("duplicate id in the misconception catalogue")


def describe(misconception_id: str, skill=None) -> tuple[str, str] | None:
    """``(description, kind)`` for an id: the skill's own wording when `skill` lists it, else the catalogue's, else
    None. `skill` is duck-typed (anything with `misconceptions` and `near_misses` lists of objects with `id` and
    `description`)."""
    for entry in getattr(skill, "misconceptions", None) or ():
        if entry.id == misconception_id:
            return entry.description, "misconception"
    for entry in getattr(skill, "near_misses", None) or ():
        if entry.id == misconception_id:
            return entry.description, "near_miss"
    entry = CATALOGUE.get(misconception_id)
    return (entry.description, entry.kind) if entry is not None else None
