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
    # -- mechanics of materials -----------------------------------------------------------------------------------
    CatalogueEntry(
        "unit-slip-kn-as-n",
        "Using a force in kilonewtons as if it were in newtons (or the reverse), so the result is out by 1000.",
        area="mechanics",
    ),
    CatalogueEntry(
        "added-dimensions-for-area",
        "Adding the two sides of a rectangle instead of multiplying them to get the cross-sectional area.",
        area="mechanics",
    ),
    CatalogueEntry(
        "used-one-side-for-area",
        "Using the square of one side, or one side alone, as the cross-sectional area of a rectangle.",
        area="mechanics",
    ),
    CatalogueEntry(
        "gave-strain-not-microstrain",
        "Giving the strain as a decimal when the answer was asked in microstrain (a factor of one million).",
        kind="near_miss",
        area="mechanics",
    ),
    CatalogueEntry(
        "unit-slip-mm-as-m",
        "Dividing an extension in millimetres by a length treated as metres, so the strain is out by 1000.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mpa-gpa-slip",
        "Mixing megapascals and gigapascals (1 GPa = 1000 MPa), so a modulus or an extension is out by 1000.",
        area="mechanics",
    ),
    CatalogueEntry(
        "ignored-length-change",
        "Reusing the first bar's length (or just scaling by the load) when the second bar has a different length.",
        area="mechanics",
    ),
    CatalogueEntry(
        "added-opposing-moments",
        "Adding the moments of two forces that turn the beam in opposite directions instead of subtracting.",
        area="mechanics",
    ),
    CatalogueEntry(
        "ignored-force-angle",
        "Using the whole force in a moment when only the part perpendicular to the lever arm turns the beam.",
        area="mechanics",
    ),
    CatalogueEntry(
        "sign-convention-reversed",
        "Reversing the stated sign convention for turning effects, so the net moment has the wrong sign.",
        area="mechanics",
    ),
    CatalogueEntry(
        "balanced-one-force-only",
        "Balancing only one of the two forces instead of the net moment of both.",
        area="mechanics",
    ),
    # -- linear algebra -------------------------------------------------------------------------------------------
    CatalogueEntry(
        "diagonal-read-as-eigenvalues",
        "Reading the diagonal entries of a matrix as its eigenvalues, which holds only for a triangular matrix.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "ignored-shift-in-eigenvalues",
        "Scaling the eigenvalues of A for sA + tI but forgetting that adding tI shifts each one by t.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "ignored-scaling-in-eigenvalues",
        "Shifting the eigenvalues of A for sA + tI but forgetting that multiplying by s scales each one.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "projected-through-origin",
        "Projecting onto the direction of a line as if it passed through the origin, ignoring where the line sits.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "projected-onto-normal",
        "Projecting onto the normal to the line instead of onto its direction.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "forgot-to-add-back-point",
        "Giving the projection vector as the closest point on a line, without adding on the point the line passes "
        "through.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "forgot-to-divide-by-length-squared",
        "Quoting the dot product alone as the multiple of the direction vector, without dividing by its squared length.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "assumed-rows-sum",
        "Assuming one row of a singular matrix is the plain sum of the others, whatever multiples the rows really use.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "dropped-one-multiple",
        "Writing a dependent row using only one of the two multiples that make it up.",
        area="linear-algebra",
    ),
    # -- fourier series -------------------------------------------------------------------------------------------
    CatalogueEntry(
        "sign-lost-at-n-pi",
        "Losing the alternating sign (-1)^n that comes from evaluating sine or cosine at n times pi.",
        area="fourier",
    ),
    CatalogueEntry(
        "integration-by-parts-incomplete",
        "Stopping after the first integration by parts of x^2 times a sine or cosine, leaving the second step out.",
        area="fourier",
    ),
    CatalogueEntry(
        "missing-normalisation-factor",
        "Leaving out the factor 2/pi in the Fourier coefficient and quoting the bare integral.",
        area="fourier",
    ),
    CatalogueEntry(
        "sign-error-in-by-parts",
        "Losing the minus sign when integrating sin(nx), so the whole by-parts result comes out with the wrong sign.",
        area="fourier",
    ),
    # -- beams and loads ------------------------------------------------------------------------------------------
    CatalogueEntry(
        "left-out-point-load",
        "Leaving a point load out of a reaction or moment, so only the distributed load is counted.",
        area="structural",
    ),
    CatalogueEntry(
        "udl-lever-arm-full-length",
        "Taking the moment of a uniformly distributed load as if its resultant acted at the far end of the span, "
        "instead of at its midpoint.",
        area="structural",
    ),
    CatalogueEntry(
        "cantilever-treated-as-simply-supported",
        "Using a simply supported beam's result (half the load, or wL^2/8) for a cantilever.",
        area="structural",
    ),
    CatalogueEntry(
        "section-moment-is-support-moment",
        "Giving the moment at the fixed support as the moment at a section part-way along the beam.",
        area="structural",
    ),
    CatalogueEntry(
        "section-moment-ignores-udl",
        "Finding the moment at a section from the support reaction and moment alone, without the part of the "
        "distributed load between the support and the section.",
        area="structural",
    ),
    CatalogueEntry(
        "wrong-udl-moment-formula",
        "Using the wrong coefficient in the bending-moment formula for a uniformly distributed load: wL^2/2 or "
        "wL^2/12 where wL^2/8 belongs, or the reverse.",
        area="structural",
    ),
    CatalogueEntry(
        "forgot-to-square-span",
        "Leaving the span unsquared in a moment formula: wL/8 where wL^2/8 belongs.",
        area="structural",
    ),
    CatalogueEntry(
        "used-total-load-as-shear",
        "Giving the total load as the maximum shear force, instead of the share each support carries.",
        area="structural",
    ),
    CatalogueEntry(
        "confused-shear-with-moment",
        "Using the bending-moment formula where the shear force was asked for, or the reverse.",
        area="structural",
    ),
    CatalogueEntry(
        "gave-maximum-for-section",
        "Giving the maximum bending moment where the moment at a stated section was asked for.",
        area="structural",
    ),
    CatalogueEntry(
        "section-moment-without-reaction",
        "Finding the moment at a section from the distributed load alone, without the support reaction.",
        area="structural",
    ),
    CatalogueEntry(
        "swapped-support-reactions",
        "Attaching each reaction to the wrong support: the moment about one end gives the reaction at the other.",
        area="structural",
    ),
    CatalogueEntry(
        "equal-reactions-assumed",
        "Sharing the total load equally between the supports though the loads are not placed symmetrically.",
        area="structural",
    ),
    CatalogueEntry(
        "picked-smaller-reaction",
        "Giving the smaller support reaction as the maximum shear force.",
        area="structural",
    ),
    # -- right-triangle trigonometry ------------------------------------------------------------------------------
    CatalogueEntry(
        "swaps-opposite-and-adjacent",
        "Mixing up which leg is opposite and which is adjacent to the angle, so the wrong ratio is used (sine "
        "where cosine belongs, or the reverse).",
        area="trigonometry",
    ),
    CatalogueEntry(
        "multiplies-instead-of-divides",
        "Choosing the right ratio but applying it the wrong way round, multiplying where the unknown side is the "
        "denominator, or dividing where it is the numerator.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "adds-where-subtracts-needed",
        "Adding two lengths measured along the same line from one point, where the question needs their "
        "difference, or the reverse.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "used-one-ramp-only",
        "Answering for one of the two triangles in a two-triangle problem instead of combining both.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "swapped-unknowns",
        "Giving the value of the other unknown: the price of the second item where the first was asked.",
        area="equations",
    ),
    CatalogueEntry(
        "elimination-wrong-operation",
        "Adding the two equations where they should be subtracted (or the reverse), so the unknown does not cancel.",
        area="equations",
    ),
    CatalogueEntry(
        "elimination-without-matching-coefficients",
        "Subtracting the two equations as they stand, without first making one coefficient match.",
        area="equations",
    ),
    CatalogueEntry(
        "ignored-other-unknown",
        "Solving one equation as if the other unknown were not there.",
        area="equations",
    ),
    CatalogueEntry(
        "dropped-coefficient-in-substitution",
        "Substituting a found value into an equation without multiplying it by that unknown's coefficient.",
        area="equations",
    ),
    CatalogueEntry(
        "unit-not-converted",
        "Using a length in centimetres in a formula that needs metres, or the reverse.",
        area="units",
    ),
    CatalogueEntry(
        "multiplied-instead-of-divided",
        "Multiplying two quantities where the relationship needs one divided by the other.",
        area="general",
    ),
    CatalogueEntry(
        "frequency-changes-at-boundary",
        "Keeping the wavelength fixed when a wave enters a new medium, though it is the frequency that stays fixed.",
        area="waves",
    ),
    CatalogueEntry(
        "gradient-term-omitted",
        "Leaving out the contribution of the cross term when taking a partial derivative.",
        area="multivariable",
    ),
    CatalogueEntry(
        "direction-not-normalised",
        "Taking the dot product with a direction vector that has not been divided by its length.",
        area="multivariable",
    ),
    CatalogueEntry(
        "direction-reversed",
        "Using the direction from the second point to the first, which changes the sign of the rate.",
        area="multivariable",
    ),
    CatalogueEntry(
        "gradient-components-swapped",
        "Pairing the x component of the gradient with the y component of the direction, and the reverse.",
        area="multivariable",
    ),
    CatalogueEntry(
        "gradient-length-for-directional",
        "Giving the length of the gradient where the rate in a stated direction was asked for.",
        area="multivariable",
    ),
    CatalogueEntry(
        "lower-limit-forgotten",
        "Evaluating an antiderivative only at the upper limit, so the lower limit is never subtracted.",
        area="integration",
    ),
    CatalogueEntry(
        "corner-value-times-area",
        "Treating a double integral as the value of the integrand at one corner times the area of the region.",
        area="integration",
    ),
    CatalogueEntry(
        "other-variable-width-dropped",
        "Integrating a term in one variable only, and never multiplying by the width of the other variable's interval.",
        area="integration",
    ),
    CatalogueEntry(
        "sum-instead-of-product",
        "Adding the two one-variable integrals of a product term instead of multiplying them.",
        area="integration",
    ),
    CatalogueEntry(
        "average-not-divided-by-area",
        "Giving the integral over a region where its average value over the region was asked for.",
        kind="near_miss",
        area="integration",
    ),
    CatalogueEntry(
        "average-divided-by-wrong-size",
        "Dividing an integral by the sum of the side lengths instead of the area of the rectangle.",
        area="integration",
    ),
    CatalogueEntry(
        "gcd-stopped-early",
        "Stopping the Euclidean algorithm at a remainder that is not the last non-zero one.",
        area="number-theory",
    ),
    CatalogueEntry(
        "common-divisor-not-greatest",
        "Giving a common divisor of both numbers that is not the greatest one.",
        area="number-theory",
    ),
    CatalogueEntry(
        "lcm-for-gcd",
        "Giving the lowest common multiple where the greatest common divisor was asked for.",
        area="number-theory",
    ),
    CatalogueEntry(
        "side-ratio-not-squared",
        "Dividing the area of a floor by the side of a square tile instead of by the tile's area.",
        area="number-theory",
    ),
    CatalogueEntry(
        "sides-counted-not-area",
        "Adding the number of tiles along each side where the total number of tiles was asked for.",
        area="number-theory",
    ),
    CatalogueEntry(
        "exponent-not-reduced-by-cycle",
        "Reducing the exponent by the modulus instead of by the length of the cycle of powers.",
        area="number-theory",
    ),
    CatalogueEntry(
        "cycle-length-off-by-one",
        "Reporting the exponent just before or just after the one where the powers return to 1 as the cycle length.",
        area="number-theory",
    ),
    CatalogueEntry(
        "cycle-length-is-modulus",
        "Taking the modulus as the length of the cycle of powers.",
        area="number-theory",
    ),
    CatalogueEntry(
        "additive-inverse-for-multiplicative",
        "Giving the number that adds to the input to make the modulus, where the multiplicative inverse was asked for.",
        area="number-theory",
    ),
    CatalogueEntry(
        "inverse-not-applied-to-shifted",
        "Multiplying by the inverse without first moving the constant term to the other side.",
        area="number-theory",
    ),
    CatalogueEntry(
        "inverse-of-wrong-modulus",
        "Finding the inverse of a number modulo the other modulus of the pair.",
        area="number-theory",
    ),
    CatalogueEntry(
        "solved-one-congruence-only",
        "Giving a number that satisfies only one of the two congruences.",
        area="number-theory",
    ),
    CatalogueEntry(
        "added-remainders",
        "Adding the two remainders instead of solving the pair of congruences.",
        area="number-theory",
    ),
    CatalogueEntry(
        "difference-of-remainders-ignored",
        "Using the second remainder alone, not its difference from the first, when finding the multiplier.",
        area="number-theory",
    ),
    # -- probability and statistics -----------------------------------------------------------------------------------
    CatalogueEntry(
        "gave-complement-probability",
        "Giving the probability of the opposite event: the chance of not winning where the chance of winning was asked, "
        "or the reverse.",
        area="probability",
    ),
    CatalogueEntry(
        "favourable-over-unfavourable",
        "Dividing the favourable outcomes by the unfavourable ones (odds) instead of by all outcomes.",
        area="probability",
    ),
    CatalogueEntry(
        "left-out-a-category-from-total",
        "Leaving one category out of the total when finding the number of equally likely outcomes.",
        area="probability",
    ),
    CatalogueEntry(
        "probability-not-scaled-by-trials",
        "Giving the probability itself where an expected number of trials was asked, without multiplying by the "
        "number of trials.",
        area="probability",
    ),
    CatalogueEntry(
        "divides-trials-by-probability",
        "Dividing the number of trials by the probability when the expected frequency is the product of the two.",
        area="probability",
    ),
    CatalogueEntry(
        "multiplies-by-probability-to-invert",
        "Multiplying by the probability when working back from an expected number to the number of trials, where "
        "division is needed.",
        area="probability",
    ),
    CatalogueEntry(
        "forgot-to-update-total",
        "Changing the number of favourable outcomes but leaving the total as it was before items were added.",
        area="probability",
    ),
    CatalogueEntry(
        "used-total-as-cell",
        "Reading a row or column total off a two-way table where a single cell was asked for.",
        area="statistics",
    ),
    CatalogueEntry(
        "subtracted-from-wrong-total",
        "Subtracting the known cell from the wrong row or column total when finding a missing cell of a two-way table.",
        area="statistics",
    ),
    CatalogueEntry(
        "joint-for-conditional",
        "Dividing by the grand total instead of by the size of the group the condition names, so a joint probability "
        "is given for a conditional one.",
        area="statistics",
    ),
    CatalogueEntry(
        "wrong-condition-group",
        "Dividing by the size of the wrong group when finding a conditional probability.",
        area="statistics",
    ),
    CatalogueEntry(
        "uses-with-replacement-model",
        "Using independent draws (each probability the same) where the items are not replaced, so the counts change "
        "after every draw.",
        area="probability",
    ),
    CatalogueEntry(
        "omits-combination-factor",
        "Counting only one ordering of the successes and failures instead of every way they can occur.",
        area="probability",
    ),
    CatalogueEntry(
        "ignored-acceptance-rule",
        "Adding up the wrong set of outcomes for a stated decision rule, such as counting 'any defective' when the "
        "rule allows one.",
        area="probability",
    ),
    CatalogueEntry(
        "counts-failures-for-success-rate",
        "Using the proportion of failures in place of the proportion of successes.",
        area="statistics",
    ),
    CatalogueEntry(
        "off-by-one-denominator",
        "Dividing by one more or one fewer than the number of observations.",
        area="statistics",
    ),
    CatalogueEntry(
        "estimate-not-squared",
        "Giving the probability of one success where two successes in a row were asked, without multiplying.",
        area="statistics",
    ),
    CatalogueEntry(
        "odds-given-as-probability",
        "Giving the probability of success where the odds (success to failure) were asked.",
        area="statistics",
    ),
    CatalogueEntry(
        "odds-inverted",
        "Writing the odds as failures over successes instead of successes over failures.",
        area="statistics",
    ),
    CatalogueEntry(
        "slope-from-end-points",
        "Taking the slope of the line through the first and last data points as the least-squares slope.",
        area="statistics",
    ),
    CatalogueEntry(
        "swaps-slope-intercept",
        "Reporting the intercept as the slope, or the slope as the intercept.",
        area="statistics",
    ),
    CatalogueEntry(
        "intercept-is-mean-of-y",
        "Giving the mean of the y-values as the intercept of the fitted line.",
        area="statistics",
    ),
    CatalogueEntry(
        "dropped-intercept-in-prediction",
        "Predicting from the slope times x alone, leaving out the intercept.",
        area="statistics",
    ),
    CatalogueEntry(
        "residual-sign-reversed",
        "Subtracting the observed value from the predicted value, giving a residual with the wrong sign.",
        area="statistics",
    ),
    # -- optimisation, partial derivatives and linear approximation ------------------------------------------------
    CatalogueEntry(
        "gradient-step-uphill",
        "Adding the learning rate times the gradient to the parameter instead of subtracting it, so the step moves "
        "uphill.",
        area="optimisation",
    ),
    CatalogueEntry(
        "reports-gradient-not-update",
        "Giving the gradient as the new parameter value, without taking the step.",
        area="optimisation",
    ),
    CatalogueEntry(
        "forgot-learning-rate",
        "Subtracting the whole gradient from the parameter without scaling it by the learning rate.",
        area="optimisation",
    ),
    CatalogueEntry(
        "stale-gradient",
        "Taking a second step with the first step's gradient instead of recomputing the gradient at the new value.",
        area="optimisation",
    ),
    CatalogueEntry(
        "dropped-factor-two-in-derivative",
        "Differentiating a squared term without the factor of 2, so the derivative of a x^2 is taken as a x.",
        area="calculus",
    ),
    CatalogueEntry(
        "evaluates-function-not-derivative",
        "Reporting the value of the function at the point instead of its derivative.",
        area="calculus",
    ),
    CatalogueEntry(
        "dropped-weight-in-gradient",
        "Differentiating a weighted square k y^2 as 2y, leaving the weight k out of the gradient.",
        area="optimisation",
    ),
    CatalogueEntry(
        "dropped-product-rule-term",
        "Differentiating only one factor of a product and treating the other factors as constants.",
        area="calculus",
    ),
    CatalogueEntry(
        "differentiated-wrong-variable",
        "Finding the partial derivative with respect to the other variable.",
        area="calculus",
    ),
    CatalogueEntry(
        "summed-both-partials",
        "Adding the two partial derivatives instead of giving only the one asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "forgot-inner-derivative",
        "Differentiating the outer power of a bracket but leaving out the derivative of what is inside the bracket.",
        area="calculus",
    ),
    CatalogueEntry(
        "divided-by-step",
        "Dividing the rate of change by the step size instead of multiplying by it.",
        area="calculus",
    ),
    CatalogueEntry(
        "added-step-not-multiplied",
        "Adding the step size to the rate of change instead of multiplying them.",
        area="calculus",
    ),
    CatalogueEntry(
        "gave-x-intercept-for-y-intercept",
        "Finding where the tangent meets the x-axis when the question asked where it meets the y-axis.",
        kind="near_miss",
        area="calculus",
    ),
    # -- data and chance --------------------------------------------------------------------------------------------
    CatalogueEntry(
        "ratio-instead-of-probability",
        "Writing favourable : unfavourable as the probability instead of favourable out of all outcomes.",
        area="probability",
    ),
    CatalogueEntry(
        "divides-trials-only",
        "Dividing the number of trials by the probability's denominator but ignoring its numerator.",
        area="probability",
    ),
    CatalogueEntry(
        "gave-complement",
        "Answering for the complement of the event asked about, the outcomes that are not in it.",
        kind="near_miss",
        area="probability",
    ),
    CatalogueEntry(
        "reports-maximum",
        "Giving the greatest value as the range instead of the greatest minus the least.",
        area="statistics",
    ),
    CatalogueEntry(
        "range-first-last",
        "Subtracting the first listed value from the last listed value instead of the least from the greatest.",
        area="statistics",
    ),
    CatalogueEntry(
        "order-reported-as-gcd",
        "Reporting gcd(a, n) as the order of [a] in Z/nZ, instead of n divided by gcd(a, n).",
        area="abstract-algebra",
    ),
    CatalogueEntry(
        "uses-group-size-always",
        "Reporting the size of the whole group as the order of every element, ignoring that different elements have different orders.",
        area="abstract-algebra",
    ),
    CatalogueEntry(
        "confuses-element-with-order",
        "Reporting the element itself as its order, rather than the number of additions needed to return to zero.",
        area="abstract-algebra",
    ),
    CatalogueEntry(
        "index-reported-as-subgroup-order",
        "Giving the size of the subgroup where the number of cosets was asked for.",
        area="abstract-algebra",
    ),
    CatalogueEntry(
        "every-element-of-subgroup-generates",
        "Counting every element of a cyclic subgroup as a generator of it.",
        area="abstract-algebra",
    ),
    CatalogueEntry(
        "all-but-identity-generate",
        "Counting every element of a cyclic subgroup except the identity as a generator; this is only right when the order is prime.",
        area="abstract-algebra",
    ),
    CatalogueEntry(
        "reverses-composition-order",
        "Applying the first-named permutation first when the composite is defined to apply the second one first.",
        area="abstract-algebra",
    ),
    CatalogueEntry(
        "copies-one-permutation",
        "Reporting one of the two given permutations instead of composing them.",
        area="abstract-algebra",
    ),
    CatalogueEntry(
        "order-as-sum-of-cycle-lengths",
        "Taking the order of a permutation to be the sum of its cycle lengths instead of their least common multiple.",
        area="abstract-algebra",
    ),
    CatalogueEntry(
        "order-as-longest-cycle",
        "Taking the order of a permutation to be its longest cycle length instead of the least common multiple of all of them.",
        area="abstract-algebra",
    ),
    CatalogueEntry(
        "order-as-number-of-cycles",
        "Taking the order of a permutation to be the number of cycles in it.",
        area="abstract-algebra",
    ),
    CatalogueEntry(
        "exponent-reduced-by-wrong-modulus",
        "Reducing the exponent of a permutation power modulo the number of points instead of modulo the order of the permutation.",
        area="abstract-algebra",
    ),
    CatalogueEntry(
        "forgot-weight-is-mass-times-g",
        "Using the mass in place of the weight when balancing a hanging mass against a spring, so g is left out.",
        area="mechanics",
    ),
    CatalogueEntry(
        "unit-slip-cm-for-m",
        "Using an extension in centimetres as if it were in metres.",
        area="mechanics",
    ),
    CatalogueEntry(
        "omits-square-root",
        "Reporting k/m as the angular frequency, without taking the square root.",
        area="mechanics",
    ),
    CatalogueEntry(
        "uses-original-mass-in-formula",
        "Using the mass from the first part of the problem in the formula after the mass has been changed.",
        area="mechanics",
    ),
    CatalogueEntry(
        "period-multiplied-by-omega",
        "Finding the period as 2 pi times the angular frequency instead of 2 pi divided by it.",
        area="mechanics",
    ),
    CatalogueEntry(
        "period-as-frequency",
        "Dividing the angular frequency by 2 pi, which gives the frequency, and reporting it as the period.",
        area="mechanics",
    ),
    CatalogueEntry(
        "period-omits-two-pi",
        "Finding the period as 1 divided by the angular frequency, leaving out the factor 2 pi.",
        area="mechanics",
    ),
    CatalogueEntry(
        "overlap-counted-twice",
        "Subtracting two overlapping group sizes from the whole without adding back the overlap, so the overlap is "
        "removed twice.",
        area="statistics",
    ),
    CatalogueEntry(
        "forgot-constant",
        "Leaving out the constant of integration, or not finding its value from the given point.",
        area="calculus",
    ),
    CatalogueEntry(
        "raised-power-ode",
        "Raising each power by one when integrating without dividing by the new power.",
        area="calculus",
    ),
    CatalogueEntry(
        "initial-value-taken-as-constant",
        "Using the y-value of the given point as the constant of integration, without allowing for the antiderivative's value there.",
        area="calculus",
    ),
    CatalogueEntry(
        "constant-sign-reversed",
        "Finding the constant of integration as the antiderivative's value minus the given y-value instead of the other way round.",
        area="calculus",
    ),
    CatalogueEntry(
        "gradient-read-as-value",
        "Substituting into the gradient expression where the value of y was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "wrong-stationary-point",
        "Evaluating the curve at the wrong stationary point, giving the minimum where the maximum was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "x-coordinate-given-for-y",
        "Giving the x-coordinate of a point where its y-coordinate was asked for.",
        area="coordinate-geometry",
    ),
    CatalogueEntry(
        "union-intersection-confusion",
        "Keeping only the common elements for a union, or all elements for an intersection.",
        area="sets",
    ),
    CatalogueEntry(
        "reversed-difference",
        "Computing B minus A when the question asks for A minus B.",
        area="sets",
    ),
    CatalogueEntry(
        "complement-left-out",
        "Leaving out the complement in a set expression, so the set is used where its complement was meant.",
        area="sets",
    ),
    CatalogueEntry(
        "complement-of-union-as-union-of-complements",
        "Taking the complement of a union to be the union of the complements, instead of the intersection of the complements.",
        area="sets",
    ),
    CatalogueEntry(
        "symmetric-difference-as-union",
        "Reading the symmetric difference of two sets as their union, so the common elements are kept.",
        area="sets",
    ),
    CatalogueEntry(
        "symmetric-difference-as-intersection",
        "Reading the symmetric difference of two sets as their intersection.",
        area="sets",
    ),
    CatalogueEntry(
        "complement-given-for-set",
        "Listing the elements outside the set where the elements of the set were asked for.",
        area="sets",
    ),
    CatalogueEntry(
        "subsets-counted-as-doubling",
        "Doubling the number of elements where the number of subsets is a power of two.",
        area="sets",
    ),
    CatalogueEntry(
        "subsets-counted-as-square",
        "Squaring the number of elements where the number of subsets is a power of two.",
        area="sets",
    ),
    CatalogueEntry(
        "nonempty-subsets-only",
        "Leaving out the empty set when counting subsets.",
        area="sets",
    ),
    CatalogueEntry(
        "ordered-pairs-for-subsets",
        "Counting ordered pairs where unordered pairs (two-element subsets) were asked for.",
        area="sets",
    ),
    CatalogueEntry(
        "improper-subsets-included",
        "Including the empty set and the whole set when counting proper non-empty subsets.",
        area="sets",
    ),
    CatalogueEntry(
        "ignores-neither",
        "Substituting the total number of people for the number in at least one group, without first subtracting those in neither.",
        area="counting",
    ),
    CatalogueEntry(
        "reports-union-for-overlap",
        "Giving the number in at least one group where the number in both was asked for.",
        area="counting",
    ),
    CatalogueEntry(
        "forgets-overlap",
        "Adding the two group sizes directly, double-counting the people in both.",
        area="counting",
    ),
    CatalogueEntry(
        "at-least-one-for-exactly-one",
        "Giving the number (or probability) of at least one group where exactly one was asked for, so the overlap is counted once.",
        area="counting",
    ),
    CatalogueEntry(
        "only-one-set-counted",
        "Counting the members of only one of the two groups where exactly one of the groups was asked for.",
        area="counting",
    ),
    CatalogueEntry(
        "wrong-sample-space",
        "Dividing by the number in at least one group instead of the whole population when finding a probability.",
        area="probability",
    ),
    CatalogueEntry(
        "moment-arm-from-end",
        "Measuring a weight's moment arm from the end of the beam instead of from the pivot.",
        area="mechanics",
    ),
    CatalogueEntry(
        "moment-arm-pivot-to-end",
        "Using the distance from the pivot to the end of the beam as the moment arm of the beam's own weight, instead of the distance from the pivot to its centre.",
        area="mechanics",
    ),
    CatalogueEntry(
        "moment-arm-full-length",
        "Using the whole length of the beam as the moment arm of its own weight.",
        area="mechanics",
    ),
    CatalogueEntry(
        "ignored-beam-weight",
        "Leaving the beam's own weight out of the moment equation.",
        area="mechanics",
    ),
    CatalogueEntry(
        "force-arm-taken-as-half-length",
        "Using half the beam's length as the arm of the balancing force instead of its distance from the pivot.",
        area="mechanics",
    ),
    CatalogueEntry(
        "flipped-distance-ratio",
        "Flipping the ratio of the distances when solving for a force from the moment equation.",
        area="mechanics",
    ),
    CatalogueEntry(
        "assumed-equal-force",
        "Assuming the same force balances the beam whatever the distances.",
        area="mechanics",
    ),
    CatalogueEntry(
        "beam-weight-at-end",
        "Placing the beam's own weight at its end instead of at its centre.",
        area="mechanics",
    ),
    CatalogueEntry(
        "reaction-omits-balancing-force",
        "Leaving the balancing force out of the sum of vertical forces when finding a reaction.",
        area="mechanics",
    ),
    CatalogueEntry(
        "reaction-omits-beam-weight",
        "Leaving the beam's own weight out of the sum of vertical forces when finding a reaction.",
        area="mechanics",
    ),
    CatalogueEntry(
        "reaction-omits-load",
        "Leaving the load out of the sum of vertical forces when finding a reaction.",
        area="mechanics",
    ),
    CatalogueEntry(
        "length-used-as-extension",
        "Using a spring's whole length where its extension (length minus natural length) belongs in F = kx.",
        area="mechanics",
    ),
    CatalogueEntry(
        "natural-length-sign-reversed",
        "Adding the extension to the stretched length instead of subtracting it when finding the natural length.",
        area="mechanics",
    ),
    CatalogueEntry(
        "extension-given-for-natural-length",
        "Giving the extension under load where the natural length was asked for.",
        area="mechanics",
    ),
    CatalogueEntry(
        "natural-length-taken-as-measured-length",
        "Taking a stretched length as the natural length.",
        area="mechanics",
    ),
    CatalogueEntry(
        "force-and-length-from-different-measurements",
        "Pairing a force from one measurement with the length from the other when finding the natural length.",
        area="mechanics",
    ),
    CatalogueEntry(
        "extension-from-wrong-length",
        "Finding an extension from a measured stretched length instead of from the natural length.",
        area="mechanics",
    ),
    CatalogueEntry(
        "force-proportional-to-length",
        "Treating the force as proportional to the spring's whole length rather than to its extension.",
        area="mechanics",
    ),
    CatalogueEntry(
        "uses-first-point-intercept",
        "Giving the first observed y-value as the intercept of the fitted line, though that point is not at x = 0.",
        area="statistics",
    ),
    CatalogueEntry(
        "gave-new-total-not-increase",
        "Giving the number of items there are after some are added where the number added was asked.",
        kind="near_miss",
        area="probability",
    ),
    CatalogueEntry(
        "reaction-not-divided-by-span",
        "Taking moments about one support but forgetting to divide the total moment by the span, so a moment is given "
        "where a reaction was asked.",
        area="structural",
    ),
    CatalogueEntry(
        "left-out-udl-load",
        "Leaving the uniformly distributed load out of a reaction or moment, so only the point load is counted.",
        area="structural",
    ),
    # -- numerical methods and optimisation ----------------------------------------------------------------------
    CatalogueEntry(
        "newton-reapplied-from-start",
        "Applying the Newton step again from the starting value instead of from the latest estimate.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "newton-halving-forgotten",
        "Writing the Newton step for x² = N as x + N/x, leaving out the division by 2.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "newton-sign-of-correction",
        "Adding the Newton correction f(x)/f'(x) instead of subtracting it.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "residual-of-estimate-not-value",
        "Reporting the estimate itself where its residual f(x) was asked for.",
        kind="near_miss",
        area="numerical methods",
    ),
    CatalogueEntry(
        "euler-omits-step-size",
        "Adding the slope to the old value without multiplying by the step size h.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "euler-sign-of-increment",
        "Subtracting the Euler increment h times the slope instead of adding it.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "euler-reuses-first-slope",
        "Taking the second Euler step with the slope from the starting point instead of the slope at the new estimate.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "equilibrium-inverted",
        "Finding the steady state of y' = r - ky as k/r instead of r/k.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "gap-sign-reversed",
        "Subtracting in the wrong order, so a distance above a target comes out as one below it, or the reverse.",
        area="general",
    ),
    CatalogueEntry(
        "trapezoid-full-endpoints",
        "Giving both end values full weight in the trapezoidal rule instead of half weight.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "trapezoid-wrong-step",
        "Using the whole interval width as the step instead of the width divided by the number of strips.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "trapezoid-left-sum",
        "Adding every value except the last and calling it the trapezoidal rule (a left rectangle sum).",
        area="numerical methods",
    ),
    CatalogueEntry(
        "antiderivative-no-division",
        "Raising the power in an antiderivative but not dividing by the new power.",
        area="integration",
    ),
    CatalogueEntry(
        "lower-limit-ignored",
        "Evaluating an antiderivative at the upper limit only, forgetting to subtract its value at the lower limit.",
        area="integration",
    ),
    CatalogueEntry(
        "condition-number-inverted",
        "Forming the condition number as smallest over largest instead of largest over smallest.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "condition-number-signed",
        "Taking the largest and smallest entries with their signs, instead of by absolute value.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "condition-number-sum",
        "Adding the diagonal entries instead of forming the ratio of the largest to the smallest.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "error-bound-divided",
        "Dividing the relative input error by the condition number where the bound multiplies it.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "error-allowance-multiplied",
        "Multiplying the output tolerance by the condition number where the allowed input error divides it.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "jacobi-equations-not-reordered",
        "Solving the equations for x and y in the order listed, though the listed order puts the large coefficient on the other unknown.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "jacobi-uses-new-value",
        "Using the freshly computed x in the same sweep when finding y (a Gauss-Seidel step) instead of the old x.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "jacobi-ignores-off-diagonal",
        "Dividing the right-hand side by the diagonal coefficient and leaving out the off-diagonal term.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "difference-quotient-wrong-width",
        "Dividing the centred difference by h instead of 2h.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "difference-forward-not-centred",
        "Using f(x+h) - f(x) over h where the centred difference was asked for.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "difference-exact-not-estimate",
        "Reporting the exact derivative where the finite-difference estimate was asked for.",
        kind="near_miss",
        area="numerical methods",
    ),
    CatalogueEntry(
        "richardson-wrong-weights",
        "Combining the two estimates with the first-order weights 2 and -1 instead of the second-order weights 4/3 and -1/3.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "richardson-plain-average",
        "Averaging the two estimates, which does not remove the h squared error term.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "bisection-unsquared-midpoint",
        "Giving the power of the midpoint as the function value, leaving out the subtraction of the constant.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "bisection-averages-endpoint-values",
        "Averaging the function values at the endpoints instead of evaluating the function at the midpoint.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "keeps-wrong-half", "Keeping the half of the bracket that does not contain the root.", area="numerical methods"
    ),
    CatalogueEntry(
        "bisection-one-step-only",
        "Stopping after one bisection step when two were asked for.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "bisection-steps-ignore-width",
        "Counting the halvings needed from a width of 1, ignoring how wide the starting bracket is.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "quadratic-factor-two-missing",
        "Using -b/a instead of -b/(2a) for the vertex of a quadratic.",
        area="optimisation",
    ),
    CatalogueEntry(
        "vertex-outside-interval-ignored",
        "Taking the vertex of the parabola as the minimiser though it lies outside the allowed interval.",
        area="optimisation",
    ),
    CatalogueEntry(
        "wrong-end-of-interval",
        "Taking the far end of the interval, not the end nearest the vertex, as the minimiser of a parabola.",
        area="optimisation",
    ),
    CatalogueEntry(
        "evaluates-at-vertex-not-minimiser",
        "Evaluating the function at the unconstrained vertex when the vertex is outside the allowed interval.",
        area="optimisation",
    ),
    CatalogueEntry(
        "newton-correction-no-derivative",
        "Subtracting f(x) from x without dividing by f'(x).",
        area="numerical methods",
    ),
    CatalogueEntry(
        "newton-derivative-missing-factor-two",
        "Differentiating m x^2 as m x, so the Newton correction comes out twice as large.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "newton-residual-at-previous-estimate",
        "Evaluating the residual at the earlier estimate instead of the latest one.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "jacobi-reuses-starting-values",
        "Using the starting values again in the second sweep instead of the results of the first.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "bisection-width-shrinks-linearly",
        "Counting the steps for a bracket to narrow by dividing its width by the number of steps, instead of halving it at every step.",
        area="numerical methods",
    ),
    CatalogueEntry(
        "vertex-sign-error", "Giving the vertex of ax^2 + bx + c as b/(2a), with the wrong sign.", area="optimisation"
    ),
    CatalogueEntry(
        "constant-term-as-minimum",
        "Reporting the constant term of the quadratic as its minimum value.",
        area="optimisation",
    ),
    CatalogueEntry(
        "endpoint-taken-though-vertex-inside",
        "Taking an end of the allowed interval as the minimiser though the vertex lies inside the interval.",
        area="optimisation",
    ),
    CatalogueEntry(
        "limits-added",
        "Adding the values of the antiderivative at the two limits instead of subtracting the lower from the upper.",
        area="integration",
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
