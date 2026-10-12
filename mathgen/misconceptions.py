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
    # -- r3 exam chains (rounds 3g to 3j), merged at integration: one id per error, whichever skill it turns up in --
    CatalogueEntry(
        "dropped-factor-of-four",
        "Leaving the factor 4 out of a formula such as b^2 - 4ac, or using 2a where 4a belongs.",
        area="general",
    ),
    CatalogueEntry(
        "forgot-to-divide-roots-by-leading-coefficient",
        "Taking the gap between the roots of ax^2 + bx + c as the square root of the discriminant, without "
        "dividing by |a|.",
        area="dynamics",
    ),
    CatalogueEntry(
        "gave-half-the-distance-between-equilibria",
        "Giving the distance from the middle of two equilibria to one of them where the distance between them "
        "was asked.",
        area="dynamics",
    ),
    CatalogueEntry(
        "gave-parameter-gap-for-equilibrium-gap",
        "Giving the difference between two values of the parameter where the distance between two equilibria "
        "was asked.",
        area="dynamics",
    ),
    CatalogueEntry(
        "gave-root-at-other-parameter-value",
        "Giving one of the two equilibria found for a different parameter value where the equilibrium at the "
        "threshold was asked.",
        area="dynamics",
    ),
    CatalogueEntry(
        "forgot-square-root",
        "Stopping at the square of the quantity without taking the square root.",
        area="general",
    ),
    CatalogueEntry(
        "dropped-factor-of-two",
        "Leaving a factor 2 (or 1/2) out of a formula, so the result is twice or half what it should be.",
        area="general",
    ),
    CatalogueEntry(
        "sign-slip-in-formula",
        "Using the wrong sign for one term of a standard formula or rearrangement, such as + where the formula has -.",
        area="general",
    ),
    CatalogueEntry(
        "force-at-end-point-reported-for-work",
        "Giving the force at the end point where the work done by a varying force was asked.",
        area="mechanics",
    ),
    CatalogueEntry(
        "forgot-initial-kinetic-energy",
        "Working out the final speed from the work alone, as if the body started at rest, and leaving out its "
        "initial kinetic energy.",
        area="mechanics",
    ),
    CatalogueEntry(
        "gave-final-kinetic-energy-for-work",
        "Giving the final kinetic energy where the work done was asked, so the initial kinetic energy is "
        "counted as work.",
        area="mechanics",
    ),
    CatalogueEntry(
        "integrated-to-half-the-distance",
        "Working out the work only to half the distance asked.",
        area="mechanics",
    ),
    CatalogueEntry(
        "added-instead-of-subtracted",
        "Adding two quantities where one should be subtracted from the other.",
        area="general",
    ),
    CatalogueEntry(
        "resistance-ignored-in-energy-balance",
        "Leaving the work done by a resisting force out of the energy balance, so the speed is unchanged.",
        area="mechanics",
    ),
    CatalogueEntry(
        "determinant-adds-cross-term",
        "Adding the product (or square) of the off-diagonal entries in a 2 by 2 determinant where it is subtracted.",
        area="general",
    ),
    CatalogueEntry(
        "trace-adds-off-diagonal",
        "Adding the off-diagonal entries (the cross term) into the trace, which is the sum of the diagonal "
        "entries only.",
        area="general",
    ),
    CatalogueEntry(
        "definiteness-assumed-from-diagonal-signs",
        "Deciding convex or concave from the signs of the x^2 and y^2 coefficients alone, without the determinant.",
        area="optimisation",
    ),
    CatalogueEntry(
        "counted-from-determinant-sign-alone",
        "Deciding the number of negative eigenvalues from the sign of the determinant alone: one if it is "
        "negative, otherwise two.",
        area="optimisation",
    ),
    CatalogueEntry(
        "counted-negative-diagonal-entries-instead-of-eigenvalues",
        "Counting negative diagonal entries of the Hessian instead of negative eigenvalues.",
        area="optimisation",
    ),
    CatalogueEntry(
        "counted-positive-instead-of-negative-eigenvalues",
        "Counting the positive eigenvalues where the negative ones were asked.",
        area="optimisation",
    ),
    CatalogueEntry(
        "determinant-ignores-cross-term",
        "Working out a 2 by 2 determinant as the product of the diagonal entries and leaving out the cross term.",
        area="general",
    ),
    CatalogueEntry(
        "eigenvalue-is-half-the-trace",
        "Giving half the trace as an eigenvalue, leaving out the square root of the discriminant.",
        area="general",
    ),
    CatalogueEntry(
        "gave-other-root",
        "Giving the other root, eigenvalue or factor where the larger (or smaller, or faster) one was asked.",
        area="general",
    ),
    CatalogueEntry(
        "indefinite-assumed-from-cross-term-present",
        "Assuming a quadratic with an xy term must be indefinite, without comparing the cross term with the diagonal.",
        area="optimisation",
    ),
    CatalogueEntry(
        "sign-of-result-reversed",
        "Giving the negative of the right answer: the working is sound but the sign is lost or flipped at the "
        "last step.",
        area="general",
    ),
    CatalogueEntry(
        "trace-from-first-diagonal-entry-only",
        "Taking the first diagonal entry of the Hessian as its trace instead of adding both.",
        area="optimisation",
    ),
    CatalogueEntry(
        "used-coefficients-instead-of-second-derivatives",
        "Building the Hessian from the coefficients a and b of ax^2 and by^2 instead of the second "
        "derivatives 2a and 2b.",
        area="optimisation",
    ),
    CatalogueEntry(
        "omitted-entry-in-total",
        "Leaving one term, entry or observation out of a sum.",
        area="general",
    ),
    CatalogueEntry(
        "gave-constant-quotient-term-for-coefficient",
        "Giving the constant term of the quotient where a coefficient of the product was asked.",
        area="algebra",
    ),
    CatalogueEntry(
        "gave-factor-value-for-product-value",
        "Giving the value of one factor (the generator or the quotient) at the point where the value of the "
        "product was asked.",
        area="algebra",
    ),
    CatalogueEntry(
        "swapped-quotient-coefficients-in-product",
        "Swapping the two quotient coefficients when finding the x coefficient of a product, so that s*q1 + "
        "t*q0 is used for s*q0 + t*q1.",
        area="algebra",
    ),
    CatalogueEntry(
        "forgot-to-square-cross-gram-entry",
        "Subtracting the off-diagonal entry itself, not its square, when working out the determinant of the "
        "normal-equations matrix.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "ignored-coupling-between-basis-functions",
        "Solving each normal equation on its own, as if the basis functions did not overlap: c1 = b1/G11 and "
        "c2 = b2/G22.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "summed-sizes-not-squares",
        "Adding the sizes (absolute values) of the quantities instead of their squares.",
        area="statistics",
    ),
    CatalogueEntry(
        "summed-signed-values",
        "Adding the quantities with their signs and without squaring, so positive and negative values cancel.",
        area="statistics",
    ),
    CatalogueEntry(
        "swapped-coefficient-order",
        "Giving the coefficients in the wrong order, or using them with the wrong basis functions.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "total-sum-of-squares-for-residual-sum",
        "Giving the sum of the squares of the data values where the sum of the squared residuals was asked.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "common-log-for-natural-log",
        "Giving log10(a) as the limit of (a^h - 1)/h, which is ln(a).",
        area="calculus",
    ),
    CatalogueEntry(
        "difference-quotient-taken-at-h-equals-1",
        "Giving a - 1, the value of (a^h - 1)/h at h = 1, instead of reading the trend as h gets small.",
        area="calculus",
    ),
    CatalogueEntry(
        "exponential-derivative-missing-ln-factor",
        "Giving the gradient of a^x as a^x, leaving out the constant factor that the limit of (a^h - 1)/h supplies.",
        area="calculus",
    ),
    CatalogueEntry(
        "gradient-taken-at-x-equals-zero",
        "Giving the gradient of a^x at x = 0, which is the limit itself, where the gradient at another point "
        "was asked.",
        area="calculus",
    ),
    CatalogueEntry(
        "power-rule-applied-to-exponential",
        "Differentiating a^x with the power rule, as x a^(x-1), instead of using the limit of (a^h - 1)/h.",
        area="calculus",
    ),
    CatalogueEntry(
        "reads-coarsest-table-row-as-limit",
        "Reading the limit off the first table row (the largest h) instead of the value the rows settle towards.",
        area="calculus",
    ),
    CatalogueEntry(
        "tangent-intercept-ratio-inverted",
        "Dividing the slope by the value, or using L where 1/L belongs, when finding where a tangent meets the axis.",
        area="calculus",
    ),
    CatalogueEntry(
        "tangent-intercept-ignores-base-point",
        "Giving the offset of the tangent's x-intercept from the point of tangency without adding that "
        "point's x-coordinate (or without the sign).",
        area="calculus",
    ),
    CatalogueEntry(
        "tangent-intercept-uses-gradient-for-subtangent",
        "Finding where a tangent to a^x meets the x-axis with the full gradient f'(c) in place of the limit "
        "L, so the intercept is c - 1/f'(c) instead of c - 1/L.",
        area="calculus",
    ),
    CatalogueEntry(
        "added-instead-of-multiplied",
        "Adding (or subtracting) two quantities where they should be multiplied.",
        area="general",
    ),
    CatalogueEntry(
        "counted-only-the-best-intermediate-town",
        "Counting the journeys through only one intermediate town, the one with the most, instead of summing "
        "over all of them.",
        area="matrices",
    ),
    CatalogueEntry(
        "reversed-direction",
        "Counting or totalling in the opposite direction from the one asked.",
        area="matrices",
    ),
    CatalogueEntry(
        "left-out-direct-services",
        "Giving only the two-stage journeys where journeys of at most two stages were asked, leaving out the "
        "direct services.",
        area="matrices",
    ),
    CatalogueEntry(
        "left-out-two-service-journeys",
        "Giving only the direct services where journeys of at most two stages were asked.",
        area="matrices",
    ),
    CatalogueEntry(
        "multiplied-instead-of-added",
        "Multiplying two quantities where they should be added.",
        area="general",
    ),
    CatalogueEntry(
        "multiplied-row-by-row-not-row-by-column",
        "Multiplying the row for the start town by the row (not the column) for the end town when counting "
        "two- stage journeys.",
        area="matrices",
    ),
    CatalogueEntry(
        "opposite-direction-total",
        "Giving the total into a town (a column) where the total out of it (a row) was asked, or the reverse.",
        area="matrices",
    ),
    CatalogueEntry(
        "read-wrong-row-or-column",
        "Reading the entry, total or case from the neighbouring row, column or case instead of the one named.",
        area="general",
    ),
    CatalogueEntry(
        "skipped-unit-conversion",
        "Using a quantity in the wrong unit without converting it.",
        area="measurement",
    ),
    CatalogueEntry(
        "gave-damped-frequency-for-decay-rate",
        "Giving the damped angular frequency, the imaginary part of the roots, where the decay rate was asked.",
        area="differential-equations",
    ),
    CatalogueEntry(
        "gave-natural-frequency-for-decay-rate",
        "Giving the undamped natural frequency sqrt(k/m) where the decay rate was asked.",
        area="differential-equations",
    ),
    CatalogueEntry(
        "squared-damping-ratio",
        "Giving the square of the damping ratio.",
        area="differential-equations",
    ),
    CatalogueEntry(
        "used-product-mk-for-critical-damping",
        "Giving mk as the critical damping constant instead of 2 sqrt(mk).",
        area="differential-equations",
    ),
    CatalogueEntry(
        "used-underdamped-decay-rate-for-all-regimes",
        "Using the underdamped decay rate b/(2m) for an overdamped system, whose slower root is smaller.",
        area="differential-equations",
    ),
    CatalogueEntry(
        "wrong-rate-time-unit",
        "Giving a rate per the wrong unit of time (per second, per hour) for the one asked.",
        area="rates",
    ),
    CatalogueEntry(
        "wrong-unit-conversion-factor",
        "Converting between units with the wrong factor, or in the wrong direction.",
        area="measurement",
    ),
    CatalogueEntry(
        "multiplied-instead-of-divided",
        "Multiplying where the quantity should be divided.",
        area="general",
    ),
    CatalogueEntry(
        "deviation-multiplied-by-factor-once",
        "Multiplying the difference from the fixed point by the growth factor once, as if one step had been "
        "taken, where the difference itself was asked.",
        area="sequences",
    ),
    CatalogueEntry(
        "divides-by-retained-fraction",
        "Dividing the added amount by the retained fraction a instead of by 1 - a to find the long-term value.",
        area="sequences",
    ),
    CatalogueEntry(
        "reports-added-amount",
        "Reporting the constant (the added amount) as the long-term value.",
        area="sequences",
    ),
    CatalogueEntry(
        "gave-start-value-for-deviation",
        "Giving the starting value where the difference between it and the fixed point was asked.",
        area="sequences",
    ),
    CatalogueEntry(
        "mistakes-recorded-value-for-longterm",
        "Reporting a recorded value (the starting value) as the long-term value.",
        area="sequences",
    ),
    CatalogueEntry(
        "linear-growth-instead-of-geometric",
        "Treating the difference from the fixed point as growing by a fixed amount each step instead of being "
        "multiplied by the factor a each step.",
        area="sequences",
    ),
    CatalogueEntry(
        "off-by-one-count",
        "Counting one too many or one too few.",
        area="general",
    ),
    CatalogueEntry(
        "reported-fixed-point-for-term",
        "Giving the fixed point where a later term of the sequence was asked.",
        area="sequences",
    ),
    CatalogueEntry(
        "added-equations-without-multipliers",
        "Adding the right-hand sides of two equations when the dependent equation is a different multiple of each.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "dropped-the-parameter-offset-from-the-z-coefficient",
        "Making the third row a combination of the first two without allowing for the parameter in its own z "
        "coefficient, so the critical parameter is out by the offset.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "set-coefficient-zero-for-determinant",
        "Setting one coefficient to zero (or giving the constant in k + 3) instead of making the determinant zero.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "gave-other-variable-of-solution-point",
        "Giving the value of a different variable of the same solution point.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "left-parameter-out",
        "Leaving the unknown parameter out of a rule or right-hand side, treating it as zero.",
        area="general",
    ),
    CatalogueEntry(
        "evaluated-function-for-derivative",
        "Reporting the value of the function at the point when the derivative's value was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "power-rule-exponent-not-lowered",
        "Differentiating a power by multiplying by the exponent but leaving the exponent as it was, so x^3 "
        "becomes 3x^3.",
        area="calculus",
    ),
    CatalogueEntry(
        "constant-term-differentiated-as-itself",
        "Carrying a constant term through differentiation instead of dropping it.",
        area="calculus",
    ),
    CatalogueEntry(
        "differentiated-twice",
        "Differentiating a second time and reporting the second derivative when the first was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "derivative-of-cubic-term-only",
        "Differentiating only the highest-power term and ignoring the other terms of the polynomial.",
        area="calculus",
    ),
    CatalogueEntry(
        "derivative-drops-cubic-term",
        "Differentiating every term except the highest-power one.",
        area="calculus",
    ),
    CatalogueEntry(
        "dropped-linear-correction",
        "Reporting f(a) as the linear approximation, leaving out the correction f'(a) times the offset.",
        area="calculus",
    ),
    CatalogueEntry(
        "evaluated-function-at-target",
        "Reporting the true value of the function at the target point when its linear approximation was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "dropped-base-value",
        "Reporting only the correction f'(a)(x - a) as the linear approximation, leaving out f(a).",
        area="calculus",
    ),
    CatalogueEntry(
        "change-in-f-for-error",
        "Reporting the change f(x) - f(a) as the error of the approximation.",
        area="calculus",
    ),
    CatalogueEntry(
        "reported-approximation-as-error",
        "Reporting the approximation itself when the error of the approximation was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "divided-instead-of-multiplied",
        "Dividing where the quantity should be multiplied.",
        area="general",
    ),
    CatalogueEntry(
        "outflow-at-inflow-concentration",
        "Using the concentration of the incoming liquid for the liquid leaving a well-mixed tank, which gives "
        "the inflow rate again.",
        area="rates",
    ),
    CatalogueEntry(
        "outflow-forgot-to-divide-by-volume",
        "Multiplying the flow rate by the amount in the tank without dividing by the tank's volume.",
        area="rates",
    ),
    CatalogueEntry(
        "outflow-adds-flow-to-volume",
        "Dividing by the tank's volume plus the flow rate instead of the volume alone.",
        area="rates",
    ),
    CatalogueEntry(
        "ignores-outflow",
        "Giving the rate in as the net rate of change and leaving out what flows out.",
        area="rates",
    ),
    CatalogueEntry(
        "reports-outflow-as-net-rate",
        "Reporting the rate out as the net rate of change, leaving out the rate in.",
        area="rates",
    ),
    CatalogueEntry(
        "equilibrium-forgot-flow-divisor",
        "Setting the rate in equal to (flow/volume) times Q but not dividing by the flow, so the long-run "
        "amount is too large by the flow rate.",
        area="rates",
    ),
    CatalogueEntry(
        "equilibrium-inverted-rate-constant",
        "Multiplying by the rate constant (flow/volume) instead of dividing when solving for the long-run amount.",
        area="rates",
    ),
    CatalogueEntry(
        "equilibrium-is-initial-amount",
        "Giving the starting amount as the amount the tank approaches in the long run.",
        area="rates",
    ),
    CatalogueEntry(
        "speed-adds-components",
        "Adding the components of a vector instead of taking the square root of the sum of their squares to "
        "get its length or speed.",
        area="calculus",
    ),
    CatalogueEntry(
        "speed-from-accelerations",
        "Using the acceleration components in place of the velocity components when finding the speed.",
        area="calculus",
    ),
    CatalogueEntry(
        "dot-and-cross-product-swapped",
        "Using the dot product where the 2-D cross product (determinant) was needed, or the other way round.",
        area="linear algebra",
    ),
    CatalogueEntry(
        "cross-term-product-of-first-derivatives",
        "Computing x'y' - x''y'' by pairing the derivatives of the same order instead of x'y'' - y'x''.",
        area="calculus",
    ),
    CatalogueEntry(
        "curvature-wrong-power-of-speed",
        "Using the speed or its square in the curvature (or radius) formula where the speed cubed belongs.",
        area="calculus",
    ),
    CatalogueEntry(
        "curvature-radius-confused",
        "Giving the curvature where the radius of curvature was asked, or the reverse.",
        area="calculus",
    ),
    CatalogueEntry(
        "forgot-absolute-value",
        "Leaving the sign on a quantity that should be taken as an absolute value, so the result is negative.",
        area="general",
    ),
    CatalogueEntry(
        "unit-price-of-the-other-pack",
        "Reading the unit price of the other pack when this pack's was asked for.",
        area="measurement",
    ),
    CatalogueEntry(
        "compares-pack-totals-not-unit-prices",
        "Comparing or subtracting the packs' total prices instead of their unit prices.",
        area="measurement",
    ),
    CatalogueEntry(
        "reports-cost-instead-of-saving",
        "Reporting the cost of the cheaper option when the saving was asked for.",
        area="measurement",
    ),
    CatalogueEntry(
        "reports-value-as-residual",
        "Reporting the predicted or the observed value of y when the residual was asked.",
        area="statistics",
    ),
    CatalogueEntry(
        "squares-the-sum-of-absolute-residuals",
        "Squaring the total of the residual sizes instead of adding the squares of each residual.",
        area="statistics",
    ),
    CatalogueEntry(
        "divided-sum-of-squares-by-count",
        "Dividing a sum of squares by n or n - 1, giving a mean or a variance where the sum was asked.",
        area="statistics",
    ),
    CatalogueEntry(
        "sums-squares-of-y-values-for-sst",
        "Adding the squares of the y values themselves instead of the squared deviations from their mean.",
        area="statistics",
    ),
    CatalogueEntry(
        "unexplained-proportion-for-explained",
        "Reporting SSE/SST, the proportion of variation not explained, instead of one minus it.",
        area="statistics",
    ),
    CatalogueEntry(
        "r-reported-as-r-squared",
        "Reporting the correlation coefficient r when its square r^2 was asked for.",
        area="statistics",
    ),
    CatalogueEntry(
        "subtracted-instead-of-added",
        "Subtracting where the quantities should be added.",
        area="general",
    ),
    CatalogueEntry(
        "trace-is-determinant",
        "Reporting the determinant when the trace was asked for.",
        area="dynamics",
    ),
    CatalogueEntry(
        "determinant-wrong-pairing",
        "Pairing the wrong entries of a 2 by 2 matrix in the determinant (same row, same column or a sum in "
        "place of a product).",
        area="general",
    ),
    CatalogueEntry(
        "discriminant-forgot-to-square-trace",
        "Computing T - 4D, leaving the trace unsquared.",
        area="dynamics",
    ),
    CatalogueEntry(
        "wrong-constant-in-formula",
        "Using the wrong numerical constant in a standard formula, such as 4a for 2a or 3v - 3 for 3v - 6.",
        area="general",
    ),
    CatalogueEntry(
        "discriminant-uses-trace-only",
        "Reporting T^2 and leaving out the 4D term.",
        area="dynamics",
    ),
    CatalogueEntry(
        "ignores-trace-in-eigenvalue",
        "Reporting half the root of the discriminant as the eigenvalue and leaving out the trace.",
        area="dynamics",
    ),
    CatalogueEntry(
        "real-part-from-wrong-invariant",
        "Taking half the determinant or half the discriminant as the real part of complex eigenvalues.",
        area="dynamics",
    ),
    CatalogueEntry(
        "group-order-as-lcm",
        "Taking the least common multiple of the factors' orders as the order of the product group.",
        area="groups",
    ),
    CatalogueEntry(
        "group-order-as-larger-factor",
        "Taking the order of the larger factor as the order of the product group.",
        area="groups",
    ),
    CatalogueEntry(
        "element-order-not-lcm",
        "Taking the sum, the product, the larger or the gcd of the component orders where the least common "
        "multiple gives the order of an element.",
        area="groups",
    ),
    CatalogueEntry(
        "element-order-is-group-exponent",
        "Giving the largest element order in the group when the order of the named element was asked for.",
        area="groups",
    ),
    CatalogueEntry(
        "largest-order-is-group-order",
        "Taking the order of the whole group as the largest order of an element, which assumes the group is cyclic.",
        area="groups",
    ),
    CatalogueEntry(
        "largest-order-is-order-of-named-element",
        "Reporting the order of the named element when the largest order in the group was asked for.",
        area="groups",
    ),
    CatalogueEntry(
        "index-confused-with-subgroup-order",
        "Reporting the order of the subgroup where its index was asked.",
        area="groups",
    ),
    CatalogueEntry(
        "index-of-trivial-subgroup-confusion",
        "Reporting the order of the whole group, the index of the trivial subgroup, as the index.",
        area="groups",
    ),
    CatalogueEntry(
        "index-as-smaller-modulus",
        "Taking the smaller of the two moduli as the index.",
        area="groups",
    ),
    CatalogueEntry(
        "index-zero-for-whole-group",
        "Reporting 0 for the index of a subgroup that is the whole group, instead of 1.",
        area="groups",
    ),
    CatalogueEntry(
        "cr-wrong-partials-paired",
        "Pairing the wrong partial derivatives in a Cauchy-Riemann equation.",
        area="complex-analysis",
    ),
    CatalogueEntry(
        "reports-one-partial-not-the-difference",
        "Reporting one partial derivative alone where a difference or sum of two was asked.",
        area="complex-analysis",
    ),
    CatalogueEntry(
        "absolute-values-taken-of-sum",
        "Taking the absolute value of the sum of the discrepancies instead of the sum of their absolute values.",
        area="complex-analysis",
    ),
    CatalogueEntry(
        "larger-discrepancy-only",
        "Reporting the larger of the two discrepancies instead of their total.",
        area="complex-analysis",
    ),
    CatalogueEntry(
        "all-four-partials-added",
        "Adding the sizes of all four partial derivatives instead of the two discrepancies.",
        area="complex-analysis",
    ),
    CatalogueEntry(
        "second-pair-of-partials-differenced",
        "Forming u_y - v_x and reporting it as the total of the two discrepancies.",
        area="complex-analysis",
    ),
    CatalogueEntry(
        "first-pair-of-absolute-partials-added",
        "Adding the sizes of u_x and v_y and reporting the result as the total discrepancy.",
        area="complex-analysis",
    ),
    CatalogueEntry(
        "row-and-column-swapped",
        "Using the row number where the column number belongs, and the reverse.",
        area="matrices",
    ),
    CatalogueEntry(
        "ratio-terms-reversed",
        "Writing a ratio with its two terms the wrong way round.",
        area="ratio",
    ),
    CatalogueEntry(
        "part-to-part-and-part-to-whole-confused",
        "Mixing up the ratio a : b of part to part with the fraction a/(a+b) of the whole.",
        area="ratio",
    ),
    CatalogueEntry(
        "ratio-decimal-cleared-by-different-powers-of-ten",
        "Clearing the decimal in one term by a different power of ten from the other, which changes the ratio.",
        area="ratio",
    ),
    CatalogueEntry(
        "other-component-given",
        "Giving the fraction or amount of the other component than the one asked for.",
        area="ratio",
    ),
    CatalogueEntry(
        "whole-over-part-fraction-inverted",
        "Writing the whole over the part, (a+b)/a, as the fraction of the whole.",
        area="ratio",
    ),
    CatalogueEntry(
        "half-of-the-batch-for-each-component",
        "Splitting a batch equally between the two components regardless of their ratio.",
        area="ratio",
    ),
    CatalogueEntry(
        "batch-times-ratio-instead-of-share",
        "Multiplying the batch by the part-to-part ratio a/b instead of the share a/(a+b).",
        area="ratio",
    ),
    CatalogueEntry(
        "batch-divided-by-one-term",
        "Dividing the batch by one term of the ratio instead of splitting it in the ratio.",
        area="ratio",
    ),
    CatalogueEntry(
        "multiplier-reported-as-amount",
        "Reporting the number of ratio units in a batch (the multiplier) as an amount.",
        area="ratio",
    ),
    CatalogueEntry(
        "ratio-term-subtracted-from-batch",
        "Subtracting a term of the ratio from the batch size to find an amount.",
        area="ratio",
    ),
    CatalogueEntry(
        "face-sum-as-edges",
        "Using the sum of the face lengths as the number of edges (or as e in Euler's formula) without "
        "halving it, though each edge borders two faces.",
        area="graph-theory",
    ),
    CatalogueEntry(
        "outer-face-left-out",
        "Leaving the outer face out of the faces counted.",
        area="graph-theory",
    ),
    CatalogueEntry(
        "spare-edges-reported-as-maximum",
        "Reporting the greatest possible number of edges when the number that could still be added was asked for.",
        area="graph-theory",
    ),
    CatalogueEntry(
        "error-uses-derivative-at-target",
        "Subtracting f'(a) times the offset from the true value, instead of subtracting the whole linear "
        "approximation, when finding the error.",
        area="calculus",
    ),
    CatalogueEntry(
        "fraction-ignores-the-second-term",
        "Giving 1 over the first term of the ratio as the fraction of the whole.",
        area="ratio",
    ),
    CatalogueEntry(
        "ratio-of-difference-to-first-term",
        "Forming a ratio from the difference between the two amounts and the first amount.",
        area="ratio",
    ),
    CatalogueEntry(
        "columns-counted-from-the-right",
        "Counting columns from the right-hand side when asked in which column an entry lies.",
        area="matrices",
    ),
    CatalogueEntry(
        "loan-term-ignores-interest",
        "Dividing the loan by the monthly repayment to get the number of months, as though no interest were charged.",
        area="financial-maths",
    ),
    CatalogueEntry(
        "loan-term-annual-rate-as-monthly",
        "Using a year's interest as though it were charged every month, which takes far too much off each repayment.",
        area="financial-maths",
    ),
    CatalogueEntry(
        "months-and-years-confused",
        "Treating a number of months as a number of years, or the reverse.",
        area="financial-maths",
    ),
    CatalogueEntry(
        "loan-total-repaid-for-interest",
        "Reporting everything repaid (principal and interest) where only the interest was asked for.",
        area="financial-maths",
    ),
    CatalogueEntry(
        "loan-interest-for-one-year-only",
        "Working out one year's interest and stopping, instead of the interest over the whole term.",
        area="financial-maths",
    ),
    CatalogueEntry(
        "loan-months-saved-for-dollars-saved",
        "Reporting how many months sooner the loan is repaid when the dollars of interest saved were asked for.",
        area="financial-maths",
    ),
    CatalogueEntry(
        "loan-interest-of-one-plan-only",
        "Giving the interest under the second plan alone instead of comparing it with the first plan.",
        area="financial-maths",
    ),
    CatalogueEntry(
        "modulus-and-totient-swapped",
        "Giving the modulus n where Euler's totient was asked, or the reverse.",
        area="number-theory",
    ),
    CatalogueEntry(
        "totient-as-n-minus-one",
        "Taking Euler's totient of a product of two primes to be n - 1.",
        area="number-theory",
    ),
    CatalogueEntry(
        "totient-dropped-the-plus-one",
        "Expanding (p - 1)(q - 1) as n - p - q and leaving out the final + 1.",
        area="number-theory",
    ),
    CatalogueEntry(
        "rsa-inverse-taken-modulo-n",
        "Finding the inverse of the public exponent modulo n instead of modulo Euler's totient.",
        area="number-theory",
    ),
    CatalogueEntry(
        "rsa-private-exponent-equals-public",
        "Reporting the public exponent as the private exponent.",
        area="number-theory",
    ),
    CatalogueEntry(
        "rsa-private-exponent-from-phi-over-e",
        "Estimating the private exponent as the totient divided by the public exponent, instead of solving e "
        "d = 1 modulo the totient.",
        area="number-theory",
    ),
    CatalogueEntry(
        "rsa-decrypted-with-public-exponent",
        "Raising the ciphertext to the public exponent again, which encrypts a second time instead of decrypting.",
        area="number-theory",
    ),
    CatalogueEntry(
        "rsa-power-reduced-modulo-totient",
        "Reducing c^d modulo the totient instead of modulo n.",
        area="number-theory",
    ),
    CatalogueEntry(
        "multiplied-instead-of-exponentiated",
        "Multiplying the base by the exponent where the base is raised to the exponent.",
        area="general",
    ),
    CatalogueEntry(
        "rsa-ciphertext-reported-as-message",
        "Giving back the ciphertext where the decrypted message was asked for.",
        area="number-theory",
    ),
    CatalogueEntry(
        "ode-coordinates-swapped",
        "Substituting the point's coordinates in the wrong order into the right-hand side of the differential "
        "equation.",
        area="differential-equations",
    ),
    CatalogueEntry(
        "ode-slope-from-one-variable-only",
        "Evaluating only one of the terms in x and y of the right-hand side, together with the constant.",
        area="differential-equations",
    ),
    CatalogueEntry(
        "euler-step-size-not-applied",
        "Adding the gradient to y without multiplying by the step size (or adding the step size itself).",
        area="differential-equations",
    ),
    CatalogueEntry(
        "euler-increment-reported-as-estimate",
        "Giving the increment (step size times gradient) where the new estimate of y was asked for.",
        area="differential-equations",
    ),
    CatalogueEntry(
        "euler-gradient-uses-old-point",
        "Finding the next gradient with an old x or y, or reusing the first gradient, instead of the gradient "
        "at the new point.",
        area="differential-equations",
    ),
    CatalogueEntry(
        "euler-restarts-from-initial-value",
        "Starting the second step from the original value of y instead of from the first estimate.",
        area="differential-equations",
    ),
    CatalogueEntry(
        "ridge-wrong-variable-squared",
        "Squaring the y values where the x values (or the products x y) were wanted.",
        area="statistics",
    ),
    CatalogueEntry(
        "ridge-product-signs-ignored",
        "Adding the sizes of the products x y and ignoring their signs.",
        area="statistics",
    ),
    CatalogueEntry(
        "ridge-observations-mispaired",
        "Multiplying each x by the y of the next observation instead of its own.",
        area="statistics",
    ),
    CatalogueEntry(
        "ridge-penalty-omitted",
        "Leaving the ridge penalty out of the calculation, which gives the ordinary least-squares value.",
        area="statistics",
    ),
    CatalogueEntry(
        "ridge-penalty-added-to-numerator",
        "Adding the penalty to Sxy instead of to Sxx.",
        area="statistics",
    ),
    CatalogueEntry(
        "ridge-penalty-squared",
        "Adding the square of the penalty to Sxx instead of the penalty itself.",
        area="statistics",
    ),
    CatalogueEntry(
        "ridge-prediction-gives-the-coefficient",
        "Giving the coefficient where the prediction (coefficient times x) was asked for.",
        area="statistics",
    ),
    CatalogueEntry(
        "ridge-prediction-wrong-input",
        "Predicting from the wrong value of x.",
        area="statistics",
    ),
    CatalogueEntry(
        "uses-n-not-n-minus-1",
        "Taking a tree on n vertices to have n edges (so reporting n, or a degree sum of 2n), instead of n - 1.",
        area="graph-theory",
    ),
    CatalogueEntry(
        "uses-n-plus-1",
        "Taking a tree on n vertices to have n + 1 edges, instead of n - 1.",
        area="graph-theory",
    ),
    CatalogueEntry(
        "tree-leaves-counted-from-vertices-only",
        "Taking the number of leaves from the count of higher-degree vertices alone, ignoring their degrees.",
        area="graph-theory",
    ),
    CatalogueEntry(
        "tree-leaves-forgets-plus-one",
        "Using (m - 1) times the number of internal vertices for the leaves of a full m-ary tree, without adding 1.",
        area="graph-theory",
    ),
    CatalogueEntry(
        "tree-leaves-all-children",
        "Counting every child as a leaf in a full m-ary tree, though internal vertices have children too.",
        area="graph-theory",
    ),
    CatalogueEntry(
        "tree-leaves-one-more-than-internal",
        "Taking the leaves to be one more than the internal vertices, which is true only for a binary tree.",
        area="graph-theory",
    ),
    CatalogueEntry(
        "tree-leaves-counts-all-vertices",
        "Giving the number of vertices of a full m-ary tree where the number of leaves was asked for.",
        area="graph-theory",
    ),
    CatalogueEntry(
        "tree-vertices-omit-the-leaves",
        "Counting only the vertices that are not leaves.",
        area="graph-theory",
    ),
    CatalogueEntry(
        "tree-vertices-leaves-only",
        "Giving the number of leaves where the number of vertices was asked for.",
        area="graph-theory",
    ),
    CatalogueEntry(
        "tree-vertices-reported-as-edges",
        "Giving the number of edges where the number of vertices was asked for.",
        area="graph-theory",
    ),
    CatalogueEntry(
        "forgets-to-halve-degree-sum",
        "Reporting the sum of the degrees as the number of edges, without dividing by 2.",
        area="graph-theory",
    ),
    CatalogueEntry(
        "tree-edges-n-minus-two",
        "Taking a tree on n vertices to have n - 2 edges.",
        area="graph-theory",
    ),
    CatalogueEntry(
        "reports-given-edges-directly",
        "Giving the number of edges the graph has, instead of the number that exceed a spanning tree.",
        area="graph-theory",
    ),
    CatalogueEntry(
        "energy-one-term-only",
        "Counting only the kinetic or only the spring energy in the total.",
        area="mechanics",
    ),
    CatalogueEntry(
        "energy-mass-and-stiffness-swapped",
        "Using the mass where the stiffness belongs, and the stiffness where the mass belongs, in the energy formula.",
        area="mechanics",
    ),
    CatalogueEntry(
        "amplitude-and-max-speed-swapped",
        "Giving the amplitude where the maximum speed was asked, or the reverse.",
        area="mechanics",
    ),
    CatalogueEntry(
        "amplitude-adds-the-displacements",
        "Adding the displacements given for two positions instead of finding where the speed is zero.",
        area="mechanics",
    ),
    CatalogueEntry(
        "max-speed-frequency-as-k-over-m",
        "Using k / m as the angular frequency instead of its square root.",
        area="mechanics",
    ),
    CatalogueEntry(
        "max-speed-reported-as-given-speed",
        "Giving the speed stated in the question as the answer, without using conservation of energy.",
        area="mechanics",
    ),
    CatalogueEntry(
        "speed-at-position-reports-max-speed",
        "Giving the maximum speed where the speed at a stated position was asked for.",
        area="mechanics",
    ),
    CatalogueEntry(
        "speed-uses-displacement-in-place-of-remainder",
        "Taking the speed to be the angular frequency times the displacement, rather than times the square "
        "root of A^2 - x^2.",
        area="mechanics",
    ),
    CatalogueEntry(
        "group-units-counted-as-all-nonzero-residues",
        "Counting every nonzero residue as a unit, instead of only the residues with no common factor with "
        "the modulus.",
        area="groups",
    ),
    CatalogueEntry(
        "group-units-exclude-the-identity",
        "Counting the units but leaving out the identity element 1.",
        area="groups",
    ),
    CatalogueEntry(
        "group-units-counted-as-all-residues",
        "Counting every residue from 0 to n - 1 as a unit.",
        area="groups",
    ),
    CatalogueEntry(
        "group-totient-ignores-prime-powers",
        "Using (p - 1) for each prime in the modulus without allowing for a repeated prime factor, so the "
        "group is counted too small.",
        area="groups",
    ),
    CatalogueEntry(
        "group-element-order-taken-as-group-order",
        "Taking the order of an element to be the order of the whole group, as though every element generated it.",
        area="groups",
    ),
    CatalogueEntry(
        "group-order-computed-additively",
        "Finding the order of an element under addition (n divided by the gcd) in a multiplicative group.",
        area="groups",
    ),
    CatalogueEntry(
        "group-index-reports-elements-outside",
        "Giving the number of elements outside the subgroup where the index was asked for.",
        area="groups",
    ),
    CatalogueEntry(
        "group-outside-counts-inside",
        "Giving the number of elements in the subgroup where the number outside it was asked for.",
        area="groups",
    ),
    CatalogueEntry(
        "group-outside-counts-everything",
        "Giving the order of the whole group where the number of elements outside the subgroup was asked for.",
        area="groups",
    ),
    CatalogueEntry(
        "group-outside-reports-index",
        "Giving the index where the number of elements outside the subgroup was asked for.",
        area="groups",
    ),
    CatalogueEntry(
        "uses-min-instead-of-max",
        "Reporting the smaller absolute value of the two entries where the larger one gives the norm.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "operator-norm-sums-the-entries",
        "Adding the absolute values of the entries to get the operator norm.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "operator-norm-multiplies-the-entries",
        "Multiplying the two entries to get the operator norm.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "forgets-absolute-value-after-power",
        "Comparing the signed values after raising the entries to a power, so an odd power of a negative "
        "entry is ranked wrongly.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "operator-norm-inverse-uses-the-largest-entry",
        "Taking the norm of the inverse from the reciprocal of the largest entry; it comes from the smallest.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "operator-norm-inverse-not-reciprocated",
        "Giving the smallest absolute entry without taking its reciprocal.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "operator-norm-inverse-sums-the-reciprocals",
        "Adding the reciprocals of the entries instead of taking the larger of them.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "norm-from-a-difference",
        "Subtracting the smallest absolute entry from the largest to get the norm of the inverse or the "
        "condition number.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "condition-number-uses-the-smallest-entry",
        "Multiplying the largest entry by the smallest, instead of by the reciprocal of the smallest.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "condition-number-inverted",
        "Dividing the smallest entry by the largest, which turns the condition number upside down.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "minimiser-uses-the-constant-term",
        "Using the constant term c in place of b when finding the minimiser.",
        area="optimisation",
    ),
    CatalogueEntry(
        "minimum-value-reports-the-minimiser",
        "Giving the minimiser where the minimum value was asked for.",
        area="optimisation",
    ),
    CatalogueEntry(
        "minimum-value-reports-the-constant-term",
        "Giving the constant term as the minimum value.",
        area="optimisation",
    ),
    CatalogueEntry(
        "new-minimum-ignores-the-change",
        "Giving the old minimum value after the coefficient has changed.",
        area="optimisation",
    ),
    CatalogueEntry(
        "new-minimum-linearised",
        "Estimating the new minimum with the old minimum plus the change times the old minimiser, where the "
        "exact value was asked for.",
        area="optimisation",
    ),
    CatalogueEntry(
        "new-minimum-drops-the-cross-term",
        "Changing the minimum by only the square of the change, losing the cross term.",
        area="optimisation",
    ),
    CatalogueEntry(
        "sensitivity-estimate-assumed-exact",
        "Taking the linear estimate to equal the exact new minimum, so the difference is zero.",
        area="optimisation",
    ),
    CatalogueEntry(
        "sensitivity-error-omits-the-square",
        "Writing the error of the linear estimate as -d / (4a), without squaring the change.",
        area="optimisation",
    ),
    CatalogueEntry(
        "dose-and-daily-total-confused",
        "Giving the mass of one dose where the daily total was asked, or the reverse.",
        area="measurement",
    ),
    CatalogueEntry(
        "ignores-body-mass",
        "Leaving the body mass out of a dose calculation.",
        area="measurement",
    ),
    CatalogueEntry(
        "dose-ignores-the-rate",
        "Giving the body mass in kilograms as the dose in milligrams.",
        area="measurement",
    ),
    CatalogueEntry(
        "grams-reports-one-dose",
        "Converting the mass of one dose to grams where the mass given in a day was asked for.",
        area="measurement",
    ),
    CatalogueEntry(
        "tub-reports-the-mass-used",
        "Giving the mass used where the mass left in the tub was asked for.",
        area="measurement",
    ),
    CatalogueEntry(
        "tub-ignores-the-mass-used",
        "Giving the whole contents of the tub, in milligrams, as the amount left.",
        area="measurement",
    ),
    CatalogueEntry(
        "tub-uses-one-dose-a-day",
        "Taking one dose a day, instead of the full daily total, as the amount used.",
        area="measurement",
    ),
    CatalogueEntry(
        "recurrence-ratio-of-consecutive-values",
        "Taking the ratio of two recorded values as the constant a, as though the sequence were purely geometric.",
        area="sequences",
    ),
    CatalogueEntry(
        "rate-and-dose-swapped",
        "Giving the reciprocal of the ratio of consecutive differences as the constant a.",
        area="sequences",
    ),
    CatalogueEntry(
        "recurrence-retained-and-lost-swapped",
        "Giving 1 - a (the fraction lost) where the fraction retained, a, was asked for.",
        area="sequences",
    ),
    CatalogueEntry(
        "simple-difference-for-added-amount",
        "Taking the added amount b to be the plain difference between two recorded values, ignoring the "
        "fraction lost each period.",
        area="sequences",
    ),
    CatalogueEntry(
        "recurrence-added-amount-from-wrong-term",
        "Working out b as u(1) - a u(1) instead of u(1) - a u(0).",
        area="sequences",
    ),
    CatalogueEntry(
        "recurrence-added-amount-from-lost-fraction",
        "Working out b as u(0) (1 - a), the part of the first value that was lost.",
        area="sequences",
    ),
    CatalogueEntry(
        "recurrence-time-from-the-start-not-the-gap",
        "Measuring the time to shrink the starting value itself to within 1 unit, instead of the gap between "
        "the starting value and the long-term value.",
        area="sequences",
    ),
    CatalogueEntry(
        "multiplied-components-for-length",
        "Multiplying the components, a x b, instead of taking the square root of the sum of their squares.",
        area="linear algebra",
    ),
    CatalogueEntry(
        "dot-product-not-normalised",
        "Giving u . v without dividing by |u|, so the component along a unit vector is left unscaled.",
        area="linear algebra",
    ),
    CatalogueEntry(
        "divided-by-squared-length",
        "Dividing by the squared length |u|^2 where the length |u| was needed.",
        area="linear algebra",
    ),
    CatalogueEntry(
        "area-for-height",
        "Giving the parallelogram's area |det| where its height (the area divided by the base length) was asked.",
        kind="near_miss",
        area="linear algebra",
    ),
    CatalogueEntry(
        "gave-first-vector-component",
        "Reporting a component of e1 where the same component of e2 was asked.",
        kind="near_miss",
        area="linear algebra",
    ),
    CatalogueEntry(
        "gram-schmidt-skipped-projection",
        "Normalising v directly, without first subtracting its projection onto e1.",
        area="linear algebra",
    ),
    CatalogueEntry(
        "forgot-final-normalisation",
        "Leaving the orthogonal vector unnormalised: forgetting to divide by its length at the end.",
        area="linear algebra",
    ),
    CatalogueEntry(
        "momentum-combined-ignoring-direction",
        "Adding or subtracting momenta as sizes, ignoring that the direction of each velocity sets its sign.",
        area="mechanics",
    ),
    CatalogueEntry(
        "momentum-of-one-object",
        "Using the momentum of only one of the two bodies as the total.",
        area="mechanics",
    ),
    CatalogueEntry(
        "divided-by-one-mass",
        "Dividing the total momentum by one body's mass instead of the combined mass.",
        area="mechanics",
    ),
    CatalogueEntry(
        "averaged-the-velocities",
        "Averaging the two initial velocities for the common velocity, ignoring the masses.",
        area="mechanics",
    ),
    CatalogueEntry(
        "impulse-on-the-other-object",
        "Giving the impulse or force on the other body, which has the opposite sign, instead of the body asked about.",
        kind="near_miss",
        area="mechanics",
    ),
    CatalogueEntry(
        "final-momentum-for-impulse",
        "Giving the final momentum where the change in momentum (the impulse) was asked.",
        kind="near_miss",
        area="mechanics",
    ),
    CatalogueEntry(
        "initial-momentum-for-impulse",
        "Giving the negative of the initial momentum, forgetting the final momentum in the change.",
        area="mechanics",
    ),
    CatalogueEntry(
        "impulse-for-force",
        "Giving the impulse (force times time) where the average force was asked, forgetting to divide by the "
        "contact time.",
        area="mechanics",
    ),
    CatalogueEntry(
        "initial-velocity-for-final-velocity",
        "Giving one body's initial velocity as the velocity after the collision.",
        area="mechanics",
    ),
    CatalogueEntry(
        "masses-and-velocities-swapped",
        "Pairing each mass with the other body's velocity when forming the momenta.",
        area="mechanics",
    ),
    CatalogueEntry(
        "ignored-numerator",
        "Leaving out the numerator when finding a residue, so only the product of the pole differences is used.",
        area="complex analysis",
    ),
    CatalogueEntry(
        "numerator-at-wrong-pole",
        "Evaluating the numerator at a different pole from the one the residue is taken at.",
        area="complex analysis",
    ),
    CatalogueEntry(
        "dropped-denominator-factor",
        "Dropping one of the remaining linear factors of the denominator when evaluating the residue.",
        area="complex analysis",
    ),
    CatalogueEntry(
        "included-pole-outside-contour",
        "Adding the residues at poles that lie outside the contour.",
        area="complex analysis",
    ),
    CatalogueEntry(
        "sum-or-difference-of-factors-for-factor",
        "Giving the sum or the difference of the two prime factors where one factor was asked.",
        area="number-theory",
    ),
    CatalogueEntry(
        "totient-used-one-prime-unreduced",
        "Leaving one prime factor unreduced in phi(pq), such as p(q - 1) in place of (p - 1)(q - 1).",
        area="number theory",
    ),
    CatalogueEntry(
        "quotient-for-remainder",
        "Giving the quotient of the division where the remainder was asked.",
        area="number theory",
    ),
    CatalogueEntry(
        "reduced-exponent-modulo-n",
        "Reducing the exponent modulo n instead of modulo phi(n).",
        area="number theory",
    ),
    CatalogueEntry(
        "stopped-reducing-the-exponent",
        "Subtracting phi(n) from the exponent only once, so the exponent is not fully reduced.",
        area="number theory",
    ),
    CatalogueEntry(
        "reduced-exponent-modulo-one-prime",
        "Reducing the exponent modulo p - 1 for one prime factor, which is valid only modulo that prime, not modulo n.",
        area="number theory",
    ),
    CatalogueEntry(
        "residue-modulo-one-factor-only",
        "Giving the residue modulo one prime factor of n where the residue modulo n was asked.",
        area="number theory",
    ),
    CatalogueEntry(
        "unweighted-instead-of-weighted",
        "Averaging or adding the values with equal weight instead of weighting each by its case probability.",
        area="probability",
    ),
    CatalogueEntry(
        "probabilities-misassigned-to-cases",
        "Pairing each probability with the wrong case when weighting.",
        area="probability",
    ),
    CatalogueEntry(
        "weights-as-counts-not-probabilities",
        "Weighting by the percentage numbers themselves rather than by probabilities that sum to 1.",
        area="probability",
    ),
    CatalogueEntry(
        "forgot-to-subtract-square-of-mean",
        "Stopping at the weighted mean of the squared means, without subtracting the square of the overall mean.",
        area="probability",
    ),
    CatalogueEntry(
        "forgot-to-square-the-mean",
        "Subtracting the overall mean instead of its square when finding a variance from second moments.",
        area="probability",
    ),
    CatalogueEntry(
        "gave-square-of-mean-for-variance",
        "Giving the square of the overall mean where the variance of the means was asked.",
        area="probability",
    ),
    CatalogueEntry(
        "dropped-variance-of-means",
        "Giving only the mean of the conditional variances as the total variance, leaving out the variance of "
        "the conditional means.",
        area="probability",
    ),
    CatalogueEntry(
        "dropped-mean-of-variances",
        "Giving only the variance of the conditional means as the total variance, leaving out the mean of the "
        "conditional variances.",
        area="probability",
    ),
    CatalogueEntry(
        "inverted-frequency-and-period",
        "Dividing the time by the number of cycles (the period) where the frequency was asked, or the reverse.",
        area="waves",
    ),
    CatalogueEntry(
        "halved-the-frequency",
        "Halving the frequency, treating one rise and fall as two cycles.",
        area="waves",
    ),
    CatalogueEntry(
        "wavelength-unchanged-in-new-medium",
        "Keeping the wavelength the same in the new medium, though only the frequency stays the same.",
        area="waves",
    ),
    CatalogueEntry(
        "speed-ratio-inverted",
        "Scaling the wavelength by the speed ratio the wrong way up, so a slower wave gets a longer wavelength.",
        area="waves",
    ),
    CatalogueEntry(
        "speed-given-as-wavelength",
        "Giving the new speed as the wavelength, as if the frequency were 1 Hz.",
        area="waves",
    ),
    CatalogueEntry(
        "rounded-up-complete-wavelengths",
        "Rounding up the number of wavelengths in a section, counting a part-wavelength as complete.",
        area="waves",
    ),
    CatalogueEntry(
        "used-deep-water-wavelength",
        "Dividing the section length by the deep-water wavelength instead of the shallow-water wavelength.",
        area="waves",
    ),
    CatalogueEntry(
        "took-looser-bound",
        "Taking the looser of two bounds, the smaller lower bound or the larger upper bound, when every "
        "inequality must hold.",
        area="metric spaces",
    ),
    CatalogueEntry(
        "swapped-lower-and-upper-bound",
        "Giving the upper bound where the lower bound was asked, or the other way round.",
        area="metric spaces",
    ),
    CatalogueEntry(
        "bound-from-wrong-pair",
        "Combining distances from two triples that do not both contain the unknown pair, so the bound has "
        "nothing to do with t.",
        area="metric spaces",
    ),
    CatalogueEntry(
        "added-the-two-bounds",
        "Adding the two bounds instead of using the larger as the least value and the smaller as the greatest.",
        area="metric spaces",
    ),
    CatalogueEntry(
        "excluded-attainable-endpoints",
        "Treating the bounds as strict inequalities and so dropping both endpoints, though the triangle "
        "inequality allows equality.",
        area="metric spaces",
    ),
    CatalogueEntry(
        "counted-from-one-to-upper-bound",
        "Counting every positive integer up to the upper bound, ignoring the lower bound.",
        area="counting",
    ),
    CatalogueEntry(
        "gave-root-of-discriminant",
        "Giving the square root of the discriminant where the discriminant itself was asked.",
        area="algebra",
    ),
    CatalogueEntry(
        "gave-part-of-discriminant",
        "Giving only the 4ac part of the discriminant.",
        area="algebra",
    ),
    CatalogueEntry(
        "degree-of-polynomial-for-degree-of-extension",
        "Giving the degree of the polynomial as the degree of the field extension without checking that it is "
        "the minimal polynomial.",
        area="field theory",
    ),
    CatalogueEntry(
        "missed-perfect-square",
        "Treating the square root of a perfect square as irrational, so a rational root gets degree 2.",
        area="field theory",
    ),
    CatalogueEntry(
        "assumed-roots-rational",
        "Treating the roots as rational, so the extension has degree 1.",
        area="field theory",
    ),
    CatalogueEntry(
        "field-degree-whole-and-one-root-confused",
        "Giving the degree for one root where the field of all the roots was asked, or the reverse.",
        kind="near_miss",
        area="field theory",
    ),
    CatalogueEntry(
        "multiplied-degrees-ignoring-dependence",
        "Multiplying the degrees of two quadratic extensions without checking whether they generate the same field.",
        area="field theory",
    ),
    CatalogueEntry(
        "assumed-roots-generate-the-same-field",
        "Treating two independent square roots as generating the same quadratic field, so the degree is 2 not 4.",
        area="field theory",
    ),
    CatalogueEntry(
        "order-and-parameter-confused",
        "Giving the parameter where the order was asked, or the reverse.",
        kind="near_miss",
        area="matrices",
    ),
    CatalogueEntry(
        "gave-entries-for-rows",
        "Giving the number of entries of a square matrix where its number of rows was asked.",
        kind="near_miss",
        area="matrices",
    ),
    CatalogueEntry(
        "doubled-instead-of-shifted",
        "Doubling the parameter when the number of rows is the parameter plus a constant.",
        area="matrices",
    ),
    CatalogueEntry(
        "all-entries-taken-as-last-entry",
        "Treating every diagonal entry, or every entry, as equal to the last one.",
        area="matrices",
    ),
    CatalogueEntry(
        "gave-trace-for-total",
        "Giving the sum of the diagonal entries where the sum of all the entries was asked.",
        kind="near_miss",
        area="matrices",
    ),
    CatalogueEntry(
        "multiplied-trace-by-order",
        "Multiplying the diagonal sum by the order of the matrix to get the total of all entries.",
        area="matrices",
    ),
    CatalogueEntry(
        "summed-i-not-i-squared-on-diagonal",
        "Adding i rather than i squared down the diagonal of a product rule, so the sum of squares becomes a "
        "sum of integers.",
        area="matrices",
    ),
    CatalogueEntry(
        "read-constant-as-parameter",
        "Giving one of the constants in the row or column count as the value of the parameter.",
        area="matrices",
    ),
    CatalogueEntry(
        "treated-minutes-as-decimals",
        "Treating minutes as hundredths of an hour, or converting minutes to hours by dividing by 100.",
        area="time",
    ),
    CatalogueEntry(
        "ignored-the-minutes",
        "Working with the hours only and ignoring the minutes of the times.",
        area="time",
    ),
    CatalogueEntry(
        "subtracted-minutes-without-borrowing",
        "Subtracting the minutes the smaller way round instead of borrowing an hour, so the time difference is wrong.",
        area="time",
    ),
    CatalogueEntry(
        "read-wrong-tide",
        "Reading the wrong low or high tide from the chart.",
        kind="near_miss",
        area="time",
    ),
    CatalogueEntry(
        "gave-height-not-rise",
        "Giving the height of the high tide where the rise from low tide was asked.",
        kind="near_miss",
        area="time",
    ),
    CatalogueEntry(
        "timed-from-the-high-tide",
        "Measuring the time from the high tide instead of from the low tide.",
        kind="near_miss",
        area="time",
    ),
    CatalogueEntry(
        "forgot-to-subtract-the-low-height",
        "Dividing the target height by the rate, without taking off the height at low tide first.",
        area="time",
    ),
    CatalogueEntry(
        "rounded-the-duration-up-to-whole-hours",
        "Rounding the time up to a whole number of hours before finding the rate.",
        area="time",
    ),
    CatalogueEntry(
        "borrowed-hour-wrongly",
        "Borrowing an hour twice, or when the minutes did not need it, when subtracting clock times.",
        area="time",
    ),
    CatalogueEntry(
        "fraction-of-an-hour-for-fraction-of-the-duration",
        "Taking the fraction of the rise as a fraction of one hour instead of a fraction of the whole time "
        "between the tides.",
        area="time",
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
