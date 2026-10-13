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
    # -- first-year calculus ----------------------------------------------------------------------------------------
    CatalogueEntry(
        "fy1ca-continuity-sign-slip",
        "Moving a term across the equals sign without changing its sign when matching the values of two pieces, so the constant comes out with the wrong sign or size.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-continuity-constant-not-divided",
        "Matching the two pieces correctly but forgetting to divide by the junction's x-value to isolate the slope parameter.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-continuity-gradients-matched",
        "Equating the gradients (derivatives) of the two pieces instead of their values: smoothness is not the same as continuity.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-continuity-one-piece-only",
        "Evaluating only one piece at the junction and reporting that value as the parameter, so the other piece is never matched.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-continuity-squared-as-doubled",
        "Reading c squared as 2c when evaluating a piece at the junction.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-continuity-first-constant-ignored",
        "Solving the second junction while assuming the unknown constant of the middle piece is 1, instead of first finding it from the first junction.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-continuity-first-constant-slip",
        "Finding the first unknown constant with a sign slip and carrying the wrong value into the second junction.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-continuity-wrong-piece",
        "Evaluating a function at a point using the piece that applies on the other side of the junction.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-continuity-constant-dropped",
        "Dropping the constant term of a linear piece when evaluating it.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-continuity-value-at-junction",
        "Evaluating a piece at the junction when the question asked for its value somewhere else.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-limit-by-substitution",
        "Substituting the junction into a 0/0 quotient and reporting 0, instead of cancelling or using a standard limit.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-limit-cancel-slip",
        "Cancelling a common factor correctly but then reporting the wrong part of what is left (the constant of the other factor, or its square).",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-limit-power-dropped",
        "Taking the limit of x squared plus a constant as the constant alone, dropping the squared term.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-limit-equals-value",
        "Treating the stated value of the function at the point as its limit there, when the limit is a separate quantity.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-standard-limit-factor-dropped",
        "Quoting the standard limit sin(u)/u or (e^u - 1)/u as 1 without the factor needed when the argument is mx.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-standard-limit-inverted",
        "Dividing by m instead of multiplying when taking the limit of sin(mx)/x or (e^(mx) - 1)/x.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-removable-called-jump",
        "Calling a break a jump discontinuity because the value at the point differs from the limit, although both one-sided limits agree.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-removable-called-continuous",
        "Concluding there is no break because the one-sided limits agree, without checking the value at the point.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-removable-called-infinite",
        "Calling a break an infinite discontinuity when the function is bounded near the point.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-ivt-wrong-end",
        "Giving the upper integer of the interval that holds the root when the lower one was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-ivt-one-step-away",
        "Giving an integer that is not one of the two either side of the root, from stopping the sign search too early or too late.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-ivt-value-at-wrong-point",
        "Evaluating the function at an end of the interval when the midpoint was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-ivt-linear-term-slip",
        "Evaluating bx at the midpoint as b, forgetting to multiply by the midpoint.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-ivt-cube-as-triple",
        "Reading the cube of the midpoint as three times it.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-ivt-sign-slip",
        "Adding the constant term where it should be subtracted when evaluating the function.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-bisection-wrong-half",
        "Keeping the half of the interval that does not contain the sign change, from reading the sign of f at the midpoint the wrong way.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-bisection-wrong-end",
        "Giving the other endpoint of the new interval than the one asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-bisection-midpoint-reported",
        "Reporting the midpoint just computed instead of the endpoint of the interval kept.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-bisection-no-update",
        "Reporting the original endpoint, as if the bisection step had not changed the interval.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-bisection-stops-early",
        "Stopping one bisection step early and reporting that interval.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-bisection-off-by-one",
        "Finding the number of halvings correctly but rounding it down or up by one, so the interval is not yet narrow enough or one step too many is counted.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-bisection-count-divided",
        "Dividing the interval width by the tolerance instead of halving repeatedly until the width is small enough.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-bisection-log-base-ten",
        "Counting digits of the ratio of width to tolerance (a base-ten logarithm) instead of halvings (a base-two logarithm).",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-bisection-two-steps",
        "Carrying on to a second bisection step and reporting its midpoint, when only one step was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rates-variable-dropped",
        "Differentiating the formula but losing the radius factor, so the rate does not depend on how large the shape is.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rates-chain-rule-forgotten",
        "Differentiating with respect to the radius only and leaving out the rate of change of the radius.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rates-power-rule-slip",
        "Differentiating a power of the variable incorrectly (the exponent not brought down, or the new exponent wrong).",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rates-formula-not-differentiated",
        "Substituting the values into the original formula and multiplying by the rate, instead of differentiating first.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rates-wrong-coefficient",
        "Differentiating correctly but keeping a constant that should have cancelled, or cancelling one that should stay.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rates-ratio-inverted",
        "Inverting the ratio of the two rates when relating them.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rates-rates-assumed-equal",
        "Assuming two linked quantities change at the same rate, when their rates are related by a ratio that depends on the position.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rates-wrong-length",
        "Dividing by the hypotenuse instead of the side that the rate belongs to.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rates-sign-for-speed",
        "Giving the signed rate of change when a speed (positive) was asked for, or a speed when the signed rate was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rates-direction-sign",
        "Treating a vehicle moving towards the junction as moving away, so the sign of its contribution is wrong.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rates-division-dropped",
        "Forgetting to divide by the distance when converting 2s ds/dt back to ds/dt.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rates-rates-combined",
        "Combining the two vehicles' speeds directly instead of resolving them along the line joining the vehicles.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rates-components-swapped",
        "Pairing each speed with the other vehicle's coordinate when applying the rate formula.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rates-shadow-tip-and-length-confused",
        "Giving the speed of the tip of the shadow when its length was asked for, or the reverse.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rates-similar-triangles-sum",
        "Setting up the similar triangles with the sum of the heights instead of the difference.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rates-ratio-not-squared",
        "Using the ratio of radius to height without squaring it in the cone's volume.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rates-cone-third-forgotten",
        "Leaving the factor of one third out of (or putting it into) the volume of a cone when relating the rates.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rates-radius-equals-height",
        "Taking the cone's radius equal to its height, ignoring the stated ratio.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rates-formula-confusion",
        "Using the surface area of a sphere where the volume was needed.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rates-sphere-third-forgotten",
        "Dropping the factor of one third from the volume of a sphere.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rates-surface-differentiated",
        "Differentiating the surface area formula when the volume's rate was needed.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rates-wrong-derivative-used",
        "Dividing the rate of volume by the volume instead of by its derivative with respect to the size.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rates-only-the-first-volume",
        "Dividing the volume at the starting size by the rate, as if the time were to build the pile from nothing.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rates-volume-added-not-subtracted",
        "Taking the whole volume at the larger size instead of the volume added between the two sizes.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rates-linear-scaling",
        "Assuming volume doubles when the size doubles.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rates-area-scaling",
        "Assuming volume quadruples when the size doubles, as an area would.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-opt-all-sides-fenced",
        "Treating all four sides of the rectangle as fenced when one side is a creek or wall.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-opt-area-not-squared",
        "Dividing the volume by the side instead of the side squared when finding the height.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-opt-closed-top-counted",
        "Counting a lid on a tank or box that is open at the top.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-opt-cost-at-trial-value",
        "Pricing the area at the trial value instead of at the optimum.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-opt-costs-swapped",
        "Swapping the two unit costs, so each route is charged at the other route's rate.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-opt-derivative-slip",
        "Making an algebra or differentiation slip in the derivative, so the stationary point is wrong.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-opt-endpoint-not-compared",
        "Reporting a stationary point or a single end of the interval without comparing all candidates.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-opt-factor-dropped",
        "Dropping a numerical factor when solving the stationary-point equation for the variable.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-opt-height-inverted",
        "Inverting the relation between height and side when using the fixed volume.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-opt-ignores-constraint",
        "Using the stationary point even though it lies outside the allowed interval.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-opt-land-only",
        "Costing the whole route at the land rate.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-opt-length-not-eliminated",
        "Leaving the second variable in the formula instead of using the constraint to eliminate it.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-opt-one-side-only",
        "Counting the area of one side face instead of all four.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-opt-square-assumed",
        "Assuming the best rectangle is a square without using the constraint.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-opt-stopped-near",
        "Stopping at a nearby whole number instead of solving the stationary-point equation exactly.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-opt-variable-reported",
        "Giving the value of the variable at the optimum when the optimum value was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-opt-volume-for-area",
        "Multiplying instead of adding the faces, giving a volume where an area was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-opt-volume-for-side",
        "Giving the volume where the side length was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-opt-water-only",
        "Costing the whole route at the under-water rate.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-opt-whole-triangle",
        "Giving the area of the whole triangle instead of the best inscribed rectangle.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-opt-wrong-coefficient",
        "Using a wrong coefficient when setting up the area or volume formula.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-opt-wrong-faces",
        "Including the wrong set of faces when finding the surface area.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-opt-wrong-root",
        "Taking the wrong root (square root for cube root) of the stationary-point equation.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-arc-root-dropped",
        "Integrates 1 + (dy/dx)² without the square root.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-arc-lower-limit-ignored",
        "Evaluates the antiderivative at the upper limit only.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-arc-chord-used",
        "Gives the straight-line distance between the ends instead of the length along the curve.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-arc-height-change-used",
        "Gives the change in y instead of the length along the curve.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-arc-wrong-antiderivative",
        "Sets the length up correctly but makes a slip when integrating.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-arc-cross-term-sign",
        "Expands the square with the wrong sign on the middle term, so the integrand is not the one that appears under the root.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-arc-components-added",
        "Adds dx/dt and dy/dt instead of combining them with Pythagoras.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-arc-log-term-dropped",
        "Drops the logarithmic term when integrating 1/x.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-arc-derivative-sign",
        "Gets the sign of the second term of the derivative wrong.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-arc-power-rule-slip",
        "Differentiates a power of x incorrectly.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-arc-function-not-differentiated",
        "Gives the value of the function where its slope was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-arc-square-dropped",
        "Does not square the slope inside the square root.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-arc-integrand-is-derivative",
        "Gives the slope where the value of the square root was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-arc-sign-slip",
        "Subtracts instead of adds inside the square root.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-arc-chord-ignored",
        "Gives the length of the curve where its excess over the straight line was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-arc-arc-ignored",
        "Gives the straight-line distance where the excess of the curve over it was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-arc-chord-added",
        "Adds the straight-line distance to the length instead of subtracting it.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-arc-ratio-for-difference",
        "Divides the two lengths where their difference was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-shell-radius-forgotten",
        "Integrates only the height of the shell and leaves out its radius.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-shell-two-forgotten",
        "Uses a factor of pi where 2 pi is needed (the circumference of the shell or of the strip).",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-shell-disc-formula",
        "Applies the disc or washer formula to a rotation that needs shells.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-shell-pi-forgotten",
        "Leaves the factor of pi out of the answer.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-shell-height-not-difference",
        "Uses the upper curve alone as the height of the shell instead of the gap between the curves.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-shell-axis-ignored",
        "Takes the radius of the shell as x when the axis of rotation is a different vertical line.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-shell-y-coordinate-given",
        "Gives the y-coordinate of the intersection point where its x-coordinate was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-shell-intersection-slip",
        "Makes an algebra slip when solving for the point where the curves meet.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-shell-height-sign",
        "Subtracts the upper curve from the lower, giving a negative height.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-shell-height-slip",
        "Subtracts the radius from the slope instead of finding the gap between the curves at that radius.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-shell-units-grams",
        "Reports the mass in grams where kilograms were asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-shell-density-divided",
        "Divides the volume by the density instead of multiplying.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-shell-density-forgotten",
        "Converts the volume to litres or kilograms without using the density.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-shell-units-slip",
        "Converts grams to kilograms by dividing by 100 instead of 1000.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-surface-root-dropped",
        "Integrates 2 pi y without the arc-length factor.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-surface-radius-dropped",
        "Integrates the arc-length factor alone, leaving out the radius y.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-surface-two-forgotten",
        "Uses pi where 2 pi is needed in the surface-area formula.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-surface-volume-formula",
        "Uses the volume integral of pi y squared for a surface area.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-surface-arc-length-only",
        "Gives the arc length instead of the area of the surface.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-surface-lower-limit-ignored",
        "Evaluates the antiderivative at the upper limit only.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-surface-wrong-antiderivative",
        "Sets the integral up correctly but makes a slip when integrating.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-surface-quarter-dropped",
        "Drops the 1/4 under the square root, integrating the square root of x + 1.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-surface-whole-sphere",
        "Gives the area of a whole sphere or hemisphere instead of the zone.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-surface-endpoint-sum",
        "Uses the sum of the limits where their difference is needed.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-avg-division-forgotten",
        "Gives the integral without dividing by the length of the interval.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-avg-divide-by-upper-limit",
        "Divides by the upper limit instead of the length of the interval.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-avg-endpoint-mean",
        "Averages the values of the function at the two ends instead of integrating.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-avg-midpoint-value",
        "Gives the value of the function at the midpoint of the interval.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-avg-multiplied",
        "Multiplies by the length of the interval instead of dividing.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-avg-lower-limit-ignored",
        "Evaluates the antiderivative at the upper limit only.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-avg-wrong-antiderivative",
        "Sets up the average correctly but integrates incorrectly.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-avg-power-rule-slip",
        "Makes a slip applying the power rule when integrating.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-avg-no-antiderivative",
        "Evaluates the function at the limits instead of its antiderivative.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-avg-substitution-factor-dropped",
        "Drops the factor of 1/2 that comes from the substitution u = x squared.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-avg-substitution-factor-slip",
        "Uses the wrong constant factor after a substitution.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-avg-chain-rule-misapplied",
        "Integrates a composite function as though it were a product of the pieces.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-avg-divide-by-sum",
        "Divides by the sum of the limits instead of their difference.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-avg-percentage-of-wrong-base",
        "Finds the excess as a percentage of the peak instead of the average.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-avg-difference-not-percentage",
        "Gives the difference, or the ratio, where a percentage was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-avg-percentage-sign",
        "Subtracts in the wrong order, giving a negative percentage.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-mvt-midpoint-assumed",
        "Takes c to be the midpoint of the interval.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-mvt-slope-reported",
        "Gives the slope of the chord instead of the value of c.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-mvt-value-for-c",
        "Gives the value of the function or of its square root where c was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-mvt-root-not-squared",
        "Stops at the square root of c and does not square it.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-mvt-geometric-mean",
        "Takes c to be the geometric mean of the ends for a square-root function.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-mvt-algebra-slip",
        "Makes an algebra slip when solving f'(c) = slope.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-mvt-root-not-taken",
        "Stops at c squared and does not take the square root.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-mvt-negative-root",
        "Gives the negative square root, which lies outside the interval.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rolle-endpoint-root",
        "Gives the root of the derivative that is an end of the interval instead of the one inside it.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rolle-wrong-factor",
        "Solves the wrong factor of the derivative, giving the other root.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-rolle-algebra-slip",
        "Makes an algebra slip when solving f'(c) = 0.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-half-forgotten",
        "Leaves out the factor of 1/2 in the definition of sinh or cosh.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-half-doubled",
        "Halves twice, dividing by 4 instead of 2.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-sinh-cosh-swapped",
        "Uses the definition of cosh where sinh was needed, or the reverse.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-negative-exponent",
        "Treats e to the power of minus ln k as k instead of 1/k.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-tanh-inverted",
        "Inverts tanh, giving cosh over sinh.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-squares-forgotten",
        "Forgets to square e^x and e^(-x) when simplifying the tanh quotient.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-sinh-for-tanh",
        "Gives sinh where tanh was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-denominator-slip",
        "Makes a slip in the denominator of the tanh quotient.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-root-forgotten",
        "Gives cosh squared minus 1 without taking the square root.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-sign-of-sinh",
        "Takes the negative root for sinh x although x is positive.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-linear-slip",
        "Subtracts 1 from cosh x instead of using the identity.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-reciprocal",
        "Gives the reciprocal of cosh x.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-product-for-ratio",
        "Multiplies sinh and cosh where their ratio was needed.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-chain-rule-forgotten",
        "Leaves out the factor from the chain rule.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-derivative-sign",
        "Borrows the minus sign from the derivative of cosine, so the derivative of cosh or tanh has the wrong sign.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-derivative-unchanged",
        "Differentiates sinh to sinh or cosh to cosh.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-chain-divided",
        "Divides by the inner coefficient instead of multiplying.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-product-rule-forgotten",
        "Differentiates only one factor of a product.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-product-rule-sign",
        "Subtracts the two terms of the product rule.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-log-derivative-slip",
        "Differentiates ln(cosh x) as 1 over cosh x.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-sech-squared-slip",
        "Differentiates tanh with the power of the denominator dropped.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-tanh-derivative-confused",
        "Gives the derivative of coth where the derivative of tanh was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-identity-sign",
        "Uses cosh squared plus sinh squared equals one, with the sign of the identity wrong.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-identity-misapplied",
        "Uses cosh squared minus sinh squared in place of the expression that was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-exponential-not-logged",
        "Gives e^x instead of x, forgetting to take the logarithm.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-ln-of-rhs",
        "Takes the natural logarithm of the right-hand side directly.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-ln-of-cosh",
        "Takes the natural logarithm of cosh x as though it were e^x.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-ln-of-sinh",
        "Takes the natural logarithm of sinh x as though it were e^x.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-negative-root",
        "Takes the root of the quadratic in e^x that gives a negative value of x.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-other-root",
        "Gives the smaller solution where the larger was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-coefficient-slip",
        "Divides by the coefficient of cosh and ignores the sinh term.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-sum-of-roots",
        "Gives the logarithm of the sum of the roots instead of the larger root.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-antiderivative-slip",
        "Uses a wrong antiderivative for the hyperbolic function.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-chain-divided-forgotten",
        "Forgets to divide by the inner coefficient when integrating.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-value-for-integral",
        "Gives the product of the two function values where the integral was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-term-dropped",
        "Drops one of the two terms of the antiderivative.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-power-rule-slip",
        "Applies the power rule to a hyperbolic function.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-double-angle-confusion",
        "Gives sinh 2x where cosh 2x was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-double-angle-linear",
        "Treats cosh 2x as twice cosh x or as 1 plus twice sinh x.",
        area="calculus",
    ),
    CatalogueEntry(
        "fy1ca-hyp-double-angle-constant-dropped",
        "Forgets to subtract 1 in cosh 2x = 2cosh squared x - 1.",
        area="calculus",
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
    # -- inequalities and matrix inverses -------------------------------------------------------------------------
    CatalogueEntry(
        "ineq-direction-not-reversed",
        "Multiplying or dividing both sides of an inequality by a negative number without reversing the inequality "
        "sign.",
        area="inequalities",
    ),
    CatalogueEntry(
        "ineq-reversed-without-need",
        "Reversing the inequality sign when nothing needed it: after adding, subtracting, or dividing by a positive "
        "number.",
        area="inequalities",
    ),
    CatalogueEntry(
        "strict-nonstrict-confused",
        "Changing between < and ≤ (or > and ≥) while solving, so the bound is wrongly included or excluded.",
        area="inequalities",
    ),
    CatalogueEntry(
        "used-wrong-inverse-operation",
        "Undoing a multiplication by multiplying again, or a division by dividing again, instead of using the "
        "inverse operation.",
        area="equations",
    ),
    CatalogueEntry(
        "operation-on-one-term-only",
        "Dividing, multiplying or expanding with only one term of a side: 2x + 6 > 10 becomes x + 6 > 5, or "
        "3(x + 2) is treated as 3x + 2.",
        area="equations",
    ),
    CatalogueEntry(
        "bound-sign-not-changed",
        "Reversing the inequality for a negative divisor but leaving the bound with the sign it had before dividing.",
        area="inequalities",
    ),
    CatalogueEntry(
        "compound-operation-missed-part",
        "In a double inequality, applying a step to the middle and one end but not to the other end.",
        area="inequalities",
    ),
    CatalogueEntry(
        "at-least-no-more-than-swapped",
        "Reading the direction of a context the wrong way: writing ≤ for 'at least' and ≥ for 'no more than', or "
        "the dearer option where the cheaper was asked for.",
        area="inequalities",
    ),
    CatalogueEntry(
        "rounds-wrong-way-in-context",
        "Rounding a bound to the wrong whole number for a limit: rounding up a maximum that must not be exceeded, "
        "or down a minimum that must be reached.",
        area="inequalities",
    ),
    CatalogueEntry(
        "strict-bound-taken-as-value",
        "Giving the bound itself as the greatest or least whole number although a strict inequality excludes it.",
        area="inequalities",
    ),
    CatalogueEntry(
        "integer-solutions-endpoint-miscount",
        "Miscounting the whole numbers in a range: including an excluded end, leaving out an included one, or "
        "subtracting the ends without adding one.",
        area="inequalities",
    ),
    CatalogueEntry(
        "compound-bound-omitted",
        "Combining two conditions on the same variable but giving only one of them.",
        area="inequalities",
    ),
    CatalogueEntry(
        "inverse-diagonal-not-swapped",
        "Inverting a 2×2 matrix without swapping the leading-diagonal entries a and d.",
        area="matrices",
    ),
    CatalogueEntry(
        "inverse-offdiag-not-negated",
        "Inverting a 2×2 matrix without changing the signs of the off-diagonal entries b and c.",
        area="matrices",
    ),
    CatalogueEntry(
        "inverse-not-divided-by-det",
        "Writing the adjugate (or cofactor matrix) as the inverse without dividing by the determinant.",
        area="matrices",
    ),
    CatalogueEntry(
        "inverse-entries-reciprocated",
        "Inverting a matrix by taking the reciprocal of each entry.",
        area="matrices",
    ),
    CatalogueEntry(
        "inverse-cofactors-not-transposed",
        "Dividing the cofactor matrix by the determinant without transposing it first.",
        area="matrices",
    ),
    CatalogueEntry(
        "inverse-cofactor-signs-missed",
        "Using minors in place of cofactors: leaving out the alternating + − + sign pattern.",
        area="matrices",
    ),
    CatalogueEntry(
        "inverse-global-sign",
        "A sign error across the whole inverse: every entry has the wrong sign.",
        area="matrices",
    ),
    CatalogueEntry(
        "matrix-equation-wrong-side",
        "Multiplying by the inverse on the wrong side: A⁻¹B where BA⁻¹ is needed, or the reverse. Matrix "
        "multiplication is not commutative.",
        area="matrices",
    ),
    CatalogueEntry(
        "det-formula-sign-slip",
        "Computing a 2×2 determinant as ad + bc, or as bc − ad.",
        area="matrices",
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
