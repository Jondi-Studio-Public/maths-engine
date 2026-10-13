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
