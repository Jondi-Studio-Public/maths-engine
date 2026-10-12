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
    # -- r3e exam chains ----------------------------------------------------------------------------------------
    CatalogueEntry(
        "inverse-trig-sign-flip",
        "Differentiating arctan with a minus sign (as for arccos), or arcsin and arccos with the signs swapped.",
        area="calculus",
    ),
    CatalogueEntry(
        "dropped-inner-constant-in-formula",
        "Dropping the constant inside the bracket when it is put into the derivative formula: using (kx)^2 where (kx + m)^2 belongs.",
        area="calculus",
    ),
    CatalogueEntry(
        "arctan-derivative-missing-square",
        "Writing the derivative of arctan u as u'/(1 + u) with the inside function not squared.",
        area="calculus",
    ),
    CatalogueEntry(
        "dropped-linear-term-in-derivative",
        "Leaving out the derivative of an added linear term when differentiating a sum.",
        area="calculus",
    ),
    CatalogueEntry(
        "inverse-trig-value-in-degrees",
        "Using an angle in degrees (arctan 1 = 45) where radians are needed.",
        area="calculus",
    ),
    CatalogueEntry(
        "inverse-trig-exact-value-wrong",
        "Quoting the wrong exact angle for an inverse trig value, such as arctan 1 = pi/2.",
        area="calculus",
    ),
    CatalogueEntry(
        "x-intercept-sign-slip-on-quotient",
        "Adding f(x0)/m to x0 where it is subtracted when finding where the tangent meets the x-axis.",
        area="calculus",
    ),
    CatalogueEntry(
        "x-intercept-shift-forgotten",
        "Giving -f(x0)/m, as if the tangent passed through the origin, without the shift by x0.",
        area="calculus",
    ),
    CatalogueEntry(
        "multiplied-by-gradient-not-divided",
        "Multiplying f(x0) by the gradient where it is divided by it to find the tangent's x-intercept.",
        area="calculus",
    ),
    CatalogueEntry(
        "multiplied-by-denominator-not-conjugate",
        "Multiplying a quotient's numerator by the denominator itself instead of its conjugate.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "sign-error-in-imaginary-numerator",
        "Getting the imaginary part of a quotient's numerator the wrong way round: ad - bc instead of bc - ad.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "imaginary-part-of-square-as-real",
        "Giving 2xy, the imaginary part of z^2, where the real part of z^2 was asked for.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "scalar-applied-to-first-vector-only",
        "Multiplying only one of the two vectors by its scalar in a combination such as 4p - 3q.",
        area="vectors",
    ),
    CatalogueEntry(
        "combination-sign-slip",
        "Adding where a combination of vectors subtracts: reading 4p - 3q as 4p + 3q.",
        area="vectors",
    ),
    CatalogueEntry(
        "solved-for-scalar-with-wrong-sign",
        "Solving k p1 + w1 = 0 and reporting k with the wrong sign.",
        area="vectors",
    ),
    CatalogueEntry(
        "used-other-component-of-vector",
        "Reading the other component of a vector (j for i, or y for x, along the other axis) than the one asked for.",
        area="vectors",
    ),
    CatalogueEntry(
        "scalar-left-off-component",
        "Leaving the scalar multiple out when finding a component of k p + w.",
        area="vectors",
    ),
    CatalogueEntry(
        "scalar-added-not-multiplied",
        "Adding the scalar to a component where it should multiply it.",
        area="vectors",
    ),
    CatalogueEntry(
        "reversed-vector-direction",
        "Using a vector the wrong way round: adding where subtracting is needed, or taking the vector from B to A as the one from A to B.",
        area="vectors",
    ),
    CatalogueEntry(
        "vector-component-as-coordinate",
        "Giving a vector's component where the coordinate of the point it reaches was asked for, without adding the starting point.",
        area="vectors",
    ),
    CatalogueEntry(
        "coordinate-as-vector-component",
        "Giving a point's coordinate where the component of a vector between points was asked for.",
        area="vectors",
    ),
    CatalogueEntry(
        "midpoint-without-start-point",
        "Halving the vector to the far end and reporting that as the midpoint's coordinate, without adding the start point.",
        area="vectors",
    ),
    CatalogueEntry(
        "endpoint-for-midpoint",
        "Giving the end point's coordinate where the midpoint was asked for.",
        area="vectors",
    ),
    CatalogueEntry(
        "reduced-columns-instead-of-rows",
        "Subtracting the column minima first and totalling those, when the row minima were asked for.",
        area="networks",
    ),
    CatalogueEntry(
        "stopped-after-row-reduction",
        "Stopping after the row reduction and giving its total as the minimum cost, with the column reduction still to come.",
        area="networks",
    ),
    CatalogueEntry(
        "column-minima-from-unreduced-matrix",
        "Taking the column minima from the original costs instead of from the matrix left after the row reduction.",
        area="networks",
    ),
    CatalogueEntry(
        "diagonal-total-as-optimum",
        "Giving the cost of the first-come assignment (worker 1 to the first task, and so on) as the minimum, or as the saving the optimum makes over it.",
        area="networks",
    ),
    CatalogueEntry(
        "permutation-replaced-by-combination",
        "Dividing by k! because the order of the choices was thought not to matter, when the order of a code does.",
        area="counting",
    ),
    CatalogueEntry(
        "repeats-allowed-used-for-no-repeats",
        "Using N^k, which allows repeats, where no repeats were allowed.",
        area="counting",
    ),
    CatalogueEntry(
        "base-and-exponent-swapped",
        "Writing k^N for N^k: the number of choices and the number of positions swapped.",
        area="counting",
    ),
    CatalogueEntry(
        "choices-not-reduced-after-use",
        "Counting N choices at every position of a code with no repeats, instead of one fewer each time.",
        area="counting",
    ),
    CatalogueEntry(
        "forgot-doubling-at-centre",
        "Giving the angle at the circumference for the angle at the centre, without doubling.",
        area="circle-geometry",
    ),
    CatalogueEntry(
        "subtracted-base-angles-from-180",
        "Subtracting the sum of the base angles of two isosceles triangles from 180 where the angle at the circumference is the sum itself.",
        area="circle-geometry",
    ),
    CatalogueEntry(
        "reflex-angle-at-centre",
        "Giving the reflex angle (360 minus the angle at the centre) where the non-reflex angle was asked for.",
        area="circle-geometry",
    ),
    CatalogueEntry(
        "isosceles-base-angle-not-halved",
        "Finding the apex of an isosceles triangle as 180 minus the angle and forgetting to halve it for the base angle.",
        area="circle-geometry",
    ),
    CatalogueEntry(
        "angle-subtracted-not-added",
        "Subtracting two adjacent angles where they add to give the whole angle.",
        area="circle-geometry",
    ),
    CatalogueEntry(
        "scale-power-of-ten-slip",
        "Converting a map distance with a factor of ten wrong: centimetres, metres and kilometres mixed up.",
        area="measurement",
    ),
    CatalogueEntry(
        "used-first-leg-only",
        "Using only one leg of a route where the whole route was asked for.",
        area="measurement",
    ),
    CatalogueEntry(
        "speed-time-inverted",
        "Multiplying the distance by the speed where it should be divided by it to find the time.",
        area="measurement",
    ),
    CatalogueEntry(
        "decimal-hours-read-as-minutes",
        "Reading 1.75 hours as 1 hour 75 minutes, or as 175 minutes.",
        area="measurement",
    ),
    CatalogueEntry(
        "forgot-half-in-power-reduction",
        "Using sin^2 or cos^2 as 1 -/+ cos(2x) without the factor of one half.",
        area="integration",
    ),
    CatalogueEntry(
        "power-reduction-sign-swapped",
        "Swapping the signs of the power-reduction identities: cos^2 as (1 - cos 2x)/2 or sin^2 as (1 + cos 2x)/2.",
        area="integration",
    ),
    CatalogueEntry(
        "trig-antiderivative-sign-slip",
        "Integrating cos as -sin, or sin as +cos.",
        area="integration",
    ),
    CatalogueEntry(
        "dropped-trig-term-at-limit",
        "Treating the sine or cosine term as zero at a limit where it is not (or the reverse).",
        area="integration",
    ),
    CatalogueEntry(
        "measure-of-complement-instead-of-set",
        "Giving the measure of the complement of the set where the measure of the set was asked for.",
        area="measure-theory",
    ),
    CatalogueEntry(
        "used-total-space-measure",
        "Using the measure of the whole space where the measure of the subset was asked for.",
        area="measure-theory",
    ),
    CatalogueEntry(
        "counted-too-few-atoms",
        "Adding up only some of the point masses in the set.",
        area="measure-theory",
    ),
    CatalogueEntry(
        "added-measures-instead-of-multiplying",
        "Adding the two measures instead of multiplying them to get the product measure of a rectangle.",
        area="measure-theory",
    ),
    CatalogueEntry(
        "complement-of-rectangle-as-rectangle-of-complements",
        "Taking the product measure of the complement of A x B to be mu(A^c) x nu(B^c), which misses the two strips.",
        area="measure-theory",
    ),
    CatalogueEntry(
        "rectangle-measure-for-complement",
        "Giving the measure of the rectangle where the measure of its complement was asked for.",
        area="measure-theory",
    ),
    CatalogueEntry(
        "fraction-numerator-ignored",
        "Finding 3/8 of an amount as the amount divided by 8, forgetting to multiply by 3.",
        area="percentages",
    ),
    CatalogueEntry(
        "fraction-inverted-in-part-of",
        "Finding a/b of an amount by multiplying by b/a.",
        area="percentages",
    ),
    CatalogueEntry(
        "fraction-rounded-to-two-places",
        "Turning a fraction such as 3/8 into the rounded decimal 0.38 before using it.",
        area="percentages",
    ),
    CatalogueEntry(
        "discount-added-not-subtracted",
        "Adding a discount to the price instead of taking it off.",
        area="percentages",
    ),
    CatalogueEntry(
        "discount-left-out",
        "Leaving the discount out of the subtotal.",
        area="percentages",
    ),
    CatalogueEntry(
        "percentage-of-wrong-base",
        "Taking a percentage of the wrong amount, such as GST on the base price alone instead of on the whole subtotal.",
        area="percentages",
    ),
    CatalogueEntry(
        "gst-subtracted-not-added",
        "Taking GST off the subtotal where it is added on to give the total.",
        area="percentages",
    ),
    # -- r3c exam chains ----------------------------------------------------------------------------------------
    CatalogueEntry(
        "half-angle-sign-swapped",
        "Using the half-angle identity with the wrong sign: sin^2 as (1 + cos 2t)/2, or cos^2 as (1 - cos 2t)/2.",
        area="calculus",
    ),
    CatalogueEntry(
        "half-angle-factor-dropped",
        "Dropping the factor of one half from the half-angle identity, so sin^2 t becomes 1 - cos 2t.",
        area="calculus",
    ),
    CatalogueEntry(
        "half-angle-halved-twice",
        "Halving twice in the half-angle identity, so sin^2 t becomes (1 - cos 2t)/4.",
        area="calculus",
    ),
    CatalogueEntry(
        "integral-inner-coefficient-not-divided",
        "Integrating cos(kx) or sin(kx), including double angles, as if there were no division by the inside coefficient k, or dividing by the wrong coefficient.",
        area="calculus",
    ),
    CatalogueEntry(
        "integral-inner-coefficient-multiplied",
        "Multiplying by the inner coefficient where integrating divides by it.",
        area="calculus",
    ),
    CatalogueEntry(
        "integral-limit-sign-slip",
        "Getting the sign of a trigonometric value wrong at a limit, or subtracting the limits the wrong way round.",
        area="calculus",
    ),
    CatalogueEntry(
        "dropped-oscillating-term",
        "Integrating only the constant term of a rewritten function and leaving out the cosine term.",
        area="calculus",
    ),
    CatalogueEntry(
        "i-squared-as-plus-one",
        "Treating i^2 as +1 instead of -1 when multiplying or squaring complex numbers, so (x + iy)^2 gives x^2 + y^2 for its real part.",
        area="complex numbers",
    ),
    CatalogueEntry(
        "reversed-subtraction",
        "Subtracting in the wrong order, so a difference, saving, float or vector multiplier comes out with the opposite sign.",
        area="general",
    ),
    CatalogueEntry(
        "dropped-i-squared-term",
        "Leaving the (bi)^2 term out of the real part when squaring a complex number.",
        area="complex numbers",
    ),
    CatalogueEntry(
        "complex-product-sign-slip",
        "Getting the sign of the imaginary part, or of a cross term, wrong when multiplying two complex numbers.",
        area="complex numbers",
    ),
    CatalogueEntry(
        "i-cubed-as-plus-i",
        "Treating i^3 as +i instead of -i.",
        area="complex numbers",
    ),
    CatalogueEntry(
        "dropped-i-cubed-term",
        "Leaving the b^3 term out of the imaginary part of a cube.",
        area="complex numbers",
    ),
    CatalogueEntry(
        "magnitude-as-sum-of-components",
        "Taking the magnitude of a vector as the sum of the sizes of its components.",
        area="vectors",
    ),
    CatalogueEntry(
        "magnitude-without-square-root",
        "Giving the sum of the squares of the components as the magnitude, with no square root.",
        area="vectors",
    ),
    CatalogueEntry(
        "magnitude-as-one-leg",
        "Giving one component of a vector as its magnitude.",
        area="vectors",
    ),
    CatalogueEntry(
        "forgot-to-normalise",
        "Giving the raw component where the component of the unit vector (the component divided by the magnitude) was asked for.",
        area="vectors",
    ),
    CatalogueEntry(
        "unit-component-sign-slip",
        "Getting the sign of a component of a unit vector wrong.",
        area="vectors",
    ),
    CatalogueEntry(
        "unit-vector-not-rotated",
        "Using the components of the original unit vector where the perpendicular unit vector was needed.",
        area="vectors",
    ),
    CatalogueEntry(
        "wrong-rotation-direction",
        "Rotating a vector through 90 degrees the wrong way, so the perpendicular points in the opposite direction.",
        area="vectors",
    ),
    CatalogueEntry(
        "forgot-to-scale-unit-vector",
        "Giving the component of the unit vector where a vector of stated length was asked for.",
        area="vectors",
    ),
    CatalogueEntry(
        "started-from-wrong-point",
        "Adding a displacement to the starting point of a journey instead of the point it starts from.",
        area="vectors",
    ),
    CatalogueEntry(
        "forgot-scalar-multiple",
        "Leaving out the scalar multiple when combining vectors, so 2b is used as b.",
        area="vectors",
    ),
    CatalogueEntry(
        "used-second-longest-route",
        "Giving the length of a route that is not the longest as the minimum completion time.",
        area="networks",
    ),
    CatalogueEntry(
        "reports-lst-not-float",
        "Giving an activity's latest start time where its float (latest start minus earliest start) was asked for.",
        area="networks",
    ),
    CatalogueEntry(
        "ignored-float-buffer",
        "Taking the whole overrun as the project's delay, though the activity's float absorbs part of it.",
        area="networks",
    ),
    CatalogueEntry(
        "reported-float-not-extension",
        "Giving the activity's float where the extension to the project was asked for.",
        area="networks",
    ),
    CatalogueEntry(
        "added-float-to-overrun",
        "Adding the float to the overrun, where the float is used up first and the rest delays the project.",
        area="networks",
    ),
    CatalogueEntry(
        "no-delay-assumed",
        "Believing a non-critical activity's overrun cannot affect the completion time at all.",
        area="networks",
    ),
    CatalogueEntry(
        "added-instead-of-multiplied",
        "Adding the two numbers where a rate per unit is applied to a quantity by multiplying.",
        area="measurement",
    ),
    CatalogueEntry(
        "reported-per-dose-for-daily",
        "Giving the amount for one dose where the amount for the whole day was asked for.",
        area="measurement",
    ),
    CatalogueEntry(
        "shared-over-wrong-number-of-doses",
        "Sharing a daily amount over the wrong number of doses, or multiplying by the number of doses.",
        area="measurement",
    ),
    CatalogueEntry(
        "forgot-to-share-over-doses",
        "Giving the whole day's amount as the amount for one dose.",
        area="measurement",
    ),
    CatalogueEntry(
        "tablet-strength-not-converted",
        "Dividing a dose in milligrams by a tablet strength in grams without converting, so the count is a thousand times too large.",
        area="measurement",
    ),
    CatalogueEntry(
        "factor-of-100-error",
        "Using one hundred as the conversion factor between grams and milligrams instead of one thousand.",
        area="measurement",
    ),
    CatalogueEntry(
        "reported-daily-for-per-dose",
        "Giving the number of tablets for the whole day where the number for one dose was asked for.",
        area="measurement",
    ),
    CatalogueEntry(
        "forgot-doses-per-day",
        "Working out how long a supply lasts as if only one dose were taken each day.",
        area="measurement",
    ),
    CatalogueEntry(
        "rounded-up-instead-of-down",
        "Rounding up a count of full days that a supply lasts, so a part day is counted as a full one.",
        area="general",
    ),
    CatalogueEntry(
        "forgot-half-in-triangle-area",
        "Giving the product of the two perpendicular sides as the area of a triangle, with no factor of one half.",
        area="geometry",
    ),
    CatalogueEntry(
        "used-hypotenuse-as-leg",
        "Using the hypotenuse as one of the two perpendicular sides in the area of a right triangle.",
        area="geometry",
    ),
    CatalogueEntry(
        "forgot-doubling-in-altitude",
        "Dividing the area of a triangle by its base to get the height, with no factor of two.",
        area="geometry",
    ),
    CatalogueEntry(
        "used-leg-as-base",
        "Dividing twice the area by a leg where the height to the hypotenuse was asked for.",
        area="geometry",
    ),
    CatalogueEntry(
        "subtracted-sides-not-pythagoras",
        "Subtracting two side lengths of a right triangle instead of using Pythagoras' theorem.",
        area="geometry",
    ),
    CatalogueEntry(
        "pythagoras-without-square-root",
        "Giving the difference of the squares of two sides as the third side, with no square root.",
        area="geometry",
    ),
    CatalogueEntry(
        "used-radius-as-chord",
        "Giving the radius where the length of a chord was asked for.",
        area="geometry",
    ),
    CatalogueEntry(
        "confuses-neither-with-union",
        "Using the whole group in the inclusion-exclusion formula where the number in at least one set belongs, or reporting the number in none.",
        area="sets",
    ),
    CatalogueEntry(
        "ignored-neither-count",
        "Taking everyone in the group as being in at least one set, ignoring those in none.",
        area="sets",
    ),
    CatalogueEntry(
        "reported-total-overlap-as-triple",
        "Giving the total of the overlaps as the number in all three sets.",
        area="sets",
    ),
    CatalogueEntry(
        "subtracts-triple-once",
        "Taking the number in exactly two sets as the pairwise overlaps less the triple overlap once, though it is counted in three overlaps.",
        area="sets",
    ),
    CatalogueEntry(
        "forgets-triple-region",
        "Giving the total of the pairwise overlaps as the number in exactly two sets, without removing those in all three.",
        area="sets",
    ),
    CatalogueEntry(
        "subtracts-triple-twice",
        "Removing the triple overlap twice from the pairwise overlaps where it must be removed three times.",
        area="sets",
    ),
    CatalogueEntry(
        "omits-triple-from-exactly-one",
        "Finding the number in exactly one set by removing those in exactly two but not those in all three.",
        area="sets",
    ),
    CatalogueEntry(
        "single-counted-overlaps",
        "Removing each overlap once from the set sizes to find the number in exactly one set, though each person in an overlap is counted in more than one set.",
        area="sets",
    ),
    CatalogueEntry(
        "inclusion-exclusion-sign-error",
        "Rearranging the three-set inclusion-exclusion formula with the wrong sign on the pairwise overlaps.",
        area="sets",
    ),
    CatalogueEntry(
        "flat-not-reducing-interest",
        "Working out a loan repayment from interest on the original amount every period, rather than on the balance that remains.",
        area="finance",
    ),
    CatalogueEntry(
        "annual-rate-as-periodic",
        "Using the stated annual interest rate as the rate for each period, without dividing it by the number of periods in the year.",
        area="finance",
    ),
    CatalogueEntry(
        "years-not-months",
        "Using the number of years as the number of monthly repayments.",
        area="finance",
    ),
    CatalogueEntry(
        "interest-on-repayment",
        "Charging the month's interest on the repayment rather than on the balance of the loan.",
        area="finance",
    ),
    CatalogueEntry(
        "interest-not-added-to-balance",
        "Taking a repayment off the balance without first adding the month's interest.",
        area="finance",
    ),
    CatalogueEntry(
        "added-repayment-to-balance",
        "Adding the repayment to the balance where it is taken off.",
        area="finance",
    ),
    CatalogueEntry(
        "interest-not-recalculated",
        "Using the first month's interest again in the second month, though the balance has changed.",
        area="finance",
    ),
    CatalogueEntry(
        "interest-on-balance-before-interest",
        "Charging interest on the balance as it stood before the month's interest was added.",
        area="finance",
    ),
    CatalogueEntry(
        "determinant-products-added",
        "Adding the two products in a 2 by 2 determinant where one is subtracted from the other.",
        area="linear algebra",
    ),
    CatalogueEntry(
        "determinant-order-reversed",
        "Subtracting the products of a 2 by 2 determinant the wrong way round, so the determinant has the opposite sign.",
        area="linear algebra",
    ),
    CatalogueEntry(
        "wrong-column-replaced",
        "Replacing the wrong column of the coefficient matrix with the constants in Cramer's rule.",
        area="linear algebra",
    ),
    CatalogueEntry(
        "forgot-to-divide-by-determinant",
        "Giving a determinant with a column replaced as the solution, without dividing by the determinant of the coefficients.",
        area="linear algebra",
    ),
    CatalogueEntry(
        "swapped-x-and-y-values",
        "Substituting the value of x for y and the value of y for x.",
        area="linear algebra",
    ),
    CatalogueEntry(
        "interest-not-credited",
        "Leaving out the interest earned in the period when finding an annuity's balance.",
        area="finance",
    ),
    CatalogueEntry(
        "withdrawal-added-not-subtracted",
        "Adding the withdrawal to the balance where it is taken off.",
        area="finance",
    ),
    CatalogueEntry(
        "off-by-one-over",
        "Counting the withdrawal that could not be paid in full as if it had been, so the count is one too many.",
        area="finance",
    ),
    CatalogueEntry(
        "off-by-one-under",
        "Stopping one withdrawal early, so the count is one too few.",
        area="finance",
    ),
    CatalogueEntry(
        "ignores-compounding-in-count",
        "Dividing the opening balance by the withdrawal to count the withdrawals, ignoring the interest earned.",
        area="finance",
    ),
    CatalogueEntry(
        "balance-before-last-withdrawal",
        "Giving the balance before the last full withdrawal where the balance after it was asked for.",
        area="finance",
    ),
    CatalogueEntry(
        "final-withdrawal-not-taken",
        "Giving the balance as if the last full withdrawal had not been taken.",
        area="finance",
    ),
    CatalogueEntry(
        "ignored-imaginary-part-when-squaring",
        "Squaring only the real part of a complex number: reading Re(z^2) as (Re z)^2.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "scalars-attached-to-wrong-vectors",
        "Attaching each scalar to the wrong vector: reading 4p - 3q as 3p - 4q.",
        area="vectors",
    ),
    CatalogueEntry(
        "second-vector-left-out-of-sum",
        "Leaving the second vector out of a sum: giving k p2 where the j-coefficient of kp + w was asked for.",
        area="vectors",
    ),
    CatalogueEntry(
        "greedy-assignment-total",
        "Assigning each worker in turn to the cheapest task still free and giving that total as the minimum.",
        area="networks",
    ),
    CatalogueEntry(
        "left-out-one-row-or-column",
        "Leaving one row or column out when totalling the minima of a cost table.",
        area="networks",
    ),
    CatalogueEntry(
        "largest-instead-of-smallest",
        "Choosing the largest value where the smallest was needed.",
        area="general",
    ),
    # -- exam chains, prefix r3d ------------------------------------------------------------------------------
    CatalogueEntry(
        "degrees-added-for-quotient",
        "Adding the degrees of the numerator and denominator instead of subtracting them to find how fast a quotient grows.",
        area="limits",
    ),
    CatalogueEntry(
        "factor-degree-undercounted",
        "Counting a powered or squared factor as one degree per bracket, so (x^2 + 1)^2 is read as degree 2 or 1 instead of 4.",
        area="limits",
    ),
    CatalogueEntry(
        "denominator-degree-ignored",
        "Using the degree of the numerator alone as the growth exponent of a quotient.",
        area="limits",
    ),
    CatalogueEntry(
        "leading-coefficient-power-ignored",
        "Taking the leading coefficient of a factor such as (2x - 3)^2 to be 2 rather than 2^2 = 4.",
        area="limits",
    ),
    CatalogueEntry(
        "limit-constant-sign-flipped",
        "Giving a limit constant, or its limit at minus infinity, with the opposite sign.",
        area="limits",
    ),
    CatalogueEntry(
        "denominator-coefficient-ignored",
        "Reading the numerator's leading coefficient as the limit constant, without dividing by the denominator's.",
        area="limits",
    ),
    CatalogueEntry(
        "coefficients-multiplied-not-divided",
        "Multiplying the leading coefficients of the numerator and denominator where the limit constant is their quotient.",
        area="limits",
    ),
    CatalogueEntry(
        "imaginary-terms-dropped-in-product",
        "Multiplying (a + bi)(c + di) by multiplying only the matching parts, so the cross terms or the bd term are left out.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "complex-parts-added-not-multiplied",
        "Adding the matching parts of two complex numbers where their product was asked for.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "moduli-squares-added",
        "Adding the squares of two moduli, |z|^2 + |w|^2, where the modulus of the product |z||w| was asked for.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "position-vectors-dotted-for-displacement",
        "Taking the dot product of the position vectors of the points where the displacement vectors between them were meant.",
        area="vectors",
    ),
    CatalogueEntry(
        "wrong-vector-pair-dotted",
        "Dotting the wrong pair of vectors, such as AB with BC or AB with itself when AB . AC was asked for.",
        area="vectors",
    ),
    CatalogueEntry(
        "length-of-wrong-vector",
        "Giving the length of a different side of the triangle from the one asked for.",
        area="vectors",
    ),
    CatalogueEntry(
        "projection-division-by-length-omitted",
        "Giving the dot product a . b as the scalar projection of a onto b, without dividing by |b|.",
        area="vectors",
    ),
    CatalogueEntry(
        "vector-coefficient-for-scalar-projection",
        "Dividing a . b by |b|^2, which gives the multiple of b in the projection, where the scalar projection divides by |b|.",
        area="vectors",
    ),
    CatalogueEntry(
        "projection-sign-lost",
        "Giving a scalar projection with the opposite sign, as if the component lay the other way along the vector.",
        area="vectors",
    ),
    CatalogueEntry(
        "scalars-k-and-m-confused",
        "Giving the scalar k that makes a vector parallel where the multiplier m in the vector = m e was asked for, or the reverse.",
        area="vectors",
    ),
    CatalogueEntry(
        "parallel-taken-as-equal",
        "Solving 'ku - w is parallel to e' by making ku - w equal to e, which fixes the multiplier as 1.",
        area="vectors",
    ),
    CatalogueEntry(
        "parallel-condition-ignored",
        "Solving ku = w and ignoring the vector e that the result was meant to be parallel to.",
        area="vectors",
    ),
    CatalogueEntry(
        "scalar-not-squared-in-length",
        "Finding the squared length of m e as m |e|^2, with the scalar m not squared.",
        area="vectors",
    ),
    CatalogueEntry(
        "direction-vector-length-ignored",
        "Giving m^2 as the squared length of m e, leaving out the squared length of e.",
        area="vectors",
    ),
    CatalogueEntry(
        "scalar-multiplier-ignored",
        "Giving the squared length of e as the squared length of m e, leaving out the factor m^2.",
        area="vectors",
    ),
    CatalogueEntry(
        "square-of-sum-for-sum-of-squares",
        "Squaring the sum of the components, (p + q)^2, where the squared length is p^2 + q^2.",
        area="vectors",
    ),
    CatalogueEntry(
        "ordered-count-for-selection",
        "Counting the order of the people chosen (permutations) where only the group matters, so each group is counted r! times.",
        area="counting",
    ),
    CatalogueEntry(
        "selection-with-repeats-assumed",
        "Using the count for choosing with repeats, C(n + r - 1, r), for a selection of different people.",
        area="counting",
    ),
    CatalogueEntry(
        "second-group-size-wrong",
        "Choosing the same number from both groups, C(x, j) C(y, j), where the second group supplies the rest of the selection.",
        area="counting",
    ),
    CatalogueEntry(
        "only-one-group-chosen",
        "Choosing the required number from the whole pool, C(n, j), and ignoring how the rest of the selection is made.",
        area="counting",
    ),
    CatalogueEntry(
        "one-excluded-case-missed",
        "Removing only one of the two cases that fail 'at least one of each' (all from one group) from the total.",
        area="counting",
    ),
    CatalogueEntry(
        "complement-counted-as-two-cases",
        "Removing just 2 from the total for 'at least one of each', as if only two selections failed.",
        area="counting",
    ),
    CatalogueEntry(
        "overcounts-by-choosing-required-members-first",
        "Choosing one member of each group first and then the rest, which counts the same selection many times.",
        area="counting",
    ),
    CatalogueEntry(
        "required-members-not-removed-from-pool",
        "Choosing the other members of a selection from the whole pool, C(n, r - 2), without removing the two required members.",
        area="counting",
    ),
    CatalogueEntry(
        "required-members-excluded",
        "Choosing the whole selection from the pool without the two named people, C(n - 2, r), where they must be in it.",
        area="counting",
    ),
    CatalogueEntry(
        "at-least-one-for-both",
        "Counting selections with at least one of the two named people where both must be included.",
        area="counting",
    ),
    CatalogueEntry(
        "only-the-boundary-case-counted",
        "Counting only the case with exactly j from a group where 'at most j' includes every smaller number too.",
        area="counting",
    ),
    CatalogueEntry(
        "upper-case-excluded",
        "Counting 'at most j' as the cases with fewer than j, leaving out the case with exactly j.",
        area="counting",
    ),
    CatalogueEntry(
        "odds-for-probability",
        "Dividing the favourable count by the unfavourable count (the odds) where the probability divides by the total.",
        area="counting",
    ),
    CatalogueEntry(
        "tangent-chord-angle-supplementary",
        "Treating the tangent-chord angle and the angle in the alternate segment as adding to 180 degrees instead of being equal.",
        area="circle-geometry",
    ),
    CatalogueEntry(
        "tangent-chord-angle-complementary",
        "Treating the tangent-chord angle and the angle in the alternate segment as adding to 90 degrees instead of being equal.",
        area="circle-geometry",
    ),
    CatalogueEntry(
        "reports-the-angle-instead-of-x",
        "Giving the size of the angle where the value of the unknown x in its expression was asked for.",
        area="circle-geometry",
    ),
    CatalogueEntry(
        "inscribed-angle-doubled",
        "Doubling an angle that is already the tangent-chord or inscribed angle, as if it were being converted to the angle at the centre.",
        area="circle-geometry",
    ),
    CatalogueEntry(
        "tangent-angles-all-equal",
        "Giving the angle between two tangents as equal to the tangent-chord angle.",
        area="circle-geometry",
    ),
    CatalogueEntry(
        "isosceles-base-angle-not-doubled",
        "Finding the apex of an isosceles triangle as 180 minus one base angle instead of 180 minus both.",
        area="circle-geometry",
    ),
    CatalogueEntry(
        "radius-tangent-right-angle-ignored",
        "Giving the tangent-chord angle where the angle between the radius and the chord was asked for, forgetting the right angle between radius and tangent.",
        area="circle-geometry",
    ),
    CatalogueEntry(
        "radius-tangent-right-angle-added",
        "Adding the tangent-chord angle to the right angle between radius and tangent where it is taken away.",
        area="circle-geometry",
    ),
    CatalogueEntry(
        "tangent-pair-angle-given-for-radius-angle",
        "Giving the angle between the two tangents where the angle between the radius and the chord was asked for.",
        area="circle-geometry",
    ),
    CatalogueEntry(
        "two-period-change-as-rate",
        "Giving the total percentage change over two periods as the percentage change per period.",
        area="percentages",
    ),
    CatalogueEntry(
        "compound-change-divided-by-periods",
        "Dividing a total percentage change by the number of periods to get a rate per period, as if growth were not compounded.",
        area="percentages",
    ),
    CatalogueEntry(
        "adds-percent-to-undo-decrease",
        "Undoing a percentage decrease by adding the same percentage of the final value, though that percentage was of the original.",
        area="percentages",
    ),
    CatalogueEntry(
        "undone-for-too-many-periods",
        "Reversing the growth for more periods than separate the reading from the start.",
        area="percentages",
    ),
    CatalogueEntry(
        "reads-off-the-given-value",
        "Giving a value stated in the question as the answer, such as the first reading as the starting value.",
        area="percentages",
    ),
    CatalogueEntry(
        "simple-growth-for-compound-count",
        "Counting the periods to a target by adding a fixed share of the starting value each period, instead of compounding.",
        area="percentages",
    ),
    CatalogueEntry(
        "threshold-period-off-by-one",
        "Giving the period before or after the one in which the target is first passed.",
        area="percentages",
    ),
    CatalogueEntry(
        "difference-instead-of-ratio",
        "Subtracting two successive terms where the common ratio divides them.",
        area="sequences",
    ),
    CatalogueEntry(
        "complement-of-ratio",
        "Giving the fraction lost each time, 1 - r, where the ratio r that is kept was asked for.",
        area="sequences",
    ),
    CatalogueEntry(
        "arithmetic-instead-of-geometric",
        "Working back along a geometric sequence by adding the common difference instead of dividing by the common ratio.",
        area="sequences",
    ),
    CatalogueEntry(
        "first-rebound-as-drop-height",
        "Treating the height of the first rebound as the height the ball was dropped from.",
        area="sequences",
    ),
    CatalogueEntry(
        "rebound-travel-counted-once",
        "Counting each rebound once, up or down, where the ball travels up and down.",
        area="sequences",
    ),
    CatalogueEntry(
        "initial-drop-counted-twice",
        "Counting the first drop both ways, as if the ball had also gone up before it fell.",
        area="sequences",
    ),
    CatalogueEntry(
        "one-term-too-many-in-series",
        "Adding one more rebound to a finite series than the question stops at.",
        area="sequences",
    ),
    CatalogueEntry(
        "one-term-too-few-in-series",
        "Stopping a finite series one rebound before the question stops.",
        area="sequences",
    ),
    CatalogueEntry(
        "series-sum-formula-inverted",
        "Using (1 - r) / (1 + r) where the sum to infinity uses (1 + r) / (1 - r).",
        area="sequences",
    ),
    CatalogueEntry(
        "power-rule-coefficient-dropped",
        "Differentiating a x^2 as a x, dropping the 2 that the power rule brings down.",
        area="calculus",
    ),
    CatalogueEntry(
        "x-values-added-not-subtracted",
        "Adding the two x-values where the difference of the limits is divided by the difference of the x-values.",
        area="calculus",
    ),
    CatalogueEntry(
        "quadratic-term-ignored",
        "Giving the limit as the value of b, ignoring the 2ax contributed by the x^2 term.",
        area="calculus",
    ),
    CatalogueEntry(
        "wrong-endpoint-for-stated-rule",
        "Using the other endpoint rule (right for left, or left for right) or the endpoint rule where the midpoint rule was asked for.",
        area="integration",
    ),
    CatalogueEntry(
        "strip-width-omitted",
        "Adding up the heights of the rectangles without multiplying by the strip width.",
        area="integration",
    ),
    CatalogueEntry(
        "off-by-one-rectangle",
        "Leaving out the last rectangle, or counting one too many.",
        area="integration",
    ),
    CatalogueEntry(
        "trapezium-for-rectangle-rule",
        "Averaging the left and right sums where a rectangle rule was asked for.",
        area="integration",
    ),
    CatalogueEntry(
        "differentiated-instead-of-integrated",
        "Subtracting the heights f(b) - f(a) where the area is the difference of the antiderivative values.",
        area="integration",
    ),
    CatalogueEntry(
        "power-rule-integration-divisor-omitted",
        "Integrating x^n as x^(n+1) without dividing by n + 1.",
        area="integration",
    ),
    CatalogueEntry(
        "estimate-reported-as-exact",
        "Giving the rectangle-rule estimate as the exact area.",
        area="integration",
    ),
    CatalogueEntry(
        "relative-error-for-absolute-error",
        "Dividing the error by the exact area where the size of the error was asked for.",
        area="integration",
    ),
    CatalogueEntry(
        "error-sign-kept",
        "Giving an error that is negative because the estimate is below the exact value, where its size was asked for.",
        area="integration",
    ),
    CatalogueEntry(
        "error-of-other-estimate",
        "Giving the error of a different estimate (the other endpoint rule, or the trapezium) from the exact area.",
        area="integration",
    ),
    CatalogueEntry(
        "second-feature-ignored",
        "Fitting a slope as if the second feature were not in the model, so each slope is a sum of products over a sum of squares.",
        area="statistics",
    ),
    CatalogueEntry(
        "slopes-swapped",
        "Giving the slope of the other feature, or using the means of the features in the wrong places.",
        area="statistics",
    ),
    CatalogueEntry(
        "coefficient-sign-flipped",
        "Giving a fitted coefficient with the opposite sign.",
        area="statistics",
    ),
    CatalogueEntry(
        "intercept-taken-as-mean-of-y",
        "Giving the mean of y as the intercept, without subtracting the slopes times the feature means.",
        area="statistics",
    ),
    CatalogueEntry(
        "feature-term-dropped-from-intercept",
        "Subtracting only one slope times its mean when finding the intercept.",
        area="statistics",
    ),
    CatalogueEntry(
        "feature-term-dropped-from-prediction",
        "Leaving one feature's slope times its value out of a fitted value.",
        area="statistics",
    ),
    CatalogueEntry(
        "intercept-omitted-in-prediction",
        "Predicting y from the slopes and feature values alone, leaving out the intercept.",
        area="statistics",
    ),
    CatalogueEntry(
        "feature-values-swapped",
        "Putting the value of the first feature where the second slope applies, and the reverse, in a prediction.",
        area="statistics",
    ),
    CatalogueEntry(
        "prediction-by-mean",
        "Giving the mean of y as the prediction at a point, ignoring the features.",
        area="statistics",
    ),
    CatalogueEntry(
        "perpendicular-distance-taken-as-hypotenuse",
        "Giving the squared length |AB|^2 of the whole side as the squared distance from B to the line, without taking away the squared projection.",
        area="vectors",
    ),
    CatalogueEntry(
        "projection-not-squared",
        "Taking away the projection itself, not its square, from |AB|^2 when finding the squared distance from a point to a line.",
        area="vectors",
    ),
    CatalogueEntry(
        "projection-squared-for-perpendicular-distance",
        "Giving the square of the scalar projection where the square of the perpendicular distance was asked for.",
        area="vectors",
    ),
    CatalogueEntry(
        "pythagoras-added-for-leg",
        "Adding the squares of the hypotenuse and the other leg where Pythagoras takes one from the other to find a leg.",
        area="vectors",
    ),
    # -- exam chains, r3a (bearings, right triangles, powers of a complex number, determinants, overtime, networks,
    # -- restricted arrangements, cyclic quadrilaterals, survey probabilities, Euler-Maruyama, definite integrals) --
    CatalogueEntry(
        "turn-angle-for-interior-angle",
        "Using the size of the turn at the corner of a journey as the angle inside the triangle, though the angle "
        "between the two legs is 180 degrees minus the turn.",
        area="geometry",
    ),
    CatalogueEntry(
        "reflex-angle-for-interior-angle",
        "Reading the angle between two bearings the long way round (360 degrees minus the turn) instead of the "
        "angle inside the triangle.",
        area="geometry",
    ),
    CatalogueEntry(
        "turn-angle-added-to-straight-angle",
        "Adding the turn to 180 degrees to get the angle between two legs, instead of subtracting it.",
        area="geometry",
    ),
    CatalogueEntry(
        "bearing-difference-for-interior-angle",
        "Taking the raw difference between two bearings as the angle inside the triangle, without checking "
        "whether it is the turn or its supplement.",
        area="geometry",
    ),
    CatalogueEntry(
        "pythagoras-on-non-right-triangle",
        "Using Pythagoras on a triangle that has no right angle, so the cosine rule's cross term is left out.",
        area="geometry",
    ),
    CatalogueEntry(
        "legs-added-for-pythagoras",
        "Adding the two shorter lengths of a triangle to get the third, instead of applying Pythagoras or the "
        "cosine rule.",
        area="geometry",
    ),
    CatalogueEntry(
        "cosine-rule-sign-error",
        "Adding the 2bc cos A term in the cosine rule instead of subtracting it.",
        area="geometry",
    ),
    CatalogueEntry(
        "angle-applied-wrong-way-to-bearing",
        "Adding a triangle angle to a bearing when the turn goes the other way, or subtracting it when it should "
        "be added.",
        area="geometry",
    ),
    CatalogueEntry(
        "triangle-angle-not-converted-to-bearing",
        "Reporting an angle from inside a triangle as if it were a compass bearing, without adding it to the "
        "reference bearing.",
        area="geometry",
    ),
    CatalogueEntry(
        "back-bearing-for-forward-bearing",
        "Giving the bearing of the start from the finish (180 degrees away) when the bearing of the finish from "
        "the start was asked.",
        kind="near_miss",
        area="geometry",
    ),
    CatalogueEntry(
        "bearing-of-wrong-point",
        "Repeating a bearing that was given in the question as the answer, though it is the bearing from a "
        "different point.",
        kind="near_miss",
        area="geometry",
    ),
    CatalogueEntry(
        "ambiguous-sine-rule-angle",
        "Taking the acute angle from the sine rule when the triangle's angle is obtuse (the supplement).",
        area="geometry",
    ),
    CatalogueEntry(
        "sine-cosine-swapped",
        "Giving the cosine ratio where the sine was asked, or the reverse, so the sides are paired the wrong way "
        "round.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "wrong-ratio-for-sine",
        "Giving the tangent (opposite over adjacent) where the sine (opposite over hypotenuse) was asked.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "reciprocal-ratio",
        "Giving the reciprocal of a trigonometric ratio, with the two sides in the wrong places.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "ratio-multiplied-not-divided",
        "Multiplying a side by a trigonometric ratio when the side is the numerator of that ratio and should be "
        "divided by it.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "other-leg-for-hypotenuse",
        "Reporting the other leg of a right-angled triangle as the hypotenuse.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "legs-subtracted-for-pythagoras",
        "Subtracting the lengths of the hypotenuse and a leg to find the other leg, instead of subtracting their "
        "squares.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "given-side-repeated",
        "Repeating the side length that was given in the question as the length asked for.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "area-for-altitude",
        "Reporting half the product of the legs (the area) as the length of the perpendicular from the right "
        "angle to the hypotenuse.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "altitude-not-divided-by-hypotenuse",
        "Reporting the product of the legs as the perpendicular to the hypotenuse, without dividing by the hypotenuse.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "altitude-ratio-inverted",
        "Inverting the ratio when finding the perpendicular to the hypotenuse, so the hypotenuse is multiplied "
        "instead of divided.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "cross-term-dropped",
        "Dropping the middle term when squaring a binomial, so (a + bi) squared loses its 2abi.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "cross-term-not-doubled",
        "Writing the middle term of a squared complex number as ab (or abi) instead of 2ab (or 2abi).",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "squared-each-part-separately",
        "Squaring the real and imaginary parts separately, so (a + bi) squared becomes a squared plus b squared i, or the imaginary part of the square is given as b^2.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "complex-product-componentwise",
        "Multiplying two complex numbers part by part, real by real and imaginary by imaginary, instead of "
        "multiplying out the brackets.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "power-as-multiplication",
        "Treating a power as repeated addition or a multiple, so z to the fourth becomes 4z or twice z squared.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "determinant-added-not-subtracted",
        "Adding the two products ad and bc instead of subtracting them in a 2 by 2 determinant.",
        area="matrices",
    ),
    CatalogueEntry(
        "determinant-columns-not-diagonals",
        "Multiplying down the columns instead of across the diagonals when finding a 2 by 2 determinant.",
        area="matrices",
    ),
    CatalogueEntry(
        "determinant-sign-reversed",
        "Subtracting the products the wrong way round, bc minus ad, so the determinant has the wrong sign.",
        area="matrices",
    ),
    CatalogueEntry(
        "determinant-one-diagonal-only",
        "Using only the leading diagonal product ad as the determinant.",
        area="matrices",
    ),
    CatalogueEntry(
        "determinants-summed-for-product",
        "Treating the determinant of a product as the sum of the two determinants.",
        area="matrices",
    ),
    CatalogueEntry(
        "determinant-product-rule-multiplied-not-divided",
        "Multiplying by one determinant instead of dividing by it when using det(AB) = det(A) det(B) to find the "
        "other.",
        area="matrices",
    ),
    CatalogueEntry(
        "second-matrix-ignored",
        "Using the determinant of the product as the determinant of one factor, ignoring the other matrix.",
        area="matrices",
    ),
    CatalogueEntry(
        "scalar-multiple-not-squared",
        "Multiplying the determinant of a 2 by 2 matrix by the scalar once instead of squaring the scalar.",
        area="matrices",
    ),
    CatalogueEntry(
        "scalar-multiple-ignored",
        "Leaving the scalar multiple out when finding the determinant of a scaled matrix.",
        area="matrices",
    ),
    CatalogueEntry(
        "inverse-determinant-not-reciprocal",
        "Using det(A) instead of 1 over det(A) for the determinant of the inverse.",
        area="matrices",
    ),
    CatalogueEntry(
        "inverse-ignored-in-determinant",
        "Leaving the inverse out of a determinant calculation, as if det(A inverse) were 1.",
        area="matrices",
    ),
    CatalogueEntry(
        "total-hours-for-overtime-hours",
        "Reporting all the hours worked as the overtime hours.",
        area="financial-maths",
    ),
    CatalogueEntry(
        "overtime-tier-hours-confused",
        "Mixing up the hours at the higher overtime rate with all the overtime hours.",
        area="financial-maths",
    ),
    CatalogueEntry(
        "ordinary-hours-for-overtime-hours",
        "Reporting the ordinary hours of the week as the overtime hours.",
        area="financial-maths",
    ),
    CatalogueEntry(
        "overtime-hours-off-by-one",
        "Counting one hour too many when finding the hours beyond the ordinary week.",
        area="financial-maths",
    ),
    CatalogueEntry(
        "single-loading-for-all-overtime",
        "Paying every overtime hour at one loading, ignoring that the first hours are paid at a lower loading "
        "than the rest.",
        area="financial-maths",
    ),
    CatalogueEntry(
        "overtime-loading-omitted",
        "Paying overtime hours at the ordinary rate, leaving out the overtime loading.",
        area="financial-maths",
    ),
    CatalogueEntry(
        "overtime-tiers-swapped",
        "Paying the first overtime hours at the higher loading and the later hours at the lower one.",
        area="financial-maths",
    ),
    CatalogueEntry(
        "overtime-multiplier-on-all-hours",
        "Applying the overtime loading to every hour worked, not only the hours beyond the ordinary week.",
        area="financial-maths",
    ),
    CatalogueEntry(
        "overtime-pay-omitted",
        "Reporting only the ordinary pay as the week's gross pay, leaving the overtime out.",
        area="financial-maths",
    ),
    CatalogueEntry(
        "average-over-ordinary-hours",
        "Dividing the week's pay by the ordinary hours instead of the hours actually worked to find the average "
        "per hour.",
        area="financial-maths",
    ),
    CatalogueEntry(
        "average-over-overtime-hours",
        "Dividing the week's pay by the overtime hours only to find the average per hour worked.",
        area="financial-maths",
    ),
    CatalogueEntry(
        "ordinary-rate-for-average-pay",
        "Giving the ordinary hourly rate as the average earned per hour, though overtime raises the average.",
        area="financial-maths",
    ),
    CatalogueEntry(
        "all-cables-summed",
        "Adding the costs of all the possible cables instead of finding the cheapest set that connects every town.",
        area="networks",
    ),
    CatalogueEntry(
        "cheapest-cables-without-cycle-check",
        "Adding the cheapest cables without checking that each one joins a new town, so some cables form a loop.",
        area="networks",
    ),
    CatalogueEntry(
        "one-town-left-unconnected",
        "Subtracting the dearest cable of a network, which leaves a town disconnected, instead of re-routing.",
        area="networks",
    ),
    CatalogueEntry(
        "extra-spare-cable-added",
        "Adding the cost of a spare cable to a network that is already connected.",
        area="networks",
    ),
    CatalogueEntry(
        "cheapest-spare-cable-not-checked",
        "Taking the cheapest unused cable without checking that it re-joins the two groups left apart by a failed "
        "cable.",
        area="networks",
    ),
    CatalogueEntry(
        "failed-cable-cost-repeated",
        "Giving the cost of the failed cable as the cost of its replacement.",
        area="networks",
    ),
    CatalogueEntry(
        "network-cost-for-cable-cost",
        "Giving the cost of the rest of the network as the cost of the one replacement cable.",
        area="networks",
    ),
    CatalogueEntry(
        "costs-added-for-replacement",
        "Adding the cost of the failed cable to the cost of its replacement.",
        area="networks",
    ),
    CatalogueEntry(
        "failed-cable-not-replaced",
        "Subtracting the failed cable's cost from the network's cost without adding a replacement, leaving a town "
        "disconnected.",
        area="networks",
    ),
    CatalogueEntry(
        "failure-ignored",
        "Giving the original network's cost when one of its cables can no longer be used.",
        area="networks",
    ),
    CatalogueEntry(
        "replacement-added-failed-kept",
        "Adding the replacement cable's cost to the network without taking the failed cable's cost off.",
        area="networks",
    ),
    CatalogueEntry(
        "repeated-letters-not-divided",
        "Counting every arrangement of letters as different, without dividing by the factorial for each repeated "
        "letter.",
        area="counting",
    ),
    CatalogueEntry(
        "restriction-ignored",
        "Ignoring the restriction on a position or letter and counting every arrangement, as if the rule did not apply.",
        area="counting",
    ),
    CatalogueEntry(
        "vowels-arranged-with-consonants",
        "Arranging the vowels and consonants together when the question has already placed one class and is "
        "counting the other.",
        area="counting",
    ),
    CatalogueEntry(
        "one-letter-fewer-arranged",
        "Arranging one letter fewer than the group, as if one letter were fixed in place.",
        area="counting",
    ),
    CatalogueEntry(
        "order-of-vowels-omitted",
        "Choosing which places the restricted letters take but not counting the orders they can stand in.",
        area="counting",
    ),
    CatalogueEntry(
        "gaps-counted-without-the-end-gap",
        "Counting the gaps between letters as one fewer than there are, leaving out the gap at one end.",
        area="counting",
    ),
    CatalogueEntry(
        "gap-choice-omitted",
        "Counting only the orders of the restricted letters, not the choice of places they take.",
        area="counting",
    ),
    CatalogueEntry(
        "wrong-class-of-letters",
        "Counting the arrangements of the other group of letters when the question asked about the restricted group.",
        area="counting",
    ),
    CatalogueEntry(
        "block-not-counted-as-an-object",
        "Leaving the block of letters that stay together out of the count, or counting its letters separately.",
        area="counting",
    ),
    CatalogueEntry(
        "counts-added-not-multiplied",
        "Adding the numbers of ways (or choices at each position) for the stages of a task instead of multiplying them.",
        area="counting",
    ),
    CatalogueEntry(
        "repeated-letters-ignored-in-sample-space",
        "Dividing by the number of arrangements of all distinct objects, though repeated letters make some "
        "arrangements identical.",
        area="probability",
    ),
    CatalogueEntry(
        "partial-count-over-total",
        "Dividing a count for one stage of the task by the total, instead of the count for the whole task.",
        area="probability",
    ),
    CatalogueEntry(
        "supplement-rule-as-full-turn",
        "Using 360 degrees where the rule for opposite angles of a cyclic quadrilateral gives 180 degrees.",
        area="geometry",
    ),
    CatalogueEntry(
        "opposite-angles-taken-as-equal",
        "Taking opposite angles of a cyclic quadrilateral as equal instead of supplementary.",
        area="geometry",
    ),
    CatalogueEntry(
        "adjacent-angles-taken-as-supplementary",
        "Adding neighbouring angles of a cyclic quadrilateral to 180 degrees instead of opposite ones.",
        area="geometry",
    ),
    CatalogueEntry(
        "unknowns-swapped",
        "Giving the value of one unknown where the other was asked.",
        area="equations",
    ),
    CatalogueEntry(
        "opposite-angle-reported",
        "Reporting the angle opposite the one asked for.",
        kind="near_miss",
        area="geometry",
    ),
    CatalogueEntry(
        "adjacent-angle-reported",
        "Reporting a neighbouring angle of the quadrilateral instead of the one asked for.",
        kind="near_miss",
        area="geometry",
    ),
    CatalogueEntry(
        "interior-angle-for-exterior-angle",
        "Giving the interior angle where the exterior angle on a straight line with it was asked.",
        area="geometry",
    ),
    CatalogueEntry(
        "union-formula-without-overlap-subtracted",
        "Adding the sizes of two sets to find how many are in both, leaving out the subtraction of the total.",
        area="probability",
    ),
    CatalogueEntry(
        "neither-group-left-out-of-union",
        "Forgetting the people in neither set when using the total to find the overlap.",
        area="probability",
    ),
    CatalogueEntry(
        "only-b-count-reported-as-both",
        "Giving the number in only one set as the number in both sets.",
        area="probability",
    ),
    CatalogueEntry(
        "total-minus-counts-for-overlap",
        "Subtracting both set sizes from the total to get the overlap, which counts the people in neither set instead.",
        area="probability",
    ),
    CatalogueEntry(
        "overlap-counted-twice",
        "Adding the two set sizes to count those in exactly one set, so the overlap is counted twice.",
        area="probability",
    ),
    CatalogueEntry(
        "at-least-one-for-exactly-one",
        "Giving the number in at least one set where the number in exactly one was asked.",
        kind="near_miss",
        area="probability",
    ),
    CatalogueEntry(
        "neither-included-in-exactly-one",
        "Counting the people in neither set as if they were in exactly one set.",
        area="probability",
    ),
    CatalogueEntry(
        "overlap-included-in-numerator",
        "Counting everyone in a set, including those also in the other set, when only those in exactly one set "
        "were wanted.",
        area="probability",
    ),
    CatalogueEntry(
        "wrong-denominator",
        "Dividing by the total population or by the count for A rather than the count for the condition.",
        area="probability",
    ),
    CatalogueEntry(
        "reversed-conditioning",
        "Computing P(B given A) instead of P(A given B).",
        area="probability",
    ),
    CatalogueEntry(
        "complement-reported",
        "Giving the probability, proportion or count of the opposite outcome (the complement, or the cases that fail) instead of the one asked for.",
        kind="near_miss",
        area="probability",
    ),
    CatalogueEntry(
        "independence-product-ignored",
        "Giving P(A and B) alone where the difference from P(A) times P(B) was asked.",
        kind="near_miss",
        area="probability",
    ),
    CatalogueEntry(
        "joint-probability-ignored",
        "Giving the product P(A) times P(B) alone, leaving out P(A and B) in the independence check.",
        kind="near_miss",
        area="probability",
    ),
    CatalogueEntry(
        "single-probability-for-product",
        "Subtracting P(A) alone instead of the product P(A) times P(B) from P(A and B).",
        area="probability",
    ),
    CatalogueEntry(
        "diffusion-term-omitted",
        "Leaving the noise term out of an Euler-Maruyama step and giving the deterministic Euler value.",
        area="stochastic-processes",
    ),
    CatalogueEntry(
        "noise-coefficient-without-state",
        "Using the constant k instead of k times the current value as the diffusion coefficient.",
        area="stochastic-processes",
    ),
    CatalogueEntry(
        "noise-scaled-by-step-size",
        "Multiplying the noise term by the step size h, as if it scaled like the drift.",
        area="stochastic-processes",
    ),
    CatalogueEntry(
        "noise-sign-flipped",
        "Subtracting the noise term instead of adding it.",
        area="stochastic-processes",
    ),
    CatalogueEntry(
        "noise-value-used-as-step-size",
        "Using the stated noise value in place of the step size in the drift term.",
        area="stochastic-processes",
    ),
    CatalogueEntry(
        "diffusion-not-re-evaluated",
        "Using the starting value instead of the latest value to work out the diffusion coefficient in a later step.",
        area="stochastic-processes",
    ),
    CatalogueEntry(
        "drift-not-re-evaluated",
        "Using the starting value instead of the latest value to work out the drift in a later step.",
        area="stochastic-processes",
    ),
    CatalogueEntry(
        "noise-value-reused",
        "Using the first step's noise value again in the second step.",
        area="stochastic-processes",
    ),
    CatalogueEntry(
        "deterministic-value-reported",
        "Giving the value without noise where the effect of the noise was asked.",
        kind="near_miss",
        area="stochastic-processes",
    ),
    CatalogueEntry(
        "compared-with-wrong-step",
        "Comparing the final value with the no-noise value from the wrong step.",
        area="stochastic-processes",
    ),
    CatalogueEntry(
        "first-step-effect-only",
        "Reporting the effect of the noise after one step as the effect after two steps.",
        kind="near_miss",
        area="stochastic-processes",
    ),
    CatalogueEntry(
        "compared-with-start-value",
        "Taking the change from the starting value as the effect of the noise.",
        area="stochastic-processes",
    ),
    CatalogueEntry(
        "swapped-limits",
        "Subtracting the antiderivative at the upper limit from the one at the lower limit, so the sign is wrong.",
        area="calculus",
    ),
    CatalogueEntry(
        "forgot-lower-limit",
        "Giving the antiderivative at the upper limit as the area, or treating the antiderivative at the lower limit as zero, instead of subtracting it.",
        area="calculus",
    ),
    CatalogueEntry(
        "dropped-power-rule-factor",
        "Integrating a power without dividing by the new exponent.",
        area="calculus",
    ),
    CatalogueEntry(
        "power-rule-factor-multiplied",
        "Multiplying by the new exponent instead of dividing by it when integrating a power.",
        area="calculus",
    ),
    CatalogueEntry(
        "power-rule-exponent-not-raised",
        "Adding the exponent to the coefficient instead of raising the power and dividing.",
        area="calculus",
    ),
    CatalogueEntry(
        "used-integrand-not-antiderivative",
        "Substituting the limits into the original function instead of its antiderivative.",
        area="calculus",
    ),
    CatalogueEntry(
        "forgot-stated-multiplier",
        "Leaving out the stated multiple of an earlier integral when setting up the equation.",
        area="calculus",
    ),
    CatalogueEntry(
        "wrong-root-index",
        "Taking a square root where a cube or higher root is needed to undo the power of k.",
        area="calculus",
    ),
    # -- r3f exam chains (integration, vectors, networks, counting, geometry, measurement, ODE, quotient rings) ---
    CatalogueEntry(
        "completing-square-constant-unchanged",
        "Completing the square and leaving the constant term as it was, without taking away the square of "
        "half the coefficient: x^2 + 6x + 13 written as (x + 3)^2 + 13.",
        area="integration",
    ),
    CatalogueEntry(
        "completing-square-adjusted-wrong-way",
        "Correcting for the square of half the coefficient in the wrong direction when completing the "
        "square, adding it where it should be taken away or the reverse.",
        area="integration",
    ),
    CatalogueEntry(
        "completing-square-reports-half-squared",
        "Giving the square of half the coefficient of x as the constant left over after completing the square.",
        area="integration",
    ),
    CatalogueEntry(
        "completing-square-reports-root-not-square",
        "Giving a, the square root of the constant left over after completing the square, where the "
        "constant a^2 was asked for.",
        area="integration",
    ),
    CatalogueEntry(
        "arctan-missing-one-over-a",
        "Integrating k/((x + h)^2 + a^2) as k arctan((x + h)/a), leaving out the factor 1/a.",
        area="integration",
    ),
    CatalogueEntry(
        "arctan-divided-by-a-squared",
        "Integrating k/((x + h)^2 + a^2) with a factor 1/a^2 in place of 1/a.",
        area="integration",
    ),
    CatalogueEntry(
        "arcsin-extra-division-by-a",
        "Integrating k/sqrt(a^2 - (x + h)^2) as (k/a) arcsin((x + h)/a), dividing by a as for arctan.",
        area="integration",
    ),
    CatalogueEntry(
        "arcsin-extra-multiplication-by-a",
        "Integrating k/sqrt(a^2 - (x + h)^2) with an extra factor of a.",
        area="integration",
    ),
    CatalogueEntry(
        "integral-dropped-numerator-constant",
        "Leaving out the constant numerator k of the integrand when writing down the antiderivative.",
        area="integration",
    ),
    CatalogueEntry(
        "mean-value-not-divided-by-interval-length",
        "Giving the integral over an interval where the mean value, the integral divided by the length of "
        "the interval, was asked for.",
        area="integration",
    ),
    CatalogueEntry(
        "mean-value-divided-by-upper-limit",
        "Dividing an integral by the upper limit of the interval instead of by the length of the interval "
        "to find a mean value.",
        area="integration",
    ),
    CatalogueEntry(
        "mean-value-multiplied-by-interval-length",
        "Multiplying the integral by the length of the interval where a mean value divides by it.",
        area="integration",
    ),
    CatalogueEntry(
        "mean-value-divided-by-half-interval",
        "Dividing the integral by half the length of the interval when finding a mean value.",
        area="integration",
    ),
    CatalogueEntry(
        "parallelogram-wrong-opposite-vertex",
        "Finding the vertex opposite a different corner of the parallelogram: taking D = A + B - C or D = "
        "B + C - A where D = A - B + C.",
        area="coordinate geometry",
    ),
    CatalogueEntry(
        "parallelogram-added-all-vertices",
        "Adding all three given vertices of a parallelogram to find the fourth.",
        area="coordinate geometry",
    ),
    CatalogueEntry(
        "parallelogram-midpoint-for-fourth-vertex",
        "Giving the midpoint of the diagonal AC as the fourth vertex D of the parallelogram ABCD.",
        area="coordinate geometry",
    ),
    CatalogueEntry(
        "parallelogram-area-by-dot-product",
        "Using the dot product of two adjacent sides where the cross product gives the area of a parallelogram.",
        area="coordinate geometry",
    ),
    CatalogueEntry(
        "parallelogram-area-single-product",
        "Giving only one of the two products in the cross product of two adjacent sides as the area of a "
        "parallelogram.",
        area="coordinate geometry",
    ),
    CatalogueEntry(
        "parallelogram-area-as-triangle",
        "Giving the area of the triangle formed by two adjacent sides, half the area of the "
        "parallelogram, for the parallelogram.",
        area="coordinate geometry",
    ),
    CatalogueEntry(
        "parallelogram-area-from-position-vectors",
        "Finding the area from the position vectors of two opposite vertices instead of from two adjacent "
        "side vectors.",
        area="coordinate geometry",
    ),
    CatalogueEntry(
        "dot-product-multiplied-across",
        "Finding a dot product as a1 b2 + a2 b1, multiplying across instead of component by component.",
        area="vectors",
    ),
    CatalogueEntry(
        "dot-product-subtracted-components",
        "Finding a dot product as a1 b1 - a2 b2, subtracting the products of the components.",
        area="vectors",
    ),
    CatalogueEntry(
        "ratio-inverted-solving-for-k",
        "Solving a + b k = 0 for k as -b/a, with the ratio the wrong way up.",
        area="vectors",
    ),
    CatalogueEntry(
        "subtracted-instead-of-divided",
        "Subtracting where the constant term is divided by the coefficient when solving a + b k = 0 for k.",
        area="vectors",
    ),
    CatalogueEntry(
        "perpendicular-condition-used-wrong-vector",
        "Setting the dot product with the wrong vector equal to zero when finding the scalar that makes "
        "two vectors perpendicular.",
        area="vectors",
    ),
    CatalogueEntry(
        "perpendicular-dropped-first-vector",
        "Giving the length of u + kv as the length of kv, leaving out u.",
        area="vectors",
    ),
    CatalogueEntry(
        "squared-sum-without-cross-term",
        "Giving |u + kv|^2 as |u|^2 + k^2 |v|^2, leaving out the cross term 2k(u . v).",
        area="vectors",
    ),
    CatalogueEntry(
        "parameter-offset-ignored",
        "Solving a t + p = x for t as x/a, leaving out the constant p.",
        area="vectors",
    ),
    CatalogueEntry(
        "parameter-offset-sign-error",
        "Solving a t + p = x for t as (x + p)/a, with the wrong sign on p.",
        area="vectors",
    ),
    CatalogueEntry(
        "parameter-offset-y-ignored",
        "Solving b/t + c = y for b as y t, leaving out the constant c.",
        area="vectors",
    ),
    CatalogueEntry(
        "reports-b-over-t-not-b",
        "Giving the value of b/t where the value of b was asked for.",
        area="vectors",
    ),
    CatalogueEntry(
        "sign-error-eliminating-parameter",
        "Getting the sign of the constant K wrong when eliminating the parameter from (a t, b/t).",
        area="vectors",
    ),
    CatalogueEntry(
        "dropped-coefficient-a",
        "Giving b for the constant K = ab in (x - p)(y - c) = K, leaving out the factor a.",
        area="vectors",
    ),
    CatalogueEntry(
        "relation-from-point-product",
        "Taking the constant in (x - p)(y - c) = K as the product x0 y0 of the coordinates of a point, "
        "ignoring the shifts p and c.",
        area="vectors",
    ),
    CatalogueEntry(
        "cartesian-offset-y-ignored",
        "Reading y - c = K/(x - p) and giving y - c as the value of y, leaving out c.",
        area="vectors",
    ),
    CatalogueEntry(
        "cartesian-sign-error-on-constant",
        "Finding y as c - K/(x - p) where y = c + K/(x - p).",
        area="vectors",
    ),
    CatalogueEntry(
        "cartesian-offset-x-ignored",
        "Using K/x in place of K/(x - p), leaving out the shift in x.",
        area="vectors",
    ),
    CatalogueEntry(
        "second-point-reuses-first-point",
        "Giving the y-value of the first point again where the y-value at a second x was asked for.",
        area="vectors",
    ),
    CatalogueEntry(
        "second-point-parameter-set-to-one",
        "Evaluating b/t + c with t taken as 1 where t had to be found from the second value of x.",
        area="vectors",
    ),
    CatalogueEntry(
        "cut-counts-backward-pipes",
        "Adding to a cut's capacity the pipes that run from the sink side back to the source side; only "
        "pipes from the source side to the sink side count.",
        area="networks",
    ),
    CatalogueEntry(
        "cut-sums-source-pipes",
        "Giving the total capacity of the pipes leaving the source as the capacity of a cut that "
        "separates more than the source.",
        area="networks",
    ),
    CatalogueEntry(
        "cut-misses-a-crossing-pipe",
        "Leaving one pipe that crosses the cut out of the cut's capacity.",
        area="networks",
    ),
    CatalogueEntry(
        "cut-sums-pipes-on-one-side",
        "Adding up the pipes that lie wholly on the source side of a cut instead of the pipes that cross it.",
        area="networks",
    ),
    CatalogueEntry(
        "sum-all-source-outgoing",
        "Giving the total capacity of the pipes leaving the source as the maximum flow, ignoring "
        "bottlenecks downstream.",
        area="networks",
    ),
    CatalogueEntry(
        "sum-all-sink-incoming",
        "Giving the total capacity of the pipes entering the sink as the maximum flow, ignoring bottlenecks upstream.",
        area="networks",
    ),
    CatalogueEntry(
        "minimum-pipe-capacity",
        "Giving the smallest single pipe capacity as the maximum flow, ignoring that flow can split across paths.",
        area="networks",
    ),
    CatalogueEntry(
        "flow-equals-capacity-of-a-non-minimum-cut",
        "Giving the capacity of a cut as the maximum flow when a smaller cut exists; the maximum flow "
        "equals the smallest cut.",
        area="networks",
    ),
    CatalogueEntry(
        "upgrade-increase-assumed-in-full",
        "Adding the whole upgrade to the maximum flow, as if the upgraded pipe were the only bottleneck.",
        area="networks",
    ),
    CatalogueEntry(
        "upgrade-ignored",
        "Leaving the maximum flow unchanged after a pipe is upgraded, or saying it cannot increase when the pipe is in every smallest cut.",
        area="networks",
    ),
    CatalogueEntry(
        "increase-equals-upgrade",
        "Giving the size of the upgrade as the increase in the maximum flow.",
        area="networks",
    ),
    CatalogueEntry(
        "reports-new-flow-not-increase",
        "Giving the new maximum flow where the increase in the maximum flow was asked for.",
        area="networks",
    ),
    CatalogueEntry(
        "reports-old-flow-not-increase",
        "Giving the original maximum flow where the increase in the maximum flow was asked for.",
        area="networks",
    ),
    CatalogueEntry(
        "one-factor-short",
        "Multiplying one factor too few when counting ordered selections, so that P(n, r) is given as P(n, r - 1).",
        area="counting",
    ),
    CatalogueEntry(
        "removed-one-member-too-few",
        "Removing only one member from the pool when a whole subgroup is excluded.",
        area="counting",
    ),
    CatalogueEntry(
        "complement-added-instead-of-subtracted",
        "Adding the count of the complement to the total where 'at least one' is the total minus the complement.",
        area="counting",
    ),
    CatalogueEntry(
        "at-least-one-counted-for-first-role-only",
        "Counting 'at least one subgroup member' by choosing one for the first role only and filling the rest freely.",
        area="counting",
    ),
    CatalogueEntry(
        "at-least-one-overcounted-by-placing-one-then-filling-freely",
        "Counting 'at least one' by placing a subgroup member in one role and filling the others freely, "
        "which counts arrangements with several subgroup members more than once.",
        area="counting",
    ),
    CatalogueEntry(
        "probability-denominator-allows-repeats",
        "Using n^r as the number of equally likely outcomes when the roles go to different members.",
        area="counting",
    ),
    CatalogueEntry(
        "subgroup-share-reported-as-probability",
        "Giving the share of the group in the subgroup, s/n, as the probability that at least one of them "
        "holds a role.",
        area="counting",
    ),
    CatalogueEntry(
        "wrong-sign-equation",
        "Setting two side expressions equal with a sign error, AB = -BC, when solving for x.",
        area="geometry",
    ),
    CatalogueEntry(
        "added-coefficients-instead-of-subtracting",
        "Adding the coefficients of x where they are subtracted when solving a x + b = c x + d.",
        area="geometry",
    ),
    CatalogueEntry(
        "x-term-on-one-side-ignored",
        "Solving a x + b = c x + d while ignoring the x term on one side.",
        area="geometry",
    ),
    CatalogueEntry(
        "reports-x-not-side",
        "Giving x where the length of the side, found by substituting x, was asked for.",
        area="geometry",
    ),
    CatalogueEntry(
        "coefficient-not-applied-to-x",
        "Evaluating a x + b as x + b, adding the constant to x without multiplying by the coefficient.",
        area="geometry",
    ),
    CatalogueEntry(
        "sign-error-on-constant",
        "Evaluating a side expression with the wrong sign on its constant: c x - d for c x + d.",
        area="geometry",
    ),
    CatalogueEntry(
        "omits-doubling-the-half-diagonal",
        "Giving the half-diagonal found from Pythagoras' theorem where the whole diagonal was asked for.",
        area="geometry",
    ),
    CatalogueEntry(
        "pythagoras-added-instead-of-subtracted",
        "Adding the squares where Pythagoras' theorem gives the half-diagonal as the side squared minus "
        "the other half-diagonal squared.",
        area="geometry",
    ),
    CatalogueEntry(
        "reports-side-for-diagonal",
        "Giving the side of the rhombus where its diagonal was asked for.",
        area="geometry",
    ),
    CatalogueEntry(
        "reports-given-diagonal-as-answer",
        "Giving the diagonal stated in the question where the other diagonal was asked for.",
        area="geometry",
    ),
    CatalogueEntry(
        "full-diagonal-used-as-pythagoras-leg",
        "Using a whole diagonal where half of it is a leg of the right triangle formed at the centre.",
        area="geometry",
    ),
    CatalogueEntry(
        "rhombus-area-diagonals-not-halved",
        "Giving the product of the two diagonals as the area of a rhombus, without the factor one half.",
        area="geometry",
    ),
    CatalogueEntry(
        "rhombus-area-halved-twice",
        "Dividing the product of the diagonals by 4 where the area of a rhombus is half that product.",
        area="geometry",
    ),
    CatalogueEntry(
        "rhombus-area-as-side-squared",
        "Giving the square of the side as the area of a rhombus, as if it were a square.",
        area="geometry",
    ),
    CatalogueEntry(
        "rhombus-area-as-side-times-half-diagonal",
        "Multiplying the side by half a diagonal to find the area of a rhombus.",
        area="geometry",
    ),
    CatalogueEntry(
        "wrong-hemisphere-rule",
        "Subtracting the two latitudes when they are in opposite hemispheres (the magnitudes are added), "
        "or adding them when they are in the same hemisphere (they are subtracted).",
        area="measurement",
    ),
    CatalogueEntry(
        "forgot-arcminute-conversion",
        "Giving the angular separation in degrees, never multiplied by 60 into arcminutes, so the "
        "distance in nautical miles is far too small.",
        area="measurement",
    ),
    CatalogueEntry(
        "halves-the-constant",
        "Multiplying the angular separation in degrees by 30 instead of 60, halving the distance in nautical miles.",
        area="measurement",
    ),
    CatalogueEntry(
        "misreads-minutes-as-hundredths",
        "Reading 48 minutes of latitude as .48 of a degree instead of .8 of a degree.",
        area="measurement",
    ),
    CatalogueEntry(
        "confuses-nm-with-statute-mile",
        "Multiplying a distance in nautical miles by 1.609, the statute-mile-to-kilometre factor, instead of 1.852.",
        area="measurement",
    ),
    CatalogueEntry(
        "forgot-unit-conversion",
        "Giving the figure in nautical miles as if it were already in kilometres.",
        area="measurement",
    ),
    CatalogueEntry(
        "used-kilometres-with-knots",
        "Dividing a distance in kilometres by a speed in knots, mixing the units.",
        area="measurement",
    ),
    CatalogueEntry(
        "converted-the-speed-instead-of-the-distance",
        "Dividing a distance in nautical miles by the speed converted to kilometres per hour, converting "
        "one side of the units only.",
        area="measurement",
    ),
    CatalogueEntry(
        "divided-by-nautical-mile-factor",
        "Dividing by 1.852 to change nautical miles to kilometres, where the conversion multiplies.",
        area="measurement",
    ),
    CatalogueEntry(
        "converted-to-statute-miles-instead-of-km",
        "Multiplying nautical miles by 1.151, which gives statute miles, to find kilometres.",
        area="measurement",
    ),
    CatalogueEntry(
        "distance-used-in-place-of-time",
        "Multiplying the distance by the fuel rate per hour, using the distance where the time was needed.",
        area="measurement",
    ),
    CatalogueEntry(
        "forgets-to-subtract-antiderivative",
        "Giving the y-value of the given point as the constant, without subtracting the antiderivative at that x.",
        area="differential equations",
    ),
    CatalogueEntry(
        "sign-error-on-antiderivative",
        "Adding the antiderivative at the given point to y0 instead of subtracting it when finding the constant.",
        area="differential equations",
    ),
    CatalogueEntry(
        "reports-antiderivative-as-constant",
        "Giving the value of the antiderivative at the point as the constant, ignoring the y-value.",
        area="differential equations",
    ),
    CatalogueEntry(
        "antiderivative-evaluated-at-wrong-value",
        "Substituting the y-value of the point into the antiderivative instead of its x-value.",
        area="differential equations",
    ),
    CatalogueEntry(
        "evaluated-derivative-instead-of-function",
        "Substituting x into the derivative of the supplied function where the function itself was needed.",
        area="differential equations",
    ),
    CatalogueEntry(
        "particular-solution-omits-constant",
        "Giving the value of the antiderivative at the point and leaving out the constant fixed by the starting "
        "condition.",
        area="differential equations",
    ),
    CatalogueEntry(
        "particular-solution-constant-sign-error",
        "Giving the particular solution with the sign of the constant reversed.",
        area="differential equations",
    ),
    CatalogueEntry(
        "particular-solution-used-y0-as-constant",
        "Using the y-value of the given point as the constant in the particular solution.",
        area="differential equations",
    ),
    CatalogueEntry(
        "reports-constant-not-value",
        "Giving the constant of integration where the value of the solution at a point was asked for.",
        area="differential equations",
    ),
    CatalogueEntry(
        "tangent-and-normal-confused",
        "Giving the normal, with gradient -1/m, where the tangent was asked for.",
        area="differential equations",
    ),
    CatalogueEntry(
        "tangent-point-slope-dropped-y-value",
        "Writing the tangent as y = m(x - x1), leaving out the y-value of the point.",
        area="differential equations",
    ),
    CatalogueEntry(
        "tangent-point-slope-sign-error",
        "Writing the tangent as y = m(x + x1) + y1, with the wrong sign on x1.",
        area="differential equations",
    ),
    CatalogueEntry(
        "y-value-used-as-gradient",
        "Using the y-value at a point as the gradient of the tangent in place of the derivative.",
        area="differential equations",
    ),
    CatalogueEntry(
        "forgets-to-reduce-mod-m",
        "Giving a product with its coefficients reduced but the polynomial left unreduced, still "
        "containing powers of x at or above the degree of the modulus.",
        area="abstract algebra",
    ),
    CatalogueEntry(
        "forgets-mod-p-reduction",
        "Leaving a coefficient or evaluated value as an ordinary integer instead of reducing it to a residue modulo p.",
        area="abstract algebra",
    ),
    CatalogueEntry(
        "sign-error-in-reduction",
        "Reducing modulo x^2 + b x + c with the wrong sign, replacing x^2 by b x + c instead of -b x - c.",
        area="abstract algebra",
    ),
    CatalogueEntry(
        "drops-a-term-of-modulus",
        "Reducing modulo a polynomial but leaving out its constant term.",
        area="abstract algebra",
    ),
    CatalogueEntry(
        "reports-larger-root-for-smaller",
        "Giving the larger root where the smaller root was asked for.",
        area="abstract algebra",
    ),
    CatalogueEntry(
        "reports-linear-coefficient-as-root",
        "Giving the coefficient of x in the modulus as a root.",
        area="abstract algebra",
    ),
    CatalogueEntry(
        "reports-constant-term-as-root",
        "Giving the constant term of the modulus as a root.",
        area="abstract algebra",
    ),
    CatalogueEntry(
        "evaluated-at-the-other-root",
        "Evaluating at the other root of the modulus than the one asked for.",
        area="abstract algebra",
    ),
    CatalogueEntry(
        "evaluated-at-zero-instead-of-the-root",
        "Giving the constant term of a polynomial where its value at a root was asked for.",
        area="abstract algebra",
    ),
    # -- exam chains, prefix r3b ------------------------------------------------------------
    CatalogueEntry(
        "differentiated-once",
        "Stopping after the first derivative when the second was asked for.",
        area="differentiation",
    ),
    CatalogueEntry(
        "product-rule-cross-term-dropped",
        "Differentiating a product twice but leaving out the middle term 2u'v' of (uv)'' = u''v + 2u'v' + uv''.",
        area="differentiation",
    ),
    CatalogueEntry(
        "chain-rule-inner-derivative-dropped",
        "Leaving out the factor from the inside function when differentiating a composite: (ax + b)^n gives n(ax + b)^(n-1), not an(ax + b)^(n-1), and arctan(kx + m) gives 1/(1 + (kx + m)^2) without the k.",
        area="differentiation",
    ),
    CatalogueEntry(
        "second-derivatives-multiplied",
        "Taking the second derivative of a product as the product of the two second derivatives.",
        area="differentiation",
    ),
    CatalogueEntry(
        "evaluated-original-function",
        "Substituting the value into the original function where its derivative was asked for.",
        area="differentiation",
    ),
    CatalogueEntry(
        "gave-other-root-of-two",
        "Finding both solutions of an equation and reporting the one the question did not ask for.",
        area="equations",
    ),
    CatalogueEntry(
        "repeated-root-taken-as-inflection",
        "Counting a repeated root of the second derivative as a point of inflection, though the second derivative does not change sign there.",
        area="differentiation",
    ),
    CatalogueEntry(
        "added-parts-without-squaring",
        "Adding the real and imaginary parts, a + b, where a^2 + b^2 was needed.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "forgot-to-divide-by-denominator-modulus",
        "Multiplying the top and bottom by the conjugate but leaving out the division by the new real denominator (c^2 + d^2).",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "swapped-real-and-imaginary-parts",
        "Reporting the imaginary part (or the coefficient of i) where the real part was asked for, or the reverse.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "divided-parts-separately",
        "Dividing the real parts and the imaginary parts separately instead of using the conjugate of the denominator.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "forgot-to-conjugate",
        "Stopping at the quotient and not taking its conjugate.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "cross-term-sign-error-in-division",
        "Getting the sign of the cross terms wrong when multiplying by the conjugate of the denominator.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "added-conjugates-for-product",
        "Adding z and its conjugate (twice the real part) where their product was asked for.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "multiplied-by-denominator-modulus",
        "Multiplying by the modulus squared of the denominator where the quotient should have been divided by it.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "forgot-square-root-in-modulus",
        "Finding the sum of the squares of the real and imaginary parts and stopping, without taking the square root to get the modulus.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "reported-root-modulus-for-w",
        "Giving the modulus of the root where the modulus of the number itself was asked for.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "argument-from-reference-angle-without-quadrant",
        "Reporting the acute reference angle as the argument, without placing it in the quadrant of the number.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "argument-in-opposite-quadrant",
        "Placing the argument in the opposite quadrant by adding or removing half a turn, usually from taking arctan(y/x) with the signs cancelled.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "argument-of-conjugate-taken",
        "Measuring the argument clockwise, which gives the argument of the conjugate (360 minus the angle).",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "divided-modulus-by-n-not-root",
        "Dividing the modulus by n where its n-th root is needed.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "kept-modulus-of-w",
        "Leaving the modulus of each root equal to the modulus of the original number.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "used-square-root-for-nth-root",
        "Taking a square root where the n-th root was needed.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "dropped-the-full-turn-term",
        "Dividing only the argument of w by n, leaving out the whole turns 360k added before dividing, so every root gets the principal argument.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "added-full-turn-after-dividing",
        "Adding the whole turns 360k after dividing by n instead of before, giving an argument far outside the quadrant.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "chose-root-in-wrong-quadrant",
        "Working out the roots correctly but reporting the one in a neighbouring quadrant.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "all-plus-expansion",
        "Expanding a determinant along a row with every cofactor taken positive, forgetting the alternating + - + signs.",
        area="matrices",
    ),
    CatalogueEntry(
        "cofactor-sign-error",
        "Using the wrong sign for a cofactor in a determinant expansion (the checkerboard of signs applied the wrong way round).",
        area="matrices",
    ),
    CatalogueEntry(
        "minor-used-for-cofactor",
        "Using the minor of an entry where its cofactor was needed, leaving out the sign from the checkerboard pattern.",
        area="matrices",
    ),
    CatalogueEntry(
        "scalar-not-cubed",
        "Pulling a scalar c out of a 3 by 3 determinant as c rather than c cubed.",
        area="matrices",
    ),
    CatalogueEntry(
        "scalar-squared-as-for-2x2",
        "Pulling a scalar c out of a 3 by 3 determinant as c squared, as for a 2 by 2 matrix.",
        area="matrices",
    ),
    CatalogueEntry(
        "months-used-as-years",
        "Using a number of months as if it were a number of years, so the time is twelve times too long.",
        area="finance",
    ),
    CatalogueEntry(
        "omitted-time-in-interest",
        "Finding one year's interest and ignoring the length of the term.",
        area="finance",
    ),
    CatalogueEntry(
        "used-compound-instead-of-simple",
        "Compounding the interest each period where the investment pays simple interest on the original principal.",
        area="finance",
    ),
    CatalogueEntry(
        "percentage-not-converted-to-decimal",
        "Using the rate as 5 rather than 5/100 (or 0.05) in the interest formula.",
        area="finance",
    ),
    CatalogueEntry(
        "reported-interest-not-total-value",
        "Giving the interest earned where the total value of the investment (principal plus interest) was asked for.",
        kind="near_miss",
        area="finance",
    ),
    CatalogueEntry(
        "subtracted-interest-from-principal",
        "Subtracting the interest from the principal where it should be added.",
        area="finance",
    ),
    CatalogueEntry(
        "time-ratio-inverted",
        "Scaling the rate by the ratio of the terms the wrong way up: a shorter term needs a higher rate, not a lower one.",
        area="finance",
    ),
    CatalogueEntry(
        "kept-original-rate",
        "Giving the original rate again, though the term has changed and the same interest needs a different rate.",
        area="finance",
    ),
    CatalogueEntry(
        "forgot-percent-conversion",
        "Leaving the rate as a decimal (0.06) where it was asked for as a percentage (6).",
        area="finance",
    ),
    CatalogueEntry(
        "sums-all-durations",
        "Taking the project's minimum completion time as the sum of every activity's duration, with no allowance for activities that run at the same time.",
        area="networks",
    ),
    CatalogueEntry(
        "takes-shorter-path",
        "Following the shorter route where two routes meet, instead of the longer one that sets the earliest start.",
        area="networks",
    ),
    CatalogueEntry(
        "stops-one-activity-early",
        "Reading the finish time of an activity that is not the last one in the project.",
        area="networks",
    ),
    CatalogueEntry(
        "float-ignores-following-activities",
        "Finding an activity's float as the project time less its own earliest finish, ignoring the activities that must follow it.",
        area="networks",
    ),
    CatalogueEntry(
        "float-from-duration-only",
        "Finding an activity's float as the project time less its own duration, ignoring the activities before and after it.",
        area="networks",
    ),
    CatalogueEntry(
        "reported-earliest-start-as-float",
        "Giving an activity's earliest start time where its float was asked for.",
        area="networks",
    ),
    CatalogueEntry(
        "float-sign-reversed",
        "Finding the float as the longest route through the activity less the project time, which gives the negative of the float.",
        area="networks",
    ),
    CatalogueEntry(
        "crash-saving-taken-in-full",
        "Subtracting the whole reduction from the project time, though another route becomes critical and limits the saving.",
        area="networks",
    ),
    CatalogueEntry(
        "ignored-the-reduction",
        "Leaving the project time unchanged after shortening a critical activity.",
        area="networks",
    ),
    CatalogueEntry(
        "subtracted-overshoot-instead-of-saving",
        "Subtracting the part of the reduction beyond the margin over the rival route, instead of the margin itself.",
        area="networks",
    ),
    CatalogueEntry(
        "forgets-one-factorial",
        "Dropping one factor from the product left after cancelling the common factorial part of a ratio.",
        area="combinatorics",
    ),
    CatalogueEntry(
        "adds-extra-factor",
        "Multiplying in one extra consecutive factor when cancelling the common part of a ratio of factorials.",
        area="combinatorics",
    ),
    CatalogueEntry(
        "summed-factors-not-multiplied",
        "Adding the factors left after cancelling a ratio of factorials where they should be multiplied.",
        area="combinatorics",
    ),
    CatalogueEntry(
        "sign-error-combining-terms",
        "Adding where the two terms should be subtracted, or the reverse, when combining the simplified terms.",
        area="combinatorics",
    ),
    CatalogueEntry(
        "gave-extraneous-negative-root",
        "Giving the root of a quadratic that lies outside the stated domain, such as a negative n where n is at least 2.",
        area="equations",
    ),
    CatalogueEntry(
        "gave-value-not-n",
        "Giving the value the expression takes where the value of n that produces it was asked for.",
        area="equations",
    ),
    CatalogueEntry(
        "area-ratio-taken-as-scale-factor",
        "Giving the ratio of the areas as the scale factor of the lengths, without taking the square root.",
        area="similarity",
    ),
    CatalogueEntry(
        "volume-ratio-taken-as-scale-factor",
        "Giving a ratio of volumes (the cube of the scale factor) as the scale factor of the lengths.",
        area="similarity",
    ),
    CatalogueEntry(
        "swapped-p-and-q",
        "Using the reciprocal of the scale factor, so the larger figure comes out smaller.",
        area="similarity",
    ),
    CatalogueEntry(
        "uses-squared-factor",
        "Applying the area scale factor k squared to a length.",
        area="similarity",
    ),
    CatalogueEntry(
        "uses-cubed-factor-for-length",
        "Applying the volume scale factor k cubed to a length.",
        area="similarity",
    ),
    CatalogueEntry(
        "forgets-to-scale",
        "Giving the small figure's measurement again as if it were the large figure's.",
        area="similarity",
    ),
    CatalogueEntry(
        "volume-scaled-by-linear-factor",
        "Scaling a volume by the linear scale factor k instead of k cubed.",
        area="similarity",
    ),
    CatalogueEntry(
        "volume-scaled-by-area-factor",
        "Scaling a volume by the area scale factor k squared instead of k cubed.",
        area="similarity",
    ),
    CatalogueEntry(
        "forgot-to-divide-by-gap",
        "Taking the difference between two terms as the common difference, without dividing by the number of steps between them.",
        area="sequences",
    ),
    CatalogueEntry(
        "counted-terms-inclusively-for-gap",
        "Counting both end terms when finding the number of steps between two terms, so the gap is one too many.",
        area="sequences",
    ),
    CatalogueEntry(
        "sign-error-common-difference",
        "Getting the sign of the common difference (or of the term it is used to build) wrong, treating growth as decay or the reverse.",
        area="sequences",
    ),
    CatalogueEntry(
        "divided-by-position-sum",
        "Dividing the change between two terms by the sum of their positions instead of the difference.",
        area="sequences",
    ),
    CatalogueEntry(
        "stated-term-taken-as-first-term",
        "Taking the first stated term as the first term of the sequence, though it is later in the sequence.",
        area="sequences",
    ),
    CatalogueEntry(
        "off-by-one-nth-term",
        "Using n instead of n - 1 in t_n = t_1 + (n - 1)d, which shifts the answer by one whole common difference.",
        area="sequences",
    ),
    CatalogueEntry(
        "wrong-term-paired-with-position",
        "Pairing a term's value with the position of the other term when working back to the first term.",
        area="sequences",
    ),
    CatalogueEntry(
        "sum-formula-not-halved",
        "Leaving out the one half in S_n = n/2 (2a + (n - 1)d).",
        area="sequences",
    ),
    CatalogueEntry(
        "total-as-count-times-last-term",
        "Finding the total of an arithmetic sequence as the number of terms times the last term.",
        area="sequences",
    ),
    CatalogueEntry(
        "reported-term-not-sum",
        "Giving the last term of the sequence where the total of the terms was asked for.",
        area="sequences",
    ),
    CatalogueEntry(
        "off-by-one-in-sum",
        "Using n instead of n - 1 in the sum formula S_n = n/2 (2a + (n - 1)d).",
        area="sequences",
    ),
    CatalogueEntry(
        "first-index-one-early",
        "Giving the last period before the threshold is passed, where the first period after it was asked for.",
        area="sequences",
    ),
    CatalogueEntry(
        "first-index-one-late",
        "Giving the period after the first one that passes the threshold.",
        area="sequences",
    ),
    CatalogueEntry(
        "reported-the-value-not-the-period",
        "Giving the value of the first term past the threshold where the period it occurs in was asked for.",
        area="sequences",
    ),
    CatalogueEntry(
        "uses-jacobi-method",
        "Using the old value of x in the second update (the Jacobi method) instead of the x just computed (Gauss-Seidel).",
        area="numerical-methods",
    ),
    CatalogueEntry(
        "wrong-variable-order",
        "Updating the variables in the wrong order, or solving each equation for the wrong variable.",
        area="numerical-methods",
    ),
    CatalogueEntry(
        "forgets-diagonal-divide",
        "Leaving out the division by the diagonal coefficient after moving the other term across.",
        area="numerical-methods",
    ),
    CatalogueEntry(
        "sign-error-on-off-diagonal",
        "Adding the off-diagonal term where it should be subtracted when solving for the variable.",
        area="numerical-methods",
    ),
    CatalogueEntry(
        "ignores-dominance-order",
        "Using the equation that is printed first to update x, though x does not have the larger coefficient there.",
        area="numerical-methods",
    ),
    CatalogueEntry(
        "residual-of-the-wrong-equation",
        "Reporting the residual of the equation just solved for y, which is zero, where the other equation's residual was asked for.",
        area="numerical-methods",
    ),
    CatalogueEntry(
        "residual-sign-reversed",
        "Taking a residual as the left side minus the right side, which gives its negative.",
        area="numerical-methods",
    ),
    CatalogueEntry(
        "residual-at-mixed-iterates",
        "Evaluating a residual at a mixture of old and new values instead of the latest values of both variables.",
        area="numerical-methods",
    ),
    CatalogueEntry(
        "error-measured-from-start",
        "Giving the error of the starting value where the error of the new estimate was asked for.",
        area="numerical-methods",
    ),
    CatalogueEntry(
        "error-in-wrong-variable",
        "Giving the error in y where the error in x was asked for.",
        area="numerical-methods",
    ),
    CatalogueEntry(
        "signed-error-not-magnitude",
        "Giving the error with its sign (estimate minus exact value) where its magnitude was asked for.",
        area="numerical-methods",
    ),
    CatalogueEntry(
        "coefficients-swapped-in-update",
        "Using the coefficient of the other variable, or of the other equation, when solving for the next value.",
        area="numerical-methods",
    ),
    CatalogueEntry(
        "step-size-for-error",
        "Giving the size of the step taken in x where the distance from the exact solution was asked for.",
        area="numerical-methods",
    ),
    CatalogueEntry(
        "bound-as-p-hat",
        "Reading one end of the interval as the sample proportion instead of its midpoint.",
        area="statistics",
    ),
    CatalogueEntry(
        "swapped-centre-and-margin",
        "Giving the centre of the interval where the margin of error was asked for, or the reverse.",
        area="statistics",
    ),
    CatalogueEntry(
        "margin-as-full-width",
        "Using the full width of the interval as the margin of error instead of its half-width.",
        area="statistics",
    ),
    CatalogueEntry(
        "halved-twice",
        "Halving the width of the interval a second time when finding the margin of error.",
        area="statistics",
    ),
    CatalogueEntry(
        "missing-z-squared",
        "Using z where z squared appears when inverting the margin formula for the sample size.",
        area="statistics",
    ),
    CatalogueEntry(
        "missing-complement",
        "Leaving out the (1 - p) factor when inverting the margin formula for the sample size.",
        area="statistics",
    ),
    CatalogueEntry(
        "no-square-root-inversion",
        "Treating the margin formula as linear in the sample size rather than in its square root, so the margin is not squared.",
        area="statistics",
    ),
    CatalogueEntry(
        "rounds-margin-target-down",
        "Rounding the required sample size down, so the margin is slightly larger than the target.",
        area="statistics",
    ),
    CatalogueEntry(
        "linear-margin-rescale",
        "Scaling the sample size by the ratio of the margins, forgetting that the margin falls as one over the square root of the sample size.",
        area="statistics",
    ),
    # -- r3e exam chains, restored ids --------------------------------------------------------------------
    CatalogueEntry(
        "gave-part-of-angle",
        "Reporting one of the angles that make up the answer, not their sum or the angle asked for.",
        area="geometry",
    ),
    CatalogueEntry(
        "centre-and-circumference-angles-confused",
        "Taking the angle at the centre as equal to the angle at the circumference on the same arc, or the circumference angle as double the centre angle.",
        area="geometry",
    ),
    CatalogueEntry(
        "excluded-only-the-all-same-codes",
        "Treating 'at least one repeat' as 'not all the same', and removing only the codes with every symbol identical.",
        area="counting",
    ),
    CatalogueEntry(
        "choices-reduced-though-repeats-allowed",
        "Reducing the number of choices at each position (as for arrangements without repeats) when repeats are allowed.",
        area="counting",
    ),
    CatalogueEntry(
        "gst-treated-as-inclusive",
        "Treating a price as already containing 10% GST and dividing by 11 to find the tax, when the tax is added on top.",
        area="percentages",
    ),
    CatalogueEntry(
        "gst-left-out-of-total",
        "Giving the amount before GST as the final total.",
        area="percentages",
    ),
    CatalogueEntry(
        "surcharge-left-out",
        "Ignoring a fee or surcharge that is added to the price when finding the amount payable.",
        area="percentages",
    ),
    CatalogueEntry(
        "percentage-decimal-point-slip",
        "Moving the decimal point the wrong way when converting a percentage, so 10% of an amount becomes 1% of it.",
        area="percentages",
    ),
    CatalogueEntry(
        "hours-not-converted-to-minutes",
        "Giving a time in hours when minutes were asked, without multiplying by 60.",
        area="measurement",
    ),
    CatalogueEntry(
        "kept-the-original-scale",
        "Converting a real distance to a map length using the first map's scale instead of the second map's.",
        area="measurement",
    ),
    CatalogueEntry(
        "assumed-total-measure-is-one",
        "Treating a measure as a probability, so the measure outside a set is taken as 1 minus the measure of the set.",
        area="probability",
    ),
    CatalogueEntry(
        "read-the-wrong-point-mass",
        "Reading the mass of a different point from the table when adding the masses of a set.",
        area="probability",
    ),
    CatalogueEntry(
        "forgot-constant-of-integration",
        "Leaving out the constant of integration from an indefinite integral.",
        area="calculus",
    ),
    CatalogueEntry(
        "double-angle-identity-misapplied",
        "Using the wrong constant or coefficient when a square of sine or cosine is rewritten with the double angle.",
        area="calculus",
    ),
    # -- r3 integration: specific ids replacing catch-alls --------------------------------------------------
    CatalogueEntry(
        "multiplied-by-doses-not-shared",
        "Multiplying the daily dose by the number of doses where it is to be shared between them.",
    ),
    CatalogueEntry(
        "rebound-height-multiplied-by-ratio",
        "Multiplying the rebound height by the bounce ratio where the drop height is its quotient.",
    ),
    CatalogueEntry(
        "growth-multiplier-applied-forward",
        "Multiplying the reading by the growth multiplier where undoing the growth needs a division.",
    ),
    CatalogueEntry(
        "divided-by-parameter-instead-of-multiplied",
        "Dividing y - c by the parameter t where b is (y - c) times t.",
    ),
    CatalogueEntry(
        "evaluations-added-not-multiplied",
        "Adding the values of two polynomials at the root where the value of their product was asked for.",
    ),
    CatalogueEntry(
        "coefficients-added-for-constant-product",
        "Adding the coefficients a and b where the constant K = ab is their product.",
    ),
    CatalogueEntry(
        "all-codes-reported-for-at-least-one-repeat",
        "Giving the number of all possible codes where the number with at least one repeated character was asked for.",
    ),
    CatalogueEntry(
        "limit-linear-term-ignored",
        "Leaving out the constant that the linear term bx adds to the limit 2ax + b.",
    ),
    CatalogueEntry(
        "inequality-linear-term-ignored",
        "Solving a quadratic inequality by looking only at the n squared term and ignoring the rest.",
    ),
    CatalogueEntry(
        "a-count-omitted",
        "Leaving out the arrangements of one stage when multiplying the counts for a restricted arrangement, such as the order of the vowels.",
    ),
    CatalogueEntry(
        "probability-inverted",
        "Giving the ratio of all arrangements to the favourable ones, the reciprocal of the probability.",
    ),
    CatalogueEntry(
        "overtime-pay-only-averaged",
        "Dividing only the overtime pay by the hours worked to get the average hourly pay.",
    ),
    CatalogueEntry(
        "cosine-applied-twice-for-leg",
        "Finding a leg of a right triangle as the hypotenuse times cos A times cos A, applying the ratio twice.",
    ),
    CatalogueEntry(
        "sine-times-bc-for-altitude",
        "Finding the altitude to the hypotenuse as BC times sin A, which is a different length in the triangle.",
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
