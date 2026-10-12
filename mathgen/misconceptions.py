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
    # -- exam tier follow-ups ---------------------------------------------------------------------------------------------
    CatalogueEntry(
        "slant-height-as-height",
        "Using the slant height of a face as the perpendicular height of a solid.",
        area="measurement",
    ),
    CatalogueEntry(
        "pythagoras-subtract-lengths",
        "Subtracting two sides of a right triangle instead of applying Pythagoras' theorem.",
        area="measurement",
    ),
    CatalogueEntry(
        "pythagoras-add-squares-for-leg",
        "Adding the squares of the hypotenuse and a leg to find the other leg, instead of subtracting.",
        area="measurement",
    ),
    CatalogueEntry(
        "pyramid-volume-no-third",
        "Finding a pyramid's volume as base area times height, leaving out the factor of one third.",
        area="measurement",
    ),
    CatalogueEntry(
        "cubic-cm-to-litres-wrong-factor",
        "Converting cubic centimetres to litres by dividing by 10 or 100 instead of 1000.",
        area="measurement",
    ),
    # -- exam tiers: measurement ------------------------------------------------------------------------------------------
    CatalogueEntry(
        "area-without-square-root",
        "Finding a radius from an area by dividing by pi and stopping, without taking the square root.",
        area="measurement",
    ),
    CatalogueEntry(
        "area-root-without-pi",
        "Finding a radius from an area by taking the square root of the area and never dividing by pi.",
        area="measurement",
    ),
    CatalogueEntry(
        "diameter-for-radius",
        "Answering with the diameter where the radius was asked for, or using a diameter as if it were the radius.",
        area="measurement",
    ),
    CatalogueEntry(
        "area-formula-for-circumference",
        "Using the area formula where the circumference was asked for, or the reverse.",
        area="measurement",
    ),
    CatalogueEntry(
        "diameter-for-circumference",
        "Giving the diameter where the circumference was asked for.",
        area="measurement",
    ),
    CatalogueEntry(
        "ignored-the-hole",
        "Giving the area or volume of the whole solid circle and leaving out the hole in the middle.",
        area="measurement",
    ),
    CatalogueEntry(
        "subtracted-radii-before-squaring",
        "Subtracting the inner radius from the outer radius and then squaring, instead of squaring each radius first.",
        area="measurement",
    ),
    CatalogueEntry(
        "added-radii-squares",
        "Adding the squares of the two radii where the inner one should be taken away.",
        area="measurement",
    ),
    CatalogueEntry(
        "diameter-minus-wall-once",
        "Taking one wall thickness off the outer diameter to get the inner diameter, when a wall lies on both sides.",
        area="measurement",
    ),
    CatalogueEntry(
        "cone-without-one-third",
        "Treating a cone as a cylinder and leaving out the factor of one third.",
        area="measurement",
    ),
    CatalogueEntry(
        "slant-height-for-height",
        "Using the slant height of a cone as its vertical height.",
        area="measurement",
    ),
    CatalogueEntry(
        "hypotenuse-added-not-subtracted",
        "Adding the squares of the radius and the slant height to find the height, when the slant height is the "
        "hypotenuse.",
        area="measurement",
    ),
    CatalogueEntry(
        "rounded-count-down",
        "Rounding a number of trips or containers down to a whole number, leaving some of the load unmoved.",
        area="measurement",
    ),
    CatalogueEntry(
        "rounded-count-up",
        "Rounding up a count of complete turns, trips or containers when only the complete ones count.",
        area="measurement",
    ),
    CatalogueEntry(
        "unit-factor-wrong-power",
        "Using the wrong power of ten between two metric units.",
        area="measurement",
    ),
    CatalogueEntry(
        "unit-conversion-wrong-way",
        "Multiplying where the conversion needs a division by the same factor, or the reverse.",
        area="measurement",
    ),
    CatalogueEntry(
        "leftover-not-converted",
        "Finding the amount left over correctly but leaving it in the smaller unit when the larger unit was asked for.",
        area="measurement",
    ),
    CatalogueEntry(
        "ignored-time-zone-change",
        "Subtracting the clock times across a time-zone border without allowing for the clocks differing.",
        area="measurement",
    ),
    CatalogueEntry(
        "time-zone-applied-backwards",
        "Moving the clock the wrong way across a time-zone border: adding the difference where it should be taken "
        "away.",
        area="measurement",
    ),
    CatalogueEntry(
        "zone-shift-within-one-clock",
        "Allowing for a time-zone difference between two times that are both on the same clock, such as a landing "
        "time and the next departure from the same city.",
        area="measurement",
    ),
    CatalogueEntry(
        "left-out-stopover",
        "Leaving the time spent waiting between flights out of the total journey time.",
        area="measurement",
    ),
    CatalogueEntry(
        "perimeter-for-area",
        "Adding the sides of a rectangle where its area was asked for.",
        area="measurement",
    ),
    CatalogueEntry(
        "area-for-perimeter",
        "Multiplying the sides of a rectangle where its perimeter was asked for.",
        area="measurement",
    ),
    CatalogueEntry(
        "border-on-one-side",
        "Adding the width of a border or path once to each dimension, when it runs along both sides.",
        area="measurement",
    ),
    CatalogueEntry(
        "half-perimeter-as-side",
        "Using half the perimeter, or the whole perimeter, as if it were a single side of the rectangle.",
        area="measurement",
    ),
    CatalogueEntry(
        "linear-factor-on-area",
        "Converting an area between units with the length factor instead of its square (100 instead of 10 000).",
        area="measurement",
    ),
    CatalogueEntry(
        "unit-not-converted",
        "Combining two measurements in different units without converting one of them first.",
        area="measurement",
    ),
    CatalogueEntry(
        "full-circle-for-semicircle",
        "Using the whole circumference where only half of the circle forms the edge.",
        area="measurement",
    ),
    CatalogueEntry(
        "surface-area-omits-ends",
        "Finding the curved surface of a cylinder and leaving out its two circular ends.",
        area="measurement",
    ),
    CatalogueEntry(
        "pi-left-out",
        "Using the formula for a circular solid without the factor of pi.",
        area="measurement",
    ),
    CatalogueEntry(
        "counts-from-wrong-mark",
        "Counting scale divisions from the labelled mark on the wrong side of the reading.",
        area="measurement",
    ),
    CatalogueEntry(
        "major-interval-as-division",
        "Treating the gap between two labelled marks as the size of one division.",
        area="measurement",
    ),
    CatalogueEntry(
        "cut-off-cone-uses-frustum-height",
        "Using the height of the frustum as the height of the small cone that was cut off.",
        area="measurement",
    ),
    CatalogueEntry(
        "forgot-unit-conversion",
        "Combining measurements given in different units without converting them to a common unit first.",
        area="measurement",
    ),
    CatalogueEntry(
        "area-divided-by-one-side",
        "Finding how many pieces cover a region by dividing its area by only one side of a piece.",
        area="measurement",
    ),
    CatalogueEntry(
        "cost-from-area-not-count",
        "Pricing a number of items as if the price applied to each unit of area instead of to each item.",
        area="measurement",
    ),
    CatalogueEntry(
        "per-100-g-not-scaled",
        "Using the energy stated per 100 g as the energy of a serve of a different size.",
        area="measurement",
    ),
    CatalogueEntry(
        "mixed-units-in-subtraction",
        "Subtracting a figure in one energy unit from a budget in another without converting.",
        area="measurement",
    ),
    CatalogueEntry(
        "minutes-over-100",
        "Turning minutes into a decimal part of an hour by dividing by 100 instead of 60.",
        area="time",
    ),
    CatalogueEntry(
        "clock-time-read-as-decimal",
        "Reading a time such as 5 h 45 min as 5.45 hours.",
        area="time",
    ),
    CatalogueEntry(
        "left-out-the-break",
        "Leaving an unpaid break out of the time worked, or taking it off the wrong way.",
        area="time",
    ),
    CatalogueEntry(
        "chord-segments-added",
        "Setting the sum of the two parts of one chord equal to the sum of the two parts of the other, instead of the "
        "products.",
        area="circle-geometry",
    ),
    CatalogueEntry(
        "chord-products-crossed",
        "Pairing a part of one chord with a part of the other chord in the same product.",
        area="circle-geometry",
    ),
    CatalogueEntry(
        "negative-root-as-length",
        "Keeping the negative solution of the quadratic as a length.",
        area="circle-geometry",
    ),
    CatalogueEntry(
        "segment-for-whole-chord",
        "Giving one part of a chord where the length of the whole chord was asked.",
        area="circle-geometry",
    ),
    CatalogueEntry(
        "triangle-without-half",
        "Finding a triangle's area as base times height, with no half.",
        area="measurement",
    ),
    CatalogueEntry(
        "slant-side-as-height",
        "Using the slanting side of a triangle in place of its perpendicular height.",
        area="measurement",
    ),
    CatalogueEntry(
        "height-from-area-without-doubling",
        "Finding a missing height by dividing a triangle's area by the base, without doubling the area first.",
        area="measurement",
    ),
    CatalogueEntry(
        "division-count-off-by-one",
        "Sharing the gap between two labelled marks over one more or one fewer division than there are.",
        area="measurement",
    ),
    # -- exam tiers: number -----------------------------------------------------------------------------------------------
    CatalogueEntry(
        "exchange-rate-direction-reversed",
        "Multiplying by an exchange rate where dividing was needed, or dividing where multiplying was needed.",
        area="money",
    ),
    CatalogueEntry(
        "converted-back-at-outbound-rate",
        "Converting money back to Australian dollars at the rate used to buy it, though the rate for selling it back is different.",
        area="money",
    ),
    CatalogueEntry(
        "converted-whole-amount-back",
        "Converting the whole amount bought back to dollars instead of only the part left unspent.",
        area="money",
    ),
    CatalogueEntry(
        "left-foreign-amount-unconverted",
        "Reporting the unspent foreign money as if it were already Australian dollars.",
        area="money",
    ),
    CatalogueEntry(
        "reported-money-back-not-cost",
        "Giving the amount received back where the net cost was asked for.",
        kind="near_miss",
        area="money",
    ),
    CatalogueEntry(
        "truncates-instead-of-rounds",
        "Rounding every number down to its leading digit instead of to the nearest one.",
        area="estimation",
    ),
    CatalogueEntry(
        "same-fixed-place-for-every-operand",
        "Rounding every number to the same place (the nearest ten, say) instead of to its own leading digit.",
        area="estimation",
    ),
    CatalogueEntry(
        "multiplied-where-division-needed",
        "Multiplying by a number that the question divides by.",
        area="estimation",
    ),
    CatalogueEntry(
        "estimate-reported-as-exact",
        "Giving the estimate where the exact value was asked for.",
        kind="near_miss",
        area="estimation",
    ),
    CatalogueEntry(
        "error-over-estimate-not-exact",
        "Working out the percentage error as a share of the estimate instead of a share of the exact value.",
        area="estimation",
    ),
    CatalogueEntry(
        "inverted-rate",
        "Dividing the wrong way round, so the rate comes out as quantity per dollar instead of dollars per quantity.",
        area="rates",
    ),
    CatalogueEntry(
        "unit-rate-not-scaled",
        "Stopping at the rate for a single unit (per gram, per kilometre) without scaling it to the unit the question asks for.",
        area="rates",
    ),
    CatalogueEntry(
        "raw-quantity-used-as-rate",
        "Using the raw quantity (litres, grams) as if it were the rate.",
        area="rates",
    ),
    CatalogueEntry(
        "rates-added-not-subtracted",
        "Adding the two rates where their difference was asked for.",
        area="rates",
    ),
    CatalogueEntry(
        "difference-not-scaled-to-quantity",
        "Giving the difference in the rates without scaling it up to the quantity bought or the distance travelled.",
        area="rates",
    ),
    CatalogueEntry(
        "total-of-dearer-not-difference",
        "Giving the full cost of the dearer option where the extra cost over the cheaper one was asked for.",
        kind="near_miss",
        area="rates",
    ),
    CatalogueEntry(
        "single-reflection-only",
        "Applying only one of two reflections.",
        area="coordinates",
    ),
    CatalogueEntry(
        "wrong-axis-reflected",
        "Reflecting in the other axis from the one the question names.",
        area="coordinates",
    ),
    CatalogueEntry(
        "swapped-x-and-y",
        "Writing the coordinates the wrong way round, (y, x) for (x, y).",
        area="coordinates",
    ),
    CatalogueEntry(
        "distance-to-axis-not-to-image",
        "Giving the distance from the point to the axis instead of the distance to its reflection, which is twice as far.",
        area="coordinates",
    ),
    CatalogueEntry(
        "other-side-of-rectangle",
        "Giving the length of the other side of the rectangle.",
        area="coordinates",
    ),
    CatalogueEntry(
        "sides-added-for-area",
        "Adding the two side lengths of a rectangle where the area, their product, was asked for.",
        area="coordinates",
    ),
    CatalogueEntry(
        "decimal-point-one-place-too-far-right",
        "Placing the decimal point one place too far to the right in a decimal product.",
        area="decimals",
    ),
    CatalogueEntry(
        "decimal-point-one-place-too-far-left",
        "Placing the decimal point one place too far to the left in a decimal product.",
        area="decimals",
    ),
    CatalogueEntry(
        "added-quantity-and-price",
        "Adding the quantity and the price per unit instead of multiplying them.",
        area="decimals",
    ),
    CatalogueEntry(
        "forgot-second-purchase",
        "Leaving the second item out of the total.",
        area="decimals",
    ),
    CatalogueEntry(
        "reported-total-not-change",
        "Giving the total spent where the change was asked for.",
        kind="near_miss",
        area="decimals",
    ),
    CatalogueEntry(
        "divided-the-wrong-way",
        "Dividing the divisor by the dividend instead of the dividend by the divisor.",
        area="decimals",
    ),
    CatalogueEntry(
        "used-note-not-change",
        "Using the whole note where only the change was available to spend.",
        area="decimals",
    ),
    CatalogueEntry(
        "exponent-applied-to-coefficient",
        "Raising the starting amount to the power as well as the growth factor, so (c x b)^h instead of c x b^h.",
        area="powers",
    ),
    CatalogueEntry(
        "dropped-starting-amount",
        "Giving the power of the growth factor alone and leaving out the starting amount.",
        area="powers",
    ),
    CatalogueEntry(
        "multiplied-base-exponent",
        "Multiplying the base by the exponent instead of using repeated multiplication.",
        area="powers",
    ),
    CatalogueEntry(
        "off-by-one-power",
        "Multiplying one factor too few or too many.",
        area="powers",
    ),
    CatalogueEntry(
        "gave-new-total-not-increase",
        "Giving the new total where the increase was asked for.",
        kind="near_miss",
        area="powers",
    ),
    CatalogueEntry(
        "one-step-for-two",
        "Applying the growth factor once where two steps were needed.",
        area="powers",
    ),
    CatalogueEntry(
        "added-instead-of-multiplied-by-power",
        "Adding a power to the amount instead of multiplying the amount by it.",
        area="powers",
    ),
    CatalogueEntry(
        "added-fractions-spent-as-remaining",
        "Adding the two fractions spent and giving that as the fraction left.",
        area="fractions",
    ),
    CatalogueEntry(
        "second-fraction-of-original",
        "Taking the second fraction of the original amount instead of of what was left.",
        area="fractions",
    ),
    CatalogueEntry(
        "multiplied-by-fraction-to-find-whole",
        "Multiplying the known part by its fraction to find the whole, where dividing is needed.",
        area="fractions",
    ),
    CatalogueEntry(
        "ignored-second-fraction",
        "Scaling back with only the first fraction and ignoring the second.",
        area="fractions",
    ),
    CatalogueEntry(
        "added-fall-instead-of-subtracting",
        "Adding a fall to the starting value instead of subtracting it.",
        area="integers",
    ),
    CatalogueEntry(
        "subtracted-rise",
        "Subtracting a rise from the value instead of adding it.",
        area="integers",
    ),
    CatalogueEntry(
        "one-step-for-many",
        "Applying a change once where it happens several times.",
        area="integers",
    ),
    CatalogueEntry(
        "ignored-sign-of-start",
        "Treating a negative starting value as if it were positive.",
        area="integers",
    ),
    CatalogueEntry(
        "change-taken-backwards",
        "Subtracting the final value from the start instead of the start from the final value, so the sign of the change is wrong.",
        area="integers",
    ),
    CatalogueEntry(
        "subtracted-sizes-ignoring-signs",
        "Subtracting the sizes of two integers and ignoring their signs.",
        area="integers",
    ),
    CatalogueEntry(
        "used-the-other-rate",
        "Using the selling rate where the buying rate applies, or the reverse.",
        area="money",
    ),
    CatalogueEntry(
        "adds-the-rate",
        "Adding the exchange rate to the amount instead of multiplying or dividing by it.",
        area="money",
    ),
    CatalogueEntry(
        "foreign-amount-treated-as-dollars",
        "Treating an amount of foreign currency as if it were the same number of Australian dollars.",
        area="money",
    ),
    CatalogueEntry(
        "change-amount-not-new-value",
        "Giving the amount of the rise or fall where the value after the change was asked for.",
        kind="near_miss",
        area="percentages",
    ),
    CatalogueEntry(
        "second-percent-on-original",
        "Taking the second percentage change of the original amount instead of the amount after the first change.",
        area="percentages",
    ),
    CatalogueEntry(
        "successive-percent-added",
        "Combining two successive percentage changes by adding or subtracting the percentages instead of "
        "multiplying the two scale factors.",
        area="percentages",
    ),
    CatalogueEntry(
        "adds-percent-of-final",
        "Undoing a percentage decrease by adding that percentage of the final value, though the percentage was "
        "of the original.",
        area="percentages",
    ),
    CatalogueEntry(
        "reported-last-step-only",
        "Giving the change made by the last step alone where the overall change was asked for.",
        kind="near_miss",
        area="percentages",
    ),
    CatalogueEntry(
        "net-percent-of-final",
        "Finding the overall change as the net percentage of the final amount instead of the original amount.",
        area="percentages",
    ),
    CatalogueEntry(
        "forgot-zero-exponent-choice",
        "Counting the choices for each prime's power as 1 up to its exponent, leaving out the choice of using none "
        "of that prime, so every factor count is too small.",
        area="factors",
    ),
    CatalogueEntry(
        "factor-count-sum-of-exponents",
        "Counting the factors by adding the exponents (or the exponents plus one) instead of multiplying "
        "the exponents plus one.",
        area="factors",
    ),
    CatalogueEntry(
        "index-ignored",
        "Multiplying the bases of a prime factorisation and ignoring the indices.",
        area="factors",
    ),
    CatalogueEntry(
        "answered-complement",
        "Giving the count of the factors that were excluded where the count of the others was asked for.",
        kind="near_miss",
        area="factors",
    ),
    CatalogueEntry(
        "assumed-half-of-factors",
        "Assuming that half of a number's factors have some property (even, odd, a multiple of a prime, a perfect "
        "square) without counting them.",
        area="factors",
    ),
    CatalogueEntry(
        "adds-radicands-when-multiplying",
        "Multiplying two surds by multiplying the coefficients and adding the numbers under the roots.",
        area="surds",
    ),
    CatalogueEntry(
        "adds-coefficients-when-multiplying",
        "Multiplying two surds by adding the coefficients and multiplying the numbers under the roots.",
        area="surds",
    ),
    CatalogueEntry(
        "coefficient-not-squared",
        "Squaring only the number under the root and leaving the coefficient unsquared.",
        area="surds",
    ),
    CatalogueEntry(
        "squares-multiplied-not-added",
        "Multiplying the two squared sides in Pythagoras' theorem instead of adding them.",
        area="surds",
    ),
    CatalogueEntry(
        "surd-extracted-without-rooting",
        "Pulling a perfect-square factor out of a root without taking its square root, so k squared stands in "
        "front of the root where k belongs.",
        area="surds",
    ),
    CatalogueEntry(
        "surd-dropped-radical",
        "Giving only the extracted coefficient and dropping the root that remains.",
        area="surds",
    ),
    CatalogueEntry(
        "counts-decimal-places-not-sig-figs",
        "Using the stated number of significant figures as a number of decimal places.",
        area="rounding",
    ),
    CatalogueEntry(
        "counts-leading-zeros",
        "Counting the zeros after the decimal point at the start of a small number as significant figures.",
        area="rounding",
    ),
    CatalogueEntry(
        "rounds-to-one-fewer-sig-fig",
        "Rounding a calculated result to one significant figure fewer than the data allows.",
        area="rounding",
    ),
    CatalogueEntry(
        "used-more-precise-measurement",
        "Rounding a calculated result to the number of significant figures of the more precise measurement "
        "instead of the less precise one.",
        area="rounding",
    ),
    CatalogueEntry(
        "added-instead-of-multiplied",
        "Adding the two measurements where their product was needed.",
        area="decimals",
    ),
    CatalogueEntry(
        "truncated-not-rounded",
        "Cutting a decimal off after the required digits without rounding the last digit up.",
        area="rounding",
    ),
    CatalogueEntry(
        "treats-recurring-as-terminating",
        "Reading the digits shown for a recurring decimal as if it ended there, and writing that finite decimal as "
        "a fraction.",
        area="decimals",
    ),
    CatalogueEntry(
        "wrong-repeat-length",
        "Using a denominator of 9 for a recurring block of two digits, which needs 99.",
        area="decimals",
    ),
    CatalogueEntry(
        "ignores-non-repeating-lead",
        "Treating digits before the repeating block as part of the repeat, instead of separating them out.",
        area="decimals",
    ),
    CatalogueEntry(
        "misclassifies-terminating-as-recurring",
        "Treating a terminating decimal as recurring and writing its digits over a string of nines.",
        area="decimals",
    ),
    CatalogueEntry(
        "dropped-lead-digits",
        "Leaving out the non-repeating digits before the repeating block altogether.",
        area="decimals",
    ),
    CatalogueEntry(
        "decimal-place-value-miscounted",
        "Writing a decimal over the wrong power of ten because the decimal places were miscounted.",
        area="decimals",
    ),
    CatalogueEntry(
        "added-numerators-and-denominators",
        "Adding two fractions by adding the numerators and adding the denominators.",
        area="fractions",
    ),
    CatalogueEntry(
        "subtracted-in-wrong-order",
        "Subtracting the smaller quantity from the larger when the question asked for the reverse, "
        "reaching the opposite sign.",
        area="general",
    ),
    CatalogueEntry(
        "added-instead-of-subtracted",
        "Adding the two quantities where their difference was asked for.",
        area="general",
    ),
    CatalogueEntry(
        "left-to-right-instead-of-precedence",
        "Working a sum of products strictly left to right instead of doing the multiplications first.",
        area="estimation",
    ),
    CatalogueEntry(
        "forgot-to-share",
        "Leaving out the division by the number of equal shares.",
        area="estimation",
    ),
    CatalogueEntry(
        "raw-total-used-as-rate",
        "Using the total price of a pack as if it were the price per unit.",
        area="rates",
    ),
    CatalogueEntry(
        "rate-for-wrong-unit",
        "Giving the rate for a different unit from the one asked for (per kilogram where per 100 g was asked).",
        area="rates",
    ),
    # -- exam tiers: university and statistics ----------------------------------------------------------------------------
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
        "unit-slip-mm-as-cm",
        "Dividing an extension in millimetres by a length treated as centimetres, so the strain is out by 10.",
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
    # -- exam tiers: primary ----------------------------------------------------------------------------------------------
    CatalogueEntry(
        "story-operation-reversed",
        "Adding where the story takes away, or taking away where it adds.",
        area="early number",
    ),
    CatalogueEntry(
        "story-one-quantity-ignored",
        "Leaving one of the quantities in the story out, so the answer is one of the numbers given or a stage "
        "that was skipped.",
        area="early number",
    ),
    CatalogueEntry(
        "story-off-by-one",
        "An answer one more or one less than the right one, from a slip in counting or in a step.",
        area="early number",
    ),
    CatalogueEntry(
        "story-earlier-quantity-reported",
        "Giving the answer to an earlier question in the problem instead of the quantity now being asked for.",
        area="early number",
    ),
    CatalogueEntry(
        "groups-count-as-size",
        "Giving the number of equal groups as the size of each group.",
        area="early number",
    ),
    CatalogueEntry(
        "groups-total-minus-count",
        "Taking the number of groups away from the total instead of dividing the total into that many groups.",
        area="early number",
    ),
    CatalogueEntry(
        "groups-one-group-for-several",
        "Giving the size of one group when the question is about several groups.",
        area="early number",
    ),
    CatalogueEntry(
        "groups-count-for-items",
        "Counting the groups that are left instead of the items in them.",
        area="early number",
    ),
    CatalogueEntry(
        "inverse-operation-not-reversed",
        "Undoing an operation by doing it again: adding when the number was added to, or taking away when it was "
        "taken from, instead of reversing it.",
        area="early number",
    ),
    CatalogueEntry(
        "names-known-addend",
        "Giving the number that was added as the unknown starting number.",
        area="early number",
    ),
    CatalogueEntry(
        "states-the-result",
        "Giving the result of the operation as the starting number, as if nothing had been done to it.",
        area="early number",
    ),
    CatalogueEntry(
        "pattern-unit-assumed-two",
        "Assuming the repeating unit has two items, whatever the pattern shows.",
        area="early patterns",
    ),
    CatalogueEntry(
        "pattern-partial-unit-counted-whole",
        "Counting a partly finished repeat as a whole one.",
        area="early patterns",
    ),
    CatalogueEntry(
        "pattern-partial-unit-dropped",
        "Ignoring the items in a partly finished repeat at the end of the pattern.",
        area="early patterns",
    ),
    CatalogueEntry(
        "division-remainder-as-answer",
        "Giving the remainder as the answer to a division with a remainder.",
        area="early patterns",
    ),
    CatalogueEntry(
        "minute-parts-only-subtraction",
        "Finding the time between two clock times by subtracting only the minute parts and ignoring the hours.",
        area="time",
    ),
    CatalogueEntry(
        "smaller-minus-larger-minutes",
        "Subtracting the smaller minute part from the larger whichever time it belongs to, instead of regrouping an "
        "hour: 8:50 to 11:20 worked as 3 hours and 30 minutes.",
        area="time",
    ),
    CatalogueEntry(
        "ignored-minute-parts",
        "Finding the time between two clock times from the hours alone and ignoring the minutes.",
        area="time",
    ),
    CatalogueEntry(
        "waiting-counted-as-travel",
        "Counting time spent waiting or stopped as time spent travelling.",
        area="time",
    ),
    CatalogueEntry(
        "waiting-added-to-elapsed",
        "Adding a stop or a wait onto a total that already includes it.",
        area="time",
    ),
    CatalogueEntry(
        "waiting-counted-twice",
        "Adding a new stop onto a total that already includes the earlier stop.",
        area="time",
    ),
    CatalogueEntry(
        "forgot-stop-time",
        "Leaving a stop or wait out of a total time.",
        area="time",
    ),
    CatalogueEntry(
        "waiting-subtracted-not-added",
        "Taking a stop away from the driving time when it should be added on.",
        area="time",
    ),
    CatalogueEntry(
        "wrong-row-in-timetable",
        "Reading the times of a different journey in the timetable from the one asked about.",
        kind="near_miss",
        area="time",
    ),
    CatalogueEntry(
        "treats-every-month-as-30-days",
        "Counting days between dates as if every month had 30 days.",
        area="time",
    ),
    CatalogueEntry(
        "inclusive-day-count",
        "Counting both the first and the last date, so the number of days is one too many.",
        area="time",
    ),
    CatalogueEntry(
        "ignored-month-boundary",
        "Subtracting the day-of-the-month numbers of two dates in different months as though they were in one month.",
        area="time",
    ),
    CatalogueEntry(
        "rounded-up-complete-weeks",
        "Rounding up the number of weeks, so a part week is counted as a complete week.",
        area="time",
    ),
    CatalogueEntry(
        "forgot-to-convert-days-to-weeks",
        "Giving a number of days when asked for weeks.",
        area="time",
    ),
    CatalogueEntry(
        "base-100-time",
        "Using 100 in place of 60 (or 24) as the factor between time units, as if an hour were 100 minutes, so a time is "
        "converted or written as a decimal of the larger unit.",
        area="time",
    ),
    CatalogueEntry(
        "dropped-smaller-unit",
        "Converting only the larger unit of a compound time such as 2 h 35 min and leaving the minutes out.",
        area="time",
    ),
    CatalogueEntry(
        "added-unconverted-units",
        "Adding the numbers of a compound time such as 2 h 35 min without converting one unit to the other.",
        area="time",
    ),
    CatalogueEntry(
        "forgot-to-multiply-by-count",
        "Finding the amount for one item and stopping, though the total for several identical items was asked.",
        area="time",
    ),
    CatalogueEntry(
        "wrote-minutes-as-decimal-part",
        "Writing a time of 6 hours 20 minutes as 6.20 hours: the leftover minutes become the decimal part.",
        area="time",
    ),
    CatalogueEntry(
        "rounded-down-not-to-nearest",
        "Cutting a decimal off at the stated place instead of rounding to the nearest.",
        area="time",
    ),
    CatalogueEntry(
        "counts-turns-not-right-angles",
        "Counting how many turns were made instead of how many right angles they add up to.",
        area="angles",
    ),
    CatalogueEntry(
        "confuses-half-and-quarter",
        "Swapping the values of a half turn (2 right angles) and a quarter turn (1 right angle).",
        area="angles",
    ),
    CatalogueEntry(
        "did-not-convert-to-turns",
        "Leaving a turn measured in right angles or half turns when asked for full turns.",
        area="angles",
    ),
    CatalogueEntry(
        "dropped-fractional-turn",
        "Giving only the whole turns and dropping the part turn left over.",
        area="angles",
    ),
    CatalogueEntry(
        "drops-one-angle",
        "Leaving one of the given angles out of the sum before taking it from a full turn.",
        area="angles",
    ),
    CatalogueEntry(
        "gave-sum-not-remainder",
        "Giving the sum of the known angles instead of the angle left over to make a full turn.",
        area="angles",
    ),
    CatalogueEntry(
        "used-one-coefficient",
        "Dividing by the coefficient of only one of the two angle expressions when solving for x.",
        area="angles",
    ),
    CatalogueEntry(
        "gave-x-not-angle",
        "Giving the value of x when the size of the angle was asked for.",
        area="angles",
    ),
    CatalogueEntry(
        "read-wrong-angle",
        "Working out the other of two angles in the problem from the one asked about.",
        kind="near_miss",
        area="angles",
    ),
    CatalogueEntry(
        "gave-whole-not-part",
        "Giving the whole angle when the size of one of its parts was asked for.",
        area="angles",
    ),
    CatalogueEntry(
        "answers-amount-used-not-remaining",
        "Giving the amount used up (or already done) where the amount left over was asked for.",
        area="fractions and percentages",
    ),
    CatalogueEntry(
        "ignores-numerator",
        "Finding one unit fraction of a quantity (dividing by the denominator) without multiplying by the numerator.",
        area="fractions and percentages",
    ),
    CatalogueEntry(
        "adds-numerators-and-denominators",
        "Adding fractions by adding the numerators and adding the denominators, so that 1/3 + 1/4 becomes 2/7.",
        area="fractions and percentages",
    ),
    CatalogueEntry(
        "subtracts-numerators-and-denominators",
        "Subtracting fractions by subtracting the numerators and subtracting the denominators.",
        area="fractions and percentages",
    ),
    CatalogueEntry(
        "applies-fraction-to-original-whole",
        "Taking the second fraction of the original amount when it was a fraction of what was left after the first.",
        area="fractions and percentages",
    ),
    CatalogueEntry(
        "omitted-a-term",
        "Leaving one of the fractions or amounts out of a total that has three or more parts.",
        area="fractions and percentages",
    ),
    CatalogueEntry(
        "reports-discount-not-sale-price",
        "Giving the amount taken off a price where the price after the discount was asked for.",
        area="percentages",
    ),
    CatalogueEntry(
        "adds-successive-percentages",
        "Adding two successive percentage discounts together, though the second applies to the already reduced price.",
        area="percentages",
    ),
    CatalogueEntry(
        "gave-discount-percent-not-remaining",
        "Giving the percentage taken off where the percentage of the original price still paid was asked for.",
        area="percentages",
    ),
    CatalogueEntry(
        "complement-instead-of-value",
        "Giving what is left over (the complement, such as 65% for 35%) where the value itself was asked for.",
        area="fractions and percentages",
    ),
    CatalogueEntry(
        "drops-the-percent-divisor",
        "Writing a percentage as a fraction over 1, never dividing by 100.",
        area="percentages",
    ),
    CatalogueEntry(
        "wrong-power-of-ten-denominator",
        "Writing a percentage over 1000 (or 10) instead of over 100.",
        area="percentages",
    ),
    CatalogueEntry(
        "off-by-one-halving-step",
        "Halving one time too few, or one time too many.",
        area="fractions and percentages",
    ),
    CatalogueEntry(
        "halvings-counted-not-compounded",
        "Treating k halvings as one k-th part, or as 1/(2k), instead of halving the previous piece each time.",
        area="fractions and percentages",
    ),
    CatalogueEntry(
        "compared-wrong-pair-of-values",
        "Finding the gap between the wrong two values, such as the largest and the second smallest.",
        area="fractions and percentages",
    ),
    CatalogueEntry(
        "compares-numbers-ignoring-form",
        "Ordering values by the numbers as printed (3 in 3/5, 0.65, 62), ignoring that they are a fraction, a decimal and a percentage.",
        area="fractions and percentages",
    ),
    CatalogueEntry(
        "decimal-place-shift",
        "Putting the decimal point one place out when moving between a percentage and a decimal.",
        area="fractions and percentages",
    ),
    CatalogueEntry(
        "reversed-order",
        "Ordering from largest to smallest when smallest to largest was asked for.",
        area="ordering",
    ),
    CatalogueEntry(
        "adjacent-swap",
        "Ordering correctly except for two neighbouring values that are swapped.",
        area="ordering",
    ),
    CatalogueEntry(
        "measured-from-wrong-end",
        "Counting a position from the far end of a line, strip or trail instead of from the start.",
        area="fractions and percentages",
    ),
    CatalogueEntry(
        "added-scale-factor",
        "Adding the scale factor to the numerator and denominator instead of multiplying both by it.",
        area="fractions and percentages",
    ),
    CatalogueEntry(
        "forgot-to-share-remainder",
        "Giving the whole amount left over where each person's equal share of it was asked for.",
        area="fractions and percentages",
    ),
    CatalogueEntry(
        "share-taken-of-whole",
        "Dividing the whole by the number of people instead of dividing the amount left over.",
        area="fractions and percentages",
    ),
    CatalogueEntry(
        "scale-factor-used-as-position",
        "Giving the unit fraction's size (the number of sections in one part) where the position of a multiple of it was asked for.",
        area="fractions and percentages",
    ),
    CatalogueEntry(
        "zero-placeholder-dropped",
        "Leaving out the zero that holds an empty place, so the digits after it slide into the wrong places.",
        area="decimals",
    ),
    CatalogueEntry(
        "decimal-digits-in-wrong-places",
        "Writing the right decimal digits in the wrong order, so a digit sits in a place it does not belong to.",
        area="decimals",
    ),
    CatalogueEntry(
        "thousandths-added-as-hundredths",
        "Adding a number of thousandths as though they were hundredths, one place too large.",
        area="decimals",
    ),
    CatalogueEntry(
        "decimal-carry-lost",
        "Adding in one decimal place and dropping the carry into the place to its left.",
        area="decimals",
    ),
    CatalogueEntry(
        "counts-wrong-decimal-unit",
        "Counting tenths or thousandths when the number of hundredths was asked for.",
        area="decimals",
    ),
    CatalogueEntry(
        "digit-read-for-count",
        "Giving the digit in a place where the number of whole units of that place was asked for.",
        area="decimals",
    ),
    CatalogueEntry(
        "rounds-instead-of-whole-units",
        "Rounding up a part unit where only the whole units were asked for.",
        area="decimals",
    ),
    CatalogueEntry(
        "rounds-only-one-operand",
        "Rounding only the first number before combining and leaving the others exact.",
        area="decimals",
    ),
    CatalogueEntry(
        "computes-exact-result-instead-of-estimating",
        "Working out the exact total where an estimate from rounded numbers was asked for.",
        area="decimals",
    ),
    CatalogueEntry(
        "adds-rounding-errors-ignoring-direction",
        "Adding the sizes of the rounding changes as though they all pushed the same way, when some rounded up and some down.",
        area="decimals",
    ),
    CatalogueEntry(
        "joins-digits-as-decimal",
        "Writing the count of tenths and the count of hundredths side by side as the digits of the decimal.",
        area="decimals",
    ),
    CatalogueEntry(
        "hundredths-read-as-tenths",
        "Giving a number of hundredths the value of the same number of tenths.",
        area="decimals",
    ),
    CatalogueEntry(
        "converts-with-wrong-power-of-ten",
        "Using the wrong power of ten to change units, such as multiplying by 100 to go from kilometres to metres.",
        area="decimals",
    ),
    CatalogueEntry(
        "counts-marks-not-gaps",
        "Counting the marks reached on a scale instead of the gaps travelled, so the reading is one gap too far.",
        area="decimals",
    ),
    CatalogueEntry(
        "gap-read-as-tenth",
        "Reading each gap between marks as a tenth when the labelled marks are a tenth apart and the gap is a fraction of that.",
        area="decimals",
    ),
    CatalogueEntry(
        "restarts-count-from-the-labelled-mark",
        "Counting a second move from the start of the scale instead of from the point already reached.",
        area="decimals",
    ),
    CatalogueEntry(
        "measures-to-the-wrong-labelled-mark",
        "Finding the distance to a labelled mark other than the one asked for.",
        area="decimals",
    ),
    CatalogueEntry(
        "uses-wrong-quantity",
        "Multiplying each price by the other item's quantity, or ignoring the quantities altogether.",
        area="decimals",
    ),
    CatalogueEntry(
        "rounds-to-nearest-ten-cents",
        "Rounding a cash total to the nearest ten cents when the nearest five cents was asked for.",
        area="decimals",
    ),
    CatalogueEntry(
        "rounds-cash-total-up",
        "Always rounding a total up to the next five cents instead of to the nearest.",
        area="decimals",
    ),
    CatalogueEntry(
        "change-from-unrounded-total",
        "Working out the change from the exact total when the amount actually paid was the rounded cash total.",
        area="decimals",
    ),
    CatalogueEntry(
        "rounded-value-taken-as-bound",
        "Giving the rounded value itself as the smallest or largest the true value could be.",
        area="decimals",
    ),
    CatalogueEntry(
        "bound-off-by-whole-unit",
        "Taking a whole rounding unit, not half of it, as how far the true value can be from the rounded one.",
        area="decimals",
    ),
    CatalogueEntry(
        "upper-bound-includes-boundary",
        "Counting the halfway value as rounding down, so the largest value is given as the boundary itself.",
        area="decimals",
    ),
    CatalogueEntry(
        "uses-wrong-bound",
        "Using the largest possible value where the smallest was needed, or the reverse.",
        area="decimals",
    ),
    CatalogueEntry(
        "decimal-point-moved-wrong-way",
        "Moving the decimal point left when multiplying by a power of ten, or right when dividing.",
        area="decimals",
    ),
    CatalogueEntry(
        "decimal-point-moved-wrong-distance",
        "Moving the decimal point one place too many or too few for the power of ten.",
        area="decimals",
    ),
    CatalogueEntry(
        "smaller-digit-taken-from-larger",
        "Subtracting the smaller digit from the larger in each column, whichever number it belongs to.",
        area="decimals",
    ),
    CatalogueEntry(
        "borrow-not-reduced-after-zero",
        "Regrouping across a zero but forgetting to take one from the digit that lent it.",
        area="decimals",
    ),
    CatalogueEntry(
        "uses-original-amount-not-remaining",
        "Taking the second amount away from the starting amount instead of from what was left.",
        area="decimals",
    ),
    CatalogueEntry(
        "uses-earlier-value",
        "Using the starting amount or an earlier answer where the latest one was needed.",
        area="decimals",
    ),
    CatalogueEntry(
        "uses-unrounded-operands",
        "Calculating with the original numbers when the question said to round them first.",
        area="estimation",
    ),
    CatalogueEntry(
        "rounded-only-one-factor",
        "Rounding one of two factors and leaving the other as it was.",
        area="estimation",
    ),
    CatalogueEntry(
        "one-place-applied-to-both",
        "Rounding both numbers to the first place named, though each was to be rounded to its own place.",
        area="estimation",
    ),
    CatalogueEntry(
        "gave-estimate-for-exact",
        "Giving the rounded estimate where the exact answer was asked for.",
        kind="near_miss",
        area="estimation",
    ),
    CatalogueEntry(
        "partial-product-not-shifted",
        "Writing the tens partial product in the ones column in a long multiplication, so it is ten times too small.",
        area="whole-numbers",
    ),
    CatalogueEntry(
        "subtracted-smaller-from-larger-each-column",
        "Subtracting column by column and always taking the smaller digit from the larger, instead of regrouping.",
        area="whole-numbers",
    ),
    CatalogueEntry(
        "stopped-before-final-step",
        "Giving a value from partway through the working as the answer to the last question.",
        kind="near_miss",
        area="general",
    ),
    CatalogueEntry(
        "ordered-by-starting-value",
        "Ordering by the starting readings and ignoring the change that happened afterwards.",
        area="integers",
    ),
    CatalogueEntry(
        "fall-treated-as-rise",
        "Adding a fall to the starting temperature as if it were a rise.",
        area="integers",
    ),
    CatalogueEntry(
        "magnitude-ignores-sign",
        "Ordering negative numbers by their size without the sign, so -7 is placed above -3 or above 2.",
        area="integers",
    ),
    CatalogueEntry(
        "ordered-by-first-number-shown",
        "Ordering large numbers by the first number written in each description (412 thousand, 3.2 million) "
        "without converting to numerals.",
        area="place-value",
    ),
    CatalogueEntry(
        "placeholder-zero-dropped",
        "Converting to a numeral without the placeholder zeros: 412 thousand and 6 written as 4126.",
        area="place-value",
    ),
    CatalogueEntry(
        "regrouped-quantity-read-as-digits",
        "Reading a regrouped place-value description digit by digit: 3 hundreds, 14 tens and 6 ones written as 3146.",
        area="place-value",
    ),
    CatalogueEntry(
        "more-less-wrong-place",
        "Changing the wrong digit for 'more than' or 'less than': 100 more treated as 10 more, or 10 less as 100 less.",
        area="place-value",
    ),
    CatalogueEntry(
        "fraction-digits-as-decimal",
        "Writing a fraction as a decimal from its digits: 3/8 as 0.38.",
        area="rational-numbers",
    ),
    CatalogueEntry(
        "inverted-fraction",
        "Multiplying by the fraction upside down: finding 8/3 of an amount for 3/8 of it.",
        area="rational-numbers",
    ),
    CatalogueEntry(
        "decimal-as-percent",
        "Reading a decimal such as 0.2 as 0.2%, a fifth of what it should be.",
        area="rational-numbers",
    ),
    CatalogueEntry(
        "decimal-of-remainder-not-total",
        "Taking a decimal share of what was left when the question gave it as a share of the original total.",
        area="rational-numbers",
    ),
    CatalogueEntry(
        "fee-added-to-balance",
        "Adding a fee to an overdrawn balance, which makes the debt smaller instead of larger.",
        area="integers",
    ),
    CatalogueEntry(
        "paid-fraction-used-as-remaining",
        "Taking the fraction that was paid off as the fraction still owing.",
        area="rational-numbers",
    ),
    CatalogueEntry(
        "sign-dropped-on-negative",
        "Writing a debt or a below-zero reading without its negative sign.",
        area="integers",
    ),
    CatalogueEntry(
        "negative-subtracted-as-positive",
        "Subtracting an amount owing from a target as if it were a positive balance, instead of adding it.",
        area="integers",
    ),
    CatalogueEntry(
        "ignored-target-value",
        "Giving the size of the debt alone, forgetting the target balance that also has to be reached.",
        kind="near_miss",
        area="integers",
    ),
    CatalogueEntry(
        "partial-benchmark-only",
        "Estimating 15% as 10% only, or 35% as 25% only, leaving out the second benchmark percentage.",
        area="percentages",
    ),
    CatalogueEntry(
        "used-unrounded-price",
        "Taking the discount off the original price when the estimate was to be made from the rounded price.",
        area="estimation",
    ),
    CatalogueEntry(
        "added-discount-to-price",
        "Adding a discount to the price instead of taking it off.",
        area="percentages",
    ),
    CatalogueEntry(
        "rounds-groups-down",
        "Giving only the number of full groups when the leftover items still need a group of their own.",
        area="primary-number",
    ),
    CatalogueEntry(
        "cost-from-rounded-down-count",
        "Pricing the full groups only, so the group that holds the leftover items is not paid for.",
        area="primary-number",
    ),
    CatalogueEntry(
        "leftover-given-for-empty-spaces",
        "Giving the number of items left over where the number of empty places in the last group was asked for.",
        area="primary-number",
    ),
    CatalogueEntry(
        "group-size-given-as-count",
        "Giving the size of each group where the number of groups was asked for.",
        area="primary-number",
    ),
    CatalogueEntry(
        "given-quantity-reported",
        "Giving one of the numbers from the story as the answer instead of working with it.",
        area="primary-number",
    ),
    CatalogueEntry(
        "multiplied-by-wrong-quantity",
        "Multiplying by a number from the story that does not count what is being priced, such as the size of a group "
        "instead of the number of groups.",
        area="primary-number",
    ),
    CatalogueEntry(
        "divided-by-wrong-group-size",
        "Dividing by the size of a later group when the share asked for is the earlier one.",
        area="primary-number",
    ),
    CatalogueEntry(
        "adjacent-table-fact",
        "Giving the answer to the next or previous fact in the table, one group too many or too few.",
        area="primary-number",
    ),
    CatalogueEntry(
        "digits-transposed",
        "Swapping the digits of a two-digit number before using it, so 36 is read as 63.",
        area="primary-number",
    ),
    CatalogueEntry(
        "tens-not-scaled",
        "Using the basic fact as the answer when both numbers were tens, so the zeros are left out.",
        area="primary-number",
    ),
    CatalogueEntry(
        "one-factor-scaled",
        "Putting a zero on the answer for only one of two numbers that were tens.",
        area="primary-number",
    ),
    CatalogueEntry(
        "wrong-number-of-zeros",
        "Scaling a basic fact by the wrong power of ten, so the answer has too many or too few zeros.",
        area="primary-number",
    ),
    CatalogueEntry(
        "doubled-instead-of-halved",
        "Doubling where the amount was to be halved, for example finding the items from the pairs.",
        area="primary-number",
    ),
    CatalogueEntry(
        "halving-skipped",
        "Giving the number before halving, as if the halving step had not been needed.",
        area="primary-number",
    ),
    CatalogueEntry(
        "carry-dropped",
        "Adding column by column and dropping the carry, so each column keeps only its ones digit.",
        area="primary-number",
    ),
    CatalogueEntry(
        "borrow-not-taken-from-next-column",
        "Adding ten to a short column but not taking one from the next column.",
        area="primary-number",
    ),
    CatalogueEntry(
        "omits-indented-sides",
        "Leaving out the sides that run around a cut-out corner when finding the perimeter of an L-shape.",
        area="measurement",
    ),
    CatalogueEntry(
        "side-added-not-subtracted",
        "Adding the cut-out length to a side instead of subtracting it to find what is left of that side.",
        area="measurement",
    ),
    CatalogueEntry(
        "wrong-dimension-for-missing-side",
        "Subtracting the cut-out's measurement from the wrong side of the shape.",
        area="measurement",
    ),
    CatalogueEntry(
        "used-cut-out-as-remainder",
        "Giving the length of the cut-out as the length that is left of the side.",
        area="measurement",
    ),
    CatalogueEntry(
        "perimeter-as-area",
        "Multiplying the dimensions to get an area where the perimeter was asked.",
        area="measurement",
    ),
    CatalogueEntry(
        "adds-only-given-lengths",
        "Finding a perimeter by adding only the lengths stated, without finding the missing sides.",
        area="measurement",
    ),
    CatalogueEntry(
        "ignored-gap-in-fence",
        "Pricing the whole boundary and ignoring a gap that needs no fence.",
        area="measurement",
    ),
    CatalogueEntry(
        "gap-added-instead-of-removed",
        "Adding the length of a gap to the boundary instead of taking it away.",
        area="measurement",
    ),
    CatalogueEntry(
        "subtracted-unlike-quantities",
        "Subtracting a length in metres from an amount of money, or a depth from a volume.",
        area="measurement",
    ),
    CatalogueEntry(
        "rate-added-not-multiplied",
        "Adding a price per unit to a quantity instead of multiplying the two.",
        area="measurement",
    ),
    CatalogueEntry(
        "direction-sign-reversed",
        "Taking a move west or south as positive, or a move east or north as negative.",
        area="coordinates",
    ),
    CatalogueEntry(
        "used-change-as-coordinate",
        "Giving the size of a move, or an end coordinate, where a coordinate or a change was asked for.",
        area="coordinates",
    ),
    CatalogueEntry(
        "swapped-coordinates",
        "Giving the y value where the x value was asked, or the reverse.",
        area="coordinates",
    ),
    CatalogueEntry(
        "total-change-for-leg",
        "Giving the change from the very start where the change over one leg was asked for.",
        area="coordinates",
    ),
    CatalogueEntry(
        "reversed-subtraction",
        "Subtracting the end value from the start value, so the answer has the wrong sign.",
        area="general",
    ),
    CatalogueEntry(
        "uses-surface-area",
        "Adding or combining face areas where the volume was asked for.",
        area="measurement",
    ),
    CatalogueEntry(
        "adds-dimensions",
        "Adding the three dimensions of a prism instead of multiplying them.",
        area="measurement",
    ),
    CatalogueEntry(
        "base-area-for-volume",
        "Giving the area of the base as the volume, leaving out the height.",
        area="measurement",
    ),
    CatalogueEntry(
        "wrong-power-of-ten-conversion",
        "Converting cm³ to litres by dividing by 10 or 100 instead of 1000.",
        area="measurement",
    ),
    CatalogueEntry(
        "gave-part-not-remainder",
        "Giving the part that is already there where the remainder was asked for.",
        area="general",
    ),
    CatalogueEntry(
        "ignored-water-already-in",
        "Working with the whole capacity and ignoring the amount that is already in the tank.",
        area="measurement",
    ),
    CatalogueEntry(
        "multiplied-instead-of-divided-rate",
        "Multiplying an amount by a rate to find a time, instead of dividing by it.",
        area="measurement",
    ),
    CatalogueEntry(
        "uses-360",
        "Using 360 degrees instead of 180 degrees for angles on a straight line.",
        area="angles",
    ),
    CatalogueEntry(
        "ignored-known-angle",
        "Leaving a given angle out of the sum or out of the angle being asked for.",
        area="angles",
    ),
    CatalogueEntry(
        "used-other-angle-expression",
        "Substituting into the expression for a different angle from the one asked for.",
        area="angles",
    ),
    CatalogueEntry(
        "bisector-not-halved",
        "Using the whole angle where its half, the bisected part, was needed.",
        area="angles",
    ),
    CatalogueEntry(
        "angle-to-wrong-arm",
        "Giving the angle to the other arm of the line, the supplement of the angle needed.",
        area="angles",
    ),
    CatalogueEntry(
        "ignored-quantity",
        "Adding unit prices without multiplying each by how many were bought.",
        area="money",
    ),
    CatalogueEntry(
        "omits-an-item",
        "Leaving one of the purchased items out of a total.",
        area="money",
    ),
    CatalogueEntry(
        "adds-price",
        "Adding the price to the payment instead of subtracting it, or adding to the money left.",
        area="money",
    ),
    CatalogueEntry(
        "reports-cost-not-change",
        "Giving the cost of the purchase where the change was asked for.",
        kind="near_miss",
        area="money",
    ),
    CatalogueEntry(
        "quantity-not-cost",
        "Subtracting the number of kilograms bought instead of what they cost.",
        area="money",
    ),
    CatalogueEntry(
        "unit-price-not-total",
        "Subtracting the price of one kilogram instead of the cost of all the kilograms bought.",
        area="money",
    ),
    CatalogueEntry(
        "ignored-first-purchase",
        "Taking the second purchase from the whole note instead of from the change.",
        area="money",
    ),
    CatalogueEntry(
        "truncates-prices",
        "Always rounding prices down when estimating.",
        area="money",
    ),
    CatalogueEntry(
        "rounds-total-not-prices",
        "Rounding the exact total instead of rounding each price first.",
        area="money",
    ),
    CatalogueEntry(
        "estimate-as-exact",
        "Giving the estimate where the exact total was asked for.",
        kind="near_miss",
        area="money",
    ),
    CatalogueEntry(
        "adds-instead-of-subtracts",
        "Adding two amounts where the difference between them was asked for.",
        area="general",
    ),
    CatalogueEntry(
        "gave-discount-percent",
        "Giving the percentage taken off where the percentage paid was asked for.",
        kind="near_miss",
        area="percentages",
    ),
    CatalogueEntry(
        "adds-discount",
        "Adding the discount percentage to 100% instead of taking it away.",
        area="percentages",
    ),
    CatalogueEntry(
        "percent-of-sale-price",
        "Finding the percentage of the sale price where it applies to the original price.",
        area="percentages",
    ),
    CatalogueEntry(
        "multiplies-sale-by-percent-paid",
        "Multiplying the sale price by the percentage paid instead of dividing by it, so the original price comes out "
        "smaller than the sale price.",
        area="percentages",
    ),
    CatalogueEntry(
        "reports-price-paid-not-saving",
        "Giving the price paid where the amount saved was asked for.",
        kind="near_miss",
        area="percentages",
    ),
    CatalogueEntry(
        "gives-remainder-not-part",
        "Giving what is left after taking the percentage where the percentage itself was asked for.",
        kind="near_miss",
        area="percentages",
    ),
    CatalogueEntry(
        "multiplies-by-whole-percent",
        "Multiplying by the percentage number, as if 25% were 25, instead of by 0.25.",
        area="percentages",
    ),
    CatalogueEntry(
        "divides-by-the-percent",
        "Dividing the quantity by the percentage number instead of finding that percentage of it.",
        area="percentages",
    ),
    CatalogueEntry(
        "subtracts-percentage",
        "Subtracting the percentage number from the quantity, as if it were an amount.",
        area="percentages",
    ),
    CatalogueEntry(
        "percent-of-wrong-group",
        "Taking a percentage of the whole group, or of the first group, where it applies to the remaining group.",
        area="percentages",
    ),
    CatalogueEntry(
        "count-given-as-percent",
        "Giving a count where a percentage of the whole was asked for.",
        area="percentages",
    ),
    CatalogueEntry(
        "adds-percentages-of-different-groups",
        "Adding two percentages that are taken of different groups.",
        area="percentages",
    ),
    CatalogueEntry(
        "wrong-denominator-for-gap",
        "Writing a length or gap as a fraction over one of the parts' own denominators instead of over the whole.",
        area="fractions and percentages",
    ),
    # -- exam tiers: secondary, senior and university ---------------------------------------------------------------------
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
    # -- exam chains: senior and university -------------------------------------------------------------------------------
    CatalogueEntry(
        "air-speed-for-ground-speed",
        "Using the speed through the air (or through the water) as the speed over the ground, ignoring the wind or "
        "current.",
        area="vectors",
    ),
    CatalogueEntry(
        "added-magnitudes-not-vectors",
        "Adding the magnitudes of two vectors instead of adding the vectors and then taking the magnitude.",
        area="vectors",
    ),
    CatalogueEntry(
        "subtracted-instead-of-added-vectors",
        "Subtracting one vector from another where the two should be added (the wind taken as blowing the other way).",
        area="vectors",
    ),
    CatalogueEntry(
        "inverted-rate-formula",
        "Dividing speed by distance instead of distance by speed (or the reverse for another rate).",
        area="vectors",
    ),
    CatalogueEntry(
        "used-wrong-component",
        "Using the wrong component of a vector: the eastward part where the northward part is needed.",
        area="vectors",
    ),
    CatalogueEntry(
        "angle-from-wrong-axis",
        "Measuring an angle or bearing from the wrong axis, or giving its complement.",
        area="vectors",
    ),
    CatalogueEntry(
        "parallelogram-vertex-mispaired",
        "Pairing the vertices of a parallelogram wrongly, so a vertex that is adjacent is treated as opposite.",
        area="vectors",
    ),
    CatalogueEntry(
        "used-wrong-vector-pair",
        "Taking the dot product or angle between the wrong two vectors, such as the sides from a different corner.",
        area="vectors",
    ),
    CatalogueEntry(
        "forgot-to-divide-by-magnitudes",
        "Stopping at the dot product when finding an angle, without dividing by the product of the magnitudes.",
        area="vectors",
    ),
    CatalogueEntry(
        "supplementary-angle",
        "Giving 180 degrees minus the angle, the angle between a vector and the reverse of the other.",
        area="vectors",
    ),
    CatalogueEntry(
        "wrong-sign-of-cosine",
        "Using the cosine of the supplementary angle, so the sign of the dot product is reversed.",
        area="vectors",
    ),
    CatalogueEntry(
        "forgot-the-cosine",
        "Giving the product of the two magnitudes as the dot product, leaving out the cosine of the angle between "
        "them.",
        area="vectors",
    ),
    CatalogueEntry(
        "dropped-cross-term",
        "Squaring a sum of vectors as the sum of the squares, leaving out the cross term 2a.b.",
        area="vectors",
    ),
    CatalogueEntry(
        "used-perpendicular-gradient",
        "Giving the gradient of the perpendicular (the negative reciprocal) where the gradient of the line was asked "
        "for, such as the slope of the segment joining a point to its image.",
        area="transformations",
    ),
    CatalogueEntry(
        "inverted-gradient",
        "Giving a gradient upside down: run over rise.",
        area="transformations",
    ),
    CatalogueEntry(
        "confused-reflection-with-rotation",
        "Using a rotation matrix where a reflection matrix is needed, so the second row has the wrong signs.",
        area="transformations",
    ),
    CatalogueEntry(
        "forgot-the-normalising-factor",
        "Leaving out the factor that scales a reflection matrix built from a gradient back to unit length.",
        area="transformations",
    ),
    CatalogueEntry(
        "composition-order-reversed",
        "Applying the transformations of a composite in the wrong order: the left-hand map first.",
        area="transformations",
    ),
    CatalogueEntry(
        "ignored-first-transformation",
        "Applying only the second map of a composite, as if the first had not been done.",
        area="transformations",
    ),
    CatalogueEntry(
        "added-determinants",
        "Adding the determinants (area factors) of two maps instead of multiplying them to get the composite's.",
        area="transformations",
    ),
    CatalogueEntry(
        "signed-determinant-as-area-factor",
        "Giving a negative determinant as the area scale factor, though an area factor is never negative.",
        area="transformations",
    ),
    CatalogueEntry(
        "determinant-order-reversed",
        "Subtracting in the wrong order when finding a determinant: bc - ad instead of ad - bc.",
        area="matrices",
    ),
    CatalogueEntry(
        "forward-map-for-inverse",
        "Applying the matrix itself to the image where its inverse is needed to find the original point.",
        area="matrices",
    ),
    CatalogueEntry(
        "forgot-to-divide-by-determinant",
        "Using the adjugate as the inverse matrix, without dividing by the determinant.",
        area="matrices",
    ),
    CatalogueEntry(
        "area-scaled-the-wrong-way",
        "Multiplying by the area scale factor where the original area is found by dividing by it, or the reverse.",
        area="matrices",
    ),
    CatalogueEntry(
        "forgot-to-take-the-root",
        "Giving the total growth factor over several periods as the growth factor for one period.",
        area="finance",
    ),
    CatalogueEntry(
        "factor-given-as-rate",
        "Giving the growth factor (1.045) or the rate as a decimal (0.045) where a percentage rate was asked for.",
        area="finance",
    ),
    CatalogueEntry(
        "uses-simple-interest",
        "Adding the same interest amount each period instead of compounding.",
        area="finance",
    ),
    CatalogueEntry(
        "wrong-number-of-periods",
        "Compounding for the wrong number of periods, such as the original term when only the extra time applies.",
        area="finance",
    ),
    CatalogueEntry(
        "interest-from-wrong-base",
        "Taking the interest as the growth since an intermediate balance rather than since the original amount.",
        area="finance",
    ),
    CatalogueEntry(
        "ignored-the-interest",
        "Leaving the interest out of a repayment or term calculation: dividing the amount borrowed by the repayments.",
        area="finance",
    ),
    CatalogueEntry(
        "used-monthly-figure-as-annual",
        "Using a monthly amount where a yearly amount is needed, forgetting to multiply by 12.",
        area="finance",
    ),
    CatalogueEntry(
        "restarted-from-the-original-principal",
        "Growing the original amount at the new rate instead of the value reached at the end of the first phase.",
        area="finance",
    ),
    CatalogueEntry(
        "off-by-one-count",
        "Being one out in a count of periods or steps, or in a single step or total of a forward or backward pass.",
        area="finance",
    ),
    CatalogueEntry(
        "repayment-before-interest",
        "Taking the repayment off before adding the interest in a recurrence step.",
        area="finance",
    ),
    CatalogueEntry(
        "rate-change-not-applied",
        "Carrying on with the old rate after the rate has changed.",
        area="finance",
    ),
    CatalogueEntry(
        "forgot-to-add-first-phase-years",
        "Giving the time for the second phase alone as the total time from the start.",
        area="finance",
    ),
    CatalogueEntry(
        "forgot-to-subtract-repayment",
        "Adding the interest each period but never taking off the repayment.",
        area="finance",
    ),
    CatalogueEntry(
        "adds-a-year-after-rounding-up",
        "Rounding the crossing time up to the next whole year and then adding another year, so the answer is "
        "one year too late.",
        area="finance",
    ),
    CatalogueEntry(
        "adds-components-for-modulus",
        "Adding the absolute values of the real and imaginary parts to get the modulus, instead of taking the "
        "square root of the sum of their squares.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "adds-every-duration",
        "Giving the sum of all the activity durations as the project time, as if every activity had to be "
        "done one after another.",
        area="networks",
    ),
    CatalogueEntry(
        "adds-probabilities-without-weights",
        "Adding the probabilities from the separate branches without weighting each by how much of the whole "
        "it makes up.",
        area="probability",
    ),
    CatalogueEntry(
        "adds-z-scores",
        "Dividing the difference of two observations by the sum of their z-scores instead of the difference.",
        area="statistics",
    ),
    CatalogueEntry(
        "annuity-as-single-deposit",
        "Growing one payment for the whole term as if it were a single deposit, instead of summing the growth "
        "of every regular payment.",
        area="finance",
    ),
    CatalogueEntry(
        "assumes-equally-likely",
        "Treating every outcome as equally likely when the probabilities are not equal: 1/n for each of n outcomes.",
        area="probability",
    ),
    CatalogueEntry(
        "assumes-the-rival-route-binds",
        "Taking the longest route that avoids a shortened activity as the new project time, when the "
        "shortening was small enough that the original route is still the longest.",
        area="networks",
    ),
    CatalogueEntry(
        "assumes-with-replacement",
        "Treating a draw without replacement as if the first item were put back, so the second probability "
        "does not change.",
        area="probability",
    ),
    CatalogueEntry(
        "averages-the-rates",
        "Averaging the two rates as if both groups were the same size, instead of weighting each rate by the "
        "share of the whole.",
        area="probability",
    ),
    CatalogueEntry(
        "compares-with-deposits-not-balance",
        "Measuring the interest earned against the total of the deposits instead of against the balance the "
        "withdrawals started from.",
        area="finance",
    ),
    CatalogueEntry(
        "confuses-the-direction-of-the-conditional",
        "Giving P(B | A) where P(A | B) was asked: the fault rate of a machine instead of the chance a faulty "
        "item came from it.",
        area="probability",
    ),
    CatalogueEntry(
        "conjugates-the-wrong-part",
        "Changing the sign of the real part of a complex number instead of the imaginary part when taking the "
        "conjugate.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "counts-terms-not-gaps",
        "Dividing a change between two terms by the number of terms from one to the other inclusive, instead "
        "of by the number of steps between them.",
        area="sequences",
    ),
    CatalogueEntry(
        "divides-by-the-number-of-values",
        "Dividing a weighted sum of values by the number of values instead of using the probabilities as weights.",
        area="probability",
    ),
    CatalogueEntry(
        "doubles-for-two-tails",
        "Doubling a one-tailed normal probability, as for a two-sided question, when only one tail was asked for.",
        area="statistics",
    ),
    CatalogueEntry(
        "first-moment-for-second-moment",
        "Giving E(X) where E(X^2) was asked for, or E(X) where the variance was.",
        area="probability",
    ),
    CatalogueEntry(
        "follows-the-longest-activity-at-each-step",
        "Choosing the critical path by taking the activity with the largest duration at each step, rather "
        "than comparing whole routes.",
        area="networks",
    ),
    CatalogueEntry(
        "forgets-the-factor-2",
        "Losing the 2 in 2xy when equating imaginary parts, so y is found from the imaginary part divided by x alone.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "forgets-the-multiplier",
        "Leaving out the multiplier in front of a term, such as using the conjugate of q where 2 times the "
        "conjugate of q is needed.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "forgets-the-standard-deviation",
        "Giving the distance from the mean, or subtracting a z-score from an observation, without scaling by "
        "the standard deviation.",
        area="statistics",
    ),
    CatalogueEntry(
        "forgets-the-z-score-term",
        "Giving an observation as the mean, ignoring the z-score times the standard deviation that separates them.",
        area="statistics",
    ),
    CatalogueEntry(
        "forgets-to-conjugate",
        "Using q where its conjugate is needed, leaving the sign of the imaginary part unchanged.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "forgets-to-divide-by-x",
        "Stopping at xy = b/2 when finding y, without dividing through by x.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "forgets-to-divide-by-z-difference",
        "Giving the difference between two observations as the standard deviation, without dividing by the "
        "difference between their z-scores.",
        area="statistics",
    ),
    CatalogueEntry(
        "forgets-to-halve",
        "Giving the numerator of the quadratic formula for x^2 without dividing by 2.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "forgets-to-normalise",
        "Giving a weighted sum with the unscaled weights, forgetting that the probabilities were found by "
        "dividing by their total.",
        area="probability",
    ),
    CatalogueEntry(
        "forgets-to-subtract-the-mean-squared",
        "Giving E(X^2) as the variance, leaving out the subtraction of the square of the mean.",
        area="probability",
    ),
    CatalogueEntry(
        "forgot-half-in-series-sum",
        "Using n(a + l) for the sum of an arithmetic series, leaving out the 1/2.",
        area="sequences",
    ),
    CatalogueEntry(
        "forgot-to-divide-by-gap",
        "Giving the difference between two terms as the common difference, without dividing by how many steps "
        "apart they are.",
        area="sequences",
    ),
    CatalogueEntry(
        "gave-total-not-interest",
        "Giving the total amount paid out or repaid where the interest earned or charged was asked for.",
        area="finance",
    ),
    CatalogueEntry(
        "gives-argument-outside-principal-range",
        "Giving an angle that points the right way but is not in the principal range, such as 2 pi more or "
        "less than the principal argument.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "gives-degrees-not-radians",
        "Giving an angle in degrees where radians were asked for.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "gives-joint-not-conditional",
        "Giving the probability of both events together where the probability of one given the other was "
        "asked: leaving out the division by the probability of the condition.",
        area="probability",
    ),
    CatalogueEntry(
        "gives-the-argument-of-the-conjugate",
        "Giving the argument with the wrong sign, which is the argument of the conjugate.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "gives-the-argument-of-w",
        "Giving the argument of w = z^2, which is twice the argument of z.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "gives-the-complement",
        "Giving the complementary probability (the chance of the opposite outcome, the other tail or the other branch: one minus the correct share) where the stated one was asked for.",
        area="probability",
    ),
    CatalogueEntry(
        "gives-the-modulus-of-w",
        "Giving the modulus of w = z^2, which is the square of the modulus of z.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "ignores-compounding",
        "Adding the payments with no interest at all, so the balance is just the sum of what was paid in.",
        area="finance",
    ),
    CatalogueEntry(
        "ignores-later-activities-in-backward-pass",
        "Finding a latest start or a float from the finish of the activity alone, without counting the "
        "longest chain of activities that must follow it.",
        area="networks",
    ),
    CatalogueEntry(
        "ignores-the-activity-being-avoided",
        "Giving the length of the critical path where the longest route that avoids a particular critical "
        "activity was asked for.",
        area="networks",
    ),
    CatalogueEntry(
        "ignores-the-change",
        "Giving the original value after a change was made, as if the change had no effect.",
        area="networks",
    ),
    CatalogueEntry(
        "ignores-the-evidence",
        "Giving the prior share of a group as the probability, ignoring the information that the item is faulty.",
        area="probability",
    ),
    CatalogueEntry(
        "ignores-the-quadrant",
        "Giving arctan(y/x) as the argument without adjusting for the quadrant, so the angle is out by pi "
        "when x is negative.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "ignores-the-rival-route",
        "Subtracting the whole saving from the project time without noticing that another route now takes as "
        "long, so the project cannot finish that early.",
        area="networks",
    ),
    CatalogueEntry(
        "ignores-the-sign-condition",
        "Giving the root with the wrong sign, ignoring the condition on the real part that picks between the "
        "two square roots.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "interest-instead-of-balance",
        "Giving the interest earned where the balance, principal plus interest, was asked for.",
        area="finance",
    ),
    CatalogueEntry(
        "leaves-out-the-unknown-activity",
        "Finding the project time with the unknown duration taken as zero, as if the activity were not there.",
        area="networks",
    ),
    CatalogueEntry(
        "mixes-up-the-two-accounts",
        "Using the rate or amount from the other account in the comparison.",
        area="finance",
    ),
    CatalogueEntry(
        "modulus-of-the-wrong-number",
        "Finding the modulus of the number given in the equation instead of the number that was solved for.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "months-and-years-confused",
        "Using the number of years as the number of monthly periods or payments, or the reverse.",
        area="finance",
    ),
    CatalogueEntry(
        "omits-centering",
        "Dividing the observation by the standard deviation without first subtracting the mean.",
        area="statistics",
    ),
    CatalogueEntry(
        "perpetuity-for-annuity",
        "Taking a withdrawal that only pays the interest each month, so the balance never runs down, where "
        "payments that use the balance up were asked for.",
        area="finance",
    ),
    CatalogueEntry(
        "present-value-for-future-value",
        "Using the present-value annuity formula where the future value of regular payments was needed.",
        area="finance",
    ),
    CatalogueEntry(
        "reads-start-time-as-duration",
        "Giving a start time, or a time from a pass, where a duration was asked for.",
        area="networks",
    ),
    CatalogueEntry(
        "removes-the-activity-from-the-critical-path",
        "Subtracting the shortened activity's whole duration from the project time, as if it were removed "
        "rather than shortened.",
        area="networks",
    ),
    CatalogueEntry(
        "reports-earliest-not-latest",
        "Giving the earliest start time where the latest start time was asked for.",
        area="networks",
    ),
    CatalogueEntry(
        "reports-latest-start-as-float",
        "Giving the latest start time where the total float was asked for.",
        area="networks",
    ),
    CatalogueEntry(
        "rounds-down-when-must-exceed",
        "Rounding the crossing time down when the quantity must strictly pass the target, so the answer is "
        "the last period before it does.",
        area="finance",
    ),
    CatalogueEntry(
        "second-moment-for-expected-value",
        "Giving E(X^2) where E(X) was asked for.",
        area="probability",
    ),
    CatalogueEntry(
        "series-total-as-n-times-last-term",
        "Giving the number of terms times the last term as the total of an arithmetic series.",
        area="sequences",
    ),
    CatalogueEntry(
        "sign-error-common-difference",
        "Getting the sign of the common difference wrong, so a term is counted forwards where it should be "
        "counted back.",
        area="sequences",
    ),
    CatalogueEntry(
        "sign-error-in-second-equation",
        "Losing the sign when solving 2xy = b for y.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "squares-the-mean",
        "Giving the square of E(X) where E(X^2) was asked for.",
        area="probability",
    ),
    CatalogueEntry(
        "stops-at-the-comparison-year",
        "Giving the year the comparison was set at as the year one account overtakes the other.",
        area="finance",
    ),
    CatalogueEntry(
        "subtracts-squares-in-modulus",
        "Taking the square root of the difference of the squares of the parts instead of their sum when "
        "finding a modulus.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "subtracts-the-mean-not-its-square",
        "Giving E(X^2) - E(X) as the variance, subtracting the mean instead of its square.",
        area="probability",
    ),
    CatalogueEntry(
        "swaps-real-and-imaginary-parts",
        "Reading the real part as the imaginary part and the reverse, so an angle or ratio is turned upside down.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "swaps-x-and-y",
        "Giving the value of x where y was asked for, and the reverse: the coordinates swapped.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "takes-a-given-part-as-x-squared",
        "Reading a number from the equation, such as the real part, as the value of x^2 without solving for it.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "takes-given-term-as-first",
        "Treating a given later term as the first term of the sequence.",
        area="sequences",
    ),
    CatalogueEntry(
        "takes-second-longest-route",
        "Choosing the second-longest route through a network as the critical path.",
        area="networks",
    ),
    CatalogueEntry(
        "takes-shortest-route",
        "Choosing the shortest route through a network where the longest was needed.",
        area="networks",
    ),
    CatalogueEntry(
        "takes-the-wrong-root-of-the-quadratic",
        "Taking the negative root of the quadratic in x^2, which gives a negative value for a square.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "total-as-starting-balance",
        "Giving the balance the withdrawals start from as the total withdrawn, forgetting the interest earned "
        "while it is paid out.",
        area="finance",
    ),
    CatalogueEntry(
        "treats-simple-as-compound",
        "Compounding an account that pays simple interest.",
        area="finance",
    ),
    CatalogueEntry(
        "unweighted-mean-for-expected-value",
        "Averaging the values without weighting by their probabilities.",
        area="probability",
    ),
    CatalogueEntry(
        "uses-area-from-the-mean",
        "Giving the area between the mean and z, the table value minus 0.5, where a tail probability was asked for.",
        area="statistics",
    ),
    CatalogueEntry(
        "uses-n-not-n-minus-one",
        "Counting n steps from the first term to the n-th term instead of n - 1.",
        area="sequences",
    ),
    CatalogueEntry(
        "uses-only-one-branch",
        "Finding the chance of an event from one branch only, leaving out the other group's contribution.",
        area="probability",
    ),
    CatalogueEntry(
        "uses-only-one-z-score",
        "Dividing the difference of two observations by one of the z-scores instead of the difference of the two.",
        area="statistics",
    ),
    CatalogueEntry(
        "uses-the-wrong-predecessor",
        "Measuring from the finish of the wrong predecessor of an activity, rather than the one that finishes last.",
        area="networks",
    ),
    CatalogueEntry(
        "variance-as-standard-deviation",
        "Dividing by the variance where the standard deviation was needed, or the reverse.",
        area="statistics",
    ),
    CatalogueEntry(
        "z-sign-reversed",
        "Finding the z-score as (mean - x) / sd, so it has the wrong sign.",
        area="statistics",
    ),
    CatalogueEntry(
        "acceleration-magnitude-for-speed",
        "Giving the size of the acceleration where the speed was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "constant-speed-assumed",
        "Treating the speed at one instant as the speed throughout, so distance is speed times time.",
        area="calculus",
    ),
    CatalogueEntry(
        "coordinates-added-not-squared",
        "Adding the two coordinates to get a distance from the origin instead of squaring, adding and "
        "taking the square root.",
        area="calculus",
    ),
    CatalogueEntry(
        "denominator-not-squared",
        "Leaving the denominator unsquared after differentiating a reciprocal: -a/(x + b) where -a/(x + "
        "b)^2 is needed.",
        area="calculus",
    ),
    CatalogueEntry(
        "derivative-of-constant-not-zero",
        "Keeping the constant term when differentiating, instead of letting it vanish.",
        area="calculus",
    ),
    CatalogueEntry(
        "differentiated-instead-of-integrated",
        "Differentiating the acceleration where integrating it was needed to find the velocity.",
        area="calculus",
    ),
    CatalogueEntry(
        "divided-by-wrong-coefficient",
        "Dividing by the coefficient of the wrong term when solving, such as 6a where 12a is needed.",
        area="calculus",
    ),
    CatalogueEntry(
        "dropped-term-derivative",
        "Differentiating only some of the terms of a sum and losing the derivative of another.",
        area="calculus",
    ),
    CatalogueEntry(
        "evaluated-wrong-function-order",
        "Evaluating the function, or its first derivative, where its second derivative was asked for (or the reverse).",
        area="calculus",
    ),
    CatalogueEntry(
        "exponent-moved-wrong-way",
        "Raising the power by one where the power rule lowers it, or lowering it where it should be raised.",
        area="calculus",
    ),
    CatalogueEntry(
        "exponent-not-lowered",
        "Multiplying by the power but leaving the power as it was: differentiating x^3 to 3x^3.",
        area="calculus",
    ),
    CatalogueEntry(
        "forgot-factor-two-from-conjugate",
        "Losing the factor of 2 in the denominator when rationalising the difference quotient of a square root.",
        area="calculus",
    ),
    CatalogueEntry(
        "forgot-half-in-triangle-area",
        "Giving base times height as the area of a triangle, without the half.",
        area="calculus",
    ),
    CatalogueEntry(
        "forgot-initial-condition",
        "Integrating but leaving out the constant fixed by the starting value (the initial velocity or position).",
        area="calculus",
    ),
    CatalogueEntry(
        "forgot-point-y-value",
        "Leaving the y-value of the point out of the point-gradient form, so the line is y = m(x - x0).",
        area="calculus",
    ),
    CatalogueEntry(
        "forgot-power-multiplier",
        "Leaving out the power that comes down in front when differentiating a power, such as differentiating x^2 as x.",
        area="calculus",
    ),
    CatalogueEntry(
        "gave-y-value-for-gradient",
        "Giving the y-value of the point where the gradient was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "given-line-used-as-tangent",
        "Giving the line the tangent is parallel to as the tangent itself.",
        area="calculus",
    ),
    CatalogueEntry(
        "inner-derivative-leading-term-only",
        "Using only the leading term of the inner function's derivative: 2ax where 2ax + b is needed.",
        area="calculus",
    ),
    CatalogueEntry(
        "integration-power-not-divided",
        "Raising the power when integrating but not dividing by the new power.",
        area="calculus",
    ),
    CatalogueEntry(
        "intercept-and-gradient-confused",
        "Using the gradient of the line where its intercept is needed, or the reverse, in an area or a length.",
        area="calculus",
    ),
    CatalogueEntry(
        "intercepts-confused",
        "Giving the intercept with the wrong axis: the y-intercept where the x-intercept was asked for, "
        "or the reverse.",
        area="calculus",
    ),
    CatalogueEntry(
        "midpoint-instead-of-turning-point",
        "Giving the midpoint of the two turning points where a turning point itself was asked for, or the reverse.",
        area="calculus",
    ),
    CatalogueEntry(
        "normal-slope-negated-only",
        "Finding the gradient of the normal by changing the sign of the tangent's gradient but not taking "
        "the reciprocal.",
        area="calculus",
    ),
    CatalogueEntry(
        "normal-slope-reciprocal-only",
        "Finding the gradient of the normal by taking the reciprocal of the tangent's gradient but not "
        "changing the sign.",
        area="calculus",
    ),
    CatalogueEntry(
        "product-rule-multiplied-derivatives",
        "Differentiating a product by multiplying the derivatives of the factors.",
        area="calculus",
    ),
    CatalogueEntry(
        "product-rule-term-dropped",
        "Using only one of the two terms of the product rule.",
        area="calculus",
    ),
    CatalogueEntry(
        "quotient-rule-differentiated-separately",
        "Differentiating a quotient by dividing the derivative of the top by the derivative of the bottom.",
        area="calculus",
    ),
    CatalogueEntry(
        "quotient-rule-forgot-square",
        "Leaving the denominator unsquared in the quotient rule.",
        area="calculus",
    ),
    CatalogueEntry(
        "quotient-rule-numerator-reversed",
        "Writing the numerator of the quotient rule as uv' - u'v, the wrong way round.",
        area="calculus",
    ),
    CatalogueEntry(
        "quotient-rule-plus-instead-of-minus",
        "Adding the two terms of the quotient rule's numerator where they are subtracted.",
        area="calculus",
    ),
    CatalogueEntry(
        "root-left-in-numerator",
        "Leaving the square root in the numerator of the derivative of a square root, where it belongs in "
        "the denominator.",
        area="calculus",
    ),
    CatalogueEntry(
        "second-derivative-at-wrong-point",
        "Evaluating the second derivative at a different stationary point from the one asked about.",
        area="calculus",
    ),
    CatalogueEntry(
        "second-derivative-sign-reversed",
        "Reading the sign of the second derivative the wrong way round, so the maximum and the minimum are swapped.",
        area="calculus",
    ),
    CatalogueEntry(
        "shift-dropped-in-denominator",
        "Losing the shift in the denominator when differentiating a reciprocal or a root: x + b becomes x.",
        area="calculus",
    ),
    CatalogueEntry(
        "sign-error-trig-derivative",
        "Getting the sign of a trigonometric derivative wrong: differentiating cos x to sin x, or sin x to -cos x.",
        area="calculus",
    ),
    CatalogueEntry(
        "sign-reversed-result",
        "Doing the working correctly but giving the answer with its sign reversed, or subtracting the two quantities the wrong way round.",
        area="calculus",
    ),
    CatalogueEntry(
        "sign-slip-point-slope",
        "Writing y - y0 = m(x + x0) in the point-gradient form, so the line is shifted the wrong way.",
        area="calculus",
    ),
    CatalogueEntry(
        "slope-used-as-x",
        "Giving the gradient of the line as the x-coordinate of the point.",
        area="calculus",
    ),
    CatalogueEntry(
        "solved-f-zero-not-f-prime",
        "Solving f(x) = 0 to find the stationary points, instead of f'(x) = 0.",
        area="calculus",
    ),
    CatalogueEntry(
        "speed-as-sum-of-components",
        "Adding the two components of the velocity to get the speed, instead of using Pythagoras' theorem.",
        area="calculus",
    ),
    CatalogueEntry(
        "tangent-instead-of-normal",
        "Giving the tangent where the normal was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "trig-function-not-changed-by-derivative",
        "Differentiating a sine to a sine, or a cosine to a cosine, without switching the function.",
        area="calculus",
    ),
    CatalogueEntry(
        "used-one-component-only",
        "Giving one component of the velocity as the speed.",
        area="calculus",
    ),
    CatalogueEntry(
        "velocity-used-as-position",
        "Using the speed, or the velocity, where the position (and so the distance from the origin) was needed.",
        area="calculus",
    ),
    CatalogueEntry(
        "wrong-initial-condition-used",
        "Using the initial position as the constant of integration for the velocity, or the initial "
        "velocity as the constant for the position.",
        area="calculus",
    ),
    CatalogueEntry(
        "wrong-power-in-reciprocal",
        "Treating a reciprocal square a/x^2 as if it were a/x when differentiating.",
        area="calculus",
    ),
    CatalogueEntry(
        "wrong-root-chosen",
        "Taking the other solution of the equation than the one the question asks for: the wrong turning "
        "point, or the root of the wrong sign.",
        area="calculus",
    ),
    CatalogueEntry(
        "x-given-for-y",
        "Giving the x-coordinate where the y-value (the height of the turning point) was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "y-value-as-intercept",
        "Using the y-value of the point of contact as the y-intercept of the line through it, so the line "
        "does not pass through the point.",
        area="calculus",
    ),
    CatalogueEntry(
        "forgot-multiplier-on-parameter",
        "Dropping the multiplier on the parameter, so the parameter is treated as if its coefficient were 1.",
        area="calculus",
    ),
    CatalogueEntry(
        "ignored-second-parameter",
        "Solving one component equation as if the second parameter were absent.",
        area="vectors",
    ),
    CatalogueEntry(
        "edge-sum-used-as-normal",
        "Taking the sum of two edge vectors as the normal to the plane instead of their cross product.",
        area="vectors",
    ),
    CatalogueEntry(
        "half-gravity-used",
        "Using g/2 as the acceleration in the velocity equation, so the time to the greatest height is doubled.",
        area="calculus",
    ),
    CatalogueEntry(
        "symmetric-flight-assumed",
        "Doubling the time to the greatest height for the whole flight, although the object lands below its launch point.",
        area="calculus",
    ),
    CatalogueEntry(
        "fall-measured-from-launch-height",
        "Measuring the fall from the launch height instead of from the greatest height.",
        area="calculus",
    ),
    CatalogueEntry(
        "upper-limit-only",
        "Evaluating an antiderivative at the upper limit only, as if the lower limit contributed nothing.",
        area="calculus",
    ),
    CatalogueEntry(
        "washer-squared-difference",
        "Squaring the difference of the two radii instead of taking the difference of their squares.",
        area="calculus",
    ),
    CatalogueEntry(
        "outer-inner-radii-swapped",
        "Taking the outer radius as the inner one and the reverse, so the cross-section area has the wrong sign.",
        area="calculus",
    ),
    CatalogueEntry(
        "axis-offset-ignored",
        "Rotating about a line parallel to the x-axis as if it were the x-axis, leaving the offset out of the radii.",
        area="calculus",
    ),
    CatalogueEntry(
        "y-value-used-as-x-limit",
        "Using a y-value of an intersection point as a limit of integration in x.",
        area="calculus",
    ),
    CatalogueEntry(
        "forgot-pi-in-volume",
        "Leaving the factor of pi out of a volume of revolution.",
        area="calculus",
    ),
    CatalogueEntry(
        "quotient-from-wrong-coefficients",
        "Reading the quotient of a polynomial division from the wrong coefficients.",
        area="calculus",
    ),
    CatalogueEntry(
        "ignored-improper-quotient",
        "Splitting an improper fraction into partial fractions without first taking out the whole-number quotient.",
        area="calculus",
    ),
    CatalogueEntry(
        "read-coefficient-without-expanding",
        "Reading a partial-fraction coefficient straight from the unexpanded numerator without substituting the root.",
        area="calculus",
    ),
    CatalogueEntry(
        "repeated-factor-integrated-as-log",
        "Integrating a repeated-factor term A/(x - p)^2 as a logarithm instead of as a reciprocal.",
        area="calculus",
    ),
    CatalogueEntry(
        "extra-factor-introduced",
        "Introducing an extra factor in the antiderivative of an exponential-trigonometric integrand.",
        area="calculus",
    ),
    CatalogueEntry(
        "cyclic-forgot-original-integral",
        "Not collecting the repeated integral on one side after two rounds of integration by parts.",
        area="calculus",
    ),
    CatalogueEntry(
        "ratio-not-squared",
        "Using the ratio a/b where its square is needed when the integral reappears on the right-hand side.",
        area="calculus",
    ),
    CatalogueEntry(
        "conjugate-sum-of-radicands",
        "Building the conjugate numerator by adding the two radicands instead of taking their difference.",
        area="limits",
    ),
    CatalogueEntry(
        "infinity-minus-infinity-zero",
        "Answering 0 for a difference of two expressions that both grow without bound.",
        area="limits",
    ),
    CatalogueEntry(
        "root-sign-at-negative-infinity",
        "Taking the square root of x squared as x when x is negative, so the limit at negative infinity has the wrong sign.",
        area="limits",
    ),
    CatalogueEntry(
        "gap-given-as-one-limit",
        "Giving one of the two limits where the gap between the two asymptotes was asked for.",
        area="limits",
    ),
    CatalogueEntry(
        "forgot-leading-coefficient",
        "Leaving out the leading coefficient when the quadratic factor of a numerator is found.",
        area="limits",
    ),
    CatalogueEntry(
        "factorising-sign-slip",
        "A sign slip in the factorised form, so the cancelled factor is not the one in the denominator.",
        area="limits",
    ),
    CatalogueEntry(
        "antiderivative-given-for-integrand",
        "Giving an antiderivative where the integrand itself was asked for, or the reverse, so the derivative and the integral are swapped.",
        area="calculus",
    ),
    CatalogueEntry(
        "chain-rule-terms-added",
        "Combining the outer and inner derivatives of a composite by adding them where the chain rule multiplies them.",
        area="calculus",
    ),
    CatalogueEntry(
        "change-instead-of-new-value",
        "Giving the estimated change in a quantity where the estimated new value was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "difference-not-taken",
        "Using one of the two values in a comparison on its own where the difference between them was needed.",
        area="calculus",
    ),
    CatalogueEntry(
        "difference-of-rates-for-total",
        "Subtracting two rates where the change they produce over the interval was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "divided-by-endpoint-not-width",
        "Dividing the change in a quantity by the end value of the interval instead of by its width.",
        area="calculus",
    ),
    CatalogueEntry(
        "du-factor-missed",
        "Substituting u in an integral but not allowing for the factor du/dx, so the integrand is left with the wrong constant.",
        area="calculus",
    ),
    CatalogueEntry(
        "dx-dt-factor-omitted",
        "Leaving out the rate dx/dt when converting a derivative with respect to x into one with respect to t.",
        area="calculus",
    ),
    CatalogueEntry(
        "dx-not-multiplied",
        "Giving the gradient or rate where the rate times the small change in x was needed.",
        area="calculus",
    ),
    CatalogueEntry(
        "exact-value-instead-of-error",
        "Giving the exact new value, or the exact change, of a quantity where the error in the estimate was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "exponential-derivative-factor-dropped",
        "Differentiating an exponential model without the factor k that comes down from the exponent.",
        area="calculus",
    ),
    CatalogueEntry(
        "first-derivative-set-to-zero",
        "Using the condition f'(x) = 0 where the condition f''(x) = 0 was needed to locate a point of inflection.",
        area="calculus",
    ),
    CatalogueEntry(
        "flipped-concavity",
        "Reading the sign of the second derivative correctly but naming the opposite concavity.",
        area="calculus",
    ),
    CatalogueEntry(
        "forgot-to-take-logarithm",
        "Leaving the exponential equation as a ratio without taking the natural logarithm to bring the exponent down.",
        area="calculus",
    ),
    CatalogueEntry(
        "initial-value-ignored",
        "Leaving the initial value out of the ratio when solving an exponential model for the time.",
        area="calculus",
    ),
    CatalogueEntry(
        "inner-function-misevaluated",
        "Evaluating the inner function of a composite wrongly, so the outer derivative is found at the wrong input.",
        area="calculus",
    ),
    CatalogueEntry(
        "instantaneous-for-average",
        "Giving the rate of change at one instant where the average rate of change over an interval was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "integrated-instead-of-differentiated",
        "Integrating the function where its derivative was needed.",
        area="calculus",
    ),
    CatalogueEntry(
        "integrated-one-curve-only",
        "Integrating only one of the two curves between two graphs, so the area is not the region between them.",
        area="calculus",
    ),
    CatalogueEntry(
        "limit-not-transformed",
        "Changing the variable in a definite integral but keeping the original limits.",
        area="calculus",
    ),
    CatalogueEntry(
        "log-base-confused",
        "Using the base 10 logarithm where the natural logarithm of the model is needed.",
        area="calculus",
    ),
    CatalogueEntry(
        "log-derivative-wrong-form",
        "Differentiating a logarithm to a reciprocal of the wrong form, such as a/x^2 where a/x is needed.",
        area="calculus",
    ),
    CatalogueEntry(
        "lower-limit-taken-as-zero",
        "Treating a definite integral from a to b as one from 0 to b.",
        area="calculus",
    ),
    CatalogueEntry(
        "normal-instead-of-tangent",
        "Giving the normal where the tangent was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "power-raised-early",
        "Raising the whole bracket to the new power in a substitution, instead of the power of u.",
        area="calculus",
    ),
    CatalogueEntry(
        "product-of-roots-taken-as-root",
        "Using the product of the roots of f'(x) = 0 as one of them.",
        area="calculus",
    ),
    CatalogueEntry(
        "rate-for-total",
        "Giving the rate of change where the total change over the interval was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "second-derivative-as-gradient",
        "Using the value of the second derivative as the gradient of the tangent.",
        area="calculus",
    ),
    CatalogueEntry(
        "second-derivative-set-to-zero",
        "Setting the second derivative to zero to find a stationary point.",
        area="calculus",
    ),
    CatalogueEntry(
        "sum-of-roots-taken-as-root",
        "Using the sum of the roots of f'(x) = 0 as one of them.",
        area="calculus",
    ),
    CatalogueEntry(
        "total-not-divided-by-width",
        "Giving the total change over an interval where the average rate over it was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "value-instead-of-rate",
        "Giving the value of the quantity where its rate of change was asked for.",
        area="calculus",
    ),
    CatalogueEntry(
        "swaps-sine-and-cosine",
        "Using sine where cosine is needed, or the reverse: the legs of a right-angled triangle swapped, or the sine of the angle between two vectors in the dot product.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "inverts-trig-ratio",
        "Dividing by a trigonometric ratio where the unknown side needs multiplying by it, or the reverse.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "tan-for-sin-or-cos",
        "Using tangent where the hypotenuse is known and sine or cosine is needed.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "product-of-legs-not-divided",
        "Multiplying the two legs of a right-angled triangle and stopping, without dividing by the "
        "hypotenuse to get the altitude.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "area-half-kept-one-side",
        "Equating the area half-product of the legs with the hypotenuse times the altitude, dropping the "
        "half on one side only.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "wrong-segment-of-hypotenuse",
        "Giving the other piece of the hypotenuse (the one beside the other acute angle).",
        area="trigonometry",
    ),
    CatalogueEntry(
        "altitude-taken-as-a-leg",
        "Giving a leg of the triangle where the altitude or a segment was asked.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "angles-added-instead-of-equated",
        "Adding two angles that stand on the same arc and setting the sum to 180 degrees, instead of "
        "setting them equal.",
        area="circle-geometry",
    ),
    CatalogueEntry(
        "gives-angle-not-unknown",
        "Giving the size of the angle where the value of the unknown x was asked.",
        area="algebra",
    ),
    CatalogueEntry(
        "assumes-supplementary",
        "Treating two angles as adding to 180 degrees when the theorem makes them equal.",
        area="circle-geometry",
    ),
    CatalogueEntry(
        "forgets-coefficient-when-substituting",
        "Substituting into an expression such as 3x + 5 and forgetting to multiply x by its coefficient.",
        area="algebra",
    ),
    CatalogueEntry(
        "central-angle-confused-with-circumference-angle",
        "Doubling an angle at the circumference as though it were at the centre, or halving the reverse.",
        area="circle-geometry",
    ),
    CatalogueEntry(
        "wrong-arc-pair-equated",
        "Equating an angle with one that stands on a different arc.",
        area="circle-geometry",
    ),
    CatalogueEntry(
        "triangle-angle-sum-not-subtracted",
        "Adding the two known angles of a triangle and giving the sum, instead of subtracting it from 180 degrees.",
        area="angles",
    ),
    CatalogueEntry(
        "adjacent-angle-for-vertically-opposite",
        "Giving the adjacent angle on a straight line (180 degrees minus the angle) where the vertically "
        "opposite angle was asked.",
        area="angles",
    ),
    CatalogueEntry(
        "omits-an-angle-in-the-triangle-sum",
        "Subtracting only one of the two known angles from 180 degrees when finding the third angle of a triangle.",
        area="angles",
    ),
    CatalogueEntry(
        "determinant-sign-error",
        "Combining the diagonal and off-diagonal products of a 2 by 2 determinant with the wrong sign: ad "
        "+ bc, or bc - ad.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "forgot-off-diagonal-product",
        "Giving the product of the diagonal entries as the determinant, without subtracting the off- diagonal product.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "origin-substitution",
        "Differentiating correctly but substituting the origin instead of the stated point.",
        area="calculus",
    ),
    CatalogueEntry(
        "reciprocal-of-partial-derivative",
        "Taking the derivative of an inverse function as the reciprocal of a single partial derivative, "
        "1/(du/dx), instead of an entry of the inverse matrix.",
        area="calculus",
    ),
    CatalogueEntry(
        "inverse-entries-not-swapped",
        "Inverting a 2 by 2 matrix without swapping its diagonal entries.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "sign-error-inverse-entry",
        "Using the off-diagonal sign pattern on a diagonal entry of an inverse matrix, so a correct entry "
        "comes out negated.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "linear-approximation-sign-error",
        "Subtracting the increment in a linear approximation, x0 - f'h, instead of adding it.",
        area="calculus",
    ),
    CatalogueEntry(
        "drops-base-point-in-linear-approximation",
        "Giving only the increment f'h of a linear approximation, without adding the starting value.",
        area="calculus",
    ),
    CatalogueEntry(
        "increment-not-scaled-by-derivative",
        "Adding the change in the output variable straight to the starting value, without scaling it by "
        "the derivative.",
        area="calculus",
    ),
    CatalogueEntry(
        "pole-outside-the-contour",
        "Taking the pole that lies outside the contour as the one the contour encloses.",
        area="complex-analysis",
    ),
    CatalogueEntry(
        "residue-at-wrong-pole",
        "Finding the residue at the pole the contour does not enclose, or using it where the other pole's was needed.",
        area="complex-analysis",
    ),
    CatalogueEntry(
        "forgot-to-divide-by-other-factor",
        "Evaluating only the numerator at the pole and not dividing by the remaining factor of the denominator.",
        area="complex-analysis",
    ),
    CatalogueEntry(
        "residue-sign-error",
        "Subtracting the poles in the wrong order when forming a residue, so the residue has the wrong sign.",
        area="complex-analysis",
    ),
    CatalogueEntry(
        "forgets-2-pi-i",
        "Giving the residue alone as the value of the contour integral, without multiplying by 2*pi*i.",
        area="complex-analysis",
    ),
    CatalogueEntry(
        "forgets-i-in-residue-theorem",
        "Multiplying the residue by 2*pi but leaving out the factor i.",
        area="complex-analysis",
    ),
    CatalogueEntry(
        "integral-sign-error",
        "Reporting the contour integral with the wrong sign.",
        area="complex-analysis",
    ),
    CatalogueEntry(
        "uses-pi-i-not-2-pi-i",
        "Multiplying the residue by pi*i instead of 2*pi*i.",
        area="complex-analysis",
    ),
    CatalogueEntry(
        "ignores-pole-enclosed-by-larger-contour",
        "Using only the residue at the first pole for a larger contour that also encloses the second pole.",
        area="complex-analysis",
    ),
    CatalogueEntry(
        "residues-subtracted",
        "Subtracting the residues at two enclosed poles instead of adding them.",
        area="complex-analysis",
    ),
    CatalogueEntry(
        "assumes-zero-outside",
        "Concluding that an integral is zero because the function tends to zero at infinity, although the "
        "contour encloses poles.",
        area="complex-analysis",
    ),
    CatalogueEntry(
        "sign-error-coordinate",
        "Negating a coordinate or a computed value: a sign slip when solving for a point.",
        area="multivariable-calculus",
    ),
    CatalogueEntry(
        "gives-the-trivial-critical-point",
        "Giving the obvious critical point (the origin) where the other one was asked for.",
        area="multivariable-calculus",
    ),
    CatalogueEntry(
        "forgot-to-square-mixed-partial",
        "Subtracting f_xy instead of (f_xy) squared in the Hessian determinant f_xx f_yy - (f_xy)^2.",
        area="multivariable-calculus",
    ),
    CatalogueEntry(
        "sign-error-cross-term",
        "Evaluating a polynomial with the sign of the cross term xy lost.",
        area="multivariable-calculus",
    ),
    CatalogueEntry(
        "drops-cross-term",
        "Evaluating a polynomial at a point and leaving out the cross term xy.",
        area="multivariable-calculus",
    ),
    CatalogueEntry(
        "sign-flip-on-b",
        "Solving r^2 - b r + c = 0 instead of r^2 + b r + c = 0, which negates the real part of the roots.",
        area="differential-equations",
    ),
    CatalogueEntry(
        "forgets-square-root",
        "Giving the squared quantity (k^2, or a squared magnitude) where k itself was asked for, forgetting to take the square root.",
        area="algebra",
    ),
    CatalogueEntry(
        "real-part-only",
        "Giving the real part of a complex root where its imaginary part was asked.",
        area="differential-equations",
    ),
    CatalogueEntry(
        "coefficients-as-roots",
        "Reporting a coefficient of the characteristic equation as though it were a root.",
        area="differential-equations",
    ),
    CatalogueEntry(
        "ignores-real-part-in-derivative",
        "Differentiating e^(ax)(C cos kx + D sin kx) as though the exponential factor were constant, so "
        "the term a*C is lost from y'(0).",
        area="differential-equations",
    ),
    CatalogueEntry(
        "confuses-c-and-d-constants",
        "Giving the constant C (from y(0)) where the constant D (from y'(0)) was asked.",
        area="differential-equations",
    ),
    CatalogueEntry(
        "drops-exponential-factor",
        "Evaluating e^(ax)(C cos kx + D sin kx) at a point and leaving out the exponential factor.",
        area="differential-equations",
    ),
    CatalogueEntry(
        "sign-error-exponent",
        "Putting the wrong sign on the exponent of an exponential when evaluating it.",
        area="differential-equations",
    ),
    CatalogueEntry(
        "forgets-k-in-exponent",
        "Evaluating the exponential at x = pi/(2k) as e^(a pi / 2), forgetting to divide the exponent by k.",
        area="differential-equations",
    ),
    CatalogueEntry(
        "drops-the-square",
        "Leaving n unsquared in the decay rate of a heat-equation mode, so n^2 pi^2 / L^2 becomes n pi^2 / L^2.",
        area="pdes",
    ),
    CatalogueEntry(
        "drops-alpha",
        "Leaving the diffusion constant out of the decay rate of a heat-equation mode.",
        area="pdes",
    ),
    CatalogueEntry(
        "forgets-to-square-length",
        "Dividing by L instead of L^2 in the decay rate (n pi / L)^2.",
        area="pdes",
    ),
    CatalogueEntry(
        "forgets-to-square-pi",
        "Squaring n and L in the decay rate (n pi / L)^2 but not pi.",
        area="pdes",
    ),
    CatalogueEntry(
        "wrong-family",
        "Using sin where the boundary condition needs cos (insulated ends), or cos where it needs sin (fixed ends).",
        area="pdes",
    ),
    CatalogueEntry(
        "wrong-coefficient",
        "Dropping the stated amplitude and using 1 as the coefficient of the mode.",
        area="pdes",
    ),
    CatalogueEntry(
        "forgets-ln-2",
        "Giving 1/rate as the half-life, leaving out the factor ln 2.",
        area="pdes",
    ),
    CatalogueEntry(
        "multiplies-instead-of-dividing-by-rate",
        "Multiplying ln 2 by the decay rate instead of dividing by it when finding a time.",
        area="pdes",
    ),
    CatalogueEntry(
        "sign-error-logarithm",
        "Solving e^(-rate T) = 1/2 and getting a negative time, from taking ln(1/2) as ln 2 or the reverse.",
        area="pdes",
    ),
    CatalogueEntry(
        "takes-ln-2-as-2",
        "Replacing ln 2 by 2 when solving e^(-rate T) = 1/2.",
        area="pdes",
    ),
    CatalogueEntry(
        "half-life-applied-twice",
        "Halving the amplitude twice: a quarter of the initial value where a half was needed.",
        area="pdes",
    ),
    CatalogueEntry(
        "uses-one-coordinate-only",
        "Substituting one coordinate of the point where the derivative needs both, for example reading "
        "2E(x + y) as 2Ex.",
        area="multivariable-calculus",
    ),
    CatalogueEntry(
        "ignores-unknown-term",
        "Leaving out the term containing the unknown constant when differentiating or evaluating.",
        area="multivariable-calculus",
    ),
    CatalogueEntry(
        "first-derivative-instead-of-second",
        "Giving a first partial derivative where a second partial derivative was asked.",
        area="multivariable-calculus",
    ),
    CatalogueEntry(
        "unweighted-average",
        "Averaging the prior mean and the sample mean with equal weight, ignoring both precisions.",
        area="bayesian-statistics",
    ),
    CatalogueEntry(
        "posterior-ignores-sample-size",
        "Treating a sample of n measurements as a single measurement when adding the data precision.",
        area="bayesian-statistics",
    ),
    CatalogueEntry(
        "precision-weights-swapped",
        "Swapping the precision weights on the prior mean and the sample mean.",
        area="bayesian-statistics",
    ),
    CatalogueEntry(
        "ignores-the-prior",
        "Reporting the sample information alone (the sample mean or the data variance) as the posterior.",
        area="bayesian-statistics",
    ),
    CatalogueEntry(
        "doesnt-shrink",
        "Reporting the prior variance unchanged as the posterior variance.",
        area="bayesian-statistics",
    ),
    CatalogueEntry(
        "reports-precision-not-variance",
        "Giving the precision (the reciprocal of the variance) where a variance was asked.",
        area="bayesian-statistics",
    ),
    CatalogueEntry(
        "discards-round-one",
        "Updating the original prior with the second round of data only, discarding what round one taught.",
        area="bayesian-statistics",
    ),
    CatalogueEntry(
        "forgets-observation-noise",
        "Giving the posterior variance of the mean as the variance of a new measurement, without adding "
        "the noise variance.",
        area="bayesian-statistics",
    ),
    CatalogueEntry(
        "discards-round-two",
        "Using the posterior from the earlier round where the later one was needed.",
        area="bayesian-statistics",
    ),
    CatalogueEntry(
        "uses-variance-of-the-mean",
        "Adding the variance of a sample mean (sigma^2 / n) where the variance of a single new "
        "measurement (sigma^2) was needed.",
        area="bayesian-statistics",
    ),
    CatalogueEntry(
        "ignores-holding-probability",
        "Giving the one-step probability of a move without conditioning on the particle moving at all, so "
        "the chance of staying put is not removed.",
        area="markov-chains",
    ),
    CatalogueEntry(
        "swaps-up-and-down",
        "Giving the probability of the downward move where the upward move was asked, or the reverse.",
        area="markov-chains",
    ),
    CatalogueEntry(
        "ignores-the-return-path",
        "Counting only the direct route to the target and ignoring that the particle can step back and try again.",
        area="markov-chains",
    ),
    CatalogueEntry(
        "stops-after-one-loop",
        "Adding one return path but not the infinite sum of repeated returns, so the first-step equation "
        "is not solved.",
        area="markov-chains",
    ),
    CatalogueEntry(
        "wrong-loop-probability",
        "Using the wrong probability for the loop back to the same state when summing repeated returns.",
        area="markov-chains",
    ),
    CatalogueEntry(
        "drops-later-step-probability",
        "Giving the probability of reaching an intermediate state, or the product of two step "
        "probabilities, as the absorption probability.",
        area="markov-chains",
    ),
    CatalogueEntry(
        "confuses-start-states",
        "Giving the absorption probability from the other starting state.",
        area="markov-chains",
    ),
    CatalogueEntry(
        "forgets-to-average",
        "Adding the two starting-state probabilities and not dividing by 2 for the equal-chance start.",
        area="probability",
    ),
    CatalogueEntry(
        "left-out-a-term-when-solving-for-constant",
        "Solving for a constant such as k or d and leaving out one of the terms of the equation that determines it.",
        area="calculus",
    ),
    CatalogueEntry(
        "dropped-the-squared-term",
        "Evaluating a quadratic position or height at a point and leaving out the squared (gravity or acceleration) term.",
        area="calculus",
    ),
    CatalogueEntry(
        "dropped-the-x-term-at-the-limit",
        "Giving only the constant of the simplified form as the limit, leaving out the term in x that survives at the point.",
        area="calculus",
    ),
    CatalogueEntry(
        "dropped-initial-velocity-term",
        "Giving the square root of the discriminant over g as a flight time, leaving out the initial vertical speed term.",
        area="calculus",
    ),
    CatalogueEntry(
        "wrong-parameter-substituted",
        "Substituting the parameter of the other line or the wrong value of k or t into a correct formula.",
        area="calculus",
    ),
    CatalogueEntry(
        "used-apex-time-for-landing",
        "Using the time of the highest point where the time of landing is needed.",
        area="calculus",
    ),
    CatalogueEntry(
        "used-initial-rate",
        "Giving the rate at t = 0 where the rate at a later moment was asked.",
        area="calculus",
    ),
    CatalogueEntry(
        "derivative-at-new-point",
        "Evaluating the derivative at x0 + h instead of x0 in a linear approximation.",
        area="calculus",
    ),
    CatalogueEntry(
        "evaluated-at-the-wrong-x",
        "Substituting an x-value from another part of the problem into a function or rate.",
        area="calculus",
    ),
    CatalogueEntry(
        "used-initial-value",
        "Giving the starting value of a model where the value at a later input was asked.",
        area="calculus",
    ),
    CatalogueEntry(
        "swapped-partial-fraction-coefficients",
        "Giving the coefficient of the other partial fraction (A for B or B for A), or exchanging them in the integral.",
        area="calculus",
    ),
    CatalogueEntry(
        "gave-the-other-parameter",
        "Giving the parameter of the other line (s for t or t for s) at the point of intersection.",
        area="calculus",
    ),
    CatalogueEntry(
        "gave-the-other-coefficient",
        "Giving the coefficient of the other variable in a plane equation (a for b or b for a).",
        area="calculus",
    ),
    CatalogueEntry(
        "confuses-k-with-the-limit",
        "Giving the unknown constant k where the limit was asked, or the limit where k was asked.",
        area="calculus",
    ),
    CatalogueEntry(
        "gave-d-for-k",
        "Stopping at the constant d of a plane equation where the unknown coordinate k was asked.",
        area="calculus",
    ),
    CatalogueEntry(
        "used-the-other-objects-speed",
        "Giving the speed of the other object where the speed of the one asked about was needed.",
        area="calculus",
    ),
    # -- exam chains: copied exam tiers -----------------------------------------------------------------------------------
    CatalogueEntry(
        "swapped-success-failure",
        "Using the probability of failure where the probability of success belongs, or the reverse, in a run of independent trials.",
        area="probability",
    ),
    CatalogueEntry(
        "odds-as-probability",
        "Giving the ratio of one outcome to the other (the odds) where a probability or a fraction of the whole was asked.",
        area="probability",
    ),
    CatalogueEntry(
        "reciprocal-of-probability",
        "Giving 1 divided by the probability instead of the probability itself.",
        area="probability",
    ),
    CatalogueEntry(
        "counts-orderings-in-ordered-sequence",
        "Multiplying the probability of one stated sequence by the number of ways to arrange the same outcomes, though the question asked for that one order only.",
        area="probability",
    ),
    CatalogueEntry(
        "drops-one-trial-from-sequence",
        "Leaving one trial's factor out of the product for a sequence of independent trials.",
        area="probability",
    ),
    CatalogueEntry(
        "ignores-failures-in-sequence",
        "Multiplying only the success factors of a sequence and leaving out the failures.",
        area="probability",
    ),
    CatalogueEntry(
        "forgets-the-extra-trial",
        "Giving the probability of the shorter sequence when the question extended it by one more trial.",
        area="probability",
    ),
    CatalogueEntry(
        "counts-partial-classes-whole",
        "Counting a class that the interval only partly covers as if it were wholly inside the interval.",
        area="probability",
    ),
    CatalogueEntry(
        "partial-class-fraction-inverted",
        "Taking the part of a class that lies outside the interval instead of the part inside it.",
        area="probability",
    ),
    CatalogueEntry(
        "drops-partial-classes",
        "Leaving out the classes the interval covers only partly, keeping only the whole classes.",
        area="probability",
    ),
    CatalogueEntry(
        "divides-by-classes-in-range-not-total",
        "Dividing the count in an interval by the count of the classes it touches instead of by the total number of observations.",
        area="probability",
    ),
    CatalogueEntry(
        "conditional-as-joint",
        "Giving the probability of both events together when a conditional probability was asked for, so the condition is never divided out.",
        area="probability",
    ),
    CatalogueEntry(
        "conditioned-on-the-wrong-event",
        "Dividing by the probability of the wrong event when finding a conditional probability.",
        area="probability",
    ),
    CatalogueEntry(
        "reports-the-conditioning-event",
        "Giving the probability of the event that is given, rather than the conditional probability asked for.",
        area="probability",
    ),
    CatalogueEntry(
        "reports-known-sum-not-remainder",
        "Giving the sum of the counts that are shown, when the question asked for the missing count that completes the total.",
        area="data",
    ),
    CatalogueEntry(
        "reports-total-as-missing-value",
        "Giving the total itself as the missing count.",
        area="data",
    ),
    CatalogueEntry(
        "left-a-class-out-of-the-sum",
        "Leaving one of the known counts out when subtracting from the total, so the missing count comes out too large.",
        area="data",
    ),
    CatalogueEntry(
        "double-counted-a-class-in-the-sum",
        "Subtracting one of the known counts twice when finding the missing count, so the missing count comes out too small.",
        area="data",
    ),
    CatalogueEntry(
        "omits-the-missing-category",
        "Leaving the category whose count had to be found out of an event made of two categories.",
        area="data",
    ),
    CatalogueEntry(
        "counts-only-one-named-category",
        "Using the count of only one of the two categories named in an event.",
        area="data",
    ),
    CatalogueEntry(
        "projects-the-complement",
        "Projecting the relative frequency of the opposite event onto future trials instead of the event asked about.",
        area="data",
    ),
    CatalogueEntry(
        "reports-raw-count-not-projection",
        "Giving the number observed in the original trials when asked to predict the number in a different number of future trials.",
        area="data",
    ),
    CatalogueEntry(
        "swapped-the-two-unknowns",
        "Swapping the values found for two unknowns, so each is attached to the wrong one.",
        area="data",
    ),
    CatalogueEntry(
        "ignores-known-values-in-mean-equation",
        "Writing the equation from the mean without subtracting the contribution of the frequencies that are known.",
        area="data",
    ),
    CatalogueEntry(
        "divides-by-wrong-difference",
        "Dividing by one of the values of the variable instead of by the difference between the two when solving a pair of equations.",
        area="data",
    ),
    CatalogueEntry(
        "forgot-to-subtract-first-unknown",
        "Giving the sum of two unknowns as the second unknown, without subtracting the first.",
        area="data",
    ),
    CatalogueEntry(
        "distribution-listed-in-reverse",
        "Listing the probabilities from the largest value of the variable down, though the question asked for increasing order.",
        area="data",
    ),
    CatalogueEntry(
        "distribution-shifted-by-one-value",
        "Pairing each probability with the next value of the variable along, so the list is shifted by one place.",
        area="data",
    ),
    CatalogueEntry(
        "ignores-the-minutes",
        "Using only the whole hours of a time given in hours and minutes.",
        area="measurement",
    ),
    CatalogueEntry(
        "wrong-watt-kilowatt-factor",
        "Dividing watts by 100 instead of 1000 to get kilowatts.",
        area="measurement",
    ),
    CatalogueEntry(
        "forgot-watt-kilowatt-conversion",
        "Multiplying the power in watts by the time in hours and calling it kilowatt-hours, with no division by 1000.",
        area="measurement",
    ),
    CatalogueEntry(
        "forgot-to-multiply-by-days",
        "Giving the cost or saving for one day when the question asked about a longer period.",
        area="measurement",
    ),
    CatalogueEntry(
        "forgot-the-tariff",
        "Multiplying the energy by the number of days but never by the price per kilowatt-hour.",
        area="measurement",
    ),
    CatalogueEntry(
        "converted-cents-to-dollars-twice",
        "Dividing by 100 twice when changing cents to dollars, so the amount is a hundred times too small.",
        area="measurement",
    ),
    CatalogueEntry(
        "reports-new-cost-not-saving",
        "Giving the running cost of the new appliance when asked how much less it costs than the old one.",
        area="measurement",
    ),
    CatalogueEntry(
        "reports-the-original-cost",
        "Giving the running cost of the original appliance when asked for the saving.",
        area="measurement",
    ),
    CatalogueEntry(
        "ignores-the-run-time",
        "Working out a saving from the drop in power alone, never multiplying by the hours the appliance runs.",
        area="measurement",
    ),
    CatalogueEntry(
        "divided-by-wrong-amount",
        "Dividing by the wrong number: a count, a length or an interval other than the one the quantity is spread "
        "over, or not dividing at all.",
        area="general",
    ),
    CatalogueEntry(
        "left-out-a-value",
        "Leaving a known quantity out of a calculation, so the result is missing its share.",
        area="general",
    ),
    CatalogueEntry(
        "left-out-a-step",
        "Skipping one step of a multi-step calculation (a discount, a tax) before carrying on with the rest.",
        area="general",
    ),
    CatalogueEntry(
        "left-out-fixed-extra",
        "Leaving out a fixed extra (a wait, a fee) that is added once to the running amount.",
        area="general",
    ),
    CatalogueEntry(
        "left-out-the-square-root",
        "Giving the squared length where the length was asked, forgetting the final square root.",
        area="general",
    ),
    CatalogueEntry(
        "evaluated-at-wrong-point",
        "Evaluating a function, a rate or a distance at the wrong input: the start of an interval instead of the "
        "end, or time zero instead of the time asked for.",
        area="functions",
    ),
    CatalogueEntry(
        "took-the-wrong-root",
        "Choosing the wrong solution of a quadratic: the one outside the interval, or the earlier time when the "
        "later was asked for.",
        area="quadratics",
    ),
    CatalogueEntry(
        "unconverted-time-unit",
        "Leaving a time in hours where minutes were asked, or a speed in kilometres per minute where kilometres per "
        "hour were asked.",
        area="measurement",
    ),
    CatalogueEntry(
        "decimal-time-base-100",
        "Treating minutes as hundredths of an hour, so 1.5 hours becomes 150 minutes and 2 hours 45 minutes becomes 2.45 hours.",
        area="measurement",
    ),
    CatalogueEntry(
        "mixed-time-units",
        "Adding minutes to a time in hours without converting one of them first.",
        area="measurement",
    ),
    CatalogueEntry(
        "unconverted-speed-unit",
        "Using a speed in metres per second as though it were in kilometres per hour.",
        area="measurement",
    ),
    CatalogueEntry(
        "ignored-the-deadline",
        "Choosing the latest service that leaves, without checking that it arrives by the deadline.",
        area="measurement",
    ),
    CatalogueEntry(
        "took-first-service-not-latest",
        "Choosing the first service that works instead of the latest one that still works.",
        area="measurement",
    ),
    CatalogueEntry(
        "used-departure-not-arrival",
        "Measuring the time to spare from the departure time instead of the arrival time.",
        area="measurement",
    ),
    CatalogueEntry(
        "skipped-binning",
        "Computing a bin-based measure from the overall means and ignoring the bins.",
        area="statistics",
    ),
    CatalogueEntry(
        "unweighted-bin-average",
        "Averaging the bins' errors equally instead of weighting each bin by its share of the points.",
        area="statistics",
    ),
    CatalogueEntry(
        "weighted-by-wrong-share",
        "Weighting each bin by the other bin's share of the points.",
        area="statistics",
    ),
    CatalogueEntry(
        "dropped-sign-of-difference",
        "Reporting the size of a difference and losing its sign, so over- and under-estimates look alike.",
        area="statistics",
    ),
    CatalogueEntry(
        "wrong-denominator-for-proportion",
        "Dividing by the wrong total: the grand total or a column total when the proportion is of a row, a total one out, or anything other than the number of trials.",
        area="statistics",
    ),
    CatalogueEntry(
        "gave-count-not-percentage",
        "Giving the count in the cell where a percentage of its group was asked for.",
        kind="near_miss",
        area="statistics",
    ),
    CatalogueEntry(
        "compared-counts-not-percentages",
        "Comparing two groups by their counts, though the groups differ in size and a comparison needs percentages.",
        area="statistics",
    ),
    CatalogueEntry(
        "solved-one-component-only",
        "Finding the time when one coordinate of two moving points agrees and stopping there, though the quantity "
        "asked for depends on both coordinates.",
        area="vectors",
    ),
    CatalogueEntry(
        "added-components-not-magnitude",
        "Adding the components of a vector to get its length instead of using Pythagoras.",
        area="vectors",
    ),
    CatalogueEntry(
        "ignored-the-closest-distance",
        "Finding how long two points stay within range as if they passed through the same point, ignoring how close "
        "they actually get.",
        area="vectors",
    ),
    CatalogueEntry(
        "subtracted-distances-not-squares",
        "Subtracting the least distance from the range directly instead of subtracting the squares and taking the "
        "square root.",
        area="vectors",
    ),
    CatalogueEntry(
        "gave-half-the-interval",
        "Giving the time on one side of the closest approach where the whole length of time was asked for, or the "
        "reverse.",
        area="vectors",
    ),
    CatalogueEntry(
        "gave-end-time-not-start",
        "Giving the time the points leave the range where the time they enter it was asked for.",
        area="vectors",
    ),
    CatalogueEntry(
        "change-not-rate",
        "Giving the total change over an interval where the average rate of change was asked for.",
        kind="near_miss",
        area="calculus",
    ),
    CatalogueEntry(
        "averaged-endpoint-values",
        "Averaging the function's values at the two ends of an interval instead of finding the change per unit.",
        area="calculus",
    ),
    CatalogueEntry(
        "averaged-endpoint-rates",
        "Averaging the instantaneous rates at the two ends of an interval to get the average rate over it, which "
        "holds only for a quadratic.",
        area="calculus",
    ),
    CatalogueEntry(
        "dropped-power-rule-coefficient",
        "Lowering the power but not multiplying by the old power when differentiating.",
        area="calculus",
    ),
    CatalogueEntry(
        "assumed-midpoint",
        "Taking the midpoint of an interval as the time the instantaneous rate equals the average rate, which holds "
        "only for a quadratic.",
        area="calculus",
    ),
    CatalogueEntry(
        "solved-wrong-derivative-equation",
        "Setting the wrong derivative to zero: the acceleration for the times of rest, or the velocity for the time "
        "of zero acceleration.",
        area="calculus",
    ),
    CatalogueEntry(
        "rest-means-zero-acceleration",
        "Believing a particle that is momentarily at rest has zero acceleration.",
        area="calculus",
    ),
    CatalogueEntry(
        "halved-second-derivative",
        "Differentiating twice but missing the factor the second power-rule step contributes, so the acceleration is "
        "half the correct value.",
        area="calculus",
    ),
    CatalogueEntry(
        "ignored-reversed-limits",
        "Adding an integral stated with its limits reversed as if it ran in the forward direction.",
        area="calculus",
    ),
    CatalogueEntry(
        "dropped-scalar-multiplier",
        "Dropping the scalar multiplier in front of an integral, so k times an integral is treated as the integral.",
        area="calculus",
    ),
    CatalogueEntry(
        "constant-integrated-as-value",
        "Integrating a constant term over an interval as the constant itself, forgetting to multiply by the length "
        "of the interval.",
        area="calculus",
    ),
    CatalogueEntry(
        "gave-observed-not-predicted",
        "Giving the mean observed outcome where the mean predicted probability was asked for.",
        area="statistics",
    ),
    CatalogueEntry(
        "used-one-bin-not-all-points",
        "Reporting the mean of one bin where the figure for all the points was asked for.",
        area="statistics",
    ),
    CatalogueEntry(
        "gave-percentage-amount-not-result",
        "Giving the amount of the discount or the tax where the price after it was asked for.",
        area="money",
    ),
    CatalogueEntry(
        "subtracted-instead-of-added",
        "Subtracting one integral from another where the two are added to give the integral over the whole interval.",
        area="calculus",
    ),
    CatalogueEntry(
        "gave-a-part-not-the-whole",
        "Giving one of the pieces that add up to the quantity, where the whole was asked for.",
        area="general",
    ),
    CatalogueEntry(
        "swapped-wait-and-journey",
        "Giving the time on the journey where the wait before it was asked for.",
        area="measurement",
    ),
    CatalogueEntry(
        "included-the-wait-in-the-journey",
        "Counting the wait as part of the journey, so the journey time is too long by the wait.",
        area="measurement",
    ),
    CatalogueEntry(
        "gave-interval-not-time",
        "Giving the length of time between two events where the time of one of them was asked for.",
        area="general",
    ),
    CatalogueEntry(
        "evaluated-wrong-derivative",
        "Evaluating the acceleration where the velocity was asked for, or the reverse.",
        area="calculus",
    ),
    CatalogueEntry(
        "gave-closest-approach-time",
        "Giving the time of closest approach where the time the points first come within range was asked for.",
        area="vectors",
    ),
    CatalogueEntry(
        "used-time-of-other-route",
        "Using the time for the other route when finding a speed or a time.",
        area="measurement",
    ),
    CatalogueEntry(
        "applied-one-price-to-all",
        "Charging every ticket or item at the first price when the group has two different prices.",
        area="money",
    ),
    CatalogueEntry(
        "forgot-to-multiply-by-quantity",
        "Adding a unit price once instead of multiplying it by the number bought.",
        area="money",
    ),
    CatalogueEntry(
        "swapped-between-groups",
        "Pairing each quantity with the other group's price.",
        area="money",
    ),
    CatalogueEntry(
        "percentage-applied-wrong-way",
        "Adding a discount to the price, or taking a tax off it, instead of the other way round.",
        area="money",
    ),
    CatalogueEntry(
        "ignored-frequencies-in-mean",
        "Averaging the class midpoints without weighting them by their frequencies.",
        area="statistics",
    ),
    CatalogueEntry(
        "used-class-limit-not-midpoint",
        "Using a class's lower (or upper) limit instead of its midpoint when estimating a mean from grouped data.",
        area="statistics",
    ),
    CatalogueEntry(
        "median-class-not-interpolated",
        "Giving the median class's midpoint or limit as the median, instead of interpolating within the class.",
        area="statistics",
    ),
    CatalogueEntry(
        "forgot-to-halve-n",
        "Using the total frequency, not half of it, to locate the median.",
        area="statistics",
    ),
    CatalogueEntry(
        "position-reported-as-median",
        "Reporting the position of the middle item (half the total frequency) as if it were the median value.",
        area="statistics",
    ),
    CatalogueEntry(
        "ratio-not-difference",
        "Dividing the two measures when asked for their difference.",
        area="statistics",
    ),
    CatalogueEntry(
        "gave-one-measure-not-difference",
        "Giving one of the two measures when asked for their difference.",
        area="statistics",
    ),
    CatalogueEntry(
        "skew-direction-confusion",
        "Reporting the opposite direction of skew: right for left, or left for right.",
        area="statistics",
    ),
    CatalogueEntry(
        "skewed-as-symmetric",
        "Calling a clearly skewed distribution symmetric.",
        area="statistics",
    ),
    CatalogueEntry(
        "symmetric-as-skewed",
        "Calling a symmetric distribution skewed.",
        area="statistics",
    ),
    CatalogueEntry(
        "unimodal-as-bimodal",
        "Calling a single-peaked distribution bimodal.",
        area="statistics",
    ),
    CatalogueEntry(
        "sign-error-in-r",
        "Getting the sign of a correlation coefficient wrong.",
        area="statistics",
    ),
    CatalogueEntry(
        "slope-reported-as-r",
        "Giving the slope of the least-squares line where the correlation coefficient was asked for.",
        area="statistics",
    ),
    CatalogueEntry(
        "forgot-to-centre-sums",
        "Using the raw sums of squares and products without subtracting the means' contribution.",
        area="statistics",
    ),
    CatalogueEntry(
        "no-square-root-in-r-denominator",
        "Leaving the square root out of the denominator of the correlation coefficient.",
        area="statistics",
    ),
    CatalogueEntry(
        "r-given-as-r-squared-percentage",
        "Giving |r| as a percentage where the percentage of variation explained (r squared) was asked for.",
        area="statistics",
    ),
    CatalogueEntry(
        "unexplained-variation-for-explained",
        "Giving the percentage of variation left unexplained instead of the percentage explained.",
        area="statistics",
    ),
    CatalogueEntry(
        "direction-sign-flip",
        "Reporting the opposite direction of an association from the one the sign of r gives.",
        area="statistics",
    ),
    CatalogueEntry(
        "form-confusion",
        "Reporting the opposite form of an association: linear for non-linear, or the reverse.",
        area="statistics",
    ),
    CatalogueEntry(
        "strength-band-off-by-one",
        "Reporting a strength one band away from the correct one.",
        area="statistics",
    ),
    CatalogueEntry(
        "r-used-for-nonlinear-strength",
        "Reading the strength of a non-linear association from r, which measures linear strength only.",
        area="statistics",
    ),
    CatalogueEntry(
        "trusts-large-r-for-linearity",
        "Taking a large |r| as proof of a linear form when the residual plot is curved.",
        area="statistics",
    ),
    CatalogueEntry(
        "gave-complement-for-event",
        "Giving the probability of the opposite event (or the complementary proportion) for the one asked.",
        area="statistics",
    ),
    CatalogueEntry(
        "wrong-count-in-probability",
        "Putting the number removed, instead of the number of successes, over the total.",
        area="statistics",
    ),
    CatalogueEntry(
        "used-sample-proportion-for-population",
        "Using the proportion in a small examined sample as the proportion in the whole population.",
        area="statistics",
    ),
    CatalogueEntry(
        "removed-from-total-only",
        "After items are removed, reducing the total but not the count of successes.",
        area="statistics",
    ),
    CatalogueEntry(
        "removed-from-successes-only",
        "After items are removed, reducing the count of successes but not the total.",
        area="statistics",
    ),
    CatalogueEntry(
        "removed-all-as-successes",
        "Treating every removed item as a success when only some of them were.",
        area="statistics",
    ),
    CatalogueEntry(
        "removed-one-not-all",
        "Allowing for one removed item when several were removed.",
        area="statistics",
    ),
    CatalogueEntry(
        "ignored-removal-effect",
        "Treating draws without replacement as if the probability could not change, whatever the size of the change.",
        area="statistics",
    ),
    CatalogueEntry(
        "dependence-assumed-regardless-of-size",
        "Rejecting a model because draws are without replacement, without checking whether the change in probability "
        "is small enough to ignore.",
        area="statistics",
    ),
    CatalogueEntry(
        "percent-of-wrong-base",
        "Finding a percentage change as a percentage of the new value, or of the wrong one of two values, instead of "
        "the stated base.",
        area="statistics",
    ),
    CatalogueEntry(
        "multiplied-by-n-not-powered",
        "Multiplying a probability by the number of trials where it should be raised to that power.",
        area="statistics",
    ),
    CatalogueEntry(
        "wrong-exponent-on-probability",
        "Raising a probability to the wrong power (one too few trials).",
        area="statistics",
    ),
    CatalogueEntry(
        "averaged-the-two-rates",
        "Averaging two relative frequencies from samples of different sizes instead of pooling the counts.",
        area="statistics",
    ),
    CatalogueEntry(
        "divided-by-failures",
        "Dividing the number of successes by the number of failures instead of by the total.",
        area="statistics",
    ),
    CatalogueEntry(
        "used-one-stage-only",
        "Using the result of one stage of an experiment where the combined result was asked for.",
        area="statistics",
    ),
    CatalogueEntry(
        "added-probabilities-for-at-least-one",
        "Adding the probabilities of the separate trials or events to get the probability of at least one, without subtracting the overlap.",
        area="statistics",
    ),
    CatalogueEntry(
        "all-instead-of-at-least-one",
        "Finding the probability that every trial succeeds (or both events occur) where at least one was asked for.",
        area="statistics",
    ),
    CatalogueEntry(
        "none-instead-of-at-least-one",
        "Finding the probability that no trial succeeds (or neither event occurs) where at least one was asked for.",
        area="statistics",
    ),
    CatalogueEntry(
        "single-trial-for-several",
        "Giving the probability for one trial where several were asked for.",
        area="statistics",
    ),
    CatalogueEntry(
        "inverted-rate-of-change",
        "Dividing the change in time by the change in value instead of the reverse.",
        area="statistics",
    ),
    CatalogueEntry(
        "counted-both-endpoints",
        "Counting both endpoints when finding the number of steps between two periods.",
        area="statistics",
    ),
    CatalogueEntry(
        "miscounted-periods-between",
        "Miscounting the periods between two points, for example by a whole year.",
        area="statistics",
    ),
    CatalogueEntry(
        "off-by-one-period",
        "Evaluating at the period just before or just after the one asked for.",
        area="statistics",
    ),
    CatalogueEntry(
        "measured-from-t-zero-not-known-point",
        "Applying a rate of change from t = 0 instead of from the point where the value is known.",
        area="statistics",
    ),
    CatalogueEntry(
        "wrong-anchor-point",
        "Projecting from the wrong one of two known values.",
        area="statistics",
    ),
    CatalogueEntry(
        "divided-by-seasonal-index",
        "Dividing by the seasonal index where a forecast is found by multiplying (that is the deseasonalising step).",
        area="statistics",
    ),
    CatalogueEntry(
        "ignored-seasonal-index",
        "Giving the trend value as the forecast, without applying the seasonal index.",
        area="statistics",
    ),
    CatalogueEntry(
        "gave-adjustment-not-forecast",
        "Giving the size of the seasonal adjustment instead of the adjusted forecast.",
        area="statistics",
    ),
    CatalogueEntry(
        "averaged-instead-of-multiplied",
        "Averaging two probabilities where the probability of both independent events is their product.",
        area="statistics",
    ),
    CatalogueEntry(
        "took-smaller-probability",
        "Giving the smaller of two probabilities as the probability of both events.",
        area="statistics",
    ),
    CatalogueEntry(
        "used-one-event-failing",
        "Multiplying the probability of one event by the probability that the other does not occur.",
        area="statistics",
    ),
    CatalogueEntry(
        "gave-complement-of-both",
        "Giving the probability that the two events do not both occur for the probability that at least one occurs.",
        area="statistics",
    ),
    CatalogueEntry(
        "uses-connotation-not-stated-ranges",
        "Choosing between neighbouring terms by everyday meaning instead of the ranges stated.",
        area="statistics",
    ),
    CatalogueEntry(
        "ignores-boundary-inclusivity",
        "Putting a value that sits exactly on the edge of a range in the neighbouring range.",
        area="statistics",
    ),
    CatalogueEntry(
        "divides-by-n",
        "Dividing by n where the standard error divides by the square root of n.",
        area="statistics",
    ),
    CatalogueEntry(
        "omits-sqrt-n",
        "Leaving the square root of n out of the standard error, so s or the raw difference is used as it stands.",
        area="statistics",
    ),
    CatalogueEntry(
        "multiplies-by-sqrt-n",
        "Multiplying by the square root of n where the standard error divides by it.",
        area="statistics",
    ),
    CatalogueEntry(
        "wrong-n-in-root",
        "Using n - 1 under the square root of the standard error.",
        area="statistics",
    ),
    CatalogueEntry(
        "one-tail-for-two-tails",
        "Giving the probability in one tail where a probability in either direction was asked for.",
        area="statistics",
    ),
    CatalogueEntry(
        "doubled-for-one-tail",
        "Doubling the tail probability where only one direction was asked for.",
        area="statistics",
    ),
    CatalogueEntry(
        "gave-complement-of-tail",
        "Giving one minus the two-tailed probability for the two-tailed probability.",
        area="statistics",
    ),
    CatalogueEntry(
        "gave-cumulative-for-tail",
        "Giving the cumulative probability up to z where the tail probability was asked for.",
        area="statistics",
    ),
    CatalogueEntry(
        "gave-area-from-centre",
        "Giving the area between the mean and z (as some tables do) as the tail probability.",
        area="statistics",
    ),
    CatalogueEntry(
        "omits-n",
        "Leaving n out of the standard deviation of a sample proportion, using the square root of p(1 - p) alone.",
        area="statistics",
    ),
    CatalogueEntry(
        "omits-sqrt",
        "Giving the variance p(1 - p)/n where the standard deviation was asked for.",
        area="statistics",
    ),
    CatalogueEntry(
        "forgot-one-minus-p",
        "Using p alone, not p(1 - p), under the square root.",
        area="statistics",
    ),
    CatalogueEntry(
        "used-population-sd-not-standard-error",
        "Standardising by the standard deviation of one observation instead of by the standard error of the sample "
        "proportion.",
        area="statistics",
    ),
    CatalogueEntry(
        "divides-by-variance",
        "Dividing by the variance where z divides by the standard deviation.",
        area="statistics",
    ),
    CatalogueEntry(
        "assumed-uniform-density",
        "Treating the density as constant across its interval, so probabilities are proportional to length.",
        area="statistics",
    ),
    CatalogueEntry(
        "assumed-uniform-mean",
        "Giving the midpoint of the interval as the mean of a density that is not uniform.",
        area="statistics",
    ),
    CatalogueEntry(
        "forgot-to-raise-the-power-in-k",
        "Integrating a power of x without raising the power by one.",
        area="statistics",
    ),
    CatalogueEntry(
        "inverted-normalising-constant",
        "Giving the value of the integral for k instead of its reciprocal.",
        area="statistics",
    ),
    CatalogueEntry(
        "forgot-normalising-constant",
        "Leaving the constant k out of an integral of the density.",
        area="statistics",
    ),
    CatalogueEntry(
        "used-upper-bound-only",
        "Integrating from the start of the interval to the upper bound instead of between the two bounds.",
        area="statistics",
    ),
    CatalogueEntry(
        "squared-instead-of-mean",
        "Giving E(X squared) where the mean E(X) was asked for.",
        area="statistics",
    ),
    CatalogueEntry(
        "variance-as-second-moment",
        "Giving E(X squared) as the variance, without subtracting the square of the mean.",
        area="statistics",
    ),
    CatalogueEntry(
        "forgot-to-square-the-mean",
        "Subtracting the mean, not its square, from E(X squared).",
        area="statistics",
    ),
    CatalogueEntry(
        "amplitude-not-squared",
        "Using the amplitude A where A squared belongs in an energy integral.",
        area="university",
    ),
    CatalogueEntry(
        "answer-sign-reversed",
        "Giving the negative of the right value: the right size with the sign reversed.",
        area="university",
    ),
    CatalogueEntry(
        "applies-the-map-incompletely",
        "Applying a linear map to only one component of the vector, or not at all, so the image is left "
        "unchanged or half scaled.",
        area="university",
    ),
    CatalogueEntry(
        "assumes-essential-members-suffice",
        "Assuming any family containing the essential members of a cover is itself a cover, so counting every "
        "such family as a subcover.",
        area="university",
    ),
    CatalogueEntry(
        "assumes-every-accepted-word-extends-twice",
        "Doubling the count of accepted short words to get the longer count, as if each accepted word had two "
        "accepted extensions.",
        area="university",
    ),
    CatalogueEntry(
        "base-and-exponent-swapped",
        "Writing a power with its base and exponent exchanged when counting functions.",
        area="university",
    ),
    CatalogueEntry(
        "basic-interval-ends-swapped",
        "Reflecting the wrong percentile in a basic bootstrap interval: using twice the estimate minus the "
        "lower percentile for the lower bound.",
        area="university",
    ),
    CatalogueEntry(
        "bias-as-percentage",
        "Reporting the bias as a percentage of the estimate instead of in the units of the estimate.",
        area="university",
    ),
    CatalogueEntry(
        "bias-correction-doubled",
        "Subtracting twice the estimated bias when correcting an estimate.",
        area="university",
    ),
    CatalogueEntry(
        "bias-correction-omitted",
        "Reporting the original estimate and leaving out the bias correction.",
        area="university",
    ),
    CatalogueEntry(
        "bias-correction-wrong-direction",
        "Adding the estimated bias to the estimate instead of subtracting it.",
        area="university",
    ),
    CatalogueEntry(
        "bias-taken-as-bootstrap-mean",
        "Reporting the mean of the bootstrap values as the bias, without subtracting the original estimate.",
        area="university",
    ),
    CatalogueEntry(
        "bound-from-first-term-only",
        "Taking the dominating function from the first function of the sequence alone, missing a larger value "
        "later or at the limit.",
        area="university",
    ),
    CatalogueEntry(
        "bound-from-limit-only",
        "Taking the dominating function from the pointwise limit alone, missing larger values earlier in the sequence.",
        area="university",
    ),
    CatalogueEntry(
        "bound-not-the-smallest",
        "Giving a valid but larger dominating function (the sum of absolute parts) where the smallest one was asked.",
        area="university",
    ),
    CatalogueEntry(
        "column-sum-for-row-sum",
        "Using the largest absolute column sum where the infinity norm needs the largest absolute row sum.",
        area="university",
    ),
    CatalogueEntry(
        "confuses-no-image-with-multiple-images",
        "Reporting how many elements have several images where the count of elements with no image was asked.",
        area="university",
    ),
    CatalogueEntry(
        "confuses-pairs-with-elements",
        "Counting the pairs of a relation where its elements were asked about, or the reverse, and "
        "subtracting one from the other.",
        area="university",
    ),
    CatalogueEntry(
        "consecutive-difference-as-tail-spread",
        "Using the gap between two consecutive terms as the largest spread over a whole tail of a sequence.",
        area="university",
    ),
    CatalogueEntry(
        "coordinates-reported-as-standard-vector",
        "Reading the coordinates in a non-standard basis as if they were the entries of the vector itself.",
        area="university",
    ),
    CatalogueEntry(
        "counts-accepting-states",
        "Giving the number of accepting states as the number of accepted words.",
        area="university",
    ),
    CatalogueEntry(
        "counts-all-failures-together",
        "Adding the elements that fail one condition to those that fail another and answering with the total.",
        area="university",
    ),
    CatalogueEntry(
        "counts-all-functions-from-a-to-b",
        "Counting every function between two sets where the choices were restricted by a relation.",
        area="university",
    ),
    CatalogueEntry(
        "counts-conjuncts-as-steps",
        "Taking the number of conjuncts in the target as the number of inference steps of one kind.",
        area="university",
    ),
    CatalogueEntry(
        "counts-elements-not-pairs",
        "Counting the elements that have surplus images instead of the surplus pairs.",
        area="university",
    ),
    CatalogueEntry(
        "counts-every-given-implication",
        "Counting every implication among the premises as used, including those beyond what the target needs.",
        area="university",
    ),
    CatalogueEntry(
        "counts-only-minimum-subcovers",
        "Counting only the smallest subcovers where every subfamily that covers was asked for.",
        area="university",
    ),
    CatalogueEntry(
        "counts-only-some-unused-premises",
        "Counting only part of the premises a derivation does not use, such as only the irrelevant ones or "
        "only the unused chain.",
        area="university",
    ),
    CatalogueEntry(
        "counts-points-not-sets",
        "Counting the points covered by exactly one member where the members that must be kept were asked for.",
        area="university",
    ),
    CatalogueEntry(
        "counts-points-over-largest-set",
        "Dividing the number of points by the size of the largest member and calling that the smallest subcover size.",
        area="university",
    ),
    CatalogueEntry(
        "counts-single-image-elements",
        "Counting the elements with exactly one image where the elements with none were asked for.",
        area="university",
    ),
    CatalogueEntry(
        "counts-steps-not-lines",
        "Counting only the inference steps of a derivation and leaving out its premise lines.",
        area="university",
    ),
    CatalogueEntry(
        "counts-the-complement",
        "Counting the items or rows that fail the condition where those that meet it were asked for.",
        area="university",
    ),
    CatalogueEntry(
        "counts-unused-codomain-elements",
        "Counting the elements of the target set that no pair reaches instead of the source elements with no image.",
        area="university",
    ),
    CatalogueEntry(
        "counts-unused-premises-as-lines",
        "Including premises a derivation never uses in its line count.",
        area="university",
    ),
    CatalogueEntry(
        "deletes-every-pair-of-a-multi-image-element",
        "Deleting all the pairs of an element with several images, instead of all but one.",
        area="university",
    ),
    CatalogueEntry(
        "determinant-formula-slip",
        "Using the wrong terms or signs in the 2 by 2 determinant: adding the cross terms, pairing the wrong "
        "entries, or dropping one product.",
        area="university",
    ),
    CatalogueEntry(
        "divides-by-n-minus-one-for-mean",
        "Dividing the total by one fewer than the number of values when finding a mean.",
        area="university",
    ),
    CatalogueEntry(
        "divides-by-total-mass",
        "Dividing an integral by the total mass of the space, turning it into an average.",
        area="university",
    ),
    CatalogueEntry(
        "divides-instead-of-multiplies",
        "Dividing two quantities where they should be multiplied.",
        area="university",
    ),
    CatalogueEntry(
        "doubles-the-energy-decay-rate",
        "Using four times the decay constant where the energy decays at twice the decay constant.",
        area="university",
    ),
    CatalogueEntry(
        "energy-decays-like-u-not-u-squared",
        "Using the decay rate of the solution u for its energy, which decays twice as fast because it is an "
        "integral of u squared.",
        area="university",
    ),
    CatalogueEntry(
        "energy-rate-for-decay-rate",
        "Giving the energy decay rate, twice the decay constant, where the decay constant itself was asked.",
        area="university",
    ),
    CatalogueEntry(
        "epsilon-taken-as-the-gap",
        "Reporting the allowed tolerance as the actual gap, though the gap at the chosen index is smaller.",
        area="university",
    ),
    CatalogueEntry(
        "essential-members-taken-as-subcover",
        "Taking the number of essential members as the size of a smallest subcover, though they need not "
        "cover the space.",
        area="university",
    ),
    CatalogueEntry(
        "first-term-as-limit",
        "Taking the first term of a sequence as its limit.",
        area="university",
    ),
    CatalogueEntry(
        "forgets-the-conjunction-premise",
        "Leaving the conjunction premise out of the count of premises a derivation uses.",
        area="university",
    ),
    CatalogueEntry(
        "forgot-diffusivity",
        "Leaving the diffusivity out of the decay constant of a heat-equation mode.",
        area="university",
    ),
    CatalogueEntry(
        "forgot-factor-two-in-energy-rate",
        "Leaving out the factor 2 when differentiating an energy that decays like the exponential of minus "
        "twice the decay constant times t.",
        area="university",
    ),
    CatalogueEntry(
        "forgot-half-from-integral-of-sin-squared",
        "Taking the integral of sin squared over a whole number of half-waves as the interval length, "
        "forgetting the factor one half.",
        area="university",
    ),
    CatalogueEntry(
        "forgot-interval-length",
        "Leaving the length of the interval out of the integral of a squared eigenfunction.",
        area="university",
    ),
    CatalogueEntry(
        "forgot-to-square-wavenumber",
        "Using the wavenumber where its square belongs in the decay constant of a heat-equation mode.",
        area="university",
    ),
    CatalogueEntry(
        "gap-taken-as-already-divided-by-n",
        "Treating the gap at the chosen index as the constant of the bound, so dividing by that index twice.",
        area="university",
    ),
    CatalogueEntry(
        "greedy-choice-taken-as-minimum",
        "Taking the size of a cover built by always picking the largest remaining member as the smallest "
        "possible subcover.",
        area="university",
    ),
    CatalogueEntry(
        "halves-the-decay-rate",
        "Halving the decay constant of a heat-equation mode.",
        area="university",
    ),
    CatalogueEntry(
        "halves-the-integral-twice",
        "Taking a quarter of the interval length times the squared amplitude, halving the integral of sin "
        "squared twice.",
        area="university",
    ),
    CatalogueEntry(
        "ignores-choice-among-several-images",
        "Counting only the free choices and ignoring the choice of image where a relation gives several.",
        area="university",
    ),
    CatalogueEntry(
        "ignores-free-elements",
        "Counting only the choices among given images and ignoring elements that may map anywhere.",
        area="university",
    ),
    CatalogueEntry(
        "ignores-second-basis-vector",
        "Dividing a coordinate of the vector by the matching entry of one basis vector, as if the other basis "
        "vector were absent.",
        area="university",
    ),
    CatalogueEntry(
        "ignores-the-index-in-tail-bound",
        "Giving the constant of a tail bound without dividing by the index.",
        area="university",
    ),
    CatalogueEntry(
        "ignores-the-measure-weights",
        "Adding the function values at the points without weighting them by the masses of the points.",
        area="university",
    ),
    CatalogueEntry(
        "integrates-absolute-value-of-limit",
        "Integrating the absolute value of the limit function where the limit itself was asked.",
        area="university",
    ),
    CatalogueEntry(
        "inverse-without-swapping-entries",
        "Forming a 2 by 2 inverse without swapping the diagonal entries and negating the others.",
        area="university",
    ),
    CatalogueEntry(
        "inverts-the-error-amplification",
        "Dividing the input error by the condition number where the bound multiplies the two.",
        area="university",
    ),
    CatalogueEntry(
        "leaves-out-a-kind-of-inference-step",
        "Leaving a whole kind of step (elimination, introduction or the final disjunction step) out of the "
        "count of inference steps.",
        area="university",
    ),
    CatalogueEntry(
        "limit-includes-constant-term",
        "Adding the constant term of the sequence's offset to the limit.",
        area="university",
    ),
    CatalogueEntry(
        "limit-inverted",
        "Giving the reciprocal of the ratio of leading coefficients as the limit.",
        area="university",
    ),
    CatalogueEntry(
        "limit-numerator-coefficient-only",
        "Giving the leading coefficient of the numerator as the limit, without dividing by the leading "
        "coefficient of the denominator.",
        area="university",
    ),
    CatalogueEntry(
        "limit-sign-error-on-constant",
        "Subtracting where the constant term should be added when reading off the limit.",
        area="university",
    ),
    CatalogueEntry(
        "limit-taken-as-constant-term",
        "Taking the constant term of the numerator over the denominator coefficient as the limit.",
        area="university",
    ),
    CatalogueEntry(
        "limit-value-for-nth-term",
        "Giving the limit's value where the value at a particular index was asked.",
        area="university",
    ),
    CatalogueEntry(
        "log-base-confusion",
        "Using a base 10 logarithm where the natural logarithm belongs in an exponential decay time.",
        area="university",
    ),
    CatalogueEntry(
        "map-factors-on-wrong-basis-vectors",
        "Multiplying each coordinate by the other basis vector's scaling factor.",
        area="university",
    ),
    CatalogueEntry(
        "median-for-mean",
        "Reporting the median where the mean was asked.",
        area="university",
    ),
    CatalogueEntry(
        "midrange-for-mean",
        "Reporting the average of the smallest and largest values as the mean.",
        area="university",
    ),
    CatalogueEntry(
        "miscounts-inference-steps",
        "Miscounting the elimination and introduction steps of a derivation, for example adding the number of "
        "conjuncts to the elimination steps.",
        area="university",
    ),
    CatalogueEntry(
        "multiplies-instead-of-adds",
        "Multiplying two counts where they should be added.",
        area="university",
    ),
    CatalogueEntry(
        "no-error-amplification",
        "Giving the input error as the output error, ignoring the condition number.",
        area="university",
    ),
    CatalogueEntry(
        "norm-of-inverse-reported-as-condition-number",
        "Reporting the norm of the inverse as the condition number, leaving out the norm of the matrix.",
        area="university",
    ),
    CatalogueEntry(
        "omits-decay-constant",
        "Leaving the decay constant out of the initial rate of change of an energy.",
        area="university",
    ),
    CatalogueEntry(
        "omits-initial-energy",
        "Leaving the initial energy out of the initial rate of change of an energy.",
        area="university",
    ),
    CatalogueEntry(
        "omits-the-whole-cover",
        "Counting the covering subfamilies other than the whole cover, which also covers.",
        area="university",
    ),
    CatalogueEntry(
        "percent-not-converted",
        "Using a percentage as if it were a fraction, so the answer is out by a factor of 100.",
        area="university",
    ),
    CatalogueEntry(
        "percentile-interval-for-basic",
        "Giving a percentile endpoint of the bootstrap values where the basic interval reflects it about the estimate.",
        area="university",
    ),
    CatalogueEntry(
        "percentile-rank-off-by-one",
        "Using the neighbouring order statistic for a percentile of a bootstrap sample.",
        area="university",
    ),
    CatalogueEntry(
        "reads-1-over-n-as-n",
        "Reading a term c over n as c times n.",
        area="university",
    ),
    CatalogueEntry(
        "reads-the-word-backwards",
        "Running an automaton on the word read from right to left.",
        area="university",
    ),
    CatalogueEntry(
        "reciprocal-of-norm-for-inverse",
        "Giving the reciprocal of the matrix norm as the norm of the inverse.",
        area="university",
    ),
    CatalogueEntry(
        "reports-steps-as-unused-premises",
        "Reporting the number of inference steps where the number of unused premises was asked.",
        area="university",
    ),
    CatalogueEntry(
        "reports-subcover-size-not-discards",
        "Reporting the size of the subcover where the number of members that can be discarded was asked.",
        area="university",
    ),
    CatalogueEntry(
        "reuses-earlier-part-value",
        "Repeating a value from an earlier part of the question instead of working out the one asked for.",
        area="university",
    ),
    CatalogueEntry(
        "rounds-down-instead-of-up",
        "Rounding a threshold down to an integer where the least integer beyond it is needed.",
        area="university",
    ),
    CatalogueEntry(
        "sign-error-in-cramers-numerator",
        "Adding the cross terms in Cramer's rule where one is subtracted.",
        area="university",
    ),
    CatalogueEntry(
        "single-point-sets-taken-as-essential",
        "Calling every one-point member of a cover essential, though another member may hold the same point.",
        area="university",
    ),
    CatalogueEntry(
        "solves-linear-not-exponential",
        "Dividing by the rate to find when a quantity decays by a factor, instead of using a logarithm.",
        area="university",
    ),
    CatalogueEntry(
        "stops-one-symbol-early",
        "Stopping the run of an automaton one symbol before the end of the word.",
        area="university",
    ),
    CatalogueEntry(
        "swaps-coefficients",
        "Giving the second coordinate for the first (or the reverse) when expressing a vector in a basis.",
        area="university",
    ),
    CatalogueEntry(
        "swaps-the-two-repairs",
        "Giving the number of elements to repair where the number of pairs to delete was asked, or the reverse.",
        area="university",
    ),
    CatalogueEntry(
        "tail-bound-from-next-index",
        "Using the index after the stated one when bounding the spread of a tail.",
        area="university",
    ),
    CatalogueEntry(
        "tail-spread-doubled",
        "Doubling the spread of a tail, as if the two terms compared could lie on opposite sides of the limit.",
        area="university",
    ),
    CatalogueEntry(
        "takes-larger-of-the-two-repairs",
        "Giving the larger of the two repair counts where they must be added.",
        area="university",
    ),
    CatalogueEntry(
        "takes-whole-cover-as-minimum",
        "Taking the size of the whole cover as the smallest subcover.",
        area="university",
    ),
    CatalogueEntry(
        "total-instead-of-mean",
        "Giving the total of the values where their mean was asked.",
        area="university",
    ),
    CatalogueEntry(
        "traces-transition-table-wrongly",
        "Following a wrong entry of the transition table during a run of an automaton.",
        area="university",
    ),
    CatalogueEntry(
        "uses-the-wrong-word-length",
        "Counting accepted words of a different length from the one asked, or adding counts for different lengths.",
        area="university",
    ),
    CatalogueEntry(
        "wrong-matrix-norm-formula",
        "Using a wrong rule for the infinity norm: the largest entry, the sum of every entry, or the smaller row sum.",
        area="university",
    ),
    CatalogueEntry(
        "sampling-fraction-from-one-stratum",
        "Working out the sampling fraction from the size of one stratum instead of the whole population.",
        area="sampling",
    ),
    CatalogueEntry(
        "part-to-part-instead-of-part-to-whole",
        "Comparing one part with the rest instead of with the whole: the sample as a fraction of the people not "
        "sampled.",
        area="sampling",
    ),
    CatalogueEntry(
        "equal-allocation-ignores-size",
        "Sharing a stratified sample equally between the strata instead of in proportion to their sizes.",
        area="sampling",
    ),
    CatalogueEntry(
        "leftover-ignores-a-stratum",
        "Finding the last stratum's share by subtracting only some of the other strata's shares from the sample size.",
        area="sampling",
    ),
    CatalogueEntry(
        "complement-fraction-taken",
        "Using the fraction of the population left out (one minus the sampling fraction) in place of the fraction "
        "sampled.",
        area="sampling",
    ),
    CatalogueEntry(
        "cdf-vs-survival-confused",
        "Giving the probability of the opposite tail: P(T > b) where P(T <= b) was asked, or the reverse.",
        area="probability",
    ),
    CatalogueEntry(
        "forgot-density-factor",
        "Integrating e^(-lambda t) without the factor lambda that the density carries, so the area is out by a "
        "factor of lambda.",
        area="probability",
    ),
    CatalogueEntry(
        "positive-exponent-in-density",
        "Losing the minus sign in the exponent of the density, so the curve grows instead of decaying.",
        area="probability",
    ),
    CatalogueEntry(
        "evaluated-one-limit-only",
        "Evaluating a definite integral at one limit only and ignoring the other.",
        area="integration",
    ),
    CatalogueEntry(
        "wrong-limits-lower-bound-zero",
        "Integrating over an interval of the right width but starting at zero instead of at the stated lower limit.",
        area="integration",
    ),
    CatalogueEntry(
        "omitted-first-interval",
        "Taking a tail probability as one minus only one of the earlier probabilities instead of both.",
        area="probability",
    ),
    CatalogueEntry(
        "conditional-ignored",
        "Giving the unconditional probability where a conditional one was asked.",
        area="probability",
    ),
    CatalogueEntry(
        "dropped-chain-rule-factor",
        "Differentiating sin(bt) as cos(bt), leaving out the chain-rule factor b.",
        area="differentiation",
    ),
    CatalogueEntry(
        "differentiated-twice",
        "Differentiating one more time than the question asks, so the answer carries an extra factor from the "
        "chain rule.",
        area="differentiation",
    ),
    CatalogueEntry(
        "forgot-divide-angle-by-b",
        "Solving cos(bt) = r for the angle and stopping there, without dividing by b to get t.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "arccos-sign-slip",
        "Looking up the inverse cosine of the opposite sign, so the angle is measured from the wrong side.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "later-root-instead-of-first",
        "Reporting a later solution of a trigonometric equation instead of the smallest positive one.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "sine-cosine-swapped",
        "Using sine where cosine is needed, or cosine where sine is needed, when evaluating at the angle.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "reflected-about-mean",
        "Reflecting the deviation from the mean level the wrong way, so the value is on the wrong side of the mean.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "took-maximum-instead",
        "Giving the greatest value of the quantity instead of its value at the stated time.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "dropped-negative-from-derivative",
        "Losing the minus sign when differentiating cosine, so the derivative of cos has the wrong sign.",
        area="differentiation",
    ),
    CatalogueEntry(
        "ignores-trig-evaluation",
        "Treating the sine or cosine factor as 1 instead of evaluating it at the stated angle.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "gave-range-for-greatest-value",
        "Giving the full range from least to greatest value where the greatest value alone was asked.",
        area="general",
    ),
    CatalogueEntry(
        "power-of-point-not-divided",
        "Using TM squared as the length of the secant without dividing by BM.",
        area="circle geometry",
    ),
    CatalogueEntry(
        "chord-for-secant",
        "Giving the chord AB where the whole secant AM was asked for, or the secant where the chord was asked for.",
        area="circle geometry",
    ),
    CatalogueEntry(
        "whole-secant-when-part-asked",
        "Reporting the whole secant AM where the chord AB, the part of it inside the circle, was asked for.",
        area="circle geometry",
    ),
    CatalogueEntry(
        "half-chord-for-chord",
        "Using half the chord where the whole chord is needed, or the whole chord where half is needed.",
        area="circle geometry",
    ),
    CatalogueEntry(
        "linear-instead-of-pythagoras",
        "Subtracting two lengths where Pythagoras' theorem is needed for the third side of a right triangle.",
        area="circle geometry",
    ),
    CatalogueEntry(
        "no-square-root",
        "Finding the sum or difference of squares and stopping before taking the square root.",
        area="general",
    ),
    CatalogueEntry(
        "added-squares-for-leg",
        "Adding the squares of the two known sides to find a leg, as if finding the hypotenuse.",
        area="circle geometry",
    ),
    CatalogueEntry(
        "ignored-half-chord",
        "Measuring along the secant from M to B only, leaving out the half-chord from B to the foot of the "
        "perpendicular.",
        area="circle geometry",
    ),
    CatalogueEntry(
        "chord-for-half-chord",
        "Measuring along the secant from M to A, the whole chord past B, where only half the chord is needed.",
        area="circle geometry",
    ),
    CatalogueEntry(
        "wrong-trig-ratio",
        "Using sine where the two sides given are the opposite and adjacent, which call for tangent.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "forgot-to-divide-by-time",
        "Finding the growth factor over the whole interval and stopping, without dividing by the length of the "
        "interval to get the rate constant.",
        area="exponential",
    ),
    CatalogueEntry(
        "average-rate-for-k",
        "Taking the average rate of change over the interval as the constant k in dy/dt = ky.",
        area="exponential",
    ),
    CatalogueEntry(
        "log-of-value-not-ratio",
        "Taking the logarithm of the final value alone instead of the logarithm of the ratio of final to initial "
        "value.",
        area="exponential",
    ),
    CatalogueEntry(
        "linear-instead-of-exponential",
        "Assuming the quantity changes by the same amount each unit of time, instead of by the same factor.",
        area="exponential",
    ),
    CatalogueEntry(
        "forgot-time-in-exponent",
        "Evaluating the exponential for one unit of time instead of for the stated time.",
        area="exponential",
    ),
    CatalogueEntry(
        "sign-error-in-exponent",
        "Using the wrong sign in the exponent, so growth is treated as decay or the reverse.",
        area="exponential",
    ),
    CatalogueEntry(
        "rate-at-start-for-rate-now",
        "Using the starting value of the quantity to find a rate at a later time.",
        area="exponential",
    ),
    CatalogueEntry(
        "multiplied-by-time-once-more",
        "Multiplying a rate by the elapsed time again after it has already been found.",
        area="exponential",
    ),
    CatalogueEntry(
        "forgot-to-divide-by-rate",
        "Solving the exponential equation for kt and stopping, without dividing by k to get t.",
        area="exponential",
    ),
    CatalogueEntry(
        "proportional-instead-of-exponential",
        "Assuming the time to reach an amount is proportional to the amount, instead of to its logarithm.",
        area="exponential",
    ),
    CatalogueEntry(
        "wrong-starting-value",
        "Starting the exponential from the value at a later reading instead of from the initial value.",
        area="exponential",
    ),
    CatalogueEntry(
        "ignores-lower-terms",
        "Evaluating only the leading term of a polynomial instead of all of its terms.",
        area="polynomials",
    ),
    CatalogueEntry(
        "only-multiplied-first-terms",
        "Multiplying only the first term of the binomial through the second factor and leaving the other term out.",
        area="expansion",
    ),
    CatalogueEntry(
        "sign-error-on-one-term",
        "Flipping the sign of one term while distributing, so a single coefficient of the expansion has the wrong "
        "sign.",
        area="expansion",
    ),
    CatalogueEntry(
        "misreads-a-branch",
        "Following the wrong branch at one test, so a shape lands in a neighbouring bin.",
        area="algorithms",
    ),
    CatalogueEntry(
        "counted-a-shape-from-the-wrong-bin",
        "Counting or including a shape that the procedure sends to a different bin.",
        area="algorithms",
    ),
    CatalogueEntry(
        "counted-the-shapes-left-out",
        "Giving the number of shapes outside the bin where the number inside was asked.",
        area="algorithms",
    ),
    CatalogueEntry(
        "left-out-a-shape",
        "Leaving one of the shapes that belongs to the group out of a count or a total.",
        area="algorithms",
    ),
    CatalogueEntry(
        "ignored-the-sorting",
        "Combining every shape in the list instead of only those the procedure sends to the bin.",
        area="algorithms",
    ),
    CatalogueEntry(
        "divided-by-all-the-shapes",
        "Dividing a total by the number of all the shapes instead of the number in the group.",
        area="statistics",
    ),
    CatalogueEntry(
        "ignored-the-new-shape",
        "Leaving a newly added value out of a mean, so the mean does not change.",
        area="statistics",
    ),
    CatalogueEntry(
        "new-shape-not-counted",
        "Adding a new value to the total of a mean but not adding one to the number of values.",
        area="statistics",
    ),
    CatalogueEntry(
        "average-of-averages",
        "Averaging the old mean with the new value instead of weighting by the number of values behind the mean.",
        area="statistics",
    ),
    CatalogueEntry(
        "forgot-period-scaling",
        "Putting t straight into the cosine instead of scaling it by 2 pi over the period first.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "period-factor-slip",
        "Using the period over pi where the period over 2 pi is needed when converting an angle back to a time.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "half-period-guess",
        "Guessing half the period as the time taken, without solving the equation.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "forgot-symmetry-of-cycle",
        "Reporting one crossing time, or the period less one crossing time, as the time spent above a level, "
        "missing that a cycle is symmetric about its peak.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "time-below-for-time-above",
        "Giving the time spent below a level where the time above it was asked for.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "percent-of-wrong-time",
        "Taking the time to first reach the level as a percentage of the period instead of the time spent above it.",
        area="percentages",
    ),
    CatalogueEntry(
        "gave-time-not-percent",
        "Giving the time spent where the share of the cycle, as a percentage, was asked for.",
        area="percentages",
    ),
    CatalogueEntry(
        "range-for-amplitude",
        "Giving the full range from lowest to highest value where the amplitude, half of it, was asked.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "mean-for-amplitude",
        "Giving the mean level of the oscillation where its amplitude was asked.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "half-highest-for-amplitude",
        "Halving the highest value to get the amplitude, forgetting that the lowest value is not zero.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "total-time-for-period",
        "Giving the total time for all the revolutions where the time for one revolution was asked.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "radical-coefficient-not-rooted",
        "Dividing by the coefficient under a square root without taking its square root first.",
        area="radicals",
    ),
    CatalogueEntry(
        "ignored-the-square-root",
        "Simplifying only the cube root and never applying the square root in the denominator.",
        area="radicals",
    ),
    CatalogueEntry(
        "dropped-sign-of-cube-root",
        "Losing the minus sign of a negative cube root, which is a real number.",
        area="radicals",
    ),
    CatalogueEntry(
        "variable-on-wrong-side-of-bar",
        "Putting a variable on the wrong side of the fraction bar, so its exponent has the wrong sign.",
        area="radicals",
    ),
    CatalogueEntry(
        "forgot-root-index",
        "Subtracting the exponents under the roots without dividing each by its root index.",
        area="radicals",
    ),
    CatalogueEntry(
        "wrong-root-index",
        "Dividing each exponent by the wrong root index: halving the cube-root exponent and thirding the square- "
        "root one.",
        area="radicals",
    ),
    CatalogueEntry(
        "swapped-values",
        "Substituting the two values into the wrong variables.",
        area="substitution",
    ),
    CatalogueEntry(
        "dropped-coefficient",
        "Evaluating the powers of the variables and leaving the coefficient out.",
        area="substitution",
    ),
    CatalogueEntry(
        "same-power-on-both",
        "Applying the combined exponent to the product of the variables instead of each exponent to its own variable.",
        area="substitution",
    ),
    CatalogueEntry(
        "poisson-forgets-factorial",
        "Using lambda^k e^(-lambda) for a Poisson probability without dividing by k!.",
        area="probability",
    ),
    CatalogueEntry(
        "poisson-rate-not-mean",
        "Using the hourly rate in place of the mean number of arrivals over the stated window.",
        area="probability",
    ),
    CatalogueEntry(
        "poisson-off-by-one-count",
        "Using the probability of one count more or fewer than the one asked for.",
        area="probability",
    ),
    CatalogueEntry(
        "poisson-cumulative-for-exact",
        "Adding the probabilities of every count up to k when exactly k was asked.",
        area="probability",
    ),
    CatalogueEntry(
        "at-least-one-complement-slip",
        "Taking the complement of the wrong event: subtracting the probability of one event, not of none, from 1.",
        area="probability",
    ),
    CatalogueEntry(
        "conditional-as-product",
        "Multiplying the two probabilities where a conditional probability is a quotient.",
        area="probability",
    ),
    CatalogueEntry(
        "conditional-ratio-inverted",
        "Dividing the other way round: P(B)/P(A and B) where P(A and B)/P(B) was needed.",
        area="probability",
    ),
    CatalogueEntry(
        "next-window-ignored",
        "Leaving out the probability for the second time window, or scaling it as if the window had the same length.",
        area="probability",
    ),
    CatalogueEntry(
        "at-least-one-for-none",
        "Using the probability of at least one event where none was asked.",
        area="probability",
    ),
    CatalogueEntry(
        "poisson-factorial-off-by-one",
        "Dividing by (k-1)! instead of k! in a Poisson probability.",
        area="probability",
    ),
    CatalogueEntry(
        "exactly-read-as-at-least",
        "Reading 'exactly k' as 'k or more' and adding the probabilities of every larger count.",
        area="probability",
    ),
    CatalogueEntry(
        "exactly-one-for-at-least-one",
        "Giving the probability of exactly one event where at least one was asked.",
        area="probability",
    ),
    CatalogueEntry(
        "ridge-penalty-omitted",
        "Solving the plain normal equations A^T A x = A^T b and leaving out the ridge penalty.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "ridge-penalty-added-once",
        "Adding the ridge penalty to only one diagonal entry of A^T A instead of both.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "inverse-off-diagonal-sign-error",
        "Keeping the sign of the off-diagonal entry when writing the inverse of a 2 by 2 matrix.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "components-swapped",
        "Giving the two components of a pair in the wrong order.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "prediction-ignores-input",
        "Predicting with the sum of the coefficients, ignoring the values of the new inputs.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "right-hand-side-swapped",
        "Swapping the two entries of the right-hand side vector A^T b when solving the system.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "gaussian-prefactor-dropped",
        "Writing the transform of a Gaussian without its sqrt(pi/a) factor.",
        area="fourier",
    ),
    CatalogueEntry(
        "gaussian-exponent-2a",
        "Using 2a instead of 4a under w^2 in the transform of a Gaussian.",
        area="fourier",
    ),
    CatalogueEntry(
        "gaussian-exponent-not-inverted",
        "Keeping a multiplied by w^2 in the exponent instead of dividing w^2 by 4a.",
        area="fourier",
    ),
    CatalogueEntry(
        "gaussian-prefactor-inverted",
        "Inverting a inside the square-root prefactor: sqrt(a/pi) in place of sqrt(pi/a).",
        area="fourier",
    ),
    CatalogueEntry(
        "modulation-half-forgotten",
        "Adding the two shifted transforms without the factor 1/2 that cos(w0 x) = (e^(i w0 x) + e^(-i w0 x))/2 brings.",
        area="fourier",
    ),
    CatalogueEntry(
        "modulation-one-sideband-only",
        "Shifting the transform in one direction only, as if cos were a single complex exponential.",
        area="fourier",
    ),
    CatalogueEntry(
        "modulation-ignored",
        "Evaluating the integral with the cosine left out.",
        area="fourier",
    ),
    CatalogueEntry(
        "sideband-distance-halved",
        "Taking the other shifted copy to be w0 away from the carrier frequency when it is 2 w0 away.",
        area="fourier",
    ),
    CatalogueEntry(
        "adds-degrees-in-composition",
        "Adding the degrees of two power maps where the degree of a composite is their product.",
        area="topology",
    ),
    CatalogueEntry(
        "drops-sign-of-degree",
        "Reporting the size of a negative degree and losing its sign.",
        area="topology",
    ),
    CatalogueEntry(
        "ignores-second-map",
        "Using only one of the two maps in a composite.",
        area="topology",
    ),
    CatalogueEntry(
        "powers-for-composition",
        "Raising one degree to the power of the other where composing power maps multiplies the degrees.",
        area="topology",
    ),
    CatalogueEntry(
        "ignores-the-loop-class",
        "Answering with the degree itself and leaving out the class of the loop it acts on.",
        area="topology",
    ),
    CatalogueEntry(
        "index-sign-kept",
        "Giving a negative index: the index of kZ in Z is |k|, never negative.",
        area="topology",
    ),
    CatalogueEntry(
        "counts-largest-residue",
        "Taking the index of kZ in Z to be one less or one more than |k|, the number of cosets.",
        area="topology",
    ),
    CatalogueEntry(
        "ignores-the-map",
        "Answering with the given output value as if the map did nothing.",
        area="general",
    ),
    CatalogueEntry(
        "counts-generators-as-atoms",
        "Taking the number of atoms to be the number of sets in the family.",
        area="measure",
    ),
    CatalogueEntry(
        "misses-the-outside-atom",
        "Leaving out the atom of points that lie in none of the sets of the family.",
        area="measure",
    ),
    CatalogueEntry(
        "counts-points-as-atoms",
        "Counting every point of the universe as an atom.",
        area="measure",
    ),
    CatalogueEntry(
        "counts-empty-atoms",
        "Counting every pattern of membership as an atom, including patterns no point has.",
        area="measure",
    ),
    CatalogueEntry(
        "omits-the-empty-set",
        "Leaving the empty set out when counting the members of a sigma-algebra.",
        area="measure",
    ),
    CatalogueEntry(
        "squares-instead-of-powering",
        "Squaring the number of atoms where the sets are counted by 2 raised to that number.",
        area="measure",
    ),
    CatalogueEntry(
        "doubles-instead-of-powering",
        "Doubling the number of atoms where the sets are counted by 2 raised to that number.",
        area="measure",
    ),
    CatalogueEntry(
        "takes-whole-power-set",
        "Counting every subset of the universe as a member of the generated sigma-algebra.",
        area="measure",
    ),
    CatalogueEntry(
        "counts-generator-patterns-as-sets",
        "Counting 2 raised to the number of generating sets, which is the number of membership patterns, not the number of sets.",
        area="measure",
    ),
    CatalogueEntry(
        "does-not-remove-given-sets",
        "Counting all the sets of sigma(F) as the ones to add, forgetting that the sets of F are already there.",
        area="measure",
    ),
    CatalogueEntry(
        "assumes-empty-and-whole-present",
        "Also removing the empty set and the whole universe, which the family does not contain.",
        area="measure",
    ),
    CatalogueEntry(
        "confuses-atoms-with-members",
        "Subtracting the number of generators from the number of atoms instead of from the number of sets.",
        area="measure",
    ),
    CatalogueEntry(
        "union-read-as-intersection",
        "Swapping union and intersection when reading a set built from several sets.",
        area="measure",
    ),
    CatalogueEntry(
        "complement-dropped",
        "Leaving out a complement when reading a set built from several sets.",
        area="measure",
    ),
    CatalogueEntry(
        "complement-misplaced",
        "Taking the complement of the whole expression where it applies to one set only, or the reverse.",
        area="measure",
    ),
    CatalogueEntry(
        "de-morgan-not-swapped",
        "Complementing each set but keeping the union, where the De Morgan law turns it into an intersection.",
        area="measure",
    ),
    CatalogueEntry(
        "symmetric-difference-one-side-only",
        "Taking only the points of A that are not in B as the symmetric difference.",
        area="measure",
    ),
    CatalogueEntry(
        "complement-for-set",
        "Measuring the complement of the set asked for.",
        area="measure",
    ),
    CatalogueEntry(
        "double-counts-shared-edges",
        "Counting an edge shared by two triangles once for each triangle.",
        area="topology",
    ),
    CatalogueEntry(
        "forgets-edges-of-triangles",
        "Counting only the edges listed on their own and leaving out the edges of the triangles.",
        area="topology",
    ),
    CatalogueEntry(
        "counts-triangles-as-edges",
        "Adding the number of triangles to the number of edges.",
        area="topology",
    ),
    CatalogueEntry(
        "forgets-the-triangles",
        "Leaving the triangles out of the Euler characteristic.",
        area="topology",
    ),
    CatalogueEntry(
        "wrong-signs-in-euler-characteristic",
        "Subtracting the triangles in vertices - edges + triangles instead of adding them.",
        area="topology",
    ),
    CatalogueEntry(
        "counts-vertices-as-components",
        "Giving the number of vertices as the number of connected components.",
        area="topology",
    ),
    CatalogueEntry(
        "assumes-connected",
        "Assuming the complex is connected, so that b0 = 1.",
        area="topology",
    ),
    CatalogueEntry(
        "counts-triangles-as-components",
        "Giving the number of triangles as the number of connected components.",
        area="topology",
    ),
    CatalogueEntry(
        "euler-relation-sign-slip",
        "Writing b1 = b0 + chi where b0 - b1 + b2 = chi gives b1 = b0 - chi when b2 = 0.",
        area="topology",
    ),
    CatalogueEntry(
        "drops-b0-from-euler-relation",
        "Taking b1 = -chi, leaving out the b0 term of the Euler relation.",
        area="topology",
    ),
    CatalogueEntry(
        "ignores-isolated-vertex",
        "Leaving out an isolated vertex when counting connected components: a lone vertex is a component.",
        area="topology",
    ),
    CatalogueEntry(
        "order-of-group-for-exponent",
        "Giving the order of the group as the largest order of an element.",
        area="algebra",
    ),
    CatalogueEntry(
        "max-of-component-orders",
        "Taking the largest element order of a product to be the larger of the two factor orders.",
        area="algebra",
    ),
    CatalogueEntry(
        "gcd-for-lcm",
        "Using the gcd of the factor orders where the lcm gives the largest element order.",
        area="algebra",
    ),
    CatalogueEntry(
        "adds-orders",
        "Adding the orders of the two factors where an order of a pair is a least common multiple.",
        area="algebra",
    ),
    CatalogueEntry(
        "counts-order-dividing-d",
        "Counting the elements whose order divides d, not those of order exactly d.",
        area="algebra",
    ),
    CatalogueEntry(
        "counts-one-cyclic-subgroup",
        "Counting only the generators of a single cyclic subgroup of order d.",
        area="algebra",
    ),
    CatalogueEntry(
        "adds-component-counts",
        "Adding the numbers of elements of order d in the two factors instead of counting pairs.",
        area="algebra",
    ),
    CatalogueEntry(
        "counts-elements-as-subgroups",
        "Giving the number of elements of order d as the number of cyclic subgroups of order d.",
        area="algebra",
    ),
    CatalogueEntry(
        "counts-extra-trivial-subgroup",
        "Adding one for the trivial subgroup when counting the cyclic subgroups of order d.",
        area="algebra",
    ),
    CatalogueEntry(
        "does-not-subtract-cyclic-count",
        "Leaving out the subtraction of the count for the cyclic subgroups when asked how many more.",
        area="algebra",
    ),
    CatalogueEntry(
        "takes-d-minus-1-as-phi",
        "Using d - 1 as the number of elements of order d in a cyclic group, which holds for prime d only.",
        area="algebra",
    ),
    CatalogueEntry(
        "assumes-unique-element-of-order",
        "Taking a cyclic group to have exactly one element of order d.",
        area="algebra",
    ),
    CatalogueEntry(
        "counts-subgroups-not-elements",
        "Giving the difference in the number of cyclic subgroups where the difference in elements was asked.",
        area="algebra",
    ),
    CatalogueEntry(
        "subtracts-subgroup-count",
        "Subtracting the number of cyclic subgroups from the number of elements.",
        area="algebra",
    ),
    CatalogueEntry(
        "sign-error-in-cross-product",
        "Adding the two products in x vy - y vx instead of subtracting.",
        area="mechanics",
    ),
    CatalogueEntry(
        "dot-product-for-cross-product",
        "Using x vx + y vy, which is not the angular momentum.",
        area="mechanics",
    ),
    CatalogueEntry(
        "forgets-mass-in-angular-momentum",
        "Leaving the mass out of the angular momentum, or of the speed found from it.",
        area="mechanics",
    ),
    CatalogueEntry(
        "cross-product-reversed",
        "Taking y vx - x vy, the angular momentum with the wrong sign.",
        area="mechanics",
    ),
    CatalogueEntry(
        "kinetic-energy-only",
        "Giving the kinetic energy as the total energy and leaving out the potential energy.",
        area="mechanics",
    ),
    CatalogueEntry(
        "forgets-half-in-kinetic-energy",
        "Using m v^2 where the kinetic energy is (1/2) m v^2.",
        area="mechanics",
    ),
    CatalogueEntry(
        "doubles-potential-energy",
        "Counting the potential energy twice in the total energy.",
        area="mechanics",
    ),
    CatalogueEntry(
        "potential-energy-only",
        "Giving the potential energy as the total energy and leaving out the kinetic energy.",
        area="mechanics",
    ),
    CatalogueEntry(
        "angular-speed-for-tangential-speed",
        "Dividing L_z by m r^2, which gives the angular speed, not the speed perpendicular to the radius.",
        area="mechanics",
    ),
    CatalogueEntry(
        "ignores-the-radius",
        "Leaving the radius out when finding a speed from the angular momentum.",
        area="mechanics",
    ),
    CatalogueEntry(
        "ignores-tangential-kinetic-energy",
        "Taking all the kinetic energy at the new radius to be radial.",
        area="mechanics",
    ),
    CatalogueEntry(
        "ignores-potential-at-new-radius",
        "Leaving out the potential energy at the new radius.",
        area="mechanics",
    ),
    CatalogueEntry(
        "adds-tangential-energy",
        "Adding the tangential kinetic energy to the radial kinetic energy where it should be taken away.",
        area="mechanics",
    ),
    CatalogueEntry(
        "drops-one-term-of-cross-product",
        "Using only one of the two products in x vy - y vx.",
        area="mechanics",
    ),
    CatalogueEntry(
        "drops-a-premise",
        "Counting with only one of the two premises.",
        area="logic",
    ),
    CatalogueEntry(
        "or-for-and",
        "Counting the rows where at least one premise is true instead of both.",
        area="logic",
    ),
    CatalogueEntry(
        "counts-conclusion-alone",
        "Counting the rows where the conclusion is true (or false), ignoring the premises.",
        area="logic",
    ),
    CatalogueEntry(
        "counts-supporting-rows",
        "Giving the number of rows where premises and conclusion are all true instead of the counterexamples.",
        area="logic",
    ),
    CatalogueEntry(
        "divides-by-all-rows",
        "Dividing by the total number of rows where only the rows satisfying the premises count.",
        area="logic",
    ),
    CatalogueEntry(
        "divides-by-wrong-rows",
        "Dividing by the number of rows where the premises fail instead of the rows where they hold.",
        area="logic",
    ),
    CatalogueEntry(
        "counts-basis-as-topology",
        "Taking the number of open sets to be the number of members of the basis.",
        area="topology",
    ),
    CatalogueEntry(
        "counts-unions-without-collapse",
        "Counting 2 raised to the number of basis members, though different unions can be the same set.",
        area="topology",
    ),
    CatalogueEntry(
        "takes-discrete-topology",
        "Counting every subset of the space as open.",
        area="topology",
    ),
    CatalogueEntry(
        "forgets-the-empty-set",
        "Leaving the empty set out when counting the open sets.",
        area="topology",
    ),
    CatalogueEntry(
        "treats-set-as-open",
        "Taking the interior of a set to be the whole set.",
        area="topology",
    ),
    CatalogueEntry(
        "interior-closure-confused",
        "Giving the interior where the closure was asked, or the reverse.",
        area="topology",
    ),
    CatalogueEntry(
        "counts-basis-members-not-points",
        "Counting the basis members inside the set, not the points of their union.",
        area="topology",
    ),
    CatalogueEntry(
        "treats-set-as-closed",
        "Taking the closure of a set to be the set itself.",
        area="topology",
    ),
    CatalogueEntry(
        "closure-as-union-of-meeting-members",
        "Taking the closure to be the union of the basis members that meet the set.",
        area="topology",
    ),
    CatalogueEntry(
        "closure-as-complement-of-interior",
        "Taking the closure to be the complement of the interior of the same set.",
        area="topology",
    ),
    CatalogueEntry(
        "boundary-as-closure",
        "Giving the closure where the boundary was asked.",
        area="topology",
    ),
    CatalogueEntry(
        "boundary-adds-interior",
        "Adding the interior to the closure instead of subtracting it.",
        area="topology",
    ),
    CatalogueEntry(
        "boundary-inside-set-only",
        "Counting only the boundary points that belong to the set.",
        area="topology",
    ),
    CatalogueEntry(
        "boundary-outside-set-only",
        "Counting only the boundary points that lie outside the set.",
        area="topology",
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
