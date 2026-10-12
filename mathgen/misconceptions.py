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
    # -- arithmetic exam chains ----------------------------------------------------------------------------------
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
        "perimeter-for-area",
        "Finding the perimeter of a rectangle where its area was asked for.",
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
    # -- successive and reverse percentages, factor counts, surds, significant figures, recurring decimals ----------
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
    # -- significant-figures exam tier: a rounding slip that must keep the answer's decimal places ---------------
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
