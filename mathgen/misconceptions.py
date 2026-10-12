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
    # -- early number word problems -------------------------------------------------------------------------------
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
    # -- time and angles (multi-step primary problems) -------------------------------------------------------------
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
    # -- fractions and percentages in multi-step problems ---------------------------------------------------------
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
    # -- decimals and place value (multi-step primary problems) ---------------------------------------------------
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
    # -- estimation, ordering and signed rational numbers -----------------------------------------------------------
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
        "added-instead-of-multiplied",
        "Adding the two numbers when the story is about equal groups and asks for the total.",
        area="whole-numbers",
    ),
    CatalogueEntry(
        "added-instead-of-subtracted",
        "Adding two values to get the difference between them.",
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
    # -- primary multiplication, division and multi-digit arithmetic (exam chains) ----------------------------------
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
        "multiplied-instead-of-divided",
        "Multiplying where the total is being shared out equally.",
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
    # -- primary measurement, coordinates, angles and money -----------------------------------------------------
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
        "unit-not-converted",
        "Leaving a volume in cm³ when litres were asked for.",
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
