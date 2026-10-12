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
    # -- r2d: vectors, linear transformations and finance chains ---------------------------------------------------
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
        "multiplied-instead-of-divided",
        "Multiplying two quantities where one should be divided by the other.",
        area="general",
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
        "dropped-a-term",
        "Leaving one of the terms out of a sum or difference of vectors.",
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
        "coefficient-not-squared",
        "Using a vector's coefficient as it stands inside a squared magnitude instead of squaring it.",
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
    # -- r2e exam chains: finance, networks, complex numbers, statistics -----------------------------------------
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
        "adds-instead-of-multiplies",
        "Adding two quantities (lengths, or the probabilities of successive draws) where they should be multiplied.",
        area="probability",
    ),
    CatalogueEntry(
        "adds-instead-of-subtracts",
        "Adding two amounts where the difference between them was asked for: the total withdrawn plus the "
        "balance rather than minus it.",
        area="finance",
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
    # -- r2a exam chains: differentiation, tangents and kinematics ----------------------------------------------
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
        "forgot-inner-derivative",
        "Differentiating the outer function but leaving out the derivative of the inner function (or of the exponent or angle, such as the factor k from sin(kx)).",
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
    # -- r2c: multi-part calculus and vector exam chains ------------------------------------------------------------
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
    # -- r2b: calculus exam chains ----------------------------------------------------
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
    # -- exam chains (r2f) --------------------------------------------------------------------------------
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
        "odds-for-probability",
        "Giving the ratio of two probabilities (odds) as though it were a probability.",
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
