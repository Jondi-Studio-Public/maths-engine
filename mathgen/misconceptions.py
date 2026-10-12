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
    # -- r2l: probability estimates, histograms, distributions and household energy -----------------------
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
        "cents-read-as-dollars",
        "Leaving a cost in cents and reporting it as dollars.",
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
    # -- r2i: calibration, trips, closest approach, tables, rates, timetables, motion, integrals and group costs ------
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
    # -- r2i: units and times --------------------------------------------------------------------------------------
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
    # -- r2i: calibration and tables -------------------------------------------------------------------------------
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
    # -- r2i: closest approach -------------------------------------------------------------------------------------
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
    # -- r2i: rates, motion and integrals --------------------------------------------------------------------------
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
        "evaluated-function-not-derivative",
        "Substituting into the function (or the modelled quantity) itself where its derivative, the rate of change, was asked for.",
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
    # -- r2i: group costs ------------------------------------------------------------------------------------------
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
    # -- statistics and probability (family r2k) ----------------------------------------
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
        "reversed-difference",
        "Subtracting in the wrong order, so the sign of the difference is reversed.",
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
        "multiplies-instead-of-dividing",
        "Multiplying where the quantity must be divided: by the standard error in z, or one length by another.",
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
        "subtracted-in-wrong-order",
        "Subtracting the square of the mean from E(X squared) the wrong way round, giving a negative variance.",
        area="statistics",
    ),
    CatalogueEntry(
        "forgot-to-square-the-mean",
        "Subtracting the mean, not its square, from E(X squared).",
        area="statistics",
    ),
    # -- r2g: bootstrap, topology, analysis, linear algebra, PDE energy, automata and derivations ------------------
    CatalogueEntry(
        "adds-instead-of-multiplies",
        "Adding two quantities (probabilities, counts) where they should be multiplied to combine them.",
        area="university",
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
        "forgot-to-divide-by-determinant",
        "Giving the adjugate-based numerator without dividing by the determinant.",
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
        "off-by-one-count",
        "An answer one or two more or less than the right count or index.",
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
    # -- r2j: exam chains for sampling, exponential models, trigonometric rates, circle geometry, expansion,
    # shape sorting and radicals ------------------------------------------------------------------------------
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
        "added-instead-of-subtracted",
        "Adding where a quantity should be taken away: the outside segment to the whole secant, a first unknown to the sum, or two figures where a difference was asked.",
        area="general",
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
        "dropped-cross-term",
        "Leaving out one of the partial products when collecting like terms, so a coefficient is short of a term.",
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
    # r2h: exam chains (counting probabilities, regularised solves, transforms, topology, algebra, physics, logic)
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
        "subtracts-instead-of-dividing",
        "Subtracting a factor where the unknown is found by dividing by it.",
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
        "symmetric-difference-as-union",
        "Reading the symmetric difference as the union.",
        area="measure",
    ),
    CatalogueEntry(
        "symmetric-difference-as-intersection",
        "Reading the symmetric difference as the intersection.",
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
        "ignores-the-conclusion",
        "Giving the number of rows satisfying the premises where the conclusion was to be included too.",
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
