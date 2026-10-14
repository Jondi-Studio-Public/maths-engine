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
    # -- sequences ------------------------------------------------------------------------------------------------
    CatalogueEntry(
        "term-position-off-by-one",
        "Using the wrong term position: counting from zero when the first term is n = 1, or going one term too far.",
        area="sequences",
    ),
    CatalogueEntry(
        "explicit-rule-applied-to-previous-term",
        "Treating an explicit rule as if it built each term from the one before: finding the previous term, then "
        "putting that value into the rule again.",
        area="sequences",
    ),
    CatalogueEntry(
        "iteration-stopped-early",
        "Stopping the repeated rule one term short, so the answer is the previous term in the sequence.",
        area="sequences",
    ),
    CatalogueEntry(
        "wrong-order-of-operations",
        "Doing the operations in the wrong order: subtracting before multiplying, or adding before multiplying.",
        area="general",
    ),
    CatalogueEntry(
        "square-applied-to-coefficient",
        "Squaring the whole product a n instead of only n, so 3n^2 at n = 4 is read as (3 × 4)^2.",
        area="functions",
    ),
    # -- t tests, chi-square tests and induction (university first year) --------------------------------------------
    CatalogueEntry(
        "se-divides-by-n",
        "Dividing the sample standard deviation by n instead of √n to get the standard error of the mean.",
        area="inference",
    ),
    CatalogueEntry(
        "se-uses-n-minus-1",
        "Dividing the sample standard deviation by √(n - 1), the degrees of freedom, instead of √n.",
        area="inference",
    ),
    CatalogueEntry(
        "se-multiplies-by-root-n",
        "Multiplying the sample standard deviation by √n instead of dividing by it.",
        area="inference",
    ),
    CatalogueEntry(
        "variance-used-as-sd",
        "Using the sample variance where the sample standard deviation belongs, so the square root is never taken.",
        area="inference",
    ),
    CatalogueEntry(
        "sd-used-as-standard-error",
        "Using the sample standard deviation itself where the standard error s/√n belongs.",
        area="inference",
    ),
    CatalogueEntry(
        "t-sign-reversed",
        "Subtracting in the wrong order, so the t statistic (or the difference behind it) has the wrong sign.",
        area="inference",
    ),
    CatalogueEntry(
        "sample-sd-divides-by-n",
        "Dividing the sum of squared deviations by n instead of n - 1 when finding a sample standard deviation.",
        area="inference",
    ),
    CatalogueEntry(
        "ci-wrong-bound",
        "Giving the other end of the confidence interval from the one asked for.",
        kind="near_miss",
        area="inference",
    ),
    CatalogueEntry(
        "ci-margin-as-bound",
        "Giving the margin of error t*·s/√n as the limit itself, without adding it to or subtracting it from the sample mean.",
        area="inference",
    ),
    CatalogueEntry(
        "inverse-forgets-null-mean",
        "Running the t formula backwards but forgetting to add the hypothesised mean, so the difference is given as the sample mean.",
        area="inference",
    ),
    CatalogueEntry(
        "critical-value-wrong-direction",
        "Moving from the hypothesised mean the wrong way when finding the boundary of the rejection region.",
        area="inference",
    ),
    CatalogueEntry(
        "critical-value-not-scaled",
        "Adding the critical value itself to the hypothesised mean, without multiplying it by the standard error.",
        area="inference",
    ),
    CatalogueEntry(
        "critical-value-wrong-df",
        "Reading the critical value from the row for n degrees of freedom instead of n - 1.",
        area="inference",
    ),
    CatalogueEntry(
        "critical-mean-from-sample-mean",
        "Measuring the critical distance from the sample mean instead of from the hypothesised mean.",
        area="inference",
    ),
    CatalogueEntry(
        "decision-compared-wrong-way",
        "Comparing the test statistic with the critical value the wrong way round: rejecting when it is smaller, or not rejecting when it is larger.",
        area="inference",
    ),
    CatalogueEntry(
        "null-accepted-as-proved",
        "Treating a failure to reject the null hypothesis as proof that it is true.",
        area="inference",
    ),
    CatalogueEntry(
        "any-difference-is-significant",
        "Rejecting the null hypothesis because the sample value differs from it at all, without comparing with the critical value.",
        area="inference",
    ),
    CatalogueEntry(
        "chi-divides-by-observed",
        "Dividing each squared difference by the observed count instead of the expected count.",
        area="inference",
    ),
    CatalogueEntry(
        "chi-no-square",
        "Adding the differences (O - E)/E without squaring them.",
        area="inference",
    ),
    CatalogueEntry(
        "chi-no-division-by-expected",
        "Adding the squared differences (O - E)² without dividing each by its expected count.",
        area="inference",
    ),
    CatalogueEntry(
        "chi-skips-a-category",
        "Leaving one category (or cell) out of the sum for the chi-square statistic.",
        area="inference",
    ),
    CatalogueEntry(
        "expected-assumes-equal-split",
        "Sharing the total equally between the categories or cells although the hypothesis does not say they are equal.",
        area="inference",
    ),
    CatalogueEntry(
        "expected-ignores-grand-total",
        "Multiplying the row total by the column total but never dividing by the grand total.",
        area="inference",
    ),
    CatalogueEntry(
        "expected-from-one-total-only",
        "Finding an expected count from a column total (or a row total) alone, as if the other variable played no part.",
        area="inference",
    ),
    CatalogueEntry(
        "expected-probability-not-count",
        "Giving the probability of the category where the expected count was asked for.",
        kind="near_miss",
        area="inference",
    ),
    CatalogueEntry(
        "expected-complement-count",
        "Finding the expected count for everything except the category asked about.",
        area="inference",
    ),
    CatalogueEntry(
        "expected-percent-used-as-count",
        "Using the percentage (or the ratio part) itself as the expected count instead of taking that share of the total.",
        area="inference",
    ),
    CatalogueEntry(
        "expected-divides-by-categories-minus-one",
        "Dividing the total by one more or one fewer than the number of equally likely categories.",
        area="inference",
    ),
    CatalogueEntry(
        "df-cells-minus-one",
        "Using the number of cells minus one as the degrees of freedom of a table, as for a goodness-of-fit test.",
        area="inference",
    ),
    CatalogueEntry(
        "df-adds-dimensions",
        "Adding the table's dimensions, r + c - 2, instead of multiplying (r - 1)(c - 1).",
        area="inference",
    ),
    CatalogueEntry(
        "df-counts-cells",
        "Taking the number of cells or categories itself as the degrees of freedom.",
        area="inference",
    ),
    CatalogueEntry(
        "df-one-dimension-only",
        "Multiplying (r - 1) by c, forgetting to subtract one from the other dimension.",
        area="inference",
    ),
    CatalogueEntry(
        "df-ignores-estimated-parameter",
        "Using categories - 1 as the degrees of freedom although parameters were estimated from the data.",
        area="inference",
    ),
    CatalogueEntry(
        "df-subtracts-too-many",
        "Taking away too many from the number of categories when finding the degrees of freedom.",
        area="inference",
    ),
    CatalogueEntry(
        "base-case-evaluated-at-wrong-n",
        "Checking the base case at the wrong value of n (n = 0 or n = 2) instead of the first value the statement covers.",
        area="induction",
    ),
    CatalogueEntry(
        "base-case-second-term-only",
        "Giving the value of the second term where the value of the sum at n = 1 was asked for.",
        area="induction",
    ),
    CatalogueEntry(
        "base-case-drops-denominator",
        "Evaluating only the numerator of the closed form at n = 1 and forgetting to divide by its denominator.",
        area="induction",
    ),
    CatalogueEntry(
        "term-index-not-shifted",
        "Writing the kth term where the (k + 1)th term was asked for.",
        area="induction",
    ),
    CatalogueEntry(
        "term-index-over-shifted",
        "Substituting k + 2 for the index where k + 1 was needed.",
        area="induction",
    ),
    CatalogueEntry(
        "term-index-shifted-wrong-way",
        "Substituting k - 1 for the index, giving the previous term instead of the next.",
        area="induction",
    ),
    CatalogueEntry(
        "target-uses-k-not-k-plus-1",
        "Writing the closed form at n = k instead of replacing n by k + 1 for the statement P(k + 1).",
        area="induction",
    ),
    CatalogueEntry(
        "substitutes-k-plus-1-in-one-place-only",
        "Replacing n by k + 1 in some places of the formula but not all of them.",
        area="induction",
    ),
    CatalogueEntry(
        "adds-kth-term-not-next",
        "Adding the kth term to the sum of the first k terms instead of the (k + 1)th.",
        area="induction",
    ),
    CatalogueEntry(
        "adds-one-not-the-next-term",
        "Adding 1 to the sum of the first k terms instead of the next term.",
        area="induction",
    ),
    CatalogueEntry(
        "forgot-to-add-the-term",
        "Stopping at the assumed sum of k terms without adding the (k + 1)th term.",
        area="induction",
    ),
    CatalogueEntry(
        "subtracts-the-term-instead-of-adding",
        "Subtracting the (k + 1)th term from the sum of k terms instead of adding it.",
        area="induction",
    ),
    CatalogueEntry(
        "off-by-one-term-count",
        "Using the sum of one term too many or too few when evaluating the closed form.",
        area="induction",
    ),
    CatalogueEntry(
        "treats-sum-as-arithmetic-series",
        "Averaging the last term, n × (last term)/2, as if every sum were an arithmetic series.",
        area="induction",
    ),
    CatalogueEntry(
        "least-n-takes-last-failing-value",
        "Giving the largest n for which the sum is still too small instead of the least n that makes it exceed the target.",
        area="induction",
    ),
    CatalogueEntry(
        "least-n-overshoots",
        "Giving one more than the least n for which the sum exceeds the target.",
        area="induction",
    ),
    CatalogueEntry(
        "term-compared-not-sum",
        "Finding when a single term exceeds the target instead of when the sum does.",
        area="induction",
    ),
    CatalogueEntry(
        "remainder-sign-reversed",
        "Getting the sign of the leftover term wrong when rearranging f(k + 1) into a·f(k) plus a remainder.",
        area="induction",
    ),
    CatalogueEntry(
        "remainder-drops-the-constant",
        "Forgetting one of the constant terms when rearranging f(k + 1) into a·f(k) plus a remainder.",
        area="induction",
    ),
    CatalogueEntry(
        "rearrangement-uses-wrong-base",
        "Leaving the leftover term in the wrong power when rearranging a^(k+1) - b^(k+1) into a(a^k - b^k) plus a remainder.",
        area="induction",
    ),
    CatalogueEntry(
        "rearrangement-drops-the-factor",
        "Dropping the factor a - b from the leftover term in the rearrangement.",
        area="induction",
    ),
    CatalogueEntry(
        "uses-sum-not-difference-of-bases",
        "Using a + b where the rearrangement gives a - b.",
        area="induction",
    ),
    CatalogueEntry(
        "expansion-slip",
        "An arithmetic slip when expanding f(k + 1) and subtracting f(k).",
        area="induction",
    ),
    CatalogueEntry(
        "gives-f-of-k-plus-1-not-the-difference",
        "Giving f(k + 1) itself where the difference f(k + 1) - f(k) was asked for.",
        kind="near_miss",
        area="induction",
    ),
    CatalogueEntry(
        "forgets-to-divide-the-remainder",
        "Dividing a·(dq) by d but leaving the remainder undivided when writing f(k + 1) ÷ d in terms of q.",
        area="induction",
    ),
    CatalogueEntry(
        "multiplier-dropped",
        "Losing the multiplier a on q when writing f(k + 1) ÷ d in terms of q.",
        area="induction",
    ),
    CatalogueEntry(
        "remainder-dropped",
        "Dropping the remainder altogether when writing f(k + 1) ÷ d in terms of q.",
        area="induction",
    ),
    CatalogueEntry(
        "quotient-not-taken",
        "Giving f(N) itself where f(N) divided by d was asked for.",
        kind="near_miss",
        area="induction",
    ),
    # -- Laplace transforms, line integrals and triple integrals (university first year) -----------------------------
    CatalogueEntry(
        "laplace-forgot-factorial",
        "Transforming c t^n as c/s^(n+1), leaving out the n! that belongs in the numerator.",
        area="laplace",
    ),
    CatalogueEntry(
        "laplace-power-off-by-one",
        "Transforming t^n with s^n in the denominator instead of s^(n+1).",
        area="laplace",
    ),
    CatalogueEntry(
        "laplace-constant-as-number",
        "Transforming a constant c as c instead of c/s.",
        area="laplace",
    ),
    CatalogueEntry(
        "laplace-exp-sign",
        "Taking the transform of e^(at) to be 1/(s + a), or shifting s the wrong way in the first shift theorem, "
        "instead of 1/(s - a).",
        area="laplace",
    ),
    CatalogueEntry(
        "laplace-sin-cos-swap",
        "Swapping the transforms of sin(bt) and cos(bt): b/(s^2 + b^2) for cosine, s/(s^2 + b^2) for sine.",
        area="laplace",
    ),
    CatalogueEntry(
        "laplace-sin-missing-b",
        "Transforming sin(bt) as 1/(s^2 + b^2), leaving out the factor b in the numerator.",
        area="laplace",
    ),
    CatalogueEntry(
        "laplace-quad-b-not-squared",
        "Writing s^2 + b in the denominator of the sine or cosine transform instead of s^2 + b^2.",
        area="laplace",
    ),
    CatalogueEntry(
        "laplace-no-shift",
        "Ignoring the exponential factor in e^(at) sin(bt), e^(at) cos(bt) or t e^(at), and transforming the "
        "rest as if a were 0.",
        area="laplace",
    ),
    CatalogueEntry(
        "laplace-shift-not-squared",
        "Shifting s correctly but leaving the denominator of t e^(at) as (s - a), or of e^(at) sin(bt) as (s - a), "
        "instead of (s - a)^2 (+ b^2).",
        area="laplace",
    ),
    CatalogueEntry(
        "laplace-shift-numerator-only",
        "Shifting the s in the denominator of e^(at) cos(bt) but leaving the numerator as s instead of (s - a).",
        area="laplace",
    ),
    CatalogueEntry(
        "laplace-inverse-coefficient-swap",
        "Attaching the partial fraction coefficients to the wrong exponentials when inverting.",
        area="laplace",
    ),
    CatalogueEntry(
        "laplace-inverse-pole-sign",
        "Reading a pole at s = a as e^(-at), or a denominator factor (s + m) as e^(mt), when inverting.",
        area="laplace",
    ),
    CatalogueEntry(
        "laplace-inverse-coefficient-undivided",
        "Finding each partial fraction coefficient by evaluating the numerator at the pole and not dividing by "
        "the other factor there.",
        area="laplace",
    ),
    CatalogueEntry(
        "laplace-inverse-term-dropped",
        "Inverting only one of the terms of a partial fraction split.",
        area="laplace",
    ),
    CatalogueEntry(
        "laplace-inverse-numerator-split",
        "Using the coefficients of the numerator as the partial fraction coefficients, A = p and B = q.",
        area="laplace",
    ),
    CatalogueEntry(
        "laplace-repeated-root-no-t",
        "Inverting the 1/(s - a)^2 part of a repeated root as e^(at), leaving out the factor t.",
        area="laplace",
    ),
    CatalogueEntry(
        "laplace-repeated-root-numerator",
        "Using q, not q + pa, as the coefficient of t e^(at) when (ps + q) is not written in terms of (s - a).",
        area="laplace",
    ),
    CatalogueEntry(
        "laplace-square-sine-undivided",
        "After completing the square, using the whole numerator remainder as the coefficient of the sine term "
        "instead of dividing it by the frequency.",
        area="laplace",
    ),
    CatalogueEntry(
        "laplace-square-numerator-unshifted",
        "After completing the square, dividing the constant in the numerator by the frequency without first "
        "taking out the shift (q - pm) for the sine coefficient.",
        area="laplace",
    ),
    CatalogueEntry(
        "laplace-square-no-exponential",
        "Inverting (ps + q)/((s + m)^2 + w^2) as a plain sine and cosine, dropping the factor e^(-mt).",
        area="laplace",
    ),
    CatalogueEntry(
        "laplace-derivative-initial-sign",
        "Transforming y' as sY + y(0), or y'' with the initial values added, instead of subtracted.",
        area="laplace",
    ),
    CatalogueEntry(
        "laplace-initial-condition-dropped",
        "Leaving the initial value y(0) out of the transformed equation.",
        area="laplace",
    ),
    CatalogueEntry(
        "laplace-forcing-ignored",
        "Solving the transformed equation for Y(s) without the transform of the right-hand side.",
        area="laplace",
    ),
    CatalogueEntry(
        "laplace-sy0-term-dropped",
        "Transforming y'' without its s y(0) term, or leaving the b y(0) term out of the transform of the y' term.",
        area="laplace",
    ),
    CatalogueEntry(
        "laplace-residue-sign",
        "Evaluating the numerator at s = -r instead of s = r when finding a partial fraction coefficient at s = r.",
        area="laplace",
    ),
    CatalogueEntry(
        "laplace-residue-undivided",
        "Finding a partial fraction coefficient as the numerator at the pole, without dividing by the other factor.",
        area="laplace",
    ),
    CatalogueEntry(
        "laplace-residue-difference-order",
        "Dividing by the difference of the two poles in the wrong order, giving a coefficient with the wrong sign.",
        area="laplace",
    ),
    CatalogueEntry(
        "laplace-residue-other-coefficient",
        "Reporting the coefficient of the other partial fraction than the one asked for.",
        area="laplace",
    ),
    CatalogueEntry(
        "line-integral-forgot-speed",
        "Leaving out |r'(t)| when turning ds into dt, so the scalar line integral is just the integral of f(r(t)) dt.",
        area="line-integrals",
    ),
    CatalogueEntry(
        "line-integral-speed-squared",
        "Using |r'(t)|^2 in place of |r'(t)| for ds, leaving out the square root.",
        area="line-integrals",
    ),
    CatalogueEntry(
        "line-integral-endpoint-only",
        "Multiplying the length of the segment by the value of f at one end instead of integrating f along it.",
        area="line-integrals",
    ),
    CatalogueEntry(
        "line-integral-ds-as-dx",
        "Treating ds as dx, so the integral is multiplied by the horizontal extent of the segment instead of its length.",
        area="line-integrals",
    ),
    CatalogueEntry(
        "line-integral-r-for-r-prime",
        "Taking the dot product of F with r(t) instead of r'(t) when evaluating F · dr.",
        area="line-integrals",
    ),
    CatalogueEntry(
        "line-integral-forgot-dr",
        "Integrating P + Q with respect to t, dropping the x'(t) and y'(t) factors of dr.",
        area="line-integrals",
    ),
    CatalogueEntry(
        "line-integral-direction-sign",
        "Getting the sign of the work wrong by taking the path, or the potential difference, in the wrong direction: "
        "phi(A) - phi(B).",
        area="line-integrals",
    ),
    CatalogueEntry(
        "line-integral-endpoint-field",
        "Taking the work to be the field at the final point dotted with the displacement, F(B) · (B - A).",
        area="line-integrals",
    ),
    CatalogueEntry(
        "line-integral-components-swapped",
        "Pairing P with y'(t) and Q with x'(t) in F · r'(t).",
        area="line-integrals",
    ),
    CatalogueEntry(
        "line-integral-endpoints-added",
        "Adding the potential at the two endpoints, phi(B) + phi(A), instead of subtracting.",
        area="line-integrals",
    ),
    CatalogueEntry(
        "line-integral-forgot-initial-potential",
        "Giving phi(B) as the work, without subtracting phi(A).",
        area="line-integrals",
    ),
    CatalogueEntry(
        "line-integral-potential-missing-y-part",
        "Finding a potential by integrating P with respect to x and stopping, leaving out the function of y alone.",
        area="line-integrals",
    ),
    CatalogueEntry(
        "line-integral-potential-missing-x-part",
        "Finding a potential by integrating Q with respect to y and stopping, leaving out the function of x alone.",
        area="line-integrals",
    ),
    CatalogueEntry(
        "line-integral-potential-double-counted",
        "Adding the integral of P with respect to x to the integral of Q with respect to y, so every term in both "
        "is counted twice.",
        area="line-integrals",
    ),
    CatalogueEntry(
        "line-integral-potential-sign",
        "Writing the potential with the wrong sign, phi where -phi was meant by the convention F = grad(phi).",
        area="line-integrals",
    ),
    CatalogueEntry(
        "line-integral-parameter-sign",
        "Getting the sign of k wrong when equating the partial derivatives dP/dy and dQ/dx.",
        area="line-integrals",
    ),
    CatalogueEntry(
        "line-integral-parameter-derivative-slip",
        "Forgetting the factor 2 from differentiating k y^2 when equating dP/dy with dQ/dx.",
        area="line-integrals",
    ),
    CatalogueEntry(
        "line-integral-parameter-other-coefficient",
        "Reading k from the wrong coefficient of the field instead of equating dP/dy with dQ/dx.",
        area="line-integrals",
    ),
    CatalogueEntry(
        "triple-integral-sum-not-product",
        "Adding the three one-variable integrals of a separable integrand instead of multiplying them.",
        area="triple-integrals",
    ),
    CatalogueEntry(
        "triple-integral-forgot-lower-bound",
        "Evaluating each antiderivative only at its upper limit and never subtracting the value at the lower limit.",
        area="triple-integrals",
    ),
    CatalogueEntry(
        "triple-integral-power-rule-slip",
        "Raising the power when integrating but not dividing by the new power.",
        area="triple-integrals",
    ),
    CatalogueEntry(
        "triple-integral-variable-treated-constant",
        "Treating one variable as a constant: multiplying by the width of its interval instead of integrating it.",
        area="triple-integrals",
    ),
    CatalogueEntry(
        "triple-integral-substituted-not-integrated",
        "Substituting the top surface z = g(x, y) into the integrand instead of integrating with respect to z.",
        area="triple-integrals",
    ),
    CatalogueEntry(
        "triple-integral-limit-as-constant",
        "Using the height of the top surface at the far corner as a constant upper limit for z.",
        area="triple-integrals",
    ),
    CatalogueEntry(
        "triple-integral-rectangle-swapped",
        "Swapping the x and y intervals of the rectangle while keeping the formula for the top surface.",
        area="triple-integrals",
    ),
    CatalogueEntry(
        "triple-integral-height-halved",
        "Taking the height of the solid as half of the top surface, from a mistake in the inner integral.",
        area="triple-integrals",
    ),
    CatalogueEntry(
        "triple-integral-bounding-box",
        "Integrating over the bounding box of the solid instead of over the solid itself.",
        area="triple-integrals",
    ),
    CatalogueEntry(
        "triple-integral-pyramid-factor",
        "Leaving out the one third in the volume of a pyramid or tetrahedron, giving (base area) x (height).",
        area="triple-integrals",
    ),
    CatalogueEntry(
        "triple-integral-base-not-halved",
        "Using the rectangle ab, not the triangle ab/2, as the base of a tetrahedron.",
        area="triple-integrals",
    ),
    CatalogueEntry(
        "triple-integral-coefficients-as-intercepts",
        "Reading the coefficients of px + qy + rz = d as the intercepts of the plane with the axes.",
        area="triple-integrals",
    ),
    CatalogueEntry(
        "triple-integral-limit-not-dependent",
        "Using a constant limit for an inner variable where the limit depends on an outer variable.",
        area="triple-integrals",
    ),
    CatalogueEntry(
        "triple-integral-limit-missing-variable",
        "Leaving one variable out of a limit that depends on two, such as z up to c(1 - x/a) instead of "
        "c(1 - x/a - y/b).",
        area="triple-integrals",
    ),
    CatalogueEntry(
        "triple-integral-top-not-slanted",
        "Using the constant height c as the upper limit for z in a tetrahedron instead of the slanted face.",
        area="triple-integrals",
    ),
    CatalogueEntry(
        "triple-integral-limits-reversed",
        "Taking the chain of inequalities the wrong way round, so each limit is on the wrong side.",
        area="triple-integrals",
    ),
    CatalogueEntry(
        "triple-integral-symmetry-assumed",
        "Taking one sixth of the integral over the cube, as if the integrand were symmetric in x, y and z.",
        area="triple-integrals",
    ),
    CatalogueEntry(
        "triple-integral-chain-misordered",
        "Swapping two variables in the chain of inequalities, so the limits belong to the wrong variables.",
        area="triple-integrals",
    ),
    CatalogueEntry(
        "triple-integral-limits-not-rewritten",
        "Changing the order of integration but keeping the original limits attached to the new variables.",
        area="triple-integrals",
    ),
    CatalogueEntry(
        "triple-integral-density-term-dropped",
        "Taking the density as the constant term only, leaving out the variable part, when finding the mass.",
        area="triple-integrals",
    ),
    CatalogueEntry(
        "triple-integral-density-maximum",
        "Using the greatest density in the solid as if it applied everywhere, mass = (maximum density) x volume.",
        area="triple-integrals",
    ),
    CatalogueEntry(
        "triple-integral-density-midpoint",
        "Using the density halfway along its range instead of integrating the density over the solid.",
        area="triple-integrals",
    ),
    CatalogueEntry(
        "triple-integral-moment-not-divided",
        "Giving the moment of the solid, the integral of x rho dV, as the centre-of-mass coordinate without dividing "
        "by the mass.",
        area="triple-integrals",
    ),
    CatalogueEntry(
        "triple-integral-centroid-uniform",
        "Giving the centroid of the solid, as if the density were uniform, for the centre of mass.",
        area="triple-integrals",
    ),
    CatalogueEntry(
        "triple-integral-moment-inverted",
        "Dividing the wrong way round: mass over moment, or volume over mass.",
        area="triple-integrals",
    ),
    CatalogueEntry(
        "triple-integral-wrong-coordinate",
        "Finding a different coordinate of the centre of mass from the one asked for.",
        area="triple-integrals",
    ),
    CatalogueEntry(
        "triple-integral-mass-not-divided",
        "Giving the mass where the average density, mass divided by volume, was asked for.",
        area="triple-integrals",
    ),
    # -- limits, improper integrals and series (university first year) --
    CatalogueEntry(
        "calc-chain-rule-factor-dropped",
        "Differentiating e^(ax), sin(ax) or ln(1 + ax) as if the argument were x, forgetting the chain-rule factor a.",
        area="limits",
    ),
    CatalogueEntry(
        "calc-ratio-inverted",
        "Turning the ratio or parameter upside down: giving the denominator over the numerator, or the reciprocal of the value asked for.",
        area="limits",
    ),
    CatalogueEntry(
        "denominator-coefficient-dropped",
        "Differentiating the denominator but leaving out its constant factor, as if (bx)' were 1, or dropping a constant factor from the result.",
        area="limits",
    ),
    CatalogueEntry(
        "calc-sign-slip",
        "Getting the right size of answer with the wrong sign: a minus sign lost or gained while differentiating, integrating or substituting.",
        area="limits",
    ),
    CatalogueEntry(
        "calc-power-rule-exponent-dropped",
        "Applying the power rule without bringing the exponent down (or dividing by the new exponent), so a factor such as the 2 in (x^2)' is lost.",
        area="limits",
    ),
    CatalogueEntry(
        "stops-after-one-application",
        "Applying L'Hôpital's rule once, meeting 0/0 or ∞/∞ again, and stopping there with 0 or a partial value instead of applying the rule again.",
        area="limits",
    ),
    CatalogueEntry(
        "quotient-rule-instead-of-lhopital",
        "Differentiating the whole quotient with the quotient rule instead of the numerator and denominator separately, and concluding that the limit is 0.",
        area="limits",
    ),
    CatalogueEntry(
        "dominant-term-misjudged",
        "Picking the wrong term as the one that dominates at infinity, so the ratio of the wrong terms is reported.",
        area="limits",
    ),
    CatalogueEntry(
        "log-exponent-ignored",
        "Treating a ratio of two logarithms of polynomials as tending to 1 because both grow like a logarithm, ignoring the powers inside them.",
        area="limits",
    ),
    CatalogueEntry(
        "leading-coefficients-kept",
        "Keeping the leading coefficients in a ratio of logarithms, where they only add constants that vanish in the limit.",
        area="limits",
    ),
    CatalogueEntry(
        "zero-times-infinity-is-zero",
        "Treating 0 × ∞ as 0 instead of rewriting the product as a quotient and applying L'Hôpital's rule.",
        area="limits",
    ),
    CatalogueEntry(
        "calc-constant-factor-dropped",
        "Dropping a constant factor that multiplies the limit, or one that should have come out of the working.",
        area="limits",
    ),
    CatalogueEntry(
        "one-term-of-difference-only",
        "Using only one of the two terms of a difference, so the other term's contribution to the limit is lost.",
        area="limits",
    ),
    CatalogueEntry(
        "infinity-minus-infinity-is-zero",
        "Treating ∞ - ∞ as 0 instead of combining the two terms into one quotient first.",
        area="limits",
    ),
    CatalogueEntry(
        "one-to-infinity-is-one",
        "Treating a 1^∞ form as 1 instead of taking logarithms to find its real value.",
        area="limits",
    ),
    CatalogueEntry(
        "forgets-to-exponentiate-log-limit",
        "Finding the limit of ln L correctly but reporting it as the limit L, without raising e to that power.",
        area="limits",
    ),
    CatalogueEntry(
        "exponent-factor-dropped",
        "Losing a constant factor in the exponent while taking the logarithm of a 1^∞ form, so the power of e is wrong.",
        area="limits",
    ),
    CatalogueEntry(
        "linear-term-ignored",
        "Ignoring a linear term next to e^(ax) near 0, so its contribution to the exponent of e is lost.",
        area="limits",
    ),
    CatalogueEntry(
        "sum-product-confusion",
        "Multiplying two constants where they should be added.",
        area="limits",
    ),
    CatalogueEntry(
        "any-constant-works-at-zero-over-zero",
        "Saying any constant will do because the numerator is 0 at x = 0, without requiring the first derivative to vanish as well.",
        area="limits",
    ),
    CatalogueEntry(
        "matches-wrong-order-coefficient",
        "Choosing the constant to match the next-order coefficient of the numerator instead of the one that cancels the lower-order term.",
        area="limits",
    ),
    CatalogueEntry(
        "factorial-factor-dropped",
        "Losing the factor 6 (from 3 × 2) that three applications of the rule on x^3 produce, so the limit is too big.",
        area="limits",
    ),
    CatalogueEntry(
        "scaled-the-wrong-way",
        "Multiplying by a constant that the limit or integral should be divided by, or dividing where it should multiply.",
        area="limits",
    ),
    CatalogueEntry(
        "divides-by-p-not-p-minus-1",
        "Integrating c/x^p to get c x^(1-p)/p, dividing by p instead of by the new exponent 1 - p (or p - 1).",
        area="integration",
    ),
    CatalogueEntry(
        "power-rule-divisor-dropped",
        "Integrating a power of x by raising the exponent but not dividing by the new exponent.",
        area="integration",
    ),
    CatalogueEntry(
        "lower-limit-power-sign",
        "Evaluating a power at a lower limit a with the exponent's sign wrong, so a^(p-1) is used where 1/a^(p-1) belongs (or the reverse).",
        area="integration",
    ),
    CatalogueEntry(
        "chain-rule-divisor-dropped",
        "Integrating e^(kx) or e^(-x/k) without dividing by the inner coefficient, so the factor 1/k (or k) is lost.",
        area="integration",
    ),
    CatalogueEntry(
        "forgets-upper-limit-factor",
        "Evaluating an antiderivative of c/x^p at the upper limit b without the factor b^(1-p), keeping only the coefficient.",
        area="integration",
    ),
    CatalogueEntry(
        "power-rule-exponent-sign",
        "Adding 1 to the exponent of x^(-p) as if it were x^p, so the new exponent is 1 + p instead of 1 - p.",
        area="integration",
    ),
    CatalogueEntry(
        "log-quotient-inverted",
        "Writing the logarithm of the ratio of the two limits upside down, so the answer has the wrong sign.",
        area="integration",
    ),
    CatalogueEntry(
        "partial-fraction-coefficient-dropped",
        "Splitting c/((x + r)(x + s)) into partial fractions but losing the factor c/(s - r) from the coefficients.",
        area="integration",
    ),
    CatalogueEntry(
        "forgets-lower-limit-term",
        "Evaluating an antiderivative at the upper limit (or the limit at infinity) and forgetting to subtract its value at the lower limit.",
        area="integration",
    ),
    CatalogueEntry(
        "partial-fraction-term-dropped",
        "Evaluating only part of the logarithm that a partial-fraction antiderivative gives at the lower limit.",
        area="integration",
    ),
    CatalogueEntry(
        "substitution-factor-dropped",
        "Substituting u = kx^m but forgetting that du carries the factor mk, so the constant in front of the integral is wrong.",
        area="integration",
    ),
    CatalogueEntry(
        "sqrt-antiderivative-factor-dropped",
        "Integrating 1/√(a - x) to get a multiple of √(a - x) but losing the factor 2 from the power rule.",
        area="integration",
    ),
    CatalogueEntry(
        "symmetry-halving-error",
        "Treating an integral over the whole real line as if it were over half of it, or the reverse, so the answer is out by a factor of 2.",
        area="integration",
    ),
    CatalogueEntry(
        "arctan-limit-misread",
        "Misreading the limit of arctan at infinity (using π instead of π/2, or reporting the coefficient alone).",
        area="integration",
    ),
    CatalogueEntry(
        "arctan-scale-factor-dropped",
        "Using arctan(x/k) as an antiderivative of 1/(x² + k²) without the factor 1/k, so the answer is out by a factor of k.",
        area="integration",
    ),
    CatalogueEntry(
        "arctan-scale-factor-squared",
        "Dividing by k² (the constant in the denominator) instead of k when integrating 1/(x² + k²).",
        area="integration",
    ),
    CatalogueEntry(
        "arcsin-value-misread",
        "Misreading arcsin(1) as π, π/4 or 1 instead of π/2.",
        area="integration",
    ),
    CatalogueEntry(
        "arcsin-scale-factor-misapplied",
        "Dividing by the endpoint b when integrating 1/√(b² - x²), as if arcsin(x/b) needed an extra factor 1/b.",
        area="integration",
    ),
    CatalogueEntry(
        "by-parts-second-integral-factor-dropped",
        "Finishing integration by parts but leaving out the factor from the second integral, so one power of 1/k (or 1/(p - 1)) is missing.",
        area="integration",
    ),
    CatalogueEntry(
        "by-parts-extra-divisor",
        "Dividing by the inner coefficient or exponent once too often while integrating by parts.",
        area="integration",
    ),
    CatalogueEntry(
        "by-parts-extra-factor",
        "Multiplying by a factor of 2 from a power rule that does not apply to this integral by parts.",
        area="integration",
    ),
    CatalogueEntry(
        "solves-without-root",
        "Solving k² = a (or (p - 1)² = a) and reporting a, forgetting to take the square root.",
        area="integration",
    ),
    CatalogueEntry(
        "solves-p-minus-one-only",
        "Solving for p - 1 (or for p - 1 squared) and reporting that instead of p.",
        area="integration",
    ),
    CatalogueEntry(
        "pdf-normalisation-skipped",
        "Taking the constant in a probability density function to be 1, or reading it off the formula, instead of choosing it so that the total probability is 1.",
        area="integration",
    ),
    CatalogueEntry(
        "tail-cdf-confusion",
        "Finding P(X ≤ t) when P(X > t) was asked, or the reverse, so the answer is 1 minus the right one.",
        area="integration",
    ),
    CatalogueEntry(
        "rate-dropped-from-exponent",
        "Evaluating a tail probability for an exponential density with the rate left out of the exponent.",
        area="integration",
    ),
    CatalogueEntry(
        "mean-and-variance-confused",
        "Reporting the square of the mean (or the mean) where the variance was asked, or the reverse.",
        area="integration",
    ),
    CatalogueEntry(
        "variance-second-moment-not-centred",
        "Reporting E(X²) as the variance, without subtracting the square of the mean.",
        area="integration",
    ),
    CatalogueEntry(
        "variance-subtracts-mean-not-mean-squared",
        "Computing the variance as E(X²) - E(X) instead of E(X²) - E(X)².",
        area="integration",
    ),
    CatalogueEntry(
        "standard-deviation-for-variance",
        "Reporting the mean, or the square root of the variance, where the variance was asked.",
        area="integration",
    ),
    CatalogueEntry(
        "by-parts-boundary-term-dropped",
        "Dropping the boundary term in integration by parts, so only the remaining integral is kept.",
        area="integration",
    ),
    CatalogueEntry(
        "power-law-exponent-off-by-one",
        "Being one out in the exponent of a power law: using x^(-p) in place of x^(1-p) for a tail, or p - 1 in place of p - 2 for a mean.",
        area="integration",
    ),
    CatalogueEntry(
        "lower-bound-factor-dropped",
        "Forgetting the power of the lower bound a that a power-law density carries, so a scale factor is missing.",
        area="integration",
    ),
    CatalogueEntry(
        "square-root-of-bound-dropped",
        "Using b where √b belongs (or the reverse) when evaluating a square-root antiderivative at the endpoint.",
        area="integration",
    ),
    CatalogueEntry(
        "volume-pi-dropped",
        "Leaving the factor π (or 2π) out of a volume of revolution.",
        area="integration",
    ),
    CatalogueEntry(
        "disc-radius-not-squared",
        "Using the disc method with the radius instead of the radius squared, so the integrand is c/x^p rather than (c/x^p)².",
        area="integration",
    ),
    CatalogueEntry(
        "shell-radius-factor-omitted",
        "Using the shell method without the factor x for the shell's radius, so the integrand is c/x^p instead of x × c/x^p.",
        area="integration",
    ),
    CatalogueEntry(
        "rotated-about-wrong-axis",
        "Giving the volume for rotation about the other axis.",
        area="integration",
    ),
    CatalogueEntry(
        "geometric-sum-formula-start-mismatch",
        "Using the sum formula for a geometric series that starts at n = 0 (a/(1 - r)) or at n = 1 (ar/(1 - r)) whatever the actual start index is.",
        area="series",
    ),
    CatalogueEntry(
        "geometric-exponent-shift-missed",
        "Reading the first term of a r^(n-1) as a r^n at the start index, so the exponent shift is ignored.",
        area="series",
    ),
    CatalogueEntry(
        "geometric-first-term-skipped",
        "Starting the geometric sum one term late, or removing the wrong term when the start index changes.",
        area="series",
    ),
    CatalogueEntry(
        "geometric-sum-first-term-only",
        "Reporting the first term of a convergent geometric series as its sum.",
        area="series",
    ),
    CatalogueEntry(
        "ratio-test-exponential-factor-ignored",
        "Taking the ratio of the polynomial parts only, so L = 1, and ignoring the factor c^n/d^n that makes the limit c/d.",
        area="series",
    ),
    CatalogueEntry(
        "ratio-test-limit-of-term",
        "Giving the limit of the terms a_n (often 0) instead of the limit of the ratio a(n+1)/a(n).",
        area="series",
    ),
    CatalogueEntry(
        "factorial-ratio-misjudged",
        "Cancelling factorials wrongly in a ratio, such as treating (2n + 2)!/(2n)! as 2 or ignoring a factorial altogether.",
        area="series",
    ),
    CatalogueEntry(
        "ratio-limit-assumed-one",
        "Assuming the ratio of consecutive terms tends to 1 without working it out.",
        area="series",
    ),
    CatalogueEntry(
        "telescoping-tail-off-by-one",
        "Keeping the wrong terms after the cancellation: starting the leftover terms one place too early or too late.",
        area="series",
    ),
    CatalogueEntry(
        "telescoping-gap-factor-dropped",
        "Splitting c/((n + r)(n + r + g)) into 1/(n + r) - 1/(n + r + g) without the factor c/g.",
        area="series",
    ),
    CatalogueEntry(
        "telescoping-leftover-terms-miscounted",
        "Keeping only one leftover term after the cancellation when the gap is larger than 1, so some of the surviving terms are missed.",
        area="series",
    ),
    CatalogueEntry(
        "telescoping-difference-added",
        "Adding the two partial-fraction terms at the ends instead of subtracting them.",
        area="series",
    ),
    CatalogueEntry(
        "telescoping-gap-factor-applied-twice",
        "Applying the 1/2 from a partial-fraction split twice.",
        area="series",
    ),
    CatalogueEntry(
        "telescoping-everything-cancels",
        "Concluding that a telescoping sum is 0 because the terms cancel, ignoring the terms at the start that survive.",
        area="series",
    ),
    CatalogueEntry(
        "radius-ratio-not-inverted",
        "Reporting L = lim |a(n+1)/a(n)| as the radius of convergence instead of 1/L.",
        area="series",
    ),
    CatalogueEntry(
        "radius-assumed-one",
        "Giving a radius of convergence of 1 without working out the limit of the ratio.",
        area="series",
    ),
    CatalogueEntry(
        "radius-x-squared-not-rooted",
        "For a series in x^(2n), giving the bound on x^2 as the radius instead of its square root.",
        area="series",
    ),
    CatalogueEntry(
        "radius-centre-added",
        "Adding the centre of the series to the radius of convergence.",
        area="series",
    ),
    CatalogueEntry(
        "radius-centre-confused",
        "Giving the centre of a series in (x - a) as its radius of convergence.",
        area="series",
    ),
    CatalogueEntry(
        "radius-polynomial-factor-multiplied",
        "Letting a polynomial factor n^m change the radius, multiplying it by m.",
        area="series",
    ),
    CatalogueEntry(
        "geometric-ratio-constant-term-dropped",
        "Forgetting the constant term when evaluating the common ratio (bx + c)/d at a given x.",
        area="series",
    ),
    CatalogueEntry(
        "geometric-ratio-coefficient-dropped",
        "Forgetting the coefficient of x when evaluating the common ratio (bx + c)/d at a given x.",
        area="series",
    ),
    CatalogueEntry(
        "geometric-ratio-set-equal-to-sum",
        "Setting the common ratio equal to the stated sum instead of solving 1/(1 - r) equal to it.",
        area="series",
    ),
    CatalogueEntry(
        "least-n-off-by-one",
        "Giving the largest n that fails the inequality, or one more than the least n that satisfies it.",
        area="series",
    ),
    CatalogueEntry(
        "series-shift-ignored",
        "Leaving the constant shift r out of the leftover terms 1/(n + r + i) when solving for n.",
        area="series",
    ),
    # -- eigenvectors, diagonalisation, null space and power series (university first year) ---------------------------
    CatalogueEntry(
        "eigvec-other-eigenvalue",
        "Finding the eigenvector for a different eigenvalue from the one asked for.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "eigvec-component-sign-slip",
        "Dropping a minus sign while solving for the eigenvector, so one component has the wrong sign: (q, p) for (-q, p).",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "eigvec-row-as-vector",
        "Writing a row of A - λI as the eigenvector instead of solving (A - λI)v = 0, whose solutions are perpendicular "
        "to the rows.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "eigvec-eigenvalue-sign",
        "Solving (A + λI)v = 0 instead of (A - λI)v = 0, so the eigenvalue is used with the wrong sign.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "eigvec-components-swapped",
        "Swapping two components of the eigenvector when reading it off the reduced system.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "eigvec-unit-vector-triangular",
        "Taking the standard basis vector in the position of the diagonal entry as the eigenvector of a triangular or "
        "block matrix, without solving for the entries above it.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "eigvec-padded-block",
        "Padding the eigenvector of a 2×2 block with 1 instead of 0 for the rest of a block diagonal matrix.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "eigvec-generalised",
        "Giving a generalised eigenvector, a solution of (A - λI)w = v, instead of an eigenvector.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "eigval-other-root",
        "Giving the other eigenvalue when the larger or smaller one was asked for.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "eigval-trace-as-eigenvalue",
        "Reporting the trace of the matrix as an eigenvalue.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "eigval-trace-difference-sign",
        "Adding the known eigenvalue to the trace instead of subtracting it to find the other eigenvalue.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "eigval-determinant-as-eigenvalue",
        "Reporting the determinant of the matrix as an eigenvalue.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "eigval-diagonal-entry",
        "Reading a diagonal entry of a matrix that is not triangular as an eigenvalue.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "eigval-repeated-given",
        "Repeating the eigenvalue given in the question as the other eigenvalue.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "diag-conjugation-reversed",
        "Multiplying in the order P⁻¹DP instead of PDP⁻¹ (or the same reversal for P⁻¹AP).",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "diag-forgot-inverse",
        "Stopping at PD, leaving out the final multiplication by P⁻¹.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "diag-transpose-for-inverse",
        "Using the transpose of P in place of its inverse.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "diag-eigenvalues-swapped",
        "Writing the eigenvalues down the diagonal of D in the opposite order from the columns of P.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "diag-columns-wrong-order",
        "Putting the eigenvectors into the columns of P in the opposite order from the eigenvalues down D.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "diag-vectors-as-rows",
        "Writing the eigenvectors as the rows of P instead of its columns.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "diag-forgot-change-of-basis",
        "Giving Dⁿ as the answer for Aⁿ, leaving out the P and P⁻¹ on either side.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "matrix-power-entrywise",
        "Raising each entry of a matrix to the power n instead of multiplying the matrix by itself n times.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "matrix-power-off-by-one",
        "Using Dⁿ⁻¹ (or Dⁿ⁺¹) in P Dⁿ P⁻¹, so the result is A to the wrong power.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "matrix-power-as-multiple",
        "Treating Aⁿ as n times A.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "inverse-adjugate-undivided",
        "Swapping the diagonal and negating the off-diagonal of a 2×2 matrix but not dividing by the determinant.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "inverse-sign-pattern",
        "Swapping the diagonal of a 2×2 matrix but leaving the off-diagonal entries unchanged, or negating the wrong pair.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "inverse-entrywise-reciprocal",
        "Inverting a matrix by taking the reciprocal of each entry.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "rank-counts-nonzero-rows",
        "Taking the rank to be the number of non-zero rows of the original matrix, without noticing that one is a "
        "combination of the others.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "rank-nullity-confused",
        "Giving the rank where the nullity was asked for, or the nullity where the rank was asked for.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "nullity-uses-rows",
        "Subtracting the rank from the number of rows instead of the number of columns to get the nullity.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "nullity-counts-columns",
        "Taking the nullity to be the number of columns, forgetting to subtract the rank.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "rank-off-by-one",
        "Miscounting the pivots during row reduction, so the rank is one too high or one too low.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "nullvec-row-as-vector",
        "Writing a row of the matrix as a null-space vector instead of a solution of Mx = 0.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "nullvec-column-as-vector",
        "Writing a column of the matrix as a null-space vector.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "nullvec-component-sign-slip",
        "Dropping a minus sign when reading the null-space vector off the reduced matrix.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "nullvec-left-null",
        "Finding the null space of the transpose (vectors y with yᵀM = 0) instead of the null space of M.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "colspace-forgot-multiplier",
        "Solving the consistency condition y·b = 0 without dividing by the coefficient of k.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "colspace-null-vector-used",
        "Testing whether b is in the column space by making it perpendicular to the null space of M instead of the "
        "null space of Mᵀ.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "det-sarrus-sign-slip",
        "Getting the sign of one of the six products wrong when expanding a 3×3 determinant.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "det-sarrus-term-dropped",
        "Leaving out one of the six products when expanding a 3×3 determinant.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "series-radius-reciprocal",
        "Giving the reciprocal of the radius of convergence, 1/R, as R (the ratio test limit read as the radius).",
        area="series",
    ),
    CatalogueEntry(
        "series-radius-as-endpoint",
        "Giving the right-hand endpoint a + R of the interval as the radius of convergence R.",
        area="series",
    ),
    CatalogueEntry(
        "series-radius-as-width",
        "Giving the length of the interval of convergence, 2R, as the radius R.",
        area="series",
    ),
    CatalogueEntry(
        "series-radius-ignores-coefficient",
        "Ignoring the coefficient of x inside the bracket, so the radius is not divided by it.",
        area="series",
    ),
    CatalogueEntry(
        "series-interval-centred-at-zero",
        "Centring the interval of convergence at 0 when the series is a power of (x - a) with a not 0.",
        area="series",
    ),
    CatalogueEntry(
        "series-centre-sign-error",
        "Taking the centre of the series (x - a)ⁿ to be -a instead of a.",
        area="series",
    ),
    CatalogueEntry(
        "series-endpoints-both-included",
        "Including both endpoints in the interval of convergence without testing them.",
        area="series",
    ),
    CatalogueEntry(
        "series-endpoints-both-excluded",
        "Excluding both endpoints without testing them, or after testing only one of them.",
        area="series",
    ),
    CatalogueEntry(
        "series-endpoints-swapped",
        "Including the wrong endpoint, because the alternating and the harmonic behaviour were attached to the wrong ends.",
        area="series",
    ),
    CatalogueEntry(
        "series-geometric-sum-sign",
        "Getting the sign wrong in the common ratio of a geometric series, so the sum has the wrong denominator.",
        area="series",
    ),
    CatalogueEntry(
        "series-geometric-first-term-only",
        "Giving the first term of a geometric series instead of its sum.",
        area="series",
    ),
    CatalogueEntry(
        "series-geometric-factor-dropped",
        "Summing a geometric series without the factor of its first term (a/(1 - r) taken as 1/(1 - r)).",
        area="series",
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
        "wrong-column-replaced",
        "Replacing the wrong column of the coefficient matrix with the constants in Cramer's rule.",
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
        "power-rule-integration-divisor-omitted",
        "Integrating x^n as x^(n+1) without dividing by n + 1.",
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
        "counted-terms-inclusively-for-gap",
        "Counting both end terms when finding the number of steps between two terms, so the gap is one too many.",
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
    CatalogueEntry(
        "compares-gradients-only",
        "Comparing two linear relationships at a given x using only their gradients, ignoring where each one starts.",
        area="linear",
    ),
    CatalogueEntry(
        "compares-starting-values-only",
        "Comparing two relationships at x = 0 (their starting values) instead of at the x that was asked about.",
        area="linear",
    ),
    CatalogueEntry(
        "forgot-to-divide-by-gradient-difference",
        "Finding where two lines meet by subtracting their y-intercepts but not dividing by the difference in gradients.",
        area="linear",
    ),
    CatalogueEntry(
        "intercept-read-with-wrong-sign",
        "Reading the y-intercept off the graph with the wrong sign, for example writing +3 for a line that crosses "
        "the y-axis at -3.",
        area="linear",
    ),
    CatalogueEntry(
        "reads-one-gridline-out",
        "Reading a value one gridline or tick to the left or right of where it really is.",
        area="graphs",
    ),
    CatalogueEntry(
        "reads-where-a-line-meets-the-y-axis",
        "Giving the point where one line meets the y-axis as the point where the two lines cross.",
        area="graphs",
    ),
    CatalogueEntry(
        "vertex-form-horizontal-sign",
        "Writing a(x + h)^2 + k for a turning point at x = h, so the horizontal shift has the wrong sign.",
        area="quadratics",
    ),
    CatalogueEntry(
        "vertex-form-vertical-sign",
        "Writing the turning point's height with the wrong sign in a(x - h)^2 + k.",
        area="quadratics",
    ),
    CatalogueEntry(
        "vertex-form-leading-coefficient-lost",
        "Writing a vertex-form rule with a leading coefficient of 1 or -1 instead of finding how steeply the curve "
        "opens from a second point on it.",
        area="quadratics",
    ),
    CatalogueEntry(
        "compares-the-x-coordinates-of-the-turning-points",
        "Comparing where the turning points are along the x-axis instead of how high or low they are.",
        area="quadratics",
    ),
    CatalogueEntry(
        "sum-of-run-not-halved",
        "Finding the sum of a run of consecutive integers as the count times the sum of the first and last, without "
        "halving.",
        area="inequalities",
    ),
    CatalogueEntry(
        "line-gradient-sign-reversed",
        "Reading the size of a gradient correctly but giving it the wrong sign, calling a falling line rising or the reverse.",
        area="linear-graphs",
    ),
    CatalogueEntry(
        "line-intercept-sign-reversed",
        "Reading the size of a y-intercept correctly but giving it the wrong sign.",
        area="linear-graphs",
    ),
    CatalogueEntry(
        "intercept-left-out",
        "Writing a line as y = mx and leaving its intercept out.",
        area="linear-graphs",
    ),
    CatalogueEntry(
        "adds-an-unstated-intercept",
        "Adding a constant to a proportional relation, although its line passes through the origin.",
        area="linear-graphs",
    ),
    CatalogueEntry(
        "gradient-read-in-grid-squares",
        "Counting rise and run in grid squares and ignoring the scale on each axis.",
        area="linear-graphs",
    ),
    CatalogueEntry(
        "added-gradient-to-x",
        "Adding the gradient to x instead of multiplying x by it when finding y on a line through the origin.",
        area="linear-graphs",
    ),
    CatalogueEntry(
        "y-value-given-as-x",
        "Giving the y-value of a point on the line where its x-value was asked for.",
        area="linear-graphs",
    ),
    CatalogueEntry(
        "solved-for-x-by-multiplying",
        "Multiplying by the gradient instead of dividing by it when finding x from a given y.",
        area="linear-graphs",
    ),
    CatalogueEntry(
        "inverse-prediction-ignores-intercept",
        "Finding x from a given y by dividing y by the gradient, without first taking away the intercept.",
        area="linear-graphs",
    ),
    CatalogueEntry(
        "prediction-rise-added-once",
        "Continuing a line by adding the rise once, however many steps along it the point is.",
        area="linear-graphs",
    ),
    CatalogueEntry(
        "x-intercept-sign-reversed",
        "Finding where a line meets the x-axis as c divided by m instead of minus c divided by m.",
        area="linear-graphs",
    ),
    CatalogueEntry(
        "x-intercept-ratio-inverted",
        "Finding where a line meets the x-axis as minus m divided by c instead of minus c divided by m.",
        area="linear-graphs",
    ),
    CatalogueEntry(
        "parallel-line-keeps-old-intercept",
        "Writing the parallel line with the first line's intercept, so it does not pass through the given point.",
        area="linear-graphs",
    ),
    CatalogueEntry(
        "parallel-gap-intercepts-added",
        "Adding the two y-intercepts where the vertical gap between two parallel lines was asked for.",
        area="linear-graphs",
    ),
    CatalogueEntry(
        "parallel-gap-horizontal",
        "Giving the horizontal distance between two parallel lines where the vertical gap was asked for.",
        area="linear-graphs",
    ),
    CatalogueEntry(
        "midpoint-sum-not-halved",
        "Adding the coordinates of the two ends of a segment and not halving the sums.",
        area="linear-graphs",
    ),
    CatalogueEntry(
        "midpoint-half-the-difference",
        "Halving the differences of the coordinates of the ends of a segment instead of halving their sums.",
        area="linear-graphs",
    ),
    CatalogueEntry(
        "distance-squared-not-rooted",
        "Adding the squares of the rise and the run and stopping, without taking the square root.",
        area="linear-graphs",
    ),
    CatalogueEntry(
        "ignores-the-horizontal-component",
        "Finding the length of a segment from its rise alone, leaving out its run.",
        area="linear-graphs",
    ),
    CatalogueEntry(
        "hist-neighbouring-bar",
        "Reading the bar beside the class the question names.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "hist-bar-to-nearest-gridline",
        "Reading a bar that ends on a half line as the labelled gridline just above or below it.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "hist-boundary-for-frequency",
        "Giving a class boundary from the horizontal axis where a frequency was asked for.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "hist-classes-counted-for-total",
        "Giving the number of classes for the total number of observations.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "hist-tallest-bar-for-total",
        "Giving the height of the tallest bar for the total number of observations.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "hist-total-misses-a-bar",
        "Adding every bar but one when finding the total.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "hist-total-counts-a-bar-twice",
        "Counting one bar twice when adding the frequencies for the total.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "hist-upper-boundary-for-lower",
        "Giving the upper boundary of the class where its lower boundary was asked for.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "hist-modal-class-for-median-class",
        "Taking the class with the highest frequency to be the class that contains the median.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "hist-middle-class-for-median-class",
        "Taking the middle class on the axis, not the class holding the middle value, as the median class.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "hist-frequency-for-boundary",
        "Giving the frequency of a class where its boundary was asked for.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "hist-count-for-percentage",
        "Giving the number of values where a percentage of the total was asked for.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "hist-complement-percentage",
        "Giving the percentage outside the stated range instead of the percentage inside it.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "hist-range-leaves-out-class",
        "Leaving out one of the classes in the stated range when adding frequencies.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "hist-range-adds-next-class",
        "Adding the class that starts at the upper boundary of the stated range, which less than excludes.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "box-iqr-as-range",
        "Giving the range, whisker end to whisker end, where the interquartile range was asked for.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "box-iqr-from-median",
        "Giving the distance from the median to a quartile for the interquartile range.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "box-quartile-for-iqr",
        "Giving a quartile on its own for the interquartile range.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "box-quarter-between-quartiles",
        "Taking the box to hold a quarter of the values rather than half.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "box-three-quarters-between-quartiles",
        "Counting the values below the upper quartile, three quarters of them, as those between the quartiles.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "box-all-values-between-quartiles",
        "Taking the whole data set to lie between the quartiles.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "box-fence-without-factor",
        "Finding an outlier fence from one IQR instead of 1.5 times the IQR.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "box-fence-omits-quartile",
        "Giving 1.5 times the IQR as the fence, without adding it to the quartile.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "box-fence-wrong-side",
        "Subtracting 1.5 times the IQR from the lower quartile to find the upper fence.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "box-fence-from-median",
        "Adding 1.5 times the IQR to the median instead of the upper quartile to find the upper fence.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "box-outlier-beyond-whisker",
        "Calling any value beyond the end of the whisker an outlier, without testing it against the 1.5 x IQR fence.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "box-outlier-fence-miscalculated",
        "Judging a value to be beyond the 1.5 x IQR fence when it is inside it.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "box-outlier-fence-too-generous",
        "Judging a value to be inside the 1.5 x IQR fence when it is beyond it.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "box-median-by-whisker-end",
        "Deciding which group has the higher median from the ends of the whiskers instead of the median lines.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "box-spread-by-range",
        "Deciding which group is more spread out from the whole length of the plot instead of the interquartile range.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "box-conclusion-both-reversed",
        "Reading both comparisons, the centres and the spreads, the wrong way round.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "box-iqr-difference-summed",
        "Adding the two interquartile ranges where their difference was asked for.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "box-iqr-difference-of-medians",
        "Giving the difference between the medians where the difference between the interquartile ranges was asked for.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "box-iqr-difference-of-ranges",
        "Giving the difference between the ranges where the difference between the interquartile ranges was asked for.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "box-iqr-difference-of-upper-quartiles",
        "Giving the difference between the upper quartiles where the difference between the interquartile ranges was asked for.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "time-pm-not-converted",
        "Writing a pm hour in 24-hour time without adding 12.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "time-hour-off-by-one",
        "Converting the hour to 24-hour time but writing the hour before or after it.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "time-base-ten-clock",
        "Treating the minutes as hundredths of an hour when counting on to the next hour, so 35 minutes leaves 65.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "time-no-minutes-carry",
        "Counting whole hours to the next hour without allowing for the minutes already past the hour.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "time-minutes-not-complemented",
        "Using the minutes past the hour as the minutes left in the hour.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "time-gap-added-to-duration",
        "Adding the time left in the day to the length of the trip, where it should be taken away.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "time-midnight-ignored",
        "Using the length of the trip as the time after midnight, without taking off the time left before midnight.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "time-counted-back-from-midnight",
        "Measuring back from midnight to the arrival instead of forward from it.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "time-am-pm-swapped",
        "Giving the arrival with am where it is pm, or the reverse.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "time-day-not-advanced",
        "Leaving the arrival on the day of departure when the trip crosses midnight.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "time-clock-hour-off-by-one",
        "Reading the arrival hour one hour early or late.",
        area="statistics and time",
    ),
    CatalogueEntry(
        "net-face-count-ignored",
        "Naming a solid with the wrong number of faces: the faces of the net were not counted.",
        area="spatial",
    ),
    CatalogueEntry(
        "net-prism-pyramid-confused",
        "Mixing up a prism and a pyramid: a prism has two matching ends and rectangle sides, a pyramid has one base and triangle sides.",
        area="spatial",
    ),
    CatalogueEntry(
        "net-face-shape-only",
        "Choosing a solid because one face matches a shape in the net, without checking the other faces.",
        area="spatial",
    ),
    CatalogueEntry(
        "net-any-six-squares",
        "Believing any six squares joined edge to edge fold into a cube.",
        area="spatial",
    ),
    CatalogueEntry(
        "net-overlap-seen-in-valid-net",
        "Thinking two squares would land on the same face when a correct cube net is folded.",
        area="spatial",
    ),
    CatalogueEntry(
        "net-faces-miscounted",
        "Counting the squares of a net wrongly, so the reason given for the answer is about the number of faces when the count is right (or the reverse).",
        area="spatial",
    ),
    CatalogueEntry(
        "net-neighbours-meet",
        "Thinking the corners next to a corner in the net are the ones that meet it when the net is folded.",
        area="spatial",
    ),
    CatalogueEntry(
        "net-corner-half-right",
        "Finding one corner that meets the given corner and guessing the other.",
        area="spatial",
    ),
    CatalogueEntry(
        "net-two-faces-for-one",
        "Doubling a face's area because the box has two faces of that size, when only one face was asked for.",
        area="measurement",
    ),
    CatalogueEntry(
        "net-face-sides-added",
        "Adding a face's length and height once instead of multiplying them to get its area.",
        area="measurement",
    ),
    CatalogueEntry(
        "net-wrong-corner",
        "Naming the corners that meet at a different corner of the cube from the one asked about.",
        area="spatial",
    ),
    CatalogueEntry(
        "net-strip-not-halved",
        "Using the whole length of the strip of side faces as one length and one width, forgetting it goes round the box once and holds two of each.",
        area="spatial",
    ),
    CatalogueEntry(
        "net-strip-length-not-removed",
        "Halving the length of the strip of side faces but not taking away the length that is already known.",
        area="spatial",
    ),
    CatalogueEntry(
        "net-strip-as-equal-faces",
        "Treating the four side faces in the strip as four equal faces and dividing its length by four.",
        area="spatial",
    ),
    CatalogueEntry(
        "net-one-of-each-face",
        "Adding the areas of three different faces of a box and forgetting that each appears twice.",
        area="spatial",
    ),
    CatalogueEntry(
        "net-sides-counted-once",
        "Counting the four side faces of a box once each as if the box had one end face and one lid-sized strip.",
        area="spatial",
    ),
    CatalogueEntry(
        "net-volume-for-area",
        "Giving the volume of a box where its surface area was asked for.",
        area="spatial",
    ),
    CatalogueEntry(
        "net-area-for-volume",
        "Giving the surface area of a box where its volume was asked for.",
        area="spatial",
    ),
    CatalogueEntry(
        "net-edges-added-for-volume",
        "Adding the three edge lengths of a box instead of multiplying them to get its volume.",
        area="spatial",
    ),
    CatalogueEntry(
        "net-base-area-for-volume",
        "Giving the area of the base of a box as its volume, leaving out the height.",
        area="spatial",
    ),
    CatalogueEntry(
        "solid-hidden-faces-missed",
        "Counting only the faces that can be seen in the drawing and missing the ones hidden at the back or underneath.",
        area="spatial",
    ),
    CatalogueEntry(
        "solid-hidden-edges-missed",
        "Counting only the solid lines and leaving out the dashed (hidden) edges.",
        area="spatial",
    ),
    CatalogueEntry(
        "solid-hidden-vertices-missed",
        "Counting only the corners that can be seen and missing those hidden at the back.",
        area="spatial",
    ),
    CatalogueEntry(
        "solid-ends-forgotten",
        "Counting the side faces of a prism or pyramid and forgetting its base and top.",
        area="spatial",
    ),
    CatalogueEntry(
        "solid-faces-edges-vertices-mixed",
        "Mixing up faces, edges and vertices: giving the count of one where another was asked.",
        area="spatial",
    ),
    CatalogueEntry(
        "solid-net-corners-counted",
        "Counting the corners drawn on the net instead of the corners of the solid it folds into.",
        area="spatial",
    ),
    CatalogueEntry(
        "solid-net-lines-counted",
        "Counting the lines drawn on the net instead of the edges of the solid it folds into.",
        area="spatial",
    ),
    CatalogueEntry(
        "solid-edges-counted-twice",
        "Adding up the sides of every face and giving that total as the number of edges, though each edge is shared by two faces.",
        area="spatial",
    ),
    CatalogueEntry(
        "solid-euler-sign",
        "Getting the signs or the 2 wrong in Euler's formula, V - E + F = 2, when solving for the vertices.",
        area="spatial",
    ),
    CatalogueEntry(
        "plan-squares-counted",
        "Counting the squares that have cubes on the plan instead of adding up the cubes stacked in them.",
        area="spatial",
    ),
    CatalogueEntry(
        "plan-tallest-times-squares",
        "Multiplying the tallest stack by the number of squares on the plan, as if every stack were as tall as the tallest.",
        area="spatial",
    ),
    CatalogueEntry(
        "plan-elevation-for-total",
        "Giving the number of squares in an elevation as the number of cubes in the whole structure, or the reverse.",
        area="spatial",
    ),
    CatalogueEntry(
        "plan-stack-left-out",
        "Leaving out a stack, or a whole row of the plan, when adding up the cubes.",
        area="spatial",
    ),
    CatalogueEntry(
        "plan-total-for-elevation",
        "Adding every stack on the plan for an elevation, though an elevation shows only the tallest stack in each column or row.",
        area="spatial",
    ),
    CatalogueEntry(
        "plan-wrong-side-viewed",
        "Reading the elevation from the wrong side of the plan: using the rows for the front view, or the columns for the side view.",
        area="spatial",
    ),
    CatalogueEntry(
        "plan-tallest-times-columns",
        "Taking the tallest stack on the whole plan as the height of every column of the elevation.",
        area="spatial",
    ),
    CatalogueEntry(
        "plan-nearest-stacks-only",
        "Counting only the stacks nearest the viewer for an elevation, and forgetting that taller stacks behind them show above.",
        area="spatial",
    ),
    CatalogueEntry(
        "plan-line-added-for-tallest",
        "Adding the stacks in a column or row of the plan where an elevation shows only the tallest of them.",
        area="spatial",
    ),
    CatalogueEntry(
        "plan-count-off-by-one",
        "Finishing a sum of stacks one cube out when working back from an elevation.",
        area="spatial",
    ),
    CatalogueEntry(
        "plan-volume-edge-squared",
        "Squaring the edge of a cube instead of cubing it to get the volume of one cube.",
        area="spatial",
    ),
    CatalogueEntry(
        "plan-volume-edge-not-cubed",
        "Multiplying the number of cubes by the edge of one cube, not by its volume.",
        area="spatial",
    ),
    CatalogueEntry(
        "plan-volume-for-surface",
        "Finding the area of every face of every cube where the volume of the structure was asked for.",
        area="spatial",
    ),
    CatalogueEntry(
        "scale-units-not-converted",
        "Multiplying a plan length by the scale and leaving the answer in centimetres, or stopping before the units are changed.",
        area="spatial",
    ),
    CatalogueEntry(
        "scale-converted-wrong-power",
        "Changing centimetres to metres with the wrong power of ten after applying the scale.",
        area="spatial",
    ),
    CatalogueEntry(
        "scale-wrong-wall-read",
        "Using the length of a different wall from the one the question names.",
        area="spatial",
    ),
    CatalogueEntry(
        "scale-applied-the-wrong-way",
        "Dividing a plan length by the scale where it should be multiplied.",
        area="spatial",
    ),
    CatalogueEntry(
        "scale-area-linear-factor",
        "Scaling a plan area by the ratio once instead of scaling each length, so the area grows by the ratio rather than its square.",
        area="spatial",
    ),
    CatalogueEntry(
        "scale-notch-ignored",
        "Finding the area of the rectangle that surrounds an L-shaped plan and not taking away the corner that is missing.",
        area="spatial",
    ),
    CatalogueEntry(
        "scale-overlap-counted-twice",
        "Adding two rectangles that overlap, so part of an L-shaped plan is counted twice.",
        area="spatial",
    ),
    CatalogueEntry(
        "scale-notch-misread",
        "Taking away a corner rectangle of the wrong size from an L-shaped plan.",
        area="spatial",
    ),
    CatalogueEntry(
        "scale-labelled-sides-only",
        "Adding only the marked sides of an L-shaped plan for its perimeter and leaving out the sides that are not marked.",
        area="spatial",
    ),
    CatalogueEntry(
        "scale-rectangles-perimeters-added",
        "Adding the perimeters of the two rectangles that make up an L-shaped plan, so the shared side is counted twice.",
        area="spatial",
    ),
    CatalogueEntry(
        "scale-perimeter-halved",
        "Adding one of each pair of opposite sides, so the answer is half the perimeter.",
        area="spatial",
    ),
    CatalogueEntry(
        "scale-tile-side-for-area",
        "Dividing the floor area by the side of a tile instead of by its area.",
        area="spatial",
    ),
    CatalogueEntry(
        "scale-tile-area-multiplied",
        "Multiplying the floor area by the size of a tile instead of dividing by it.",
        area="spatial",
    ),
    CatalogueEntry(
        "scale-area-given-as-tiles",
        "Giving the floor area in square metres as the number of tiles.",
        area="spatial",
    ),
    CatalogueEntry(
        "scale-one-side-scaled",
        "Scaling only one side of the room to real size and multiplying it by the other side as measured on the plan.",
        area="spatial",
    ),
    CatalogueEntry(
        "map-squares-not-scaled",
        "Counting the grid squares between places and giving that count as the distance, without multiplying by the length of a square.",
        area="spatial",
    ),
    CatalogueEntry(
        "map-off-by-one-square",
        "Counting the square a place sits in as well as the squares between, or missing one, so the distance is a square out.",
        area="spatial",
    ),
    CatalogueEntry(
        "map-scale-added",
        "Adding the length of a grid square to the number of squares instead of multiplying.",
        area="spatial",
    ),
    CatalogueEntry(
        "map-wrong-place-read",
        "Reading the position of the wrong place on the map, such as a nearby place with a different name.",
        area="spatial",
    ),
    CatalogueEntry(
        "map-route-one-leg",
        "Giving the length of one leg of a walk as the length of the whole walk.",
        area="spatial",
    ),
    CatalogueEntry(
        "map-legs-subtracted",
        "Subtracting the lengths of two legs of a walk instead of adding them.",
        area="spatial",
    ),
    CatalogueEntry(
        "map-bearing-not-reversed",
        "Giving the bearing of the way there when the bearing of the way back was asked for.",
        area="spatial",
    ),
    CatalogueEntry(
        "map-bearing-anticlockwise",
        "Measuring a bearing anticlockwise from north instead of clockwise.",
        area="spatial",
    ),
    CatalogueEntry(
        "map-bearing-from-east",
        "Measuring a bearing from east, as an angle in a graph, instead of clockwise from north.",
        area="spatial",
    ),
    CatalogueEntry(
        "map-bearing-reversed-wrong",
        "Trying to reverse a bearing by adding the wrong angle, for example 90 degrees instead of 180.",
        area="spatial",
    ),
    CatalogueEntry(
        "adds-a-group-of-trials-to-the-successes",
        "Adding the number of trials in one group to the number of successes when combining results.",
        area="probability",
    ),
    CatalogueEntry(
        "adds-instead-of-dividing",
        "Adding the target count to a number of trials where the target count should be divided by the probability.",
        area="probability",
    ),
    CatalogueEntry(
        "adds-the-distances",
        "Adding two distances from the theoretical probability where their difference was asked for.",
        area="probability",
    ),
    CatalogueEntry(
        "adds-the-given-shares-instead-of-the-remainder",
        "Giving the total of the relative frequencies that were shown where the share that was left out (the remainder to 100%) was needed.",
        area="probability",
    ),
    CatalogueEntry(
        "and-instead-of-but-not",
        "Counting the trials where both events happen for an event described as one happening but not the other.",
        area="probability",
    ),
    CatalogueEntry(
        "and-instead-of-or",
        'Counting only the trials where both events happen for an event described with "or".',
        area="probability",
    ),
    CatalogueEntry(
        "answered-neither-not-or",
        "Giving the probability that neither event happens where at least one was asked for.",
        area="probability",
    ),
    CatalogueEntry(
        "answered-the-event-not-its-complement",
        'Giving the probability of the event itself where "not" the event was asked for.',
        area="probability",
    ),
    CatalogueEntry(
        "assumes-equal-likelihood-ignores-data",
        "Ignoring the results and giving one over the number of categories, as if every outcome were equally likely.",
        area="probability",
    ),
    CatalogueEntry(
        "averages-percentages-ignoring-trial-numbers",
        "Averaging the percentages from several experiments as if each had the same number of trials, instead of combining the successes and the trials.",
        area="probability",
    ),
    CatalogueEntry(
        "compares-the-two-estimates-not-the-distance-to-p",
        "Finding how far apart the two estimates are instead of how far each is from the theoretical probability.",
        area="probability",
    ),
    CatalogueEntry(
        "compares-with-the-rest-not-the-whole",
        "Dividing the favourable outcomes by the other outcomes instead of by all the outcomes.",
        area="probability",
    ),
    CatalogueEntry(
        "divides-by-one-other-count",
        "Dividing the count of an event by the count of one other category instead of by the total number of results.",
        area="probability",
    ),
    CatalogueEntry(
        "divides-by-one-row-total",
        "Dividing the count where both events happen by the total for just one of the events instead of by all the trials.",
        area="probability",
    ),
    CatalogueEntry(
        "divides-by-the-complement",
        "Dividing the target count by the probability of the opposite event instead of the probability of the event.",
        area="probability",
    ),
    CatalogueEntry(
        "expected-count-multiplies-by-favourable-only",
        "Multiplying the number of trials by the number of favourable outcomes without dividing by the number of possible outcomes.",
        area="probability",
    ),
    CatalogueEntry(
        "expected-count-uses-one-outcome",
        "Dividing the number of trials by the number of possible outcomes, as though only one outcome were favourable.",
        area="probability",
    ),
    CatalogueEntry(
        "ignores-the-overlap-for-but-not",
        "Giving everything in the first event when the part that is also in the second event had to be taken out.",
        area="probability",
    ),
    CatalogueEntry(
        "ignores-the-removed-marbles",
        "Using the original numbers in the bag when some marbles had been taken out before the draw.",
        area="probability",
    ),
    CatalogueEntry(
        "ignores-the-second-event",
        "Answering for the first event alone when a question about both events was asked.",
        area="probability",
    ),
    CatalogueEntry(
        "leaves-a-group-out-of-the-total",
        "Leaving one group's results out when adding up the successes, so the combined estimate is wrong.",
        area="probability",
    ),
    CatalogueEntry(
        "leaves-out-the-overlap-for-or",
        'Counting only the trials where exactly one event happens for an "or" event, leaving out the trials where both happen.',
        area="probability",
    ),
    CatalogueEntry(
        "miscounts-by-one",
        "Being one out when adding or subtracting counts.",
        area="probability",
    ),
    CatalogueEntry(
        "off-by-a-favourable-outcome",
        "Adding or leaving out one favourable outcome's share of the expected count.",
        area="probability",
    ),
    CatalogueEntry(
        "or-instead-of-and",
        "Giving the probability that at least one event happens where both happening was asked for.",
        area="probability",
    ),
    CatalogueEntry(
        "or-instead-of-exactly-one",
        "Counting the trials where both events happen as well, for an event where exactly one must happen.",
        area="probability",
    ),
    CatalogueEntry(
        "overlap-counted-twice-in-or",
        "Adding the totals for the two events without taking off the overlap, so the trials where both happen are counted twice.",
        area="probability",
    ),
    CatalogueEntry(
        "pools-when-asked-for-one-result",
        "Combining all the results into one estimate when the question asked which single result to rely on.",
        area="probability",
    ),
    CatalogueEntry(
        "reports-expected-not-difference",
        "Giving the expected count where the difference between observed and expected was asked for.",
        area="probability",
    ),
    CatalogueEntry(
        "reports-frequency-not-relative-frequency",
        "Giving the count of the event where a fraction of all the results was asked for.",
        area="probability",
    ),
    CatalogueEntry(
        "reports-observed-not-difference",
        "Giving the observed count where the difference between observed and expected was asked for.",
        area="probability",
    ),
    CatalogueEntry(
        "reports-observed-not-expected",
        "Giving the number that happened in the experiment where the number theory predicts was asked for.",
        area="probability",
    ),
    CatalogueEntry(
        "reports-one-distance-only",
        "Giving the distance from the theoretical probability for only one of the two estimates.",
        area="probability",
    ),
    CatalogueEntry(
        "reports-the-estimate-not-the-gap",
        "Giving the estimate itself where its distance from the theoretical probability was asked for.",
        area="probability",
    ),
    CatalogueEntry(
        "reports-the-target-not-the-trials",
        "Giving the number of times wanted where the number of trials needed to expect it was asked for.",
        area="probability",
    ),
    CatalogueEntry(
        "reports-the-theoretical-value-not-the-gap",
        "Giving the theoretical percentage where the gap between it and the estimate was asked for.",
        area="probability",
    ),
    CatalogueEntry(
        "reports-the-trials-not-the-successes",
        "Giving the number of trials where the number of successes was asked for.",
        area="probability",
    ),
    CatalogueEntry(
        "removes-from-the-red-but-not-the-total",
        "Taking removed marbles off the count of one colour but forgetting that the total number of marbles drops as well.",
        area="probability",
    ),
    CatalogueEntry(
        "slips-a-decimal-place",
        "Moving the decimal point one place when working with percentages.",
        area="probability",
    ),
    CatalogueEntry(
        "subtracts-both-totals-without-adding-back-the-overlap",
        "Subtracting the totals for both events from all the trials without adding back the trials where both happened.",
        area="probability",
    ),
    CatalogueEntry(
        "swaps-the-two-events",
        "Answering for the second event where the first was asked for, or the reverse.",
        area="probability",
    ),
    CatalogueEntry(
        "trusts-a-small-experiment",
        "Using the result of a small experiment as the best estimate, when a larger experiment gives a more reliable one.",
        area="probability",
    ),
    CatalogueEntry(
        "uses-a-half-turn-as-the-whole-circle",
        "Dividing a sector angle by 180 degrees instead of 360 degrees to find its share of the circle.",
        area="probability",
    ),
    CatalogueEntry(
        "uses-observed-as-theoretical-probability",
        "Using what happened in the experiment as the theoretical probability of the event.",
        area="probability",
    ),
    CatalogueEntry(
        "uses-one-experiment-only",
        "Giving the result of one experiment where the results of the experiments had to be combined.",
        area="probability",
    ),
    CatalogueEntry(
        "uses-theoretical-not-the-estimate",
        "Using the theoretical probability to make a prediction where the experimental estimate was asked for.",
        area="probability",
    ),
    CatalogueEntry(
        "adds-a-fixed-amount-to-a-proportional-rule",
        "Adding a fixed amount to a rule where every output is a plain multiple of its input.",
        area="mathematics",
    ),
    CatalogueEntry(
        "adds-the-extra-count-instead-of-pricing-it",
        "Adding the number of extra items where the cost of those items had to be added.",
        area="mathematics",
    ),
    CatalogueEntry(
        "adds-the-first-output-to-the-input",
        "Writing a rule that adds a number to the input, matching one row of the table, where the outputs are multiples of the inputs.",
        area="mathematics",
    ),
    CatalogueEntry(
        "adds-the-fixed-amount-and-the-rate-first",
        "Adding the fixed amount to the rate and multiplying the sum by the input, instead of keeping the fixed amount separate.",
        area="mathematics",
    ),
    CatalogueEntry(
        "adds-the-fixed-charge-and-the-rate-first",
        "Adding the fixed charge to the rate and multiplying the sum by the quantity, instead of keeping the fixed charge separate.",
        area="mathematics",
    ),
    CatalogueEntry(
        "adds-the-fixed-charge-instead-of-taking-it-off",
        "Adding the fixed charge to the total where it had to be taken off before dividing by the rate.",
        area="mathematics",
    ),
    CatalogueEntry(
        "adds-the-rates-instead-of-subtracting",
        "Adding the two rates where the difference between them decides when two plans agree.",
        area="mathematics",
    ),
    CatalogueEntry(
        "adds-the-starting-value-and-the-rate-first",
        "Adding the starting value to the rate and multiplying by the input, instead of multiplying the rate by the input and then adding the starting value.",
        area="mathematics",
    ),
    CatalogueEntry(
        "adds-the-steps-to-one-number",
        "Adding the steps 1 and 2 to a single number, instead of adding the steps to each number in turn.",
        area="mathematics",
    ),
    CatalogueEntry(
        "adds-the-whole-change-not-the-rate",
        "Adding the whole change between two readings to the last reading, whatever the number of steps ahead.",
        area="mathematics",
    ),
    CatalogueEntry(
        "adds-where-dividing-is-needed",
        "Adding two quantities where one had to be divided by the other.",
        area="mathematics",
    ),
    CatalogueEntry(
        "adds-where-multiplying-is-needed",
        "Adding two quantities where a rate had to be multiplied by a count.",
        area="mathematics",
    ),
    CatalogueEntry(
        "applies-the-division-to-only-one-part",
        "Dividing only the first part of an expression when the division applies to the number as a whole.",
        area="mathematics",
    ),
    CatalogueEntry(
        "applies-the-multiplication-to-only-one-part",
        "Multiplying only one term inside a bracket and leaving the other term unchanged.",
        area="mathematics",
    ),
    CatalogueEntry(
        "applies-the-rate-for-only-one-step",
        "Adding the rate for a single step where several steps lie between the reading and the prediction.",
        area="mathematics",
    ),
    CatalogueEntry(
        "assumes-the-quantity-is-proportional-to-the-input",
        "Treating the output as a fixed multiple of the input, so a starting value or a changing ratio is ignored.",
        area="mathematics",
    ),
    CatalogueEntry(
        "assumes-the-sum-stays-fixed",
        "Treating the input and output as adding to a constant, in a relationship where their product is constant.",
        area="mathematics",
    ),
    CatalogueEntry(
        "compares-the-wrong-quantities",
        "Subtracting two quantities that are not the two the question compares.",
        area="mathematics",
    ),
    CatalogueEntry(
        "counts-one-step-too-many",
        "Using the input one more than the story states, or counting the start as a step.",
        area="mathematics",
    ),
    CatalogueEntry(
        "divides-instead-of-multiplying",
        "Dividing by a number where the story multiplies by it.",
        area="mathematics",
    ),
    CatalogueEntry(
        "divides-where-multiplying-is-needed",
        "Dividing by a rate where a quantity had to be multiplied by it.",
        area="mathematics",
    ),
    CatalogueEntry(
        "doubles-instead-of-squaring",
        "Doubling the input where the rule squares it.",
        area="mathematics",
    ),
    CatalogueEntry(
        "fits-only-the-first-row-of-the-table",
        "Choosing a rule that works for the first pair in the table but fails for the others.",
        area="mathematics",
    ),
    CatalogueEntry(
        "forgets-to-divide-by-the-rate",
        "Finishing the solution with the rate still multiplying the variable.",
        area="mathematics",
    ),
    CatalogueEntry(
        "gets-the-direction-of-change-wrong",
        "Treating a quantity that falls as one that rises, or the reverse, so the sign of the rate is wrong.",
        area="mathematics",
    ),
    CatalogueEntry(
        "gives-both-kinds-the-same-count",
        "Using one count for two kinds of item, when one kind has more than the other.",
        area="mathematics",
    ),
    CatalogueEntry(
        "ignores-the-fixed-charge-when-solving",
        "Dividing the whole total by the rate when a fixed charge had to be taken off first.",
        area="mathematics",
    ),
    CatalogueEntry(
        "ignores-the-rate-in-the-first-plan",
        "Dividing the joining fee by the second plan's rate alone instead of by the difference between the rates.",
        area="mathematics",
    ),
    CatalogueEntry(
        "ignores-the-starting-value",
        "Dividing the target by the rate without taking the starting value off it first.",
        area="mathematics",
    ),
    CatalogueEntry(
        "inverts-the-rate",
        "Dividing the change in the input by the change in the output, so the rate is upside down.",
        area="mathematics",
    ),
    CatalogueEntry(
        "inverts-the-rule",
        "Writing the rule with the input and the constant the wrong way up.",
        area="mathematics",
    ),
    CatalogueEntry(
        "leaves-a-number-out",
        "Leaving one of the numbers in the sum out of the expression.",
        area="mathematics",
    ),
    CatalogueEntry(
        "leaves-out-the-bracket",
        "Writing the multiplication without a bracket, so it applies to only one of the two terms.",
        area="mathematics",
    ),
    CatalogueEntry(
        "leaves-out-the-fixed-amount",
        "Writing a rule with only the rate, and leaving out the amount that is there at the start.",
        area="mathematics",
    ),
    CatalogueEntry(
        "leaves-out-the-fixed-charge",
        "Writing an expression with only the per-unit charge and leaving out the fixed charge.",
        area="mathematics",
    ),
    CatalogueEntry(
        "leaves-out-the-starting-value",
        "Multiplying the rate by the input without adding the starting value.",
        area="mathematics",
    ),
    CatalogueEntry(
        "leaves-out-the-voucher",
        "Writing the full price with nothing taken off for the voucher.",
        area="mathematics",
    ),
    CatalogueEntry(
        "leaves-out-the-width-sides",
        "Writing the two long sides but not the two short sides of the perimeter.",
        area="mathematics",
    ),
    CatalogueEntry(
        "miscounts-the-multiplier",
        "Writing a multiplier that is one more than the pattern in the table.",
        area="mathematics",
    ),
    CatalogueEntry(
        "multiplies-where-adding-is-needed",
        "Multiplying two quantities where one had to be added to the other.",
        area="mathematics",
    ),
    CatalogueEntry(
        "multiplies-where-dividing-is-needed",
        "Multiplying by the rate where the quantity had to be divided by it.",
        area="mathematics",
    ),
    CatalogueEntry(
        "multiplies-where-subtracting-is-needed",
        "Multiplying by the amount taken away where it had to be subtracted.",
        area="mathematics",
    ),
    CatalogueEntry(
        "operation-from-keyword-alone",
        'Choosing the operation from a keyword such as "then" or "total" without checking what the story does to the number.',
        area="mathematics",
    ),
    CatalogueEntry(
        "puts-the-total-in-the-expression",
        "Using the stated total inside the expression for the left-hand side, where it belongs on the right.",
        area="mathematics",
    ),
    CatalogueEntry(
        "reports-a-difference-as-the-constant",
        "Giving the difference between two outputs where the constant value of the products was asked for.",
        area="mathematics",
    ),
    CatalogueEntry(
        "reports-the-change-as-the-starting-value",
        "Giving the change between two readings as the starting value.",
        area="mathematics",
    ),
    CatalogueEntry(
        "reports-the-extra-time-since-the-first-reading",
        "Giving the time since the first reading where the total time was asked for.",
        area="mathematics",
    ),
    CatalogueEntry(
        "reports-the-extra-time-since-the-last-reading",
        "Giving the extra time after the last reading where the total time was asked for.",
        area="mathematics",
    ),
    CatalogueEntry(
        "reports-the-first-output-as-the-constant",
        "Giving the first output in the table as the constant of the relationship.",
        area="mathematics",
    ),
    CatalogueEntry(
        "reports-the-first-reading-as-the-starting-value",
        "Giving the first reading as the starting value, as though it had been taken at the start.",
        area="mathematics",
    ),
    CatalogueEntry(
        "reports-the-fixed-charge-as-the-saving",
        "Giving the fixed charge as the amount saved, without comparing the two totals.",
        area="mathematics",
    ),
    CatalogueEntry(
        "reports-the-other-total-not-the-difference",
        "Giving the other supplier's total where the difference between the totals was asked for.",
        area="mathematics",
    ),
    CatalogueEntry(
        "reverses-the-division",
        "Dividing the other way round, so the number divides the unknown quantity instead of the unknown dividing the number.",
        area="mathematics",
    ),
    CatalogueEntry(
        "reverses-the-order-of-the-steps",
        "Subtracting before dividing, where the story divides first and then subtracts.",
        area="mathematics",
    ),
    CatalogueEntry(
        "reverses-the-subtraction",
        "Subtracting in the wrong order, so the expression is the negative of the one the story describes.",
        area="mathematics",
    ),
    CatalogueEntry(
        "rounds-up-where-the-model-stops-making-sense",
        "Rounding the point where a model reaches zero up to the next whole number, so the model gives a negative value there.",
        area="mathematics",
    ),
    CatalogueEntry(
        "shifts-the-input-before-dividing",
        "Adding one to the input before dividing, as though the count started from zero.",
        area="mathematics",
    ),
    CatalogueEntry(
        "squares-the-multiplier-too",
        "Squaring the multiplier together with the input, where only the input is squared.",
        area="mathematics",
    ),
    CatalogueEntry(
        "stops-one-early",
        "Giving a whole number that is one lower than the last value for which the model still makes sense.",
        area="mathematics",
    ),
    CatalogueEntry(
        "subtracts-the-rate-from-the-starting-value",
        "Taking the rate off the starting value once, where the rate is taken off for every step.",
        area="mathematics",
    ),
    CatalogueEntry(
        "subtracts-the-rate-instead-of-dividing",
        "Subtracting the rate from the total where the total had to be divided by the rate.",
        area="mathematics",
    ),
    CatalogueEntry(
        "swaps-the-fixed-amount-and-the-rate",
        "Swapping the fixed amount and the rate, so the rule multiplies the input by the fixed amount.",
        area="mathematics",
    ),
    CatalogueEntry(
        "swaps-the-fixed-charge-and-the-rate",
        "Swapping the fixed charge and the rate, so the fixed charge is multiplied by the quantity.",
        area="mathematics",
    ),
    CatalogueEntry(
        "swaps-the-starting-value-and-the-rate",
        "Swapping the starting value and the rate, so the starting value is multiplied by the input.",
        area="mathematics",
    ),
    CatalogueEntry(
        "swaps-which-quantity-is-subtracted",
        "Subtracting the fixed amount from the rate term where the rate term is taken from the fixed amount.",
        area="mathematics",
    ),
    CatalogueEntry(
        "takes-the-voucher-off-the-price-not-the-bill",
        "Taking the voucher off the price of one item before multiplying, instead of off the whole bill.",
        area="mathematics",
    ),
    CatalogueEntry(
        "treats-consecutive-numbers-as-equal",
        "Treating three consecutive numbers as three copies of the smallest.",
        area="mathematics",
    ),
    CatalogueEntry(
        "treats-the-extra-as-the-whole-length",
        "Treating the extra length as the whole of the longer side.",
        area="mathematics",
    ),
    CatalogueEntry(
        "treats-the-extra-as-the-whole-number",
        "Treating the extra items as the whole number of that kind of item.",
        area="mathematics",
    ),
    CatalogueEntry(
        "treats-the-input-as-the-power",
        "Writing the input as an exponent where the rule squares the input.",
        area="mathematics",
    ),
    CatalogueEntry(
        "uses-addition-for-a-per-unit-rate",
        "Adding the price of one item to the number of items where the cost of all the items is the price times the number.",
        area="mathematics",
    ),
    CatalogueEntry(
        "uses-half-the-perimeter",
        "Adding the length and width once, which gives half the perimeter.",
        area="mathematics",
    ),
    CatalogueEntry(
        "uses-the-area-formula-for-perimeter",
        "Multiplying the length and width, which gives the area, where the perimeter was described.",
        area="mathematics",
    ),
    CatalogueEntry(
        "uses-the-change-between-readings-as-the-rate",
        "Taking the change between two readings as the rate per step, when the readings are several steps apart.",
        area="mathematics",
    ),
    CatalogueEntry(
        "uses-the-total-in-place-of-the-quantity",
        "Using the stated total where the unknown quantity belongs.",
        area="mathematics",
    ),
    CatalogueEntry(
        "uses-the-wrong-operation-for-the-change",
        "Adding where the story takes away, or taking away where it adds.",
        area="mathematics",
    ),
    CatalogueEntry(
        "divides-by-one-plans-rate-only",
        "Dividing the joining fee by the rate of one plan alone, when the plans agree where the fee equals the difference in cost per unit.",
        area="mathematics",
    ),
    CatalogueEntry(
        "gives-the-sibling-branch",
        "Giving the probability written on the branch beside the missing one instead of working out what is left from 1.",
        area="probability",
    ),
    CatalogueEntry(
        "second-draw-total-on-first-draw",
        "Dividing by the number left after the first draw (one fewer) when the probability is for the first draw.",
        area="probability",
    ),
    CatalogueEntry(
        "first-draw-total-on-second-draw",
        "Dividing by the whole bag when the probability is for the second draw, forgetting that one item has gone.",
        area="probability",
    ),
    CatalogueEntry(
        "copies-a-branch-from-another-set",
        "Copying a probability from a different set of branches into the missing branch.",
        area="probability",
    ),
    CatalogueEntry(
        "forgets-the-removed-counter",
        "Dividing by the number left but not taking away the item of the same colour that was removed, so the "
        "numerator is one too big.",
        area="probability",
    ),
    CatalogueEntry(
        "adds-along-a-path",
        "Adding the probabilities along a path of a tree diagram instead of multiplying them.",
        area="probability",
    ),
    CatalogueEntry(
        "stops-at-the-first-branch",
        "Giving the probability on the first branch of a path and ignoring the second branch.",
        area="probability",
    ),
    CatalogueEntry(
        "multiplies-the-wrong-path",
        "Multiplying along a different path from the one asked for, for example the other colour twice.",
        area="probability",
    ),
    CatalogueEntry(
        "leaves-out-a-path",
        "Finding the probability of one path of a tree and leaving out the other path that also fits the event.",
        area="probability",
    ),
    CatalogueEntry(
        "multiplies-paths-instead-of-adding",
        "Multiplying the probabilities of two different paths together where they should be added.",
        area="probability",
    ),
    CatalogueEntry(
        "adds-shown-without-total",
        "Adding the numbers shown on a diagram and giving that as the missing one, never using the stated total.",
        area="statistics",
    ),
    CatalogueEntry(
        "leaves-out-a-region",
        "Subtracting all but one of the shown regions from the total, so the answer is too big by the region left out.",
        area="statistics",
    ),
    CatalogueEntry(
        "overlap-left-out-of-set",
        "Counting only the part of a circle outside the overlap when asked for everyone in that set.",
        area="statistics",
    ),
    CatalogueEntry(
        "reads-the-other-circle",
        "Adding up the circle of the other set when the set named in the question was asked for.",
        area="statistics",
    ),
    CatalogueEntry(
        "counts-either-set-for-one-set",
        "Adding every region inside both circles when only the people in one named set were asked for.",
        area="statistics",
    ),
    CatalogueEntry(
        "gives-the-overlap-alone",
        "Giving only the overlap of two sets as the number in one of them.",
        area="statistics",
    ),
    CatalogueEntry(
        "exactly-one-with-overlap",
        "Including the overlap when asked for exactly one of two sets.",
        area="probability",
    ),
    CatalogueEntry(
        "denominator-leaves-out-neither",
        "Dividing by the number of people inside the circles instead of the whole group, leaving out those who do "
        "neither.",
        area="probability",
    ),
    CatalogueEntry(
        "counts-one-circle-only",
        "Counting only the people in one circle only, when the event includes the other circle's only-part as well.",
        area="probability",
    ),
    CatalogueEntry(
        "gives-the-smallest-value-as-the-lower-quartile",
        "Giving the smallest value in the data as the lower quartile.",
        area="statistics",
    ),
    CatalogueEntry(
        "gives-the-largest-value-as-the-upper-quartile",
        "Giving the largest value in the data as the upper quartile.",
        area="statistics",
    ),
    CatalogueEntry(
        "quartile-one-position-out",
        "Reading the value next to the quartile in the ordered list, one place too early or too late.",
        area="statistics",
    ),
    CatalogueEntry(
        "reads-the-leaf-alone",
        "Giving only the leaf as the value (for example 7 from 3 | 7), ignoring the stem and the key.",
        area="statistics",
    ),
    CatalogueEntry(
        "gives-the-range-as-the-iqr",
        "Giving the largest value minus the smallest (the range) where the interquartile range was asked for.",
        area="statistics",
    ),
    CatalogueEntry(
        "subtracts-the-median-from-q3",
        "Finding the interquartile range as the upper quartile minus the median instead of minus the lower quartile.",
        area="statistics",
    ),
    CatalogueEntry(
        "subtracts-q1-from-the-median",
        "Finding the interquartile range as the median minus the lower quartile.",
        area="statistics",
    ),
    CatalogueEntry(
        "adds-the-quartiles",
        "Adding the two quartiles where the interquartile range needs their difference.",
        area="statistics",
    ),
    CatalogueEntry(
        "uses-one-times-the-iqr",
        "Adding one interquartile range to the upper quartile instead of one and a half.",
        area="statistics",
    ),
    CatalogueEntry(
        "uses-three-times-the-iqr",
        "Adding three interquartile ranges to the upper quartile, reading 1.5 x IQR as 3 x IQR.",
        area="statistics",
    ),
    CatalogueEntry(
        "leaves-out-q3-from-the-boundary",
        "Giving 1.5 times the interquartile range alone, without adding it to the upper quartile.",
        area="statistics",
    ),
    CatalogueEntry(
        "uses-the-lower-boundary",
        "Finding the boundary below which values are outliers (Q1 less 1.5 x IQR) where the upper one was asked for.",
        area="statistics",
    ),
    CatalogueEntry(
        "net-degree-miscount",
        "Miscounting the edges at a vertex (or the vertices of a stated kind), so the answer is one too high or one too low.",
        area="networks",
    ),
    CatalogueEntry(
        "net-degree-wrong-vertex",
        "Giving the degree of a different vertex from the one asked about.",
        area="networks",
    ),
    CatalogueEntry(
        "net-degree-as-edge-total",
        "Giving the total number of edges in the network when the degree of one vertex was asked for.",
        area="networks",
    ),
    CatalogueEntry(
        "net-degree-sum-as-edges",
        "Taking the sum of the degrees as the number of edges, forgetting that every edge is counted at both of its ends.",
        area="networks",
    ),
    CatalogueEntry(
        "net-edges-as-degree-sum",
        "Using the number of edges where the sum of the degrees belongs, forgetting to double it first.",
        area="networks",
    ),
    CatalogueEntry(
        "net-handshake-slip",
        "Right method for the degree sum, but the arithmetic is out by one or two.",
        area="networks",
    ),
    CatalogueEntry(
        "net-vertices-as-edges",
        "Giving the number of vertices when the number of edges was asked for.",
        area="networks",
    ),
    CatalogueEntry(
        "net-odd-even-swapped",
        "Counting the vertices of even degree when the vertices of odd degree were asked for, or the reverse.",
        area="networks",
    ),
    CatalogueEntry(
        "net-odd-degrees-summed",
        "Adding up the odd degrees instead of counting how many vertices have an odd degree.",
        area="networks",
    ),
    CatalogueEntry(
        "net-half-of-missing",
        "Halving the missing amount after subtracting from the degree sum, as if it were the number of edges at the vertex.",
        area="networks",
    ),
    CatalogueEntry(
        "net-average-degree-used",
        "Giving the average of the known degrees for the missing degree instead of using the degree sum.",
        area="networks",
    ),
    CatalogueEntry(
        "net-degree-class-swapped",
        "Giving the number of vertices of the other degree than the one asked for.",
        area="networks",
    ),
    CatalogueEntry(
        "net-assumes-equal-split",
        "Assuming half the vertices have each of the two stated degrees without using the number of edges.",
        area="networks",
    ),
    CatalogueEntry(
        "net-forgot-divide-by-gap",
        "Stopping at the amount the degree sum exceeds the smaller degree times the vertices, without dividing by the "
        "difference between the two degrees.",
        area="networks",
    ),
    CatalogueEntry(
        "net-rounded-up-not-down",
        "Rounding up when the greatest whole number of vertices was wanted, which would break the degree condition.",
        area="networks",
    ),
    CatalogueEntry(
        "net-degree-sum-as-vertices",
        "Giving the degree sum as the number of vertices.",
        area="networks",
    ),
    CatalogueEntry(
        "net-halved-instead-of-doubled",
        "Halving the number of edges where the degree sum (twice the edges) belongs.",
        area="networks",
    ),
    CatalogueEntry(
        "net-listed-sum-only",
        "Giving the sum of the degrees that are listed instead of the sum of all of them.",
        area="networks",
    ),
    CatalogueEntry(
        "net-hidden-vertex-ignored",
        "Counting only the listed vertices, leaving out the vertex whose degree had to be found first.",
        area="networks",
    ),
    CatalogueEntry(
        "net-deleted-edges-kept",
        "Giving the original number of edges after a vertex and its edges are deleted.",
        area="networks",
    ),
    CatalogueEntry(
        "net-edges-removed-twice",
        "Removing the edges at a deleted vertex twice, once at each end.",
        area="networks",
    ),
    CatalogueEntry(
        "net-answered-removed-edges",
        "Giving the number of edges removed instead of the number that remain.",
        area="networks",
    ),
    CatalogueEntry(
        "net-route-too-general",
        "Naming a route with a correct but less specific name (a walk for a trail, or a trail for a path), when the most "
        "specific name was wanted.",
        area="networks",
    ),
    CatalogueEntry(
        "net-trail-allows-repeated-edge",
        "Calling a route a trail (or a closed trail) although it uses an edge more than once.",
        area="networks",
    ),
    CatalogueEntry(
        "net-path-allows-repeated-vertex",
        "Calling a route a path although it visits a vertex more than once.",
        area="networks",
    ),
    CatalogueEntry(
        "net-circuit-cycle-confused",
        "Calling a closed trail a cycle although it visits a vertex other than the start more than once, or the reverse.",
        area="networks",
    ),
    CatalogueEntry(
        "net-open-closed-confused",
        "Not checking whether the route ends where it starts, so an open route is given a closed name or the reverse.",
        area="networks",
    ),
    CatalogueEntry(
        "net-vertices-counted-as-edges",
        "Counting the vertices in a route instead of the edges (steps) it uses.",
        area="networks",
    ),
    CatalogueEntry(
        "net-repeats-not-counted",
        "Counting each different edge of a route once when every step was to be counted.",
        area="networks",
    ),
    CatalogueEntry(
        "net-repeats-counted-as-different",
        "Counting a repeated edge as a different edge each time it is used.",
        area="networks",
    ),
    CatalogueEntry(
        "net-all-network-edges-counted",
        "Giving the number of edges in the whole network instead of the number the route uses.",
        area="networks",
    ),
    CatalogueEntry(
        "net-different-edges-for-reuse",
        "Giving the number of different edges used where the number of repeated uses was asked for.",
        area="networks",
    ),
    CatalogueEntry(
        "net-repeated-vertices-for-edges",
        "Counting the repeated vertices of a route instead of the repeated uses of an edge.",
        area="networks",
    ),
    CatalogueEntry(
        "net-euler-circuit-trail-confused",
        "Mixing up an Eulerian trail (two odd vertices, different ends) with an Eulerian circuit (no odd vertices).",
        area="networks",
    ),
    CatalogueEntry(
        "net-euler-parity-reversed",
        "Thinking a network with every degree even cannot be travelled, or that odd degrees are what make a circuit work.",
        area="networks",
    ),
    CatalogueEntry(
        "net-euler-two-odd-rejected",
        "Ruling out an Eulerian trail because there are odd vertices, forgetting that exactly two odd vertices are allowed.",
        area="networks",
    ),
    CatalogueEntry(
        "net-euler-odd-vertices-ignored",
        "Not counting the odd vertices at all, so a network with four odd vertices (or two) is judged to have an Eulerian circuit.",
        area="networks",
    ),
    CatalogueEntry(
        "net-euler-many-odd-accepted",
        "Accepting an Eulerian trail with more than two odd vertices.",
        area="networks",
    ),
    CatalogueEntry(
        "net-euler-ignores-connectedness",
        "Checking only that every degree is even and not that the network is in one piece.",
        area="networks",
    ),
    CatalogueEntry(
        "net-euler-start-highest-degree",
        "Starting an Eulerian trail at the vertices of highest degree instead of the two odd vertices.",
        area="networks",
    ),
    CatalogueEntry(
        "net-euler-one-odd-one-even",
        "Choosing one odd vertex and one even vertex as the ends of an Eulerian trail.",
        area="networks",
    ),
    CatalogueEntry(
        "net-euler-start-even-vertices",
        "Choosing even vertices as the ends of an Eulerian trail.",
        area="networks",
    ),
    CatalogueEntry(
        "net-euler-edge-per-odd-vertex",
        "Adding one new edge for every odd vertex, when each new edge fixes two odd vertices at once.",
        area="networks",
    ),
    CatalogueEntry(
        "net-euler-added-edges-not-counted",
        "Giving the original number of edges for the length of the circuit, leaving out the edges that were added.",
        area="networks",
    ),
    CatalogueEntry(
        "net-hamilton-skipped-vertex-accepted",
        "Accepting a closed route as a Hamiltonian cycle although it misses a vertex.",
        area="networks",
    ),
    CatalogueEntry(
        "net-hamilton-repeat-accepted",
        "Accepting a route as a Hamiltonian cycle although it visits a vertex twice.",
        area="networks",
    ),
    CatalogueEntry(
        "net-hamilton-path-counted-as-cycle",
        "Counting a route that visits every vertex once but does not return to its start as a Hamiltonian cycle.",
        area="networks",
    ),
    CatalogueEntry(
        "net-hamilton-non-edge-accepted",
        "Accepting a route that moves between two vertices that no edge joins.",
        area="networks",
    ),
    CatalogueEntry(
        "net-hamilton-valid-rejected",
        "Rejecting a genuine Hamiltonian cycle, for example because it is written from a different start or in reverse.",
        area="networks",
    ),
    CatalogueEntry(
        "net-hamilton-miscount",
        "Listing the Hamiltonian cycles or paths and missing one, or counting one twice.",
        area="networks",
    ),
    CatalogueEntry(
        "net-hamilton-direction-counted",
        "Counting a Hamiltonian cycle and the same cycle travelled in reverse as two different cycles.",
        area="networks",
    ),
    CatalogueEntry(
        "net-hamilton-start-counted",
        "Counting the same Hamiltonian cycle once for each vertex it can be started from.",
        area="networks",
    ),
    CatalogueEntry(
        "net-hamilton-all-orderings-counted",
        "Counting every starting vertex and both directions of each Hamiltonian cycle as different cycles.",
        area="networks",
    ),
    CatalogueEntry(
        "net-hamilton-direction-ignored",
        "Counting tours by starting vertex only, so a tour and its reverse are not told apart when they should be.",
        area="networks",
    ),
    CatalogueEntry(
        "net-hamilton-start-ignored",
        "Counting tours by direction only (or not at all), so tours from different starting vertices are not told apart "
        "when they should be.",
        area="networks",
    ),
    CatalogueEntry(
        "net-hamilton-any-start-counted",
        "Counting Hamiltonian paths from every vertex when only those from the named vertex were wanted.",
        area="networks",
    ),
    CatalogueEntry(
        "net-hamilton-path-cycle-confused",
        "Counting only the Hamiltonian paths that can return to the start, which are the cycles, instead of all the paths.",
        area="networks",
    ),
    CatalogueEntry(
        "net-spanning-tree-instead-of-tour",
        "Giving the length of a minimum spanning tree for the shortest tour, which must also return to the start.",
        area="networks",
    ),
    CatalogueEntry(
        "net-longest-tour-chosen",
        "Choosing the longest tour when the shortest was asked for.",
        area="networks",
    ),
    CatalogueEntry(
        "net-second-cheapest-tour-chosen",
        "Stopping at the second-shortest tour.",
        area="networks",
    ),
    CatalogueEntry(
        "net-cheapest-roads-chosen",
        "Adding the shortest few roads without checking that they make a tour that visits every place once and returns.",
        area="networks",
    ),
    CatalogueEntry(
        "net-cents-as-dollars",
        "Multiplying by 15 instead of 0.15, reading the price in cents as dollars.",
        area="networks",
    ),
    CatalogueEntry(
        "net-divided-by-price",
        "Dividing the distance by the price per kilometre instead of multiplying.",
        area="networks",
    ),
    CatalogueEntry(
        "net-doubled-return",
        "Doubling the cost of a tour that already returns to its start.",
        area="networks",
    ),
    CatalogueEntry(
        "net-price-decimal-slip",
        "Misplacing the decimal point in the price per kilometre.",
        area="networks",
    ),
    CatalogueEntry(
        "net-adjacency-edges-for-degree",
        "Giving the number of edges in the network for the degree of one vertex.",
        area="networks",
    ),
    CatalogueEntry(
        "net-adjacency-sum-for-degree",
        "Adding every entry of the matrix for the degree of one vertex instead of one row.",
        area="networks",
    ),
    CatalogueEntry(
        "net-adjacency-vertices-for-degree",
        "Giving the number of other vertices for the degree of one vertex, as if it were joined to all of them.",
        area="networks",
    ),
    CatalogueEntry(
        "net-adjacency-sum-as-edges",
        "Taking the sum of all the entries of an adjacency matrix as the number of edges, forgetting each edge appears twice.",
        area="networks",
    ),
    CatalogueEntry(
        "net-adjacency-one-entry-per-edge",
        "Putting a 1 for each edge in one place only (above the diagonal), or counting one 1 per edge, when the matrix is "
        "symmetric.",
        area="networks",
    ),
    CatalogueEntry(
        "net-adjacency-all-entries-counted",
        "Counting every entry of the matrix, 0s as well as 1s.",
        area="networks",
    ),
    CatalogueEntry(
        "net-adjacency-walk-length-confused",
        "Reading the entry of the matrix itself (walks of length 1) where walks of length 2 or 3 were asked for.",
        area="networks",
    ),
    CatalogueEntry(
        "net-adjacency-squared-entrywise",
        "Squaring (or cubing) each entry instead of multiplying the matrix by itself.",
        area="networks",
    ),
    CatalogueEntry(
        "net-adjacency-row-column-added",
        "Adding the row and the column together instead of multiplying them term by term and adding.",
        area="networks",
    ),
    CatalogueEntry(
        "net-adjacency-degrees-multiplied",
        "Multiplying the degrees of the two end vertices to count walks of length 2.",
        area="networks",
    ),
    CatalogueEntry(
        "net-adjacency-one-step-only",
        "Counting walks of length 1 (the degree) where walks of length 2 were asked for.",
        area="networks",
    ),
    CatalogueEntry(
        "net-adjacency-all-starts-counted",
        "Counting the walks of length 2 from every vertex when only those from one vertex were wanted.",
        area="networks",
    ),
    CatalogueEntry(
        "net-degree-sum-as-walks",
        "Giving the sum of all the degrees as the number of walks of length 2 from one vertex.",
        area="networks",
    ),
    CatalogueEntry(
        "net-adjacency-ends-not-restricted",
        "Giving all the walks of length 2 from the vertex when only those that return to it were wanted.",
        area="networks",
    ),
    CatalogueEntry(
        "net-adjacency-ends-elsewhere",
        "Giving the walks that end elsewhere (or the walks that return) when the other kind was asked for.",
        area="networks",
    ),
    CatalogueEntry(
        "net-closed-walks-not-subtracted",
        "Not taking the walks that return to the start away from the total when only those ending elsewhere were wanted.",
        area="networks",
    ),
    CatalogueEntry(
        "net-adjacency-diagonal-ones",
        "Writing 1s on the diagonal of an adjacency matrix, as if each vertex were joined to itself.",
        area="networks",
    ),
    CatalogueEntry(
        "net-adjacency-edge-missed",
        "Leaving an edge out of the adjacency matrix.",
        area="networks",
    ),
    CatalogueEntry(
        "net-adjacency-wrong-vertex-order",
        "Writing the rows and columns of the adjacency matrix in a different order from the alphabetical order of the vertices.",
        area="networks",
    ),
    CatalogueEntry(
        "mech-forgot-divide-by-mass",
        "Reporting the resultant force in newtons where the acceleration was asked, without dividing by the mass.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-resistance-added",
        "Adding a resisting force to the forces that drive the motion instead of subtracting it.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-ignored-one-force",
        "Leaving one of the stated forces out of the resultant.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-multiplied-by-mass",
        "Multiplying by the mass where F = ma needs a division by it.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-resultant-for-resistance",
        "Reporting the resultant force as the unknown resistance.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-forgot-mass-in-resultant",
        "Using the acceleration as the resultant force, without multiplying it by the mass.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-ignored-resistance",
        "Leaving the resisting force out of the resultant force.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-speed-used-as-acceleration",
        "Using the speed reached as if it were the acceleration, without dividing by the time.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-forgot-two-in-kinematics",
        "Dropping the 2 in v² = u² + 2as when finding the acceleration or speed.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-forgot-half-in-kinematics",
        "Dropping the one half in s = ½at² when finding the acceleration from a distance and a time.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-kinetic-energy-for-force",
        "Reporting the kinetic energy ½mu² (or part of it) as the braking force, without dividing by the stopping distance.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-weight-and-acceleration-opposed",
        "Subtracting the acceleration from g where the load accelerates upward, or adding it where it accelerates downward.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-ignored-weight",
        "Leaving the weight out of the vertical force balance, so the cable is thought to supply only m×a.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-ignored-acceleration",
        "Taking the tension to be just the weight although the load is accelerating.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-resultant-sign-reversed",
        "Subtracting the resultant from the known forces instead of the known forces from the resultant, which reverses the sign of the unknown force.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-added-known-forces-to-resultant",
        "Adding the known forces to the resultant instead of subtracting them when finding the unknown force.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-used-one-component",
        "Reporting one component of a vector as its magnitude.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-added-components",
        "Adding the components of a vector to get its magnitude instead of using Pythagoras.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-resultant-for-unknown",
        "Giving the magnitude or direction of the resultant force where the unknown force was asked for.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-angle-from-wrong-axis",
        "Measuring the angle from the north axis instead of from east, which gives its complement.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-quadrant-ignored",
        "Taking the angle from the inverse tangent of the positive ratio without placing the vector in its true quadrant.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-angle-reflected",
        "Giving 180° minus the angle instead of the angle.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-system-mass-omitted",
        "Using only one body's mass as the mass of the whole system, or the wrong mass in a single body's equation.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-forgot-acceleration-of-car",
        "Taking the tension to be the driving force less the resistance on the car, forgetting that the car is accelerating too.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-masses-swapped",
        "Putting the block's mass where the hanging mass belongs, so the wrong weight drives the motion.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-free-fall-assumed",
        "Giving g as the acceleration, as if the hanging mass fell freely.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-tension-equals-weight",
        "Taking the tension in the string to equal the weight of one of the masses, which is only true when there is no acceleration.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-wrong-body-mass",
        "Using the mass of the other body in one body's equation of motion.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-ratio-inverted",
        "Inverting the ratio of the masses, so the acceleration comes out larger than g.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-tension-averaged",
        "Taking the tension to be the average of the two weights.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-forgot-g",
        "Multiplying the coefficient of friction by the mass and leaving out g.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-friction-without-coefficient",
        "Giving the normal reaction as the friction force, leaving out the coefficient of friction.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-g-applied-twice",
        "Multiplying by g a second time after the weight has already been found.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-ignored-friction",
        "Leaving the friction force out of the equation of motion.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-friction-direction-reversed",
        "Adding the friction force to the driving weight, as if friction helped the motion.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-friction-omitted-from-block",
        "Leaving friction out of the block's own equation, so the tension is just m×a.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-forgot-square-root",
        "Reporting v² as the speed.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-sin-cos-swapped",
        "Using sin where cos belongs (or the reverse) when resolving the weight along and perpendicular to a slope.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-weight-not-resolved",
        "Using the whole weight mg where a component of it is needed, as if the plane were horizontal or vertical.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-component-resolved-twice",
        "Resolving the weight into a component and then resolving that component again.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-tan-inverted",
        "Giving the coefficient of friction as cos θ / sin θ instead of sin θ / cos θ.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-sin-used-as-mu",
        "Taking the coefficient of friction on the point of slipping to be sin θ.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-cos-used-as-mu",
        "Taking the coefficient of friction on the point of slipping to be cos θ.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-limiting-friction-assumed",
        "Giving the greatest friction μR for a block that is at rest, when the friction actually acting only balances the weight component along the slope.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-ignored-weight-component",
        "Leaving the weight component along the slope out of the forces on the block.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-forgot-square-on-time",
        "Using t where t² is needed in s = ½at².",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-speed-used-as-distance",
        "Reporting the speed after t seconds (at) as the distance travelled.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-sign-error-in-balance",
        "Adding the known components and stopping, instead of making the unknown one cancel them: the sign of the answer is reversed.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-wrong-component",
        "Balancing the forces in the wrong direction, using the north components where the east components are needed or the reverse.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-strings-swapped",
        "Giving the tension in the other string.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-tan-inverted-in-balance",
        "Using cos θ / sin θ where tan θ is needed, or the reverse, when relating the horizontal and vertical forces.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-assumed-symmetric",
        "Assuming the two strings share the weight equally, so each tension is W / (2 sin angle), although the angles differ.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-other-string-ignored",
        "Balancing the weight against one string alone, as if the other string took no part.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-horizontal-balance-skipped",
        "Using only the vertical balance with both sines added, and never balancing the horizontal components.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-halved-weight",
        "Taking half the weight.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-subtracted-instead-of-pythagoras",
        "Subtracting the weight from the greatest tension where the horizontal force, the weight and the tension form a right-angled triangle.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-added-squares",
        "Adding the squares of the weight and the greatest tension where the tension is the hypotenuse.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-limit-resolved-as-fixed-angle",
        "Resolving the greatest tension at the original angle, although the angle grows as the horizontal force increases.",
        area="mechanics",
    ),
    CatalogueEntry(
        "mech-predicted-reported-as-residual",
        "Reporting the predicted value where the residual was asked for.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-observed-reported-as-residual",
        "Reporting the observed value where the residual was asked for.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-intercept-left-out",
        "Leaving the intercept out when predicting from the line, so the prediction is just slope times x.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-slope-term-left-out",
        "Leaving the slope term out when predicting from the line, so the prediction is just the intercept.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-sign-of-residual-dropped",
        "Reporting the size of the residual without its sign.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-wrong-extreme-chosen",
        "Choosing the greatest residual by value, not by size, so a large negative residual is passed over.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-first-residual-only",
        "Computing one residual and reporting it as the answer for all the observations.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-sum-sign-not-reversed",
        "Giving the sum of the residuals that are listed instead of the number that makes the total zero.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-missing-residual-taken-as-zero",
        "Assuming a residual that is not listed must be zero.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-mean-of-the-others",
        "Giving the mean of the listed residuals as the missing residual.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-residuals-always-sum-to-zero",
        "Saying residuals always sum to zero, which is only true about the least-squares line.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-absolute-residuals-summed",
        "Adding the sizes of the residuals instead of the residuals themselves.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-squared-residuals-summed",
        "Adding the squares of the residuals where their plain sum was asked.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-observed-used-as-predicted",
        "Using the observed value as the predicted value.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-x-value-replaced-by-one",
        "Putting x = 1 into the line instead of the stated value.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-intercept-and-slope-swapped",
        "Swapping the intercept and the slope when substituting into the line.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-absolute-not-squared",
        "Adding the sizes of the residuals where their squares were asked for.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-one-residual-squared",
        "Squaring only the largest residual instead of all five.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-mean-instead-of-sum",
        "Dividing the sum of squared residuals by the number of observations.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-pattern-seen-in-random-scatter",
        "Seeing a pattern in residuals that are scattered at random, and rejecting a linear model that fits well.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-pattern-read-as-random-scatter",
        "Reading residuals that form a curve or fan as random scatter, and accepting a linear model that does not fit.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-fan-and-curve-confused",
        "Describing residuals that fan out as a curve, or a curved residual pattern as fanning.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-pattern-taken-as-good-fit",
        "Taking a clear pattern in the residuals as a sign that the linear model fits well, when a pattern is the sign that it does not.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-nonzero-residuals-mean-poor-fit",
        "Rejecting a linear model because the residuals are not all zero, although a good fit leaves small residuals with no pattern.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-r-squared-trusted",
        "Accepting a linear model because r² is high, without checking the residuals for a pattern.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-sum-zero-taken-as-linear",
        "Taking residuals that add to zero as evidence of a linear relationship, when the residuals of every least-squares line add to zero.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-large-residuals-mean-poor-fit",
        "Judging a linear model by the size of a few residuals instead of by whether they form a pattern.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-low-r-squared-rejects",
        "Rejecting a linear model because r² is low, although patternless residuals mean only that predictions are less precise.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-sign-mix-taken-as-pattern",
        "Treating a mix of positive and negative residuals as a pattern, when both signs are expected of a good fit.",
        area="statistics",
    ),
    CatalogueEntry(
        "mech-one-residual-left-out",
        "Leaving the residual just calculated out of the sum of squares.",
        area="statistics",
    ),
    CatalogueEntry(
        "binom-pascal-coefficients-omitted",
        "Expanding a power of a binomial with the powers of each term but no binomial coefficients: every coefficient "
        "is left as 1 (or as a bare power of the number).",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-middle-terms-dropped",
        "Writing (a + b)^n as a^n + b^n: leaving out every term between the first and the last.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-coefficients-from-previous-row",
        "Reading the coefficients from row n - 1 of Pascal's triangle instead of row n, so one coefficient is missing "
        "and the last term is left with a coefficient of 1.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-base-not-raised",
        "Multiplying each term by the number once instead of raising it to the power that matches its position: "
        "reading the term in b^3 as b times the binomial coefficient.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-base-times-exponent",
        "Multiplying the number by its exponent instead of raising it to that power: reading 3^2 as 3 x 2 in a term of "
        "the expansion.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-sign-not-alternating",
        "Expanding (a - b)^n with every term after the first negative, instead of letting the sign alternate with the "
        "power of the negative number.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-negative-sign-dropped",
        "Expanding (a - b)^n as if it were (a + b)^n: the minus sign is lost and every term is positive.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-power-of-base-missed",
        "Finding a coefficient in a binomial expansion with the right binomial coefficient but only the first power of "
        "the number instead of the power that matches the term.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-wrong-term-picked",
        "Giving the coefficient of a neighbouring term (x^1 or x^3) in the expansion instead of the term asked for.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-pascal-coefficient-misread",
        "Reading the wrong entry from the row of Pascal's triangle: taking 4 instead of 6 as the middle coefficient of "
        "the fourth row.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-constant-term-only",
        "Giving the sum of the coefficients of an expansion as its constant term alone.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-pascal-coefficients-only",
        "Giving the sum of the binomial coefficients, 2^n, as the sum of the coefficients of an expansion whose terms "
        "also carry powers of a number.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-permutation-for-combination",
        "Counting arrangements, n(n - 1)...(n - r + 1), where the binomial coefficient C(n, r) was needed.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-row-sum-given",
        "Giving the sum of row n of Pascal's triangle, 2^n, where a single binomial coefficient was asked for.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-numbers-omitted",
        "Giving the binomial coefficient alone as the coefficient of a term, without the powers of the numbers in the "
        "binomial.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-powers-swapped",
        "Attaching the exponents to the wrong numbers: using the power r on one number and n - r on the other the "
        "wrong way round.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-first-number-not-raised",
        "Raising the second number to its power but leaving the number in front of x as it is, so the power of that "
        "number is missing from the coefficient.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-second-number-not-raised",
        "Raising the number in front of x to its power but leaving the constant as it is, so the power of the "
        "constant is missing from the coefficient.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-first-number-only",
        "Finding the constant term of (ax + b/x)^n as C(n, n/2) a^n, ignoring that half the factors are b/x.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-multipliers-omitted",
        "Adding the two earlier coefficients without multiplying each by the number it comes with when multiplying a "
        "power of (px + q) by (px + q).",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-multipliers-swapped",
        "Multiplying the coefficient of x^r by q and the coefficient of x^(r+1) by p, the wrong way round, when "
        "multiplying a power of (px + q) by (px + q).",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-one-product-only",
        "Counting only one of the two ways a term of the next power arises when multiplying a power of (px + q) by "
        "(px + q).",
        area="binomial",
    ),
    CatalogueEntry(
        "recip-not-taken",
        "Giving the value of cos, sin or tan where sec, cosec or cot was asked for: forgetting to take the reciprocal.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "recip-sec-cosec-swapped",
        "Mixing up sec and cosec: taking sec as the reciprocal of sine, or cosec as the reciprocal of cosine.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "recip-cot-as-tan",
        "Reading cot as tan (or tan as cot): giving sin/cos where cos/sin was asked for.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "recip-cot-as-cosec",
        "Confusing cot with cosec: taking cot as the reciprocal of sine.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "trig-sign-wrong-quadrant",
        "Giving a negative value where the function is positive in the angle's quadrant.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "trig-quadrant-sign-ignored",
        "Giving the value of the reference angle without the negative sign the angle's quadrant requires.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "trig-wrong-reference-angle",
        "Using the complement of the reference angle: taking the value at 60 degrees where 30 degrees was needed, or "
        "the reverse.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "trig-sin-cos-swapped",
        "Giving the value of cos at the angle where sin was asked for, or the reverse.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "recip-reciprocal-given",
        "Giving the reciprocal function's value, sec for cos or cosec for sin, where cos or sin was asked for.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "recip-identity-root-not-taken",
        "Stopping at sec^2 (or tan^2, cosec^2, cot^2) from 1 + tan^2 = sec^2 and giving it as the answer, without "
        "taking the square root.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "recip-identity-tan-not-squared",
        "Adding 1 to tan (or cot) instead of to tan squared when using 1 + tan^2 = sec^2, so sec is found as 1 + tan.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "recip-identity-one-omitted",
        "Reading sec^2 as tan^2: leaving the 1 out of 1 + tan^2 = sec^2.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "recip-identity-sign-slip",
        "Misapplying the identity with the wrong sign, so sec^2 is taken as tan^2 - 1 or 1 - tan^2.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "recip-identity-given-repeated",
        "Giving the value that was stated in the question (sec or cosec) as the answer for tan or cot.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "recip-identity-factor-left",
        "Stopping part way through a simplification and leaving a squared function or a factor that should have "
        "cancelled.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "recip-identity-not-applied",
        "Solving an equation in sec^2 and tan^2 as if sec^2 were another tan^2, without using 1 + tan^2 = sec^2.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "recip-identity-difference-as-zero",
        "Taking sec^2 x - tan^2 x as 0 instead of 1, so the constant terms are left on their own.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "recip-tan-not-applied",
        "Giving cos instead of sin by forgetting to multiply by tan when finding sin from cos and tan.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "recip-tan-divided",
        "Dividing cos by tan instead of multiplying, when finding sin from cos and tan.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "trig-standard-angle-misread",
        "Reading the wrong standard angle from the value of tan or cot, such as 45 degrees where tan is sqrt(3), or "
        "30 degrees where tan is 1.",
        area="trigonometry",
    ),
    CatalogueEntry(
        "polar-moduli-added",
        "Adding the moduli instead of multiplying them when multiplying complex numbers in polar form.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "polar-moduli-subtracted",
        "Subtracting the moduli instead of dividing them when dividing complex numbers in polar form.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "polar-arguments-subtracted-on-multiplying",
        "Subtracting the arguments instead of adding them when multiplying complex numbers in polar form.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "polar-division-arguments-added",
        "Adding the arguments instead of subtracting them when dividing complex numbers in polar form.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "polar-argument-of-first-only",
        "Keeping the argument of the first number and ignoring the second, or the modulus of one number only.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "polar-arguments-subtracted-in-wrong-order",
        "Subtracting the arguments the wrong way round when dividing, so the argument has the wrong size.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "polar-division-as-multiplication",
        "Multiplying the moduli (or dividing nothing) where a quotient was needed.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "polar-angle-not-reduced",
        "Leaving the argument as 360 degrees or more (or negative) instead of bringing it back into the range 0 to 360.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "polar-angle-reduced-by-half-turn",
        "Reducing an argument of more than 360 degrees by 180 degrees instead of by a full turn.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "polar-argument-signs-mixed",
        "Subtracting the arguments of w and u from that of z, instead of adding the argument of the factor in the "
        "numerator.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "polar-modulus-of-first-only",
        "Giving the modulus of one of the numbers as the modulus of the product.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "demoivre-modulus-not-raised",
        "Applying De Moivre's theorem and leaving the modulus as it was, instead of raising it to the power n.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "demoivre-modulus-multiplied",
        "Multiplying the modulus by n instead of raising it to the power n.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "demoivre-argument-not-multiplied",
        "Raising the modulus to the power n but leaving the argument unchanged, instead of multiplying it by n.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "demoivre-negative-power-sign-lost",
        "Giving the argument of z^n for a negative n as if n were positive, so the argument has the wrong sign.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "demoivre-quadrant-ignored",
        "Finding the argument of a number from its reference angle alone, without placing it in the quadrant of the "
        "point.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "demoivre-modulus-root-not-taken",
        "Giving the sum of the squares of the real and imaginary parts as the modulus, without the square root.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "demoivre-modulus-is-one-component",
        "Taking the modulus to be one of the two parts of the number, or their sum, instead of the square root of "
        "the sum of their squares.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "demoivre-argument-from-wrong-ratio",
        "Finding the argument from the real part over the imaginary part, so the reference angle is the complement "
        "of the right one.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "demoivre-modulus-raised-the-wrong-way",
        "Raising n to the power of the modulus instead of the modulus to the power n.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "demoivre-conjugate-given",
        "Writing the power in Cartesian form with the sign of the imaginary part reversed.",
        area="complex-numbers",
    ),
    CatalogueEntry(
        "demoivre-sine-cosine-swapped",
        "Writing the power in Cartesian form with sine as the real part and cosine as the imaginary part.",
        area="complex-numbers",
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
    CatalogueEntry(
        "unif-pdf-height-width",
        "Giving the width b - a of the interval as the height of the uniform density instead of its reciprocal 1/(b - a).",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "unif-pdf-height-upper-bound",
        "Using 1/b for the height of the uniform density instead of 1/(b - a).",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "unif-continuous-discrete-confused",
        "Counting the end points as if the uniform distribution were discrete, dividing by b - a + 1 instead of b - a.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "unif-pdf-height-sum",
        "Using 1/(a + b) for the height of the uniform density instead of 1/(b - a).",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "unif-pdf-height-half-width",
        "Giving half the width of the interval as the height of the density.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "unif-prob-over-upper-bound",
        "Dividing the length of the interval by b rather than by the full width b - a.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "unif-prob-complement",
        "Giving the probability of the complement of the stated interval.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "unif-prob-cdf-not-difference",
        "Giving P(X < d) = (d - a)/(b - a) for P(c < X < d), forgetting to subtract the part below c.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "unif-prob-width-not-divided",
        "Giving the length d - c of the interval as the probability, without dividing by b - a.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "unif-mean-half-width",
        "Giving half the width, (b - a)/2, as the mean of a uniform distribution instead of the midpoint (a + b)/2.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "unif-mean-half-upper",
        "Giving b/2 as the mean of a uniform distribution, which is only right when a = 0.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "unif-mean-sum-not-halved",
        "Giving a + b as the mean of a uniform distribution, without halving.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "unif-variance-no-square",
        "Using (b - a)/12 for the variance of a uniform distribution, leaving out the square of the width.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "unif-variance-divisor-two",
        "Using (b - a)²/2 for the variance of a uniform distribution instead of (b - a)²/12.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "unif-variance-divisor-forgotten",
        "Giving (b - a)² as the variance of a uniform distribution, forgetting to divide by 12.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "unif-variance-sd-confused",
        "Giving the standard deviation of a uniform distribution where the variance was asked for.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "unif-variance-divisor-four",
        "Using (b - a)²/4 for the variance of a uniform distribution instead of (b - a)²/12.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "unif-percentile-offset-forgotten",
        "Giving q(b - a) as the q-th percentile of a uniform distribution, forgetting to add the lower end a.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "unif-percentile-from-top",
        "Measuring the percentile down from b instead of up from a.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "unif-percentile-fraction-of-upper",
        "Giving q × b as the q-th percentile of a uniform distribution, which is only right when a = 0.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "unif-percentile-complement",
        "Using the proportion 1 - q instead of q when finding a percentile of a uniform distribution.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "unif-conditional-ignored",
        "Ignoring the given condition and using the unconditional probability for a uniform distribution.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "unif-conditional-complement",
        "Giving the probability that X is at most c, given X > d, instead of the probability that it exceeds c.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "unif-conditional-numerator-width",
        "Giving (c - d)/(b - a) for a conditional probability, with the unconditional width in the denominator and the wrong length on top.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "unif-conditional-wrong-denominator",
        "Dividing by c - d instead of by the length b - d of the interval that the condition leaves.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "expo-rate-mean-confused",
        "Using the mean where the rate belongs, or the rate where the mean belongs: e^(-μt) in place of e^(-t/μ).",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "expo-tail-complement",
        "Giving P(X ≤ t) where P(X > t) was asked, or the reverse.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "expo-time-forgotten",
        "Leaving the time out of the exponent, so the result depends only on the rate.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "expo-linear-not-exponential",
        "Treating the exponential tail as linear: 1 - t/μ in place of e^(-t/μ).",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "expo-pdf-as-cdf",
        "Giving the density λe^(-λt) where the distribution function 1 - e^(-λt) was asked for.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "expo-density-factor-forgotten",
        "Giving e^(-λt) for the density, leaving out the factor λ.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "expo-cdf-as-pdf",
        "Giving the distribution function 1 - e^(-λt) where the density λe^(-λt) was asked for.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "expo-density-times-cdf",
        "Multiplying λ by 1 - e^(-λt) for the density instead of by e^(-λt).",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "expo-median-as-mean",
        "Giving the mean as the median of an exponential distribution, as if it were symmetric.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "expo-median-halved-mean",
        "Giving half the mean as the median of an exponential distribution.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "expo-log-base-ten",
        "Using log base 10 of 2 in place of the natural logarithm ln 2 when solving for the median.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "expo-percentile-wrong-tail",
        "Solving e^(-m/μ) = p instead of 1 - e^(-m/μ) = p, so the percentile uses the wrong tail.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "expo-percentile-linear",
        "Giving p × μ as a percentile of an exponential distribution, as if it were uniform.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "expo-percentile-sign",
        "Losing the minus sign when solving for a percentile of an exponential distribution, so the time is negative.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "expo-memoryless-ignored",
        "Giving the probability of lasting s + t from the start, ignoring that the item has already lasted s.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "expo-memoryless-wrong-time",
        "Using the elapsed time s in place of the extra time t in the exponent.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "expo-memoryless-ratio-inverted",
        "Dividing the probabilities the wrong way round in the conditional probability, which gives a value above 1.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "expo-interval-width-only",
        "Giving P(X > t - s) for P(s < X < t): applying the memoryless property to an interval, which it does not describe.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "expo-interval-sign-reversed",
        "Subtracting the tail probabilities in the wrong order, so P(s < X < t) comes out negative.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "expo-tail-only",
        "Giving P(X > s) for P(s < X < t), forgetting to subtract P(X > t).",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "expo-interval-density-times-width",
        "Approximating P(s < X < t) by the density at s times the width of the interval.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "expo-rate-reciprocal-squared",
        "Giving 1/μ² as the rate of an exponential distribution with mean μ.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "expo-rate-from-median",
        "Using ln 2 divided by the mean as the rate.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "expo-rate-half-mean",
        "Giving half the mean as the rate.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "expo-pdf-as-tail",
        "Giving the density value λe^(-λt) where the tail probability e^(-λt) was asked for.",
        area="statistics-probability",
    ),
    CatalogueEntry(
        "ztest-se-uses-sample-proportion",
        "Building the standard error from the sample proportion, √(p̂(1 − p̂)/n), when a test must use the hypothesised proportion p0.",
        area="statistics",
    ),
    CatalogueEntry(
        "ztest-se-no-root",
        "Forgetting the square root: giving p0(1 − p0)/n as the standard error of a sample proportion.",
        area="statistics",
    ),
    CatalogueEntry(
        "ztest-se-complement-forgotten",
        "Using √(p0/n) as the standard error, leaving out the factor 1 − p0.",
        area="statistics",
    ),
    CatalogueEntry(
        "ztest-se-divides-by-n",
        "Dividing √(p0(1 − p0)) by n instead of by √n.",
        area="statistics",
    ),
    CatalogueEntry(
        "ztest-se-root-n-forgotten",
        "Giving √(p0(1 − p0)) as the standard error without dividing by √n.",
        area="statistics",
    ),
    CatalogueEntry(
        "ztest-sign-reversed",
        "Subtracting in the wrong order, p0 − p̂, so the test statistic has the wrong sign.",
        area="statistics",
    ),
    CatalogueEntry(
        "ztest-count-used-as-proportion",
        "Using the number of successes where the sample proportion belongs, so the difference from p0 is hundreds of times too large.",
        area="statistics",
    ),
    CatalogueEntry(
        "ztest-percent-not-converted",
        "Using a sample percentage such as 41.5 where the proportion 0.415 belongs.",
        area="statistics",
    ),
    CatalogueEntry(
        "ztest-hypothesised-proportion-ignored",
        "Giving the sample proportion itself when asked for its difference from the hypothesised proportion.",
        area="statistics",
    ),
    CatalogueEntry(
        "ztest-sample-proportion-wrong-denominator",
        "Dividing the number of successes by the successes plus the sample size instead of by the sample size.",
        area="statistics",
    ),
    CatalogueEntry(
        "ztest-pvalue-doubling-forgotten",
        "Giving a single tail area as the p-value of a two-sided test.",
        area="statistics",
    ),
    CatalogueEntry(
        "ztest-pvalue-cdf-as-pvalue",
        "Giving the area to the left of |z| as the p-value instead of the area in the tail.",
        area="statistics",
    ),
    CatalogueEntry(
        "ztest-pvalue-complement",
        "Subtracting the two-sided p-value from 1.",
        area="statistics",
    ),
    CatalogueEntry(
        "ztest-pvalue-wrong-tail",
        "Taking the tail on the wrong side of z for the alternative hypothesis: the area to the left for H1: p > p0, or to the right for H1: p < p0.",
        area="statistics",
    ),
    CatalogueEntry(
        "ztest-pvalue-doubled-for-one-sided",
        "Doubling the tail area for a one-sided alternative hypothesis.",
        area="statistics",
    ),
    CatalogueEntry(
        "ztest-pvalue-sign-ignored",
        "Using the tail beyond |z| whatever the sign of z, so a statistic on the wrong side of p0 for a one-sided alternative gives a small p-value instead of a large one.",
        area="statistics",
    ),
    CatalogueEntry(
        "ztest-cutoff-wrong-side",
        "Subtracting z* × SE from p0 for the upper rejection boundary of a test of H1: p > p0.",
        area="statistics",
    ),
    CatalogueEntry(
        "ztest-cutoff-margin-as-bound",
        "Giving the margin z* × SE as the boundary instead of adding it to p0.",
        area="statistics",
    ),
    CatalogueEntry(
        "ztest-wrong-critical-value",
        "Using a critical value that does not match the stated one, such as the two-sided 1.96 where the question gives the one-sided 1.645.",
        area="statistics",
    ),
    CatalogueEntry(
        "ztest-count-rounded-down",
        "Rounding the boundary count down, so the count named does not actually reject H0.",
        area="statistics",
    ),
    CatalogueEntry(
        "ztest-count-rounded-up-twice",
        "Adding one to the smallest whole number above the boundary count.",
        area="statistics",
    ),
    CatalogueEntry(
        "ztest-conclusion-decision-reversed",
        "Reading the comparison with 0.05 the wrong way round: rejecting H0 for a large p-value or not rejecting it for a small one.",
        area="statistics",
    ),
    CatalogueEntry(
        "ztest-conclusion-sample-differs",
        "Rejecting H0 just because the sample proportion is not exactly p0, without looking at the p-value.",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-pooled-sd-as-standard-error",
        "Dividing the difference in means by the pooled standard deviation alone, without scaling it by √(1/n1 + 1/n2).",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-welch-pooled-confused",
        "Using the pooled standard error when the question asks for the unequal-variance (Welch) statistic, or the reverse.",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-paired-sd-as-se",
        "Dividing the mean difference by the standard deviation of the differences without dividing that by √n.",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-paired-se-root-df",
        "Dividing the standard deviation of the differences by √(n - 1) instead of √n.",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-se-sds-added",
        "Adding the two standard errors s1/√n1 + s2/√n2 instead of adding the variances s1²/n1 + s2²/n2 and then taking the square root.",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-se-no-root",
        "Forgetting the final square root: giving s1²/n1 + s2²/n2 as the standard error of the difference.",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-se-single-sample",
        "Using only the first sample, s1/√n1, as the standard error of a difference between two sample means.",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-se-sd-over-n",
        "Dividing each variance by n² (each standard deviation by n) instead of dividing each variance by n.",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-se-total-n",
        "Pooling the two samples into one: √((s1² + s2²)/(n1 + n2)), which ignores that each sample has its own size.",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-diff-over-sds",
        "Dividing the difference in means by √(s1² + s2²), forgetting to divide each variance by its sample size.",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-sign-reversed",
        "Subtracting the means (or the paired values) in the wrong order, so the statistic has the wrong sign.",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-pooled-simple-average",
        "Averaging the two sample variances, (s1² + s2²)/2, without weighting by the degrees of freedom.",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-pooled-weights-sds",
        "Weighting the standard deviations by their degrees of freedom instead of the variances, so the pooled value is a standard deviation.",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-pooled-divisor-total-n",
        "Dividing the pooled sum of squares by n1 + n2 instead of by the degrees of freedom n1 + n2 - 2.",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-pooled-weights-by-n",
        "Weighting each variance by its sample size n instead of by n - 1.",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-pooled-sd-as-variance",
        "Giving the pooled standard deviation where the pooled variance was asked for (or the reverse).",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-pooled-se-one-sum",
        "Using √(1/(n1 + n2)) in place of √(1/n1 + 1/n2) when scaling the pooled standard deviation.",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-pooled-se-unrooted",
        "Leaving the square root off the factor in sp·√(1/n1 + 1/n2).",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-pooled-se-single-n",
        "Scaling the pooled standard deviation by one sample size only, sp/√n1.",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-pooled-variance-as-sd",
        "Putting the pooled variance where the pooled standard deviation belongs in the standard error.",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-paired-sd-divisor-n",
        "Finding the standard deviation of the paired differences with divisor n instead of n - 1.",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-paired-uncentred-spread",
        "Finding the spread of the differences from the sum of d² without subtracting the mean first.",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-paired-se-over-n",
        "Dividing the standard deviation of the differences by n instead of √n to get the standard error.",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-paired-variance-as-sd",
        "Using the variance of the differences where their standard deviation belongs.",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-ci-other-bound",
        "Giving the opposite end of the confidence interval from the one asked for.",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-ci-margin-as-bound",
        "Giving the margin of error as if it were the confidence limit instead of adding it to (or subtracting it from) the estimate.",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-ci-normal-critical",
        "Using the normal critical value 1.96 in place of the t critical value for the stated degrees of freedom.",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-conclusion-decision-reversed",
        "Reading the comparison with the critical value the wrong way round: not rejecting H0 when the statistic is beyond the critical value, or rejecting when it is not.",
        area="statistics",
    ),
    CatalogueEntry(
        "ttest-conclusion-means-differ",
        "Rejecting H0 just because the two sample means are not equal, without comparing the statistic with the critical value.",
        area="statistics",
    ),
    CatalogueEntry(
        "volume-extra-dot-term",
        "Adding a stray dot product to a volume: |a·(b × c)| + |a·b|.",
        area="vectors",
    ),
    CatalogueEntry(
        "distance-signed-left-negative",
        "Leaving the distance from a point to a plane negative: the signed value (n·P - d)/|n| without taking the absolute value.",
        area="vectors",
    ),
    CatalogueEntry(
        "distance-plane-denominator-not-rooted",
        "Dividing by a² + b² + c² instead of by its square root √(a² + b² + c²) in a distance to a plane.",
        area="vectors",
    ),
    CatalogueEntry(
        "distance-plane-constant-sign",
        "Adding the constant of the plane's equation to n·P instead of subtracting it.",
        area="vectors",
    ),
    CatalogueEntry(
        "distance-plane-not-normalised",
        "Giving |n·P - d| as the distance without dividing by the length of the normal vector.",
        area="vectors",
    ),
    CatalogueEntry(
        "distance-plane-denominator-sum-of-coefficients",
        "Dividing by |a| + |b| + |c| instead of √(a² + b² + c²) in a distance to a plane.",
        area="vectors",
    ),
    CatalogueEntry(
        "distance-line-dot-product-used",
        "Using the dot product instead of the cross product, so the length along the line is found instead of the distance from it.",
        area="vectors",
    ),
    CatalogueEntry(
        "distance-line-not-normalised",
        "Giving the size of the cross product AP × u without dividing by the length of the direction vector u.",
        area="vectors",
    ),
    CatalogueEntry(
        "distance-line-point-not-shifted",
        "Crossing the position vector of P with u instead of the vector from a point on the line to P.",
        area="vectors",
    ),
    CatalogueEntry(
        "distance-line-to-base-point",
        "Giving the distance from P to the line's base point A, not the shortest distance to the line.",
        area="vectors",
    ),
    CatalogueEntry(
        "volume-negative-not-absolute",
        "Leaving a volume negative: giving the scalar triple product itself without its absolute value.",
        area="vectors",
    ),
    CatalogueEntry(
        "determinant-expansion-signs-forgotten",
        "Expanding a 3×3 determinant with every term added, missing the alternating signs of the cofactor expansion.",
        area="vectors",
    ),
    CatalogueEntry(
        "volume-as-base-area-only",
        "Giving the area of the base, |b × c|, as the volume of the parallelepiped.",
        area="vectors",
    ),
    CatalogueEntry(
        "parallelepiped-as-tetrahedron",
        "Treating a parallelepiped's volume as the volume of the tetrahedron on the same edges (a sixth of it), or the reverse.",
        area="vectors",
    ),
    CatalogueEntry(
        "skew-distance-not-normalised",
        "Giving |(A₂ - A₁)·(u₁ × u₂)| as the distance between skew lines without dividing by |u₁ × u₂|.",
        area="vectors",
    ),
    CatalogueEntry(
        "skew-distance-one-line-only",
        "Using the point-to-line formula with one of the skew lines, ignoring the direction of the other.",
        area="vectors",
    ),
    CatalogueEntry(
        "skew-distance-between-base-points",
        "Giving the distance between the two lines' base points as the distance between the lines.",
        area="vectors",
    ),
    CatalogueEntry(
        "tetrahedron-volume-factor-forgotten",
        "Giving |AB · (AC × AD)| as the volume of a tetrahedron without dividing by 6.",
        area="vectors",
    ),
    CatalogueEntry(
        "tetrahedron-volume-third-of-parallelepiped",
        "Taking a third of the parallelepiped's volume for the tetrahedron instead of a sixth.",
        area="vectors",
    ),
    CatalogueEntry(
        "tetrahedron-volume-half-of-parallelepiped",
        "Taking half of the parallelepiped's volume for the tetrahedron instead of a sixth.",
        area="vectors",
    ),
    CatalogueEntry(
        "height-pyramid-factor-forgotten",
        "Dividing the tetrahedron's volume by the base area without the factor 3 in h = 3V/A.",
        area="vectors",
    ),
    CatalogueEntry(
        "height-pyramid-factor-applied-twice",
        "Dividing by 3 once too often when finding the height from |AD·(AB × AC)| and |AB × AC|.",
        area="vectors",
    ),
    CatalogueEntry(
        "triple-product-order-reversed",
        "Taking the scalar triple product with two of the vectors in the opposite order, which gives the negative.",
        area="vectors",
    ),
    CatalogueEntry(
        "triple-product-extra-dot-term",
        "Adding a stray dot product a·b to a scalar triple product.",
        area="vectors",
    ),
    CatalogueEntry(
        "triple-product-dots-multiplied",
        "Multiplying dot products (a·b)(b·c) in place of forming a·(b × c).",
        area="vectors",
    ),
    CatalogueEntry(
        "volume-doubled",
        "Doubling a volume: counting the parallelepiped's volume twice.",
        area="vectors",
    ),
    CatalogueEntry(
        "area-from-dot-product",
        "Using the dot product |a·b| for the area of the parallelogram on a and b, where the cross product |a × b| belongs.",
        area="vectors",
    ),
    CatalogueEntry(
        "area-as-product-of-lengths",
        "Giving |a||b| as the area of the parallelogram, which ignores the angle between the edges.",
        area="vectors",
    ),
    CatalogueEntry(
        "area-cross-magnitude-squared",
        "Giving the sum of squares of the cross product's components, |a × b|², where its length |a × b| was needed.",
        area="vectors",
    ),
    CatalogueEntry(
        "area-parallelogram-as-triangle",
        "Halving |a × b| for a parallelogram, which is the area of the triangle on a and b.",
        area="vectors",
    ),
    CatalogueEntry(
        "height-volume-multiplied-by-area",
        "Multiplying the volume by the base area instead of dividing, when finding the height.",
        area="vectors",
    ),
    CatalogueEntry(
        "height-ratio-inverted",
        "Dividing base area by volume instead of volume by base area, when finding the height.",
        area="vectors",
    ),
    CatalogueEntry(
        "height-over-the-wrong-face",
        "Dividing the volume by the area of a different face from the one the height is measured from.",
        area="vectors",
    ),
    CatalogueEntry(
        "height-pyramid-factor-applied-to-prism",
        "Using the pyramid formula h = 3V/A for a prism or parallelepiped, whose height is V/A.",
        area="vectors",
    ),
    CatalogueEntry(
        "plane-constant-sign",
        "Giving the constant of the plane with the wrong sign, as if the normal had been found in the opposite direction.",
        area="vectors",
    ),
    CatalogueEntry(
        "cross-product-middle-sign-slip",
        "Forgetting the minus sign on the middle component of a cross product.",
        area="vectors",
    ),
    CatalogueEntry(
        "cross-product-componentwise",
        "Multiplying two vectors component by component and calling the result the cross product.",
        area="vectors",
    ),
    CatalogueEntry(
        "cross-product-components-swapped",
        "Writing the components of a cross product in the wrong order, such as (a, c, b) for (a, b, c).",
        area="vectors",
    ),
    CatalogueEntry(
        "cross-product-reversed-order",
        "Taking the cross product in the wrong order, so the normal vector points the opposite way.",
        area="vectors",
    ),
    CatalogueEntry(
        "plane-constant-sum-of-points",
        "Dotting the normal with the sum of the three points instead of with one of them.",
        area="vectors",
    ),
    CatalogueEntry(
        "line-plane-sign-in-numerator",
        "Adding n·A to the plane's constant instead of subtracting it when solving for the parameter where a line meets a plane.",
        area="vectors",
    ),
    CatalogueEntry(
        "line-plane-not-divided-by-direction-dot",
        "Stopping at k - n·A without dividing by n·u, the rate at which the line moves through the plane's values.",
        area="vectors",
    ),
    CatalogueEntry(
        "line-plane-ratio-inverted",
        "Dividing in the wrong order: (n·u)/(k - n·A) where (k - n·A)/(n·u) is needed.",
        area="vectors",
    ),
    CatalogueEntry(
        "line-plane-point-term-forgotten",
        "Leaving the line's starting point out of the substitution into the plane's equation: solving k = (n·u)t.",
        area="vectors",
    ),
    CatalogueEntry(
        "angle-between-planes-supplement",
        "Giving the obtuse angle 180° - θ where the acute angle between the planes or the line and the plane was asked for.",
        area="vectors",
    ),
    CatalogueEntry(
        "angle-between-planes-sine-used",
        "Using sine where cosine belongs (or the reverse), which gives 90° - θ for the angle between the normals.",
        area="vectors",
    ),
    CatalogueEntry(
        "angle-between-planes-constants-included",
        "Including the constants of the two plane equations in the vectors whose angle is found, instead of only the normals.",
        area="vectors",
    ),
    CatalogueEntry(
        "angle-given-in-radians",
        "Leaving an angle in radians when degrees were asked for.",
        area="vectors",
    ),
    CatalogueEntry(
        "line-plane-angle-cosine-used",
        "Finding the angle between the line and the plane's normal (using cosine) and giving it as the angle between the line and the plane.",
        area="vectors",
    ),
    CatalogueEntry(
        "line-plane-angle-tangent-used",
        "Using the inverse tangent of |n·u|/(|n||u|) in place of the inverse sine for the angle between a line and a plane.",
        area="vectors",
    ),
    CatalogueEntry(
        "polar-jacobian-omitted",
        "Integrating in polar coordinates without the extra factor r from dA = r dr dθ.",
        area="multivariable",
    ),
    CatalogueEntry(
        "polar-angular-factor-ignored",
        "Replacing the integral of the angular factor (cos θ, sin θ, ...) by the width of the angle range, as if the integrand did not depend on the angle.",
        area="multivariable",
    ),
    CatalogueEntry(
        "polar-angle-factor-forgotten",
        "Integrating over r but never multiplying by the integral over the angle.",
        area="multivariable",
    ),
    CatalogueEntry(
        "polar-angle-halved",
        "Using an angle range of half the size needed: π for a whole disc, π/2 for a half disc, or π/4 for a quarter disc.",
        area="multivariable",
    ),
    CatalogueEntry(
        "polar-inner-radius-ignored",
        "Using 0 as the lower radius for a ring, so the hole in the middle is included.",
        area="multivariable",
    ),
    CatalogueEntry(
        "polar-ring-difference-before-power",
        "Raising the difference of the radii to the power, (b - a)^n, instead of taking the difference of the powers, b^n - a^n.",
        area="multivariable",
    ),
    CatalogueEntry(
        "polar-radius-limit-squared",
        "Using x² + y² = R² as r² = R, so that the upper limit of r is R² instead of R.",
        area="multivariable",
    ),
    CatalogueEntry(
        "integrand-boundary-value-times-area",
        "Multiplying the area of the region by the value of the integrand on its outer edge instead of integrating.",
        area="multivariable",
    ),
    CatalogueEntry(
        "polar-quadrant-angle-forgotten",
        "Using the whole circle, 2π, as the angle range for a region that is only a quarter of it.",
        area="multivariable",
    ),
    CatalogueEntry(
        "substitution-factor-half-dropped",
        "Integrating r·e^(-r²) as e^(-r²) without the factor 1/2 that the substitution u = r² gives.",
        area="multivariable",
    ),
    CatalogueEntry(
        "polar-gaussian-limit-not-squared",
        "Substituting r = R instead of r² = R² into the antiderivative of r·e^(-r²).",
        area="multivariable",
    ),
    CatalogueEntry(
        "polar-gaussian-complement-sign",
        "A sign slip in e^(-R²) in the value (1 - e^(-R²))/2: adding it, or subtracting in the other order.",
        area="multivariable",
    ),
    CatalogueEntry(
        "polar-gaussian-bound-ignored",
        "Taking the radius of the region to infinity so that the e^(-R²) term disappears.",
        area="multivariable",
    ),
    CatalogueEntry(
        "polar-bound-radius-from-height",
        "Using the height of the surface, instead of the radius where it meets the plane z = 0, as the outer radius of the region.",
        area="multivariable",
    ),
    CatalogueEntry(
        "volume-as-base-area-times-peak-height",
        "Giving the volume as the area of the base times the greatest height of the solid.",
        area="multivariable",
    ),
    CatalogueEntry(
        "polar-substitution-x-as-r-squared",
        "Writing x as r² in polar coordinates instead of r cos θ.",
        area="multivariable",
    ),
    CatalogueEntry(
        "centroid-not-divided-by-area",
        "Giving the first moment ∬ x dA as the centroid coordinate, without dividing by the area.",
        area="multivariable",
    ),
    CatalogueEntry(
        "mean-position-ratio-inverted",
        "Dividing the area by the first moment instead of the first moment by the area.",
        area="multivariable",
    ),
    CatalogueEntry(
        "mean-position-divided-by-pi",
        "Dividing a first moment by a full-circle area where the area of the plate itself belongs.",
        area="multivariable",
    ),
    CatalogueEntry(
        "centroid-distance-uses-one-coordinate",
        "Giving the x-coordinate of a centroid where the distance from the origin was asked for.",
        area="multivariable",
    ),
    CatalogueEntry(
        "centroid-distance-coordinates-added",
        "Adding the two coordinates of the centroid instead of using Pythagoras' theorem for the distance.",
        area="multivariable",
    ),
    CatalogueEntry(
        "centroid-distance-root-forgotten",
        "Giving x² + y² for a distance without taking the square root.",
        area="multivariable",
    ),
    CatalogueEntry(
        "centroid-distance-divided-by-root-two",
        "Dividing the x-coordinate by √2 instead of multiplying, when the distance is the hypotenuse of two equal sides.",
        area="multivariable",
    ),
    CatalogueEntry(
        "chain-rule-inner-derivatives-omitted",
        "Adding the two partial derivatives of z without multiplying each by the derivative of its variable with respect to the parameter.",
        area="multivariable",
    ),
    CatalogueEntry(
        "chain-rule-one-term-only",
        "Using only the x term of the chain rule, fx·(dx/dt), and leaving out the y term.",
        area="multivariable",
    ),
    CatalogueEntry(
        "chain-rule-partials-swapped",
        "Pairing each partial derivative with the other variable's rate of change: fy with dx/dt and fx with dy/dt.",
        area="multivariable",
    ),
    CatalogueEntry(
        "chain-rule-partials-evaluated-at-parameter",
        "Evaluating the partial derivatives at the parameter value (t, t) or (u, v) instead of at the point (x, y) that the parameter gives.",
        area="multivariable",
    ),
    CatalogueEntry(
        "chain-rule-wrong-variable-differentiated",
        "Differentiating the inner functions with respect to the other new variable, so the answer is for dz/dv where dz/du was asked (or the reverse).",
        area="multivariable",
    ),
    CatalogueEntry(
        "partial-multiplied-by-path-derivative",
        "Giving fx·(dx/dt) where the partial derivative fx alone was asked for.",
        area="multivariable",
    ),
    CatalogueEntry(
        "chain-rule-point-coordinates-swapped",
        "Substituting the position into a partial derivative with x and y the wrong way round.",
        area="multivariable",
    ),
    CatalogueEntry(
        "second-derivative-cross-term-dropped",
        "Leaving out the mixed term 2·fxy·x'·y' in the second derivative along a path.",
        area="multivariable",
    ),
    CatalogueEntry(
        "second-derivative-cross-term-not-doubled",
        "Including the mixed term of the second derivative along a path as fxy·x'·y' instead of 2·fxy·x'·y'.",
        area="multivariable",
    ),
    CatalogueEntry(
        "second-derivative-inner-derivatives-not-squared",
        "Multiplying fxx and fyy by x' and y' instead of by their squares in the second derivative along a path.",
        area="multivariable",
    ),
    CatalogueEntry(
        "second-derivative-stopped-at-first",
        "Giving the first derivative dz/dt where the second derivative was asked for.",
        area="multivariable",
    ),
    CatalogueEntry(
        "rate-divided-by-interval",
        "Dividing a rate by the time interval where the change is the rate multiplied by that interval.",
        area="multivariable",
    ),
    CatalogueEntry(
        "rate-times-time-not-interval",
        "Multiplying a rate of change by the time t at which it was found instead of by the length of the time interval.",
        area="multivariable",
    ),
    CatalogueEntry(
        "rate-plus-interval",
        "Adding the length of the time interval to the rate instead of multiplying the rate by the interval.",
        area="multivariable",
    ),
    CatalogueEntry(
        "rate-times-final-time",
        "Multiplying a rate of change by the final time t + Δt instead of by the length Δt of the interval.",
        area="multivariable",
    ),
    CatalogueEntry(
        "linear-approx-omits-base-value",
        "Giving only the change fx·h + fy·k predicted by the tangent plane, without adding the value f(x0, y0) at the point.",
        area="multivariable",
    ),
    CatalogueEntry(
        "tangent-plane-partials-swapped",
        "Using f_y where f_x belongs and f_x where f_y belongs in the tangent plane or the linear approximation.",
        area="multivariable",
    ),
    CatalogueEntry(
        "linear-approx-exact-value-given",
        "Evaluating f itself at the nearby point instead of using the tangent plane, so the exact value is given where the linear estimate was asked for.",
        area="multivariable",
    ),
    CatalogueEntry(
        "linear-approx-one-variable-only",
        "Changing only x in the linear approximation and leaving out the f_y·k term for the change in y.",
        area="multivariable",
    ),
    CatalogueEntry(
        "linear-approx-slope-times-coordinate",
        "Multiplying each slope by the coordinate of the point instead of by the small step in that variable.",
        area="multivariable",
    ),
    CatalogueEntry(
        "tangent-plane-shift-forgotten",
        "Reading the constant term of z = Ax + By + C as f(x0, y0), forgetting that the plane is written about the point so A·x0 + B·y0 must be subtracted.",
        area="multivariable",
    ),
    CatalogueEntry(
        "tangent-plane-shift-sign",
        "Adding A·x0 + B·y0 to f(x0, y0) instead of subtracting it when expanding the point form of the tangent plane.",
        area="multivariable",
    ),
    CatalogueEntry(
        "tangent-plane-shift-one-variable",
        "Subtracting the A·x0 term but leaving out B·y0 when expanding the point form of the tangent plane.",
        area="multivariable",
    ),
    CatalogueEntry(
        "tangent-plane-shift-uses-slopes-only",
        "Subtracting the slopes A and B themselves instead of A·x0 and B·y0 when expanding the point form of the tangent plane.",
        area="multivariable",
    ),
    CatalogueEntry(
        "tangent-plane-constant-negated",
        "A sign error across the answer: the constant term or intercept has the opposite sign to the correct one.",
        area="multivariable",
    ),
    CatalogueEntry(
        "radial-slope-taken-as-one",
        "Taking the partial derivatives of the square-root surface as 1, forgetting the factor x/√(x² + y²) from the chain rule.",
        area="multivariable",
    ),
    CatalogueEntry(
        "radial-slope-denominator-squared",
        "Dividing by x² + y² instead of √(x² + y²) when differentiating a square root of a sum of squares.",
        area="multivariable",
    ),
    CatalogueEntry(
        "implicit-plane-coefficients-dropped",
        "Writing the tangent plane to an implicit surface as x0·x + y0·y + z0·z = k, forgetting the coefficients of the surface's equation.",
        area="multivariable",
    ),
    CatalogueEntry(
        "implicit-plane-constant-halved",
        "Setting the gradient dotted with (x, y, z) equal to k, which leaves out the factor 2 that the constant needs once the gradient 2a·x0, 2b·y0, 2c·z0 is used.",
        area="multivariable",
    ),
    CatalogueEntry(
        "tangent-plane-wrong-axis-intercept",
        "Finding where the tangent plane meets a different axis from the one asked for.",
        area="multivariable",
    ),
    CatalogueEntry(
        "partial-power-rule-multiplier-dropped",
        "Differentiating a power by lowering its exponent but not multiplying by the old exponent: the derivative of x³ taken as x².",
        area="multivariable",
    ),
    CatalogueEntry(
        "tangent-plane-point-coordinates-swapped",
        "Substituting the point's coordinates into the partial derivative the wrong way round: x0 for y and y0 for x.",
        area="multivariable",
    ),
    CatalogueEntry(
        "partial-sign-slip",
        "A sign slip in a partial derivative, giving the negative of its value.",
        area="multivariable",
    ),
    CatalogueEntry(
        "cx-arg-ignores-quadrant",
        "Giving the angle between the point and the real axis, without placing it in the right quadrant.",
        area="complex",
    ),
    CatalogueEntry(
        "cx-arg-conjugate-sign",
        "Getting the sign of the argument wrong, so the answer is the argument of the conjugate.",
        area="complex",
    ),
    CatalogueEntry(
        "cx-arg-swaps-real-imag",
        "Using the real part over the imaginary part inside the inverse tangent, so the angle is measured from the "
        "imaginary axis.",
        area="complex",
    ),
    CatalogueEntry(
        "cx-arg-not-reduced",
        "Leaving the argument outside the stated range (-π, π] instead of adding or subtracting a full turn.",
        area="complex",
    ),
    CatalogueEntry(
        "cx-arg-quarter-turn-off",
        "Measuring the angle from the wrong axis, so the argument is a quarter turn out.",
        area="complex",
    ),
    CatalogueEntry(
        "cx-arg-ignores-second-factor",
        "Giving the argument of one of the numbers instead of the argument of the combination of both.",
        area="complex",
    ),
    CatalogueEntry(
        "cx-arg-power-ignored",
        "Not applying the power to the argument: using θ where nθ is needed.",
        area="complex",
    ),
    CatalogueEntry(
        "cx-arg-sign-after-power",
        "Taking the argument of a power to be π minus the right angle, or its mirror image.",
        area="complex",
    ),
    CatalogueEntry(
        "cx-product-subtracts-args",
        "Subtracting the arguments when multiplying: the arguments add under multiplication.",
        area="complex",
    ),
    CatalogueEntry(
        "cx-quotient-adds-args",
        "Adding the arguments when dividing: the arguments subtract under division.",
        area="complex",
    ),
    CatalogueEntry(
        "cx-root-skips-2pi-k",
        "Giving the principal root and not adding 2π k before dividing by n to find the others.",
        area="complex",
    ),
    CatalogueEntry(
        "cx-root-wrong-k",
        "Choosing the wrong root: taking k = 1, or the largest argument, instead of the root asked for.",
        area="complex",
    ),
    CatalogueEntry(
        "cx-root-divides-only-2k",
        "Dividing only the argument of w by n, or only the 2π k term, instead of dividing their sum.",
        area="complex",
    ),
    CatalogueEntry(
        "cx-root-power-instead-of-root",
        "Multiplying the argument of w by n, as if finding a power, instead of dividing by n.",
        area="complex",
    ),
    CatalogueEntry(
        "cx-modulus-forgot-root",
        "Leaving out the square root in the modulus: giving a² + b² for |z|.",
        area="complex",
    ),
    CatalogueEntry(
        "cx-modulus-real-part-only",
        "Using only the real part or one coefficient as the modulus.",
        area="complex",
    ),
    CatalogueEntry(
        "cx-modulus-drops-scale",
        "Forgetting the common factor when finding the modulus of a multiple of a number.",
        area="complex",
    ),
    CatalogueEntry(
        "cx-power-sign",
        "Getting the sign of a real power of z wrong: cos(nθ) is -1 when nθ is an odd multiple of π.",
        area="complex",
    ),
    CatalogueEntry(
        "cx-modulus-ignores-imaginary-part",
        "Treating a purely imaginary number as having modulus 0, because it has no real part.",
        area="complex",
    ),
    CatalogueEntry(
        "cx-modulus-negative",
        "Giving a negative modulus: the modulus is a distance, never negative.",
        area="complex",
    ),
    CatalogueEntry(
        "cx-power-divides-arg",
        "Dividing the argument by n instead of multiplying it when finding a power.",
        area="complex",
    ),
    CatalogueEntry(
        "cx-power-exponent-off-by-one",
        "Raising the modulus to the power n - 1 instead of n.",
        area="complex",
    ),
    CatalogueEntry(
        "cx-power-as-multiplication",
        "Multiplying the modulus by n instead of raising it to the power n.",
        area="complex",
    ),
    CatalogueEntry(
        "cpoly-sum-forgets-conjugate",
        "Counting only one non-real root in the sum of the roots, forgetting that its conjugate is a root too.",
        area="complex-roots",
    ),
    CatalogueEntry(
        "cpoly-vieta-sign-slip",
        "Taking the sum of the roots to be the coefficient itself, not its negative.",
        area="complex-roots",
    ),
    CatalogueEntry(
        "cpoly-ignores-complex-pair",
        "Taking the real root to be minus the coefficient, as if the non-real roots contributed nothing to the sum.",
        area="complex-roots",
    ),
    CatalogueEntry(
        "cpoly-product-sign-slip",
        "Taking a product of roots to be the constant term, forgetting the sign that Vieta's formulas give.",
        area="complex-roots",
    ),
    CatalogueEntry(
        "cpoly-product-not-divided",
        "Using the whole constant term as the product of the real roots, without dividing out the product of the non-real pair.",
        area="complex-roots",
    ),
    CatalogueEntry(
        "cpoly-modulus-imag-part-only",
        "Using q² in place of p² + q² for the product of a conjugate pair of roots p ± qi.",
        area="complex-roots",
    ),
    CatalogueEntry(
        "cpoly-modulus-real-part-only",
        "Using p² in place of p² + q² for the product of a conjugate pair of roots p ± qi.",
        area="complex-roots",
    ),
    CatalogueEntry(
        "cpoly-sum-instead-of-product",
        "Giving the sum of the real roots where their product was asked for.",
        area="complex-roots",
    ),
    CatalogueEntry(
        "cpoly-sumsq-forgot-cross-term",
        "Giving the square of the sum of two roots as the sum of their squares, leaving out the cross term.",
        area="complex-roots",
    ),
    CatalogueEntry(
        "cpoly-sumsq-cross-term-sign",
        "Adding twice the product of the roots to the square of their sum, where it must be subtracted.",
        area="complex-roots",
    ),
    CatalogueEntry(
        "cpoly-sumsq-cross-term-not-doubled",
        "Subtracting the product of the roots once, not twice, from the square of their sum.",
        area="complex-roots",
    ),
    CatalogueEntry(
        "cpoly-sum-instead-of-sumsq",
        "Giving the sum of the real roots where the sum of their squares was asked for.",
        area="complex-roots",
    ),
    CatalogueEntry(
        "cpoly-pairwise-sum-missing-term",
        "Leaving a term out of the sum of the products of pairs of roots: the pair of conjugates, or the real root with each of them.",
        area="complex-roots",
    ),
    CatalogueEntry(
        "cpoly-pairwise-sign-slip",
        "Subtracting where the real root's products with the conjugate pair must be added.",
        area="complex-roots",
    ),
    CatalogueEntry(
        "cpoly-wrong-real-root",
        "Giving the smaller real root where the largest was asked for.",
        area="complex-roots",
    ),
    CatalogueEntry(
        "cpoly-root-sign-slip",
        "Giving the negative of a root, by moving the sign the wrong way when solving the quadratic factor.",
        area="complex-roots",
    ),
    CatalogueEntry(
        "cpoly-product-instead-of-root",
        "Giving the product of the real roots instead of the root asked for.",
        area="complex-roots",
    ),
    CatalogueEntry(
        "cpoly-sum-instead-of-root",
        "Giving the sum of the real roots instead of the root asked for.",
        area="complex-roots",
    ),
    CatalogueEntry(
        "cpoly-conjugate-negates-both",
        "Writing the conjugate as the negative of the root, changing the sign of both parts.",
        area="complex-roots",
    ),
    CatalogueEntry(
        "cpoly-conjugate-negates-real",
        "Writing the conjugate by changing the sign of the real part instead of the imaginary part.",
        area="complex-roots",
    ),
    CatalogueEntry(
        "cpoly-conjugate-repeats-root",
        "Giving the same root again as the other non-real root.",
        area="complex-roots",
    ),
    CatalogueEntry(
        "cpoly-conjugate-swaps-parts",
        "Writing the conjugate by swapping the real and imaginary parts.",
        area="complex-roots",
    ),
    CatalogueEntry(
        "cpoly-square-of-sum",
        "Squaring p + q where p² + q² is needed for the product of the conjugate pair.",
        area="complex-roots",
    ),
    CatalogueEntry(
        "cpoly-i-squared-sign",
        "Taking i² as +1 when multiplying the conjugate pair, so the product becomes p² - q².",
        area="complex-roots",
    ),
    CatalogueEntry(
        "cpoly-product-multiplied",
        "Multiplying by the product of the pair where it must divide the constant term.",
        area="complex-roots",
    ),
    CatalogueEntry(
        "cpoly-constant-subtracted",
        "Subtracting the pair constant from the constant term where it must divide it.",
        area="complex-roots",
    ),
    CatalogueEntry(
        "binom-term-forgot-a-power",
        "Leaving out the power of the first term: the coefficient is C(n, r) multiplied by only the power of the second term.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-term-forgot-b-power",
        "Leaving out the power of the second term: the coefficient is C(n, r) multiplied by only the power of the first term.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-term-powers-swapped",
        "Putting the powers on the wrong terms, so a is raised to r and b to n - r.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-term-off-by-one",
        "Taking the wrong term: the (r + 1)th term has r in its binomial coefficient, and a slip of one gives the neighbouring term.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-term-sign-lost",
        "Dropping the minus sign of the second term when it is raised to an odd power.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-term-coefficients-dropped",
        "Giving the binomial coefficient or the coefficient without the powers of the numbers a and b.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-independent-term-middle",
        "Assuming the term independent of x is the middle term, instead of solving for r from the power of x.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-product-ignores-cross-terms",
        "In a product of two expansions, adding the coefficients of x^k from each and leaving out the products of lower-power terms.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-product-multiplies-coefficients",
        "In a product of two expansions, multiplying the two coefficients of x^k instead of pairing powers that add to k.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-product-drops-end-terms",
        "In a product of two expansions, leaving out the pairs that use the constant term of one factor.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-term-number-confusion",
        "Giving the number of the term (r + 1) where r, the index in the general term, was asked for.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-count-from-wrong-end",
        "Counting r from the other end of the expansion, giving n - r.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-coefficient-as-product",
        "Writing the binomial coefficient C(n, r) as the product n × r.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-next-term-not-updated",
        "Giving the term already found where the coefficient of the next term was asked for.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-coefficient-forgot-denominator",
        "Writing C(n, r) as n!/r!, leaving out the (n - r)! in the denominator.",
        area="binomial",
    ),
    CatalogueEntry(
        "binom-term-sign-flipped",
        "Getting the sign of a term wrong, so the coefficient has the opposite sign.",
        area="binomial",
    ),
    CatalogueEntry(
        "equiv-class-rounds-down",
        "Rounding the number of elements in a class down (N divided by n, ignoring the remainder), so classes that take a share of the leftover elements are undercounted.",
        area="equivalence",
    ),
    CatalogueEntry(
        "equiv-class-count-for-size",
        "Giving the number of classes (the modulus) where the size of a class was asked for.",
        area="equivalence",
    ),
    CatalogueEntry(
        "equiv-class-counts-one-direction",
        "Counting only the elements above (or below) the given element in its class, not the whole class.",
        area="equivalence",
    ),
    CatalogueEntry(
        "equiv-class-rounds-up-always",
        "Adding one to the quotient whether or not there is a remainder, so a class that fits exactly is overcounted.",
        area="equivalence",
    ),
    CatalogueEntry(
        "equiv-adds-only-reflexive",
        "Counting only the missing pairs (a, a), and not the pairs needed for symmetry and transitivity.",
        area="equivalence",
    ),
    CatalogueEntry(
        "equiv-adds-only-symmetric",
        "Counting only the pairs needed to make the relation symmetric.",
        area="equivalence",
    ),
    CatalogueEntry(
        "equiv-forgets-transitivity",
        "Adding the reflexive and symmetric pairs but not the pairs that transitivity then requires.",
        area="equivalence",
    ),
    CatalogueEntry(
        "equiv-counts-whole-relation",
        "Giving the size of the finished equivalence relation instead of the number of pairs to add to R.",
        area="equivalence",
    ),
    CatalogueEntry(
        "equiv-classes-as-elements",
        "Taking the number of classes to be the number of elements of the set, as if no two elements were related.",
        area="equivalence",
    ),
    CatalogueEntry(
        "equiv-squares-assumes-pm-only",
        "Counting the classes of a² ≡ b² as if only a ≡ ±b (mod n) made elements related, which is true only when n is prime.",
        area="equivalence",
    ),
    CatalogueEntry(
        "equiv-gcd-counts-units",
        "Counting only the elements that share no factor with n (Euler's totient) instead of every divisor of n.",
        area="equivalence",
    ),
    CatalogueEntry(
        "equiv-gcd-misses-n",
        "Leaving out one divisor (1 or n itself) when counting the classes by the divisors of n, or counting an extra one.",
        area="equivalence",
    ),
    CatalogueEntry(
        "equiv-lcm-as-product",
        "Using m × k where the least common multiple of m and k is needed.",
        area="equivalence",
    ),
    CatalogueEntry(
        "equiv-lcm-as-sum",
        "Using m + k where the least common multiple of m and k is needed.",
        area="equivalence",
    ),
    CatalogueEntry(
        "equiv-lcm-as-larger",
        "Using the larger of m and k where the least common multiple is needed.",
        area="equivalence",
    ),
    CatalogueEntry(
        "equiv-lcm-as-gcd",
        "Using the greatest common divisor where the least common multiple is needed.",
        area="equivalence",
    ),
    CatalogueEntry(
        "equiv-pairs-all-classes-largest",
        "Treating every class as if it had the size of the largest class when counting ordered pairs.",
        area="equivalence",
    ),
    CatalogueEntry(
        "equiv-pairs-reflexive-only",
        "Counting only the pairs (a, a) as the ordered pairs of the relation.",
        area="equivalence",
    ),
    CatalogueEntry(
        "equiv-pairs-each-element-with-largest",
        "Counting, for each element, the size of the largest class, instead of the size of its own class.",
        area="equivalence",
    ),
    CatalogueEntry(
        "equiv-pairs-equal-classes",
        "Splitting the pairs evenly over the classes, as if every class had the same size.",
        area="equivalence",
    ),
    CatalogueEntry(
        "equiv-pairs-ordered-not-halved",
        "Giving the ordered pairs of different elements where unordered pairs were asked for.",
        area="equivalence",
    ),
    CatalogueEntry(
        "equiv-pairs-reflexive-not-removed",
        "Halving the ordered pairs without first removing the pairs (a, a), or removing them twice.",
        area="equivalence",
    ),
    CatalogueEntry(
        "trigsub-inverse-swapped",
        "Using the wrong inverse trigonometric function for the limit or the antiderivative: arccos where arcsin is needed, or the complement of the right angle.",
        area="integration",
    ),
    CatalogueEntry(
        "trigsub-limits-not-converted",
        "Keeping the x limits as the θ limits, instead of converting them with the substitution.",
        area="integration",
    ),
    CatalogueEntry(
        "trigsub-spurious-factor",
        "Getting the constant factor wrong after substituting: leaving out the 1/a, or multiplying or dividing by an extra a.",
        area="integration",
    ),
    CatalogueEntry(
        "trigsub-wrong-inverse",
        "Using arctan where arcsin is needed, or the reverse, after choosing the wrong substitution.",
        area="integration",
    ),
    CatalogueEntry(
        "trigsub-double-angle-term-dropped",
        "Integrating sin² θ or cos² θ as θ/2 and leaving out the sin 2θ term.",
        area="integration",
    ),
    CatalogueEntry(
        "trigsub-double-angle-sign",
        "Getting the sign of the sin 2θ term wrong when integrating sin² θ or cos² θ.",
        area="integration",
    ),
    CatalogueEntry(
        "trigsub-half-dropped",
        "Leaving out the factor 1/2 from the identity for sin² θ or cos² θ, or doubling it.",
        area="integration",
    ),
    CatalogueEntry(
        "trigsub-constant-not-squared",
        "Using a where a² is needed, because x² = a² sin² θ and the constant is squared.",
        area="integration",
    ),
    CatalogueEntry(
        "trigsub-constant-cubed",
        "Raising the constant to the third power when the substitution gives a².",
        area="integration",
    ),
    CatalogueEntry(
        "trigsub-constant-dropped",
        "Leaving the constant factor out of the final answer.",
        area="integration",
    ),
    CatalogueEntry(
        "trigsub-power-ignored",
        "Treating (a² + x²)^(3/2) as a² + x², losing the sec θ left over after cancelling.",
        area="integration",
    ),
    CatalogueEntry(
        "trigsub-root-dropped",
        "Dropping the square root when converting the result back from θ to x.",
        area="integration",
    ),
    CatalogueEntry(
        "trigsub-wrong-power-of-a",
        "Getting the power of a wrong in the final answer after the substitution.",
        area="integration",
    ),
    CatalogueEntry(
        "trigsub-sin-cos-swapped",
        "Integrating cos θ as -sin θ or swapping sin θ and cos θ when converting back to x.",
        area="integration",
    ),
    CatalogueEntry(
        "trigsub-sec-as-tan",
        "Using the arctan limits where the secant substitution needs arcsec limits.",
        area="integration",
    ),
    CatalogueEntry(
        "trigsub-sign-reversed-limits",
        "Subtracting the limits in the wrong order, so the answer has the wrong sign.",
        area="integration",
    ),
    CatalogueEntry(
        "trigsub-limits-not-shifted",
        "Not shifting the limits when a square has been completed, so x - h is replaced by x.",
        area="integration",
    ),
    CatalogueEntry(
        "trigsub-limit-half-angle",
        "Halving the angle for the new upper limit.",
        area="integration",
    ),
    CatalogueEntry(
        "trigsub-limit-double-angle",
        "Doubling the angle for the new upper limit.",
        area="integration",
    ),
    CatalogueEntry(
        "trigsub-evaluates-instead-of-integrates",
        "Evaluating sin² θ at the limit instead of integrating it.",
        area="integration",
    ),
    CatalogueEntry(
        "redform-boundary-dropped",
        "Dropping the boundary term from the reduction formula that integration by parts produces, so only the multiple of the previous integral is left.",
        area="integration",
    ),
    CatalogueEntry(
        "redform-boundary-not-divided",
        "Leaving the boundary term undivided by the exponent's coefficient a in the reduction formula for the integral of x^n e^(ax).",
        area="integration",
    ),
    CatalogueEntry(
        "redform-boundary-value-wrong",
        "Evaluating the boundary term of the reduction formula wrongly, for example taking 1 where the upper limit gives e.",
        area="integration",
    ),
    CatalogueEntry(
        "redform-coefficient-dropped",
        "Leaving out the factor n that integration by parts brings in front of the previous integral.",
        area="integration",
    ),
    CatalogueEntry(
        "redform-denominator-off-by-one",
        "Using 1/n where the reduction formula has 1/(n - 1), so the denominator is off by one.",
        area="integration",
    ),
    CatalogueEntry(
        "redform-factor-one-over-k",
        "Using the factor 1/k in place of (k - 1)/k when stepping the reduction formula down by two.",
        area="integration",
    ),
    CatalogueEntry(
        "redform-limit-sign",
        "Adding the value at the lower limit instead of subtracting it when evaluating the integral of e^(ax).",
        area="integration",
    ),
    CatalogueEntry(
        "redform-lower-limit-dropped",
        "Forgetting the contribution of the lower limit when evaluating the base case or the boundary term.",
        area="integration",
    ),
    CatalogueEntry(
        "redform-n-term-not-divided",
        "Leaving the multiple of the previous integral undivided by a, so the term is n times the previous integral where it should be n/a times it.",
        area="integration",
    ),
    CatalogueEntry(
        "redform-order-off-by-one",
        "Applying the reduction formula from the wrong earlier integral, one step too few or too many, or using the previous integral where the one two steps back is needed.",
        area="integration",
    ),
    CatalogueEntry(
        "redform-power-of-b-dropped",
        "Dropping the power of the upper limit b from the boundary term, so b^n e^(ab) becomes e^(ab).",
        area="integration",
    ),
    CatalogueEntry(
        "redform-ratio-index-shift",
        "Using a ratio with the index shifted by one, such as (2n - 1)/(2n) where 2n/(2n + 1) is needed.",
        area="integration",
    ),
    CatalogueEntry(
        "redform-ratio-inverted",
        "Inverting the ratio in the reduction formula, so the integral grows with n instead of shrinking.",
        area="integration",
    ),
    CatalogueEntry(
        "redform-ratio-simplified-wrongly",
        "Using a simpler ratio such as n/(n + 1) where the reduction formula has 2n/(2n + 1).",
        area="integration",
    ),
    CatalogueEntry(
        "redform-sign-of-recurrence-term",
        "Getting the sign of the term in the previous integral wrong in the reduction formula, adding where it should be subtracted.",
        area="integration",
    ),
    CatalogueEntry(
        "redform-stopped-early",
        "Stopping before the base case is reached, so the answer is only the factor from the last step or starts from the wrong value.",
        area="integration",
    ),
    CatalogueEntry(
        "redform-term-dropped",
        "Leaving the previous integral out of the reduction formula, so only the boundary or constant term remains.",
        area="integration",
    ),
    CatalogueEntry(
        "redform-upper-limit-ignored",
        "Ignoring the upper limit b when evaluating the base case, as if the integral ran from 0 to 1.",
        area="integration",
    ),
    CatalogueEntry(
        "redform-wrong-base",
        "Starting the recurrence from swapped or wrong base cases, such as using the value for n = 1 where n = 0 is needed.",
        area="integration",
    CatalogueEntry(
        "rref-stopped-at-echelon",
        "Stopping at a row echelon form: clearing the entries below each pivot but not the entries above it, so the "
        "matrix is not reduced.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "rref-adds-instead-of-subtracts",
        "Clearing the entries above a pivot by adding the multiple of the pivot row instead of subtracting it, so "
        "the non-pivot columns come out with the wrong sign.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "rref-rows-scaled-only",
        "Dividing each row by its leading entry and stopping, without eliminating the other entries in the pivot "
        "columns.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "row-operation-arithmetic-slip",
        "A slip of one in a single entry while adding a multiple of one row to another.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "rank-counts-all-rows",
        "Giving the number of rows as the rank, without noticing that a row reduces to zero.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "rank-counts-all-columns",
        "Giving the number of columns as the rank.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "rank-misses-a-pivot",
        "Counting one pivot too few when reading the rank off the reduced matrix.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "free-variables-from-rows",
        "Counting free variables as the number of rows minus the rank instead of the number of unknowns minus the rank.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "free-variables-miscounted",
        "Miscounting the non-pivot columns when listing the free variables.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "free-variables-all-columns",
        "Treating every unknown as a free variable instead of subtracting the pivots.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "rref-free-column-read-unsigned",
        "Reading a solution straight off a free column of the reduced matrix without changing its sign (the pivot "
        "variable is minus that entry, not that entry).",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "free-variable-values-swapped",
        "Giving the value 1 to the wrong free variable, so the solution is the one for the other free variable.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "pivot-variables-left-zero",
        "Setting the free variables as asked but leaving the pivot variables at zero.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "free-column-mixed-up",
        "Reading the pivot variables from the column of the other free variable.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "linear-solution-inverted",
        "Solving Ck + D = 0 as k = C/D instead of k = -D/C.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "singular-means-infinitely-many",
        "Assuming that every value of the parameter that makes the determinant zero gives infinitely many solutions, "
        "without checking the right-hand side for consistency.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "singular-means-no-solution",
        "Assuming that a determinant of zero always means there is no solution, when the system may be consistent "
        "with infinitely many.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "singular-still-unique",
        "Saying a system still has exactly one solution when its coefficient matrix has determinant zero.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "dropped-constant-term-det",
        "Leaving the constant term out of a determinant that depends on a parameter.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "solution-sign-slip",
        "Reaching the right size of answer for a solution of a linear system with the wrong sign.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "cramer-numerator-not-divided",
        "Using Cramer's rule but stopping at the determinant with the column replaced, without dividing by det A.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "cramer-ratio-inverted",
        "Using Cramer's rule with the ratio upside down: det A over the replaced-column determinant.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "det-expansion-sign-slip",
        "Getting a sign wrong when expanding a determinant by cofactors, so a coefficient has the wrong sign.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "free-variable-not-substituted",
        "Writing the constants of the reduced equations as the pivot variables without subtracting the multiples of "
        "the free variables.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "free-variable-value-dropped",
        "Using the stated value of a free variable in the working but leaving it out of the final solution, so it is "
        "reported as zero.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "free-variable-sign-slip",
        "Giving a free variable the wrong sign, or adding where the free-column multiple should be subtracted, when "
        "writing out the solution.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "particular-solution-as-direction",
        "Giving a particular solution of the system as the direction vector of its line of solutions.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "direction-free-component-zero",
        "Writing the direction vector with 0 for the free variable instead of 1.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "direction-row-as-vector",
        "Giving a row of the coefficient matrix as the direction vector of the line of solutions.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "rhs-read-as-solution",
        "Reading the right-hand side b of the system as if it were the solution.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "free-variables-set-to-one",
        "Setting the free variables to 1 where the question asks for them to be zero.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "particular-solution-omitted",
        "Writing the multiples of the parameters but leaving out the constant (the particular solution) in a general "
        "solution.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "coefficients-order-swapped",
        "Giving the coefficients of a linear combination in the wrong order: b for u1 and a for u2.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "span-coefficient-sign-slip",
        "Getting the sign of a coefficient wrong when solving for the scalars in a linear combination.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "vector-entries-read-as-coefficients",
        "Reading the entries of the target vector as the coefficients of the combination.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "span-membership-assumed",
        "Assuming a vector is in a span without checking: saying a combination exists after the condition has failed.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "span-dependence-means-infinite",
        "Saying there are infinitely many combinations because the three vectors are dependent, when the first two "
        "vectors are independent and so any combination that exists is unique.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "dependence-relation-sign-slip",
        "Giving a coefficient of the wrong sign in a dependence relation: reading w = a·u + b·v as c1·u + c2·v + w = 0 "
        "with c1 = a, when c1 = -a.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "dependence-coefficients-swapped",
        "Reporting the coefficient of the other vector in a dependence relation.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "dependence-coefficient-of-w-reported",
        "Reporting the coefficient of the vector that was normalised to 1 in the relation instead of the one asked for.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "root-chosen-ignoring-condition",
        "Solving the determinant equation correctly but giving the root that the condition in the question rules out.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "rank-nullity-swapped",
        "Giving the rank where the nullity was asked for, or the nullity where the rank was asked for.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "rank-nullity-uses-rows",
        "Subtracting the rank from the number of rows where rank-nullity uses the number of columns.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "rank-nullity-sum-reported",
        "Giving the number of columns (rank plus nullity) as the answer instead of subtracting.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "rank-nullity-off-by-one",
        "Counting the dimension one too high when applying rank-nullity.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "dimension-counts-vectors",
        "Taking the dimension of a span to be the number of vectors listed, without checking for dependence.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "dimension-counts-coordinates",
        "Taking the dimension of a span to be the dimension of the space the vectors live in.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "dimension-counts-dependent-vector",
        "Counting a vector that is a combination of the others as one more dimension.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "left-null-space-confused-with-null-space",
        "Giving the dimension of the null space of A (columns minus rank) where the null space of the transpose "
        "(rows minus rank) was asked for.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "free-columns-as-basis",
        "Choosing the non-pivot columns as a basis of the column space instead of the pivot columns.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "first-columns-assumed-independent",
        "Taking the first columns as a basis of the column space without checking which columns are pivot columns.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "basis-mixes-pivot-and-free-columns",
        "Choosing a basis made of one pivot column and one non-pivot column, so the columns are not the pivot columns.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "coefficients-read-from-wrong-column",
        "Reading the coefficients of a dependent column from the wrong non-pivot column of the reduced matrix.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "eigenvalue-diagonal-entries-reported",
        "Reporting the diagonal entries of a matrix as its eigenvalues, which is right only for a triangular matrix.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "eigenvalue-sign-flipped",
        "Giving every eigenvalue the wrong sign by solving det(A + λI) = 0, or by reading the roots of the "
        "characteristic polynomial from the factors (λ - a) as λ = -a.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "eigenvalue-third-from-trace-sign-slip",
        "Finding two eigenvalues correctly and the third from the trace with a sign error, so it has the wrong sign.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "eigenvalue-missed-one-root",
        "Listing only two of the three eigenvalues of a 3×3 matrix.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "eigenvalue-trace-reported-as-root",
        "Reporting the trace of the matrix as one of its eigenvalues.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "eigenvalue-trace-and-determinant-reported",
        "Reporting the trace and the determinant as the two eigenvalues of a 2×2 matrix.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "sum-of-all-entries-as-trace",
        "Adding every entry of the matrix instead of only the leading-diagonal entries to find the trace.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "trace-sign-slip",
        "Giving the trace of a matrix with the wrong sign.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "trace-misses-a-diagonal-entry",
        "Leaving one of the leading-diagonal entries out of the trace.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "trace-reported-as-determinant",
        "Giving the trace of a matrix where its determinant was asked for.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "determinant-as-product-of-diagonal",
        "Taking the determinant of a matrix to be the product of its leading-diagonal entries, which is right only for "
        "a triangular matrix.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "product-of-others-sign-slip",
        "Giving the product of the remaining eigenvalues with the wrong sign.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "product-multiplied-instead-of-divided",
        "Multiplying the determinant by the known eigenvalue instead of dividing, to find the product of the others.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "whole-determinant-reported-as-product",
        "Giving the whole determinant as the product of the other two eigenvalues, without removing the known one.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "sum-of-others-reported-as-product",
        "Giving the sum of the other two eigenvalues where their product was asked for.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "discriminant-sign-slip",
        "Computing the discriminant t² - 4d as t² + 4d.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "discriminant-forgot-to-square",
        "Computing the discriminant t² - 4d without squaring t.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "discriminant-wrong-factor",
        "Using the wrong multiple of d in the discriminant t² - 4d.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "taylor-derivative-not-divided",
        "Giving the k-th derivative at the centre as the coefficient, without dividing by k!.",
        area="series",
    ),
    CatalogueEntry(
        "taylor-inner-coefficient-ignored",
        "Using the series of the outer function and leaving out the inner coefficient: the coefficient of x^k in e^(ax) given as 1/k!.",
        area="series",
    ),
    CatalogueEntry(
        "taylor-inner-coefficient-not-powered",
        "Writing the inner coefficient a where a^k belongs: the coefficient of x^k in e^(ax) given as a/k! instead of a^k/k!.",
        area="series",
    ),
    CatalogueEntry(
        "taylor-wrong-factorial",
        "Dividing by the wrong factorial, or by k where k! belongs, so the coefficient is out by a factor of k or k!/(k-1)!.",
        area="series",
    ),
    CatalogueEntry(
        "taylor-series-sign-pattern",
        "Getting the sign pattern of the standard series wrong, so the coefficient has the opposite sign.",
        area="series",
    ),
    CatalogueEntry(
        "taylor-derivative-order-off-by-one",
        "Using the (k-1)-th derivative for the coefficient of the k-th power.",
        area="series",
    ),
    CatalogueEntry(
        "taylor-expansion-point-ignored",
        "Evaluating the derivative at x = 1 (or x = 0) instead of at the stated centre of the expansion.",
        area="series",
    ),
    CatalogueEntry(
        "taylor-product-coefficientwise",
        "Multiplying the two coefficients of x^k instead of collecting every pair of terms whose powers add to k.",
        area="series",
    ),
    CatalogueEntry(
        "taylor-product-cross-terms-dropped",
        "Collecting only the terms that pair x^k with a constant, and missing the middle pairs of a product of series.",
        area="series",
    ),
    CatalogueEntry(
        "taylor-product-pair-missing",
        "Leaving one pair of terms out when collecting the coefficient of a product of series.",
        area="series",
    ),
    CatalogueEntry(
        "taylor-composite-order-mismatch",
        "Taking the term of the outer series that matches the power of x instead of the power of the inner expression.",
        area="series",
    ),
    CatalogueEntry(
        "taylor-order-confused",
        "Giving the coefficient of the wrong power of x (x^3 for x^2, or the reverse).",
        area="series",
    ),
    CatalogueEntry(
        "taylor-polynomial-term-dropped",
        "Leaving the highest-degree term out when evaluating a Taylor polynomial.",
        area="series",
    ),
    CatalogueEntry(
        "taylor-polynomial-wrong-degree",
        "Evaluating a polynomial of the wrong degree, with one term too many.",
        area="series",
    ),
    CatalogueEntry(
        "taylor-powers-of-x-ignored",
        "Multiplying each coefficient by x instead of by the matching power of x when evaluating a polynomial.",
        area="series",
    ),
    CatalogueEntry(
        "taylor-error-of-lower-degree",
        "Finding the error of a lower-degree polynomial than the one asked about.",
        area="series",
    ),
    CatalogueEntry(
        "taylor-approximation-reported-as-error",
        "Reporting the approximate value where the size of the error was asked for.",
        area="series",
    ),
    CatalogueEntry(
        "taylor-value-reported-as-error",
        "Reporting the true value of the function where the size of the error was asked for.",
        area="series",
    ),
    CatalogueEntry(
        "taylor-error-sign-ignored",
        "Reporting the signed difference between the approximation and the true value where the (positive) error was asked for.",
        area="series",
    ),
    CatalogueEntry(
        "taylor-relative-error",
        "Reporting the error as a fraction of the true value where the absolute error was asked for.",
        area="series",
    ),
    CatalogueEntry(
        "taylor-error-of-higher-degree",
        "Finding the error of a higher-degree polynomial than the one asked about.",
        area="series",
    ),
    CatalogueEntry(
        "remainder-power-off-by-one",
        "Using |x - a|^n where the Lagrange remainder of a degree-n polynomial has |x - a|^(n+1).",
        area="series",
    ),
    CatalogueEntry(
        "remainder-factorial-index",
        "Dividing by n! where the Lagrange remainder of a degree-n polynomial has (n+1)!.",
        area="series",
    ),
    CatalogueEntry(
        "remainder-factorial-dropped",
        "Leaving the factorial out of the Lagrange remainder altogether.",
        area="series",
    ),
    CatalogueEntry(
        "remainder-degree-off-by-one",
        "Applying the remainder formula for the polynomial one degree higher or lower than the one asked about.",
        area="series",
    ),
    CatalogueEntry(
        "remainder-derivative-order-off-by-one",
        "Bounding the n-th derivative where the (n+1)-th derivative is needed (or the reverse) for the Lagrange remainder.",
        area="series",
    ),
    CatalogueEntry(
        "remainder-derivative-bound-ignored",
        "Leaving the bound M on the derivative out of the Lagrange error bound, as if it were 1.",
        area="series",
    ),
    CatalogueEntry(
        "remainder-max-at-wrong-endpoint",
        "Taking the largest size of a decreasing derivative at the far end of the interval instead of at the near end.",
        area="series",
    ),
    CatalogueEntry(
        "remainder-terms-for-degree",
        "Giving the number of terms of the polynomial (one more than its degree) where the degree was asked for.",
        area="series",
    ),
    CatalogueEntry(
        "remainder-wrong-end-of-interval",
        "Giving the upper end of the interval (the approximation plus the bound) where the lower end was asked for.",
        area="series",
    ),
    CatalogueEntry(
        "remainder-bound-not-applied",
        "Giving the approximation itself where the end of the interval it fixes was asked for.",
        area="series",
    ),
    CatalogueEntry(
        "remainder-bound-subtracted-the-wrong-way",
        "Subtracting the approximation from the bound where the bound is taken away from the approximation.",
        area="series",
    ),
    CatalogueEntry(
        "series-integral-power-rule-slip",
        "Integrating (x + b)^(-q) without dividing by the new exponent, or with the wrong exponent.",
        area="series",
    ),
    CatalogueEntry(
        "series-integral-constant-dropped",
        "Dropping the constant multiplier when integrating c/(x + b)^q.",
        area="series",
    ),
    CatalogueEntry(
        "series-integral-wrong-lower-limit",
        "Evaluating an improper integral at the wrong lower limit.",
        area="series",
    ),
    CatalogueEntry(
        "series-integral-replaced-by-first-term",
        "Giving the first term of the series where the value of the integral was asked for.",
        area="series",
    ),
    CatalogueEntry(
        "series-integral-sign-slip",
        "Getting the sign of an improper integral wrong when the antiderivative is evaluated at the limits.",
        area="series",
    ),
    CatalogueEntry(
        "series-bound-integral-omitted",
        "Giving only the first term as the integral-test bound, leaving out the integral.",
        area="series",
    ),
    CatalogueEntry(
        "series-bound-first-term-omitted",
        "Giving only the integral as the bound for the sum, leaving out the first term.",
        area="series",
    ),
    CatalogueEntry(
        "series-bound-wrong-first-term",
        "Adding the wrong term of the series to the integral in the integral-test bound.",
        area="series",
    ),
    CatalogueEntry(
        "series-partial-sum-terms-miscounted",
        "Adding the wrong number of terms, or starting from the wrong index, in a partial sum.",
        area="series",
    ),
    CatalogueEntry(
        "series-tail-integral-start-off-by-one",
        "Starting the tail integral at the wrong point, one step before or after the partial sum ends.",
        area="series",
    ),
    CatalogueEntry(
        "series-tail-integral-omitted",
        "Leaving the tail integral out of the sharper bound, or adding a single term in its place.",
        area="series",
    ),
    CatalogueEntry(
        "series-partial-sum-omitted",
        "Giving the tail integral alone where the partial sum plus the tail integral was asked for.",
        area="series",
    ),
    CatalogueEntry(
        "alt-error-last-term-used",
        "Bounding the alternating-series error by the last term kept, b_N, instead of the first term dropped, b_(N+1).",
        area="series",
    ),
    CatalogueEntry(
        "alt-error-term-after-next",
        "Bounding the alternating-series error by b_(N+2), skipping the first dropped term.",
        area="series",
    ),
    CatalogueEntry(
        "alt-error-halved",
        "Halving the first dropped term to get the alternating-series error bound.",
        area="series",
    ),
    CatalogueEntry(
        "alt-error-two-terms-added",
        "Adding the next two terms, b_(N+1) + b_(N+2), as the error bound.",
        area="series",
    ),
    CatalogueEntry(
        "alt-error-signed",
        "Giving the alternating-series error with the sign of the first dropped term rather than as a size.",
        area="series",
    ),
    CatalogueEntry(
        "alt-terms-uses-last-term",
        "Requiring b_N below the tolerance rather than b_(N+1), so one term too few.",
        area="series",
    ),
    CatalogueEntry(
        "alt-terms-off-by-one",
        "Counting the terms needed one too many for the tolerance.",
        area="series",
    ),
    CatalogueEntry(
        "alt-tolerance-misread",
        "Misreading the tolerance as a different power of ten when finding the number of terms.",
        area="series",
    ),
    CatalogueEntry(
        "alt-partial-sum-wrong-count",
        "Summing the wrong number of terms of the alternating series.",
        area="series",
    ),
    CatalogueEntry(
        "alt-partial-sum-sign-pattern",
        "Using the wrong sign pattern when forming an alternating partial sum.",
        area="series",
    ),
    CatalogueEntry(
        "alt-partial-sum-bound-subtracted",
        "Subtracting the error bound from the partial sum instead of reporting the partial sum.",
        area="series",
    ),
    CatalogueEntry(
        "alt-interval-wrong-end",
        "Taking the partial sum plus the bound as the lower end of the interval.",
        area="series",
    ),
    CatalogueEntry(
        "alt-interval-bound-not-applied",
        "Reporting the partial sum as the end of the interval without applying the bound.",
        area="series",
    ),
    CatalogueEntry(
        "alt-interval-bound-subtracted-the-wrong-way",
        "Applying the error bound in the wrong direction when bracketing the sum.",
        area="series",
    ),
    CatalogueEntry(
        "ode1-factor-sign-flipped",
        "Using e^(-integral of P) rather than e^(integral of P) as the integrating factor.",
        area="ode",
    ),
    CatalogueEntry(
        "ode1-factor-integral-skipped",
        "Using e^(P(x)) as the integrating factor without integrating P.",
        area="ode",
    ),
    CatalogueEntry(
        "ode1-factor-integral-not-exponentiated",
        "Giving the integral of P as the integrating factor instead of e raised to it.",
        area="ode",
    ),
    CatalogueEntry(
        "ode1-factor-coefficient-inverted",
        "Using e^(x/a) as the integrating factor for a constant coefficient a.",
        area="ode",
    ),
    CatalogueEntry(
        "ode1-factor-coefficient-dropped",
        "Dropping the coefficient k from the integral of k/x, so the factor becomes x.",
        area="ode",
    ),
    CatalogueEntry(
        "ode1-factor-power-rule-misapplied",
        "Integrating P(x) with the wrong power-rule step (forgetting to halve or raise the power) before exponentiating.",
        area="ode",
    ),
    CatalogueEntry(
        "ode1-particular-coefficient-not-divided",
        "Leaving the forcing coefficient undivided by the factor the integral produces, so the particular solution is too large.",
        area="ode",
    ),
    CatalogueEntry(
        "ode1-ic-ignores-particular-part",
        "Setting the constant equal to y(0) without subtracting the particular solution's value at 0.",
        area="ode",
    ),
    CatalogueEntry(
        "ode1-ic-sign-error",
        "Subtracting the wrong way round when fixing the constant from the initial condition.",
        area="ode",
    ),
    CatalogueEntry(
        "ode1-ic-evaluated-at-wrong-point",
        "Evaluating the particular solution at x = 1 rather than at the point where the initial condition is given.",
        area="ode",
    ),
    CatalogueEntry(
        "ode1-homogeneous-exponent-sign",
        "Writing the complementary term with the wrong sign in its exponent (e^(+ax) instead of e^(-ax)).",
        area="ode",
    ),
    CatalogueEntry(
        "ode1-particular-denominator-sign",
        "Dividing the forcing coefficient by a - k instead of a + k.",
        area="ode",
    ),
    CatalogueEntry(
        "ode1-integrating-factor-division-skipped",
        "Forgetting to divide the constant of integration by the integrating factor.",
        area="ode",
    ),
    CatalogueEntry(
        "ode1-lead-coefficient-not-divided",
        "Not dividing the right-hand side by x before using the integrating factor, so the power is one too high.",
        area="ode",
    ),
    CatalogueEntry(
        "ode1-resonance-factor-x-missing",
        "Missing the factor x in the solution when the forcing term matches the complementary solution.",
        area="ode",
    ),
    CatalogueEntry(
        "ode1-ic-homogeneous-term-dropped",
        "Dropping the arbitrary-constant term from the solution once the initial condition is applied.",
        area="ode",
    ),
    CatalogueEntry(
        "ode1-integral-over-integrated",
        "Integrating a constant right-hand side twice (giving x^2/2 instead of x).",
        area="ode",
    ),
    CatalogueEntry(
        "ode1-particular-constant-correction-missed",
        "Finding the constant of the particular solution as r/a, missing the -m/a correction from integration by parts.",
        area="ode",
    ),
    CatalogueEntry(
        "ode1-integration-by-parts-sign",
        "A sign error in the integration by parts, giving (r + m)/a instead of (r - m)/a for the constant term.",
        area="ode",
    ),
    CatalogueEntry(
        "ode1-eval-exponent-sign",
        "Evaluating the decaying term with the sign of its exponent flipped.",
        area="ode",
    ),
    CatalogueEntry(
        "ode1-eval-exponential-term-dropped",
        "Dropping the decaying term when evaluating the solution.",
        area="ode",
    ),
    CatalogueEntry(
        "ode1-eval-exponential-dropped",
        "Treating the exponential factor as 1 when evaluating the solution.",
        area="ode",
    ),
    CatalogueEntry(
        "ode1-eval-particular-dropped",
        "Dropping the particular solution when evaluating, keeping only the decaying term.",
        area="ode",
    ),
    CatalogueEntry(
        "ode1-eval-point-wrong",
        "Evaluating the solution at x = 1 rather than the requested x.",
        area="ode",
    ),
    CatalogueEntry(
        "ode1-eval-initial-value-reported",
        "Reporting the initial value y(0) instead of the value at the requested x.",
        area="ode",
    ),
    CatalogueEntry(
        "ode1-particular-coefficient-divided-twice",
        "Dividing by the exponent coefficient a twice when finding the particular solution.",
        area="ode",
    ),
    CatalogueEntry(
        "ode1-linear-coefficients-swapped",
        "Swapping the coefficient of x and the constant on the right-hand side when finding the particular solution.",
        area="ode",
    ),
    CatalogueEntry(
        "ode1-particular-coefficient-sign-flipped",
        "Giving the particular-solution coefficient with the opposite sign.",
        area="ode",
    ),
    CatalogueEntry(
        "ode2-arctan-inverse-confused",
        "Writing arctan of the right-hand side instead of tan of it when undoing arctan y.",
        area="ode",
    ),
    CatalogueEntry(
        "ode2-arctan-not-inverted",
        "Stopping at arctan y = (the integral) and giving that as y, without applying tan.",
        area="ode",
    ),
    CatalogueEntry(
        "ode2-constant-added-not-multiplied",
        "Writing the exponentiated solution as y0 + e^(...) instead of y0 e^(...): the constant from e^C multiplies, not adds.",
        area="ode",
    ),
    CatalogueEntry(
        "ode2-exponent-sign",
        "Taking the exponent of the logistic solution with the wrong sign.",
        area="ode",
    ),
    CatalogueEntry(
        "ode2-exponential-inverted",
        "Giving e^(...) as y, when e^y is what equals that expression.",
        area="ode",
    ),
    CatalogueEntry(
        "ode2-factor-two-mishandled",
        "Mishandling the factor of 2 produced by integrating y or x (halving or doubling once too often).",
        area="ode",
    ),
    CatalogueEntry(
        "ode2-half-forgotten",
        "Forgetting the 1/2 when integrating y with respect to y, so the constant is y0^2 rather than y0^2/2.",
        area="ode",
    ),
    CatalogueEntry(
        "ode2-ic-constant-dropped",
        "Dropping the constant from the initial condition: giving e^(...) without the factor y(0).",
        area="ode",
    ),
    CatalogueEntry(
        "ode2-ic-constant-misapplied",
        "Putting the initial condition in the wrong place when solving for the constant.",
        area="ode",
    ),
    CatalogueEntry(
        "ode2-ic-squared-forgotten",
        "Forgetting to square (or square root) the initial value when working with y^2.",
        area="ode",
    ),
    CatalogueEntry(
        "ode2-integration-constant-dropped",
        "Dropping the constant of integration, so the solution misses its '+ 1'.",
        area="ode",
    ),
    CatalogueEntry(
        "ode2-ln-not-exponentiated",
        "Leaving ln y = (the integral) as the answer instead of exponentiating to solve for y.",
        area="ode",
    ),
    CatalogueEntry(
        "ode2-partial-fraction-factor-missed",
        "Missing the 1/M factor in the partial fractions of 1/(y(M - y)), so the exponent lacks the factor M.",
        area="ode",
    ),
    CatalogueEntry(
        "ode2-partial-fraction-sign",
        "A sign error in the partial fractions, giving y0 - (M - y0)e^(...) in the denominator.",
        area="ode",
    ),
    CatalogueEntry(
        "ode2-power-rule-misapplied",
        "Using the wrong power for the antiderivative (x^2 for x, or no power at all).",
        area="ode",
    ),
    CatalogueEntry(
        "ode2-power-rule-not-divided",
        "Integrating a power of x without dividing by the new exponent.",
        area="ode",
    ),
    CatalogueEntry(
        "ode2-reciprocal-forgotten",
        "Using y0 where 1/y0 is needed (or the reverse) after integrating 1/y^2.",
        area="ode",
    ),
    CatalogueEntry(
        "ode2-reciprocal-sign",
        "Losing the minus sign of the integral of 1/y^2 = -1/y.",
        area="ode",
    ),
    CatalogueEntry(
        "ode2-reciprocal-squared",
        "Squaring the reciprocal: giving -1/y0^2 instead of -1/y0.",
        area="ode",
    ),
    CatalogueEntry(
        "ode2-sign-error-root",
        "A sign error when moving to a square root or a root's argument.",
        area="ode",
    ),
    CatalogueEntry(
        "ode2-square-root-forgotten",
        "Forgetting to take the square root after solving for x^2.",
        area="ode",
    ),
    CatalogueEntry(
        "ode2-y-dy-integrated-as-y",
        "Integrating y dy as y rather than y^2/2.",
        area="ode",
    ),
    CatalogueEntry(
        "ode2-y-power-ignored",
        "Treating dy/dx = a x y^2 as dy/dx = a x y and giving an exponential.",
        area="ode",
    ),
    CatalogueEntry(
        "ode3-characteristic-polynomial-differentiated",
        "Dividing by the derivative of the characteristic polynomial, 2k + b, instead of its value at k.",
        area="ode",
    ),
    CatalogueEntry(
        "ode3-coefficient-not-divided",
        "Giving the forcing coefficient as the coefficient of the particular solution, without dividing by the characteristic polynomial's value.",
        area="ode",
    ),
    CatalogueEntry(
        "ode3-constant-term-dropped",
        "Dropping the constant term of the particular solution.",
        area="ode",
    ),
    CatalogueEntry(
        "ode3-constant-term-slip",
        "An arithmetic slip in the constant term of the particular solution.",
        area="ode",
    ),
    CatalogueEntry(
        "ode3-constants-swapped",
        "Swapping the roles of the two homogeneous constants, so each belongs to the wrong exponential.",
        area="ode",
    ),
    CatalogueEntry(
        "ode3-derivative-ic-particular-dropped",
        "Leaving the particular solution's derivative out of the condition on y'(0).",
        area="ode",
    ),
    CatalogueEntry(
        "ode3-derivative-order-ignored",
        "Treating y' and y'' as y when substituting, so the characteristic polynomial becomes k^2 + b + c.",
        area="ode",
    ),
    CatalogueEntry(
        "ode3-derivative-terms-dropped",
        "Dropping the y' (and y'') terms when matching coefficients, so only c times the trial solution is compared.",
        area="ode",
    ),
    CatalogueEntry(
        "ode3-eval-exponent-sign",
        "Evaluating the particular term with the sign of its exponent flipped.",
        area="ode",
    ),
    CatalogueEntry(
        "ode3-eval-exponential-dropped",
        "Treating every exponential factor as 1 when evaluating the solution.",
        area="ode",
    ),
    CatalogueEntry(
        "ode3-eval-initial-value-reported",
        "Reporting the initial value y(0) instead of y at the requested point.",
        area="ode",
    ),
    CatalogueEntry(
        "ode3-eval-particular-dropped",
        "Dropping the particular term when evaluating the solution.",
        area="ode",
    ),
    CatalogueEntry(
        "ode3-eval-point-wrong",
        "Evaluating the solution at x = 1 rather than the requested x.",
        area="ode",
    ),
    CatalogueEntry(
        "ode3-factor-two-mishandled",
        "Mishandling the factor of 2 produced by differentiating a repeated or resonant trial solution.",
        area="ode",
    ),
    CatalogueEntry(
        "ode3-first-derivative-dropped",
        "Leaving the y' contribution out when matching the coefficient of x.",
        area="ode",
    ),
    CatalogueEntry(
        "ode3-ic-ignores-particular",
        "Fixing the constants from the initial conditions without allowing for the particular solution's value and slope.",
        area="ode",
    ),
    CatalogueEntry(
        "ode3-resonance-derivative-incomplete",
        "Dividing by 2r instead of 2r + b when the trial solution is x e^(rx).",
        area="ode",
    ),
    CatalogueEntry(
        "ode3-resonance-factor-x-missing",
        "Trying A e^(rx) when e^(rx) already solves the homogeneous equation, so the factor x is missing.",
        area="ode",
    ),
    CatalogueEntry(
        "ode3-resonance-order-too-high",
        "Multiplying the trial solution by x more times than the multiplicity of the root.",
        area="ode",
    ),
    CatalogueEntry(
        "ode3-resonance-order-too-low",
        "Multiplying by x only once when the root is repeated.",
        area="ode",
    ),
    CatalogueEntry(
        "ode3-second-derivative-dropped",
        "Leaving the y'' contribution (2a) out of the constant-term equation.",
        area="ode",
    ),
    CatalogueEntry(
        "ode3-sign-error",
        "A sign error when solving for the coefficients.",
        area="ode",
    ),
    CatalogueEntry(
        "ode3-sign-in-characteristic-polynomial",
        "Substituting k^2 + bk - c instead of k^2 + bk + c for the characteristic polynomial.",
        area="ode",
    ),
    CatalogueEntry(
        "ode3-trig-coefficients-swapped",
        "Swapping the cosine and sine coefficients of the particular solution.",
        area="ode",
    ),
    CatalogueEntry(
        "ode3-trig-partner-confused",
        "Using x cos(wx) instead of x sin(wx) as the resonant trial solution's partner.",
        area="ode",
    ),
    CatalogueEntry(
        "ode3-trig-partner-missed",
        "Taking only a cosine term, missing the sine term that the first derivative forces.",
        area="ode",
    ),
    CatalogueEntry(
        "ode3-wrong-root-used",
        "Using the other characteristic root in the resonant trial solution.",
        area="ode",
    ),
    CatalogueEntry(
        "series-absolute-confused-with-conditional",
        "Giving the count of absolutely convergent cases when asked for the conditionally convergent ones.",
        area="series",
    ),
    CatalogueEntry(
        "series-conditional-confused-with-divergent",
        "Counting the conditionally convergent cases as divergent (or the reverse) because the series does not converge absolutely.",
        area="series",
    ),
    CatalogueEntry(
        "series-converges-counted-as-absolute",
        "Counting every convergent case, absolute and conditional together, as the absolutely convergent ones.",
        area="series",
    ),
    CatalogueEntry(
        "series-denominator-coefficient-dropped",
        "Dropping the denominator's leading coefficient when finding the limit in a limit comparison.",
        area="series",
    ),
    CatalogueEntry(
        "series-k-overshoot",
        "Giving a larger exponent than the least one that works.",
        area="series",
    ),
    CatalogueEntry(
        "series-limit-inverted",
        "Inverting the limit: finding lim b_n / a_n instead of lim a_n / b_n.",
        area="series",
    ),
    CatalogueEntry(
        "series-lower-terms-kept",
        "Letting the lower-order terms change the limit, adding them to the leading coefficients or multiplying by them.",
        area="series",
    ),
    CatalogueEntry(
        "series-numerator-coefficient-dropped",
        "Dropping the numerator's leading coefficient when finding the limit in a limit comparison.",
        area="series",
    ),
    CatalogueEntry(
        "series-numerator-power-ignored",
        "Ignoring the power of n in the numerator when finding the exponent p of the p-series.",
        area="series",
    ),
    CatalogueEntry(
        "series-p-boundary-excluded",
        "Excluding a boundary case that belongs in the count (off by one at p = 1 or p = 0).",
        area="series",
    ),
    CatalogueEntry(
        "series-p-boundary-included",
        "Treating p = 1 as convergent (or counting the boundary case on the wrong side): the harmonic series diverges.",
        area="series",
    ),
    CatalogueEntry(
        "series-root-index-ignored",
        "Ignoring the root index when finding the exponent p, so the exponent of n under the root is not divided by it.",
        area="series",
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
