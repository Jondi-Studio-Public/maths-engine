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
        "adds-instead-of-subtracts",
        "Adding f(k) to f(k + 1) where the difference was needed.",
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
        "evaluated-at-wrong-point",
        "Substituting the wrong value into the derivatives, such as x = 0 when the limit is as x approaches c.",
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
        "product-rule-term-dropped",
        "Differentiating a product and keeping only one of the two product-rule terms.",
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
