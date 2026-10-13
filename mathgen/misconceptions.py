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
    # -- first-year multivariable, 3D geometry and statistics --
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
