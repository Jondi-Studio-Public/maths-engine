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
    # -- complex numbers in exponential form ----------------------------------------------------------------------
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
