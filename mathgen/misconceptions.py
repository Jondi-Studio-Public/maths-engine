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
    # -- first-year linear algebra: row reduction -------------------------------------------------------------------
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
    # -- first-year linear algebra: systems with a parameter -------------------------------------------------------
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
        "sum-of-roots-as-root",
        "Giving the sum of the roots of the determinant equation (minus the linear coefficient over the quadratic "
        "coefficient) as one of the roots.",
        area="linear-algebra",
    ),
    CatalogueEntry(
        "product-of-roots-as-root",
        "Giving the product of the roots of the determinant equation (constant over leading coefficient) as one of "
        "the roots.",
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
    # -- first-year linear algebra: general solutions ---------------------------------------------------------------
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
    # -- first-year linear algebra: span ----------------------------------------------------------------------------
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
    # -- first-year linear algebra: linear independence ------------------------------------------------------------
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
    # -- first-year linear algebra: column space and rank-nullity ---------------------------------------------------
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
    # -- first-year linear algebra: eigenvalues ---------------------------------------------------------------------
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
