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
