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
