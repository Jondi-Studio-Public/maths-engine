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
    # -- circles, solids and units --------------------------------------------------------------------------------
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
    # -- measurement: area, volume, energy, time, chords, triangles ------------------------------------------------
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
