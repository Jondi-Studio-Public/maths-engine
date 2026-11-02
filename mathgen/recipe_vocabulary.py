"""The closed vocabulary recipe subjects declare: function families and subject properties.

One file, extended only by pull request, so a subject cannot invent a name. ``@subject`` refuses a family or
property that is not listed here. A property may also have a *check* (``@property_check("name")``): a function of a
drawn subject that returns whether the property really holds; ``mathgen.recipes.conformance`` runs every registered
check for every seed, so a declared property is true of the maths and the list cannot drift from it.
Meanings are in ``docs/design/recipes.md``. First list for VCE Methods; Specialist, HSC and QCE add to it later.
"""

from __future__ import annotations

from collections.abc import Callable

FAMILIES = frozenset(
    {
        "polynomial",
        "power",
        "rational",
        "exponential",
        "logarithmic",
        "circular",
        "hybrid",
        "composite",
        "inverse",
        "pdf",
        "discrete_pmf",
    }
)

PROPERTIES = frozenset(
    {
        # argument shape
        "linear_argument",
        "quadratic_argument",
        "polynomial_argument",
        # algebraic structure
        "quadratic_in_exponential",
        "has_surd_roots",
        "has_log_root",
        "has_pi_roots",
        "rational_roots_only",
        # symmetry and periodicity
        "even_candidate",
        "odd_candidate",
        "periodic",
        # calculus
        "differentiable_everywhere",
        "has_stationary_point",
        "has_inflection",
        "has_closed_form_antiderivative",
        "by_parts_integrable",
        "substitution_integrable",
        # domain and range
        "restricted_domain",
        "has_asymptote",
        # probability
        "unknown_constants_1",
        "unknown_constants_2",
        "bounded_support",
        # composition
        "invertible_on_domain",
        "composite_of_translation",
    }
)

#: property name -> check(subject) -> bool, for properties whose truth can be tested on a drawn subject.
PROPERTY_CHECKS: dict[str, Callable] = {}


def property_check(name: str):
    """Register ``fn(subject) -> bool`` as the test that ``subject`` really has property ``name``."""
    if name not in PROPERTIES:
        raise ValueError(f"unknown property {name!r}")

    def deco(fn: Callable) -> Callable:
        if name in PROPERTY_CHECKS:
            raise ValueError(f"duplicate property check: {name}")
        PROPERTY_CHECKS[name] = fn
        return fn

    return deco
