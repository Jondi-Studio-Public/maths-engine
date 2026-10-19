"""The cross-checks of the flux and integral-theorem generators.

`make()` computes each answer by one exact route. The second routes live here: with `CHECKS` on, every exact integral is
also done by Gauss-Legendre quadrature, and each theorem is checked by working out both of its sides (Green: the double
integral against the line integrals along every piece of the boundary; divergence and flux: the volume integral against
the fluxes through every piece; Stokes: the boundary against the flux of the curl, parametrised). Any disagreement raises.
The questions built with the checks on must also be identical to those built with them off.
"""

from __future__ import annotations

import pytest

from mathgen import make
from mathgen.generators import _s2a3_vc as vc
from mathgen.generators import (  # noqa: F401  (registers the topics)
    s2a3_divergence_flux_build,
    s2a3_flux_graph_build,
    s2a3_flux_pieces_build,
    s2a3_green_area_build,
    s2a3_green_circulation_build,
    s2a3_stokes_circulation_build,
    s2a3_theorem_verification_build,
)

TOPICS = [
    "s2a3_flux_graph",
    "s2a3_flux_pieces",
    "s2a3_green_circulation",
    "s2a3_green_area",
    "s2a3_divergence_flux",
    "s2a3_stokes_circulation",
    "s2a3_theorem_verification",
]
SEEDS = range(300, 330)


def _fingerprint(q):
    parts = [(p.label, p.prompt, p.answer_text, tuple(p.distractors)) for p in q.parts or []]
    return (q.text, q.answer_text, tuple(q.distractors), tuple(parts))


def _build(topic, tier, seed):
    return make(topic, seed=seed, difficulty=tier)


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize("tier", ["easy", "medium", "hard", "exam"])
def test_every_route_agrees_and_changes_nothing(topic, tier, monkeypatch):
    name = f"exd_x_{topic}" if tier == "exam" else topic
    plain = [_fingerprint(_build(name, tier, s)) for s in SEEDS]
    monkeypatch.setattr(vc, "CHECKS", True)
    checked = [_fingerprint(_build(name, tier, s)) for s in SEEDS]  # raises if any second route disagrees
    assert checked == plain


def test_quadrature_catches_a_wrong_exact_value(monkeypatch):
    """The quadrature check is live: a polynomial integrated with a deliberately wrong region disagrees."""
    import sympy as sp

    monkeypatch.setattr(vc, "CHECKS", True)
    dom = vc.Dom((vc.u, vc.v), ((0, 2), (0, 3)), {})
    assert vc.integrate(vc.u**2 * vc.v, dom) == sp.Rational(8, 3) * sp.Rational(9, 2)
    wrong = vc.Dom((vc.u, vc.v), ((0, 2), (0, 3)), {})
    monkeypatch.setattr(vc, "_numeric_poly", lambda terms, d: 0.0)
    with pytest.raises(Exception):
        vc.integrate(vc.u**2 * vc.v, wrong)
