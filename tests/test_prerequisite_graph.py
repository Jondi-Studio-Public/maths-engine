"""The prerequisite graph: the validator's rules, the shipped library's shape, and the helpers in
mathgen/scheduling/prerequisites.py."""

from __future__ import annotations

import pytest

from mathgen.scheduling import prerequisites as graph
from mathgen.skills import Domain, Skill, Source, load_skills, validate


def _skill(sid, domain="arithmetic", status="active", prerequisites=(), **extra):
    reason = {"deferred_reason": "generator"} if status == "deferred" else {}
    return Skill(
        id=sid,
        name=sid,
        domain=domain,
        status=status,
        prerequisites=list(prerequisites),
        answer="number",
        sources=[Source(topic="addition", params={}, difficulty="easy")],
        **reason,
        **extra,
    )


DOMAINS = [
    Domain(id="arithmetic", name="Arithmetic", order=1, band="primary", band_name="Primary"),
    Domain(id="calculus", name="Calculus", order=2, band="senior", band_name="Senior"),
    Domain(id="analysis", name="Analysis", order=3, band="university", band_name="University"),
]


def _graph_errors(skills):
    return [e for e in validate(DOMAINS, skills, gate_registry={}).errors if "prerequisite" in e]


# -- validator rules ------------------------------------------------------------------------------


def test_an_active_skill_may_not_build_on_a_deferred_one():
    skills = [
        _skill("count"),
        _skill("later", status="deferred", prerequisites=["count"]),
        _skill("add", prerequisites=["later"]),
    ]
    assert _graph_errors(skills) == [
        "add: active skill has deferred prerequisite 'later'; a host that waits for prerequisites would never offer it"
    ]


def test_a_deferred_skill_may_build_on_a_deferred_one():
    skills = [
        _skill("count"),
        _skill("a", status="deferred", prerequisites=["count"]),
        _skill("b", status="deferred", prerequisites=["a"]),
    ]
    assert _graph_errors(skills) == []


@pytest.mark.parametrize("domain", ["calculus", "analysis"])
def test_an_active_senior_or_university_skill_needs_a_prerequisite(domain):
    errors = _graph_errors([_skill("count"), _skill("deep", domain=domain)])
    band = "senior" if domain == "calculus" else "university"
    assert errors == [f"deep: active {band} skill has no prerequisites"]


def test_a_deferred_senior_skill_may_have_none_yet():
    assert _graph_errors([_skill("count"), _skill("deep", domain="calculus", status="deferred")]) == []


def test_a_primary_starting_point_may_have_none():
    assert _graph_errors([_skill("count"), _skill("deep", domain="calculus", prerequisites=["count"])]) == []


# -- the shipped library --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def library():
    _, skills = load_skills()
    return {s.id: s for s in skills}


def test_no_active_skill_is_isolated(library):
    named = {pre for s in library.values() for pre in s.prerequisites}
    isolated = sorted(
        s.id for s in library.values() if s.status == "active" and not s.prerequisites and s.id not in named
    )
    assert isolated == []


def test_no_active_skill_builds_on_a_deferred_one(library):
    bad = sorted(
        (s.id, pre)
        for s in library.values()
        if s.status == "active"
        for pre in s.prerequisites
        if library[pre].status != "active"
    )
    assert bad == []


def test_every_active_skill_can_be_reached_from_a_starting_point(library):
    starts = {s.id for s in library.values() if s.status == "active" and not s.prerequisites}
    for s in library.values():
        if s.status == "active" and s.prerequisites:
            assert graph.prerequisite_closure(s.id, library) & starts, s.id


# -- helpers --------------------------------------------------------------------------------------

CHAIN = [
    _skill("count"),
    _skill("add", prerequisites=["count"]),
    _skill("subtract", prerequisites=["count"]),
    _skill("multiply", prerequisites=["add"]),
    _skill("fractions", prerequisites=["multiply", "subtract"]),
    _skill("ratios", status="deferred", prerequisites=["fractions"]),
]


def test_closure_is_every_ancestor_and_not_the_skill():
    assert graph.prerequisite_closure("fractions", CHAIN) == {"multiply", "add", "subtract", "count"}
    assert graph.prerequisite_closure("count", CHAIN) == set()


def test_closure_accepts_a_dict_and_rejects_an_unknown_id():
    assert graph.prerequisite_closure("multiply", {s.id: s for s in CHAIN}) == {"add", "count"}
    with pytest.raises(KeyError):
        graph.prerequisite_closure("nope", CHAIN)


def test_ready_to_start_needs_every_direct_prerequisite_mastered():
    ids = lambda mastered: [s.id for s in graph.ready_to_start(CHAIN, mastered)]  # noqa: E731
    assert ids([]) == ["count"]
    assert ids(["count"]) == ["add", "subtract"]
    assert ids(["count", "add"]) == ["subtract", "multiply"]
    assert ids(["count", "add", "multiply"]) == ["subtract"]
    # Deferred skills are never offered, even when everything before them is mastered.
    assert ids(["count", "add", "subtract", "multiply", "fractions"]) == []


def test_gaps_before_lists_unmastered_ancestors_earliest_first():
    assert graph.gaps_before("fractions", CHAIN, []) == ["count", "add", "subtract", "multiply"]
    assert graph.gaps_before("fractions", CHAIN, ["count", "subtract"]) == ["add", "multiply"]
    assert graph.gaps_before("fractions", CHAIN, ["count", "add", "subtract", "multiply"]) == []


def test_gaps_before_puts_a_skill_after_what_it_builds_on_in_the_library(library):
    target = max(
        (s for s in library.values() if s.status == "active"),
        key=lambda s: len(graph.prerequisite_closure(s.id, library)),
    )
    order = graph.gaps_before(target.id, library, [])
    seen: set[str] = set()
    for sid in order:
        assert all(pre in seen or pre not in order for pre in library[sid].prerequisites), sid
        seen.add(sid)
