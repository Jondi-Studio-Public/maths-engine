"""More regressions from the 2026-10-14 core bug hunt: bad skill files and an id seed at the top of the range."""

import pytest

from mathgen import service
from mathgen.skills import Domain, Skill, Source, validate


def _skill(sid, **extra):
    return Skill(
        id=sid,
        name=sid,
        domain="arithmetic",
        status="active",
        answer="number",
        sources=[Source(topic="addition", params={}, difficulty="easy", **extra)],
    )


DOMAINS = [Domain(id="arithmetic", name="Arithmetic", order=1, band="primary", band_name="Primary")]


@pytest.mark.parametrize("stages", [3, "all", {"a": 1}])
def test_a_source_whose_stages_is_not_a_list_is_reported_not_a_crash(stages):
    report = validate(DOMAINS, [_skill("a", stages=stages)], gate_registry={})
    assert any("stages must be a list" in e for e in report.errors)


def test_prerequisites_given_as_nested_lists_or_a_string_are_reported_not_a_crash(tmp_path):
    from mathgen.skills import _skill_from_dict

    nested = _skill_from_dict({"id": "a", "prerequisites": [["x"]]}, tmp_path)
    assert nested.prerequisites == ["['x']"]
    single = _skill_from_dict({"id": "b", "prerequisites": "abc"}, tmp_path)
    assert single.prerequisites == ["abc"]


def test_a_domain_entry_without_an_id_says_so(tmp_path):
    from mathgen.skills import _load_domains

    (tmp_path / "domains.yaml").write_text("domains:\n  - name: No id here\n")
    with pytest.raises(ValueError, match="needs an `id`"):
        _load_domains(tmp_path, [])


def test_a_seed_too_close_to_the_top_of_the_range_is_refused_up_front():
    lib = service.Library.load(id_secret="s" * 32)
    skill = next(s for s in lib.skills if s.status == "active" and s.sources)
    with pytest.raises(service.BadRequest, match="too close"):
        service.make_questions(lib, skill.id, count=3, seed=2**63 - 1)
    (view,) = service.make_questions(lib, skill.id, count=1, seed=2**63 - 100)
    service.question_by_id(lib, view["id"])
