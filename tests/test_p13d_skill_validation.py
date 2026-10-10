"""The live loader validates the WHOLE skill library, and the report has no errors."""

from mathgen import skills as skills_mod


def test_whole_library_passes_live_loader_validation():
    domains, loaded = skills_mod.load_skills()
    report = skills_mod.validate(domains, loaded)
    assert report.ok, report.errors
