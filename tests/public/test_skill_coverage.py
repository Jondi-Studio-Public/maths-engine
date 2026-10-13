"""year_level, naplan_domain and exam_board: read by the loader, checked by validate(), and kept equal to the
curriculum codes in each file (scripts/backfill_year_level.py is the one place the rule lives)."""

from pathlib import Path

import pytest
import yaml

from mathgen import skills as skills_mod
from mathgen.skills import Domain, _skill_from_dict, validate
from scripts import backfill_year_level as backfill

SKILLS = Path(__file__).resolve().parents[2] / "skills"


def test_the_library_agrees_with_what_its_codes_derive():
    bad = []
    for path in sorted(SKILLS.glob("*/*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        want = backfill.derive(data)
        have = backfill.current(data)
        # A skill with no code or tag evidence may set the fields by hand; one with evidence may not disagree.
        if any(have.get(name) != values for name, values in want.items()):
            bad.append(f"{path.parent.name}/{path.name}: has {have}, derives {want}")
    assert bad == []


def test_a_naplan_domain_only_comes_with_a_year_between_f_and_9():
    for path in sorted(SKILLS.glob("*/*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data.get("naplan_domain"):
            years = {str(y) for y in data.get("year_level") or []}
            assert years & set(skills_mod.YEAR_LEVELS[:10]), path.name


@pytest.mark.parametrize(
    ("tags", "years", "domains"),
    [
        (["ac9mfn05", "ac9m1n05"], ["F", "1"], ["number"]),
        (["AC9M10ST02"], ["10"], []),  # Year 10 is beyond NAPLAN
        (["ac9m10ap01"], ["10"], []),  # 10A optional content
        (["ac9m9sp01", "ac9m10a01"], ["9", "10"], ["space"]),
        (["acara-v9-foundation"], ["F"], []),
        (["acara-v9-year-7-year-8"], ["7", "8"], []),
        (["acara-v9-year-9-10"], ["9", "10"], []),
        (["acara-senior-v8-4", "acmmm070"], ["11"], []),
        (["acmsm077", "ACMSM144"], ["12"], []),
        (["acmgm047", "acmgm048"], ["11", "12"], []),
        (["acmmm040", "acmmm181", "acmxx001"], [], []),  # absent, past the last code, unknown subject
        (["university-core-v1"], ["university"], []),
    ],
)
def test_the_derivation_rules(tags, years, domains):
    got = backfill.derive({"id": "x", "tags": tags})
    assert got.get("year_level", []) == years
    assert got.get("naplan_domain", []) == domains


def test_the_loader_reads_the_fields_as_string_tuples():
    skill = _skill_from_dict({"id": "x", "year_level": ["F", 1, "10"], "naplan_domain": ["number"]}, Path("x.yaml"))
    assert skill.year_level == ("F", "1", "10")
    assert skill.naplan_domain == ("number",)
    assert skill.exam_board == ()
    assert _skill_from_dict({"id": "x"}, Path("x.yaml")).year_level == ()


def problems(**fields):
    skill = _skill_from_dict(
        {"id": "x", "name": "X", "domain": "d", "status": "deferred", "deferred_reason": "hard", **fields},
        Path("d/x.yaml"),
    )
    report = validate([Domain(id="d", name="D", order=1)], [skill])
    return [e for e in report.errors if any(k in e for k in ("year_level", "naplan_domain", "exam_board"))]


def test_validate_accepts_good_labels_and_rejects_bad_ones():
    assert (
        problems(year_level=["F", "1", "12", "university"], naplan_domain=["number"], exam_board=["nesa", "ib"]) == []
    )
    assert problems(year_level=["13"])
    assert problems(year_level=["1", "F"])  # out of order
    assert problems(year_level=["1", "1"])  # duplicate
    assert problems(naplan_domain=["geometry"])
    assert problems(exam_board=["hsc"])


@pytest.mark.parametrize(
    ("code", "unit"),
    [
        ("ACMMM001", 1), ("ACMMM060", 1), ("ACMMM061", 2), ("acmmm097", 2), ("ACMMM098", 3), ("ACMMM150", 3),
        ("ACMMM151", 4), ("ACMMM180", 4), ("ACMSM041", 1), ("ACMSM042", 2), ("ACMSM076", 2), ("ACMSM077", 3),
        ("ACMSM115", 3), ("ACMSM116", 4), ("ACMSM144", 4), ("ACMGM025", 1), ("ACMGM026", 2), ("ACMGM047", 2),
        ("ACMGM048", 3), ("ACMGM086", 3), ("ACMGM087", 4), ("ACMGM111", 4), ("ACMEM042", 1), ("ACMEM043", 2),
        ("ACMEM089", 2), ("ACMEM090", 3), ("ACMEM147", 3), ("ACMEM148", 4), ("ACMEM175", 4),
        ("ACMMM040", None), ("ACMSM097", None), ("ACMMM181", None), ("ACMEM176", None), ("ACMMM000", None),
        ("ACMXX001", None), ("ac9m6n03", None),
    ],
)  # fmt: skip
def test_senior_code_to_unit(code, unit):
    assert backfill.senior_unit(code) == unit
    assert backfill.senior_year(code) == {1: "11", 2: "11", 3: "12", 4: "12", None: None}[unit]


def test_the_unit_table_is_contiguous_and_ordered():
    for prefix, ranges in backfill.SENIOR["ranges"].items():
        assert [u for _, _, u in ranges] == [1, 2, 3, 4], prefix
        assert ranges[0][0] == 1, prefix
        for (_, hi, _), (lo, _, _) in zip(ranges, ranges[1:], strict=False):
            assert lo == hi + 1, prefix
