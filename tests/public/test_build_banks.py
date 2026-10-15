"""scripts/build_banks.py: the release question banks parse, match their manifest and come out the same every time."""

import dataclasses
import io
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

from mathgen import service
from scripts import build_banks

MCQ = "ucv1-finite-subgroup-test"
NUMBER_MULTIPART = "cyclic-group-element-order"  # a number skill whose exam tier is a multi-part question
FIGURE = "count-to-5"
SENIOR = "senior-convert-between-cartesian-and-polar-form"
SKILLS = [MCQ, NUMBER_MULTIPART, FIGURE, SENIOR]
QTI_NS = "{http://www.imsglobal.org/xsd/imsqti_v2p1}"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("banks")
    outs = {}
    for name, jobs in (("one", 1), ("two", 2)):
        out = root / name
        code = build_banks.main(
            ["--out", str(out), "--skills", ",".join(SKILLS), "--per-difficulty", "3", "--jobs", str(jobs)]
        )
        assert code == 0
        outs[name] = out
    return outs


def archive(out: Path, kind: str) -> zipfile.ZipFile:
    (path,) = out.glob(f"maths-engine-*-{kind}-banks.zip")
    return zipfile.ZipFile(path)


def tree(out: Path) -> dict[str, bytes]:
    return {str(p.relative_to(out)): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}


def test_files_exist_and_parse(built):
    out = built["one"]
    manifest = json.loads(archive(out, "moodle").read("manifest.json"))
    keys = [(e["domain"], e["id"]) for e in manifest["skills"]]
    assert keys == sorted(keys)
    assert {e["id"] for e in manifest["skills"]} == set(SKILLS)
    for e in manifest["skills"]:
        xml_file = out / e["file"]
        assert xml_file.is_file() and e["file"] == f"moodle/{e['domain']}/{e['id']}.xml"
        assert ET.parse(xml_file).getroot().tag == "quiz"
        qti_file = out / f"qti/{e['domain']}/{e['id']}.zip"
        with zipfile.ZipFile(qti_file) as inner:
            assert "imsmanifest.xml" in inner.namelist()
            for name in inner.namelist():
                if name.endswith((".xml", ".svg")):
                    ET.fromstring(inner.read(name))


def test_manifest_matches_the_banks(built):
    out = built["one"]
    moodle = json.loads(archive(out, "moodle").read("manifest.json"))
    qti = json.loads(archive(out, "qti").read("manifest.json"))
    assert moodle["per_difficulty"] == 3 and moodle["engine_version"] == qti["engine_version"]
    by_id = {e["id"]: e for e in moodle["skills"]}
    assert by_id[NUMBER_MULTIPART]["difficulties"] == ["easy", "medium", "hard", "exam"]
    assert by_id[NUMBER_MULTIPART]["questions"] == 12
    assert by_id[FIGURE]["difficulties"] == ["easy"] and by_id[FIGURE]["questions"] == 3
    assert by_id[SENIOR]["curriculum_codes"] and by_id[SENIOR]["years"] is not None
    for e in moodle["skills"]:
        root = ET.fromstring(archive(out, "moodle").read(e["file"]))
        assert sum(1 for q in root if q.get("type") != "category") == e["questions"]
        assert e["questions"] == 3 * len(e["difficulties"])
    for e in qti["skills"]:
        with zipfile.ZipFile(io.BytesIO(archive(out, "qti").read(e["file"]))) as inner:
            refs = ET.fromstring(inner.read("test.xml")).iter(f"{QTI_NS}assessmentItemRef")
            assert len(list(refs)) == e["questions"]
    assert [e["id"] for e in moodle["skills"]] == [e["id"] for e in qti["skills"]]


def test_bank_titles_and_sections(built):
    out = built["one"]
    moodle = (out / "moodle/abstract-algebra" / f"{NUMBER_MULTIPART}.xml").read_text(encoding="utf-8")
    assert "$course$/top/Maths Engine/abstract-algebra/Find an element order in a small cyclic group/Exam" in moodle
    assert 'type="cloze"' in moodle  # the multi-part exam questions
    with zipfile.ZipFile(out / "qti/abstract-algebra" / f"{NUMBER_MULTIPART}.zip") as inner:
        test = ET.fromstring(inner.read("test.xml"))
    assert test.get("title") == "Find an element order in a small cyclic group"
    assert [s.get("title") for s in test.iter(f"{QTI_NS}assessmentSection")] == ["Easy", "Medium", "Hard", "Exam"]


def test_figures_are_embedded(built):
    out = built["one"]
    domain = json.loads(archive(out, "moodle").read("manifest.json"))
    domain = next(e["domain"] for e in domain["skills"] if e["id"] == FIGURE)
    with zipfile.ZipFile(out / f"qti/{domain}/{FIGURE}.zip") as inner:
        assert any(n.startswith("figures/") and n.endswith(".svg") for n in inner.namelist())
    assert "@@PLUGINFILE@@/figure-1.svg" in (out / f"moodle/{domain}/{FIGURE}.xml").read_text(encoding="utf-8")


def test_readme_carries_the_attribution(built):
    for kind in ("moodle", "qti"):
        readme = archive(built["one"], kind).read("README.txt").decode("utf-8")
        assert service.SKILLS_ATTRIBUTION in readme
        assert service.ACARA_ATTRIBUTION in readme  # the senior skill is in the set
        assert "Engine version:" in readme and "HOW TO IMPORT" in readme


def test_same_bytes_on_every_run_and_for_any_job_count(built, tmp_path):
    again = tmp_path / "again"
    assert build_banks.main(["--out", str(again), "--skills", ",".join(SKILLS), "--per-difficulty", "3"]) == 0
    assert tree(built["one"]) == tree(again)
    assert tree(built["one"]) == tree(built["two"])
    names = archive(built["one"], "qti").namelist()
    assert names == sorted(names)
    assert {i.date_time for i in archive(built["one"], "qti").infolist()} == {(1980, 1, 1, 0, 0, 0)}


def test_seed_override_changes_the_questions(built, tmp_path):
    other = tmp_path / "other"
    assert build_banks.main(["--out", str(other), "--skills", MCQ, "--per-difficulty", "3", "--seed", "5"]) == 0
    manifest = json.loads(archive(other, "moodle").read("manifest.json"))
    assert manifest["seed"] == 5 and [e["id"] for e in manifest["skills"]] == [MCQ]
    (default,) = (built["one"] / "moodle").rglob(f"{MCQ}.xml")
    (changed,) = (other / "moodle").rglob(f"{MCQ}.xml")
    assert default.read_bytes() != changed.read_bytes()


def test_rubric_skills_are_left_out(monkeypatch):
    lib = service.Library.load()
    base = lib.by_id[MCQ]
    rubric = dataclasses.replace(base, id="zz-rubric-skill", answer="rubric")
    lib.skills.append(rubric)
    lib.by_id[rubric.id] = rubric
    ids, notes = build_banks.select_skills(lib, None, None)
    assert rubric.id not in ids and MCQ in ids
    ids, notes = build_banks.select_skills(lib, [rubric.id, MCQ], None)
    assert ids == [MCQ] and any(rubric.id in n for n in notes)


def test_a_failing_skill_is_reported_not_dropped(tmp_path, monkeypatch, capsys):
    real = build_banks.build_one

    def broken(skill_id, per_difficulty, seed):
        if skill_id == MCQ:
            return {"id": skill_id, "error": "VerificationError: boom"}
        return real(skill_id, per_difficulty, seed)

    monkeypatch.setattr(build_banks, "build_one", broken)
    code = build_banks.main(["--out", str(tmp_path), "--skills", f"{MCQ},{FIGURE}", "--per-difficulty", "2"])
    assert code == 1
    err = capsys.readouterr().err
    assert MCQ in err and "boom" in err
    manifest = json.loads(archive(tmp_path, "moodle").read("manifest.json"))
    assert [e["id"] for e in manifest["skills"]] == [FIGURE]


def test_bad_arguments_are_usage_errors(tmp_path):
    for args in (
        ["--skills", "no-such-skill"],
        ["--domain", "no-such-domain"],
        ["--per-difficulty", "0"],
        ["--jobs", "0"],
    ):
        with pytest.raises(SystemExit) as stopped:
            build_banks.main(["--out", str(tmp_path), *args])
        assert stopped.value.code == 2
