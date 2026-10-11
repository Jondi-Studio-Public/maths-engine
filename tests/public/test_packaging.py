"""pyproject.toml: the wheel must carry the package, every subpackage and the whole skill library.

Building a wheel needs the network (an isolated build environment), so these read the configuration and compare
it with the tree instead. The install itself is exercised by CI's `pip install -e` and by hand with `pip wheel`.
"""

import fnmatch
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_every_package_folder_is_listed_so_the_wheel_is_not_missing_a_module(pyproject):
    on_disk = {".".join(init.parent.relative_to(ROOT).parts) for init in (ROOT / "mathgen").rglob("__init__.py")}
    listed = set(pyproject["tool"]["setuptools"]["packages"])
    assert listed - {"mathgen._skills"} == on_disk


def test_the_skill_library_is_mapped_into_the_package_without_moving_it(pyproject):
    setuptools = pyproject["tool"]["setuptools"]
    assert setuptools["package-dir"] == {"mathgen._skills": "skills"}
    assert "mathgen._skills" in setuptools["packages"]
    assert (ROOT / "skills" / "domains.yaml").is_file()
    assert not (ROOT / "mathgen" / "_skills").exists(), "the in-package copy exists only inside a built wheel"


def test_package_data_globs_cover_every_yaml_file_in_the_library(pyproject):
    """A skill nested deeper than the globs reach would be left out of the wheel without any warning."""
    globs = pyproject["tool"]["setuptools"]["package-data"]["mathgen._skills"]
    files = sorted(p.relative_to(ROOT / "skills").as_posix() for p in (ROOT / "skills").rglob("*.yaml"))
    assert len(files) > 700
    # setuptools' `*` does not cross a folder boundary, which fnmatch's does not model; count the separators.
    missed = [f for f in files if not any(fnmatch.fnmatch(f, g) and f.count("/") == g.count("/") for g in globs)]
    assert missed == []
    # With the default (true) setuptools warns about every domain folder, as a package it was not told about.
    assert pyproject["tool"]["setuptools"]["include-package-data"] is False


def test_the_skill_schema_ships_with_the_skill_library(pyproject):
    """The schema sits beside domains.yaml, so the wheel carries it as mathgen/_skills/skill.schema.yaml."""
    globs = pyproject["tool"]["setuptools"]["package-data"]["mathgen._skills"]
    assert (ROOT / "skills" / "skill.schema.yaml").is_file()
    assert any(fnmatch.fnmatch("skill.schema.yaml", g) and "/" not in g for g in globs)


def test_jsonschema_is_a_dev_dependency_only(pyproject):
    """The schema is for editors and tests; the package must run without jsonschema installed."""
    assert not any(d.lower().startswith("jsonschema") for d in pyproject["project"]["dependencies"])
    assert any(d.lower().startswith("jsonschema") for d in pyproject["project"]["optional-dependencies"]["dev"])
    runtime_sources = [p for p in (ROOT / "mathgen").rglob("*.py") if "jsonschema" in p.read_text(encoding="utf-8")]
    assert runtime_sources == []


def test_the_distribution_is_not_named_after_the_taken_pypi_project(pyproject):
    project = pyproject["project"]
    assert project["name"] == "maths-engine"  # `mathgen` on PyPI is an unrelated project
    assert pyproject["project"]["scripts"] == {"mathgen": "mathgen.__main__:main"}


def test_metadata_is_complete(pyproject):
    project = pyproject["project"]
    assert project["license"] == "MIT AND CC-BY-4.0"
    for name in project["license-files"]:
        assert (ROOT / name).is_file(), name
    assert (ROOT / project["readme"]).is_file()
    assert project["authors"] and project["classifiers"] and project["description"]
    assert set(project["urls"]) >= {"Homepage", "Source", "Issues"}
    assert all(
        url.startswith("https://github.com/Jondi-Studio-Public/maths-engine") for url in project["urls"].values()
    )
    assert not any(c.startswith("License ::") for c in project["classifiers"]), "use the SPDX `license` field only"
    setuptools_floor = pyproject["build-system"]["requires"][0]
    assert setuptools_floor.startswith("setuptools>=") and tuple(map(int, setuptools_floor[12:].split("."))) >= (
        77,
        0,
        3,
    )


def test_dependency_floors_are_versions_the_generators_were_checked_against(pyproject):
    deps = {d.split(">=")[0]: d.split(">=")[1] for d in pyproject["project"]["dependencies"]}
    as_tuple = lambda v: tuple(int(x) for x in v.split("."))  # noqa: E731
    assert as_tuple(deps["sympy"]) >= (1, 12, 1)  # 1.12.0 cannot parse p11c_factorial_notation's output
    assert as_tuple(deps["pyyaml"]) >= (6, 0, 1)  # 6.0 does not build on Python 3.12


def test_a_bare_pytest_can_import_scripts_and_tests(pyproject):
    assert pyproject["tool"]["pytest"]["ini_options"]["pythonpath"] == ["."]


def test_the_sdist_carries_every_folder_the_tests_read():
    """An unpacked sdist runs its own tests; `tests/test_ci_*.py` read `.github/`."""
    lines = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    grafted = {line.split()[1] for line in lines if line.startswith("graft ")}
    folders = {name for name in (".github", "skills", "scripts", "tests") if (ROOT / name).is_dir()}
    assert folders and folders <= grafted
