"""Loading the skill library: where it is found, and that a missing or broken one is an error, not an empty result."""

import pytest

from mathgen import _paths, skills as skills_mod
from mathgen.skills import DuplicateSkillError, SkillsNotFoundError, load_problems, load_skills, validate

SKILL = """\
id: add-two
name: Add two
domain: arithmetic
status: {status}
{extra}prerequisites: []
answer: {answer}
sources:
  - topic: addition
    params: {{}}
    difficulty: easy
"""


def library(root, **files):
    (root / "arithmetic").mkdir(parents=True)
    for stem, text in files.items():
        (root / "arithmetic" / f"{stem}.yaml").write_text(text, encoding="utf-8")
    return root


def good(status="active", extra="", answer="number"):
    return SKILL.format(status=status, extra=extra, answer=answer)


# -- where the library is -----------------------------------------------------------------------


def test_a_checkout_uses_the_skills_folder_beside_the_package():
    assert _paths.default_skills_dir() == _paths.checkout_skills_dir()
    assert (_paths.checkout_skills_dir() / "domains.yaml").is_file()
    assert skills_mod.skills_root() == _paths.checkout_skills_dir()


def test_the_environment_variable_overrides_the_default(monkeypatch, tmp_path):
    monkeypatch.setenv("MATHS_SKILLS_DIR", str(tmp_path))
    assert skills_mod.skills_root() == tmp_path
    monkeypatch.setenv("MATHS_SKILLS_DIR", "")  # empty means unset
    assert skills_mod.skills_root() == skills_mod.DEFAULT_SKILLS_DIR


def test_an_unrelated_skills_folder_is_not_mistaken_for_the_library(monkeypatch, tmp_path):
    """No domains.yaml, so it is not ours; with no packaged copy either, the checkout path comes back."""
    (tmp_path / "skills").mkdir()
    monkeypatch.setattr(_paths, "_PACKAGE_DIR", tmp_path / "mathgen")
    monkeypatch.setattr(_paths, "packaged_skills_dir", lambda: None)
    assert _paths.default_skills_dir() == tmp_path / "skills"
    (tmp_path / "skills" / "domains.yaml").write_text("domains: []\n")
    assert _paths.default_skills_dir() == tmp_path / "skills"


def test_the_packaged_copy_is_used_when_there_is_no_checkout(monkeypatch, tmp_path):
    packaged = tmp_path / "mathgen" / "_skills"
    packaged.mkdir(parents=True)
    monkeypatch.setattr(_paths, "_PACKAGE_DIR", tmp_path / "mathgen")
    monkeypatch.setattr(_paths, "packaged_skills_dir", lambda: packaged)
    assert _paths.default_skills_dir() == packaged
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills" / "domains.yaml").write_text("domains: []\n")
    assert _paths.default_skills_dir() == tmp_path / "skills"  # a checkout wins


def test_the_capability_gate_registry_is_found_with_the_skills():
    from mathgen import capabilities

    assert capabilities.DEFAULT_CAPABILITY_GATES_FILE == _paths.default_skills_dir() / "capability-gates.yaml"
    assert capabilities.load_capability_gates()


# -- an empty library is an error ---------------------------------------------------------------


def test_load_skills_raises_when_the_folder_is_missing_or_empty(tmp_path):
    with pytest.raises(SkillsNotFoundError, match="no skills found under .*MATHS_SKILLS_DIR"):
        load_skills(tmp_path / "missing")
    with pytest.raises(SkillsNotFoundError):
        load_skills(tmp_path)
    (tmp_path / "arithmetic").mkdir()  # a domain folder with no skills is still empty
    with pytest.raises(SkillsNotFoundError):
        load_skills(tmp_path)
    assert issubclass(SkillsNotFoundError, FileNotFoundError)


def test_validate_does_not_pass_an_empty_library():
    report = validate([], [])
    assert not report.ok and "no skills were loaded" in report.errors[0]


# -- a file that did not load -------------------------------------------------------------------


def test_load_problems_names_each_kind_of_unloadable_file(tmp_path):
    root = library(
        tmp_path,
        **{
            "add-two": good(),
            "bad-yaml": "id: [unclosed\n",
            "not-a-mapping": "- just\n- a list\n",
            "empty": "",
            "renamed": good(),  # id add-two inside renamed.yaml
            "wrong-domain": good().replace("add-two", "wrong-domain").replace("domain: arithmetic", "domain: other"),
        },
    )
    _, loaded = load_skills(root)
    problems = load_problems(loaded)
    assert len(problems) == 5, problems
    by_file = {p.split(":")[0]: p for p in problems}
    assert "failed to load bad-yaml.yaml" in by_file["arithmetic/bad-yaml.yaml"]
    assert "failed to load not-a-mapping.yaml" in by_file["arithmetic/not-a-mapping.yaml"]
    assert "missing id" in by_file["arithmetic/empty.yaml"]
    assert "id 'add-two' != filename stem 'renamed'" in by_file["arithmetic/renamed.yaml"]
    assert "domain 'other' != folder 'arithmetic'" in by_file["arithmetic/wrong-domain.yaml"]
    assert "arithmetic/add-two.yaml" not in by_file


def test_a_valid_deferred_skill_is_not_a_load_problem(tmp_path):
    root = library(tmp_path, **{"add-two": good("deferred", "deferred_reason: hard\n")})
    _, loaded = load_skills(root)
    assert loaded[0].status == "deferred" and loaded[0].deferred_reason == "hard"
    assert load_problems(loaded) == []


def test_the_shipped_library_has_no_load_problems():
    assert load_problems(load_skills()[1]) == []


# -- source difficulty tiers -------------------------------------------------------------------


def errors_for(tmp_path, text):
    """Only the tier errors: whether `addition` has a preset of the label is a separate, generation error."""
    domains, loaded = load_skills(library(tmp_path, **{"add-two": text}))
    return [e for e in validate(domains, loaded).errors if "difficulty" in e and "source" in e and "failed" not in e]


@pytest.mark.parametrize("label", ["easy", "medium", "hard", "exam"])
def test_the_four_tiers_are_accepted_as_a_source_difficulty(tmp_path, label):
    assert errors_for(tmp_path, good().replace("difficulty: easy", f"difficulty: {label}")) == []


def test_an_active_source_without_a_difficulty_is_reported(tmp_path):
    errors = errors_for(tmp_path, good().replace("    difficulty: easy\n", ""))
    assert errors == [
        "add-two: source 0 (topic='addition') has no difficulty; it must be one of easy, medium, hard, exam"
    ]


@pytest.mark.parametrize("label", ["integers", "Easy", "''"])
def test_an_active_source_with_another_difficulty_label_is_reported(tmp_path, label):
    errors = errors_for(tmp_path, good().replace("difficulty: easy", f"difficulty: {label}"))
    assert len(errors) == 1 and "source 0" in errors[0] and "must be one of easy, medium, hard, exam" in errors[0]


def test_only_the_offending_source_is_named(tmp_path):
    two = good() + "  - topic: addition\n    params: {}\n    difficulty: nonsense\n"
    errors = errors_for(tmp_path, two)
    assert len(errors) == 1 and "source 1" in errors[0] and "'nonsense'" in errors[0]


def test_a_deferred_skill_may_leave_the_difficulty_off(tmp_path):
    text = good("deferred", "deferred_reason: generator\n").replace("    difficulty: easy\n", "")
    assert errors_for(tmp_path, text) == []


def test_the_shipped_library_has_a_tier_on_every_active_source():
    """Cheap stand-in for validate() over the whole library, which generates questions and takes a minute."""
    _, loaded = load_skills()
    untiered = [
        f"{s.id}[{i}]={src.difficulty!r}"
        for s in loaded
        if s.status == "active"
        for i, src in enumerate(s.sources)
        if src.difficulty not in skills_mod.VALID_DIFFICULTIES
    ]
    assert untiered == []


# -- rubric hook --------------------------------------------------------------------------------

RUBRIC_SKILL = good(answer="rubric", extra="rubric: {id: proof-v1}\n")


def test_rubric_ids_are_resolved_only_by_a_loader_the_caller_supplies(tmp_path):
    root = library(
        tmp_path, **{"add-two": RUBRIC_SKILL.replace("status: active", "status: deferred\ndeferred_reason: hard")}
    )
    domains, loaded = load_skills(root)
    assert validate(domains, loaded).ok  # no loader, no api package consulted: the id is required, not resolved

    def missing(rubric_id):
        raise FileNotFoundError(rubric_id)

    report = validate(domains, loaded, rubric_loader=missing)
    assert report.errors == ["add-two: rubric 'proof-v1' is invalid: proof-v1"]
    assert validate(domains, loaded, rubric_loader=lambda rubric_id: object()).ok


def test_validate_does_not_import_an_application_api_package(monkeypatch, tmp_path):
    """An unrelated top-level `api` package on sys.path must not be imported by the validator."""
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "__init__.py").write_text("raise RuntimeError('imported an unrelated api package')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(__import__("sys").modules, "api", raising=False)
    root = library(tmp_path / "lib", **{"add-two": good()})
    domains, loaded = load_skills(root)
    assert validate(domains, loaded).ok
    assert "api" not in __import__("sys").modules


# -- extra skill folders ------------------------------------------------------------------------


def extra_folder(root, domain="arithmetic", stem="add-three"):
    (root / domain).mkdir(parents=True)
    text = good().replace("add-two", stem).replace("domain: arithmetic", f"domain: {domain}")
    (root / domain / f"{stem}.yaml").write_text(text, encoding="utf-8")
    return root


def test_an_extra_folder_adds_its_skills_to_the_library(tmp_path):
    root = library(tmp_path / "lib", **{"add-two": good()})
    extra = extra_folder(tmp_path / "private")
    domains, loaded = load_skills(root, [extra])
    assert [s.id for s in loaded] == ["add-two", "add-three"]
    assert [d.id for d in domains] == ["arithmetic"]
    assert validate(domains, loaded).ok


def test_extra_folders_come_from_the_environment_by_default(monkeypatch, tmp_path):
    root = library(tmp_path / "lib", **{"add-two": good()})
    first = extra_folder(tmp_path / "one")
    second = extra_folder(tmp_path / "two", stem="add-four")
    monkeypatch.setenv("MATHS_EXTRA_SKILLS_DIRS", f"{first}{__import__('os').pathsep}{second}")
    assert skills_mod.extra_skills_roots() == [first, second]
    assert [s.id for s in load_skills(root)[1]] == ["add-two", "add-three", "add-four"]
    assert [s.id for s in load_skills(root, [])[1]] == ["add-two"]  # an explicit empty list means none
    monkeypatch.setenv("MATHS_EXTRA_SKILLS_DIRS", "")
    assert skills_mod.extra_skills_roots() == []


def test_an_extra_folder_cannot_replace_a_library_skill(tmp_path):
    """Refused at load, not only in validate(): a caller that keys skills by id must never get the extra copy."""
    root = library(tmp_path / "lib", **{"add-two": good()})
    extra = extra_folder(tmp_path / "private", stem="add-two")
    with pytest.raises(DuplicateSkillError, match="'add-two' .* is already loaded"):
        load_skills(root, [extra])
    other = extra_folder(tmp_path / "other")  # add-three, twice across two extra folders
    with pytest.raises(DuplicateSkillError, match="'add-three'"):
        load_skills(root, [other, extra_folder(tmp_path / "again")])


def test_an_extra_skill_must_use_a_library_domain(tmp_path):
    root = library(tmp_path / "lib", **{"add-two": good()})
    extra = extra_folder(tmp_path / "private", domain="private-only")
    domains, loaded = load_skills(root, [extra])
    assert [d.id for d in domains] == ["arithmetic"]
    assert "add-three: domain 'private-only' not in domains.yaml" in validate(domains, loaded).errors


def test_a_missing_extra_folder_is_an_error_but_an_empty_one_is_not(tmp_path):
    root = library(tmp_path / "lib", **{"add-two": good()})
    with pytest.raises(
        FileNotFoundError, match="extra skills folder .*missing does not exist.*MATHS_EXTRA_SKILLS_DIRS"
    ):
        load_skills(root, [tmp_path / "missing"])
    (tmp_path / "empty").mkdir()
    assert [s.id for s in load_skills(root, [tmp_path / "empty"])[1]] == ["add-two"]


def test_a_home_relative_extra_folder_is_expanded(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MATHS_EXTRA_SKILLS_DIRS", "~/private")
    assert skills_mod.extra_skills_roots() == [tmp_path / "private"]


def test_extra_skills_are_copies_too(tmp_path):
    root = library(tmp_path / "lib", **{"add-two": good()})
    extra = extra_folder(tmp_path / "private")
    load_skills(root, [extra])[1][1].name = "changed"
    assert load_skills(root, [extra])[1][1].name == "Add two"
