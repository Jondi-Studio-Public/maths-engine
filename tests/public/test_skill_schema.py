"""skills/skill.schema.json: the written-down shape of a skill file.

The library must satisfy it, bad labels must not, and its field lists must not drift from what the loader reads.
`jsonschema` is a dev dependency only; the package itself never imports it.
"""

import ast
import inspect
import json
from pathlib import Path

import pytest
import yaml

jsonschema = pytest.importorskip("jsonschema")

from mathgen import skills as skills_mod  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SKILLS = ROOT / "skills"
SCHEMA_PATH = SKILLS / "skill.schema.json"

GOOD = """\
id: add-two
name: Add two
domain: arithmetic
description: Add two small numbers.
status: active
prerequisites: []
answer: number
sources:
- topic: addition
  params: {}
  difficulty: easy
  stages: [0, 1]
misconceptions:
- id: subtracts
  description: Subtracts instead.
tags: []
"""


@pytest.fixture(scope="module")
def schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def validator(schema):
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def skill_files():
    return sorted(p for p in SKILLS.glob("*/*.yaml"))


def errors(validator, data):
    return sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))


def paths_of(errs):
    return [list(e.absolute_path) for e in errs]


def test_the_schema_is_a_valid_draft_2020_12_schema(schema):
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    jsonschema.Draft202012Validator.check_schema(schema)


def test_every_skill_file_validates(validator):
    files = skill_files()
    assert len(files) > 700
    bad = {}
    for path in files:
        errs = errors(validator, yaml.safe_load(path.read_text(encoding="utf-8")))
        if errs:
            bad[f"{path.parent.name}/{path.name}"] = [f"{list(e.absolute_path)}: {e.message[:120]}" for e in errs[:3]]
    assert bad == {}


def test_the_baseline_skill_validates(validator):
    assert errors(validator, yaml.safe_load(GOOD)) == []


DELETE = object()


def mutate(**changes):
    data = yaml.safe_load(GOOD)
    for key, value in changes.items():
        if value is DELETE:
            data.pop(key, None)
        else:
            data[key] = value
    return data


def test_a_source_difficulty_outside_the_four_tiers_is_rejected(validator):
    for label in ("integers", "Easy", "", None, 1):
        data = mutate(sources=[{"topic": "addition", "difficulty": label}])
        assert paths_of(errors(validator, data)) == [["sources", 0, "difficulty"]], label
    for label in ("easy", "medium", "hard", "exam"):
        assert errors(validator, mutate(sources=[{"topic": "addition", "difficulty": label}])) == []


def test_an_active_skill_needs_a_source_and_a_difficulty_on_each(validator):
    assert errors(validator, mutate(sources=[]))
    assert errors(validator, mutate(sources=DELETE))
    missing = mutate(sources=[{"topic": "addition", "params": {}}])
    assert paths_of(errors(validator, missing)) == [["sources", 0]]


def test_a_deferred_skill_needs_a_reason_but_not_a_source(validator):
    deferred = mutate(status="deferred", sources=[])
    assert errors(validator, deferred)
    deferred["deferred_reason"] = "generator"
    assert errors(validator, deferred) == []
    deferred["deferred_reason"] = "because"
    assert paths_of(errors(validator, deferred)) == [["deferred_reason"]]


def test_an_unknown_status_is_rejected(validator):
    assert paths_of(errors(validator, mutate(status="live"))) == [["status"]]
    assert paths_of(errors(validator, mutate(status="Active"))) == [["status"]]


def test_a_misspelt_or_unknown_field_is_rejected_at_every_level(validator):
    assert errors(validator, mutate(prerequisite=[]))
    data = mutate(sources=[{"topic": "addition", "difficulty": "easy", "param": {}}])
    assert paths_of(errors(validator, data)) == [["sources", 0]]
    data = mutate(misconceptions=[{"id": "x", "description": "y", "tone": "z"}])
    assert paths_of(errors(validator, data)) == [["misconceptions", 0]]


def test_stages_and_learner_band_are_checked(validator):
    data = mutate(sources=[{"topic": "addition", "difficulty": "easy", "stages": [7]}])
    assert paths_of(errors(validator, data)) == [["sources", 0, "stages", 0]]
    assert errors(validator, mutate(learner_band="advanced"))


def test_alignment_rules_match_the_validator(validator):
    verified = {"source_id": "x", "status": "verified", "framework": "f", "area": "a"}
    assert errors(validator, mutate(alignments=[verified]))  # no verified_on
    assert errors(validator, mutate(alignments=[{**verified, "verified_on": "2026-01-31"}])) == []
    assert errors(validator, mutate(alignments=[{"source_id": "x", "framework": "f"}]))  # no area
    assert errors(validator, mutate(alignments=[{"source_id": "x", "status": "maybe"}]))


def test_mental_and_figure_rules_match_the_validator(validator):
    level = {"topic": "addition", "params": {}, "target_s": 3, "limit_s": 8}
    assert errors(validator, mutate(mental={"levels": [level]})) == []
    assert errors(validator, mutate(mental={"levels": []}))
    assert errors(validator, mutate(mental={"levels": [{**level, "target_s": 0}]}))
    assert errors(validator, mutate(mental={"levels": [level] * 6}))
    assert errors(validator, mutate(needs_figure=True, mental={"levels": [level]}))
    assert errors(validator, mutate(needs_figure=True)) == []


def test_a_rubric_skill_needs_a_rubric_block(validator):
    assert errors(validator, mutate(answer="rubric"))
    assert errors(validator, mutate(answer="rubric", rubric={"id": "proof-v1", "high_stakes": True})) == []


def test_the_validator_agrees_with_the_schema_on_a_bad_label(tmp_path, validator):
    """The same bad file is an error to both, so the two rule sets cannot quietly disagree on this label."""
    root = tmp_path / "skills"
    (root / "arithmetic").mkdir(parents=True)
    bad = GOOD.replace("difficulty: easy", "difficulty: integers")
    (root / "arithmetic" / "add-two.yaml").write_text(bad, encoding="utf-8")
    assert errors(validator, yaml.safe_load(bad))
    domains, loaded = skills_mod.load_skills(root)
    assert any("difficulty" in e for e in skills_mod.validate(domains, loaded).errors)


# -- the schema's field lists against the fields the loader reads -------------------------------------------------


def keys_read(function_name, variable):
    """String keys the named loader function reads from `variable`, as variable.get("k") or variable["k"]."""
    tree = ast.parse(inspect.getsource(skills_mod))
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function_name)
    found = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            target, args = node.func.value, node.args
        elif isinstance(node, ast.Subscript):
            target, args = node.value, [node.slice]
        else:
            continue
        if isinstance(target, ast.Name) and target.id == variable and args and isinstance(args[0], ast.Constant):
            found.add(args[0].value)
    return found


NOT_READ_BY_THE_LOADER = {"difficulty_progression"}  # prose for authors, documented in the schema


def test_the_top_level_fields_are_exactly_the_ones_the_loader_reads(schema):
    read = keys_read("_skill_from_dict", "d")
    assert read, "the AST probe found nothing; has _skill_from_dict been restructured?"
    assert set(schema["properties"]) - NOT_READ_BY_THE_LOADER == read


@pytest.mark.parametrize(
    "definition, function_name, variable",
    [
        ("source", "_source_from_dict", "d"),
        ("alignment", "_alignment_from_dict", "d"),
        ("mental_level", "_mental_level_from_dict", "d"),
        ("mental", "_mental_from_dict", "d"),
        ("rubric", "_rubric_block_from_dict", "d"),
    ],
)
def test_each_nested_object_lists_exactly_the_fields_the_loader_reads(schema, definition, function_name, variable):
    read = keys_read(function_name, variable)
    assert read, f"the AST probe found nothing in {function_name}"
    assert set(schema["$defs"][definition]["properties"]) == read


def test_misconceptions_and_near_misses_list_the_fields_the_loader_reads(schema):
    # Both are built inline in _skill_from_dict, from comprehension variables `m` and `nm`.
    assert keys_read("_skill_from_dict", "m") == keys_read("_skill_from_dict", "nm") == {"id", "description"}
    assert set(schema["$defs"]["note"]["properties"]) == {"id", "description"}


def test_the_schema_enums_match_the_loader_constants(schema):
    assert set(schema["properties"]["deferred_reason"]["enum"]) == skills_mod.VALID_DEFERRED_REASONS - {"invalid"}
    assert set(schema["$defs"]["alignment"]["properties"]["status"]["enum"]) == skills_mod.VALID_ALIGNMENT_STATUSES
    assert set(schema["$defs"]["source"]["properties"]["stages"]["items"]["enum"]) == skills_mod.VALID_STAGES
    assert schema["$defs"]["source"]["properties"]["difficulty"]["enum"] == list(skills_mod.VALID_DIFFICULTIES)
