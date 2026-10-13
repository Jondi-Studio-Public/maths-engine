"""Question ids: source keys, content hashes, `regenerate` and `QuestionChanged`."""

import copy
import re

import pytest

from mathgen import engine
from mathgen.skills import SOURCE_KEY_RE, Skill, Source, validate

SKILLS = {s.id: s for s in engine.load_skills()[1]}
SKILL = SKILLS["simplify-surds"]


def test_a_sources_key_defaults_to_its_difficulty():
    src = Source(topic="addition", difficulty="easy")
    assert src.effective_key == "easy"
    assert Source(topic="addition", difficulty="easy", key="small").effective_key == "small"
    assert Source(topic="addition").effective_key is None


def test_every_library_source_has_a_valid_unique_key():
    for skill in SKILLS.values():
        keys = [src.effective_key for src in skill.sources]
        assert len(keys) == len(set(keys)), skill.id
        assert all(k and SOURCE_KEY_RE.match(k) for k in keys), skill.id


def test_id_has_four_parts_and_round_trips():
    q, qid = engine.make_with_id(SKILL, 7)
    skill_id, key, seed, digest = engine.parse_question_id(qid)
    assert (skill_id, key, seed) == ("simplify-surds", SKILL.sources[0].effective_key, 7)
    assert re.fullmatch(r"[0-9a-f]{12}", digest) and digest == engine.content_hash(q)
    assert qid == engine.question_id(SKILL, 0, 7, q) == engine.question_id(SKILL, SKILL.sources[0], 7, q)
    assert engine.regenerate(qid, [SKILL]).to_dict() == q.to_dict()
    assert engine.regenerate(qid, SKILLS).to_dict() == q.to_dict()


def test_make_with_id_gives_the_same_question_as_make_for_skill():
    for source in (None, 0, SKILL.sources[-1].effective_key):
        q, _ = engine.make_with_id(SKILL, 3, source=source)
        assert q.to_dict() == engine.make_for_skill(SKILL, 3, source=source).to_dict()


def test_different_seeds_and_sources_give_different_ids():
    ids = {engine.make_with_id(SKILL, seed, source=i)[1] for seed in range(3) for i in range(len(SKILL.sources))}
    assert len(ids) == 3 * len(SKILL.sources)


def test_regenerate_detects_a_changed_question():
    _, qid = engine.make_with_id(SKILL, 5)
    head, digest = qid.rsplit("/", 1)
    tampered = f"{head}/{'0' * 12 if digest != '0' * 12 else '1' * 12}"
    with pytest.raises(engine.QuestionChanged) as caught:
        engine.regenerate(tampered, [SKILL])
    err = caught.value
    assert err.old_id == tampered and err.new_id == qid
    assert err.question.to_dict() == engine.make_for_skill(SKILL, 5).to_dict()


def test_regenerate_detects_an_edited_source():
    """Editing a source (here its difficulty, which seeds the generator) changes the content, not the key."""
    skill = copy.deepcopy(SKILL)
    key = skill.sources[0].effective_key
    _, qid = engine.make_with_id(skill, 5, source=0)
    skill.sources[0].key = key
    skill.sources[0].difficulty = "hard" if key != "hard" else "medium"
    with pytest.raises(engine.QuestionChanged) as caught:
        engine.regenerate(qid, [skill])
    assert caught.value.old_id == qid and caught.value.new_id.startswith(f"simplify-surds/{key}/5/")


def test_unknown_skill_and_key_and_malformed_ids():
    _, qid = engine.make_with_id(SKILL, 1)
    with pytest.raises(KeyError, match="unknown skill"):
        engine.regenerate(qid.replace("simplify-surds", "no-such-skill", 1), [SKILL])
    key = SKILL.sources[0].effective_key
    with pytest.raises(KeyError, match="no source with key"):
        engine.regenerate(qid.replace(f"/{key}/", "/no-such-key/", 1), [SKILL])
    for bad in ("", "a/b/c", "a/b/x/abc", "a/b/1/c/d", "//1/x"):
        with pytest.raises(ValueError):
            engine.regenerate(bad, [SKILL])


def test_pick_source_accepts_a_key_or_a_difficulty_label():
    skill = SKILLS["sine-rule"]
    assert engine.pick_source(skill, "solve").topic == "sine_rule_solve"
    assert engine.pick_source(skill, "angle").topic == "sine_rule_angle"
    assert engine.pick_source(skill, "hard") is skill.sources[3]  # the first source keyed by its label
    with pytest.raises(KeyError, match="its source keys are"):
        engine.pick_source(skill, "nope")


def _errors_for(*sources):
    skill = copy.deepcopy(SKILL)
    skill.sources = list(sources)
    report = validate([], [skill], graders={skill.answer: object()}, gate_registry={})
    return [e for e in report.errors if "key" in e]


def test_validate_rejects_a_repeated_key_and_a_bad_key():
    a = Source(topic="addition", difficulty="easy")
    b = Source(topic="addition", difficulty="easy")
    assert any("share the key 'easy'" in e for e in _errors_for(a, b))
    assert not _errors_for(a, Source(topic="addition", difficulty="easy", key="other"))
    assert any(
        "explicit" in e or "share the key" in e
        for e in _errors_for(a, Source(topic="addition", difficulty="medium", key="easy"))
    )
    for bad in ("Easy", "two words", "-x", "x-", "x--y", "x_y", ""):
        assert any(
            "must be lower-case" in e for e in _errors_for(Source(topic="addition", difficulty="easy", key=bad))
        ), bad
