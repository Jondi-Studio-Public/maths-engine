"""technology_active: the `make()` argument, the generator declaration and the skill/source label."""

import pytest

from mathgen import core, engine, make
from mathgen.skills import Skill, Source, skills_for_technology, sources_for_technology, validate

TOPIC = "zz_technology_probe"
PLAIN = "zz_technology_plain"


@pytest.fixture()
def probe_topics():
    @core.generator(TOPIC, 1, technology=(True, False))
    def _probe(rng, technology_active=None):
        n = rng.randrange(2, 9)
        text = f"Find {n}/4 as a decimal." if technology_active else f"Write {n}/4 as a fraction."
        return dict(text=text, latex=text, answer=n, answer_text=str(n))

    @core.generator(PLAIN, 1)
    def _plain(rng):
        return dict(text="Plain.", latex="Plain.", answer=1, answer_text="1")

    yield
    for t in (TOPIC, PLAIN):
        core.REGISTRY.pop(t, None)
        core.TECHNOLOGY_MODES.pop(t, None)


def test_default_is_unchanged_and_unflagged(probe_topics):
    q = make(PLAIN, seed=3)
    assert q.technology_active is None
    assert "technology_active" not in q.to_dict()


def test_each_mode_reaches_the_generator_and_is_recorded(probe_topics):
    on, off = make(TOPIC, seed=5, technology_active=True), make(TOPIC, seed=5, technology_active=False)
    assert "decimal" in on.text and "fraction" in off.text
    assert on.technology_active is True and off.technology_active is False
    assert on.to_dict()["technology_active"] is True
    assert off.to_dict()["technology_active"] is False


def test_same_mode_and_seed_repeat(probe_topics):
    a = make(TOPIC, seed=9, technology_active=False)
    b = make(TOPIC, seed=9, technology_active=False)
    assert a.to_json() == b.to_json()


def test_a_topic_that_has_not_declared_the_mode_refuses(probe_topics):
    for flag in (True, False):
        with pytest.raises(ValueError, match="does not support technology_active"):
            make(PLAIN, seed=1, technology_active=flag)


def test_flag_must_be_a_bool(probe_topics):
    with pytest.raises(TypeError):
        make(TOPIC, seed=1, technology_active="yes")


def test_declaring_a_non_bool_mode_is_refused():
    with pytest.raises(ValueError, match="booleans"):

        @core.generator("zz_technology_bad", 1, technology=("tech",))
        def _bad(rng):  # pragma: no cover
            return {}

    core.REGISTRY.pop("zz_technology_bad", None)


def _skill(technology=None, sources=()):
    return Skill(id="zz", name="zz", domain="arithmetic", answer="number", technology=technology, sources=list(sources))


def test_untagged_required_either_none_filter(probe_topics):
    untagged = Source(topic=PLAIN, difficulty="easy")
    required = Source(topic=PLAIN, difficulty="medium", technology="required")
    either = Source(topic=TOPIC, difficulty="hard", technology="either")
    none = Source(topic=PLAIN, difficulty="exam", technology="none")
    skill = _skill(sources=[untagged, required, either, none])
    assert sources_for_technology(skill, False) == [either, none]
    assert sources_for_technology(skill, True) == skill.sources
    assert sources_for_technology(skill, None) == skill.sources
    assert skills_for_technology([skill, _skill(sources=[untagged])], False) == [skill]


def test_the_skill_label_applies_unless_a_source_overrides_it(probe_topics):
    skill = _skill(
        "none", [Source(topic=PLAIN, difficulty="easy"), Source(topic=PLAIN, difficulty="hard", technology="required")]
    )
    assert sources_for_technology(skill, False) == [skill.sources[0]]


def test_make_for_skill_picks_a_usable_source_and_passes_the_flag(probe_topics):
    skill = _skill(
        sources=[Source(topic=PLAIN, difficulty="easy"), Source(topic=TOPIC, difficulty="hard", technology="either")]
    )
    q = engine.make_for_skill(skill, seed=2, technology_active=False)
    assert q.topic == TOPIC and q.technology_active is False
    with pytest.raises(ValueError, match="not usable with technology off"):
        engine.make_for_skill(skill, seed=2, source=0, technology_active=False)
    assert engine.make_for_skill(skill, seed=2, source=0, technology_active=True).technology_active is None


def test_make_for_skill_with_nothing_usable_refuses(probe_topics):
    skill = _skill(sources=[Source(topic=PLAIN, difficulty="easy")])
    with pytest.raises(ValueError, match="no source that is usable"):
        engine.make_for_skill(skill, seed=1, technology_active=False)


def test_validate_flags_a_bad_label_and_an_either_without_modes(probe_topics):
    domains, skills = __import__("mathgen.skills", fromlist=["load_skills"]).load_skills()
    base = next(s for s in skills if s.status == "active")
    bad = Skill(**{**base.__dict__, "technology": "maybe"})
    report = validate(domains, [bad])
    assert any("technology must be one of" in e for e in report.errors)
    src = Source(topic=PLAIN, difficulty="easy", technology="either")
    either = Skill(**{**base.__dict__, "sources": [src], "technology": None})
    report = validate(domains, [either])
    assert any("does not declare both modes" in e for e in report.errors)


def test_the_real_library_has_no_technology_errors():
    from mathgen.skills import load_and_validate

    report = load_and_validate()[-1] if isinstance(load_and_validate(), tuple) else None
    assert report is not None and not [e for e in report.errors if "technology" in e]


def test_ids_round_trip_with_the_mode_and_need_it_to_rebuild(probe_topics):
    skill = _skill(sources=[Source(topic=TOPIC, difficulty="easy", technology="either")])
    q, qid = engine.make_with_id(skill, 4, technology_active=False)
    assert q.technology_active is False
    assert engine.regenerate(qid, [skill], technology_active=False).to_json() == q.to_json()
    with pytest.raises(engine.QuestionChanged):
        engine.regenerate(qid, [skill])


def test_engine_exports_the_filter_helpers():
    for name in ("skills_for_technology", "sources_for_technology", "source_usable", "source_technology"):
        assert name in engine.__all__
