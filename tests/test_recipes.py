"""The recipe composer (mathgen/recipes.py): registration, validation, composition, conformance helper."""

import pytest

from mathgen import core, recipes
from mathgen.grading import grade
from mathgen.recipes import Recipe, RecipeError, Subject, ask, subject


@pytest.fixture
def toy():
    """A toy line subject with two per-subject asks and one generic ask, removed again afterwards."""
    snapshot = (dict(recipes.SUBJECTS), dict(recipes.ASKS), dict(core.REGISTRY), dict(core.PRESETS))

    @subject("toy_line")
    class ToyLine(Subject):
        properties = frozenset({"linear_argument"})

        def __init__(self, m, r, noun):
            self.m, self.r, self.noun = m, r, noun

        @classmethod
        def draw(cls, rng, technology_active):
            return cls(rng.randint(2, 9), rng.randint(-400, 400), rng.choice(["A line", "A straight path", "A ramp"]))

        def stem(self):
            return f"{self.noun} has equation y = {self.m}x {-self.m * self.r:+d}."

    @ask("toy_line", "gradient")
    def gradient(s, rng, ctx):
        return dict(
            prompt="Find the gradient of the line.", answer_kind="number", answer=s.m, answer_text=str(s.m), marks=1
        )

    @ask("toy_line", "root", default_marks=2)
    def root(s, rng, ctx):
        return dict(
            prompt="Find where the line crosses the x-axis.", answer_kind="number", answer=s.r, answer_text=str(s.r)
        )

    @ask("toy_line", "hence_value", requires=("gradient",))
    def hence(s, rng, ctx):
        g = ctx.label_of("gradient")
        return dict(
            prompt="Hence find the change in y when x increases by 3.",
            answer_kind="number",
            answer=3 * s.m,
            answer_text=str(3 * s.m),
            follow_through=f"3*part_{g}",
        )

    @ask(None, "is_linear", needs=("linear_argument",))
    def generic(s, rng, ctx):
        return dict(prompt="Is the line straight? Answer 1 for yes.", answer_kind="number", answer=1, answer_text="1")

    yield
    recipes.SUBJECTS.clear(), recipes.ASKS.clear(), core.REGISTRY.clear(), core.PRESETS.clear()
    recipes.SUBJECTS.update(snapshot[0]), recipes.ASKS.update(snapshot[1])
    core.REGISTRY.update(snapshot[2]), core.PRESETS.update(snapshot[3])


def _recipe(asks, **kw):
    return Recipe(id="toy_one", title="Toy", level=9, subject="toy_line", asks=tuple(asks), **kw)


def test_a_recipe_builds_a_multi_part_question_with_follow_through(toy):
    topic = recipes.register_recipe(_recipe([("gradient", None), ("root", None), ("hence_value", 3)]))
    q = core.make(topic, seed=4)
    assert [p.label for p in q.parts] == ["a", "b", "c"]
    assert [p.marks for p in q.parts] == [1, 2, 3]  # gradient's own 1, root's default 2, the recipe's 3
    assert q.parts[2].follow_through == "3*part_a"
    # a wrong (a) carried into (c) earns the follow-through mark
    wrong = grade(
        "number",
        q,
        {"a": str(q.parts[0].answer + 1), "b": str(q.parts[1].answer), "c": str(3 * (q.parts[0].answer + 1))},
    )
    assert wrong.parts["c"].correct and wrong.parts["c"].follow_through


def test_same_seed_same_question(toy):
    topic = recipes.register_recipe(_recipe([("gradient", None), ("root", None)]))
    assert core.make(topic, seed=7).to_json() == core.make(topic, seed=7).to_json()


def test_technology_flag_reaches_the_subject(toy):
    seen = []

    @ask("toy_line", "spy")
    def spy(s, rng, ctx):
        seen.append(s.technology_active)
        return dict(prompt="Spy.", answer_kind="number", answer=1, answer_text="1")

    topic = recipes.register_recipe(_recipe([("spy", None)], technology_active=False))
    core.make(topic, seed=1)
    core.make(topic, seed=1, technology_active=True)
    assert seen == [False, True]


@pytest.mark.parametrize(
    "asks, message",
    [
        ([("nope", None)], "unknown ask 'nope'"),
        ([("hence_value", None)], r"requires \['gradient'\]"),
        ([("gradient", None), ("gradient", None)], "not repeatable"),
        ([("gradient", 0)], "marks"),
        ([], "no asks"),
    ],
)
def test_bad_recipes_fail_with_a_clear_message(toy, asks, message):
    with pytest.raises(RecipeError, match=message):
        recipes.validate(_recipe(asks))


def test_unknown_subject_names_the_known_ones(toy):
    with pytest.raises(RecipeError, match="unknown subject 'ghost'.*toy_line"):
        recipes.validate(Recipe(id="r", title="t", level=9, subject="ghost", asks=(("gradient", None),)))


def test_generic_ask_needs_subject_properties(toy):
    recipes.validate(_recipe([("is_linear", None)]))
    recipes.SUBJECTS["toy_line"].properties = frozenset()
    with pytest.raises(RecipeError, match="needs \\['linear_argument'\\]"):
        recipes.validate(_recipe([("is_linear", None)]))


def test_a_per_subject_ask_beats_a_generic_one(toy):
    @ask("toy_line", "is_linear")
    def specific(s, rng, ctx):
        return dict(prompt="Specific.", answer_kind="number", answer=1, answer_text="1")

    topic = recipes.register_recipe(_recipe([("is_linear", None)]))
    assert core.make(topic, seed=1).parts[0].prompt == "Specific."


def test_conformance_passes_a_good_recipe_and_flags_a_leaked_answer(toy):
    topic = recipes.register_recipe(_recipe([("gradient", None), ("hence_value", None), ("root", None)]))
    assert topic == "recipe_toy_one"
    assert recipes.conformance("toy_one", seeds=300) == []

    @ask("toy_line", "gives_away")
    def gives_away(s, rng, ctx):
        return dict(prompt=f"The line crosses at {s.r}.", answer_kind="number", answer=s.m, answer_text=str(s.m))

    recipes.register_recipe(
        Recipe(id="toy_leak", title="t", level=9, subject="toy_line", asks=(("gives_away", None), ("root", None)))
    )
    assert any("appears in the stem or an earlier prompt" in p for p in recipes.conformance("toy_leak", seeds=60))


def test_loading_yaml_recipes(toy, tmp_path):
    (tmp_path / "toy_two.yaml").write_text(
        "id: toy_two\ntitle: Toy two\nlevel: 9\nsubject: toy_line\ntechnology_active: false\n"
        "asks:\n  - ask: gradient\n  - ask: root\n    marks: 2\n",
        encoding="utf-8",
    )
    [loaded] = recipes.load_recipes(tmp_path)
    assert loaded.asks == (("gradient", None), ("root", 2)) and loaded.technology_active is False
    assert recipes.register_all(tmp_path) == ["recipe_toy_two"]


@pytest.mark.parametrize(
    "body, message",
    [
        ("id: other\ntitle: t\nlevel: 9\nsubject: s\nasks: [{ask: a}]\n", "must equal the file name"),
        ("id: bad_file\ntitle: t\nlevel: 9\nsubject: s\n", "missing 'asks'"),
        ("id: bad_file\ntitle: t\nlevel: 9\nsubject: s\nextra: 1\nasks: [{ask: a}]\n", "unknown keys"),
        ("id: bad_file\ntitle: t\nlevel: 9\nsubject: s\nasks: [{ask: a, mark: 1}]\n", "each ask is a mapping"),
    ],
)
def test_malformed_recipe_files_are_refused(tmp_path, body, message):
    (tmp_path / "bad_file.yaml").write_text(body, encoding="utf-8")
    with pytest.raises(RecipeError, match=message):
        recipes.load_recipes(tmp_path)


def test_no_recipes_directory_is_fine(tmp_path):
    assert recipes.load_recipes(tmp_path / "absent") == []


def test_subject_names_outside_the_vocabulary_are_refused(toy):
    with pytest.raises(RecipeError, match="outside mathgen/recipe_vocabulary.py.*made_up"):

        @subject("bad_toy")
        class Bad(Subject):
            properties = frozenset({"made_up"})


def test_a_declared_property_must_hold_on_every_draw(toy, monkeypatch):
    from mathgen import recipe_vocabulary

    monkeypatch.setitem(recipe_vocabulary.PROPERTY_CHECKS, "linear_argument", lambda s: s.m > 100)
    recipes.SUBJECTS["toy_line"].properties = frozenset({"linear_argument"})
    topic = recipes.register_recipe(_recipe([("gradient", None)]))
    with pytest.raises(core.VerificationError, match="declares 'linear_argument'"):
        core.make(topic, seed=1)
    monkeypatch.setitem(recipe_vocabulary.PROPERTY_CHECKS, "linear_argument", lambda s: s.m > 0)
    assert core.make(topic, seed=1).parts[0].label == "a"
