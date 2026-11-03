"""ExprSubject: properties computed from the sympy expression, and checked on every draw."""

import pytest
import sympy as sp

from mathgen import core, recipes
from mathgen.recipe_expr import X, ExprSubject, expr_text
from mathgen.recipes import Recipe, Subject, ask, subject

x = X
a = sp.Symbol("a")
CASES = [
    # (expression, domain, properties that hold, properties that do not)
    (
        sp.log(-(x**2) + 4 * x - 3),
        sp.S.Reals,
        {"quadratic_argument", "restricted_domain", "has_stationary_point"},
        {"differentiable_everywhere", "periodic", "linear_argument"},
    ),
    (
        3 ** (2 * x) - 10 * 3**x + 9,
        sp.S.Reals,
        {
            "quadratic_in_exponential",
            "differentiable_everywhere",
            "has_closed_form_antiderivative",
            "linear_argument",
            "has_asymptote",
        },
        {"restricted_domain", "periodic"},
    ),
    (
        sp.sin(x + sp.pi / 3),
        sp.Interval(0, 2 * sp.pi),
        {"periodic", "linear_argument", "has_pi_roots", "differentiable_everywhere", "has_stationary_point"},
        {"restricted_domain", "has_log_root", "has_surd_roots"},
    ),
    (
        1 / (x - 2) + 1,
        sp.S.Reals,
        {"restricted_domain", "has_asymptote", "rational_roots_only"},
        {"differentiable_everywhere", "periodic", "has_stationary_point"},
    ),
    (
        x**3 - 3 * x,
        sp.S.Reals,
        {
            "has_stationary_point",
            "has_inflection",
            "differentiable_everywhere",
            "has_surd_roots",
            "has_closed_form_antiderivative",
        },
        {"rational_roots_only", "restricted_domain", "has_asymptote", "periodic"},
    ),
    (
        2**x - 5,
        sp.S.Reals,
        {"has_log_root", "linear_argument", "has_asymptote"},
        {"quadratic_in_exponential", "has_stationary_point"},
    ),
    (sp.sqrt(x**2 - 4), sp.S.Reals, {"restricted_domain", "quadratic_argument"}, {"periodic"}),
]


@pytest.mark.parametrize("expr, domain, holds, fails", CASES, ids=[expr_text(c[0])[:30] for c in CASES])
def test_properties_are_computed_from_the_expression(expr, domain, holds, fails):
    s = ExprSubject(expr, domain=domain)
    for name in holds:
        assert s.check_property(name) is not False, f"{name} should hold or be untestable for {expr}"
    for name in fails:
        assert s.check_property(name) is not True, f"{name} should not hold for {expr}"


def test_computed_verdicts_are_definite_for_common_cases():
    ln = ExprSubject(sp.log(-(x**2) + 4 * x - 3))
    assert ln.check_property("quadratic_argument") is True
    assert ln.check_property("restricted_domain") is True
    assert ln.check_property("differentiable_everywhere") is False
    assert ExprSubject(3 ** (2 * x) - 10 * 3**x + 9).check_property("quadratic_in_exponential") is True
    assert ExprSubject(x**3 - 3 * x).check_property("quadratic_in_exponential") is False
    assert ExprSubject(sp.sin(x)).check_property("periodic") is True
    assert ExprSubject(x**2).check_property("periodic") is False


def test_properties_without_a_generic_test_are_unchecked():
    assert ExprSubject(x**2).check_property("even_candidate") is None
    assert ExprSubject(x**2).check_property("unknown_constants_1") is None


def test_expr_text_writes_powers_with_a_caret():
    assert expr_text(x**2 - 3 * x) == "x^2 - 3*x"


@pytest.fixture
def registered():
    snapshot = (dict(recipes.SUBJECTS), dict(recipes.ASKS), dict(core.REGISTRY), dict(core.PRESETS))
    yield
    recipes.SUBJECTS.clear(), recipes.ASKS.clear(), core.REGISTRY.clear(), core.PRESETS.clear()
    recipes.SUBJECTS.update(snapshot[0]), recipes.ASKS.update(snapshot[1])
    core.REGISTRY.update(snapshot[2]), core.PRESETS.update(snapshot[3])


def _define(props):
    @subject("expr_toy")
    class ExprToy(ExprSubject):
        properties = frozenset(props)

        @classmethod
        def draw(cls, rng, technology_active):
            k = rng.randint(2, 9)
            return cls(sp.log(x**2 - k))  # restricted domain, quadratic argument, differentiable on its domain

        def stem(self):
            return f"Let f(x) = {expr_text(self.expr)}."

    @ask("expr_toy", "one")
    def one(s, rng, ctx):
        return dict(prompt="One.", answer_kind="number", answer=1, answer_text="1")

    return Recipe(id="expr_one", title="t", level=11, subject="expr_toy", asks=(("one", None),))


def test_a_true_declaration_passes_on_every_draw(registered):
    topic = recipes.register_recipe(_define({"quadratic_argument", "restricted_domain", "has_stationary_point"}))
    for seed in range(5):
        assert core.make(topic, seed=seed).stem.startswith("Let f(x) = log(x^2 -")


def test_a_false_declaration_fails_the_draw(registered):
    topic = recipes.register_recipe(_define({"periodic"}))
    with pytest.raises(core.VerificationError, match="declares 'periodic'"):
        core.make(topic, seed=1)
