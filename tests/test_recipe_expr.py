"""ExprSubject: declared properties are checked against the expression on every draw, cheaply.

The adversarial cases are the ones a first, solveset-based version got wrong (false passes and a crash): a
stationary point outside the domain, an asymptote that is a removable gap, an antiderivative that is not
elementary, a pi that is not a rational multiple of pi, a kink, an inflection of odd order and two exponential
look-alikes.
"""

import time

import pytest
import sympy as sp

from mathgen import core, recipes
from mathgen.recipe_expr import X, ExprSubject, expr_text
from mathgen.recipes import Recipe, ask, subject

x = X


def sub(expr, domain=sp.S.Reals, **facts):
    return ExprSubject(expr, domain=domain, facts=facts)


# ----- structural ------------------------------------------------------------------------------------------------


def test_argument_shapes():
    ln = sub(sp.log(-(x**2) + 4 * x - 3))
    assert ln.check_property("quadratic_argument") is True
    assert ln.check_property("linear_argument") is False
    assert sub(sp.sin(x + 1)).check_property("linear_argument") is True
    assert sub(sp.exp(x**3)).check_property("polynomial_argument") is True
    assert sub(sp.sqrt(x**2 - 4)).check_property("quadratic_argument") is True


@pytest.mark.parametrize(
    "expr, expected",
    [
        (3 ** (2 * x) - 10 * 3**x + 9, True),
        (3 ** (2 * x + 1) - 3**x, True),  # the constant in the exponent is a factor
        (x * 3 ** (2 * x) + 3**x, False),  # x survives the substitution
        (3 ** (3 * x) - 3**x, False),  # cubic, not quadratic
        (3**x - 5, False),
        (2 ** (2 * x) - 3**x, False),  # two bases
        (sp.exp(2 * x) - 3 * sp.exp(x) + 2, True),
        (x**2 - 3, False),
    ],
)
def test_quadratic_in_exponential(expr, expected):
    assert sub(expr).check_property("quadratic_in_exponential") is expected


@pytest.mark.parametrize(
    "expr, verdict",
    [
        (x**3 - 3 * x, True),
        (sp.sin(x) + sp.exp(x), True),
        (sp.Abs(x), False),
        (sp.Max(x, 1), False),
        (sp.log(x**2 + 1), None),  # smooth, but not provably so here: unchecked
        (1 / x, False),
        (sp.log(x), False),
        (sp.sqrt(x), False),
        (sp.tan(x), False),
        (x ** sp.Rational(1, 3), False),
        (sp.sin(x) / x, False),
    ],
)
def test_differentiable_everywhere_never_raises(expr, verdict):
    assert sub(expr).check_property("differentiable_everywhere") is verdict


@pytest.mark.parametrize(
    "expr, verdict",
    [
        (sp.sin(x), True),
        (sp.sin(3 * x) * sp.cos(2 * x) + sp.tan(x / 2), True),
        (sp.sin(x + sp.pi / 3) ** 2 + 1, True),
        (x**2, False),
        (sp.exp(x**3) + sp.log(x**2 - 3), False),
        (sp.sin(x) + x, None),  # x outside the trig: not decided here
        (sp.sin(sp.sqrt(2) * x), None),  # irrational frequency: not a rational multiple of pi
        (sp.sin(x**2), None),
    ],
)
def test_periodic_is_structural(expr, verdict):
    assert sub(expr).check_property("periodic") is verdict


# ----- witnesses -------------------------------------------------------------------------------------------------


def test_stationary_point_must_be_in_the_domain_and_defined():
    assert sub(x**3 - 3 * x, stationary_points=[1, -1]).check_property("has_stationary_point") is True
    assert sub(sp.sqrt(x**2 - 4), stationary_points=[0]).check_property("has_stationary_point") is False
    assert sub(sp.log(x**2 - 5), stationary_points=[0]).check_property("has_stationary_point") is False
    assert sub(sp.Abs(x), stationary_points=[0]).check_property("has_stationary_point") is False
    assert sub(x**3, stationary_points=[1]).check_property("has_stationary_point") is False  # f'(1) = 3
    assert sub(x**2).check_property("has_stationary_point") is None  # no witness: unchecked, never a pass


def test_stationary_point_outside_a_restricted_domain_fails():
    assert sub(x**2, sp.Interval(1, 3), stationary_points=[0]).check_property("has_stationary_point") is False


def test_asymptotes_must_be_real_ones():
    rational = sub(1 / (x - 2) + 1, vertical_asymptotes=[2], horizontal_asymptotes=[(1, 1), (1, -1)])
    assert rational.check_property("has_asymptote") is True
    assert sub(sp.log(x), vertical_asymptotes=[0]).check_property("has_asymptote") is True
    assert sub(2**x - 5, horizontal_asymptotes=[(-5, -1)]).check_property("has_asymptote") is True
    assert sub(sp.sin(x), horizontal_asymptotes=[(0, 1)]).check_property("has_asymptote") is False
    assert sub(sp.cos(x) + 2, horizontal_asymptotes=[(2, 1)]).check_property("has_asymptote") is False
    gap = sub((x**2 - 1) / (x - 1), vertical_asymptotes=[1])  # a removable gap, not an asymptote
    assert gap.check_property("has_asymptote") is False


def test_antiderivative_must_be_elementary_and_correct():
    good = sub(x * sp.exp(x**2), antiderivative=sp.exp(x**2) / 2)
    assert good.check_property("has_closed_form_antiderivative") is True
    gauss = sub(sp.exp(-(x**2)), antiderivative=sp.sqrt(sp.pi) * sp.erf(x) / 2)
    assert gauss.check_property("has_closed_form_antiderivative") is False
    assert sub(sp.sin(x) / x, antiderivative=sp.Si(x)).check_property("has_closed_form_antiderivative") is False
    assert sub(x**2, antiderivative=x**3).check_property("has_closed_form_antiderivative") is False  # wrong F


def test_roots_must_be_roots_and_classified_exactly():
    assert sub(sp.sin(x), sp.Interval(0, 2 * sp.pi), roots=[0, sp.pi]).check_property("has_pi_roots") is True
    # sin(x) = 1/3 has the root pi - asin(1/3): it contains pi but is not a rational multiple of pi
    one_third = sp.asin(sp.Rational(1, 3))
    s = sub(sp.sin(x) - sp.Rational(1, 3), sp.Interval(0, sp.pi), roots=[one_third, sp.pi - one_third])
    assert s.check_property("has_pi_roots") is False
    assert sub(3**x - 7, roots=[sp.log(7) / sp.log(3)]).check_property("has_log_root") is True
    assert sub(x**2 - 3, roots=[sp.sqrt(3), -sp.sqrt(3)]).check_property("has_surd_roots") is True
    assert sub(x**2 - 4, roots=[2, -2]).check_property("has_surd_roots") is False
    assert sub(x**2 - 4, roots=[3]).check_property("has_surd_roots") is False  # 3 is not a root: the witness fails


def test_rational_roots_only_for_polynomials():
    assert sub(x**2 - 5 * x + 6).check_property("rational_roots_only") is True
    assert sub(x**3 - 3 * x).check_property("rational_roots_only") is False
    assert sub(sp.sin(x)).check_property("rational_roots_only") is None


def test_inflection_includes_odd_order_points():
    assert sub(x**3, inflection_points=[0]).check_property("has_inflection") is True
    assert sub(x**5, inflection_points=[0]).check_property("has_inflection") is True  # f''' = 0 at 0, f'' changes sign
    assert sub(x**4, inflection_points=[0]).check_property("has_inflection") is False  # f'' = 12x^2 keeps its sign


def test_restricted_domain_needs_an_undefined_witness():
    assert sub(sp.log(x**2 - 5), excluded_points=[0]).check_property("restricted_domain") is True
    assert sub(1 / (x - 2), excluded_points=[2]).check_property("restricted_domain") is True
    assert sub(x**2, excluded_points=[0]).check_property("restricted_domain") is False


def test_properties_without_a_generic_test_are_unchecked():
    assert sub(x**2).check_property("even_candidate") is None
    assert sub(x**2).check_property("unknown_constants_1") is None


def test_a_property_check_registered_for_the_name_is_honoured(monkeypatch):
    from mathgen import recipe_vocabulary

    monkeypatch.setitem(recipe_vocabulary.PROPERTY_CHECKS, "periodic", lambda s: False)
    assert sub(sp.sin(x)).check_property("periodic") is False


def test_expr_text_writes_powers_with_a_caret():
    assert expr_text(x**2 - 3 * x) == "x^2 - 3*x"


# ----- cost ------------------------------------------------------------------------------------------------------


def test_every_check_is_cheap():
    """Checks run on every make(); the whole battery of declared properties must stay well under the 250 ms ceiling."""
    pi = sp.pi
    cases = [
        sub(sp.log(-(x**2) + 4 * x - 3), excluded_points=[0]),
        sub(3 ** (2 * x) - 10 * 3**x + 9, horizontal_asymptotes=[(9, -1)], roots=[0, 2]),
        sub(sp.sin(x + pi / 3), sp.Interval(0, 2 * pi), roots=[2 * pi / 3, 5 * pi / 3], stationary_points=[pi / 6]),
        sub(1 / (x - 2) + 1, vertical_asymptotes=[2], horizontal_asymptotes=[(1, 1)]),
    ]
    props = [
        "linear_argument",
        "quadratic_argument",
        "quadratic_in_exponential",
        "periodic",
        "differentiable_everywhere",
        "restricted_domain",
        "has_stationary_point",
        "has_asymptote",
        "has_pi_roots",
        "has_log_root",
        "has_surd_roots",
    ]
    for s in cases:
        start = time.perf_counter()
        for p in props:
            s.check_property(p)
        assert time.perf_counter() - start < 0.15, expr_text(s.expr)


# ----- through compose -------------------------------------------------------------------------------------------


@pytest.fixture
def registered():
    snapshot = (dict(recipes.SUBJECTS), dict(recipes.ASKS), dict(core.REGISTRY), dict(core.PRESETS))
    yield
    recipes.SUBJECTS.clear(), recipes.ASKS.clear(), core.REGISTRY.clear(), core.PRESETS.clear()
    recipes.SUBJECTS.update(snapshot[0]), recipes.ASKS.update(snapshot[1])
    core.REGISTRY.update(snapshot[2]), core.PRESETS.update(snapshot[3])


def _define(props, **facts):
    @subject("expr_toy")
    class ExprToy(ExprSubject):
        properties = frozenset(props)

        @classmethod
        def draw(cls, rng, technology_active):
            k = rng.randint(2, 9)
            # log(x^2 - k) is undefined at 0 (x^2 - k < 0), so x = 0 is excluded and there is NO stationary point
            return cls(sp.log(x**2 - k), facts=dict(facts))

        def stem(self):
            return f"Let f(x) = {expr_text(self.expr)}."

    @ask("expr_toy", "one")
    def one(s, rng, ctx):
        return dict(prompt="One.", answer_kind="number", answer=1, answer_text="1")

    return Recipe(id="expr_one", title="t", level=11, subject="expr_toy", asks=(("one", None),))


def test_a_true_declaration_passes_on_every_draw(registered):
    topic = recipes.register_recipe(_define({"quadratic_argument", "restricted_domain"}, excluded_points=[0]))
    for seed in range(5):
        assert core.make(topic, seed=seed).stem.startswith("Let f(x) = log(x^2 -")


def test_a_false_declaration_fails_the_draw(registered):
    topic = recipes.register_recipe(_define({"periodic"}))
    with pytest.raises(core.VerificationError, match="declares 'periodic'"):
        core.make(topic, seed=1)


def test_a_stationary_point_outside_the_domain_fails_the_draw(registered):
    """log(x^2 - k) has no stationary point (x = 0 is outside its domain); declaring one used to pass."""
    topic = recipes.register_recipe(_define({"has_stationary_point"}, stationary_points=[0]))
    with pytest.raises(core.VerificationError, match="declares 'has_stationary_point'"):
        core.make(topic, seed=1)


def test_text_is_refused_as_an_expression_or_a_witness():
    """Nothing in ExprSubject reads a string as maths, so no string can be evaluated (see _exact)."""
    with pytest.raises(TypeError, match="not 'x'"):
        ExprSubject("x")
    with pytest.raises(TypeError):
        sub(x**2, stationary_points=["0"]).check_property("has_stationary_point")
    with pytest.raises(TypeError):
        sub(x**2, antiderivative="x**3/3").check_property("has_closed_form_antiderivative")
