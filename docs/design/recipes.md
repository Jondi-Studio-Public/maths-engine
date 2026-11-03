# Recipes: chained exam questions from a subject and a list of asks

Status: implemented by the recipe composer change (`mathgen/recipes.py`). No existing skill is edited.

## Why

An exam question often builds one object (a function, a pdf) and then asks several things about it. The
multi-part model (`docs/design/multi-part-questions.md`) can mark such a question, but each chain was a
hand-written generator. A recipe names the object and the asks in a short data file, and the composer builds
the multi-part question from it.

## Vocabulary

- **Subject**: one object drawn once per question, with every fact about it known (function, roots, vertex,
  domain, constants). A Python class registered with `@subject("name")`.
- **Ask**: one part about a subject. A function registered with `@ask("subject_name", "ask_name")`, written next
  to the subject. It states the prompt, computes the answer, builds tagged distractors, and sets follow-through.
- **Recipe**: a YAML file under `recipes/` naming a subject and an ordered list of asks with marks. It becomes
  the generator topic `recipe_<id>`, so `make()`, `grade()` and skills use it unchanged.

## Subject interface

```python
from mathgen.recipes import Subject, subject


@subject("log_quadratic")
class LogQuadratic(Subject):
    # Subject.technology_active: bool | None is set by the composer before stem() or any ask runs.
    @classmethod
    def draw(cls, rng, technology_active):  # build the object; use core.attempts(), draw only from rng
        return cls(...)  # exact facts; technology_active None/False = exact, True = decimals

    def stem(self) -> str:  # the shared information, plain text, goes before part (a)
        ...

    # optional: def figure(self) -> dict | None
```

`technology_active` comes from `make(..., technology_active=...)` (owned by the technology-flag change) or, when
`None`, from the recipe's own `technology_active` field, else stays `None`. Every ask reads
`subject.technology_active`, so all parts are exact together or decimal together.

## Ask interface

```python
from mathgen.recipes import ask


@ask("log_quadratic", "implied_domain", requires=())  # requires: ask names that must come earlier
def implied_domain(subject, rng, ctx) -> dict:
    return dict(  # exactly the keys a Part takes (see multi-part-questions.md)
        prompt="Find the implied domain of f.",
        answer_kind="expression",
        answer=...,
        answer_text="...",
        distractors=[Distractor(...), ...],  # tagged with catalogue misconception ids
        sensible_range=None,
        working=[...],
        # `label` and `marks` are filled by the composer (marks from the recipe, else `default_marks`).
    )
```

`ctx` gives: `ctx.label` (this part's label), `ctx.label_of("x_of_maximum")` (label of an earlier ask, for
writing `follow_through="2*part_%s" % ctx.label_of(...)`), and `ctx.earlier` (the earlier part dicts). An ask
must not draw from `rng` before it needs to, and must be deterministic per seed. The same ask may appear twice
in a recipe only if it declares `repeatable=True`.

Rules the composer enforces: unknown subject or ask fails with a message listing what exists; an ask used before
the asks it `requires` fails; an ask registered for another subject fails; the part dict goes through the normal
`make()` checks (follow-through reproduces the canonical answer, tagged distractors, answer kinds).

## Generic asks (compatible with a grid-driven design)

Jondi has not yet chosen between one ask set per subject type and generic asks over expression families (thread
"Design engine to cover all papers without sampling"). The interface supports both so no subject work is wasted:

- `Subject.properties: frozenset[str]` names what the subject can supply (`"quadratic_argument"`,
  `"closed_form_antiderivative"`, ...).
- `@ask(None, "stationary_x", needs=("differentiable",))` registers a generic ask, valid for any subject whose
  `properties` include everything in `needs`. `@ask("log_quadratic", ...)` stays the per-subject form, and a
  per-subject ask of the same name wins over a generic one.

Jondi chose this grid design (2026-11-02). Vocabulary: `mathgen/recipe_vocabulary.py` holds the closed lists of
`FAMILIES` and `PROPERTIES` (first list for VCE Methods, extended only by PR; meanings in the plan folder's
`subject-vocabulary.md`). `@subject` refuses names outside it. `Subject.families` and `Subject.properties` are
declared per subject class (so which asks apply is fixed by the recipe, not by the draw), and a property can have a
check, `@property_check("has_stationary_point")` taking the drawn subject, which `compose` runs on every draw: a
declared property that is false fails the seed, so the declaration cannot drift from the maths.

## Recipe YAML

```yaml
id: methods_ln_quadratic_1        # lower-case, digits, underscores; topic is recipe_<id>
title: Natural log of a quadratic
level: 12                         # registry level
subject: log_quadratic
technology_active: false          # optional; null/absent = leave to make()
asks:
  - ask: implied_domain
    marks: 2
  - ask: x_of_maximum
    marks: 2
  - ask: solve_f_equals_c
    marks: 3
```

Recipes are scanned from `recipes/` (or `MATHS_RECIPES_DIR`) and registered when `mathgen.generators` loads. A
malformed recipe file, or one naming an unknown subject or ask, raises then, so a bad file breaks the whole
`mathgen` import loudly (`tests/test_recipes.py` loads the real tree so CI catches it before a merge).
Every recipe declares both technology modes (`technology=(True, False)`) because it hands the flag to its subject.
`Question.technology_active` records only the value passed to `make()`; a recipe's own `technology_active` field
is the default the subject sees when `make()` was not asked.

## Testing a subject thread's work

`from mathgen.recipes import conformance` runs a recipe over N seeds (default 300) and returns problems: build
failures, answers not clean, a part whose prompt or the stem contains a later part's answer, fewer than 100
distinct final-part answers per 300 seeds, fewer than 3 distinct prompt shapes. Subject tests call it and assert
an empty list, plus their own independent-oracle tests for each ask.

## Not done here

Wheel packaging of `recipes/` (add when the first real recipe ships), a site renderer, a general way to chain
arbitrary skills (Plan B).

## Expression subjects

`mathgen/recipe_expr.py` has `ExprSubject`, a Subject holding a sympy `expr`, `var`, `domain` and a `facts` dict. It
answers `Subject.check_property(name)` from the expression itself, which `compose` calls for every declared
property on every draw (`False` fails the seed; `None`, for a property with no generic test, is skipped). Computed:
the three argument shapes, `quadratic_in_exponential` (one exponential base), `periodic`, `restricted_domain`,
`differentiable_everywhere`, `has_stationary_point`, `has_inflection`, `has_asymptote`,
`has_closed_form_antiderivative`, and the four root properties (roots found in `domain`, so give a circular subject
a bounded interval). Not computed, so a subject that declares one overrides `check_property` itself:
`even_candidate`, `odd_candidate`, `unknown_constants_*`, `bounded_support`, `invertible_on_domain`,
`composite_of_translation`, `by_parts_integrable`, `substitution_integrable`. `expr_text(expr)` writes an expression
as question text (`x^2`, not `x**2`).
