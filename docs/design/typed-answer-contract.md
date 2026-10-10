# Typed answers and misconception-tagged distractors

Status: contract and pilot implemented (five generators). Bulk migration is follow-on work.

## Why

Before this change a question's answer was four loose attributes (`answer`, `answer_text`, `answer_latex`,
`answer_form`) whose kind only the serving skill knew, and its distractors were bare strings. A wrong answer could be
marked wrong but not explained:

- 6 of the 400 generator modules tagged any distractor (`option_notes`), all in the `pv1*` figure batch.
- A tag was an id into the serving skill's own `misconceptions` list. Those ids are mostly placeholders
  (`misconception-1`), and one generator serves several skills (`limits` serves four, `linear_equation` eight), so a
  generator could not know which id to emit.
- A typed reply matched a tag only if it was spelled exactly like the option: `x=84` never matched `x = 84`.

## The contract (`mathgen/answers.py`)

Everything is read by attribute name, never by type, so a question, or any object with the same attributes (a part
of a multi-part question), is graded and diagnosed the same way.

| Attribute | Meaning |
|---|---|
| `answer` | The structured, JSON-serialisable value. Unchanged. |
| `answer_text`, `answer_latex` | Canonical spellings. Unchanged. |
| `answer_kind` | **New, optional.** The grader kind of `answer`, one of `ANSWER_KINDS` (= the keys of `GRADERS`). |
| `answer_form` | The shape a reply must have, one of `ANSWER_FORMS`; `make()` now rejects any other value. Order is not a form: it is the difference between the `list`/`set` kinds and `ordered_list`. |
| `answer_alt` | A numeric alternative a grader may accept. Unchanged. |
| `distractors` | Option texts, still plain strings to every consumer. |
| `option_notes` | Option text -> misconception id. Existing; now usually filled by `make()`. |
| `distractor_values` | **New, optional.** Option text -> the value a learner making that error reaches, shaped like `answer`. |
| `distractor_credit` | **New, optional.** Option text -> share of the marks that error still earns, as a fraction string (`"1/2"`). |

`answer_spec(obj, kind=None)` returns a frozen `AnswerSpec(kind, value, text, latex, form, alt)` view; nothing stores
it. The question's `answer_kind` is the kind of the *value*; the skill's `answer` is how it is *graded*. They agree,
or the skill is `mcq` (any value can be an option), or one of the listed widenings holds (`kind_accepts`).

### Tagged distractors

A generator returns `Distractor(text, misconception=None, value=None, credit=None)` objects in `distractors`, mixed
freely with plain strings. `pick_distractors` keeps them, tag and all. `make()` flattens them into the attributes
above (`flatten_distractors`) and rejects an id that is not in the catalogue or an option tagged twice.
`tagged_distractors(obj)` reassembles them.

`value` is what lets a reply be matched by meaning rather than spelling. Give it for every distractor that has one;
leave it out only for words with no value (`undefined`, `does not exist`). `credit` is for a right method with a slip
(1/2 for a rounding error, say); it must be at least 0 and below 1. The engine stores it and does not compute marks.

### Backwards compatibility

- A generator that returns strings produces byte-identical output: the new attributes are omitted from `to_dict()`
  and `to_json()` when empty, as `answer_alt` is.
- Existing `option_notes` with skill-local ids (the `pv1*` generators) resolve exactly as before.
- `GradeResult` gains two optional fields, `misconception` (the id) and `matched_distractor` (the option text, the key
  for its credit). Nothing else in it changes.

## The catalogue (`mathgen/misconceptions.py`)

Ids are the engine's, not a skill's: an id names an error ("sign-flip-one-root"), means the same thing everywhere,
and so can be counted across skills for one learner. Each entry has a learner-readable description, a `kind`
(`misconception` for wrong thinking, `near_miss` for right thinking that answered a different question, matching the
two YAML lists' feedback tone) and an `area`.

A skill's YAML `misconceptions` or `near_misses` entry with the same id overrides the wording for that skill; grading
looks there first, then the catalogue. Add an entry when a generator needs one; reuse an id whenever it says the same
thing.

## Diagnosis

Only a reply graded wrong is diagnosed. Its option is found in this order, and the first hit wins:

1. The option's exact text (the existing `option_notes` path; for `mcq`, the chosen option).
2. For each tagged option with a value: the reply is graded, by the same grader, against a shallow copy of the
   question whose answer is that value (`answer_form` and `answer_alt` cleared). Matching is exactly as forgiving as
   grading: `84`, `x=84` and `x = 84.0` all match the option `x = 84`; `7, -6` matches `x = -6 or x = 7` for a list.
3. For a tagged option without a value: the text, ignoring case and spacing.

A reply that two errors both explain (the change and the change over the final value are both 20 for 80 -> 100) is
credited to the first-listed option, so list distractors in order of likelihood. The cost is at most one extra grader
call per tagged option, on wrong replies only.

`engine.grade(skill, q, reply)` and `grade(kind, q, reply)` both fill `misconception` and `matched_distractor`; `note`
and `note_kind` come from the skill when it is passed and lists the id, else from the catalogue. The command line
prints the diagnosis under a wrong verdict.

## Seam with the multi-part question model

A part carries the same answer attributes and is graded with `grade(kind, part, reply)`. The part's kind field is
named `answer_kind` so one accessor serves both. The multi-part model owns `marks`, `max_marks` and `parts` on
`GradeResult`, and computes marks from `distractor_credit[matched_distractor]`.

## The pilot

| Generator | Paths tagged | Kinds |
|---|---|---|
| `limits` | every path | number, fraction, expression |
| `linear_equation` | `standard` | number |
| `quadratic` | `factor` | list |
| `function_evaluation` | the one path | number |
| `percentage_change` | both directions | number |

Checking each distractor was an error a learner really makes (review recommendation R13) removed some that were not:

- `function_evaluation` offered `-a n^2` ("squared the sign away") for a positive input, where no sign is lost. Sign
  slips are now offered only for a negative input; squaring as doubling and dropping the constant are offered always.
- `percentage_change` offered `100 - p` and a `0` (p/100 rounded to a whole number) going forward, and `final x 100 / p`
  and `original + p` (which needs the answer to compute) in reverse. They are replaced by dividing by the final value,
  or dropped.
- `quadratic` now offers the positive root alone for x^2 - k^2 = 0, where the sign flips cannot move the roots.
- Fallbacks that stand for no error (off by one) stay, untagged.

`percentage_change` is set in Australian contexts (rent in Brisbane, a surf life saving club, a price including 10%
GST) instead of "a value".

## Migrating a generator

1. Import `Distractor` (and `exact_value` for rationals) from `mathgen.answers`.
2. Build each candidate as `Distractor(text, "<catalogue id>", value)`, adding catalogue entries as needed. Ask of
   each one: would a real learner make this error? Drop it if not, or keep it untagged if it is a needed filler.
3. Set `answer_kind`.
4. Add the topic to `_PILOT_TOPICS` in `tests/test_typed_answer_contract.py` (rename the list when it is no longer a
   pilot), which checks every tagged distractor, spelled as shown and spelled plainly, is diagnosed as its own error
   for every skill the generator serves.

## Open questions for the maintainer

1. **Skill YAML misconception ids.** Most are placeholders (`misconception-1`). Recommend rewriting them to catalogue
   ids as each generator migrates, keeping a skill-specific description only where it adds something.
2. **Logging.** The app that records attempts should store `GradeResult.misconception` with each wrong attempt; that
   is work in the private app, not the engine.
3. **Why not tag with the skill YAML ids?** 686 of 719 skills have a list, so it is tempting. But a generator serves
   several skills whose lists differ, and most ids are placeholders, so a tag could only be right for one of them.
   Catalogue ids with skill wording on top keep the YAML lists useful without tying a generator to one skill.
4. **Multiple-choice presentation** (review F4, not in this change): the answer sits inside the option range too
   often, its slot depends on the seed alone (`mcq_options` shuffles with `Random(seed)`), some skills show two or
   three options, and some probability skills offer impossible values. Each is a separate change; the shuffle one
   changes every stored option order, so it needs a decision on in-flight questions first.
5. **Grader gaps found on the way** (not fixed here): the expression grader does not accept `infinity` for an answer
   of `∞`, and the fraction grader rejects `x = 50/7` against its own canonical `x = 50/7`.
