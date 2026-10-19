# Open-response hand-off and typed proof steps

Status: built (task w10). Code: `mathgen/grading/open_response.py`, `mathgen/grading/chain.py`.

## Ruling

1. Proofs and derivations are typed as steps and the engine checks every line (`step_chain` kind).
2. Genuinely open answers (explain, justify, describe) are **not marked by the engine**. It hands a
   standard marking request to whoever the implementer chooses (a model, a teacher, the learner) and folds the
   standard marking result back. No model runs in this repo.

## What main already had

- Answer kinds are the `GRADERS` registry; `ANSWER_KINDS` must equal it. No rubric or free-text kind existed.
- Skills with an `assessment_contract` but no marking path were deferred; there is no rubric library in this repo.
- Chain-proof generators (`_chain_proof_kit`) carried a marking dict but no grader read it.

## Marking request (`maths.marking-request/1`)

JSON object: `schema`, `request_id` (16 hex of sha256 over question, reference and rubric, so the same
question gives the same id), `skill`, `question`, `answer` (the learner's text, at most 8000 characters),
`reference` (model answer), `criteria` (ordered list of `{id, description, marks, levels?}`),
`total_marks`, `exemplars?` (`{label, marks, answer}`). Built by `build_marking_request(question, given)`.

## Marking result (`maths.marking-result/1`)

`schema`, `request_id`, `marker` (`model`, `teacher`, `self` or `other`), `marks` (`{criterion id: whole
number}`), `total`, `feedback` (text). `parse_marking_result` accepts JSON text only (no NaN, 64 kB cap, nothing
evaluated). `validate_marking_result(request, result)` rejects: wrong schema, request id mismatch, unknown or
missing criteria ids, marks that are not whole numbers (bool and float refused), marks outside 0..criterion
marks, a total that is not the sum, an unknown marker. `fold_marking_result` returns the final
`OpenResponseResult`; `correct` is true only for full marks.

## The `open_response` kind

`Question.answer` is the rubric `{reference, criteria, exemplars?}` (checked by `rubric_of`). The grader
returns an `OpenResponseResult` with `pending=True`, `correct=False`, `max_marks` = rubric total, and no model
answer in any result field the learner sees.

## Self-marking

`self_marking_checklist(request)` lists the criteria as yes/no statements; `self_marking_result(request, ticks)`
builds a result with `marker="self"` and full marks per ticked criterion.

## Prompt helper

`render_marking_prompt(request)` returns text only: instructions, rubric, reference, then the learner answer in a
fence that is stated to be data, and a JSON skeleton to fill in. It makes no API call.

## Scheduler and mastery

Minimal safe treatment: a pending result is neither right nor wrong. `scheduling.review.outcome_from_marks`
drops pending results; if every result is pending it raises `ValueError`. `counts_toward_mastery(result,
accept_markers=...)` lets the implementer decide whether, say, self-marks count. The default accepts all
markers; strict deployments pass `("model", "teacher")`.

## Typed steps (`step_chain`)

The learner types one expression per line (or joins them with `=`). The grader checks the first line has the
question's start structure, each line equals the one before at exact rational probe points on the domain the
question states (`real`, `positive`, `integer`, `nonzero`), under an optional `given` hypothesis, and the last
line has the target's structure. Relations are refused, 12 lines at most, notes name the failing line and never
the answer. Roots may be typed `sqrt(2)` or the surd sign.

## Skills built

Open response (`*-explain`): meaning-of-equality, inverse-operations, solution-and-solution-set,
identity-and-no-solution, check-solution-by-substitution, recognise-equivalent-expressions,
independent-dependent-variables, proportional-vs-non-proportional.
Typed proofs: proof-irrational-surd-steps, proof-parity-steps, proof-am-gm-steps, proof-triangle-inequality-steps.

## Needs a human

- Choose and operate a marker (a model service, a teacher queue, or self-marking UI); the engine has none.
- Decide whether self-marks count toward mastery and whether `pending` items show in progress views.
- Rubric wording review by a teacher; the rubrics here are drafts.
- Regenerate `skills/index.yaml` and `tests/golden/questions.txt`.
- Deferred assessment-contract skills that need a sketch or draw answer kind are not covered here.
