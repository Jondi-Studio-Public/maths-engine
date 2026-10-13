# Paper spec

The spec is a JSON object passed to `build_paper`. This describes exactly what the engine reads; no other fields do anything.

```json
{
  "title": "Year 8 surds check",
  "seed": 42,
  "versions": 2,
  "shuffle": false,
  "sections": [
    {
      "title": "Part 1",
      "items": [
        {"skill": "simplify-surds", "source": "easy", "count": 4}
      ]
    }
  ]
}
```

## Top-level fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `sections` | list of sections | yes | At least one. |
| `title` | string | no | Truncated to 200 characters. Defaults to "Worksheet". |
| `seed` | integer, 0 to 2^63 - 1 | no | A whole number (not true/false). If left out, a random seed is chosen and returned in the built paper; note it down to rebuild the same paper. The same spec, seed and library give the same paper. |
| `versions` | integer, 1 to 10 | no | Default 1. Versions are labelled A, B, C ... and have different numbers. |
| `shuffle` | boolean | no | Default false. Shuffles question order within each section, per version. Questions are numbered after shuffling. |

## Sections

| Field | Type | Required | Notes |
|---|---|---|---|
| `items` | list of items | yes | At least one. |
| `title` | string | no | A heading for the section. |

## Items

| Field | Type | Required | Notes |
|---|---|---|---|
| `skill` | string | yes | A skill id from `find_skills`. |
| `source` | string or integer | no | A source key or difficulty label (`easy`, `medium`, `hard`, `exam`; the skill's `sources` list shows what exists), or a 0-based index into the skill's sources. If left out, the skill's first source is used. |
| `count` | integer, 1 to 100 | no | Default 1. Questions from this item. |

The engine tries a few times to avoid repeating the same wording within one item, then accepts a repeat if the question pool is small.

## Limits

- At most 400 questions in total across all versions (`total count x versions`).
- Unknown skills or sources are rejected with an error that names the options.

## What comes back (format "json")

An object with `title`, `seed`, `answers` (whether the answers are included), `versions` (each with a `label` and `sections`, whose `questions` are question views with a `number`), `attribution` (list of strings) and `engine`. Every question has an `id`, so it can be marked later with `grade_answer`.

## Examples

### A single-topic worksheet

Ten questions climbing in difficulty, one version.

```json
{
  "title": "Simplifying surds",
  "seed": 101,
  "sections": [
    {
      "items": [
        {"skill": "simplify-surds", "source": "easy", "count": 4},
        {"skill": "simplify-surds", "source": "medium", "count": 4},
        {"skill": "simplify-surds", "source": "hard", "count": 2}
      ]
    }
  ]
}
```

### A mixed test with two versions

Sections per topic, shuffled order, two versions against copying.

```json
{
  "title": "Algebra test",
  "seed": 2026,
  "versions": 2,
  "shuffle": true,
  "sections": [
    {"title": "Section A: Skills",
     "items": [{"skill": "factorise-quadratic", "source": "easy", "count": 5}]},
    {"title": "Section B: Problems",
     "items": [{"skill": "factorise-quadratic", "source": "exam", "count": 3},
               {"skill": "polynomial-long-division", "source": "medium", "count": 2}]}
  ]
}
```

### A curriculum-coded revision sheet

Skills found by `find_skills(curriculum_code="AC9M8N")`, one question each.

```json
{
  "title": "Year 8 Number revision",
  "seed": 7,
  "sections": [
    {"title": "Number",
     "items": [
       {"skill": "f10-rational-percentage-financial-model", "count": 1},
       {"skill": "f10-rational-operation-strategy", "count": 1},
       {"skill": "f10-terminating-recurring-decimals", "count": 1}
     ]}
  ]
}
```

Always run these through `make_questions` or a `json` build before relying on them: a skill that has no source called `exam` is rejected, and the list of valid sources differs per skill.
