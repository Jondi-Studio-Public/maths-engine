---
name: maths-skill-finder
description: Finds the right maths skill in the maths-engine library of over 1,300 skills and 2,500 question generators, by topic words, school year, Australian Curriculum (ACARA) code, domain or answer kind, and follows prerequisites to easier or harder skills. Use when someone names a topic, year level or curriculum code and you need a skill id before making questions, or when a learner is stuck and you need a skill to step back to.
license: MIT
---

# Finding maths skills

The library is a set of skills. Each skill generates an unlimited supply of questions on one narrow idea, for example `simplify-surds` or `factorise-quadratic`. Every other tool (`make_questions`, `build_paper`) needs a skill `id`, so finding the id comes first.

## Tools

- `find_skills(text, year, code, domain, framework, answer_kind, difficulty, tag, gradeable, limit)` returns `{"count": n, "skills": [...]}`. All filters combine with AND. Leave out any you do not need.
- `get_skill(skill_id)` returns the same summary plus `description`, `prerequisites`, `tags` and `misconceptions`.

Read `count`. If it is larger than the number of skills returned, narrow the search rather than paging through everything.

## Summary fields

| Field | Meaning |
|---|---|
| `id`, `name`, `domain` | Identifier, readable name and topic area (`quadratics`, `trigonometry`, `statistics-probability`, `primary-number` ...). |
| `answer_kind` | `number`, `mcq`, `expression`, `pair`, `fraction`, `ordered_list`, `list`, `table`, `inequality`, `matrix`, `set` or `grid_cells`. A skill whose answer needs a person to mark it would show `rubric`; the library today has none. |
| `gradeable` | `false` for a skill the engine cannot mark (a rubric skill, which this library does not currently contain); `true` for every skill here. |
| `sources` | The difficulty keys that `make_questions(source=...)` accepts, commonly `easy`, `medium`, `hard`, `exam`. |
| `years` | Years the skill belongs to: `F`, `1` to `10`, `10A`, `11`, `12` or `university`. Empty for some skills, see below. |
| `curriculum_codes` | ACARA content description codes the skill is tagged with. |
| `frameworks` | `acara-f10-v9`, `acara-senior-v8-4` or `university-core-v1`. |
| `learner_band` | `early`, `primary` or `general`. |
| `needs_figure` | The questions come with a diagram. |

## Searching

1. Text: `find_skills(text="surd")` matches the id, name and description, ignoring case. Results come in library order, not ranked by relevance, so read the whole list (or add a filter) rather than taking the first. Use Australian spelling ("factorise", "metre") and try a synonym if the first word finds nothing.
2. Year: `find_skills(year="8")`. Values are `F`, `1` to `10`, `10A`, `11`, `12` and `university`.
3. Curriculum code: `find_skills(code="AC9M8N")` matches by prefix.
4. Domain, answer kind, framework: use these to narrow a broad result, for example `domain="trigonometry", year="10"`.
5. Check the best candidates with `get_skill` and read the description. Names can be close and descriptions say exactly what is covered and what is excluded.

Combine filters gradually. A text search with a year filter is the most common useful query.

## Reading curriculum codes

Australian Curriculum Version 9 (F to 10) codes look like `AC9M<year><strand><nn>`:

- `AC9M` is Australian Curriculum v9 Mathematics.
- `<year>` is `F`, `1` to `10` or `10A`.
- `<strand>` is a letter or two: `N` Number, `A` Algebra, `M` Measurement, `SP` Space, `ST` Statistics, `P` Probability.
- `<nn>` is the two-digit descriptor number within that year and strand.

So `AC9M8N05` is Year 8, Number, descriptor 5. A prefix works as a filter: `AC9M8` is all of Year 8, `AC9M8A` is Year 8 Algebra.

Senior Secondary (Version 8.4) codes look like `ACMMM092` or `ACMGM003`: `ACM`, a two-letter course code (for example `MM` Mathematical Methods, `SM` Specialist Mathematics, `GM` General Mathematics, `EM` Essential Mathematics), then a three-digit number. Use the code prefix of a course (`ACMMM`) to list a whole course.

A skill can carry several codes, even from different years, when one skill serves several descriptors. A skill with no code is not necessarily outside the curriculum; it may simply be untagged.

## Gaps to know about

- `years` is empty for about a quarter of the library (roughly 330 skills): about 60 senior skills and many primary and secondary ones such as `add-within-10`, `simplify-surds` and the differential-equations and financial skills. A year filter therefore finds fewer skills than exist. If a year search comes back thin, also search by text or code, or by the neighbouring years.
- The library includes university-level skills (framework `university-core-v1`, ids often starting `ucv1-`): groups, rings, fields, linear algebra and so on. They carry the year `university`, so `year="university"` finds them; `framework="university-core-v1"` finds the same skills by framework.
- A skill summary has an `attribution` field when ACARA attribution is required (senior skills have it). Keep that text whenever you share or print material from such a skill. Decide by the field, not by the shape of the id.

## Prerequisites

`get_skill` lists `prerequisites`, which are skill ids. For example `simplify-surds` lists `f10-square-roots-perfect-squares` and `prime-factorisation`. Use them to step back when a learner keeps getting a skill wrong, or to check what a topic builds on when planning a sequence. Fetch the prerequisite with `get_skill` to confirm it is gradeable and see its sources. To find what a skill leads to, search by text or code for the next topic; the library does not list dependants.

## Reporting back

When you offer choices, give the `id`, the name, the years or codes, and one line of what the description covers. Do not invent ids: if nothing matches, say so and offer the nearest topics.
