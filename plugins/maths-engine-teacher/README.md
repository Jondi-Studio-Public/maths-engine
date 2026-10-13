# maths-engine-teacher

Worksheets, quizzes and tests for teachers, built by an AI agent from verified question generators. The agent maps
your request (topic, year, curriculum codes, length, difficulty mix) to skills, samples the questions with their
answers, builds the paper, and writes a student sheet and a separate answer key. Several versions of a test differ in
their numbers, to discourage copying. It can also export Moodle XML and QTI 2.1 question banks.

## Skills

- `maths-worksheets`: from a request to a checked paper spec, a printable worksheet and a key.
- `maths-lms-export`: Moodle XML and QTI 2.1 (Canvas and others), including which answer kinds an LMS can mark
  itself and which are text matches a teacher should review.
- `maths-skill-finder`: finding skills by topic, school year, ACARA code, domain or answer kind.

## Tools (server role `author`)

`engine_info`, `find_skills`, `get_skill`, `make_questions` (`answers`), `get_question` (`answers`), `grade_answer`
(`reveal`), `build_paper` (`json`, `html`, `html-key`, `moodle`, `qti`). Files are written to the plugin's data
directory and the agent reports the path. See `docs/mcp.md`.

This server can show every answer. Keep it away from learners.

## Install

Needs [uv](https://docs.astral.sh/uv/) on your `PATH`. See [../README.md](../README.md) for Claude Code and other
clients.

## Licences

Code MIT, skills CC BY 4.0, senior content adapted from ACARA under CC BY 4.0. Keep the attribution printed on
papers when you share them.
