# AGENTS.md

Course code repo for Udacity **CD13318 — Large Language Models and Text Generation**.
~24 Python exercise files, no build / test / lint tooling.

## Layout
- One top-level folder per lesson (e.g. `Strategic Model Selection & Economics /`, `Implementing RAG with Vector Databases`).
- Each lesson has one+ exercises, each with `starter/` (skeleton with TODOs) and `solution/` (reference answer key).
- Every exercise ships its own `README.md` (objectives + setup + completion checklist). Root `README.md` is only the course title.

## ⚠️ Paths have spaces — often a trailing space
Two top-level directories **end with a trailing space** (verified):
`Implementing Tokens Embeddings and Vector Search ` and `Strategic Model Selection & Economics `.
- **Always quote paths** and never assume a path has no spaces. Copying a path from `ls`/`find`
  output silently drops the trailing space and breaks the command.
- `cd`/`ls` into these dirs with a quoted name or a glob.

## Work in `starter/`, not `solution/`
- Exercises to edit live in `starter/`. Treat `solution/` as a reference — read it to check work,
  do not modify it.
- `solution/` is also indexed alongside `starter/`, so don't assume a file only exists in one place.

## Setup & running
- Python 3.14 locally. No shared `requirements.txt` — the NASA project is the exception
  (`Project-NASA-Mission-Intelligence-Starter/requirements.txt`).
- Exercises call the OpenAI SDK. Set the key or API calls fail:
  `export OPENAI_API_KEY=...` — stored in root `.env` (gitignored; course keys are Vocareum `voc-...`).
- Install deps per exercise (common: `openai pandas matplotlib seaborn numpy`), then run the script directly:
  `python3 <path>/file.py`
- No tests / CI. Syntax-check before running: `python3 -m py_compile <path>/file.py`.

## Tooling (both present and verified)
- `.serena/` configured (Python LSP, project `cd13318-exercises-project`). Scope symbol lookups with an
  explicit `relative_path` to stay inside `starter/`.
- `.codegraph/` index at root, but it indexes `starter` **and** `solution` — filter out `solution/`
  matches when working an exercise.
