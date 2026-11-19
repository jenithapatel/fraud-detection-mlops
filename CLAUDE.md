# CLAUDE.md: rules for every Claude Code session in this repo

Team project for ADSP 32021 (MLOps). Four humans share this repo, each working
with their own Claude Code. Read `HANDOFF.md` for context, roles and tasks.

## Attribution (strict)
- Never add `Co-Authored-By`, `Claude-Session`, "Generated with Claude Code",
  or any other Claude/Anthropic attribution to commit messages, PR titles,
  PR descriptions, code comments, docs or the README.
- Commit only with the human's own git identity (`git config user.name/email`).
  Never change git identity and never commit as a bot.
- `.claude/settings.json` disables attribution and `.githooks/commit-msg`
  strips it; CI fails any PR that still contains it. Don't bypass these
  (no `--no-verify`).

## Git workflow
- Never push to `main`, never force-push, never rewrite shared history.
- Before starting work: `git checkout main && git pull`, then create a branch
  named `<area>/<short-task>` (area = data, model, serving, monitoring, docs).
- Small, focused commits with imperative messages ("Add Evidently drift report").
- Finish by pushing the branch and opening a PR with `gh pr create`, using the
  PR template. A human teammate approves and merges.
- If the branch is behind main: `git fetch && git rebase origin/main`.

## Code rules
- Python 3.11, package code in `src/fraud/`, run modules with `python -m fraud.<...>`.
- All paths, column names, model names and thresholds come from
  `src/fraud/config.py`. Don't hardcode them. Changing `config.py`, the
  `/predict` schema or the MLflow model name/alias is a team decision: say so
  in the PR description.
- Stay inside your own stage's folder unless the task needs otherwise; if you
  must edit a shared file (`requirements*.txt`, CI, `docker-compose.yml`),
  call it out in the PR.
- Every new function gets a docstring; non-obvious lines get comments (graded).
- Add or update tests in `tests/`. Before committing run: `ruff check . && pytest -q`.
- Never commit data, `mlruns/`, `mlflow.db`, model files, `.env` or `kaggle.json`.
- Never touch `data/processed/test.csv` during training or tuning; it is only
  for production validation through the API.
