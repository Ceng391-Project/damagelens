# DamageLens review guidelines

Read by the Claude review workflow on every pull request. Review against these project rules, not general style preferences. Skip anything a linter, formatter or the test suite already checks. If nothing below is violated, say so in one line.

## Severity
- **Blocking** — breaks an evaluation, leaks data, or commits something that must not be in git. Explain the concrete failure.
- **Should fix** — makes a reported number misleading or a result irreproducible.
- **Nit** — only if it changes meaning; otherwise do not comment.

## Evaluation integrity (blocking)
- No overlap between training and evaluation data: KATE-CD `test` must never be used for training, threshold tuning or early stopping (thresholds come from `validation`). Hand-labelled areas used for fine-tuning must not be the ones used as the raw-Maxar test region.
- Every metric in README, `docs/` or a PR description must state dataset, split and level (pixel / building / AUC) and must match a JSON written by code in this repo. Flag numbers that appear in prose but are not produced by a script.
- Thresholds must be chosen on validation data. A threshold picked on the test set is only acceptable when the text explicitly calls it optimistic.
- Changing a loss, augmentation or split must come with re-run numbers, not the old ones.

## Pre/post imagery
- Labels are drawn in the **post** frame. Any synthetic misregistration or alignment must move the **pre** image only (see `damagelens.augment.misregister`, `damagelens.align`).
- Channel order is pre RGB then post RGB (6 channels), normalised with `damagelens.models.normalize`. Flag code that builds inputs another way.
- Raw Maxar reads must keep the UTM grid and GSD from the manifest; flag code that resamples without recording the GSD.
- Snow/cloud and missing pixels must be masked or excluded from metrics, not counted as "no damage".

## Labels (`labels/<area>/`)
- Only `manifest.json` and `annotations/*.json` are committed; never image tiles or exported GeoJSON.
- Editing a committed `manifest.json` (dates, GSD, size) silently changes every tile and invalidates existing annotations — blocking unless all annotations of that area are redone.
- Bulk-deleting or rewriting other people's annotations needs a reason in the PR description.

## Data, weights and licences (blocking)
- Never commit datasets, image tiles, model weights (`*.pt`), `runs/`, `outputs/` or anything under `data/`.
- xBD (CC BY-NC-SA 4.0) and Maxar Open Data (CC BY-NC 4.0) are non-commercial; KATE-CD's licence is unknown (#4). Flag anything that publishes data or models publicly (e.g. a public Hugging Face repo) without addressing this.
- No credentials, tokens or API keys in code, notebooks or workflow files.

## Code
- New system code goes in `src/damagelens/`; `experiments/` is frozen and only changes to regenerate existing figures. Flag new features added under `experiments/`.
- Use uv only: no `pip install`, `requirements.txt` or manual venv instructions; new dependencies go through `uv add` with an updated `uv.lock`.
- Paths are derived from `damagelens.REPO_ROOT` / `DATA_DIR` / `RUNS_DIR` or the script's own location — no absolute or user-specific paths.
- Comments only for a non-obvious reason; no comments or docstrings that restate the code.
- Everything user-facing (README, docs, CLI help, UI text, figure labels) is in English.
- New behaviour in `src/damagelens/` comes with a test in `tests/` that runs without downloaded data.

## Git
- Commit messages: one line, `type(scope): ...`, referencing the issue (`(#12)`); no body, no trailers.
- The PR description maps each part to its issue and uses `Closes #n`.
- The PR title becomes the squash commit on `main`: `type(scope): summary`, lower-case, ≤72 characters, no `#` references (checked by CI).
