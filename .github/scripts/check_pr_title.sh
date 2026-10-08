#!/usr/bin/env bash
# usage: TITLE="..." .github/scripts/check_pr_title.sh — the squash-merge commit title comes from the PR title
set -u
re='^(feat|fix|docs|test|refactor|perf|build|ci|chore|i18n|revert)(\([a-z0-9-]+\))?!?: [a-z0-9`].*[^.]$'
fail() { echo "::error title=PR title::$1"; echo "  title: $TITLE"; echo "  expected: type(scope): lower-case summary, e.g. 'feat(label): add a validate command'"; exit 1; }
[[ "$TITLE" =~ $re ]] || fail "must look like 'type(scope): summary' (types: feat fix docs test refactor perf build ci chore i18n revert), start lower-case and not end with a period"
(( ${#TITLE} <= 72 )) || fail "is ${#TITLE} characters; keep it at 72 or fewer (squash merges append ' (#PR)')"
[[ "$TITLE" =~ \#[0-9] ]] && fail "must not reference issues or PRs; link issues with 'Closes #n' in the description (GitHub appends the PR number itself)"
echo "PR title ok: $TITLE"
