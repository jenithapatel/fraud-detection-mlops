#!/bin/sh
# CI guard: fail if any commit in the range mentions Claude/Anthropic attribution.
# Usage: scripts/check_attribution.sh <base_ref> <head_ref>
range="$1..$2"
bad=$(git log --format='%H%n%an <%ae>%n%cn <%ce>%n%B' "$range" \
  | grep -iE 'co-authored-by:.*(claude|anthropic)|claude-session:|generated with \[?claude|noreply@anthropic\.com' || true)
if [ -n "$bad" ]; then
  echo "AI attribution found in commits ($range):"; echo "$bad"
  echo "Fix with: git rebase -i <base> and reword those commits, or see CLAUDE.md."
  exit 1
fi
echo "No AI attribution found."
