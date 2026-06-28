#!/usr/bin/env bash
# PostToolUse hook: auto-commit and push progress_log.md when Claude edits it.
set -euo pipefail

# Read tool input JSON from stdin
input=$(cat)

# Only act if the edited file is progress_log.md
echo "$input" | python3 -c "
import json, sys
d = json.load(sys.stdin)
fp = d.get('tool_input', {}).get('file_path', '')
sys.exit(0 if 'progress_log.md' in fp else 1)
" || exit 0

REPO="/home/jovyan/scaleworm-student-lab"
cd "$REPO"

# Nothing to do if file is unchanged
if git diff --quiet progress_log.md 2>/dev/null && ! git ls-files --others --exclude-standard | grep -q "^progress_log.md$"; then
    exit 0
fi

git add progress_log.md
git commit -m "Update progress log [$(date -u '+%Y-%m-%d %H:%M UTC')]"
git push origin main
