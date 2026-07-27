#!/bin/bash
# Periodically commit scene-sort logs as an off-site backup while sorting is
# in progress. Surgical: only stages sort_log.csv files, never other changes.
# Exits (and does a final commit) once the full batch has no pending sheets.
set -u
REPO=/home/jovyan/scaleworm-student-lab
FULL="$REPO/scene_sorting/full_2023_2024/contact_sheets"
COAUTHOR="Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"

commit_logs() {
    local msg="$1"
    cd "$REPO" || return
    git add scene_sorting/*/sort_log.csv 2>/dev/null
    if ! git diff --cached --quiet 2>/dev/null; then
        git commit -q -m "$msg" -m "$COAUTHOR" 2>/dev/null && git push -q origin main 2>/dev/null
    fi
}

while true; do
    sleep 1200  # 20 minutes
    commit_logs "Checkpoint scene-sort logs ($(date -u +%Y-%m-%dT%H:%MZ))"
    pending=$(ls "$FULL"/*.png 2>/dev/null | wc -l)
    if [ "$pending" -eq 0 ]; then
        commit_logs "Final scene-sort log checkpoint — full batch sorted"
        break
    fi
done
