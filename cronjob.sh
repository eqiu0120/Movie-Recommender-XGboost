#!/bin/bash
set -e

export PATH=/usr/bin:/bin:/usr/local/bin
echo "[CRON] Script entered at $(date)" >&2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/." && pwd)"
echo $REPO_ROOT

COMPOSE_FILE="$REPO_ROOT/docker/docker-compose.yml"

mkdir -p "$REPO_ROOT/logs"
echo "[CRON] Started at $(date)" >> "$REPO_ROOT/logs/retrain.log"

LOG_FILE="$REPO_ROOT/logs/retrain.log"
MAX_LINES=100000

if [ -f "$LOG_FILE" ]; then
  tail -n "$MAX_LINES" "$LOG_FILE" > "$LOG_FILE.tmp"
  mv "$LOG_FILE.tmp" "$LOG_FILE"
fi

LOCKFILE=/tmp/recsys_retrain.lock
exec 9>"$LOCKFILE" || exit 1
flock -n 9 || {
  echo "[CRON] Lock held, exiting" >> "$REPO_ROOT/logs/retrain.log"
  exit 0
}

cd "$REPO_ROOT"

# /usr/bin/docker compose -f "$COMPOSE_FILE" \
#   --profile retrain \
#   build retrainer >> "$REPO_ROOT/logs/retrain.log" 2>&1 # only need to build once

/usr/bin/docker compose -f "$COMPOSE_FILE" \
  --profile retrain \
  run --rm retrainer  >> "$REPO_ROOT/logs/retrain.log" 2>&1


STATUS=$?

if [ "$STATUS" -eq 10 ]; then
  echo "Model promoted → deploying service"
  /usr/bin/docker compose -f "$COMPOSE_FILE"  build flask-app >> "$REPO_ROOT/logs/retrain.log" 2>&1
  /usr/bin/docker compose -f "$COMPOSE_FILE"  down flask-app >> "$REPO_ROOT/logs/retrain.log" 2>&1
  /usr/bin/docker compose -f "$COMPOSE_FILE" up -d flask-app >> "$REPO_ROOT/logs/retrain.log" 2>&1
else
  echo "No promotion → skipping deploy"
fi

