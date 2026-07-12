#!/usr/bin/env bash
# Credential watchdog for the dashboard's Bedrock-backed API.
#
# WHY: app_server.py holds the AWS_SESSION_TOKEN it launched with. These are AWS Workshop
# Studio (WSParticipantRole) temporary STS creds — they expire (~1 h) and CANNOT be refreshed
# programmatically; a human re-pastes them into .owkin_hack_aws.sh from the workshop portal.
# When that happens the running server still holds the OLD token, so InvokeModel fails with
# ExpiredTokenException until it is restarted.
#
# WHAT THIS DOES: every INTERVAL seconds it re-reads .owkin_hack_aws.sh; when the access key
# changes AND is valid, it restarts app_server so the new creds take effect — no manual
# restart. It CANNOT invent creds: if the file still holds expired ones it just logs that they
# need refreshing. Runs detached (setsid) so it survives terminal/session close (not reboot).
#
#   start:  setsid bash dashboard/cred_watchdog.sh </dev/null >/tmp/cred_watchdog.log 2>&1 &
#   stop:   pkill -f cred_watchdog.sh
#   log:    /tmp/cred_watchdog.log
set +e
ROOT=/home/orthodim/owkin_hack
CREDS="$ROOT/.owkin_hack_aws.sh"
PY="$HOME/miniconda3/envs/owkin-env/bin/python"
PORT="${PORT:-8080}"
INTERVAL="${INTERVAL:-120}"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null && conda activate owkin-env 2>/dev/null
cd "$ROOT/dashboard" || exit 1

log() { echo "$(date '+%F %T') $*"; }
last=""
log "watchdog up (interval=${INTERVAL}s, port=${PORT}) — watching $CREDS"
while true; do
  # shellcheck disable=SC1090
  source "$CREDS" 2>/dev/null
  key="$AWS_ACCESS_KEY_ID"
  if aws sts get-caller-identity >/dev/null 2>&1; then
    # creds are valid. Restart if they changed since we last (re)started the server, or if the
    # server isn't running at all.
    if [ "$key" != "$last" ] || ! pgrep -f app_server.py >/dev/null 2>&1; then
      pkill -f app_server.py 2>/dev/null
      PORT="$PORT" setsid "$PY" app_server.py </dev/null >/tmp/biolayer-dashboard.log 2>&1 &
      last="$key"
      log "restarted app_server on :$PORT with creds ${key:0:12}… (Bedrock live)"
    fi
  else
    log "creds in $CREDS are INVALID/EXPIRED (key ${key:0:12}…) — paste fresh ones from the workshop portal; will pick them up within ${INTERVAL}s"
  fi
  sleep "$INTERVAL"
done
