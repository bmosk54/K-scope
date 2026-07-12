#!/usr/bin/env bash
# Expose the dashboard backend (this SageMaker box) on a public URL via a Cloudflare
# quick tunnel — no account, no Studio presigned URL, no manual port-forward. SageMaker
# stays backend-only; Cloudflare just forwards the port.
#
#   bash dashboard/serve.sh     # 1) start the Node server on :4173 (once)
#   bash dashboard/tunnel.sh    # 2) print a https://<random>.trycloudflare.com URL
#
# NOTE: this is a TEMPORARY, PUBLIC URL. It exposes /api/certify_answer (which runs
# compute + Bedrock on POST). Random subdomain + short-lived; kill cloudflared when done.
set -euo pipefail
PORT="${PORT:-4173}"
BIN="${CLOUDFLARED:-/tmp/cloudflared}"

if ! [ -x "$BIN" ]; then
  echo "downloading cloudflared -> $BIN"
  curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
    -o "$BIN"
  chmod +x "$BIN"
fi

LOG="/tmp/biolayer-tunnel.log"
: > "$LOG"
nohup "$BIN" tunnel --url "http://localhost:${PORT}" --no-autoupdate > "$LOG" 2>&1 &
echo "cloudflared PID $! (log: $LOG)"
echo -n "waiting for URL"
for _ in $(seq 1 30); do
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" | head -1 || true)
  [ -n "${URL:-}" ] && break
  echo -n "."; sleep 1
done
echo
if [ -n "${URL:-}" ]; then echo "OPEN: $URL"; else echo "no URL yet — see $LOG"; fi
