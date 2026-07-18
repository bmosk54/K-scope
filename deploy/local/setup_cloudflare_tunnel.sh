#!/usr/bin/env bash
# One-time setup for a PERMANENT public URL, replacing the random-subdomain
# `trycloudflare.com` quick tunnel with a NAMED Cloudflare Tunnel bound to a hostname
# you choose (e.g. kscope.yourdomain.com). The hostname never changes; only the tunnel
# process needs to be running (bin/kscope start/stop), and it's free.
#
# Requires:
#   - a free Cloudflare account (https://dash.cloudflare.com/sign-up)
#   - a domain added to that Cloudflare account as a "zone" (any registrar's domain
#     can be added to Cloudflare for free — Cloudflare just needs to manage its DNS).
#     If you don't have a domain yet, buy any cheap one (~$10/yr) and add it to
#     Cloudflare, or skip this script entirely — `bin/kscope start` still works fine
#     serving http://localhost:4173 with no tunnel at all.
#
# This script is interactive in two places (both open your browser):
#   1. `cloudflared tunnel login`  — approve the CLI against your Cloudflare account
#   2. nothing else — DNS routing is done here for you via the API once logged in
set -euo pipefail

HOSTNAME="${1:-${CLOUDFLARE_HOSTNAME:-}}"
TUNNEL_NAME="${CLOUDFLARE_TUNNEL_NAME:-kscope}"
CFG_DIR="$HOME/.cloudflared"
CFG_FILE="$CFG_DIR/kscope-config.yml"
PORT="${PORT:-4173}"

if [ -z "$HOSTNAME" ]; then
  echo "usage: $0 <hostname>   e.g. $0 kscope.yourdomain.com" >&2
  echo "  (or export CLOUDFLARE_HOSTNAME=kscope.yourdomain.com first)" >&2
  exit 1
fi

echo "=== 1/4  cloudflared binary ==="
if ! command -v cloudflared >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    brew install cloudflared
  else
    echo "install cloudflared first: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/" >&2
    exit 1
  fi
fi
cloudflared --version

echo "=== 2/4  Cloudflare login (opens your browser once) ==="
if [ ! -f "$CFG_DIR/cert.pem" ]; then
  cloudflared tunnel login
else
  echo "already logged in ($CFG_DIR/cert.pem exists)"
fi

echo "=== 3/4  create (or reuse) the named tunnel '$TUNNEL_NAME' ==="
if cloudflared tunnel list 2>/dev/null | awk '{print $2}' | grep -qx "$TUNNEL_NAME"; then
  echo "tunnel '$TUNNEL_NAME' already exists — reusing it"
else
  cloudflared tunnel create "$TUNNEL_NAME"
fi
TUNNEL_ID="$(cloudflared tunnel list 2>/dev/null | awk -v n="$TUNNEL_NAME" '$2==n {print $1}' | head -1)"
CRED_FILE="$CFG_DIR/${TUNNEL_ID}.json"

echo "=== 4/4  point $HOSTNAME at the tunnel + write config ==="
cloudflared tunnel route dns "$TUNNEL_NAME" "$HOSTNAME" || \
  echo "  (route may already exist — that's fine)"

cat > "$CFG_FILE" <<YML
tunnel: ${TUNNEL_ID}
credentials-file: ${CRED_FILE}
ingress:
  - hostname: ${HOSTNAME}
    service: http://localhost:${PORT}
  - service: http_status:404
YML

echo
echo "Wrote $CFG_FILE"
echo "Done. From now on:"
echo "  bin/kscope start    -> backend + tunnel come up, dashboard live at https://${HOSTNAME}"
echo "  bin/kscope stop     -> both go down, URL stops resolving (nothing left running)"
