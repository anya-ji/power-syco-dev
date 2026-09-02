#!/usr/bin/env bash
# Publish the dashboard on a public URL through a Cloudflare tunnel.
#
#   ./scripts/analysis/tunnel.sh setup     # one-time: create the named tunnel + DNS
#   ./scripts/analysis/tunnel.sh up        # build + serve + tunnel; prints the URL
#   ./scripts/analysis/tunnel.sh refresh   # rebuild in place -- same URL, new data
#   ./scripts/analysis/tunnel.sh url       # the current public URL
#   ./scripts/analysis/tunnel.sh status
#   ./scripts/analysis/tunnel.sh down
#
# Everything runs in two detached screens, so both outlive the SSH session that
# started them (this box has no reachable user systemd, so no --user service):
#
#   syco-dashboard   serve_dashboard.py --all, bound to 127.0.0.1:$PORT
#   syco-tunnel      cloudflared, joining that port to the public hostname
#
# The server stays on 127.0.0.1 rather than 0.0.0.0: cloudflared reaches it over
# loopback, so binding wider would only expose the port to the lab network too.
#
# Two tunnel modes. A *named* tunnel (DASH_TUNNEL_NAME + DASH_TUNNEL_HOSTNAME, set up
# once by `setup`) keeps one hostname forever and holds 4 redundant edge
# connections. Without one it falls back to a quick tunnel: no setup, but a
# random *.trycloudflare.com name that changes on every reconnect, over a single
# connection. `up` picks whichever is available.
#
# `refresh` is the sync step. The pages are plain files under experiments/, so
# rebuilding rewrites them underneath the running server; the next request gets
# the new bytes (responses carry Cache-Control: no-store, so browsers do not
# hold a stale copy). Nothing restarts and the public URL does not change.
#
# DASH_AUTH=user:pass  puts HTTP basic auth in front of the server. The pages
# carry every model output in the run -- set this if the link leaves people you
# trust. Cloudflare Access is the stronger option once a named tunnel is up.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PORT="${PORT:-8000}"
DASH_AUTH="${DASH_AUTH:-}"
# DASH_-prefixed on purpose: cloudflared reads TUNNEL_NAME and TUNNEL_HOSTNAME
# from the environment as its own --name/--hostname flags, so naming these the
# obvious thing silently turns a quick tunnel into a named one and it dies
# looking for a cert.pem that is not there.
DASH_TUNNEL_NAME="${DASH_TUNNEL_NAME:-power-syco-dashboard}"
DASH_TUNNEL_HOSTNAME="${DASH_TUNNEL_HOSTNAME:-}"   # e.g. syco.yourdomain.org
unset TUNNEL_NAME TUNNEL_HOSTNAME TUNNEL_URL       # ditto, if the caller has them set
STATE="${STATE:-$ROOT/.tunnel}"
SERVER_SESSION=syco-dashboard
TUNNEL_SESSION=syco-tunnel
SERVER_LOG="$STATE/server.log"
TUNNEL_LOG="$STATE/cloudflared.log"
HOSTNAME_FILE="$STATE/hostname"
mkdir -p "$STATE"

# The hostname `setup` routed, so later `up`/`url` calls need no env vars.
[[ -z "$DASH_TUNNEL_HOSTNAME" && -f "$HOSTNAME_FILE" ]] && DASH_TUNNEL_HOSTNAME="$(<"$HOSTNAME_FILE")"

have_session() { screen -ls 2>/dev/null | grep -q "[.]$1[[:space:]]"; }
port_listening() { (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; }
logged_in() { [[ -f "$HOME/.cloudflared/cert.pem" ]]; }
tunnel_exists() { logged_in && cloudflared tunnel list 2>/dev/null | grep -q "[[:space:]]$DASH_TUNNEL_NAME[[:space:]]"; }
named_ready() { [[ -n "$DASH_TUNNEL_HOSTNAME" ]] && tunnel_exists; }

metrics_port() {
  # 20241 is taken by the host's own root cloudflared, so ours lands elsewhere.
  # Read the port back from our log rather than guessing, or `status` reports
  # the host service's health as if it were this tunnel's.
  grep -oh 'Starting metrics server on 127\.0\.0\.1:[0-9]*' "$TUNNEL_LOG" 2>/dev/null \
    | tail -1 | grep -o '[0-9]*$'
}

public_url() {
  if named_ready; then
    echo "https://$DASH_TUNNEL_HOSTNAME"
  else
    grep -oh 'https://[a-z0-9-]*\.trycloudflare\.com' "$TUNNEL_LOG" 2>/dev/null | tail -1
  fi
}

# --- the tunnel process, supervised -------------------------------------
# Runs inside the screen. cloudflared already reconnects dropped edge
# connections by itself without exiting, so this loop is a last resort for the
# process dying outright, not a refresh cycle. The backoff matters: a tunnel
# that dies instantly is misconfigured, and retrying that every 5s just hammers
# Cloudflare's API until it rate-limits the account.
supervise() {
  local delay=5 start rc
  while true; do
    start=$SECONDS
    # cloudflared exiting non-zero is the normal case here, and under
    # `set -e -o pipefail` that status would kill this supervisor on the first
    # drop -- i.e. the restart loop would never once restart. Disarm around it.
    set +e
    if named_ready; then
      cloudflared tunnel --no-autoupdate run --url "http://127.0.0.1:$PORT" "$DASH_TUNNEL_NAME" 2>&1 | tee -a "$TUNNEL_LOG"
    else
      cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" 2>&1 | tee -a "$TUNNEL_LOG"
    fi
    rc=${PIPESTATUS[0]}
    set -e
    if (( SECONDS - start > 60 )); then
      delay=5                       # ran a while, so this was a transient drop
    else
      delay=$(( delay * 2 )); (( delay > 600 )) && delay=600
    fi
    echo "[tunnel] cloudflared exited rc=$rc after $((SECONDS-start))s $(date -Is); retry in ${delay}s" \
      | tee -a "$TUNNEL_LOG"
    sleep "$delay"
  done
}

start_server() {
  if have_session "$SERVER_SESSION"; then
    # An already-running server was started with whatever flags it was started
    # with; asking for auth now has to restart it or the password is a no-op.
    if [[ -n "$DASH_AUTH" ]] && ! pgrep -af 'serve_dashboard\.py' | grep -q -- '--auth'; then
      echo "[server] running without auth but DASH_AUTH is set — restarting it"
      screen -S "$SERVER_SESSION" -X quit || true
      sleep 1
    else
      echo "[server] $SERVER_SESSION already up on 127.0.0.1:$PORT"
      return
    fi
  fi
  local auth_arg=()
  [[ -n "$DASH_AUTH" ]] && auth_arg=(--auth "$DASH_AUTH")
  echo "[server] starting on 127.0.0.1:$PORT -> $SERVER_LOG"
  screen -dmS "$SERVER_SESSION" bash -c "
    cd '$ROOT'
    uv run python scripts/analysis/serve_dashboard.py --all \
      --port $PORT --host 127.0.0.1 ${auth_arg[*]} 2>&1 | tee '$SERVER_LOG'
    echo '--- server exited; Ctrl-D to close ---'
    exec bash
  "
  # --all rebuilds any stale page before it binds, and those run to tens of MB,
  # so the first bind can be minutes away on a fresh run, not seconds.
  echo -n "[server] waiting for the port"
  for _ in $(seq 1 300); do
    port_listening && break
    echo -n "."; sleep 2
  done
  echo
  port_listening || echo "[server] not listening yet — see $SERVER_LOG"
}

start_tunnel() {
  if have_session "$TUNNEL_SESSION"; then
    echo "[tunnel] $TUNNEL_SESSION already up"
    return
  fi
  if named_ready; then
    echo "[tunnel] named: $DASH_TUNNEL_NAME -> https://$DASH_TUNNEL_HOSTNAME"
  else
    echo "[tunnel] quick tunnel (run 'tunnel.sh setup' for a stable hostname)"
  fi
  # Start this session's log clean. A quick tunnel's hostname is read back out
  # of it, and a previous run's line sitting there is indistinguishable from
  # this one's -- `up` would report the dead URL of the tunnel it just replaced.
  [[ -s "$TUNNEL_LOG" ]] && mv -f "$TUNNEL_LOG" "$TUNNEL_LOG.prev"
  : > "$TUNNEL_LOG"
  # The loop lives in this script rather than inline in `screen -dmS bash -c`,
  # so the backoff arithmetic is not fighting two levels of quoting.
  screen -dmS "$TUNNEL_SESSION" bash -c \
    "cd '$ROOT' && DASH_TUNNEL_NAME='$DASH_TUNNEL_NAME' DASH_TUNNEL_HOSTNAME='$DASH_TUNNEL_HOSTNAME' PORT=$PORT \
     ./scripts/analysis/tunnel.sh _supervise"
  echo -n "[tunnel] waiting for the URL"
  for _ in $(seq 1 40); do
    [[ -n "$(public_url)" ]] && break
    echo -n "."; sleep 1
  done
  echo
}

banner() {
  local url; url="$(public_url)"
  echo
  echo "================================================================"
  if [[ -n "$url" ]]; then
    echo "  Public:  $url"
    echo "           $url/exp2/dashboard/dashboard.html"
  else
    echo "  Public:  not up yet — check $TUNNEL_LOG"
  fi
  echo "  Local:   http://localhost:$PORT/"
  if named_ready; then
    echo "  Tunnel:  named '$DASH_TUNNEL_NAME' — hostname is stable across restarts"
  else
    echo "  Tunnel:  quick — hostname changes on every reconnect"
  fi
  if [[ -n "$DASH_AUTH" ]]; then
    echo "  Auth:    basic, user '${DASH_AUTH%%:*}'"
  else
    echo "  Auth:    none — anyone with the link reads every model output"
  fi
  echo "  Update:  ./scripts/analysis/tunnel.sh refresh   (same URL)"
  echo "================================================================"
  echo
}

case "${1:-up}" in
  _supervise) supervise ;;

  setup)
    # One-time. `cloudflared tunnel login` is browser OAuth and cannot be done
    # from here, so it is checked rather than run.
    if ! logged_in; then
      cat <<MSG
No ~/.cloudflared/cert.pem — log in first, on a machine with a browser:

    cloudflared tunnel login

It opens a Cloudflare page, asks you to pick a domain you own there, and writes
cert.pem. Copy that file to ~/.cloudflared/ on this box if you logged in
elsewhere. Then re-run this with the hostname you want:

    DASH_TUNNEL_HOSTNAME=syco.yourdomain.org ./scripts/analysis/tunnel.sh setup
MSG
      exit 1
    fi
    if [[ -z "$DASH_TUNNEL_HOSTNAME" ]]; then
      echo "Set the hostname you want to serve on, under a domain in your Cloudflare account:"
      echo "    DASH_TUNNEL_HOSTNAME=syco.yourdomain.org ./scripts/analysis/tunnel.sh setup"
      exit 1
    fi
    tunnel_exists || { echo "[setup] creating tunnel '$DASH_TUNNEL_NAME'"; cloudflared tunnel create "$DASH_TUNNEL_NAME"; }
    echo "[setup] routing $DASH_TUNNEL_HOSTNAME -> $DASH_TUNNEL_NAME"
    cloudflared tunnel route dns "$DASH_TUNNEL_NAME" "$DASH_TUNNEL_HOSTNAME"
    echo "$DASH_TUNNEL_HOSTNAME" > "$HOSTNAME_FILE"
    echo "[setup] done — 'tunnel.sh down && tunnel.sh up' to switch over"
    ;;

  up)
    start_server
    start_tunnel
    banner
    ;;

  refresh)
    # Rebuild only. The running server serves these files off disk, so this is
    # the whole sync: no restart, no new URL. Note it rebuilds a page only when
    # a *source* is newer, so hand-editing a built .html will not trigger one.
    uv run python scripts/analysis/serve_dashboard.py --all --build-only
    have_session "$SERVER_SESSION" || echo "[warn] no server running — 'up' first"
    banner
    ;;

  url) public_url ;;

  status)
    screen -ls | grep -E "syco-(dashboard|tunnel)" || echo "no dashboard screens"
    port_listening && echo "port $PORT: listening" || echo "port $PORT: closed"
    mport="$(metrics_port)"
    echo -n "cloudflared edge connections: "
    if [[ -n "$mport" ]]; then
      curl -s --max-time 3 "http://127.0.0.1:$mport/ready" || echo -n "(metrics not answering)"
    else
      echo -n "(metrics port unknown)"
    fi
    echo
    # grep -c prints 0 and *exits 1* on no match, so the fallback has to be on
    # the assignment, not chained onto the echo -- else it prints 0 twice.
    n="$(grep -c "cloudflared exited" "$TUNNEL_LOG" 2>/dev/null || true)"
    echo "restarts since launch: ${n:-0}"
    banner
    ;;

  down)
    for s in "$TUNNEL_SESSION" "$SERVER_SESSION"; do
      have_session "$s" && { screen -S "$s" -X quit; echo "[$s] stopped"; } \
                        || echo "[$s] not running"
    done
    ;;

  *) sed -n '2,32p' "$0"; exit 2 ;;
esac
