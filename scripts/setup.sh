#!/bin/sh
set -eu
accept=false
case "${1-}" in
  '') ;;
  --accept-noncommercial) accept=true; shift ;;
  *) printf '%s\n' 'Usage: sh scripts/setup.sh [--accept-noncommercial]' >&2; exit 2 ;;
esac
[ "$#" -eq 0 ] || { printf '%s\n' 'Unexpected arguments' >&2; exit 2; }
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
umask 077
mkdir -p "$root/.secrets"
chmod 700 "$root/.secrets"
for name in worker_token mcp_token; do
  path="$root/.secrets/$name"
  if [ ! -e "$path" ]; then
    # noclobber preserves existing tokens, including concurrent setup runs.
    token=$(od -An -N32 -tx1 /dev/urandom | tr -d ' \n')
    [ "${#token}" -eq 64 ] || { printf '%s\n' 'Random token generation failed' >&2; exit 1; }
    (set -C; printf '%s\n' "$token" > "$path")
  fi
  [ -s "$path" ] || { printf '%s\n' "Empty secret file: $name" >&2; exit 1; }
  # UID 10001 must read bind-mounted Compose secrets on Linux; private parent protects host access.
  chmod 644 "$path"
done
if [ ! -e "$root/.env" ]; then
  (set -C; cat "$root/.env.example" > "$root/.env")
fi
if [ "$accept" = true ]; then
  temp=$(mktemp "$root/.env.XXXXXX")
  trap 'rm -f "$temp"' EXIT HUP INT TERM
  awk 'BEGIN { found=0 } /^ACCEPT_NONCOMMERCIAL_MODELS=/ { print "ACCEPT_NONCOMMERCIAL_MODELS=true"; found=1; next } { print } END { if (!found) print "ACCEPT_NONCOMMERCIAL_MODELS=true" }' "$root/.env" > "$temp"
  mv "$temp" "$root/.env"
  trap - EXIT HUP INT TERM
fi
printf '%s\n' 'Setup complete. Secrets preserved, never printed. Model use requires explicit non-commercial acceptance.'
