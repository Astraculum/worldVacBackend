#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PG_BIN="${PG_BIN:-/usr/lib/postgresql/16/bin}"
PGDATA="${PGDATA:-$ROOT/.pgdata}"
PG_RUN="${PG_RUN:-$ROOT/.pgrun}"
PG_PORT="${PG_PORT:-5433}"
PG_USER="${PG_USER:-worldvac}"
PG_DB="${PG_DB:-worldvac}"
LOG_FILE="${PGDATA}/postgres.log"

usage() {
  echo "Usage: $0 {start|stop|status|restart}"
}

ensure_cluster() {
  mkdir -p "$PG_RUN"
  if [[ ! -f "$PGDATA/PG_VERSION" ]]; then
    mkdir -p "$PGDATA"
    "$PG_BIN/initdb" \
      -D "$PGDATA" \
      -U "$PG_USER" \
      --auth-local=trust \
      --auth-host=trust \
      --encoding=UTF8 \
      --locale=C.UTF-8
  fi
}

is_running() {
  "$PG_BIN/pg_ctl" -D "$PGDATA" status >/dev/null 2>&1
}

start_cluster() {
  ensure_cluster
  if is_running; then
    echo "PostgreSQL already running on port ${PG_PORT}"
    return
  fi
  "$PG_BIN/pg_ctl" \
    -D "$PGDATA" \
    -l "$LOG_FILE" \
    -o "-p ${PG_PORT} -h 127.0.0.1 -k ${PG_RUN}" \
    start
  for _ in $(seq 1 30); do
    if "$PG_BIN/pg_isready" -h 127.0.0.1 -p "$PG_PORT" -U "$PG_USER" >/dev/null 2>&1; then
      break
    fi
    sleep 0.2
  done
  if ! "$PG_BIN/psql" -h 127.0.0.1 -p "$PG_PORT" -U "$PG_USER" -d postgres -Atqc \
      "SELECT 1 FROM pg_database WHERE datname='${PG_DB}'" | grep -q 1; then
    "$PG_BIN/createdb" -h 127.0.0.1 -p "$PG_PORT" -U "$PG_USER" "$PG_DB"
  fi
  echo "PostgreSQL ready: postgresql://${PG_USER}@127.0.0.1:${PG_PORT}/${PG_DB}"
}

stop_cluster() {
  if is_running; then
    "$PG_BIN/pg_ctl" -D "$PGDATA" stop -m fast
  else
    echo "PostgreSQL is not running"
  fi
}

status_cluster() {
  "$PG_BIN/pg_ctl" -D "$PGDATA" status || true
  "$PG_BIN/pg_isready" -h 127.0.0.1 -p "$PG_PORT" -U "$PG_USER" || true
}

case "${1:-}" in
  start) start_cluster ;;
  stop) stop_cluster ;;
  restart) stop_cluster; start_cluster ;;
  status) status_cluster ;;
  *) usage; exit 1 ;;
esac
