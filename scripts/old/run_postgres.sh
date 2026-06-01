#!/usr/bin/env bash
set -euo pipefail

PGDATA_DIR="${PGDATA_DIR:-$PWD/optuna_pgdata}"
PGPORT="${PGPORT:-5432}"
PGHOST="${PGHOST:-127.0.0.1}"
POSTGRES_BIN_DIR="${POSTGRES_BIN_DIR:-}"
READY_FILE="${READY_FILE:-$PGDATA_DIR/connection.txt}"

if [[ -n "${POSTGRES_BIN_DIR}" ]]; then
  export PATH="${POSTGRES_BIN_DIR}:$PATH"
fi

mkdir -p "${PGDATA_DIR}"

if [[ ! -f "${PGDATA_DIR}/PG_VERSION" ]]; then
  initdb -D "${PGDATA_DIR}"
fi

cat > "${PGDATA_DIR}/postgresql.conf" <<EOF
listen_addresses = '${PGHOST}'
port = ${PGPORT}
unix_socket_directories = '${PGDATA_DIR}'
EOF

postgres -D "${PGDATA_DIR}" &
POSTGRES_PID=$!

cleanup() {
  if kill -0 "${POSTGRES_PID}" >/dev/null 2>&1; then
    kill "${POSTGRES_PID}"
    wait "${POSTGRES_PID}" || true
  fi
}
trap cleanup EXIT INT TERM

until pg_isready -h "${PGHOST}" -p "${PGPORT}" >/dev/null 2>&1; do
  sleep 1
done

cat > "${READY_FILE}" <<EOF
host=${PGHOST}
port=${PGPORT}
pgdata=${PGDATA_DIR}
EOF

wait "${POSTGRES_PID}"
