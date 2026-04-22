#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH_DIR="${SCRATCH:?SCRATCH must be set}"
PGROOT="${PGROOT:-${SCRATCH_DIR}/mandala_optuna_short/postgres}"
PGDATA_DIR="${PGDATA_DIR:-${PGROOT}/data}"
PGPORT="${PGPORT:-55432}"
URL_FILE="${URL_FILE:-${PGROOT}/storage_url.txt}"

mkdir -p "${PGROOT}" "${PGDATA_DIR}"

if [[ ! -f "${PGDATA_DIR}/PG_VERSION" ]]; then
  initdb -D "${PGDATA_DIR}" >/dev/null
fi

cat > "${PGDATA_DIR}/postgresql.conf" <<EOF
listen_addresses = '*'
port = ${PGPORT}
unix_socket_directories = '${PGDATA_DIR}'
EOF

cat > "${PGDATA_DIR}/pg_hba.conf" <<'EOF'
local   all             all                                     trust
host    all             all             127.0.0.1/32            trust
host    all             all             ::1/128                 trust
host    all             all             0.0.0.0/0               trust
host    all             all             ::/0                    trust
EOF

HOST_FQDN="$(hostname -f 2>/dev/null || hostname)"
STORAGE_URL="postgresql://${USER}@${HOST_FQDN}:${PGPORT}/postgres"
echo "${STORAGE_URL}" > "${URL_FILE}"

echo "Optuna PostgreSQL service starting"
echo "PGDATA_DIR=${PGDATA_DIR}"
echo "PGPORT=${PGPORT}"
echo "STORAGE_URL=${STORAGE_URL}"
echo "URL_FILE=${URL_FILE}"

postgres -D "${PGDATA_DIR}"
