#!/usr/bin/env bash
# 01_upload_to_minio.sh — Upload the bundle to MinIO.
# Uses mc (MinIO client). Must be configured first.
#
# Prerequisites:
#   1. Install mc: https://min.io/docs/minio/linux/reference/minio-mc.html
#   2. export env vars from .env:
#        set -a && source .env && set +a
#
# Usage:
#   set -a && source .env && set +a && bash scripts/01_upload_to_minio.sh

set -euo pipefail

cd "$(dirname "$0")/.."

: "${MINIO_ENDPOINT:?MINIO_ENDPOINT must be set in .env}"
: "${MINIO_ACCESS_KEY:?MINIO_ACCESS_KEY must be set}"
: "${MINIO_SECRET_KEY:?MINIO_SECRET_KEY must be set}"
: "${MINIO_BUCKET:=hw03-bundles}"
: "${STUDENT_USERNAME:?STUDENT_USERNAME must be set}"
: "${MINIO_PREFIX:=${STUDENT_USERNAME}/}"

if [ ! -d "bundle" ]; then
    echo "ERROR: bundle/ directory not found."
    exit 1
fi

if ! command -v mc >/dev/null 2>&1; then
    echo "ERROR: mc command not found. Install MinIO client first."
    exit 1
fi

# Normalize endpoint.
# If MINIO_ENDPOINT already includes http:// or https://, use it as-is.
# Otherwise default to http://.
if [[ "${MINIO_ENDPOINT}" == http://* || "${MINIO_ENDPOINT}" == https://* ]]; then
    MINIO_URL="${MINIO_ENDPOINT}"
else
    MINIO_URL="http://${MINIO_ENDPOINT}"
fi

# Configure mc alias.
echo "Configuring MinIO alias 'qbc12'..."
if ! mc alias set qbc12 "${MINIO_URL}" "${MINIO_ACCESS_KEY}" "${MINIO_SECRET_KEY}" >/dev/null 2>&1; then
    echo "ERROR: Failed to configure mc alias. Check endpoint and credentials."
    echo "Endpoint used: ${MINIO_URL}"
    exit 1
fi

# Create bucket if missing.
mc mb "qbc12/${MINIO_BUCKET}" 2>/dev/null || true
echo "Bucket: ${MINIO_BUCKET}"

# Create a temporary clean bundle copy because this mc version does not support --exclude.
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

CLEAN_BUNDLE="${TMP_DIR}/bundle"
mkdir -p "${CLEAN_BUNDLE}"

echo "Preparing clean bundle copy..."
cp -a bundle/. "${CLEAN_BUNDLE}/"

# Remove files/folders that should not be uploaded.
find "${CLEAN_BUNDLE}" -name ".git" -type d -prune -exec rm -rf {} +
find "${CLEAN_BUNDLE}" -name ".ipynb_checkpoints" -type d -prune -exec rm -rf {} +
find "${CLEAN_BUNDLE}" -name "__pycache__" -type d -prune -exec rm -rf {} +
find "${CLEAN_BUNDLE}" -name "*.pyc" -type f -delete
find "${CLEAN_BUNDLE}" -name "*.pyo" -type f -delete
find "${CLEAN_BUNDLE}" -name "*.commit" -type f -delete
find "${CLEAN_BUNDLE}" -name ".commit" -type f -delete

# Upload the clean bundle.
echo "Uploading bundle/ -> s3://${MINIO_BUCKET}/${MINIO_PREFIX}"
mc cp --recursive \
    "${CLEAN_BUNDLE}/" \
    "qbc12/${MINIO_BUCKET}/${MINIO_PREFIX}"

echo ""
echo "=== Verification ==="
echo "Files on MinIO:"
mc ls --recursive "qbc12/${MINIO_BUCKET}/${MINIO_PREFIX}" | head -30

echo ""
echo "Done. Your bundle is at: s3://${MINIO_BUCKET}/${MINIO_PREFIX}"