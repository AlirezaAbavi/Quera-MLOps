#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://127.0.0.1:8000}"

echo "Using API_URL=${API_URL}"

echo ""
echo "=== / ==="
curl -fsS "${API_URL}/" | python -m json.tool

echo ""
echo "=== /health ==="
curl -fsS "${API_URL}/health" | python -m json.tool

echo ""
echo "=== /model-info ==="
curl -fsS "${API_URL}/model-info" | python -m json.tool

echo ""
echo "=== /embed ==="
EMBED_RESPONSE="$(
  curl -fsS -X POST "${API_URL}/embed" \
    -H "Content-Type: application/json" \
    -d '{"texts":["I love this so much","I am angry right now","This is neutral text"]}'
)"

printf '%s\n' "${EMBED_RESPONSE}" | python -c '
import json
import sys

data = json.load(sys.stdin)

summary = {
    "count": data["count"],
    "dim": data["dim"],
    "embedding_rows": len(data["embeddings"]),
    "first_embedding_len": len(data["embeddings"][0]) if data["embeddings"] else 0,
}

print(json.dumps(summary, indent=2))
'

echo ""
echo "=== /search ==="
set +e
SEARCH_RESPONSE="$(
  curl -fsS -X POST "${API_URL}/search" \
    -H "Content-Type: application/json" \
    -d '{"query":"I am very happy today","top_k":5,"lang":"en","exclude_neutral":true}' 2>&1
)"
SEARCH_CODE=$?
set -e

if [ "${SEARCH_CODE}" -eq 0 ]; then
  printf '%s\n' "${SEARCH_RESPONSE}" | python -m json.tool
else
  echo "Search request failed or backend has no indexed data yet."
  echo "${SEARCH_RESPONSE}"
fi

echo ""
echo "Smoke test completed."