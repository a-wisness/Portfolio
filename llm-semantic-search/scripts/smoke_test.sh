#!/usr/bin/env bash
#
# Smoke test for a RUNNING stack (e.g. after `docker compose up`).
# Verifies the live HTTP surface: health -> ingest -> stats -> search.
#
# Usage:
#   ./scripts/smoke_test.sh [BASE_URL]
#
# BASE_URL defaults to the frontend origin (which proxies /api to the backend),
# so this also exercises the nginx reverse proxy. Pass http://localhost:8000 to
# hit the backend directly.
#
# The search step needs a valid ANTHROPIC_API_KEY in the backend; if it isn't
# set, that single step is reported as SKIPPED rather than failing the run.

set -u

BASE="${1:-http://localhost:8080}"
PASS=0
FAIL=0

green() { printf '\033[32m%s\033[0m\n' "$1"; }
red()   { printf '\033[31m%s\033[0m\n' "$1"; }
yellow(){ printf '\033[33m%s\033[0m\n' "$1"; }

ok()   { green "  PASS: $1"; PASS=$((PASS+1)); }
bad()  { red   "  FAIL: $1"; FAIL=$((FAIL+1)); }

echo "Smoke testing: $BASE"
echo

# 1. Health -------------------------------------------------------------------
echo "[1/4] GET /api/health"
if curl -fsS "$BASE/api/health" | grep -q '"status":"ok"'; then
  ok "backend is healthy"
else
  bad "health check did not return status ok"
  red "Is the stack running? Try: docker compose up --build"
  exit 1
fi

# 2. Ingest -------------------------------------------------------------------
echo "[2/4] POST /api/ingest"
TMP="$(mktemp --suffix=.txt)"
cat > "$TMP" <<'EOF'
Semantic Search Studio indexes documents and answers questions about them.
The refund policy allows returns within 30 days of purchase.
EOF
INGEST="$(curl -fsS -F "file=@${TMP};type=text/plain" "$BASE/api/ingest" || true)"
rm -f "$TMP"
if echo "$INGEST" | grep -q '"chunks_indexed"'; then
  ok "document ingested ($(echo "$INGEST" | grep -o '"chunks_indexed":[0-9]*'))"
else
  bad "ingest did not report chunks_indexed -> $INGEST"
fi

# 3. Stats --------------------------------------------------------------------
echo "[3/4] GET /api/stats"
STATS="$(curl -fsS "$BASE/api/stats" || true)"
if echo "$STATS" | grep -q '"total_chunks"'; then
  ok "stats reported -> $(echo "$STATS" | grep -o '"total_chunks":[0-9]*')"
else
  bad "stats did not return total_chunks -> $STATS"
fi

# 4. Search (needs ANTHROPIC_API_KEY) ----------------------------------------
echo "[4/4] POST /api/search"
CODE="$(curl -s -o /tmp/_search_body -w '%{http_code}' \
  -H 'Content-Type: application/json' \
  -d '{"query":"how long do I have to return something?"}' \
  "$BASE/api/search")"
BODY="$(cat /tmp/_search_body 2>/dev/null || true)"
rm -f /tmp/_search_body
if [ "$CODE" = "200" ] && echo "$BODY" | grep -q '"answer"'; then
  ok "search returned a grounded answer"
elif echo "$BODY" | grep -qiE 'api[_ ]key|authentication|anthropic'; then
  yellow "  SKIP: search reachable but ANTHROPIC_API_KEY missing/invalid (HTTP $CODE)"
else
  bad "search failed (HTTP $CODE) -> $BODY"
fi

# Summary ---------------------------------------------------------------------
echo
echo "------------------------------------------"
green "PASS: $PASS"
[ "$FAIL" -gt 0 ] && red "FAIL: $FAIL" || echo "FAIL: 0"
echo "------------------------------------------"
[ "$FAIL" -eq 0 ]
