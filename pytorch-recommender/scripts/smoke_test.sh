#!/usr/bin/env bash
#
# Smoke test for a RUNNING stack (e.g. after `docker compose up`).
# Verifies the live HTTP surface: health -> movies -> recommend -> user recs -> similar.
#
# Usage:
#   ./scripts/smoke_test.sh [BASE_URL]
#
# BASE_URL defaults to the frontend origin (which proxies /api to the backend),
# so this also exercises the nginx reverse proxy. Pass http://localhost:8000 to
# hit the backend directly.
#
# If no model is trained yet, the recommendation steps are reported SKIPPED
# (the API returns 503) rather than failing the run.

set -u

BASE="${1:-http://localhost:8080}"
PASS=0
FAIL=0

green() { printf '\033[32m%s\033[0m\n' "$1"; }
red()   { printf '\033[31m%s\033[0m\n' "$1"; }
yellow(){ printf '\033[33m%s\033[0m\n' "$1"; }
ok()    { green "  PASS: $1"; PASS=$((PASS+1)); }
bad()   { red   "  FAIL: $1"; FAIL=$((FAIL+1)); }

echo "Smoke testing: $BASE"
echo

# 1. Health -------------------------------------------------------------------
echo "[1/5] GET /api/health"
HEALTH="$(curl -fsS "$BASE/api/health" 2>/dev/null || true)"
if echo "$HEALTH" | grep -q '"status":"ok"'; then
  ok "backend healthy"
else
  bad "health check failed"
  red "Is the stack running? Try: docker compose up --build"
  exit 1
fi

MODEL_LOADED="false"
echo "$HEALTH" | grep -q '"model_loaded":true' && MODEL_LOADED="true"
if [ "$MODEL_LOADED" = "true" ]; then
  ok "model is loaded"
else
  yellow "  NOTE: model_loaded=false — recommendation steps will be SKIPPED"
  yellow "  Train with: docker compose run --rm backend python -m app.train && docker compose restart backend"
fi

# Helper: GET expecting JSON; echoes body, sets global CODE.
get() { CODE="$(curl -s -o /tmp/_rb -w '%{http_code}' "$BASE$1")"; cat /tmp/_rb; }

# 2. Movies -------------------------------------------------------------------
echo "[2/5] GET /api/movies"
BODY="$(get "/api/movies?limit=5")"
if [ "$MODEL_LOADED" = "true" ]; then
  echo "$BODY" | grep -q '"movie_id"' && ok "catalog returned movies" || bad "no movies -> $BODY"
else
  [ "$CODE" = "503" ] && yellow "  SKIP: /api/movies (503, no model)" || bad "expected 503, got $CODE"
fi

# 3. Recommend (cold-start) ---------------------------------------------------
echo "[3/5] POST /api/recommend"
if [ "$MODEL_LOADED" = "true" ]; then
  FIRST_ID="$(echo "$BODY" | grep -o '"movie_id":[0-9]*' | head -1 | grep -o '[0-9]*')"
  CODE="$(curl -s -o /tmp/_rb -w '%{http_code}' -H 'Content-Type: application/json' \
    -d "{\"liked_movie_ids\":[${FIRST_ID:-1}],\"top_k\":5}" "$BASE/api/recommend")"
  R="$(cat /tmp/_rb)"
  [ "$CODE" = "200" ] && echo "$R" | grep -q '"recommendations"' \
    && ok "cold-start recommendations returned" || bad "recommend failed ($CODE) -> $R"
else
  yellow "  SKIP: /api/recommend (no model)"
fi

# 4. User recommendations -----------------------------------------------------
echo "[4/5] GET /api/users/1/recommendations"
if [ "$MODEL_LOADED" = "true" ]; then
  BODY="$(get "/api/users/1/recommendations?top_k=5")"
  [ "$CODE" = "200" ] && echo "$BODY" | grep -q '"recommendations"' \
    && ok "user recommendations returned" || bad "user recs failed ($CODE) -> $BODY"
else
  yellow "  SKIP: /api/users/1/recommendations (no model)"
fi

# 5. Similar movies -----------------------------------------------------------
echo "[5/5] GET /api/movies/{id}/similar"
if [ "$MODEL_LOADED" = "true" ]; then
  BODY="$(get "/api/movies/${FIRST_ID:-1}/similar?top_k=5")"
  [ "$CODE" = "200" ] && echo "$BODY" | grep -q '"recommendations"' \
    && ok "similar movies returned" || bad "similar failed ($CODE) -> $BODY"
else
  yellow "  SKIP: /api/movies/{id}/similar (no model)"
fi

rm -f /tmp/_rb
echo
echo "------------------------------------------"
green "PASS: $PASS"
[ "$FAIL" -gt 0 ] && red "FAIL: $FAIL" || echo "FAIL: 0"
echo "------------------------------------------"
[ "$FAIL" -eq 0 ]
