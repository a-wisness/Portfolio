#!/usr/bin/env bash
# Smoke test for a running stack. Hits the live API and checks a real prediction.
#
#   ./scripts/smoke_test.sh [BASE_URL] [IMAGE_PATH]
#
# Defaults: BASE_URL=http://localhost:8080 (frontend nginx proxy),
#           IMAGE_PATH=a sample leaf from the original dataset.
set -euo pipefail

BASE_URL="${1:-http://localhost:8080}"
IMAGE_PATH="${2:-../LeafDiseaseDetection-DL/Classification Data/test/Apple rust leaf/2011-011.jpg}"

echo "== Health =="
curl -fsS "${BASE_URL}/api/health" | python3 -m json.tool

echo "== Classes =="
curl -fsS "${BASE_URL}/api/classes" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(len(d["classes"]), "classes")'

echo "== Predict =="
if [[ ! -f "${IMAGE_PATH}" ]]; then
  echo "Sample image not found at ${IMAGE_PATH}; pass one as the 2nd arg." >&2
  exit 1
fi
curl -fsS -X POST "${BASE_URL}/api/predict" -F "file=@${IMAGE_PATH}" \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print("predicted:", d["predicted_class"], f"({d[\"confidence\"]:.2%})"); print("leaf coverage:", f"{d[\"leaf_coverage\"]:.0%}"); print("mask bytes:", len(d["mask_png_base64"]))'

echo "OK"
