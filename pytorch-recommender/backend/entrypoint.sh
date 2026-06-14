#!/bin/sh
# Seed the artifacts volume with the baked-in versioned model(s) on first run, so
# the API serves a trained model immediately. Retraining (python -m app.train)
# writes a new version into the volume and activates it.
set -e

if [ -z "$(ls -A /app/artifacts 2>/dev/null | grep -v '^\.gitkeep$')" ] && [ -d /app/seed_artifacts ]; then
  cp -a /app/seed_artifacts/. /app/artifacts/
  echo "Seeded artifacts volume from the baked-in model registry."
fi

exec "$@"
