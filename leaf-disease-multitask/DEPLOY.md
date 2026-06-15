# Deploying LeafLens

A small, single-host deployment guide for the multi-task leaf analyzer. The
production manifest is [`docker-compose.prod.yml`](docker-compose.prod.yml):
nginx serves the SPA and reverse-proxies `/api` to an **internal-only** backend,
both containers restart automatically and expose healthchecks, and the dataset
is mounted read-only.

> Built to the same pattern verified in dev. The Docker image build and the live
> compose flow were **not run in the dev environment** (no Docker there) — treat
> the commands below as the intended, untested-here deploy path.

## 1. Prerequisites

- Docker + Docker Compose v2 on the host.
- The dataset available next to this repo at `../LeafDiseaseDetection-DL`
  (mounted read-only). Override the two `LEAFLENS_*_DIR` env vars in the manifest
  if it lives elsewhere.
- ~2 vCPU / 2 GB RAM minimum to serve; more to train.

## 2. Train once, then serve

The API answers `/api/health` immediately, reporting `model_loaded: false` until
a model exists — so you can start the stack first and train after, or train
first. Recommended order:

```bash
# Build + train a first model version into the shared `artifacts` volume.
docker compose -f docker-compose.prod.yml --profile train run --rm trainer

# Bring up the public stack (frontend :80 -> internal backend :8000).
docker compose -f docker-compose.prod.yml up -d --build
```

Visit `http://<host>/` for the UI; the API is at `http://<host>/api/...`.

## 3. No-downtime model updates

Model artifacts are **versioned** (`artifacts/versions/<id>/`) with an
`active.txt` pointer (see [`backend/app/registry.py`](backend/app/registry.py)).
To ship a new model without dropping traffic:

```bash
# 1. Train a new version (becomes active on disk).
docker compose -f docker-compose.prod.yml --profile train run --rm trainer

# 2. Tell the running server to reload the active version — no restart.
curl -X POST http://<host>/api/models/reload
```

Or pin/roll back to a specific version:

```bash
curl -s http://<host>/api/models                       # list versions
curl -X POST http://<host>/api/models/<version>/activate
```

`activate` updates the pointer **and** hot-reloads the served model in one call.
Because each version is its own directory, rollback is just re-activating the
previous id.

## 4. Observability

- `GET /api/health` — liveness, active model version, cache stats.
- `GET /api/metrics` — request volume, latency p50/p95/p99/max, status-code
  buckets, and the distribution of predicted classes (in-process, no external
  backend). Scrape it on an interval or eyeball it during load.
- Each request is logged (`method path -> status (ms)`) to stdout — pick it up
  with `docker compose logs -f backend` or your log shipper.

## 5. TLS

The frontend listens on plain `:80`. Terminate TLS in front of it — e.g. a
reverse proxy (Caddy/Traefik/nginx) or a cloud load balancer handling certs and
forwarding to the container's `:80`. Don't expose the backend directly; it has
no auth and is meant to sit behind the nginx `/api` proxy.

## 6. Scaling caveats

- **State:** the only state is the `artifacts` volume (models) and the
  **in-process** metrics + prediction cache. Running multiple backend replicas
  would give each its own cache and metrics — fine for throughput, but
  `/api/metrics` then reflects a single replica, and a `reload` must be sent to
  each. For a single-host demo, one replica is the intended setup.
- **CPU-bound inference:** TensorFlow-CPU inference is the bottleneck; the cache
  absorbs repeat images. Scale vertically (more CPU, raise the `cpus` limit)
  before horizontally.
- **Resource limits:** tune the `deploy.resources.limits` in the manifest to the
  host. Training is much heavier than serving — run the `trainer` profile when
  the host is otherwise idle.

## 7. Health & teardown

```bash
docker compose -f docker-compose.prod.yml ps          # health status
docker compose -f docker-compose.prod.yml logs -f     # tail logs
docker compose -f docker-compose.prod.yml down        # stop (keeps the volume)
docker compose -f docker-compose.prod.yml down -v      # stop + delete artifacts
```
