# Deployment

How to run CineMatch on a single host: a containerized FastAPI backend serving
the versioned NeuMF model, behind an nginx-served static frontend that proxies
`/api` to the backend.

> **Status:** these manifests follow the same pattern as the verified dev setup,
> but the production build/run has **not** been executed in this repo's
> environment (no Docker available here). Treat the steps below as a tested-shape
> reference and validate on your host. The offline Python test suite (85 tests)
> and the model artifact are verified.

## Architecture

```
            Internet
               │  :80 (add TLS in front)
               ▼
      ┌──────────────────┐        internal network
      │ frontend (nginx) │ ──/api──▶ ┌──────────────────┐
      │  serves the SPA  │           │ backend (FastAPI)│
      └──────────────────┘           │  :8000 (private) │
                                      └────────┬─────────┘
                                               │ loads active version
                                      ┌────────▼─────────┐
                                      │ artifacts volume │  versions/ + active.txt
                                      └──────────────────┘
```

- Only the **frontend** is published (port 80). The backend is reachable solely
  on the internal Docker network via the nginx proxy — there is no public 8000.
- The trained model lives in the **`artifacts` named volume**. On first boot the
  backend's entrypoint seeds it from the model baked into the image (the
  committed version under `backend/artifacts/versions/`), so a fresh deploy
  serves a real model immediately.

## Prerequisites

- A Linux host with Docker Engine + the Compose plugin (`docker compose`).
- Ports: 80 (and 443 if you terminate TLS here).

## Deploy

```bash
git clone <repo> && cd pytorch-recommender
docker compose -f docker-compose.prod.yml up -d --build
```

Then:

```bash
curl -fsS http://localhost/api/health        # {"status":"ok","model_loaded":true,"version":"..."}
./scripts/smoke_test.sh http://localhost      # health → movies → recommend → user → similar
```

Open `http://<host>/` for the UI.

## Shipping / updating the model

The model is a versioned artifact in the `artifacts` volume. Two ways to get one
there:

1. **Baked-in (default):** the committed version under `backend/artifacts/` is
   copied into the image and seeded into the volume on first boot. Nothing to do.
2. **Train on the host:** run a one-off training job that writes a new version
   into the same volume, then activate it **without downtime** via the registry:

   ```bash
   # Train a new version into the artifacts volume. `run` reuses the backend
   # service's volume mounts, so the new version lands in the shared volume.
   docker compose -f docker-compose.prod.yml run --rm backend python -m app.train

   # List versions, then hot-swap the served one (no restart):
   curl http://localhost/api/models
   curl -X POST http://localhost/api/models/<new-version>/activate
   ```

   (Or `POST /api/models/reload` to reload the active version after replacing it.)

Rollback is the same call pointed at an older version id — every version stays
in the volume until you prune it.

## Configuration

All settings are environment variables (see `backend/app/config.py`). Set them in
the compose `environment:` block or an `.env` file next to the compose file:

| Var | Default | Purpose |
|---|---|---|
| `USE_FAISS` | `false` | Use the FAISS ANN backend (install `requirements-faiss.txt` in the image first) |
| `GENRE_WEIGHT` | `0.2` | Cold-start genre-prior blend weight |
| `CACHE_SIZE` | `512` | Max entries per recommendation LRU cache |
| `TOP_K` | `10` | Default recommendation size |

## TLS / public exposure

The frontend speaks plain HTTP on :80. For a public deployment, terminate TLS in
front of it with a reverse proxy that auto-manages certificates — e.g. **Caddy**
or **Traefik** — or your platform's load balancer. Point it at the frontend
container and forward `:443 → frontend:80`. Don't expose the backend directly.

## Health & monitoring

- **Liveness/readiness:** `GET /api/health` (also the container healthcheck) —
  reports `model_loaded` and the active version.
- **Metrics:** `GET /api/metrics` — request volume, latency p50/p95/max, status
  buckets, and recommendation coverage. Scrape it or wire it into a dashboard.
- **Logs:** request logging is on stdout (`docker compose logs -f backend`).
- **Restart:** `restart: unless-stopped` recovers both services across crashes
  and host reboots.

## Scaling & limitations

- The backend is **stateless with respect to the model** (loaded from the shared
  artifacts volume), so it scales horizontally for read traffic — **except** for
  three pieces of per-process state: the in-process **metrics collector**, the
  **LRU caches**, and the **active-version slot**. Consequences:
  - Run the backend with a **single uvicorn worker** per container (the default
    here). Multiple workers/replicas each keep their own metrics, caches, and
    active version — so `/api/metrics` is per-process and a `activate`/`reload`
    call only affects the worker that served it.
  - To scale out: put replicas behind a load balancer and either (a) treat
    metrics as per-replica and aggregate downstream, and roll model activations
    by restarting replicas (all re-read `active.txt` from the shared volume), or
    (b) externalize metrics (Prometheus client) and the active pointer if you
    need cluster-wide live switching. This is intentionally out of scope for the
    single-host target.
- CPU inference is fine for MovieLens-scale catalogs; for very large catalogs,
  enable FAISS and raise the backend resource limits.

## What is NOT covered here

- A managed/cloud deployment (ECS/Kubernetes/Fly/Render) — the image is standard
  and portable, but platform manifests aren't included.
- Authentication/rate limiting (the demo is unauthenticated by design).
- Pushing images to a registry — add `image:` tags and `docker compose push` if
  you deploy from a registry rather than building on the host.
