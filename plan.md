# Scalar MVP Implementation Plan

## Scope & Deliverables

- Deliver production-ready services under `services/` for API gateway, scheduler, capacity manager, runtime agent (Docker), billing, forecasting.
- Wire Supabase (Postgres + GoTrue auth) for persistence and auth; manage migrations in `migrations/`.
- Integrate real SFCompute contracts via adapter with buying/selling, node provisioning, monitoring.
- Package FastAPI apps for Railway auto-deploy (Procfile-style detection of `main.py`), plus CI/CD workflow.

## Architecture & Data Model

- Define shared Pydantic models in `libs/common/types.py`; implement unified config/secrets loader `libs/common/config.py` supporting Supabase + SFCompute credentials from env/Secrets Manager.
- Create `migrations/001_init.sql` with tables: `users`, `api_keys`, `jobs`, `blocks`, `nodes`, `gpu_assignments`, `job_runs`, `usage_events`, `invoices`.
- Implement Supabase triggers/views for auth mapping and job activity auditing.

## Service Implementations

- `services/api_gateway`: FastAPI app with routers (`jobs.py`, `endpoints.py`, `auth.py`) supporting job submission, status, endpoint CRUD; integrate Supabase auth and rate-limiting middleware; publish events to `job_queue` (Supabase channel or Redis-on-Railway).
- `services/scheduler`: Worker consuming queue, running bin-packer (`bin_packer.py`) using heuristics from `idea.md`; call `capacity_client` for reservations and `runtime_client` for placements; handle retries, job prioritization, dead-letter queue.
- `services/capacity_manager`: FastAPI/Async app exposing `/capacity_snapshot`, `/reserve`, `/release`; maintain in-memory state synced with Supabase tables; run autoscaler loop for Fast & Flex tiers interacting with SFCompute adapter.
- `services/runtime_agent`: Docker-based node agent using Python + Docker SDK; receive placement requests, pull images, start containers, stream logs/metrics back to API gateway; enforce resource cgroups.
- `services/billing`: Periodic worker aggregating `usage_events`, computing costs vs SFCompute spend, emitting invoices to Supabase tables, firing Stripe API stub (configurable for later live key).
- `services/forecasting`: Background job (Celery/Arq) reading historical demand & price data, writing forecasts consumed by autoscaler.

## SFCompute Integration

- Build `services/capacity_manager/sfcompute_client.py` with REST/CLI hybrid wrapper, handling auth, retries, rate limits; support contract lifecycle (quote, buy, sell, provision nodes, terminate).
- Add monitoring for order fill latency and block expiry; surface metrics to Forecasting & Billing services.

## Messaging & State

- Choose queue (Supabase Realtime or Redis via Railway) for job orchestration; document schema and failure semantics.
- Implement idempotent message handling and persistence of scheduler decisions.

## Deployment & Infrastructure

- Provide Railway deployment configs (`railway.json`, service definitions) and Dockerfiles per service; ensure FastAPI apps can be started via `uvicorn main:app` without extra scaffolding.
- Add GitHub Actions workflow for lint/test/build; include Railway deploy step.
- Configure Supabase project via `supabase/config.toml`; document env vars required (`SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `SF_API_KEY`, etc.).

## Observability & Testing

- Introduce `libs/common/logging.py` for structured logs; integrate OpenTelemetry exporters (OTLP) for tracing.
- Set up Prometheus-compatible metrics endpoints for each service; add Railway log drain instructions.
- Write unit tests (pytest) for bin-packer, autoscaler, SFCompute adapter mocks; integration tests using docker-compose to spin services with Supabase test container.
- Define load test scripts (Locust) to validate Fast tier latency SLO.

## Security & Compliance

- Implement API key management, Supabase auth checks, RBAC for tiers; secure secret management via Railway variables.
- Add audit logging for capacity changes, SFCompute orders, and job lifecycle events.

## Documentation & Handoff

- Expand `docs/architecture.md`, `docs/api.md`, `docs/scheduler.md`, `docs/capacity_planning.md` with final schemas, sequence diagrams, and operational runbooks.
- Provide onboarding guide covering local dev (Makefile), migrations, test suite, deployment steps, rollback procedures.