# Deployment Guide

## Prerequisites

- Railway account
- Supabase project
- SFCompute API credentials

## Environment Variables

See `.env.example` for required environment variables.

Set these in Railway for each service:

### API Gateway
- `SUPABASE_URL`
- `SUPABASE_SERVICE_KEY`
- `REDIS_URL` (optional)

### Scheduler
- `SUPABASE_URL`
- `SUPABASE_SERVICE_KEY`
- `CAPACITY_MANAGER_URL`
- `RUNTIME_AGENT_URL`
- `REDIS_URL` (optional)

### Capacity Manager
- `SUPABASE_URL`
- `SUPABASE_SERVICE_KEY`
- `SF_API_KEY`
- `SF_API_URL`

### Runtime Agent
- Docker socket access (requires privileged mode)

### Billing
- `SUPABASE_URL`
- `SUPABASE_SERVICE_KEY`
- `STRIPE_API_KEY` (optional)

### Forecasting
- `SUPABASE_URL`
- `SUPABASE_SERVICE_KEY`
- `SF_API_KEY`
- `SF_API_URL`

## Database Setup

1. Create Supabase project
2. Run migrations:
   ```bash
   supabase db reset
   ```
   Or apply `migrations/001_init.sql` manually

## Railway Deployment

### Option 1: Individual Services

Deploy each service separately:

1. Create new Railway project
2. Add service from GitHub repo
3. Set root directory to service directory (e.g., `services/api_gateway`)
4. Set environment variables
5. Deploy

### Option 2: Monorepo

Use Railway's monorepo support:

1. Create Railway project
2. Add services as separate services
3. Configure each service's root directory
4. Set environment variables per service

## Docker Compose (Local)

For local development:

```bash
docker-compose up -d
```

## Health Checks

Each service exposes `/health` endpoint:

- API Gateway: `http://api-gateway:8000/health`
- Capacity Manager: `http://capacity-manager:8001/health`
- Runtime Agent: `http://runtime-agent:8002/health`

## Monitoring

- Logs: Railway provides log streaming
- Metrics: Prometheus-compatible endpoints (TODO)
- Tracing: OpenTelemetry (if `OTLP_ENDPOINT` configured)

## Scaling

- API Gateway: Scale horizontally
- Scheduler: Single instance (or partitioned by tier)
- Capacity Manager: Single instance
- Runtime Agent: One per SFCompute cluster/node
- Billing: Single instance
- Forecasting: Single instance

## Rollback

Railway automatically keeps previous deployments. Rollback via Railway dashboard.

