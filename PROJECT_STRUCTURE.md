# Scalar Project Structure

## Overview

This is a monorepo containing all Scalar services and shared libraries.

## Directory Structure

```
scalar/
├── README.md                 # Main README
├── PROJECT_STRUCTURE.md      # This file
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── .gitignore               # Git ignore rules
├── Makefile                 # Development commands
├── docker-compose.yml       # Local development setup
├── railway.json             # Railway deployment config
│
├── libs/                    # Shared libraries
│   └── common/
│       ├── __init__.py
│       ├── types.py         # Pydantic models
│       ├── config.py         # Configuration loader
│       ├── db.py            # Database clients
│       └── logging.py        # Logging setup
│
├── services/                # Microservices
│   ├── api_gateway/        # REST API
│   │   ├── main.py
│   │   ├── routers/
│   │   │   ├── auth.py
│   │   │   ├── jobs.py
│   │   │   └── endpoints.py
│   │   └── Dockerfile
│   │
│   ├── scheduler/          # Job scheduler
│   │   ├── main.py
│   │   ├── scheduler.py
│   │   ├── bin_packer.py
│   │   ├── capacity_client.py
│   │   ├── runtime_client.py
│   │   └── Dockerfile
│   │
│   ├── capacity_manager/   # Capacity planning
│   │   ├── main.py
│   │   ├── api.py
│   │   ├── capacity_state.py
│   │   ├── autoscaler.py
│   │   ├── sfcompute_client.py
│   │   └── Dockerfile
│   │
│   ├── runtime_agent/      # Docker runtime
│   │   ├── main.py
│   │   └── Dockerfile
│   │
│   ├── billing/            # Billing worker
│   │   ├── main.py
│   │   ├── usage_collector.py
│   │   ├── rating_engine.py
│   │   ├── stripe_client.py
│   │   └── Dockerfile
│   │
│   └── forecasting/        # Forecasting worker
│       ├── main.py
│       ├── demand_model.py
│       ├── price_model.py
│       └── Dockerfile
│
├── migrations/              # Database migrations
│   └── 001_init.sql        # Initial schema
│
└── docs/                    # Documentation
    ├── architecture.md
    ├── api.md
    ├── scheduler.md
    ├── capacity_planning.md
    └── deployment.md
```

## Services

### API Gateway (`services/api_gateway`)
- FastAPI REST API
- Port: 8000
- Endpoints: `/jobs`, `/endpoints`, `/auth`

### Scheduler (`services/scheduler`)
- Background worker
- Consumes job queue
- Runs bin-packing algorithm

### Capacity Manager (`services/capacity_manager`)
- FastAPI service
- Port: 8001
- Manages capacity state
- Autoscales tiers

### Runtime Agent (`services/runtime_agent`)
- FastAPI service
- Port: 8002
- Manages Docker containers
- Requires Docker socket access

### Billing (`services/billing`)
- Background worker
- Collects usage events
- Generates invoices

### Forecasting (`services/forecasting`)
- Background worker
- Forecasts demand and prices

## Development

### Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

3. Run migrations:
   ```bash
   supabase db reset
   ```

4. Start services locally:
   ```bash
   docker-compose up -d
   ```

### Testing

```bash
make test
```

### Linting

```bash
make lint
```

## Deployment

See `docs/deployment.md` for detailed deployment instructions.

Each service can be deployed independently to Railway by:
1. Setting the service root directory
2. Configuring environment variables
3. Deploying

