# Scalar Architecture

## Overview

Scalar is a serverless compute platform built on top of SFCompute that dynamically buys and sells compute on the marketplace to optimize GPU unit economics.

## System Architecture

Scalar operates in three planes:

1. **Control Plane**: API, scheduler, capacity manager, billing, forecasting
2. **Data Plane**: SFCompute-backed clusters where jobs actually run
3. **Market Plane**: Integration with SFCompute's order book & cluster APIs

## Components

### API Gateway

- **Purpose**: REST API for job submission and management
- **Technology**: FastAPI
- **Endpoints**:
  - `POST /jobs` - Submit job
  - `GET /jobs/{id}` - Get job status
  - `POST /endpoints` - Create serverless endpoint
- **Auth**: Supabase API keys

### Scheduler

- **Purpose**: Job orchestrator with bin-packing algorithm
- **Technology**: Python async worker
- **Key Features**:
  - Consumes job queue
  - Bin-packs jobs onto nodes
  - Communicates with Capacity Manager and Runtime Agents

### Capacity Manager

- **Purpose**: Capacity planning and autoscaling
- **Technology**: FastAPI with background worker
- **Key Features**:
  - Maintains capacity state
  - Autoscales Fast and Flex tiers
  - Integrates with SFCompute marketplace

### Runtime Agent

- **Purpose**: Docker-based runtime for job execution
- **Technology**: FastAPI + Docker SDK
- **Key Features**:
  - Receives placement requests
  - Creates Docker containers
  - Streams logs and metrics

### Billing

- **Purpose**: Usage tracking and invoicing
- **Technology**: Python worker
- **Key Features**:
  - Collects usage events
  - Generates invoices
  - Integrates with Stripe

### Forecasting

- **Purpose**: Demand and price forecasting
- **Technology**: Python worker
- **Key Features**:
  - EWMA demand forecasting
  - Price forecasting
  - Feeds autoscaler

## Data Model

See `migrations/001_init.sql` for complete schema.

### Key Tables

- `jobs`: Job definitions and status
- `blocks`: SFCompute contracts
- `nodes`: Compute instances
- `gpu_assignments`: GPU slot allocations
- `usage_events`: Billing events
- `invoices`: Customer invoices

## Bin-Packing Algorithm

The scheduler uses a multi-dimensional bin-packing algorithm to maximize GPU utilization:

1. Filter candidate nodes by tier, expiry, CPU/RAM
2. Find GPUs with sufficient VRAM
3. Score nodes using heuristics:
   - Utilization after job
   - Fragmentation penalty
   - Time to expiry penalty
4. Select best node and allocate

See `services/scheduler/bin_packer.py` for implementation.

## Autoscaling

### Fast Tier

- Goal: Keep warm capacity for low latency
- Strategy: Pre-buy blocks based on demand forecast
- Safety factor: 1.5x expected load

### Flex Tier

- Goal: Minimize cost, accept delays
- Strategy: Buy blocks only when queue exists and price is good
- Backfill: Use idle Fast tier capacity

## Deployment

Services are deployed on Railway with Docker containers. Each service has its own Dockerfile and can be scaled independently.

See `docs/deployment.md` for detailed deployment instructions.

