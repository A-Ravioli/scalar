# Scalar

A serverless compute platform built on top of SFCompute that dynamically buys and sells compute on the marketplace to optimize GPU unit economics.

## Architecture

Scalar operates on a two-tier model:

- **Fast Tier**: Low-latency serverless compute with warm capacity
- **Flex Tier**: Cost-optimized compute with just-in-time capacity

See [docs/architecture.md](docs/architecture.md) for detailed architecture documentation.

## Services

- `api_gateway`: REST API for job submission and management
- `scheduler`: Job orchestrator with bin-packing algorithm
- `capacity_manager`: Capacity planning and autoscaling
- `runtime_agent`: Docker-based runtime for job execution
- `billing`: Usage tracking and invoicing
- `forecasting`: Demand and price forecasting

## Development

### Prerequisites

- Python 3.11+
- Supabase project
- SFCompute API credentials
- Railway account (for deployment)

### Environment Variables

See `.env.example` for required environment variables.

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run migrations
supabase db reset

# Start services locally
make dev
```

### Testing

```bash
pytest
```

## Deployment

Deploy to Railway:

```bash
railway up
```

See [docs/deployment.md](docs/deployment.md) for detailed deployment instructions.

## License

MIT

