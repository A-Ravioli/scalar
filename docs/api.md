# Scalar API Documentation

## Authentication

All API requests require authentication via API key:

```
Authorization: Bearer sk_your_api_key
```

### Create API Key

```http
POST /auth/api-keys
Authorization: Bearer <existing_key>
Content-Type: application/json

{
  "name": "My API Key"
}
```

Response:
```json
{
  "api_key": "sk_...",
  "id": "uuid"
}
```

## Jobs

### Submit Job

```http
POST /jobs
Authorization: Bearer sk_your_api_key
Content-Type: application/json

{
  "tier": "FAST",
  "gpu_count": 1,
  "vram_per_gpu_gb": 40.0,
  "cpu_cores": 4,
  "ram_gb": 16.0,
  "expected_duration_sec": 3600,
  "priority": 0,
  "image": "nvidia/cuda:11.8.0-base-ubuntu22.04",
  "command": ["python", "train.py"],
  "env": {
    "MODEL_PATH": "/models/bert"
  }
}
```

Response:
```json
{
  "id": "uuid",
  "user_id": "uuid",
  "tier": "FAST",
  "status": "pending",
  "gpu_count": 1,
  "vram_per_gpu_gb": 40.0,
  "cpu_cores": 4,
  "ram_gb": 16.0,
  "image": "nvidia/cuda:11.8.0-base-ubuntu22.04",
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

### Get Job Status

```http
GET /jobs/{job_id}
Authorization: Bearer sk_your_api_key
```

Response:
```json
{
  "id": "uuid",
  "status": "running",
  "node_id": "uuid",
  "gpu_indices": [0],
  ...
}
```

### List Jobs

```http
GET /jobs?status=running&tier=FAST&limit=100
Authorization: Bearer sk_your_api_key
```

### Cancel Job

```http
POST /jobs/{job_id}/cancel
Authorization: Bearer sk_your_api_key
```

## Endpoints

### Create Endpoint

```http
POST /endpoints
Authorization: Bearer sk_your_api_key
Content-Type: application/json

{
  "name": "my-endpoint",
  "image": "my-image:latest",
  "gpu_count": 1,
  "vram_per_gpu_gb": 40.0,
  "cpu_cores": 4,
  "ram_gb": 16.0
}
```

## Job Statuses

- `pending`: Job submitted, waiting to be scheduled
- `queued`: Job queued, waiting for capacity
- `scheduled`: Job scheduled on a node
- `running`: Job currently running
- `completed`: Job completed successfully
- `failed`: Job failed
- `cancelled`: Job cancelled by user

## Tiers

### FAST

- Low latency (seconds to tens of seconds)
- Higher cost per GPU-hour
- Warm capacity pre-bought

### FLEX

- Accepts delays (hours)
- Lower cost per GPU-hour
- Just-in-time capacity

