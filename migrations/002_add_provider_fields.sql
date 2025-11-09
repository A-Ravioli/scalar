-- Migration: Add provider fields to blocks table
-- This migration adds support for multiple compute providers (Prime Intellect, SFCompute)

-- Add provider-specific fields to blocks table
ALTER TABLE blocks
ADD COLUMN IF NOT EXISTS provider_instance_id TEXT,
ADD COLUMN IF NOT EXISTS region TEXT,
ADD COLUMN IF NOT EXISTS preemptible BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS drain_deadline TIMESTAMPTZ;

-- Update existing blocks to have default provider
UPDATE blocks SET provider = 'sfcompute' WHERE provider IS NULL;

-- Create index on provider for faster queries
CREATE INDEX IF NOT EXISTS idx_blocks_provider ON blocks(provider);

-- Create index on provider_instance_id for provider lookups
CREATE INDEX IF NOT EXISTS idx_blocks_provider_instance_id ON blocks(provider_instance_id);

-- Create index on drain_deadline for consolidation queries
CREATE INDEX IF NOT EXISTS idx_blocks_drain_deadline ON blocks(drain_deadline) WHERE drain_deadline IS NOT NULL;

-- Add comment to document provider field
COMMENT ON COLUMN blocks.provider IS 'Compute provider: sfcompute or prime';
COMMENT ON COLUMN blocks.provider_instance_id IS 'Provider-specific instance ID (contract_id, pod_id, etc.)';
COMMENT ON COLUMN blocks.region IS 'Provider region/zone';
COMMENT ON COLUMN blocks.preemptible IS 'Whether instance can be preempted';
COMMENT ON COLUMN blocks.drain_deadline IS 'When to drain this capacity for consolidation';

