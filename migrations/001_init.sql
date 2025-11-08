-- Scalar MVP Database Schema
-- Supabase migration: 001_init.sql

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table (maps to Supabase auth.users)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- API keys for authentication
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_hash TEXT NOT NULL UNIQUE,
    name TEXT,
    last_used_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX idx_api_keys_key_hash ON api_keys(key_hash);

-- Blocks (SFCompute contracts)
CREATE TABLE IF NOT EXISTS blocks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    provider TEXT NOT NULL DEFAULT 'sfcompute',
    instance_type TEXT NOT NULL,
    gpus_per_node INTEGER NOT NULL,
    vram_per_gpu_gb FLOAT NOT NULL,
    cpu_per_node INTEGER NOT NULL,
    ram_per_node_gb FLOAT NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    cost_per_hour FLOAT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'active', 'selling', 'sold')),
    tier TEXT CHECK (tier IN ('FAST', 'FLEX')),
    sfcompute_contract_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_blocks_status ON blocks(status);
CREATE INDEX idx_blocks_tier ON blocks(tier);
CREATE INDEX idx_blocks_end_time ON blocks(end_time);

-- Nodes (compute instances spawned from blocks)
CREATE TABLE IF NOT EXISTS nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    block_id UUID NOT NULL REFERENCES blocks(id) ON DELETE CASCADE,
    tier TEXT NOT NULL CHECK (tier IN ('FAST', 'FLEX')),
    cpu_used FLOAT DEFAULT 0.0,
    ram_used_gb FLOAT DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_nodes_block_id ON nodes(block_id);
CREATE INDEX idx_nodes_tier ON nodes(tier);

-- GPU assignments per node
CREATE TABLE IF NOT EXISTS gpu_assignments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_id UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    gpu_index INTEGER NOT NULL,
    vram_total_gb FLOAT NOT NULL,
    vram_used_gb FLOAT DEFAULT 0.0,
    job_id UUID,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(node_id, gpu_index)
);

CREATE INDEX idx_gpu_assignments_node_id ON gpu_assignments(node_id);
CREATE INDEX idx_gpu_assignments_job_id ON gpu_assignments(job_id);

-- Jobs
CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tier TEXT NOT NULL CHECK (tier IN ('FAST', 'FLEX')),
    gpu_count INTEGER NOT NULL,
    vram_per_gpu_gb FLOAT NOT NULL,
    cpu_cores INTEGER NOT NULL,
    ram_gb FLOAT NOT NULL,
    expected_duration_sec INTEGER DEFAULT 3600,
    priority INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'queued', 'scheduled', 'running', 'completed', 'failed', 'cancelled')),
    image TEXT NOT NULL,
    command JSONB,
    env JSONB,
    node_id UUID REFERENCES nodes(id),
    gpu_indices INTEGER[],
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    exit_code INTEGER,
    usage_collected BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_jobs_user_id ON jobs(user_id);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_tier ON jobs(tier);
CREATE INDEX idx_jobs_node_id ON jobs(node_id);
CREATE INDEX idx_jobs_created_at ON jobs(created_at);

-- Job runs (for tracking multiple runs/retries)
CREATE TABLE IF NOT EXISTS job_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    node_id UUID REFERENCES nodes(id),
    status TEXT NOT NULL CHECK (status IN ('scheduled', 'running', 'completed', 'failed')),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    exit_code INTEGER,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_job_runs_job_id ON job_runs(job_id);
CREATE INDEX idx_job_runs_node_id ON job_runs(node_id);

-- Usage events (for billing)
CREATE TABLE IF NOT EXISTS usage_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    node_id UUID NOT NULL REFERENCES nodes(id),
    gpu_hours FLOAT NOT NULL,
    cpu_hours FLOAT NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    cost_usd FLOAT NOT NULL,
    sfcompute_cost_usd FLOAT NOT NULL,
    margin_usd FLOAT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_usage_events_user_id ON usage_events(user_id);
CREATE INDEX idx_usage_events_job_id ON usage_events(job_id);
CREATE INDEX idx_usage_events_created_at ON usage_events(created_at);

-- Invoices
CREATE TABLE IF NOT EXISTS invoices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    total_usd FLOAT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'paid', 'failed')),
    stripe_invoice_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    paid_at TIMESTAMPTZ
);

CREATE INDEX idx_invoices_user_id ON invoices(user_id);
CREATE INDEX idx_invoices_status ON invoices(status);

-- Reservations (for capacity locking)
CREATE TABLE IF NOT EXISTS reservations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    node_id UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    gpu_indices INTEGER[] NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_reservations_job_id ON reservations(job_id);
CREATE INDEX idx_reservations_node_id ON reservations(node_id);
CREATE INDEX idx_reservations_expires_at ON reservations(expires_at);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_blocks_updated_at BEFORE UPDATE ON blocks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_nodes_updated_at BEFORE UPDATE ON nodes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_gpu_assignments_updated_at BEFORE UPDATE ON gpu_assignments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_jobs_updated_at BEFORE UPDATE ON jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- RLS Policies (Row Level Security)
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE usage_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE invoices ENABLE ROW LEVEL SECURITY;

-- Users can read their own data
CREATE POLICY "Users can read own data" ON users
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can read own api keys" ON api_keys
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can read own jobs" ON jobs
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can read own usage events" ON usage_events
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can read own invoices" ON invoices
    FOR SELECT USING (auth.uid() = user_id);

-- Service role can do everything (for internal services)
CREATE POLICY "Service role full access" ON users
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access" ON api_keys
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access" ON jobs
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access" ON usage_events
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role full access" ON invoices
    FOR ALL USING (auth.role() = 'service_role');

