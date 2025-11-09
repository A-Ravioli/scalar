/**
 * TypeScript types for the Scalar compute platform
 */

export type Tier = 'FAST' | 'FLEX';

export type AppStatus = 
  | 'pending' 
  | 'queued' 
  | 'scheduled' 
  | 'running' 
  | 'completed' 
  | 'failed' 
  | 'cancelled';

export interface AppConfig {
  name: string;
  tier: Tier;
  gpu_count: number;
  vram_per_gpu_gb: number;
  cpu_cores: number;
  ram_gb: number;
  expected_duration_sec?: number;
  priority?: number;
  image: string;
  command?: string[];
  env?: Record<string, string>;
}

export interface App {
  id: string;
  user_id: string;
  name?: string;
  tier: Tier;
  status: AppStatus;
  gpu_count: number;
  vram_per_gpu_gb: number;
  cpu_cores: number;
  ram_gb: number;
  image: string;
  created_at: string;
  updated_at: string;
  node_id?: string;
  gpu_indices?: number[];
  command?: string[];
  env?: Record<string, string>;
}

export interface CapacitySnapshot {
  total_gpus: number;
  available_gpus: number;
  reserved_gpus: number;
  allocated_gpus: number;
  nodes_count: number;
}

export interface Node {
  id: string;
  provider: string;
  gpu_type: string;
  gpu_count: number;
  available_gpus: number;
  status: string;
  tier?: Tier;
}

export interface ApiError {
  error: string;
  detail?: string;
}

export interface OrderbookEntry {
  price: number;
  quantity_gpus: number;
  duration_hours: number;
  cumulative_quantity?: number;
}

export interface OrderbookData {
  instance_type: string;
  asks: OrderbookEntry[];
  bids: OrderbookEntry[];
  optimal_price: number | null;
  optimal_index: number | null;
  spread: number | null;
  total_ask_liquidity: number;
  total_bid_liquidity: number;
  last_updated: string;
  metadata: {
    required_gpus: number;
    node_count: number;
    gpus_per_node: number;
    mock_data?: boolean;
  };
}

export interface OrderbookRequest {
  instance_type: string;
  node_count: number;
}

