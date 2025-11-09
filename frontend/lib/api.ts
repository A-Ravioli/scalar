/**
 * API client for the Scalar compute platform
 */

import { App, AppConfig, CapacitySnapshot, Node, Tier, OrderbookData } from './types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || '';

class ApiClient {
  private baseUrl: string;
  private apiKey: string;

  constructor(baseUrl: string, apiKey: string) {
    this.baseUrl = baseUrl;
    this.apiKey = apiKey;
  }

  private async fetch<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
    }

    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        error: 'An error occurred',
      }));
      throw new Error(error.detail || error.error || 'An error occurred');
    }

    return response.json();
  }

  /**
   * Create a new app (backend creates a job)
   */
  async createApp(config: AppConfig): Promise<App> {
    return this.fetch<App>('/jobs', {
      method: 'POST',
      body: JSON.stringify({
        tier: config.tier,
        gpu_count: config.gpu_count,
        vram_per_gpu_gb: config.vram_per_gpu_gb,
        cpu_cores: config.cpu_cores,
        ram_gb: config.ram_gb,
        expected_duration_sec: config.expected_duration_sec || 3600,
        priority: config.priority || 0,
        image: config.image,
        command: config.command,
        env: config.env,
      }),
    });
  }

  /**
   * List all apps (optionally filter by status or tier)
   */
  async listApps(params?: {
    status?: string;
    tier?: string;
    limit?: number;
  }): Promise<App[]> {
    const queryParams = new URLSearchParams();
    if (params?.status) queryParams.append('status', params.status);
    if (params?.tier) queryParams.append('tier', params.tier);
    if (params?.limit) queryParams.append('limit', params.limit.toString());

    const query = queryParams.toString();
    return this.fetch<App[]>(`/jobs${query ? `?${query}` : ''}`);
  }

  /**
   * Get a single app by ID
   */
  async getApp(id: string): Promise<App> {
    return this.fetch<App>(`/jobs/${id}`);
  }

  /**
   * Delete/cancel an app
   */
  async deleteApp(id: string): Promise<{ status: string }> {
    return this.fetch<{ status: string }>(`/jobs/${id}/cancel`, {
      method: 'POST',
    });
  }

  /**
   * Get capacity snapshot (from capacity manager)
   */
  async getCapacity(tier?: Tier): Promise<CapacitySnapshot> {
    const query = tier ? `?tier=${tier}` : '';
    // Note: This endpoint is on the capacity manager service
    // In production, this might be a different port/service
    return this.fetch<CapacitySnapshot>(
      `/capacity_snapshot${query}`
    ).catch(() => {
      // Fallback if capacity manager isn't running
      return {
        total_gpus: 0,
        available_gpus: 0,
        reserved_gpus: 0,
        allocated_gpus: 0,
        nodes_count: 0,
      };
    });
  }

  /**
   * Get orderbook for an instance type with optimal price recommendation
   */
  async getOrderbook(
    instanceType: string,
    nodeCount: number = 1
  ): Promise<OrderbookData> {
    const queryParams = new URLSearchParams({
      instance_type: instanceType,
      node_count: nodeCount.toString(),
    });
    return this.fetch<OrderbookData>(`/orderbook?${queryParams.toString()}`);
  }
}

// Export a singleton instance
export const api = new ApiClient(API_URL, API_KEY);

// Also export the class for testing or custom instances
export { ApiClient };

