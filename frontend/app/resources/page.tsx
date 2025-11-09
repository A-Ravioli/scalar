'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { CapacitySnapshot } from '@/lib/types';
import { Server, Cpu, HardDrive, RefreshCw } from 'lucide-react';

export default function ResourcesPage() {
  const [capacity, setCapacity] = useState<CapacitySnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchCapacity = async () => {
    try {
      setLoading(true);
      const data = await api.getCapacity();
      setCapacity(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch capacity');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCapacity();

    // Auto-refresh every 10 seconds
    const interval = setInterval(fetchCapacity, 10000);
    return () => clearInterval(interval);
  }, []);

  const getUtilization = () => {
    if (!capacity || capacity.total_gpus === 0) return 0;
    return Math.round(((capacity.allocated_gpus + capacity.reserved_gpus) / capacity.total_gpus) * 100);
  };

  return (
    <div className="max-w-7xl mx-auto px-8 py-12">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-4xl font-serif font-bold text-gray-900 mb-2">
            Resources
          </h1>
          <p className="text-gray-600">View available compute capacity</p>
        </div>
        <button
          onClick={fetchCapacity}
          className="p-2 text-gray-600 hover:text-indigo-600 transition-colors"
          title="Refresh"
        >
          <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Error Message */}
      {error && (
        <div className="rounded-xl p-4 bg-red-50 text-red-800 mb-6">
          {error}
        </div>
      )}

      {/* Capacity Overview */}
      {loading && !capacity ? (
        <div className="text-center py-12">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
          <p className="mt-4 text-gray-600">Loading capacity...</p>
        </div>
      ) : capacity ? (
        <>
          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div className="rounded-xl p-6 bg-gray-50">
              <div className="flex items-center gap-3 mb-2">
                <Server className="w-5 h-5 text-indigo-600" />
                <h3 className="text-sm font-medium text-gray-600">Total Nodes</h3>
              </div>
              <p className="text-3xl font-semibold text-gray-900">
                {capacity.nodes_count}
              </p>
            </div>

            <div className="rounded-xl p-6 bg-gray-50">
              <div className="flex items-center gap-3 mb-2">
                <Cpu className="w-5 h-5 text-indigo-600" />
                <h3 className="text-sm font-medium text-gray-600">Total GPUs</h3>
              </div>
              <p className="text-3xl font-semibold text-gray-900">
                {capacity.total_gpus}
              </p>
            </div>

            <div className="rounded-xl p-6 bg-gray-50">
              <div className="flex items-center gap-3 mb-2">
                <HardDrive className="w-5 h-5 text-green-600" />
                <h3 className="text-sm font-medium text-gray-600">Available GPUs</h3>
              </div>
              <p className="text-3xl font-semibold text-gray-900">
                {capacity.available_gpus}
              </p>
            </div>

            <div className="rounded-xl p-6 bg-gray-50">
              <div className="flex items-center gap-3 mb-2">
                <Cpu className="w-5 h-5 text-orange-600" />
                <h3 className="text-sm font-medium text-gray-600">Utilization</h3>
              </div>
              <p className="text-3xl font-semibold text-gray-900">
                {getUtilization()}%
              </p>
            </div>
          </div>

          {/* Capacity Breakdown */}
          <div className="rounded-xl p-6 bg-gray-50 mb-8">
            <h2 className="text-2xl font-serif font-semibold text-gray-900 mb-6">
              Capacity Breakdown
            </h2>
            
            {/* Visual bar */}
            <div className="mb-6">
              <div className="flex items-center gap-4 text-sm text-gray-600 mb-2">
                <span className="flex items-center gap-2">
                  <span className="w-3 h-3 bg-green-500 rounded"></span>
                  Available: {capacity.available_gpus}
                </span>
                <span className="flex items-center gap-2">
                  <span className="w-3 h-3 bg-yellow-500 rounded"></span>
                  Reserved: {capacity.reserved_gpus}
                </span>
                <span className="flex items-center gap-2">
                  <span className="w-3 h-3 bg-indigo-500 rounded"></span>
                  Allocated: {capacity.allocated_gpus}
                </span>
              </div>
              <div className="w-full h-8 bg-gray-200 rounded-lg overflow-hidden flex">
                {capacity.total_gpus > 0 && (
                  <>
                    <div
                      className="bg-green-500"
                      style={{
                        width: `${(capacity.available_gpus / capacity.total_gpus) * 100}%`,
                      }}
                    ></div>
                    <div
                      className="bg-yellow-500"
                      style={{
                        width: `${(capacity.reserved_gpus / capacity.total_gpus) * 100}%`,
                      }}
                    ></div>
                    <div
                      className="bg-indigo-500"
                      style={{
                        width: `${(capacity.allocated_gpus / capacity.total_gpus) * 100}%`,
                      }}
                    ></div>
                  </>
                )}
              </div>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div>
                <p className="text-sm text-gray-600 mb-1">Available</p>
                <p className="text-2xl font-semibold text-green-600">
                  {capacity.available_gpus}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600 mb-1">Reserved</p>
                <p className="text-2xl font-semibold text-yellow-600">
                  {capacity.reserved_gpus}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600 mb-1">Allocated</p>
                <p className="text-2xl font-semibold text-indigo-600">
                  {capacity.allocated_gpus}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600 mb-1">Total</p>
                <p className="text-2xl font-semibold text-gray-900">
                  {capacity.total_gpus}
                </p>
              </div>
            </div>
          </div>

          {/* Info Box */}
          <div className="rounded-xl p-6 bg-gray-50">
            <h3 className="font-semibold text-gray-900 mb-2">About Capacity</h3>
            <p className="text-sm text-gray-600 mb-2">
              This page shows the current capacity across all compute nodes in the cluster.
            </p>
            <ul className="text-sm text-gray-600 space-y-1 list-disc list-inside">
              <li>
                <strong>Available:</strong> GPUs ready to be allocated to new applications
              </li>
              <li>
                <strong>Reserved:</strong> GPUs temporarily reserved by the scheduler
              </li>
              <li>
                <strong>Allocated:</strong> GPUs currently running applications
              </li>
            </ul>
          </div>
        </>
      ) : (
        <div className="text-center py-12 rounded-xl bg-gray-50">
          <p className="text-gray-600">No capacity data available</p>
        </div>
      )}
    </div>
  );
}

