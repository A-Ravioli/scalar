'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { api } from '@/lib/api';
import { App } from '@/lib/types';
import { StatusBadge } from '@/components/StatusBadge';
import { ArrowLeft, Trash2, Terminal } from 'lucide-react';
import Link from 'next/link';

export default function AppDetailPage({ params }: { params: { id: string } }) {
  const router = useRouter();
  const [app, setApp] = useState<App | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchApp = async () => {
      try {
        const fetchedApp = await api.getApp(params.id);
        setApp(fetchedApp);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch application');
      } finally {
        setLoading(false);
      }
    };

    fetchApp();

    // Auto-refresh every 3 seconds
    const interval = setInterval(fetchApp, 3000);
    return () => clearInterval(interval);
  }, [params.id]);

  const handleDelete = async () => {
    if (!confirm('Are you sure you want to delete this application?')) {
      return;
    }

    try {
      await api.deleteApp(params.id);
      router.push('/apps');
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to delete application');
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const getRuntime = (createdAt: string) => {
    const start = new Date(createdAt);
    const now = new Date();
    const diff = now.getTime() - start.getTime();
    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) return `${days}d ${hours % 24}h`;
    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
    return `${seconds}s`;
  };

  if (loading) {
    return (
      <div className="max-w-5xl mx-auto px-8 py-12">
        <div className="text-center py-12">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
          <p className="mt-4 text-gray-600">Loading application...</p>
        </div>
      </div>
    );
  }

  if (error || !app) {
    return (
      <div className="max-w-5xl mx-auto px-8 py-12">
        <div className="rounded-xl p-6 bg-red-50 text-red-800">
          {error || 'Application not found'}
        </div>
        <Link
          href="/apps"
          className="inline-flex items-center gap-2 mt-4 text-indigo-600 hover:text-indigo-700 font-medium"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Applications
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto px-8 py-12">
      <div className="mb-6">
        <Link
          href="/apps"
          className="inline-flex items-center gap-2 text-gray-600 hover:text-gray-900 mb-4"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Applications
        </Link>
      </div>

      {/* Header */}
      <div className="mb-8">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h1 className="text-4xl font-serif font-bold text-gray-900 mb-2">
              {app.name || `Application ${app.id.slice(0, 8)}`}
            </h1>
            <p className="text-gray-600 font-mono text-sm">{app.id}</p>
          </div>
          <StatusBadge status={app.status} />
        </div>
      </div>

      {/* Overview Card */}
      <div className="rounded-xl p-6 bg-gray-50 mb-6">
        <h2 className="text-xl font-serif font-semibold text-gray-900 mb-4">
          Overview
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
          <div>
            <p className="text-sm text-gray-500 mb-1">Tier</p>
            <p className="text-gray-900 font-medium">
              <span className="px-2.5 py-1 bg-white rounded-md">
                {app.tier}
              </span>
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-500 mb-1">GPU Count</p>
            <p className="text-gray-900 font-medium">{app.gpu_count}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500 mb-1">VRAM per GPU</p>
            <p className="text-gray-900 font-medium">{app.vram_per_gpu_gb} GB</p>
          </div>
          <div>
            <p className="text-sm text-gray-500 mb-1">CPU Cores</p>
            <p className="text-gray-900 font-medium">{app.cpu_cores}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500 mb-1">RAM</p>
            <p className="text-gray-900 font-medium">{app.ram_gb} GB</p>
          </div>
          {app.status === 'running' && (
            <div>
              <p className="text-sm text-gray-500 mb-1">Runtime</p>
              <p className="text-gray-900 font-medium">{getRuntime(app.created_at)}</p>
            </div>
          )}
        </div>
      </div>

      {/* Configuration Card */}
      <div className="rounded-xl p-6 bg-gray-50 mb-6">
        <h2 className="text-xl font-serif font-semibold text-gray-900 mb-4">
          Configuration
        </h2>
        <div className="space-y-4">
          <div>
            <p className="text-sm text-gray-500 mb-1">Docker Image</p>
            <p className="text-gray-900 font-mono text-sm bg-white px-4 py-2.5 rounded-lg">
              {app.image}
            </p>
          </div>
          {app.command && (
            <div>
              <p className="text-sm text-gray-500 mb-1">Command</p>
              <p className="text-gray-900 font-mono text-sm bg-white px-4 py-2.5 rounded-lg">
                {JSON.stringify(app.command)}
              </p>
            </div>
          )}
          {app.env && Object.keys(app.env).length > 0 && (
            <div>
              <p className="text-sm text-gray-500 mb-2">Environment Variables</p>
              <div className="space-y-1">
                {Object.entries(app.env).map(([key, value]) => (
                  <div
                    key={key}
                    className="font-mono text-sm bg-white px-4 py-2.5 rounded-lg"
                  >
                    <span className="text-indigo-600">{key}</span>
                    <span className="text-gray-600">=</span>
                    <span className="text-gray-900">{value}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Allocation Card (if running) */}
      {app.node_id && (
        <div className="rounded-xl p-6 bg-gray-50 mb-6">
          <h2 className="text-xl font-serif font-semibold text-gray-900 mb-4">
            Allocation
          </h2>
          <div className="space-y-3">
            <div>
              <p className="text-sm text-gray-500 mb-1">Node ID</p>
              <p className="text-gray-900 font-mono text-sm">{app.node_id}</p>
            </div>
            {app.gpu_indices && (
              <div>
                <p className="text-sm text-gray-500 mb-1">GPU Indices</p>
                <p className="text-gray-900 font-mono text-sm">
                  {app.gpu_indices.join(', ')}
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Timestamps Card */}
      <div className="rounded-xl p-6 bg-gray-50 mb-6">
        <h2 className="text-xl font-serif font-semibold text-gray-900 mb-4">
          Timeline
        </h2>
        <div className="space-y-3">
          <div>
            <p className="text-sm text-gray-500 mb-1">Created</p>
            <p className="text-gray-900 text-sm">{formatDate(app.created_at)}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500 mb-1">Last Updated</p>
            <p className="text-gray-900 text-sm">{formatDate(app.updated_at)}</p>
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-4">
        <button
          className="inline-flex items-center gap-2 px-6 py-3 rounded-lg font-medium bg-gray-100 text-gray-700 hover:bg-gray-200 transition-colors disabled:opacity-50"
          title="View logs (coming soon)"
          disabled
        >
          <Terminal className="w-5 h-5" />
          View Logs
        </button>
        <button
          onClick={handleDelete}
          className="inline-flex items-center gap-2 px-6 py-3 rounded-lg font-medium bg-red-50 text-red-700 hover:bg-red-100 transition-colors"
        >
          <Trash2 className="w-5 h-5" />
          Delete Application
        </button>
      </div>
    </div>
  );
}

