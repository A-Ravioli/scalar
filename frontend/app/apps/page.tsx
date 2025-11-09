'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { api } from '@/lib/api';
import { App } from '@/lib/types';
import { AppCard } from '@/components/AppCard';
import { Plus, RefreshCw } from 'lucide-react';

export default function AppsPage() {
  const [apps, setApps] = useState<App[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<'all' | 'running' | 'pending' | 'completed'>('all');
  const [error, setError] = useState<string | null>(null);

  const fetchApps = async () => {
    try {
      setLoading(true);
      const params = filter !== 'all' ? { status: filter } : {};
      const fetchedApps = await api.listApps(params);
      setApps(fetchedApps);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch applications');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchApps();
    
    // Auto-refresh every 5 seconds
    const interval = setInterval(fetchApps, 5000);
    return () => clearInterval(interval);
  }, [filter]);

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this application?')) {
      return;
    }

    try {
      await api.deleteApp(id);
      setApps(apps.filter((app) => app.id !== id));
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to delete application');
    }
  };

  const filterTabs: Array<{
    key: 'all' | 'running' | 'pending' | 'completed';
    label: string;
  }> = [
    { key: 'all', label: 'All' },
    { key: 'running', label: 'Running' },
    { key: 'pending', label: 'Pending' },
    { key: 'completed', label: 'Completed' },
  ];

  return (
    <div className="max-w-7xl mx-auto px-8 py-12">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-4xl font-serif font-bold text-gray-900 mb-2">
            Your Applications
          </h1>
          <p className="text-gray-600">Manage and monitor your GPU compute applications</p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={fetchApps}
            className="p-2 text-gray-600 hover:text-indigo-600 transition-colors"
            title="Refresh"
          >
            <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
          </button>
          <Link
            href="/deploy"
            className="inline-flex items-center gap-2 bg-indigo-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-indigo-700 transition-colors"
          >
            <Plus className="w-5 h-5" />
            Deploy New App
          </Link>
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex gap-2 mb-6">
        {filterTabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setFilter(tab.key)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              filter === tab.key
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Error Message */}
      {error && (
        <div className="rounded-xl p-4 bg-red-50 text-red-800 mb-6">
          {error}
        </div>
      )}

      {/* Apps Grid */}
      {loading && apps.length === 0 ? (
        <div className="text-center py-12">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
          <p className="mt-4 text-gray-600">Loading applications...</p>
        </div>
      ) : apps.length === 0 ? (
        <div className="text-center py-12 rounded-xl bg-gray-50">
          <p className="text-gray-600 mb-4">No applications found</p>
          <Link
            href="/deploy"
            className="inline-flex items-center gap-2 text-indigo-600 hover:text-indigo-700 font-medium"
          >
            <Plus className="w-4 h-4" />
            Deploy your first application
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4">
          {apps.map((app) => (
            <AppCard key={app.id} app={app} onDelete={handleDelete} />
          ))}
        </div>
      )}
    </div>
  );
}

