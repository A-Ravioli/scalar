'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { api } from '@/lib/api';
import { App } from '@/lib/types';
import { StatusBadge } from '@/components/StatusBadge';
import { ArrowRight, Activity, Cpu, DollarSign } from 'lucide-react';

export default function Home() {
  const [recentApps, setRecentApps] = useState<App[]>([]);
  const [stats, setStats] = useState({
    activeApps: 0,
    totalGPUs: 0,
    runningApps: 0,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        const apps = await api.listApps({ limit: 5 });
        setRecentApps(apps);

        const runningApps = apps.filter((app) => app.status === 'running');
        const totalGPUs = runningApps.reduce((sum, app) => sum + app.gpu_count, 0);

        setStats({
          activeApps: apps.length,
          totalGPUs,
          runningApps: runningApps.length,
        });
      } catch (error) {
        console.error('Failed to fetch apps:', error);
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, []);

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) return `${days}d ago`;
    if (hours > 0) return `${hours}h ago`;
    if (minutes > 0) return `${minutes}m ago`;
    return 'Just now';
  };

  return (
    <div className="max-w-6xl mx-auto px-8 py-12">
      {/* Hero Section */}
      <div className="text-center mb-16">
        <h1 className="text-5xl font-serif font-bold text-gray-900 mb-6">
          Autoscaling GPU Compute
        </h1>
        <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
          Deploy GPU-intensive workloads with automatic scaling. Never pay to move
          your data. Run clusters from UEFI on up and get support just like a
          traditional cloud.
        </p>
        <Link
          href="/deploy"
          className="inline-flex items-center gap-2 bg-indigo-600 text-white px-8 py-4 rounded-lg font-medium hover:bg-indigo-700 transition-colors text-lg"
        >
          Deploy Application
          <ArrowRight className="w-5 h-5" />
        </Link>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
        <div className="rounded-xl p-6 bg-gray-50">
          <div className="flex items-center gap-3 mb-2">
            <Activity className="w-5 h-5 text-indigo-600" />
            <h3 className="text-sm font-medium text-gray-600">Running Apps</h3>
          </div>
          <p className="text-3xl font-semibold text-gray-900">
            {loading ? '-' : stats.runningApps}
          </p>
        </div>
        <div className="rounded-xl p-6 bg-gray-50">
          <div className="flex items-center gap-3 mb-2">
            <Cpu className="w-5 h-5 text-indigo-600" />
            <h3 className="text-sm font-medium text-gray-600">Active GPUs</h3>
          </div>
          <p className="text-3xl font-semibold text-gray-900">
            {loading ? '-' : stats.totalGPUs}
          </p>
        </div>
        <div className="rounded-xl p-6 bg-gray-50">
          <div className="flex items-center gap-3 mb-2">
            <DollarSign className="w-5 h-5 text-indigo-600" />
            <h3 className="text-sm font-medium text-gray-600">Total Apps</h3>
          </div>
          <p className="text-3xl font-semibold text-gray-900">
            {loading ? '-' : stats.activeApps}
          </p>
        </div>
      </div>

      {/* Recent Apps */}
      <div>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-serif font-semibold text-gray-900">
            Recent Applications
          </h2>
          <Link
            href="/apps"
            className="text-sm text-indigo-600 hover:text-indigo-700 font-medium"
          >
            View all
          </Link>
        </div>

        {loading ? (
          <div className="text-center py-8 text-gray-500">Loading...</div>
        ) : recentApps.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-gray-600 mb-4">No applications yet</p>
            <Link
              href="/deploy"
              className="inline-flex items-center gap-2 text-indigo-600 hover:text-indigo-700 font-medium"
            >
              Deploy your first application
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-500">
                    Name
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-500">
                    Status
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-500">
                    Tier
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-500">
                    GPUs
                  </th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-500">
                    Created
                  </th>
                </tr>
              </thead>
              <tbody>
                {recentApps.map((app) => (
                  <tr key={app.id} className="hover:bg-gray-50 transition-colors">
                    <td className="py-4 px-4">
                      <Link
                        href={`/apps/${app.id}`}
                        className="text-gray-900 hover:text-indigo-600 font-medium"
                      >
                        {app.name || `${app.id.slice(0, 8)}...${app.id.slice(-4)}`}
                      </Link>
                    </td>
                    <td className="py-4 px-4">
                      <StatusBadge status={app.status} />
                    </td>
                    <td className="py-4 px-4">
                      <span className="px-2.5 py-1 bg-gray-100 rounded-md text-sm text-gray-700">
                        {app.tier}
                      </span>
                    </td>
                    <td className="py-4 px-4 text-gray-900">{app.gpu_count}</td>
                    <td className="py-4 px-4 text-gray-500 text-sm">
                      {formatDate(app.created_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
