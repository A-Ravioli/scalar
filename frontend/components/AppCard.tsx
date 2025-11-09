import { App } from '@/lib/types';
import { StatusBadge } from './StatusBadge';
import { Trash2, ExternalLink } from 'lucide-react';
import Link from 'next/link';

interface AppCardProps {
  app: App;
  onDelete?: (id: string) => void;
}

export function AppCard({ app, onDelete }: AppCardProps) {
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

  const truncateId = (id: string) => {
    return `${id.slice(0, 8)}...${id.slice(-4)}`;
  };

  const truncateImage = (image: string) => {
    if (image.length > 40) {
      return image.slice(0, 37) + '...';
    }
    return image;
  };

  return (
    <div className="border border-gray-300 rounded-lg p-6 bg-white hover:border-gray-400 transition-colors">
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-2">
            <h3 className="text-lg font-semibold text-gray-900">
              {app.name || truncateId(app.id)}
            </h3>
            <StatusBadge status={app.status} />
          </div>
          <div className="flex items-center gap-4 text-sm text-gray-600">
            <span className="px-2 py-0.5 bg-gray-100 rounded border border-gray-300">
              {app.tier}
            </span>
            <span>{app.gpu_count} GPU{app.gpu_count !== 1 ? 's' : ''}</span>
            <span>{formatDate(app.created_at)}</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Link
            href={`/apps/${app.id}`}
            className="p-2 text-gray-600 hover:text-indigo-600 transition-colors"
          >
            <ExternalLink className="w-4 h-4" />
          </Link>
          {onDelete && (
            <button
              onClick={() => onDelete(app.id)}
              className="p-2 text-gray-600 hover:text-red-600 transition-colors"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>
      <div className="text-sm text-gray-600">
        <p className="font-mono">{truncateImage(app.image)}</p>
      </div>
      {app.node_id && (
        <div className="mt-3 pt-3 border-t border-gray-200 text-xs text-gray-500">
          Node: {truncateId(app.node_id)}
          {app.gpu_indices && ` • GPUs: ${app.gpu_indices.join(', ')}`}
        </div>
      )}
    </div>
  );
}

