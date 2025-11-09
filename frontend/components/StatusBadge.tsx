import { AppStatus } from '@/lib/types';

interface StatusBadgeProps {
  status: AppStatus;
}

export function StatusBadge({ status }: StatusBadgeProps) {
  const statusConfig: Record<AppStatus, { color: string; label: string }> = {
    pending: { color: 'bg-yellow-100 text-yellow-800 border-yellow-300', label: 'Pending' },
    queued: { color: 'bg-yellow-100 text-yellow-800 border-yellow-300', label: 'Queued' },
    scheduled: { color: 'bg-blue-100 text-blue-800 border-blue-300', label: 'Scheduled' },
    running: { color: 'bg-green-100 text-green-800 border-green-300', label: 'Running' },
    completed: { color: 'bg-gray-100 text-gray-800 border-gray-300', label: 'Completed' },
    failed: { color: 'bg-red-100 text-red-800 border-red-300', label: 'Failed' },
    cancelled: { color: 'bg-gray-100 text-gray-800 border-gray-300', label: 'Cancelled' },
  };

  const config = statusConfig[status];

  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-md text-xs font-medium border ${config.color}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full mr-1.5 ${status === 'running' ? 'bg-green-500 animate-pulse' : ''}`}></span>
      {config.label}
    </span>
  );
}

