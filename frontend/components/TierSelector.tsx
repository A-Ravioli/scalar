import { Tier } from '@/lib/types';
import { Zap, Clock } from 'lucide-react';

interface TierSelectorProps {
  selected: Tier;
  onChange: (tier: Tier) => void;
}

export function TierSelector({ selected, onChange }: TierSelectorProps) {
  const tiers: Array<{
    value: Tier;
    label: string;
    description: string;
    Icon: React.ComponentType<{ className?: string }>;
  }> = [
    {
      value: 'FAST',
      label: 'Fast',
      description: 'Low latency (seconds), higher cost, warm capacity',
      Icon: Zap,
    },
    {
      value: 'FLEX',
      label: 'Flex',
      description: 'Accepts delays (hours), lower cost, just-in-time capacity',
      Icon: Clock,
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {tiers.map((tier) => (
        <button
          key={tier.value}
          type="button"
          onClick={() => onChange(tier.value)}
          className={`text-left p-6 rounded-lg transition-all ${
            selected === tier.value
              ? 'bg-indigo-600 text-white'
              : 'bg-white hover:bg-gray-100'
          }`}
        >
          <div className="flex items-start gap-3">
            <div className="mt-1">
              <tier.Icon className={`w-5 h-5 ${
                selected === tier.value ? 'text-white' : 'text-indigo-600'
              }`} />
            </div>
            <div className="flex-1">
              <h3 className={`text-lg font-semibold mb-1 ${
                selected === tier.value ? 'text-white' : 'text-gray-900'
              }`}>
                {tier.label}
              </h3>
              <p className={`text-sm ${
                selected === tier.value ? 'text-indigo-100' : 'text-gray-600'
              }`}>
                {tier.description}
              </p>
            </div>
          </div>
        </button>
      ))}
    </div>
  );
}

