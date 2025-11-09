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
    icon: React.ReactNode;
  }> = [
    {
      value: 'FAST',
      label: 'Fast',
      description: 'Low latency (seconds), higher cost, warm capacity',
      icon: <Zap className="w-5 h-5 text-indigo-600" />,
    },
    {
      value: 'FLEX',
      label: 'Flex',
      description: 'Accepts delays (hours), lower cost, just-in-time capacity',
      icon: <Clock className="w-5 h-5 text-indigo-600" />,
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {tiers.map((tier) => (
        <button
          key={tier.value}
          type="button"
          onClick={() => onChange(tier.value)}
          className={`text-left p-6 rounded-lg border-2 transition-all ${
            selected === tier.value
              ? 'border-indigo-600 bg-indigo-50'
              : 'border-gray-300 bg-white hover:border-gray-400'
          }`}
        >
          <div className="flex items-start gap-3">
            <div className="mt-1">{tier.icon}</div>
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-gray-900 mb-1">
                {tier.label}
              </h3>
              <p className="text-sm text-gray-600">{tier.description}</p>
            </div>
          </div>
        </button>
      ))}
    </div>
  );
}

