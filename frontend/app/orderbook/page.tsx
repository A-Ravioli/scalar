'use client';

import { useState, useEffect } from 'react';
import { api } from '@/lib/api';
import { OrderbookData, OrderbookEntry } from '@/lib/types';
import { TrendingUp, TrendingDown, AlertCircle } from 'lucide-react';

export default function OrderbookPage() {
  const [instanceType, setInstanceType] = useState('8xH100');
  const [nodeCount, setNodeCount] = useState(1);
  const [orderbook, setOrderbook] = useState<OrderbookData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load orderbook on mount and when parameters change
  useEffect(() => {
    fetchOrderbook();
  }, []);

  const fetchOrderbook = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getOrderbook(instanceType, nodeCount);
      setOrderbook(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch orderbook');
      console.error('Failed to fetch orderbook:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    fetchOrderbook();
  };

  const formatPrice = (price: number) => {
    return `$${price.toFixed(2)}`;
  };

  const formatDuration = (hours: number) => {
    if (hours >= 720) return `${Math.floor(hours / 720)}mo`;
    if (hours >= 168) return `${Math.floor(hours / 168)}w`;
    if (hours >= 24) return `${Math.floor(hours / 24)}d`;
    return `${hours}h`;
  };

  const calculateCumulativeQuantity = (entries: OrderbookEntry[]) => {
    let cumulative = 0;
    return entries.map((entry) => {
      cumulative += entry.quantity_gpus;
      return { ...entry, cumulative_quantity: cumulative };
    });
  };

  const renderOrderbookTable = (
    entries: OrderbookEntry[],
    type: 'ask' | 'bid',
    optimalIndex: number | null
  ) => {
    const sortedEntries = [...entries].sort((a, b) =>
      type === 'ask' ? a.price - b.price : b.price - a.price
    );
    const withCumulative = calculateCumulativeQuantity(sortedEntries);

    const maxQuantity = Math.max(...sortedEntries.map((e) => e.quantity_gpus), 1);

    return (
      <div className="overflow-hidden">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-300 bg-gray-50">
              <th className="text-left py-3 px-4 text-xs font-semibold text-gray-600 uppercase">
                Price ($/GPU-hr)
              </th>
              <th className="text-right py-3 px-4 text-xs font-semibold text-gray-600 uppercase">
                Quantity (GPUs)
              </th>
              <th className="text-right py-3 px-4 text-xs font-semibold text-gray-600 uppercase">
                Duration
              </th>
              <th className="text-right py-3 px-4 text-xs font-semibold text-gray-600 uppercase">
                Cumulative
              </th>
              <th className="w-32 py-3 px-4 text-xs font-semibold text-gray-600 uppercase">
                Depth
              </th>
            </tr>
          </thead>
          <tbody>
            {withCumulative.length === 0 ? (
              <tr>
                <td colSpan={5} className="text-center py-8 text-gray-500">
                  No orders
                </td>
              </tr>
            ) : (
              withCumulative.map((entry, idx) => {
                const isOptimal =
                  type === 'ask' &&
                  optimalIndex !== null &&
                  entry.price === sortedEntries[optimalIndex]?.price;
                const depthPercent = (entry.quantity_gpus / maxQuantity) * 100;

                return (
                  <tr
                    key={idx}
                    className={`border-b border-gray-200 last:border-0 transition-colors ${
                      isOptimal
                        ? 'bg-green-50 border-green-200'
                        : 'hover:bg-gray-50'
                    }`}
                  >
                    <td
                      className={`py-3 px-4 font-semibold ${
                        type === 'ask' ? 'text-red-600' : 'text-green-600'
                      }`}
                    >
                      {formatPrice(entry.price)}
                      {isOptimal && (
                        <span className="ml-2 text-xs bg-green-600 text-white px-2 py-0.5 rounded font-medium">
                          OPTIMAL
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-right text-gray-900">
                      {entry.quantity_gpus}
                    </td>
                    <td className="py-3 px-4 text-right text-gray-600 text-sm">
                      {formatDuration(entry.duration_hours)}
                    </td>
                    <td className="py-3 px-4 text-right text-gray-900 font-medium">
                      {entry.cumulative_quantity}
                    </td>
                    <td className="py-3 px-4">
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full ${
                            type === 'ask' ? 'bg-red-500' : 'bg-green-500'
                          }`}
                          style={{ width: `${depthPercent}%` }}
                        />
                      </div>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div className="max-w-7xl mx-auto px-8 py-12">
      <div className="mb-8">
        <h1 className="text-4xl font-serif font-bold text-gray-900 mb-3">
          SFCompute Order Book
        </h1>
        <p className="text-lg text-gray-600">
          View the marketplace orderbook and find the optimal price point for your
          compute needs
        </p>
      </div>

      {/* Input Form */}
      <div className="rounded-xl p-6 bg-gray-50 mb-6">
        <form onSubmit={handleSubmit} className="flex gap-4 items-end">
          <div className="flex-1">
            <label
              htmlFor="instanceType"
              className="block text-sm font-medium text-gray-700 mb-2"
            >
              Instance Type
            </label>
            <select
              id="instanceType"
              value={instanceType}
              onChange={(e) => setInstanceType(e.target.value)}
              className="w-full px-4 py-2.5 bg-white rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none"
            >
              <option value="8xH100">8xH100</option>
              <option value="8xA100">8xA100</option>
              <option value="8xA6000">8xA6000</option>
            </select>
          </div>
          <div className="flex-1">
            <label
              htmlFor="nodeCount"
              className="block text-sm font-medium text-gray-700 mb-2"
            >
              Number of Nodes
            </label>
            <input
              type="number"
              id="nodeCount"
              min="1"
              max="100"
              value={nodeCount}
              onChange={(e) => setNodeCount(parseInt(e.target.value) || 1)}
              className="w-full px-4 py-2.5 bg-white rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none"
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="px-6 py-2.5 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? 'Loading...' : 'Fetch Orderbook'}
          </button>
        </form>
      </div>

      {/* Error Message */}
      {error && (
        <div className="rounded-xl p-4 bg-red-50 mb-6 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 flex-shrink-0" />
          <div>
            <p className="text-red-900 font-medium">Error loading orderbook</p>
            <p className="text-red-700 text-sm mt-1">{error}</p>
          </div>
        </div>
      )}

      {/* Orderbook Stats */}
      {orderbook && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div className="rounded-xl p-4 bg-gray-50">
              <div className="text-sm text-gray-500 mb-1">Spread</div>
              <div className="text-2xl font-bold text-gray-900">
                {orderbook.spread !== null
                  ? formatPrice(orderbook.spread)
                  : 'N/A'}
              </div>
            </div>
            <div className="rounded-xl p-4 bg-gray-50">
              <div className="text-sm text-gray-500 mb-1">Required GPUs</div>
              <div className="text-2xl font-bold text-gray-900">
                {orderbook.metadata.required_gpus}
              </div>
            </div>
            <div className="rounded-xl p-4 bg-gray-50">
              <div className="text-sm text-gray-500 mb-1">Ask Liquidity</div>
              <div className="text-2xl font-bold text-red-600">
                {orderbook.total_ask_liquidity} GPUs
              </div>
            </div>
            <div className="rounded-xl p-4 bg-gray-50">
              <div className="text-sm text-gray-500 mb-1">Bid Liquidity</div>
              <div className="text-2xl font-bold text-green-600">
                {orderbook.total_bid_liquidity} GPUs
              </div>
            </div>
          </div>

          {/* Optimal Price Recommendation */}
          {orderbook.optimal_price !== null && (
            <div className="rounded-xl p-6 bg-green-50 mb-6">
              <div className="flex items-start gap-4">
                <div className="p-3 bg-green-600 rounded-lg">
                  <TrendingUp className="w-6 h-6 text-white" />
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-green-900 mb-2">
                    Recommended Order Price
                  </h3>
                  <p className="text-3xl font-bold text-green-700 mb-2">
                    {formatPrice(orderbook.optimal_price)}
                    <span className="text-base font-normal text-green-600 ml-2">
                      per GPU-hour
                    </span>
                  </p>
                  <p className="text-sm text-green-800">
                    This price level offers the best balance of cost, liquidity, and
                    execution certainty for your {orderbook.metadata.required_gpus}{' '}
                    GPU order. The highlighted row in the asks table shows this
                    optimal entry point.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Orderbook Tables */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Asks (Sellers) */}
            <div className="rounded-xl bg-white overflow-hidden">
              <div className="bg-red-50 px-6 py-4">
                <div className="flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-red-600" />
                  <h2 className="text-lg font-semibold text-red-900">
                    Asks (Sellers)
                  </h2>
                </div>
                <p className="text-sm text-red-700 mt-1">
                  Orders offering compute for sale
                </p>
              </div>
              {renderOrderbookTable(
                orderbook.asks,
                'ask',
                orderbook.optimal_index
              )}
            </div>

            {/* Bids (Buyers) */}
            <div className="rounded-xl bg-white overflow-hidden">
              <div className="bg-green-50 px-6 py-4">
                <div className="flex items-center gap-2">
                  <TrendingDown className="w-5 h-5 text-green-600" />
                  <h2 className="text-lg font-semibold text-green-900">
                    Bids (Buyers)
                  </h2>
                </div>
                <p className="text-sm text-green-700 mt-1">
                  Orders looking to buy compute
                </p>
              </div>
              {renderOrderbookTable(orderbook.bids, 'bid', null)}
            </div>
          </div>

          {/* Timestamp */}
          <div className="mt-4 text-center text-sm text-gray-500">
            Last updated: {new Date(orderbook.last_updated).toLocaleString()}
          </div>
        </>
      )}

      {/* Loading State */}
      {loading && !orderbook && (
        <div className="text-center py-16">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600" />
          <p className="mt-4 text-gray-600">Loading orderbook...</p>
        </div>
      )}

      {/* Empty State */}
      {!loading && !orderbook && !error && (
        <div className="text-center py-16 rounded-xl bg-gray-50">
          <TrendingUp className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-600">
            Enter your requirements and fetch the orderbook to get started
          </p>
        </div>
      )}
    </div>
  );
}

