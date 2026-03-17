import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const Charts = ({ chartData }) => {
  const displayData = chartData || [
    { year: '2020', actual: 4.2, predicted: 4.1 },
    { year: '2021', actual: 4.4, predicted: 4.5 },
    { year: '2022', actual: 4.7, predicted: 4.6 },
    { year: '2023', actual: 4.9, predicted: 5.0 },
    { year: '2024', actual: 5.1, predicted: 5.2 },
  ];

  return (
    <div className="glass-panel p-6 h-full flex flex-col">
      <h3 className="text-lg font-semibold text-emerald-300 mb-6">Yield Forecast Trend</h3>
      <div className="flex-1 w-full min-h-[250px] relative">
         <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={displayData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="colorPredicted" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
              </linearGradient>
              <linearGradient id="colorActual" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#94a3b8" stopOpacity={0.2}/>
                <stop offset="95%" stopColor="#94a3b8" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#ffffff15" vertical={false} />
            <XAxis dataKey="year" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 12 }} axisLine={false} tickLine={false} />
            <YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 12 }} axisLine={false} tickLine={false} />
            <Tooltip content={<CustomTooltip />} />
            <Area type="monotone" dataKey="actual" stroke="#94a3b8" strokeWidth={2} fillOpacity={1} fill="url(#colorActual)" />
            <Area type="monotone" dataKey="predicted" stroke="#10b981" strokeWidth={2} fillOpacity={1} fill="url(#colorPredicted)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default Charts;
