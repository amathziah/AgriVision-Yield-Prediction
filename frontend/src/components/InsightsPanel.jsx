import React from 'react';
import { Lightbulb, TrendingUp, Droplets, ThermometerSun, Activity } from 'lucide-react';

const InsightsPanel = ({ formData, predictionResult }) => {

  const getInsights = () => {
    if (!predictionResult || !predictionResult.insights) {
      return [
        { icon: Lightbulb, text: "Awaiting data input to generate personalized agronomic insights.", color: "text-slate-400" }
      ];
    }

    return predictionResult.insights.map((text, idx) => {
      // Pick an icon based on keywords
      let icon = Lightbulb;
      let color = "text-emerald-400";
      
      const lower = text.toLowerCase();
      if (lower.includes('rain') || lower.includes('precip')) {
        icon = Droplets;
        color = "text-blue-400";
      } else if (lower.includes('temp') || lower.includes('heat')) {
        icon = ThermometerSun;
        color = "text-rose-400";
      } else if (lower.includes('hybrid') || lower.includes('fusion') || lower.includes('model')) {
        icon = Activity;
        color = "text-purple-400";
      } else if (lower.includes('yield') || lower.includes('improve') || lower.includes('potential')) {
        icon = TrendingUp;
        color = "text-teal-400";
      }
      
      return { icon, text, color };
    });
  };

  const insights = getInsights();

  return (
    <div className="glass-panel p-6">
      <h3 className="text-lg font-semibold text-emerald-300 mb-4 flex items-center gap-2">
        <Lightbulb className="w-5 h-5" />
        AI Insights
      </h3>
      
      <div className="space-y-4">
        {insights.map((insight, idx) => {
          const Icon = insight.icon;
          return (
            <div key={idx} className="glass-card flex gap-4 items-start animate-in fade-in slide-in-from-right-4" style={{ animationDelay: `${idx * 150}ms` }}>
              <div className={`p-2 rounded-lg bg-white/5 ${insight.color}`}>
                <Icon className="w-5 h-5" />
              </div>
              <p className="text-sm text-slate-300 leading-relaxed pt-0.5">
                {insight.text}
              </p>
            </div>
          )
        })}
      </div>
    </div>
  );
};

export default InsightsPanel;
