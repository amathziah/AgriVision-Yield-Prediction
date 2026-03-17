import React from 'react';
import { Target, Info, Activity } from 'lucide-react';
import clsx from 'clsx';

const PredictionCard = ({ predictionResult, isLoading }) => {
  return (
    <div className="glass-panel p-6 flex flex-col items-center justify-center relative overflow-hidden min-h-[300px]">
      {/* Decorative background blur */}
      <div className="absolute top-0 right-0 w-64 h-64 bg-emerald-500/10 rounded-full blur-3xl -mr-32 -mt-32"></div>
      <div className="absolute bottom-0 left-0 w-64 h-64 bg-teal-500/10 rounded-full blur-3xl -ml-32 -mb-32"></div>

      {isLoading ? (
        <div className="flex flex-col items-center justify-center space-y-4 z-10 transition-all">
           <div className="relative">
             <div className="w-20 h-20 border-4 border-emerald-500/20 rounded-full"></div>
             <div className="w-20 h-20 border-4 border-emerald-400 rounded-full border-t-transparent animate-spin absolute top-0 left-0"></div>
           </div>
           <p className="text-emerald-300 font-medium animate-pulse">Running SVR Pipeline...</p>
        </div>
      ) : predictionResult ? (
        <div className="z-10 text-center w-full animate-in fade-in slide-in-from-bottom-4 duration-700">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/20 border border-emerald-500/30 text-emerald-300 text-xs font-semibold uppercase tracking-widest mb-6">
            <Target className="w-3 h-3" />
            Predicted Yield
          </div>
          
          <div className="flex items-baseline justify-center gap-2 mb-2">
            <h3 className="text-6xl font-bold text-white tracking-tight">
              {predictionResult.prediction.toFixed(2)}
            </h3>
            <span className="text-xl text-emerald-400/80 font-medium">t/ha</span>
          </div>

          <div className="mt-8 grid grid-cols-2 gap-4 text-left">
            <div className="glass-card flex items-start gap-3">
              <Activity className="w-5 h-5 text-emerald-400 mt-0.5" />
              <div>
                <p className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold mb-1">Model Used</p>
                <p className="text-sm text-slate-200 font-medium">
                  {predictionResult.model || 'SVR RBF Tuned'}
                </p>
              </div>
            </div>
            <div className="glass-card flex items-start gap-3">
              <Info className="w-5 h-5 text-teal-400 mt-0.5" />
              <div>
                <p className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold mb-1">Confidence Score</p>
                <p className="text-sm text-slate-200 font-medium">
                  {predictionResult.confidence ? `${(predictionResult.confidence * 100).toFixed(1)}%` : '± 5.73 t/ha'}
                </p>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="z-10 text-center text-slate-400 flex flex-col items-center max-w-xs">
          <Target className="w-12 h-12 mb-4 text-slate-600" />
          <p className="text-sm">Enter parameters and run analysis to view the predicted crop yield.</p>
        </div>
      )}
    </div>
  );
};

export default PredictionCard;
