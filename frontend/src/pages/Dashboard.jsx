import React, { useState } from 'react';
import Navbar from '../components/Navbar';
import InputForm from '../components/InputForm';
import PredictionCard from '../components/PredictionCard';
import InsightsPanel from '../components/InsightsPanel';
import Charts from '../components/Charts';
import ComingSoonBadge from '../components/ComingSoonBadge';
import { predictYield } from '../api/predict';
import { AlertCircle } from 'lucide-react';

const Dashboard = () => {
  const [predictionResult, setPredictionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [formData, setFormData] = useState(null);
  const [error, setError] = useState(null);

  const handlePredict = async (data) => {
    setIsLoading(true);
    setFormData(data);
    setError(null);
    setPredictionResult(null);
    
    try {
      const responseData = await predictYield(data);
      setPredictionResult(responseData);
    } catch (err) {
      console.error("Prediction error:", err);
      setError(err.message || 'Failed to connect to the prediction server.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 pb-12 sm:px-6 lg:px-8">
      <Navbar />
      
      {error && (
        <div className="mb-6 p-4 rounded-lg bg-rose-500/10 border border-rose-500/30 flex items-start gap-3 backdrop-blur-sm animate-in fade-in slide-in-from-top-4">
          <AlertCircle className="w-5 h-5 text-rose-400 mt-0.5 flex-shrink-0" />
          <div>
            <h3 className="text-sm font-semibold text-rose-300">Connection Error</h3>
            <p className="text-sm text-rose-200/80 mt-1">{error}</p>
          </div>
        </div>
      )}

      <main className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column - Input Panel */}
        <section className="lg:col-span-4 flex flex-col gap-6">
          <InputForm onSubmit={handlePredict} isLoading={isLoading} />
        </section>
        
        {/* Right Column - Results Dashboard */}
        <section className="lg:col-span-8 flex flex-col gap-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
             {/* Top Right - Core Prediction & Insights */}
             <PredictionCard predictionResult={predictionResult} isLoading={isLoading} />
             <InsightsPanel formData={formData} predictionResult={predictionResult} />
          </div>
          
          {/* Bottom Right - Visualizations & Upcoming */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Charts chartData={predictionResult?.chart_data} />
            <div className="flex flex-col h-full justify-between">
              <div className="glass-panel p-6 flex-1">
                 <h3 className="text-lg font-semibold text-emerald-300 mb-4">Model Pipeline Status</h3>
                 <ul className="space-y-3 text-sm text-slate-300">
                    <li className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-emerald-500"></span> Tabular SVR Engine (Active)</li>
                    <li className="flex items-center gap-2 opacity-50"><span className="w-2 h-2 rounded-full bg-slate-600"></span> Geo-Spatial Preprocessing (Offline)</li>
                    <li className="flex items-center gap-2 opacity-50"><span className="w-2 h-2 rounded-full bg-slate-600"></span> API Gateway (Local Only)</li>
                 </ul>
              </div>
              <ComingSoonBadge />
            </div>
          </div>
        </section>
      </main>
    </div>
  );
};

export default Dashboard;
