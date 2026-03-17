import React from 'react';
import { Satellite, Construction } from 'lucide-react';

const LiveStatusBadge = () => {
  return (
    <div className="glass-panel p-4 flex items-center justify-between border-emerald-500/30 bg-emerald-500/5 mt-8 overflow-hidden relative group">
       <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/0 via-emerald-500/20 to-teal-500/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000 ease-in-out"></div>
       
       <div className="flex items-center gap-4 z-10 w-full">
         <div className="p-3 rounded-full bg-slate-800/80 shadow-inner flex items-center justify-center relative">
            <Satellite className="w-5 h-5 text-emerald-400 animate-pulse" />
         </div>
         <div className="flex-1">
           <h4 className="text-sm font-semibold text-emerald-300">Hybrid System Active</h4>
           <p className="text-xs text-slate-400 italic">Multi-modal CNN + SVR inference enabled</p>
         </div>
       </div>
    </div>
  );
}

export default LiveStatusBadge;
