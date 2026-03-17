import React from 'react';
import { Satellite, Construction } from 'lucide-react';

const ComingSoonBadge = () => {
  return (
    <div className="glass-panel p-4 flex items-center justify-between border-dashed border-white/30 bg-white/5 mt-8 overflow-hidden relative group">
       <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/0 via-emerald-500/10 to-teal-500/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000 ease-in-out"></div>
       
       <div className="flex items-center gap-4 z-10 w-full">
         <div className="p-3 rounded-full bg-slate-800/80 shadow-inner flex items-center justify-center relative">
            <Satellite className="w-5 h-5 text-emerald-400" />
            <Construction className="w-3 h-3 text-amber-500 absolute bottom-1 right-1" />
         </div>
         <div className="flex-1">
           <h4 className="text-sm font-semibold text-slate-200">Satellite Image Analysis</h4>
           <p className="text-xs text-slate-400 italic">🚧 CNN Integration Coming Soon</p>
         </div>
       </div>
    </div>
  );
}

export default ComingSoonBadge;
