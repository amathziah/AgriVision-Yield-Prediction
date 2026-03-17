import React from 'react';
import { Sprout } from 'lucide-react';

const Navbar = () => {
  return (
    <nav className="glass-panel sticky top-0 z-50 px-6 py-4 flex items-center justify-between mb-8">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-emerald-500/20 rounded-lg">
          <Sprout className="w-6 h-6 text-emerald-400 pl-0.5" />
        </div>
        <h1 className="text-xl font-bold bg-gradient-to-r from-emerald-400 to-teal-300 bg-clip-text text-transparent">
          AgriVision AI
        </h1>
      </div>
      <div className="flex items-center gap-4 text-sm font-medium text-slate-300">
        <span className="flex items-center gap-2">
          <span className="relative flex h-2.5 w-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
          </span>
          System Online
        </span>
      </div>
    </nav>
  );
};

export default Navbar;
