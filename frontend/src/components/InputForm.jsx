import React, { useState } from 'react';

const InputForm = ({ onSubmit, isLoading }) => {
  const [formData, setFormData] = useState({
    country: '',
    crop: '',
    year: '',
    rainfall: '',
    temperature: '',
    pesticides: '',
  });

  const [validationError, setValidationError] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setValidationError('');
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Simple validation
    if (!formData.country || !formData.crop || !formData.year || formData.rainfall === '' || formData.temperature === '' || formData.pesticides === '') {
      setValidationError('Please fill in all fields before analyzing.');
      return;
    }

    const submissionData = {
      country: formData.country,
      crop: formData.crop,
      year: Number(formData.year),
      rainfall: Number(formData.rainfall),
      temperature: Number(formData.temperature),
      pesticides: Number(formData.pesticides)
    };

    onSubmit(submissionData);
  };

  const countries = ['India', 'Brazil', 'USA', 'China', 'France', 'Australia'];
  const crops = ['Wheat', 'Rice', 'Maize', 'Soybeans', 'Potatoes'];

  return (
    <div className="glass-panel p-6 flex flex-col h-full">
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-emerald-300">Parameters</h2>
        <p className="text-slate-400 text-sm mt-1">Configure input for SVR prediction</p>
      </div>

      <form onSubmit={handleSubmit} className="flex-1 space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-slate-300 uppercase tracking-wider">Country</label>
            <select
              name="country"
              value={formData.country}
              onChange={handleChange}
              className="glass-input appearance-none bg-slate-900/50"
            >
              <option value="" disabled>Select Country</option>
              {countries.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-slate-300 uppercase tracking-wider">Crop</label>
            <select
              name="crop"
              value={formData.crop}
              onChange={handleChange}
              className="glass-input appearance-none bg-slate-900/50"
            >
              <option value="" disabled>Select Crop</option>
              {crops.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
        </div>

        <div className="space-y-4 pt-2">
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-slate-300 uppercase tracking-wider flex justify-between">
              <span>Year</span>
              <span className="text-emerald-400/70">{formData.year}</span>
            </label>
            <input
              type="range"
              name="year"
              min="1990"
              max="2030"
              value={formData.year}
              onChange={handleChange}
              className="w-full accent-emerald-500"
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-medium text-slate-300 uppercase tracking-wider flex justify-between">
              <span>Rainfall (mm)</span>
              <span className="text-emerald-400/70">{formData.rainfall}</span>
            </label>
             <input
              type="number"
              name="rainfall"
              value={formData.rainfall}
              onChange={handleChange}
              className="glass-input"
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-medium text-slate-300 uppercase tracking-wider flex justify-between">
              <span>Temperature (°C)</span>
              <span className="text-emerald-400/70">{formData.temperature}</span>
            </label>
            <input
              type="number"
              step="0.1"
              name="temperature"
              value={formData.temperature}
              onChange={handleChange}
              className="glass-input"
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-medium text-slate-300 uppercase tracking-wider flex justify-between">
              <span>Pesticides (tonnes)</span>
              <span className="text-emerald-400/70">{formData.pesticides}</span>
            </label>
             <input
              type="number"
              name="pesticides"
              value={formData.pesticides}
              onChange={handleChange}
              className="glass-input"
              placeholder="e.g. 200"
            />
          </div>
        </div>

        {validationError && (
          <div className="text-red-400 text-sm py-1 font-medium animate-pulse">
            {validationError}
          </div>
        )}

        <div className="pt-6 mt-auto">
          <button
            type="submit"
            disabled={isLoading}
            className="glass-button flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <div className="flex items-center gap-2">
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Processing...
              </div>
            ) : (
              'Analzye Yield Potential'
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

export default InputForm;
