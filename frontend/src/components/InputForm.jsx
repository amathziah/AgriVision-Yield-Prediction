import React, { useState } from 'react';

const InputForm = ({ onSubmit, isLoading }) => {
  const [formData, setFormData] = useState({
    country: 'Afghanistan',
    crop: 'Maize',
    year: '2024',
    rainfall: '1030',
    temperature: '15',
    pesticides: '1500',
    model_type: 'hybrid',
    image: null,
  });

  const [validationError, setValidationError] = useState('');
  const [preview, setPreview] = useState(null);

  const handleChange = (e) => {
    const { name, value, type, files } = e.target;
    setValidationError('');
    
    if (type === 'file') {
      const file = files[0];
      setFormData((prev) => ({ ...prev, [name]: file }));
      if (file) {
        const reader = new FileReader();
        reader.onloadend = () => setPreview(reader.result);
        reader.readAsDataURL(file);
      }
    } else {
      setFormData((prev) => ({
        ...prev,
        [name]: value,
      }));
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!formData.country || !formData.crop || !formData.year || formData.rainfall === '' || formData.temperature === '' || formData.pesticides === '') {
      setValidationError('Please fill in all tabular fields.');
      return;
    }

    if ((formData.model_type === 'cnn' || formData.model_type === 'hybrid') && !formData.image) {
      setValidationError('Please upload a satellite image for this model type.');
      return;
    }

    const data = new FormData();
    data.append('country', formData.country);
    data.append('crop', formData.crop);
    data.append('year', formData.year);
    data.append('rainfall', formData.rainfall);
    data.append('temperature', formData.temperature);
    data.append('pesticides', formData.pesticides);
    data.append('model_type', formData.model_type);
    if (formData.image) {
      data.append('image', formData.image);
    }

    onSubmit(data);
  };

  const countries = ['Afghanistan', 'India', 'Brazil', 'USA', 'China', 'France', 'Australia'];
  const crops = ['Maize', 'Wheat', 'Rice', 'Soybeans', 'Potatoes'];

  return (
    <div className="glass-panel p-6 flex flex-col h-full">
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-emerald-300">Analysis Engine</h2>
        <p className="text-slate-400 text-sm mt-1">Configure parameters & sensor data</p>
      </div>

      <form onSubmit={handleSubmit} className="flex-1 space-y-4">
        {/* Model Selection */}
        <div className="space-y-1.5 pb-2 border-b border-slate-700/50">
          <label className="text-xs font-medium text-slate-300 uppercase tracking-wider">Prediction Strategy</label>
          <div className="grid grid-cols-3 gap-2">
            {['svr', 'cnn', 'hybrid'].map((type) => (
              <button
                key={type}
                type="button"
                onClick={() => setFormData(prev => ({ ...prev, model_type: type }))}
                className={`py-2 text-xs font-bold rounded-md transition-all ${
                  formData.model_type === type 
                    ? 'bg-emerald-500 text-white shadow-lg shadow-emerald-500/20' 
                    : 'bg-slate-800/50 text-slate-400 hover:bg-slate-800'
                }`}
              >
                {type.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

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

        {/* Image Upload (Conditional) */}
        {(formData.model_type === 'cnn' || formData.model_type === 'hybrid') && (
          <div className="space-y-1.5 animate-in fade-in slide-in-from-left-4 duration-300">
            <label className="text-xs font-medium text-emerald-400 uppercase tracking-wider">Satellite Imagery (Required)</label>
            <div className="relative group">
              <input
                type="file"
                name="image"
                accept="image/*"
                onChange={handleChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
              />
              <div className={`p-4 border-2 border-dashed rounded-lg flex flex-col items-center justify-center transition-all ${
                formData.image ? 'border-emerald-500/50 bg-emerald-500/5' : 'border-slate-700 bg-slate-800/30 group-hover:border-slate-500'
              }`}>
                {preview ? (
                  <img src={preview} alt="Preview" className="h-20 w-auto rounded object-cover mb-2" />
                ) : (
                  <svg className="w-8 h-8 text-slate-500 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                )}
                <span className="text-xs text-slate-400">{formData.image ? formData.image.name : 'Upload NIR/RGB Imagery'}</span>
              </div>
            </div>
          </div>
        )}

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

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-slate-300 uppercase tracking-wider">Rainfall (mm)</label>
              <input
                type="number"
                name="rainfall"
                value={formData.rainfall}
                onChange={handleChange}
                className="glass-input"
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-slate-300 uppercase tracking-wider">Temp (°C)</label>
              <input
                type="number"
                step="0.1"
                name="temperature"
                value={formData.temperature}
                onChange={handleChange}
                className="glass-input"
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-medium text-slate-300 uppercase tracking-wider">Pesticides (tonnes)</label>
            <input
              type="number"
              name="pesticides"
              value={formData.pesticides}
              onChange={handleChange}
              className="glass-input"
            />
          </div>
        </div>

        {validationError && (
          <div className="text-rose-400 text-sm py-1 font-medium animate-pulse flex items-center gap-2">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            {validationError}
          </div>
        )}

        <div className="pt-4 mt-auto">
          <button
            type="submit"
            disabled={isLoading}
            className={`glass-button flex items-center justify-center gap-2 ${
              isLoading ? 'opacity-70 cursor-not-allowed' : 'btn-glow active:scale-95'
            }`}
          >
            {isLoading ? (
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                Quantifying Yield...
              </div>
            ) : (
              'Execute Inference'
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

export default InputForm;
