import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import apiClient from '../api/apiClient.ts'

interface Model {
  id: string;
  name: string;
  description: string;
  input_type: string;
  supported_formats?: string[];
}

interface PredictionResponse {
  prediction: string;
  confidence: number;
  processing_time: number;
  class_probabilities?: Record<string, number>;
}

// Define the feature field types
interface FeatureField {
  name: string;
  label: string;
  type: "text" | "number" | "select";
  options?: string[];
  min?: number;
  max?: number;
}

// Define model-specific feature fields
const MODEL_FEATURE_FIELDS: Record<string, FeatureField[]> = {
  "schizo-predictor": [
    { name: "Age", label: "Age", type: "number", min: 1, max: 120 },
    { name: "Gender", label: "Gender", type: "select", options: ["Male", "Female"] },
    { name: "Marital_Status", label: "Marital Status", type: "select", options: ["Single", "Married", "Divorced", "Widowed"] },
    { name: "Fatigue", label: "Fatigue Level", type: "number", min: 1, max: 5 },
    { name: "Slowing", label: "Movement Slowing", type: "number", min: 1, max: 5 },
    { name: "Pain", label: "Pain Level", type: "number", min: 1, max: 5 },
    { name: "Hygiene", label: "Hygiene Level", type: "number", min: 1, max: 5 },
    { name: "Movement", label: "Movement Quality", type: "number", min: 1, max: 5 },
  ],
  // Add other model-specific fields here
  "cardio-predictor": [
    { name: "age", label: "Age", type: "number" },
    { name: "gender", label: "Gender", type: "select", options: ["1", "2"] },
    { name: "height", label: "Height (cm)", type: "number" },
    { name: "weight", label: "Weight (kg)", type: "number" },
    { name: "ap_hi", label: "Systolic Blood Pressure", type: "number" },
    { name: "ap_lo", label: "Diastolic Blood Pressure", type: "number" },
    { name: "cholesterol", label: "Cholesterol", type: "select", options: ["1", "2", "3"] },
    { name: "gluc", label: "Glucose", type: "select", options: ["1", "2", "3"] },
    { name: "smoke", label: "Smoking", type: "select", options: ["0", "1"] },
    { name: "alco", label: "Alcohol", type: "select", options: ["0", "1"] },
    { name: "active", label: "Physical Activity", type: "select", options: ["0", "1"] },
  ],
  "diabetes-predictor": [
    { name: "Pregnancies", label: "Pregnancies", type: "number" },
    { name: "Glucose", label: "Glucose", type: "number" },
    { name: "BloodPressure", label: "Blood Pressure", type: "number" },
    { name: "SkinThickness", label: "Skin Thickness", type: "number" },
    { name: "Insulin", label: "Insulin", type: "number" },
    { name: "BMI", label: "BMI", type: "number" },
    { name: "DiabetesPedigreeFunction", label: "Diabetes Pedigree Function", type: "number" },
    { name: "Age", label: "Age", type: "number" },
  ],
  "asthma-predictor": [
    // Add the asthma model fields once you have them
  ]
};

const PredictionPage: React.FC = () => {
  const { modelId } = useParams<{ modelId: string }>();
  const [model, setModel] = useState<Model | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [predicting, setPredicting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [text, setText] = useState<string>('');
  const [features, setFeatures] = useState<Record<string, any>>({});
  const [customFeatureKeys, setCustomFeatureKeys] = useState<string[]>([]);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [requiredFeatures, setRequiredFeatures] = useState<string[]>([]);

  useEffect(() => {
    const fetchModelDetails = async () => {
      if (!modelId) return;
      
      try {
        setLoading(true);
        const data = await apiClient.getModelDetails(modelId);
        setModel(data);
        setFeatures({}); // Reset features when model changes
        setError(null);
      } catch (err) {
        setError('Failed to fetch model details. Please try again later.');
        console.error('Error fetching model details:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchModelDetails();
  }, [modelId]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      
      // Create preview URL for image
      if (selectedFile.type.startsWith('image/')) {
        const url = URL.createObjectURL(selectedFile);
        setPreviewUrl(url);
      } else {
        setPreviewUrl(null);
      }
    }
  };

  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value);
  };

  const handleFeatureChange = (key: string, value: any) => {
    setFeatures(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const handleAddCustomFeature = () => {
    const newKey = `feature${customFeatureKeys.length + 1}`;
    setCustomFeatureKeys(prev => [...prev, newKey]);
  };

  const handleRemoveCustomFeature = (keyToRemove: string) => {
    setCustomFeatureKeys(prev => prev.filter(key => key !== keyToRemove));
    setFeatures(prev => {
      const newFeatures = { ...prev };
      delete newFeatures[keyToRemove];
      return newFeatures;
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!model || !modelId) return;
    
    setRequiredFeatures([]);
    
    try {
      setPredicting(true);
      setError(null);
      
      let response;
      
      if (model.input_type === 'image' && file) {
        response = await apiClient.predictImage(modelId, file);
      } else if (model.input_type === 'text' && text) {
        response = await apiClient.predictText(modelId, text);
      } else if (model.input_type === 'tabular' && Object.keys(features).length > 0) {
        response = await apiClient.predictTabular(modelId, features);
      } else {
        throw new Error('Invalid input for model type');
      }
      
      setResult(response);
    } catch (err: any) {
      if (err.response?.data?.required_features) {
        setRequiredFeatures(err.response.data.required_features);
        setError(`Missing required features: ${err.response.data.detail}`);
      } else {
        setError('Failed to make prediction. Please check your input and try again.');
      }
      console.error('Error making prediction:', err);
    } finally {
      setPredicting(false);
    }
  };

  const renderModelSpecificFields = () => {
    if (!model || !modelId) return null;
    
    const fields = MODEL_FEATURE_FIELDS[modelId];
    if (!fields || fields.length === 0) return null;
    
    return (
      <div className="space-y-4">
        <h3 className="text-lg font-medium text-gray-700">Model-Specific Features</h3>
        {fields.map((field) => (
          <div key={field.name} className="sm:grid sm:grid-cols-3 sm:gap-4 sm:items-start">
            <label htmlFor={field.name} className="block text-sm font-medium text-gray-700 sm:mt-px sm:pt-2">
              {field.label}
              <span className="text-red-500">*</span>
            </label>
            <div className="mt-1 sm:mt-0 sm:col-span-2">
              {field.type === 'select' ? (
                <select
                  id={field.name}
                  name={field.name}
                  value={features[field.name] || ''}
                  onChange={(e) => handleFeatureChange(field.name, e.target.value)}
                  className={`max-w-lg block w-full shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:max-w-xs sm:text-sm border-gray-300 rounded-md ${
                    requiredFeatures.includes(field.name) ? 'border-red-500 ring-1 ring-red-500' : ''
                  }`}
                >
                  <option value="">Select {field.label}</option>
                  {field.options?.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="number"
                  id={field.name}
                  name={field.name}
                  min={field.min}
                  max={field.max}
                  value={features[field.name] || ''}
                  onChange={(e) => handleFeatureChange(field.name, field.type === 'number' ? Number(e.target.value) : e.target.value)}
                  className={`max-w-lg block w-full shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:max-w-xs sm:text-sm border-gray-300 rounded-md ${
                    requiredFeatures.includes(field.name) ? 'border-red-500 ring-1 ring-red-500' : ''
                  }`}
                />
              )}
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderCustomFeatureFields = () => {
    if (!model || model.input_type !== 'tabular') return null;
    
    return (
      <div className="mt-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-medium text-gray-700">Custom Features</h3>
          <button
            type="button"
            onClick={handleAddCustomFeature}
            className="inline-flex items-center px-3 py-1 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          >
            Add Custom Feature
          </button>
        </div>
        
        {customFeatureKeys.length > 0 ? (
          <div className="space-y-3">
            {customFeatureKeys.map((key) => (
              <div key={key} className="flex items-center space-x-2">
                <div className="w-1/3">
                  <input
                    type="text"
                    value={key}
                    disabled
                    className="shadow-sm focus:ring-indigo-500 focus:border-indigo-500 block w-full sm:text-sm border-gray-300 rounded-md bg-gray-50"
                  />
                </div>
                <div className="w-2/3 flex items-center space-x-2">
                  <input
                    type="text"
                    placeholder="Value"
                    value={features[key] || ''}
                    onChange={(e) => handleFeatureChange(key, isNaN(Number(e.target.value)) ? e.target.value : Number(e.target.value))}
                    className="shadow-sm focus:ring-indigo-500 focus:border-indigo-500 block w-full sm:text-sm border-gray-300 rounded-md"
                  />
                  <button
                    type="button"
                    onClick={() => handleRemoveCustomFeature(key)}
                    className="inline-flex items-center p-1 border border-transparent rounded-full shadow-sm text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                  >
                    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-gray-500">No custom features added yet.</p>
        )}
      </div>
    );
  };

  const renderInputForm = () => {
    if (!model) return null;
    
    switch (model.input_type) {
      case 'image':
        return (
          <div>
            <label className="block text-sm font-medium text-gray-700">Upload Image</label>
            <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md">
              <div className="space-y-1 text-center">
                {previewUrl ? (
                  <div className="mb-4">
                    <img src={previewUrl} alt="Preview" className="mx-auto h-64 object-contain" />
                  </div>
                ) : (
                  <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                    <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                )}
                <div className="flex text-sm text-gray-600">
                  <label htmlFor="file-upload" className="relative cursor-pointer bg-white rounded-md font-medium text-indigo-600 hover:text-indigo-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-indigo-500">
                    <span>Upload a file</span>
                    <input id="file-upload" name="file-upload" type="file" className="sr-only" onChange={handleFileChange} accept="image/*" />
                  </label>
                  <p className="pl-1">or drag and drop</p>
                </div>
                <p className="text-xs text-gray-500">
                  {model.supported_formats 
                    ? `Supported formats: ${model.supported_formats.join(', ')}` 
                    : 'PNG, JPG, GIF up to 10MB'}
                </p>
              </div>
            </div>
          </div>
        );
      
      case 'text':
        return (
          <div>
            <label htmlFor="text-input" className="block text-sm font-medium text-gray-700">
              Enter Text
            </label>
            <div className="mt-1">
              <textarea
                id="text-input"
                name="text-input"
                rows={4}
                className="shadow-sm focus:ring-indigo-500 focus:border-indigo-500 block w-full sm:text-sm border-gray-300 rounded-md"
                placeholder="Enter text for analysis..."
                value={text}
                onChange={handleTextChange}
              />
            </div>
          </div>
        );
      
      case 'tabular':
        return (
          <div>
            {renderModelSpecificFields()}
            {renderCustomFeatureFields()}
          </div>
        );
      
      default:
        return <p>Unsupported model type</p>;
    }
  };

  const renderResult = () => {
    if (!result) return null;
    
    return (
      <div className="mt-8 bg-white shadow overflow-hidden sm:rounded-lg">
        <div className="px-4 py-5 sm:px-6 bg-green-50">
          <h3 className="text-lg leading-6 font-medium text-gray-900">Prediction Result</h3>
          <p className="mt-1 max-w-2xl text-sm text-gray-500">
            Processed in {result.processing_time.toFixed(3)} seconds
          </p>
        </div>
        <div className="border-t border-gray-200">
          <dl>
            <div className="bg-white px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
              <dt className="text-sm font-medium text-gray-500">Prediction</dt>
              <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2 font-semibold">
                {result.prediction}
              </dd>
            </div>
            <div className="bg-gray-50 px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
              <dt className="text-sm font-medium text-gray-500">Confidence</dt>
              <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                <div className="flex items-center">
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div 
                      className="bg-indigo-600 h-2.5 rounded-full" 
                      style={{ width: `${result.confidence * 100}%` }}
                    ></div>
                  </div>
                  <span className="ml-2">{(result.confidence * 100).toFixed(1)}%</span>
                </div>
              </dd>
            </div>
            
            {/* Display class probabilities if available */}
            {result.class_probabilities && (
              <div className="bg-white px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
                <dt className="text-sm font-medium text-gray-500">Class Probabilities</dt>
                <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
                  <ul className="space-y-2">
                    {Object.entries(result.class_probabilities).map(([className, prob]) => (
                      <li key={className} className="flex items-center justify-between">
                        <span>{className}</span>
                        <div className="flex items-center w-2/3">
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div 
                              className="bg-blue-600 h-2 rounded-full" 
                              style={{ width: `${prob * 100}%` }}
                            ></div>
                          </div>
                          <span className="ml-2 text-xs">{(prob * 100).toFixed(1)}%</span>
                        </div>
                      </li>
                    ))}
                  </ul>
                </dd>
              </div>
            )}
          </dl>
        </div>
      </div>
    );
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="py-8">
        {loading ? (
          <div className="flex justify-center">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-500"></div>
          </div>
        ) : error ? (
          <div className="bg-red-50 p-4 rounded-md">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">{error}</h3>
              </div>
            </div>
          </div>
        ) : model ? (
          <div>
            <div className="md:flex md:items-center md:justify-between">
              <div className="flex-1 min-w-0">
                <h2 className="text-2xl font-bold leading-7 text-gray-900 sm:text-3xl sm:truncate">
                  Make Prediction with {model.name}
                </h2>
                <p className="mt-1 text-sm text-gray-500">{model.description}</p>
              </div>
              <div className="mt-4 flex md:mt-0 md:ml-4">
                <Link
                  to={`/models/${model.id}`}
                  className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                >
                  Back to Model Details
                </Link>
              </div>
            </div>
            <div className="mt-8 bg-white shadow overflow-hidden sm:rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <form onSubmit={handleSubmit}>
                  {renderInputForm()}
                  
                  <div className="mt-6">
                    <button
                      type="submit"
                      disabled={predicting || (
                        (model.input_type === 'image' && !file) ||
                        (model.input_type === 'text' && !text) ||
                        (model.input_type === 'tabular' && Object.keys(features).length === 0)
                      )}
                      className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-gray-300 disabled:cursor-not-allowed"
                    >
                      {predicting ? (
                        <>
                          <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                          </svg>
                          Processing...
                        </>
                      ) : (
                        'Make Prediction'
                      )}
                    </button>
                  </div>
                </form>
              </div>
            </div>
            {renderResult()}
          </div>
        ) : (
          <div className="text-center py-12">
            <p className="text-gray-500">Model not found.</p>
            <Link to="/models" className="mt-4 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700">
              Back to Models
            </Link>
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictionPage;