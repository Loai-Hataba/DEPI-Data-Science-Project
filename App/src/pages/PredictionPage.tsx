import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import apiClient from '../api/apiClient.ts'

interface Model {
  id: string;
  name: string;
  description: string;
  input_type: string;
  supported_formats?: string[];
  expected_features?: string[]; // <-- ADD THIS LINE
}

interface PredictionResponse {
  prediction: string;
  confidence: number;
  processing_time: number;
}

const PredictionPage: React.FC = () => {
  const { modelId } = useParams<{ modelId: string }>();
  const [model, setModel] = useState<Model | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [predicting, setPredicting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [text, setText] = useState<string>('');
  const [features, setFeatures] = useState<Record<string, any>>({});
  const [featureKeys, setFeatureKeys] = useState<string[]>(['feature1', 'feature2', 'feature3']);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  useEffect(() => {
    const fetchModelDetails = async () => {
      if (!modelId) return;
      
      try {
        setLoading(true);
        const data = await apiClient.getModelDetails(modelId);
        setModel(data);
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

  const handleFeatureChange = (key: string, value: string) => {
    setFeatures(prev => ({
      ...prev,
      [key]: isNaN(Number(value)) ? value : Number(value)
    }));
  };

  const handleAddFeature = () => {
    const newKey = `feature${featureKeys.length + 1}`;
    setFeatureKeys(prev => [...prev, newKey]);
  };

  const handleRemoveFeature = (keyToRemove: string) => {
    setFeatureKeys(prev => prev.filter(key => key !== keyToRemove));
    setFeatures(prev => {
      const newFeatures = { ...prev };
      delete newFeatures[keyToRemove];
      return newFeatures;
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!model || !modelId) return;

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
    } catch (err) {
      setError('Failed to make prediction. Please check your input and try again.');
      console.error('Error making prediction:', err);
    } finally {
      setPredicting(false);
    }
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
            <div className="flex justify-between items-center mb-4">
              <label className="block text-sm font-medium text-gray-700">
                Enter Features
              </label>
              <button
                type="button"
                onClick={handleAddFeature}
                className="inline-flex items-center px-3 py-1 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              >
                Add Feature
              </button>
            </div>
            <div className="space-y-3">
              {featureKeys.map((key) => (
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
                      onChange={(e) => handleFeatureChange(key, e.target.value)}
                      className="shadow-sm focus:ring-indigo-500 focus:border-indigo-500 block w-full sm:text-sm border-gray-300 rounded-md"
                    />
                    <button
                      type="button"
                      onClick={() => handleRemoveFeature(key)}
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
