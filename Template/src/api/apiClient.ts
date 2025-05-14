import axios from 'axios';

const API_URL = 'http://localhost:8000';

// API client for interacting with the backend
const apiClient = {
  // Get list of all available models
  getModels: async () => {
    try {
      const response = await axios.get(`${API_URL}/api/models`);
      return response.data;
    } catch (error) {
      console.error('Error fetching models:', error);
      throw error;
    }
  },

  // Get details of a specific model
  getModelDetails: async (modelId: string) => {
    try {
      const response = await axios.get(`${API_URL}/api/models/${modelId}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching model ${modelId}:`, error);
      throw error;
    }
  },

  // Make prediction with image file
  predictImage: async (modelId: string, file: File) => {
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await axios.post(
        `${API_URL}/api/models/${modelId}/predict`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      return response.data;
    } catch (error) {
      console.error('Error making image prediction:', error);
      throw error;
    }
  },

  // Make prediction with text
  predictText: async (modelId: string, text: string) => {
    try {
      const response = await axios.post(
        `${API_URL}/api/models/${modelId}/predict/text`,
        { text }
      );
      return response.data;
    } catch (error) {
      console.error('Error making text prediction:', error);
      throw error;
    }
  },

  // Make prediction with tabular data
  predictTabular: async (modelId: string, features: Record<string, any>) => {
    try {
      const response = await axios.post(
        `${API_URL}/api/models/${modelId}/predict/tabular`,
        { features }
      );
      return response.data;
    } catch (error) {
      console.error('Error making tabular prediction:', error);
      throw error;
    }
  },

  // Upload a new model
  uploadModel: async (name: string, description: string, inputType: string, modelFile: File) => {
    try {
      const formData = new FormData();
      formData.append('name', name);
      formData.append('description', description);
      formData.append('input_type', inputType);
      formData.append('model_file', modelFile);
      
      const response = await axios.post(
        `${API_URL}/api/models/upload`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      return response.data;
    } catch (error) {
      console.error('Error uploading model:', error);
      throw error;
    }
  },

  // Delete a model
  deleteModel: async (modelId: string) => {
    try {
      const response = await axios.delete(`${API_URL}/api/models/${modelId}`);
      return response.data;
    } catch (error) {
      console.error(`Error deleting model ${modelId}:`, error);
      throw error;
    }
  },

  // Health check
  healthCheck: async () => {
    try {
      const response = await axios.get(`${API_URL}/api/health`);
      return response.data;
    } catch (error) {
      console.error('Error checking API health:', error);
      throw error;
    }
  }
};

export default apiClient;
