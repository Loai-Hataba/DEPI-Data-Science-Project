# ML Web Application Documentation

## Overview
This documentation provides instructions for using, extending, and deploying the ML Web Application. The application consists of a FastAPI backend and a React frontend, designed to integrate and serve machine learning models through a web interface.

## Architecture
The application follows a client-server architecture:
- **Backend**: FastAPI application that provides API endpoints for model management and prediction
- **Frontend**: React application with TypeScript and Tailwind CSS that provides a user interface
- **ML Model Integration**: Extensible system for integrating various types of machine learning models

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 14+
- npm or yarn

### Installation

#### Backend Setup
1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Install the required dependencies:
   ```
   pip install fastapi uvicorn python-multipart
   ```

3. Start the backend server:
   ```
   python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

#### Frontend Setup
1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install the required dependencies:
   ```
   npm install
   ```

3. Start the frontend development server:
   ```
   npm start
   ```

4. Access the application at `http://localhost:3000`

## Backend API

### Endpoints

#### Model Management
- `GET /api/models`: Get a list of all available models
- `GET /api/models/{model_id}`: Get details of a specific model
- `POST /api/models/upload`: Upload a new model
- `DELETE /api/models/{model_id}`: Delete a model

#### Predictions
- `POST /api/models/{model_id}/predict`: Make a prediction with an image model
- `POST /api/models/{model_id}/predict/text`: Make a prediction with a text model
- `POST /api/models/{model_id}/predict/tabular`: Make a prediction with a tabular data model

#### Health Check
- `GET /api/health`: Check the health status of the API

## Integrating ML Models

The application supports three types of ML models:
1. **Image Classification Models**: For processing image inputs
2. **Text Analysis Models**: For processing text inputs
3. **Tabular Data Models**: For processing structured data inputs

### Adding a Custom Model

To add a custom model, follow these steps:

1. Extend the appropriate base class:
   ```python
   from main import ImageClassificationModel, TextAnalysisModel, TabularModel
   
   class MyCustomImageModel(ImageClassificationModel):
       def __init__(self, model_id, name, description):
           super().__init__(model_id, name, description)
       
       def load(self):
           # Load your model here
           self.model = load_your_model()
           return True
       
       def predict(self, image_path):
           # Implement prediction logic
           result = self.model.predict(preprocess_image(image_path))
           return {
               "prediction": result.label,
               "confidence": result.confidence
           }
   ```

2. Register your model with the ModelManager:
   ```python
   from main import model_manager
   
   custom_model = MyCustomImageModel(
       model_id="my-custom-model",
       name="My Custom Image Model",
       description="A custom image classification model"
   )
   
   model_manager.add_model(custom_model)
   ```

## Deployment

### Production Deployment

#### Backend Deployment
1. Install production dependencies:
   ```
   pip install gunicorn
   ```

2. Create a `Procfile` or systemd service file to run:
   ```
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
   ```

#### Frontend Deployment
1. Build the production version of the frontend:
   ```
   cd frontend
   npm run build
   ```

2. Serve the static files using a web server like Nginx or serve them directly from the FastAPI application by uncommenting the following line in `main.py`:
   ```python
   app.mount("/", StaticFiles(directory="../frontend/build", html=True), name="frontend")
   ```

### Docker Deployment
For containerized deployment, you can use the following Dockerfile:

```dockerfile
# Backend Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```dockerfile
# Frontend Dockerfile
FROM node:14-alpine as build

WORKDIR /app

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## Extending the Application

### Adding New API Endpoints
To add new API endpoints, modify the `main.py` file in the backend directory:

```python
@app.get("/api/custom-endpoint")
async def custom_endpoint():
    return {"message": "This is a custom endpoint"}
```

### Adding New Frontend Components
To add new frontend components, create new files in the `src/components` directory:

```tsx
import React from 'react';

const CustomComponent: React.FC = () => {
  return (
    <div className="p-4 bg-white shadow rounded-lg">
      <h2 className="text-xl font-bold">Custom Component</h2>
      <p className="mt-2 text-gray-600">This is a custom component</p>
    </div>
  );
};

export default CustomComponent;
```

## Troubleshooting

### Backend Issues
- **Port already in use**: Kill the process using the port or change the port number
- **Module not found**: Ensure all dependencies are installed
- **Model loading errors**: Check model file paths and formats

### Frontend Issues
- **Build errors**: Check for syntax errors in your components
- **API connection issues**: Verify the API URL in the frontend configuration
- **Styling issues**: Ensure Tailwind CSS is properly configured

## Conclusion
This ML Web Application provides a flexible and extensible platform for integrating and serving machine learning models through a web interface. By following the documentation, you can customize and extend the application to suit your specific needs.
