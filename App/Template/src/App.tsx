import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar.tsx';
import HomePage from './pages/HomePage.tsx';
import ModelsPage from './pages/ModelsPage.tsx';
import ModelDetailPage from './pages/ModelDetailPage.tsx';
import PredictionPage from './pages/PredictionPage.tsx';
import UploadModelPage from './pages/UploadModelPage.tsx';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/models" element={<ModelsPage />} />
            <Route path="/models/:modelId" element={<ModelDetailPage />} />
            <Route path="/models/:modelId/predict" element={<PredictionPage />} />
            <Route path="/upload" element={<UploadModelPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
