import React, { useState, useEffect } from 'react';
import { Upload, X, CheckCircle, Loader2, Image as ImageIcon } from 'lucide-react';
import blobberLogo from '../public/Elegant_Minimalist_Calligraphy_Initials_logo__1_-removebg-preview.png';
import './App.css';

// Define the shape of the data returned by the AI model
interface PredictionResult {
  prediction: string;
  confidence: string; // Stored as string for precision (e.g., "0.9876")
  modelId: string;
  studentId: string;
  name: string;
  regNo: string;
  department: string;
  level: number;
}

// Main App Component (Fully compatible with Vite React setup)
const App: React.FC = () => {
  // Explicitly type the state variables
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [animatedConfidence, setAnimatedConfidence] = useState<number>(0);

  /**
   * Handles file selection, setting the file state and creating a temporary
   * URL for image preview.
   * @param {React.ChangeEvent<HTMLInputElement>} event
   */
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    // We check for event.target.files before accessing [0]
    const selectedFile = event.target.files ? event.target.files[0] : null;

    // Reset states on new file selection
    setResult(null);
    setError(null);
    setFile(selectedFile);

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl); // Clean up previous preview URL
    }

    if (selectedFile) {
      setPreviewUrl(URL.createObjectURL(selectedFile));
    } else {
      setPreviewUrl(null);
    }
  };

  /**
   * Simulates the API call to the backend for AI prediction.
   */
  const handleUpload = async () => {
    if (!file) {
      setError("Please select an image file to analyze.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    // --- START: API Mocking Section ---
    const formData = new FormData();
    formData.append('image', file);

    // Simulate Network Delay and AI Processing
    await new Promise(resolve => setTimeout(resolve, 2000));

    try {
      // Simulate a successful JSON response from the AI model
      const simulatedResponse: PredictionResult = {
        prediction: 'Student',
        confidence: (Math.random() * 0.4 + 0.59).toFixed(4), // Random score between 59% and 99%
        modelId: 'Blobber',
        studentId: '1242534',
        name: 'student',
        regNo: '124323/22',
        department: 'Electronics',
        level: 300, 
      };

      setResult(simulatedResponse);

      // Example of actual API call structure (if this were real):
      /*
      // Assuming your Vite proxy handles routing this to a Python backend:
      const apiResponse = await fetch('/api/predict', {
        method: 'POST',
        body: formData, // FormData sends the file correctly
      });

      if (!apiResponse.ok) {
        throw new Error(`Server responded with status: ${apiResponse.status}`);
      }

      const data: PredictionResult = await apiResponse.json();
      setResult(data);
      */

    } catch (err: unknown) {
      console.error("Upload Error:", err);
      // Safely check if error is an instance of Error
      if (err instanceof Error) {
        setError(`Failed to connect to the AI model server: ${err.message}`);
      } else {
        setError("An unknown error occurred during upload.");
      }
    } finally {
      setLoading(false);
    }
    // --- END: API Mocking Section ---
  };

  /**
   * Clears the selected file and preview.
   */
  const handleClear = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    // Explicitly cast to HTMLInputElement to access .value property
    const fileInput = document.getElementById('file-input') as HTMLInputElement | null;
    if (fileInput) {
        fileInput.value = ''; // Clear input element
    }
  };



  // Animate confidence score
  useEffect(() => {
    if (result) {
      const targetConfidence = parseFloat(result.confidence) * 100;
      setAnimatedConfidence(0);
      const duration = 2000; // 2 seconds
      const steps = 60;
      const increment = targetConfidence / steps;
      let current = 0;
      const timer = setInterval(() => {
        current += increment;
        if (current >= targetConfidence) {
          setAnimatedConfidence(targetConfidence);
          clearInterval(timer);
        } else {
          setAnimatedConfidence(current);
        }
      }, duration / steps);
      return () => clearInterval(timer);
    }
  }, [result]);

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center p-4">
      <div className="w-full max-w-6xl bg-white shadow-2xl rounded-xl p-8 border border-indigo-100 relative">        
        <div className="absolute top-8 left-8 w-30 h-40">
          <img src={blobberLogo} alt="blubbai" className='logo'/>
        </div>
      

        <h1 className="text-3xl font-bold text-center text-indigo-700 mb-6 flex items-center justify-center">
          <ImageIcon className="w-8 h-8 mr-3" />
          Handwriting Analysis Tool
        </h1>
        <p className="text-center text-gray-500 mb-8">
          Upload a handwriting image to identify the student and get confidence scores.
        </p>

        {/* Main Grid Layout */}
        <div className="grid md:grid-cols-2 gap-8">

          {/* Left Side: Upload and Preview */}
          <div className="space-y-6">
            {/* File Input Area */}
            <div className="border-2 border-dashed border-indigo-300 rounded-lg p-6 bg-indigo-50 transition duration-300 hover:border-indigo-500">
              <input
                id="file-input"
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
              />
              <label
                htmlFor="file-input"
                className="cursor-pointer flex flex-col items-center justify-center"
              >
                <Upload className="w-10 h-10 text-indigo-500 mb-2" />
                <p className="text-sm text-indigo-700 font-semibold">
                  {file ? file.name : 'Click to select image or drag and drop'}
                </p>
                <p className="text-xs text-gray-500">
                  (JPG, PNG, up to 10MB recommended)
                </p>
              </label>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-4">
              <button
                onClick={handleUpload}
                disabled={!file || loading}
                className={`flex-1 flex items-center justify-center py-3 px-4 rounded-lg text-white font-medium transition-all duration-300 transform
                  ${!file || loading
                    ? 'bg-indigo-300 cursor-not-allowed scale-95'
                    : 'bg-indigo-600 hover:bg-indigo-700 hover:scale-105 shadow-md hover:shadow-lg active:scale-95'
                  }`}
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  'Run AI Analysis'
                )}
              </button>

              <button
                onClick={handleClear}
                disabled={!file}
                className={`py-3 px-4 rounded-lg border text-sm font-medium transition-all duration-300 transform
                  ${!file
                    ? 'border-gray-200 text-gray-400 cursor-not-allowed scale-95'
                    : 'border-gray-300 text-gray-600 hover:bg-gray-100 hover:scale-105 active:scale-95'
                  }`}
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Image Preview */}
            <div className="bg-gray-100 p-4 rounded-lg shadow-inner flex flex-col items-center">
              <h2 className="text-lg font-semibold text-gray-700 mb-3">Image Preview</h2>
              <div className="w-full h-64 bg-gray-200 rounded-md overflow-hidden flex items-center justify-center">
                {previewUrl ? (
                  <img
                    src={previewUrl}
                    alt="Selected Preview"
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <p className="text-gray-400 text-sm">No image selected</p>
                )}
              </div>
            </div>
          </div>

          {/* Right Side: Results and Student Details */}
          <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
            <h2 className="text-lg font-semibold text-gray-700 mb-4">Analysis Result & Student Details</h2>
            <div className="min-h-96 flex flex-col justify-center items-center">
              {error && (
                <div className="text-red-600 text-center p-4 bg-red-50 rounded-md w-full">
                  <p className="font-semibold">Error:</p>
                  <p className="text-sm">{error}</p>
                </div>
              )}

              {loading && (
                <p className="text-gray-500 flex items-center">
                  <Loader2 className="w-5 h-5 mr-2 animate-spin text-indigo-500" />
                  Running model...
                </p>
              )}

              {result && !loading && (
                <div className="w-full text-left space-y-4">
                  <div className="p-4 bg-green-50 rounded-lg border-l-4 border-green-500">
                    <div className="flex items-center mb-3">
                      <CheckCircle className="w-6 h-6 text-green-600 mr-2" />
                      <h3 className="text-xl font-bold text-green-700">Prediction Found!</h3>
                    </div>
                    <p className="text-sm text-gray-700 mb-2">
                      <span className="font-medium">Predicted Class:</span>
                      <span className="ml-2 font-bold text-indigo-700">{result.prediction}</span>
                    </p>
                    <p className="text-sm text-gray-700 mb-2">
                      <span className="font-medium">Confidence Score:</span>
                      <span className="ml-2 font-bold text-xl text-green-600">{animatedConfidence.toFixed(2)}%</span>
                    </p>

                  </div>

                  <div className="p-4 bg-blue-50 rounded-lg border-l-4 border-blue-500">
                    <h4 className="text-lg font-semibold text-blue-700 mb-3">Student Details</h4>
                    <div className="space-y-1 text-sm text-gray-700">
                      <p><span className="font-medium">Student ID:</span> {result.studentId}</p>
                      <p><span className="font-medium">Name:</span> {result.name}</p>
                      <p><span className="font-medium">Registration No:</span> {result.regNo}</p>
                      <p><span className="font-medium">Department:</span> {result.department}</p>
                      <p><span className="font-medium">Level:</span> {result.level}</p>
                    </div>
                  </div>
                </div>
              )}

              {!file && !loading && !result && !error && (
                <p className="text-gray-400 text-center">Awaiting image upload.</p>
              )}
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default App;
