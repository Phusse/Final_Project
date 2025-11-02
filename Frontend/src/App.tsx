import React, { useState } from 'react';
import { Upload, X, CheckCircle, Loader2, Image as ImageIcon } from 'lucide-react';

// Define the shape of the data returned by the AI model
interface PredictionResult {
  prediction: string;
  confidence: string; // Stored as string for precision (e.g., "0.9876")
  modelId: string;
}

// Main App Component (Fully compatible with Vite React setup)
const App: React.FC = () => {
  // Explicitly type the state variables
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

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
        prediction: 'Golden Retriever',
        confidence: (Math.random() * 0.4 + 0.59).toFixed(4), // Random score between 59% and 99%
        modelId: 'VGG16-FineTuned-v2'
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

  // Convert confidence (0.0 to 1.0) string to percentage string
  const getConfidenceText = (conf: string): string => {
    const percentage = (parseFloat(conf) * 100).toFixed(2);
    return `${percentage}%`;
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center p-4">
      <div className="w-full max-w-2xl bg-white shadow-2xl rounded-xl p-8 border border-indigo-100">

        <h1 className="text-3xl font-bold text-center text-indigo-700 mb-6 flex items-center justify-center">
          <ImageIcon className="w-8 h-8 mr-3" />
          AI Confidence Scorer
        </h1>
        <p className="text-center text-gray-500 mb-8">
          Upload an image to get a prediction and confidence score from the model.
        </p>

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
        <div className="flex gap-4 mt-6">
          <button
            onClick={handleUpload}
            disabled={!file || loading}
            className={`flex-1 flex items-center justify-center py-3 px-4 rounded-lg text-white font-medium transition duration-200
              ${!file || loading
                ? 'bg-indigo-300 cursor-not-allowed'
                : 'bg-indigo-600 hover:bg-indigo-700 shadow-md hover:shadow-lg'
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
            className={`py-3 px-4 rounded-lg border text-sm font-medium transition duration-200
              ${!file
                ? 'border-gray-200 text-gray-400 cursor-not-allowed'
                : 'border-gray-300 text-gray-600 hover:bg-gray-100'
              }`}
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Image Preview and Results Area */}
        <div className="mt-8 grid md:grid-cols-2 gap-6">
          {/* Preview Card */}
          <div className="bg-gray-100 p-4 rounded-lg shadow-inner flex flex-col items-center">
            <h2 className="text-lg font-semibold text-gray-700 mb-3">Image Preview</h2>
            <div className="w-full h-48 bg-gray-200 rounded-md overflow-hidden flex items-center justify-center">
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

          {/* Results Card */}
          <div className="bg-white p-4 rounded-lg shadow-md border border-gray-200">
            <h2 className="text-lg font-semibold text-gray-700 mb-3">Analysis Result</h2>
            <div className="h-48 flex flex-col justify-center items-center">
              {error && (
                <div className="text-red-600 text-center p-3 bg-red-50 rounded-md">
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
                <div className="w-full text-left p-4 bg-green-50 rounded-lg border-l-4 border-green-500">
                  <div className="flex items-center mb-3">
                    <CheckCircle className="w-6 h-6 text-green-600 mr-2" />
                    <h3 className="text-xl font-bold text-green-700">Prediction Found!</h3>
                  </div>
                  <p className="text-sm text-gray-700 mb-2">
                    <span className="font-medium">Predicted Class:</span>
                    <span className="ml-2 font-bold text-indigo-700">{result.prediction}</span>
                  </p>
                  <p className="text-sm text-gray-700">
                    <span className="font-medium">Confidence Score:</span>
                    <span className="ml-2 font-bold text-xl text-green-600">{getConfidenceText(result.confidence)}</span>
                  </p>
                  <p className="text-xs text-gray-500 mt-2">
                    Model: {result.modelId}
                  </p>
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
