import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { uploadDocuments, captureCamera, submitCalibration, rotatePage } from '../api';
import { useToast } from '../hooks/useToast';
import BboxHighlightViewer from '../components/BboxHighlightViewer';
import CameraCapture from '../components/CameraCapture';

const DEFAULT_TEXT = 'The quick brown fox jumps over the lazy gray dog.';

export default function Calibrate() {
  const navigate = useNavigate();
  const toast = useToast();
  const queryClient = useQueryClient();
  const fileRef = useRef(null);

  const [step, setStep] = useState('upload'); // upload | define | done
  const [page, setPage] = useState(null); // { id, image_url }
  const [bbox, setBbox] = useState(null);
  const [groundTruth, setGroundTruth] = useState(DEFAULT_TEXT);
  const [showCamera, setShowCamera] = useState(false);

  // ── Upload step ──

  const uploadMutation = useMutation({
    mutationFn: uploadDocuments,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      const p = data?.pages?.[0];
      if (p) {
        setPage({ id: p.id, image_url: p.image_url });
        setStep('define');
      }
    },
    onError: () => toast.error('Upload failed.'),
  });

  const cameraMutation = useMutation({
    mutationFn: captureCamera,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      setShowCamera(false);
      const p = data?.pages?.[0];
      if (p) {
        setPage({ id: p.id, image_url: p.image_url });
        setStep('define');
      }
    },
    onError: () => toast.error('Camera capture failed.'),
  });

  const handleFiles = (e) => {
    const files = e.target.files;
    if (files?.length) uploadMutation.mutate(Array.from(files));
  };

  // ── Define step ──

  const rotateMutation = useMutation({
    mutationFn: () => {
      return rotatePage(page.id, 90);
    },
    onSuccess: (data) => {
      // Backend returns new page with updated image_url (file renamed on rotate)
      if (data?.image_url) {
        setPage(prev => ({ ...prev, image_url: data.image_url }));
      }
      setBbox(null);
      toast.success('Rotated');
    },
    onError: () => toast.error('Rotate failed.'),
  });

  // ── Submit ──

  const calibrateMutation = useMutation({
    mutationFn: () => submitCalibration(page.id, bbox, groundTruth),
    onSuccess: () => {
      toast.success('Calibration submitted! Training started.');
      setStep('done');
    },
    onError: () => toast.error('Calibration failed.'),
  });

  // ── Render ──

  if (step === 'done') {
    return (
      <div className="max-w-xl mx-auto px-4 py-12 text-center">
        <div className="w-16 h-16 bg-green-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
          <svg className="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        </div>
        <h2 className="text-lg font-bold text-gray-900 mb-2">Training Started</h2>
        <p className="text-gray-500 mb-6">
          Your model is being fine-tuned with the calibration data. This may take a few minutes.
        </p>
        <button
          onClick={() => navigate('/documents')}
          className="px-4 py-2 bg-primary-600 text-white rounded-lg text-sm font-medium hover:bg-primary-700 transition-colors"
        >
          Go to Dashboard
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-xl mx-auto px-4 py-6">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <button
          onClick={() => navigate('/documents')}
          className="p-1.5 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100 transition-colors"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
        </button>
        <h1 className="text-xl font-bold text-gray-900">Calibrate Model</h1>
      </div>

      {/* Step indicator */}
      <div className="flex items-center gap-2 mb-6">
        {['Upload', 'Define Region'].map((label, i) => (
          <div key={label} className="flex items-center gap-2">
            {i > 0 && <div className="w-8 h-px bg-gray-300" />}
            <div className={`flex items-center gap-1.5 text-sm font-medium ${
              (i === 0 && step === 'upload') || (i === 1 && step === 'define')
                ? 'text-primary-600'
                : step === 'define' && i === 0
                  ? 'text-green-600'
                  : 'text-gray-400'
            }`}>
              <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                (i === 0 && step === 'upload') || (i === 1 && step === 'define')
                  ? 'bg-primary-100 text-primary-700'
                  : step === 'define' && i === 0
                    ? 'bg-green-100 text-green-700'
                    : 'bg-gray-100 text-gray-400'
              }`}>
                {step === 'define' && i === 0 ? (
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                  </svg>
                ) : i + 1}
              </span>
              {label}
            </div>
          </div>
        ))}
      </div>

      {/* Upload step */}
      {step === 'upload' && (
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <div className="mb-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-1">Write this text on paper:</h3>
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
              <p className="text-lg font-mono text-amber-900 leading-relaxed">{DEFAULT_TEXT}</p>
            </div>
            <p className="text-xs text-gray-400 mt-2">
              You can edit the text below if you wrote something different.
            </p>
          </div>

          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">Ground truth text</label>
            <input
              type="text"
              value={groundTruth}
              onChange={(e) => setGroundTruth(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>

          <p className="text-sm text-gray-500 mb-4">
            Then upload or photograph the handwritten page.
          </p>

          <div className="flex gap-2">
            <label className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-primary-600 text-white rounded-lg text-sm font-medium hover:bg-primary-700 transition-colors cursor-pointer">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
              </svg>
              Upload
              <input ref={fileRef} type="file" accept="image/*" onChange={handleFiles} className="hidden" />
            </label>
            <button
              onClick={() => setShowCamera(true)}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-white border border-gray-300 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-50 transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              Camera
            </button>
          </div>

          {(uploadMutation.isPending || cameraMutation.isPending) && (
            <div className="mt-3 flex items-center justify-center gap-2 text-sm text-primary-600">
              <div className="w-4 h-4 border-2 border-primary-600 border-t-transparent rounded-full animate-spin" />
              Uploading...
            </div>
          )}
        </div>
      )}

      {/* Define step */}
      {step === 'define' && page && (
        <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
          <div className="px-4 py-3 bg-gray-50 border-b border-gray-200 flex items-center justify-between">
            <span className="text-sm text-gray-600 font-medium">Draw a box around the text region</span>
            <button
              onClick={() => rotateMutation.mutate()}
              disabled={rotateMutation.isPending}
              className="p-1 text-gray-500 hover:text-gray-700 disabled:opacity-30 transition-colors"
              title="Rotate 90"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
          </div>

          <div className="p-2">
            <BboxHighlightViewer
              imageSrc={page.image_url}
              bbox={bbox}
              drawMode={!bbox}
              onDrawBbox={(b) => setBbox(b)}
            />
          </div>

          {bbox && (
            <div className="px-4 py-3 border-t border-gray-200">
              <div className="mb-3">
                <label className="block text-xs font-medium text-gray-500 mb-1">Ground truth text</label>
                <input
                  type="text"
                  value={groundTruth}
                  onChange={(e) => setGroundTruth(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                />
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => calibrateMutation.mutate()}
                  disabled={calibrateMutation.isPending || !groundTruth.trim()}
                  className="flex-1 py-2.5 bg-primary-600 text-white rounded-lg text-sm font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors"
                >
                  {calibrateMutation.isPending ? (
                    <span className="flex items-center justify-center gap-2">
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      Submitting...
                    </span>
                  ) : 'Submit & Train'}
                </button>
                <button
                  onClick={() => setBbox(null)}
                  className="px-4 py-2.5 bg-gray-100 text-gray-600 rounded-lg text-sm font-medium hover:bg-gray-200 transition-colors"
                >
                  Redraw
                </button>
              </div>
            </div>
          )}

          {!bbox && (
            <div className="px-4 py-2 text-xs text-blue-500">
              Click and drag on the image to select the text region
            </div>
          )}
        </div>
      )}

      {showCamera && (
        <CameraCapture
          onCapture={(base64) => cameraMutation.mutate(base64)}
          onClose={() => setShowCamera(false)}
        />
      )}
    </div>
  );
}
