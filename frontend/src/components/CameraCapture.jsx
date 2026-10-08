import { useState, useRef, useCallback, useEffect } from 'react';

export default function CameraCapture({ onCapture, onClose }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const [error, setError] = useState(null);
  const [ready, setReady] = useState(false);
  const [captured, setCaptured] = useState(null);

  useEffect(() => {
    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'environment',
            width: { ideal: 1920 },
            height: { ideal: 1080 },
          },
        });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current.play();
            setReady(true);
          };
        }
      } catch (err) {
        setError('Camera access denied. Please allow camera permissions and try again.');
      }
    }
    startCamera();

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const capture = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    const base64 = canvas.toDataURL('image/jpeg', 0.9);
    setCaptured(base64);
  }, []);

  const confirmCapture = useCallback(() => {
    if (captured) {
      onCapture(captured);
    }
  }, [captured, onCapture]);

  const retake = useCallback(() => {
    setCaptured(null);
  }, []);

  if (error) {
    return (
      <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-xl p-6 max-w-sm text-center">
          <svg className="w-12 h-12 text-red-400 mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l-4 4m0-4l4 4m6-4a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <p className="text-gray-700 mb-4">{error}</p>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black z-50 flex flex-col">
      <canvas ref={canvasRef} className="hidden" />

      {/* Camera preview or captured image */}
      <div className="flex-1 relative overflow-hidden">
        {captured ? (
          <img
            src={captured}
            alt="Captured"
            className="w-full h-full object-contain"
          />
        ) : (
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-full object-contain"
          />
        )}
        {!ready && !captured && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-white text-lg">Starting camera...</div>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="bg-black/90 p-4 flex items-center justify-center gap-6">
        <button
          onClick={onClose}
          className="w-12 h-12 rounded-full bg-gray-700 text-white flex items-center justify-center hover:bg-gray-600 transition-colors"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        {captured ? (
          <>
            <button
              onClick={retake}
              className="px-6 py-3 rounded-full bg-gray-700 text-white font-medium hover:bg-gray-600 transition-colors"
            >
              Retake
            </button>
            <button
              onClick={confirmCapture}
              className="px-6 py-3 rounded-full bg-primary-600 text-white font-medium hover:bg-primary-700 transition-colors"
            >
              Use Photo
            </button>
          </>
        ) : (
          <button
            onClick={capture}
            disabled={!ready}
            className="w-16 h-16 rounded-full bg-white border-4 border-gray-300 hover:border-primary-400 transition-colors disabled:opacity-50"
            title="Capture"
          >
            <span className="sr-only">Capture</span>
          </button>
        )}
      </div>
    </div>
  );
}
