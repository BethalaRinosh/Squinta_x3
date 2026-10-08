import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getPlayItems, submitPlayCorrection, rotatePage, processPage } from '../api';
import { useToast } from '../hooks/useToast';
import ConfidenceBadge from '../components/ConfidenceBadge';
import BboxHighlightViewer from '../components/BboxHighlightViewer';

function SpeechInput({ onResult, listening, onToggle }) {
  const recognitionRef = useRef(null);

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return;

    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      onResult(transcript);
    };

    recognition.onerror = () => {
      onToggle(false);
    };

    recognition.onend = () => {
      onToggle(false);
    };

    recognitionRef.current = recognition;

    return () => {
      recognition.abort();
    };
  }, [onResult, onToggle]);

  useEffect(() => {
    if (!recognitionRef.current) return;
    if (listening) {
      recognitionRef.current.start();
    } else {
      recognitionRef.current.stop();
    }
  }, [listening]);

  const hasSpeech = !!(window.SpeechRecognition || window.webkitSpeechRecognition);
  if (!hasSpeech) return null;

  return (
    <button
      onClick={() => onToggle(!listening)}
      className={`p-2.5 rounded-lg transition-colors ${
        listening
          ? 'bg-red-100 text-red-600 animate-pulse'
          : 'bg-gray-100 text-gray-500 hover:bg-gray-200 hover:text-gray-700'
      }`}
      title={listening ? 'Stop dictation' : 'Dictate correction'}
    >
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
      </svg>
    </button>
  );
}

export default function Play() {
  const navigate = useNavigate();
  const toast = useToast();
  const queryClient = useQueryClient();
  const [currentIndex, setCurrentIndex] = useState(0);
  const [correctionText, setCorrectionText] = useState('');
  const [isCorrectMode, setIsCorrectMode] = useState(false);
  const [listening, setListening] = useState(false);
  const [transitioning, setTransitioning] = useState(false);
  const [drawMode, setDrawMode] = useState(false);
  const [correctedBbox, setCorrectedBbox] = useState(null);
  const inputRef = useRef(null);

  const { data: playItems, isLoading } = useQuery({
    queryKey: ['playItems'],
    queryFn: getPlayItems,
  });

  const items = Array.isArray(playItems) ? playItems : playItems?.items || [];
  const remaining = playItems?.remaining ?? 0;
  const totalCount = items.length + remaining;
  const currentItem = items[currentIndex];

  // Pre-fill correction text when viewing a new item
  useEffect(() => {
    if (currentItem) {
      setCorrectionText(currentItem.recognized_text || '');
      setIsCorrectMode(false);
      setDrawMode(false);
      setCorrectedBbox(null);
    }
  }, [currentItem]);

  const submitMutation = useMutation({
    mutationFn: ({ ocrResultId, text, bbox }) => submitPlayCorrection(ocrResultId, text, bbox),
    onSuccess: (_data, variables) => {
      toast.success(variables._toastMsg || 'Submitted');
      advanceToNext();
    },
    onError: () => {
      toast.error('Failed to submit. Please try again.');
    },
  });

  const rotateMutation = useMutation({
    mutationFn: async () => {
      if (!currentItem) return;
      await rotatePage(currentItem.page_id, 90);
      // Trigger OCR reprocessing so new results appear in Play
      await processPage(currentItem.page_id);
    },
    onSuccess: () => {
      toast.success('Page rotated — reprocessing OCR');
      // Delay refetch to give OCR time to run
      setTimeout(() => {
        queryClient.invalidateQueries({ queryKey: ['playItems'] });
      }, 5000);
    },
    onError: () => {
      toast.error('Failed to rotate page.');
    },
  });

  const advanceToNext = useCallback(() => {
    setTransitioning(true);
    setTimeout(() => {
      if (currentIndex < items.length - 1) {
        setCurrentIndex((prev) => prev + 1);
      } else {
        queryClient.invalidateQueries({ queryKey: ['playItems'] });
        setCurrentIndex(0);
      }
      setCorrectionText('');
      setIsCorrectMode(false);
      setDrawMode(false);
      setCorrectedBbox(null);
      setTransitioning(false);
    }, 200);
  }, [currentIndex, items.length, queryClient]);

  const handleConfirmCorrect = () => {
    if (!currentItem) return;
    submitMutation.mutate({
      ocrResultId: currentItem.ocr_result_id,
      text: currentItem.recognized_text,
      bbox: correctedBbox,
      _toastMsg: 'Marked as correct',
    });
  };

  const handleSubmitCorrection = () => {
    if (!currentItem || !correctionText.trim()) return;
    submitMutation.mutate({
      ocrResultId: currentItem.ocr_result_id,
      text: correctionText.trim(),
      bbox: correctedBbox,
      _toastMsg: 'Correction submitted',
    });
  };

  const handleSkip = () => {
    advanceToNext();
  };

  const handleSpeechResult = useCallback((transcript) => {
    setCorrectionText(transcript);
    setIsCorrectMode(true);
    setListening(false);
  }, []);

  const handleDrawBbox = useCallback((bbox) => {
    setCorrectedBbox(bbox);
    setDrawMode(false);
  }, []);

  // Focus input when entering correct mode
  useEffect(() => {
    if (isCorrectMode) {
      inputRef.current?.focus();
      inputRef.current?.select();
    }
  }, [isCorrectMode]);

  if (isLoading) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-6">
        <div className="skeleton h-8 w-48 mb-6" />
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <div className="skeleton w-full h-32 mb-4" />
          <div className="skeleton h-6 w-3/4 mb-3" />
          <div className="skeleton h-10 w-full mb-3" />
          <div className="flex gap-2">
            <div className="skeleton h-10 flex-1" />
            <div className="skeleton h-10 flex-1" />
          </div>
        </div>
      </div>
    );
  }

  if (items.length === 0) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-12 text-center">
        <div className="w-20 h-20 bg-green-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
          <svg className="w-10 h-10 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        <h2 className="text-lg font-bold text-gray-900 mb-2">All caught up!</h2>
        <p className="text-gray-500 mb-4">
          No items need review right now. Process more documents to generate new OCR results.
        </p>
        <button
          onClick={() => navigate('/documents')}
          className="px-4 py-2 bg-primary-600 text-white rounded-lg text-sm font-medium hover:bg-primary-700 transition-colors"
        >
          Go to Documents
        </button>
      </div>
    );
  }

  const activeBbox = correctedBbox || (currentItem ? {
    x: currentItem.bbox_x,
    y: currentItem.bbox_y,
    w: currentItem.bbox_w,
    h: currentItem.bbox_h,
  } : null);

  return (
    <div className="max-w-2xl mx-auto px-4 py-6">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-bold text-gray-900">Play Mode</h1>
        <span className="text-sm text-gray-400">
          {currentIndex + 1} of {totalCount} to review
        </span>
      </div>

      {/* Progress bar */}
      <div className="h-1.5 bg-gray-100 rounded-full mb-6 overflow-hidden">
        <div
          className="h-full bg-primary-500 rounded-full transition-all duration-500"
          style={{ width: `${totalCount > 0 ? ((currentIndex + 1) / totalCount) * 100 : 0}%` }}
        />
      </div>

      <div
        className={`transition-all duration-200 ${
          transitioning ? 'opacity-0 translate-x-4' : 'opacity-100 translate-x-0'
        }`}
      >
        {/* Card */}
        <div className="bg-white rounded-xl border border-gray-200 overflow-hidden shadow-sm">
          {/* Handwriting image with bbox highlight */}
          <div className="bg-gray-50 border-b border-gray-200 p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-gray-400 font-medium uppercase tracking-wide">
                {currentItem.document_name}
              </span>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => rotateMutation.mutate()}
                  disabled={rotateMutation.isPending}
                  className="p-1.5 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-200 transition-colors disabled:opacity-30"
                  title="Rotate page 90°"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                </button>
                <button
                  onClick={() => {
                    setDrawMode(!drawMode);
                    if (drawMode) setCorrectedBbox(null);
                  }}
                  className={`p-1.5 rounded-lg transition-colors ${
                    drawMode
                      ? 'bg-blue-100 text-blue-600'
                      : 'text-gray-400 hover:text-gray-600 hover:bg-gray-200'
                  }`}
                  title={drawMode ? 'Cancel drawing' : 'Adjust region'}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h4a1 1 0 010 2H6v3a1 1 0 01-2 0V5zm16 0a1 1 0 00-1-1h-4a1 1 0 000 2h3v3a1 1 0 002 0V5zM4 19a1 1 0 001 1h4a1 1 0 000-2H6v-3a1 1 0 00-2 0v4zm16 0a1 1 0 01-1 1h-4a1 1 0 010-2h3v-3a1 1 0 012 0v4z" />
                  </svg>
                </button>
              </div>
            </div>
            {currentItem.image_url ? (
              <BboxHighlightViewer
                imageSrc={currentItem.image_url}
                bbox={activeBbox}
                drawMode={drawMode}
                onDrawBbox={handleDrawBbox}
              />
            ) : (
              <div className="text-center text-gray-400 py-8">
                <svg className="w-12 h-12 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <p className="text-sm">Image not available</p>
              </div>
            )}
            {correctedBbox && (
              <div className="mt-2 flex items-center gap-2">
                <span className="text-xs text-blue-600 font-medium">Custom region set</span>
                <button
                  onClick={() => setCorrectedBbox(null)}
                  className="text-xs text-gray-400 hover:text-gray-600"
                >
                  Reset
                </button>
              </div>
            )}
            {drawMode && (
              <p className="mt-2 text-xs text-blue-500">Click and drag on the image to draw a new bounding box</p>
            )}
          </div>

          {/* Model's text */}
          <div className="p-5">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                Model's reading
              </span>
              {currentItem.confidence !== undefined && (
                <ConfidenceBadge confidence={currentItem.confidence} />
              )}
            </div>

            {/* Confidence progress bar */}
            {currentItem.confidence !== undefined && (
              <div className="h-2 bg-gray-100 rounded-full mb-3 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${
                    currentItem.confidence > 0.8
                      ? 'bg-green-500'
                      : currentItem.confidence >= 0.5
                        ? 'bg-amber-500'
                        : 'bg-red-500'
                  }`}
                  style={{ width: `${Math.round(currentItem.confidence * 100)}%` }}
                />
              </div>
            )}

            <p className="text-lg text-gray-900 font-mono bg-gray-50 rounded-lg px-4 py-3 border border-gray-200">
              {currentItem.recognized_text || <span className="italic text-gray-400">No text recognized</span>}
            </p>

            {/* Correction input */}
            {isCorrectMode && (
              <div className="mt-3">
                <label className="block text-xs font-medium text-gray-500 mb-1">
                  Your correction
                </label>
                <div className="flex gap-2">
                  <input
                    ref={inputRef}
                    type="text"
                    value={correctionText}
                    onChange={(e) => setCorrectionText(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleSubmitCorrection();
                      if (e.key === 'Escape') setIsCorrectMode(false);
                    }}
                    placeholder="Type the correct text..."
                    className="flex-1 px-4 py-2.5 border border-gray-300 rounded-lg text-base focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  />
                  <SpeechInput
                    onResult={handleSpeechResult}
                    listening={listening}
                    onToggle={setListening}
                  />
                </div>
              </div>
            )}

            {/* Action buttons */}
            <div className="mt-4 flex gap-2">
              {isCorrectMode ? (
                <>
                  <button
                    onClick={handleSubmitCorrection}
                    disabled={submitMutation.isPending || !correctionText.trim()}
                    className="flex-1 py-2.5 bg-primary-600 text-white rounded-lg text-sm font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors"
                  >
                    {submitMutation.isPending ? (
                      <span className="flex items-center justify-center gap-2">
                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                        Saving...
                      </span>
                    ) : (
                      'Submit Correction'
                    )}
                  </button>
                  <button
                    onClick={() => setIsCorrectMode(false)}
                    className="px-4 py-2.5 bg-gray-100 text-gray-600 rounded-lg text-sm font-medium hover:bg-gray-200 transition-colors"
                  >
                    Cancel
                  </button>
                </>
              ) : (
                <>
                  <button
                    onClick={handleConfirmCorrect}
                    disabled={submitMutation.isPending}
                    className="flex-1 py-2.5 bg-green-600 text-white rounded-lg text-sm font-medium hover:bg-green-700 disabled:opacity-50 transition-colors flex items-center justify-center gap-2"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    Correct!
                  </button>
                  <button
                    onClick={() => setIsCorrectMode(true)}
                    className="flex-1 py-2.5 bg-primary-600 text-white rounded-lg text-sm font-medium hover:bg-primary-700 transition-colors flex items-center justify-center gap-2"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                    </svg>
                    Fix It
                  </button>
                  <button
                    onClick={handleSkip}
                    className="px-4 py-2.5 bg-gray-100 text-gray-500 rounded-lg text-sm font-medium hover:bg-gray-200 hover:text-gray-700 transition-colors"
                    title="Skip"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                    </svg>
                  </button>
                </>
              )}
            </div>

            {/* Keyboard shortcuts hint */}
            <p className="mt-3 text-xs text-gray-400 text-center">
              Press <kbd className="px-1.5 py-0.5 bg-gray-100 rounded text-gray-500 font-mono">Enter</kbd> to submit
              {' '}&middot;{' '}
              <kbd className="px-1.5 py-0.5 bg-gray-100 rounded text-gray-500 font-mono">Esc</kbd> to cancel
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
