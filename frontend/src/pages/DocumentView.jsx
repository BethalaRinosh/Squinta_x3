import { useState, useEffect, useCallback, useRef } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  getDocument, processDocument, processPage, getResults,
  submitCorrection, rotatePage, processBbox, setPageCrop, clearPageCrop,
  autoCropPage, getProcessingStatus, updateResultBbox, summarizeText,
} from '../api';
import { useToast } from '../hooks/useToast';
import PageViewer from '../components/PageViewer';
import OcrResultList from '../components/OcrResultList';
import StructuredOcrView from '../components/StructuredOcrView';

export default function DocumentView() {
  const { id } = useParams();
  const navigate = useNavigate();
  const toast = useToast();
  const queryClient = useQueryClient();
  const [selectedPageIndex, setSelectedPageIndex] = useState(0);
  const [selectedResultId, setSelectedResultId] = useState(null);
  const [drawMode, setDrawMode] = useState(false);
  const [cropMode, setCropMode] = useState(false);
  const [trainMode, setTrainMode] = useState(false);
  const [trainIndex, setTrainIndex] = useState(0);
  const [resultView, setResultView] = useState('formatted');
  const [speechRate, setSpeechRate] = useState(1);
  const [speakingResultId, setSpeakingResultId] = useState(null);
  const [aiSummary, setAiSummary] = useState('');
  const [visualMode, setVisualMode] = useState(() => (
    localStorage.getItem('squinta.visualMode') === 'true'
  ));
  const notifiedProcessingPagesRef = useRef(new Set());

  useEffect(() => {
    localStorage.setItem('squinta.visualMode', String(visualMode));
  }, [visualMode]);

  const { data: doc, isLoading: docLoading } = useQuery({
    queryKey: ['document', id],
    queryFn: () => getDocument(id),
    enabled: !!id,
  });

  const pages = doc?.pages || [];
  const currentPage = pages[selectedPageIndex];
  const currentPageId = currentPage?.id;

  // A newly uploaded document can already be processing before this page
  // mounts. Derive the initial state from the page records so we keep polling
  // instead of rendering a stale one-shot OCR query.
  const [isProcessing, setIsProcessing] = useState(false);

  useEffect(() => {
    if (!pages.length) return;
    setIsProcessing(pages.some((page) => page.processing_status === 'processing'));
  }, [pages]);

  // Poll processing status for this document's pages.
  const { data: processingStatus } = useQuery({
    queryKey: ['processingStatus'],
    queryFn: getProcessingStatus,
    refetchInterval: isProcessing ? 2000 : false,
    retry: false,
  });

  useEffect(() => {
    if (!isProcessing || !processingStatus?.pages) return;
    const docPages = pages.map(p => p.id);
    const relevantPages = processingStatus.pages.filter(p => docPages.includes(p.page_id));
    const stillProcessing = relevantPages.some(p => p.status === 'processing');
    const newlyFinished = relevantPages.filter((p) => {
      const key = `${p.page_id}:${p.status}`;
      return (p.status === 'done' || p.status === 'error')
        && !notifiedProcessingPagesRef.current.has(key);
    });

    if (newlyFinished.length > 0) {
      queryClient.invalidateQueries({ queryKey: ['document', id] });
      queryClient.invalidateQueries({ queryKey: ['ocrResults', currentPageId] });

      const doneCount = newlyFinished.filter(p => p.status === 'done').length;
      const errorCount = newlyFinished.filter(p => p.status === 'error').length;
      newlyFinished.forEach((p) => {
        notifiedProcessingPagesRef.current.add(`${p.page_id}:${p.status}`);
      });

      if (doneCount > 0) {
        toast.success(
          doneCount === 1
            ? 'OCR processing complete!'
            : `OCR processing complete for ${doneCount} pages!`
        );
      }
      if (errorCount > 0) {
        toast.error(
          errorCount === 1
            ? 'OCR processing failed for a page.'
            : `OCR processing failed for ${errorCount} pages.`
        );
      }
    }

    if (relevantPages.length > 0 && !stillProcessing) {
      setIsProcessing(false);
    }
  }, [processingStatus, isProcessing, pages, currentPageId, id, queryClient, toast]);

  const { data: ocrResults, isLoading: ocrLoading } = useQuery({
    queryKey: ['ocrResults', currentPageId],
    queryFn: () => getResults(currentPageId),
    enabled: !!currentPageId,
    refetchInterval: isProcessing ? 3000 : false,
  });

  const results = Array.isArray(ocrResults) ? ocrResults : ocrResults?.results || [];

  useEffect(() => {
    setSelectedResultId(null);
    setDrawMode(false);
    setCropMode(false);
    setAiSummary('');
  }, [selectedPageIndex]);

  const processDocMutation = useMutation({
    mutationFn: () => processDocument(id, visualMode),
    onSuccess: () => {
      notifiedProcessingPagesRef.current.clear();
      queryClient.setQueryData(['processingStatus'], { pages: [] });
      setIsProcessing(true);
      queryClient.invalidateQueries({ queryKey: ['document', id] });
      queryClient.invalidateQueries({ queryKey: ['ocrResults'] });
      queryClient.invalidateQueries({ queryKey: ['processingStatus'] });
      toast.success('OCR processing started for all pages');
    },
    onError: () => {
      toast.error('Failed to start OCR processing.');
    },
  });

  const processPageMutation = useMutation({
    mutationFn: () => processPage(currentPageId, visualMode),
    onSuccess: () => {
      notifiedProcessingPagesRef.current.delete(`${currentPageId}:done`);
      notifiedProcessingPagesRef.current.delete(`${currentPageId}:error`);
      queryClient.setQueryData(['processingStatus'], { pages: [] });
      setIsProcessing(true);
      queryClient.invalidateQueries({ queryKey: ['ocrResults', currentPageId] });
      queryClient.invalidateQueries({ queryKey: ['processingStatus'] });
      toast.success('Page processing started');
    },
    onError: () => {
      toast.error('Failed to process page.');
    },
  });

  const correctionMutation = useMutation({
    mutationFn: ({ resultId, text }) => submitCorrection(resultId, text),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ocrResults', currentPageId] });
      toast.success('Correction saved');
    },
    onError: () => {
      toast.error('Failed to save correction.');
    },
  });

  const rotateMutation = useMutation({
    mutationFn: () => rotatePage(currentPageId, 90),
    onSuccess: () => {
      // image_path changes on rotation (new filename), so URL updates naturally
      queryClient.invalidateQueries({ queryKey: ['document', id] });
      queryClient.invalidateQueries({ queryKey: ['ocrResults', currentPageId] });
      toast.success('Page rotated');
    },
    onError: () => {
      toast.error('Failed to rotate page.');
    },
  });

  const processBboxMutation = useMutation({
    mutationFn: (bbox) => processBbox(currentPageId, bbox),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ocrResults', currentPageId] });
      toast.success('Box OCR result added');
      setDrawMode(false);
    },
    onError: () => {
      toast.error('Failed to process bounding box.');
    },
  });

  const setCropMutation = useMutation({
    mutationFn: (bbox) => setPageCrop(currentPageId, bbox),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['document', id] });
      toast.success('Crop region set');
      setCropMode(false);
    },
    onError: () => {
      toast.error('Failed to set crop region.');
    },
  });

  const clearCropMutation = useMutation({
    mutationFn: () => clearPageCrop(currentPageId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['document', id] });
      toast.success('Crop cleared');
    },
    onError: () => {
      toast.error('Failed to clear crop.');
    },
  });

  const autoCropMutation = useMutation({
    mutationFn: () => autoCropPage(currentPageId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['document', id] });
      toast.success('Auto-crop applied');
    },
    onError: () => {
      toast.error('Failed to auto-crop.');
    },
  });

  const summaryMutation = useMutation({
    mutationFn: () => {
      const pageText = results
        .map((result) => String(result?.translated_text || result?.text || '').trim())
        .filter(Boolean)
        .join('\n');
      return summarizeText(pageText);
    },
    onSuccess: (data) => {
      setAiSummary(data.summary || '');
    },
    onError: (error) => {
      const detail = error.response?.data?.detail;
      toast.error(detail || 'Failed to generate AI summary.');
    },
  });

  const trainBboxMutation = useMutation({
    mutationFn: ({ resultId, bbox }) => updateResultBbox(resultId, bbox),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ocrResults', currentPageId] });
      setTrainIndex(prev => prev + 1);
    },
    onError: () => {
      toast.error('Failed to save bbox.');
    },
  });

  const handleDrawBbox = useCallback((bbox) => {
    if (trainMode) {
      const trainResult = results[trainIndex];
      if (trainResult) {
        trainBboxMutation.mutate({ resultId: trainResult.id, bbox });
      }
    } else if (cropMode) {
      setCropMutation.mutate(bbox);
    } else if (drawMode) {
      processBboxMutation.mutate(bbox);
    }
  }, [trainMode, trainIndex, results, cropMode, drawMode, trainBboxMutation, setCropMutation, processBboxMutation]);

  const handleCorrect = (resultId, text) => {
    correctionMutation.mutate({ resultId, text });
  };

  const stopSpeaking = useCallback(() => {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    setSpeakingResultId(null);
  }, []);

  const speakResult = useCallback((result) => {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
      toast.error('Text-to-speech is not supported in this browser.');
      return;
    }

    const text = String(result?.translated_text || result?.text || '').trim();
    if (!text) return;

    if (speakingResultId === result.id) {
      stopSpeaking();
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = speechRate;
    utterance.pitch = 1;
    utterance.onend = () => setSpeakingResultId(null);
    utterance.onerror = () => setSpeakingResultId(null);

    setSpeakingResultId(result.id);
    window.speechSynthesis.speak(utterance);
  }, [speakingResultId, speechRate, stopSpeaking, toast]);

  const speakSelection = useCallback(() => {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
      toast.error('Text-to-speech is not supported in this browser.');
      return;
    }

    const selection = window.getSelection?.();
    const text = selection?.toString().trim();
    if (!text) {
      toast.error('Select some OCR text first, then click Read selection.');
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = speechRate;
    utterance.pitch = 1;
    utterance.onend = () => setSpeakingResultId(null);
    utterance.onerror = () => setSpeakingResultId(null);

    setSpeakingResultId('__selection__');
    window.speechSynthesis.speak(utterance);
  }, [speechRate, toast]);

  useEffect(() => () => {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
  }, []);

  const goToPrevPage = () => {
    if (selectedPageIndex > 0) setSelectedPageIndex(selectedPageIndex - 1);
  };

  const goToNextPage = () => {
    if (selectedPageIndex < pages.length - 1) setSelectedPageIndex(selectedPageIndex + 1);
  };

  const activeDrawMode = drawMode || cropMode || trainMode;

  const hasCrop = currentPage?.crop_x != null;
  const cropData = hasCrop ? {
    crop_x: currentPage.crop_x,
    crop_y: currentPage.crop_y,
    crop_w: currentPage.crop_w,
    crop_h: currentPage.crop_h,
  } : null;

  if (docLoading) {
    return (
      <div className="max-w-6xl mx-auto px-4 py-6">
        <div className="skeleton h-8 w-48 mb-4" />
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <div className="skeleton w-full" style={{ paddingBottom: '141%' }} />
          </div>
          <div>
            <div className="skeleton h-6 w-32 mb-3" />
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="skeleton h-12 w-full mb-2" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (!doc) {
    return (
      <div className="max-w-6xl mx-auto px-4 py-12 text-center">
        <p className="text-gray-500">Document not found.</p>
        <button
          onClick={() => navigate('/documents')}
          className="mt-4 text-primary-600 hover:text-primary-700 text-sm font-medium"
        >
          Back to documents
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3 min-w-0">
          <button
            onClick={() => navigate('/documents')}
            className="p-1.5 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100 transition-colors shrink-0"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          <div className="min-w-0">
            <h1 className="text-lg font-bold text-gray-900 truncate">
              {doc.name || doc.title || 'Untitled Document'}
            </h1>
            <p className="text-xs text-gray-400">
              {pages.length} page{pages.length !== 1 ? 's' : ''}
              {doc.created_at && <> &middot; {new Date(doc.created_at).toLocaleDateString()}</>}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <Link
            to={`/play?document=${id}`}
            className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-amber-700 bg-amber-50 rounded-lg hover:bg-amber-100 transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Play Mode
          </Link>
          <label
            className="hidden sm:flex items-center gap-2 px-2.5 py-1.5 text-xs text-gray-600 border border-gray-200 rounded-lg bg-white cursor-pointer select-none"
            title="Off: faster OCR with text, text positions, and arrows. On: also detect tables, boxes, circles, brackets, underlines, connectors, and diagrams."
          >
            <input
              type="checkbox"
              checked={visualMode}
              onChange={(e) => setVisualMode(e.target.checked)}
              className="accent-primary-600"
            />
            Detailed visuals
          </label>
          <button
            onClick={() => processDocMutation.mutate()}
            disabled={processDocMutation.isPending}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-primary-600 text-white text-sm font-medium rounded-lg hover:bg-primary-700 disabled:opacity-50 transition-colors"
          >
            {(processDocMutation.isPending || isProcessing) ? (
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            )}
            {isProcessing ? 'Processing...' : 'Process OCR'}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Page thumbnails sidebar */}
        {pages.length > 1 && (
          <div className="lg:col-span-1">
            <div className="flex lg:flex-col gap-2 overflow-x-auto lg:overflow-y-auto lg:max-h-[calc(100vh-12rem)] pb-2 lg:pb-0 lg:pr-2">
              {pages.map((page, index) => (
                <button
                  key={page.id || index}
                  onClick={() => setSelectedPageIndex(index)}
                  className={`shrink-0 w-14 h-18 rounded-lg border-2 overflow-hidden transition-all ${
                    index === selectedPageIndex
                      ? 'border-primary-500 shadow-md'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  {page.image_url ? (
                    <img src={page.image_url} alt={`Page ${index + 1}`} className="w-full h-full object-cover" loading="lazy" />
                  ) : (
                    <div className="w-full h-full bg-gray-100 flex items-center justify-center text-xs text-gray-400">
                      {index + 1}
                    </div>
                  )}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Main page viewer */}
        <div className={pages.length > 1 ? 'lg:col-span-7' : 'lg:col-span-8'}>
          {currentPage ? (
            <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
              {/* Page navigation */}
              <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-b border-gray-200">
                <button
                  onClick={goToPrevPage}
                  disabled={selectedPageIndex === 0}
                  className="p-1 text-gray-500 hover:text-gray-700 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                  </svg>
                </button>
                <div className="flex items-center gap-2">
                  <span className="text-sm text-gray-500">
                    Page {selectedPageIndex + 1} of {pages.length}
                  </span>
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
                  {/* Train Boxes button */}
                  {results.length > 0 && (
                    <button
                      onClick={() => {
                        const next = !trainMode;
                        setTrainMode(next);
                        setDrawMode(false);
                        setCropMode(false);
                        if (next) setTrainIndex(0);
                      }}
                      className={`p-1 rounded transition-colors ${
                        trainMode ? 'bg-purple-100 text-purple-600' : 'text-gray-500 hover:text-gray-700'
                      }`}
                      title={trainMode ? 'Exit training mode' : 'Train bounding boxes'}
                    >
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
                      </svg>
                    </button>
                  )}
                  {/* Draw Box button */}
                  <button
                    onClick={() => { setDrawMode(!drawMode); setCropMode(false); setTrainMode(false); }}
                    className={`p-1 rounded transition-colors ${
                      drawMode ? 'bg-blue-100 text-blue-600' : 'text-gray-500 hover:text-gray-700'
                    }`}
                    title={drawMode ? 'Cancel draw box' : 'Draw box for OCR'}
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h4a1 1 0 010 2H6v3a1 1 0 01-2 0V5zm16 0a1 1 0 00-1-1h-4a1 1 0 000 2h3v3a1 1 0 002 0V5zM4 19a1 1 0 001 1h4a1 1 0 000-2H6v-3a1 1 0 00-2 0v4zm16 0a1 1 0 01-1 1h-4a1 1 0 010-2h3v-3a1 1 0 012 0v4z" />
                    </svg>
                  </button>
                  {/* Crop button */}
                  <button
                    onClick={() => { setCropMode(!cropMode); setDrawMode(false); setTrainMode(false); }}
                    className={`p-1 rounded transition-colors ${
                      cropMode ? 'bg-green-100 text-green-600' : 'text-gray-500 hover:text-gray-700'
                    }`}
                    title={cropMode ? 'Cancel crop' : 'Set crop region'}
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7V5a2 2 0 012-2h2M17 3h2a2 2 0 012 2v2M21 17v2a2 2 0 01-2 2h-2M7 21H5a2 2 0 01-2-2v-2" />
                    </svg>
                  </button>
                  {/* Auto crop */}
                  <button
                    onClick={() => autoCropMutation.mutate()}
                    disabled={autoCropMutation.isPending}
                    className="p-1 text-gray-500 hover:text-gray-700 disabled:opacity-30 transition-colors"
                    title="Auto-detect crop"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </button>
                  {/* Clear crop */}
                  {hasCrop && (
                    <button
                      onClick={() => clearCropMutation.mutate()}
                      disabled={clearCropMutation.isPending}
                      className="p-1 text-red-500 hover:text-red-700 transition-colors"
                      title="Clear crop"
                    >
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  )}
                </div>
                <button
                  onClick={goToNextPage}
                  disabled={selectedPageIndex === pages.length - 1}
                  className="p-1 text-gray-500 hover:text-gray-700 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </button>
              </div>

              {/* Active mode indicator */}
              {activeDrawMode && (
                <div className={`px-4 py-2 text-sm font-medium ${
                  trainMode ? 'bg-purple-50 text-purple-800' :
                  cropMode ? 'bg-green-50 text-green-700' : 'bg-blue-50 text-blue-700'
                }`}>
                  {trainMode ? (
                    trainIndex < results.length ? (
                      <div className="flex items-center justify-between">
                        <div className="flex-1 min-w-0">
                          <span className="text-purple-500 text-xs">Line {trainIndex + 1} of {results.length} — draw a box around:</span>
                          <div className="mt-0.5 font-semibold text-purple-900 truncate">
                            &ldquo;{results[trainIndex]?.text}&rdquo;
                          </div>
                        </div>
                        <div className="flex items-center gap-2 ml-3 shrink-0">
                          <button
                            onClick={() => setTrainIndex(prev => Math.max(0, prev - 1))}
                            disabled={trainIndex === 0}
                            className="px-2 py-0.5 text-xs bg-purple-100 rounded disabled:opacity-30"
                          >
                            Prev
                          </button>
                          <button
                            onClick={() => setTrainIndex(prev => prev + 1)}
                            className="px-2 py-0.5 text-xs bg-purple-100 rounded"
                          >
                            Skip
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="flex items-center justify-between">
                        <span>All lines trained!</span>
                        <button
                          onClick={() => { setTrainMode(false); }}
                          className="px-2 py-0.5 text-xs bg-purple-200 rounded"
                        >
                          Done
                        </button>
                      </div>
                    )
                  ) : cropMode ? 'Draw a rectangle to set the crop region' : 'Draw a rectangle to OCR that region'}
                </div>
              )}

              {/* Image with overlays */}
              <div className="p-2">
                <PageViewer
                  imageSrc={currentPage.image_url || currentPage.url}
                  ocrResults={results}
                  visualElements={
                    visualMode
                      ? (currentPage.visual_elements || [])
                      : (currentPage.visual_elements || []).filter((element) => (
                          String(element.element_type || '').toLowerCase() === 'arrow'
                        ))
                  }
                  selectedResultId={selectedResultId}
                  onSelectResult={setSelectedResultId}
                  crop={cropData}
                  drawMode={activeDrawMode}
                  onDrawBbox={handleDrawBbox}
                />
              </div>
            </div>
          ) : (
            <div className="bg-white rounded-xl border border-gray-200 p-12 text-center">
              <p className="text-gray-400">No pages in this document.</p>
            </div>
          )}
        </div>

        {/* OCR results panel */}
        <div className="lg:col-span-4">
          <div className="bg-white rounded-xl border border-gray-200 overflow-hidden sticky top-20">
            <div className="px-4 py-3 border-b border-gray-100">
              <div className="flex items-center justify-between gap-3">
                <h2 className="text-sm font-semibold text-gray-700">OCR Results</h2>
                {results.length > 0 && (
                  <div className="flex items-center gap-2">
                    <label className="flex items-center gap-1 text-[11px] text-gray-400" title="Speech speed">
                      <span>Speed</span>
                      <select
                        value={speechRate}
                        onChange={(e) => setSpeechRate(Number(e.target.value))}
                        className="rounded border border-gray-200 bg-white px-1.5 py-0.5 text-[11px] text-gray-600 outline-none"
                        aria-label="Speech speed"
                      >
                        <option value="0.8">0.8×</option>
                        <option value="1">1×</option>
                        <option value="1.2">1.2×</option>
                        <option value="1.5">1.5×</option>
                      </select>
                    </label>
                    <button
                      type="button"
                      onClick={speakingResultId ? stopSpeaking : speakSelection}
                      className={"inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium transition-colors " + (speakingResultId ? "bg-gray-100 text-gray-700 hover:bg-gray-200" : "bg-primary-600 text-white hover:bg-primary-700")}
                      title={speakingResultId ? "Stop reading" : "Select OCR text and read it aloud"}
                    >
                      {speakingResultId ? "Stop" : "Read selection"}
                    </button>
                  </div>
                )}
                {currentPageId && (
                  <button
                    onClick={() => processPageMutation.mutate()}
                    disabled={processPageMutation.isPending}
                    className="text-xs text-primary-600 hover:text-primary-700 font-medium disabled:opacity-50"
                  >
                    {processPageMutation.isPending ? 'Processing...' : 'Re-process'}
                  </button>
                )}
              </div>
              {results.length > 0 && (
                <>
                  <div className="mt-3 flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => summaryMutation.mutate()}
                      disabled={summaryMutation.isPending}
                      className="inline-flex flex-1 items-center justify-center gap-1.5 rounded-lg border border-primary-200 bg-primary-50 px-3 py-2 text-xs font-semibold text-primary-700 hover:bg-primary-100 disabled:cursor-not-allowed disabled:opacity-60 transition-colors"
                    >
                      {summaryMutation.isPending ? (
                        <>
                          <div className="w-3.5 h-3.5 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
                          Summarizing...
                        </>
                      ) : (
                        <>
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7l-1 7 9-11h-6z" />
                          </svg>
                          AI Summary
                        </>
                      )}
                    </button>
                  </div>
                  {aiSummary && (
                    <div className="mt-3 rounded-lg border border-gray-200 bg-gray-50 p-3">
                      <div className="mb-2 flex items-center justify-between">
                        <span className="text-xs font-semibold text-gray-700">AI Summary</span>
                        <button
                          type="button"
                          onClick={() => setAiSummary('')}
                          className="text-[11px] text-gray-400 hover:text-gray-600"
                          aria-label="Dismiss AI summary"
                        >
                          Close
                        </button>
                      </div>
                      <div className="whitespace-pre-line text-sm leading-6 text-gray-700">
                        {aiSummary}
                      </div>
                    </div>
                  )}
                  <div className="mt-3 flex rounded-lg bg-gray-100 p-0.5">
                  <button
                    onClick={() => setResultView('formatted')}
                    className={`flex-1 rounded-md px-2 py-1 text-xs font-medium transition-colors ${resultView === 'formatted' ? 'bg-white text-gray-800 shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}
                  >
                    Formatted
                  </button>
                  <button
                    onClick={() => setResultView('list')}
                    className={`flex-1 rounded-md px-2 py-1 text-xs font-medium transition-colors ${resultView === 'list' ? 'bg-white text-gray-800 shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}
                  >
                    Results
                  </button>
                  </div>
                </>
              )}
            </div>

            <div className="max-h-[calc(100vh-16rem)] overflow-y-auto">
              {ocrLoading ? (
                <div className="p-4 space-y-3">
                  {[1, 2, 3, 4, 5].map((i) => (
                    <div key={i}>
                      <div className="skeleton h-4 w-full mb-1" />
                      <div className="skeleton h-3 w-1/3" />
                    </div>
                  ))}
                </div>
              ) : results.length > 0 ? (
                resultView === 'formatted' ? (
                  <StructuredOcrView
                    results={results}
                    visualElements={currentPage?.visual_elements || []}
                    imageSrc={currentPage.image_url || currentPage.url}
                    onSelectResult={setSelectedResultId}
                    selectedResultId={selectedResultId}
                    onSpeak={speakResult}
                    speakingResultId={speakingResultId}
                  />
                ) : (
                  <OcrResultList
                    results={results}
                    selectedResultId={selectedResultId}
                    onSelectResult={setSelectedResultId}
                    onCorrect={handleCorrect}
                    onSpeak={speakResult}
                    speakingResultId={speakingResultId}
                  />
                )
              ) : (
                <div className="p-6 text-center">
                  <svg className="w-10 h-10 text-gray-300 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <p className="text-sm text-gray-400 mb-2">No OCR results for this page</p>
                  <button
                    onClick={() => processPageMutation.mutate()}
                    disabled={processPageMutation.isPending || !currentPageId}
                    className="text-sm text-primary-600 hover:text-primary-700 font-medium disabled:opacity-50"
                  >
                    {processPageMutation.isPending ? 'Processing...' : 'Process this page'}
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
