import { useState, useEffect, useRef } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { listDocuments, uploadDocuments, captureCamera, deleteDocument, getPlayItems, getProcessingStatus } from '../api';
import { useToast } from '../hooks/useToast';
import CameraCapture from '../components/CameraCapture';

function DocumentCard({ doc, onDelete }) {
  const navigate = useNavigate();
  const [confirmDelete, setConfirmDelete] = useState(false);

  const statusColors = {
    new: 'bg-gray-100 text-gray-600',
    processed: 'bg-green-100 text-green-700',
  };

  const statusLabel = (doc.ocr_result_count || 0) > 0 ? 'processed' : 'new';

  return (
    <div
      onClick={() => navigate(`/documents/${doc.id}`)}
      className="block bg-white rounded-xl border border-gray-200 hover:border-primary-300 hover:shadow-md transition-all p-4 group cursor-pointer"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <h3 className="font-medium text-gray-900 truncate group-hover:text-primary-700 transition-colors">
            {doc.name || doc.title || 'Untitled Document'}
          </h3>
          <p className="text-xs text-gray-400 mt-1">
            {doc.page_count ?? doc.pages?.length ?? 0} page{(doc.page_count ?? doc.pages?.length ?? 0) !== 1 ? 's' : ''}
            {doc.created_at && (
              <> &middot; {new Date(doc.created_at).toLocaleDateString()}</>
            )}
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${statusColors[statusLabel] || statusColors.new}`}>
            {statusLabel}
          </span>
          <button
            onClick={(e) => {
              e.stopPropagation();
              if (confirmDelete) {
                onDelete(doc.id);
                setConfirmDelete(false);
              } else {
                setConfirmDelete(true);
                setTimeout(() => setConfirmDelete(false), 3000);
              }
            }}
            className={`p-1.5 rounded-lg transition-colors ${
              confirmDelete
                ? 'bg-red-100 text-red-600'
                : 'text-gray-400 hover:text-red-500 hover:bg-red-50'
            }`}
            title={confirmDelete ? 'Click again to confirm' : 'Delete'}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}

function SkeletonCard() {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4">
      <div className="skeleton h-5 w-3/4 mb-2" />
      <div className="skeleton h-3 w-1/2 mb-3" />
      <div className="flex gap-1.5">
        {[1, 2, 3].map((i) => (
          <div key={i} className="skeleton w-12 h-16 shrink-0" />
        ))}
      </div>
    </div>
  );
}

export default function Dashboard() {
  const navigate = useNavigate();
  const toast = useToast();
  const queryClient = useQueryClient();
  const [showCamera, setShowCamera] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const { data: documents, isLoading } = useQuery({
    queryKey: ['documents'],
    queryFn: listDocuments,
  });

  const { data: playItems } = useQuery({
    queryKey: ['playItems'],
    queryFn: getPlayItems,
    retry: false,
  });

  // Poll for processing status to show notifications.
  const processingSeenRef = useRef(new Set());
  const { data: processingStatus } = useQuery({
    queryKey: ['processingStatus'],
    queryFn: getProcessingStatus,
    refetchInterval: 3000,
    retry: false,
  });

  useEffect(() => {
    if (!processingStatus?.pages) return;
    for (const p of processingStatus.pages) {
      const key = `${p.page_id}`;
      if (p.status === 'done' && !processingSeenRef.current.has(key)) {
        processingSeenRef.current.add(key);
        toast.success(`OCR complete: ${p.document_name}`);
        queryClient.invalidateQueries({ queryKey: ['documents'] });
        queryClient.invalidateQueries({ queryKey: ['playItems'] });
      } else if (p.status === 'error' && !processingSeenRef.current.has(key)) {
        processingSeenRef.current.add(key);
        toast.error(`OCR failed: ${p.document_name}`);
      }
    }
  }, [processingStatus, toast, queryClient]);

  const uploadMutation = useMutation({
    mutationFn: uploadDocuments,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      toast.success('Documents uploaded successfully');
      if (data?.id) {
        navigate(`/documents/${data.id}`);
      }
    },
    onError: () => {
      toast.error('Upload failed. Please try again.');
    },
  });

  const cameraMutation = useMutation({
    mutationFn: captureCamera,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      setShowCamera(false);
      toast.success('Photo captured and uploaded');
      if (data?.id) {
        navigate(`/documents/${data.id}`);
      }
    },
    onError: () => {
      toast.error('Failed to upload captured photo.');
    },
  });

  const deleteMutation = useMutation({
    mutationFn: deleteDocument,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      toast.success('Document deleted');
    },
    onError: () => {
      toast.error('Failed to delete document.');
    },
  });

  const handleFileUpload = (e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      uploadMutation.mutate(Array.from(files));
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      uploadMutation.mutate(Array.from(files));
    }
  };

  const handleCameraCapture = (base64) => {
    cameraMutation.mutate(base64);
  };

  const filteredDocs = documents?.filter((doc) => {
    if (!searchQuery.trim()) return true;
    const q = searchQuery.toLowerCase();
    return (
      (doc.name || doc.title || '').toLowerCase().includes(q)
    );
  }) || [];

  const playCount = playItems?.items?.length
    ? playItems.items.length + (playItems.remaining || 0)
    : 0;

  return (
    <div className="max-w-4xl mx-auto px-4 py-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-bold text-gray-900">Documents</h1>
        <div className="flex items-center gap-2">
          <Link
            to="/calibrate"
            className="flex items-center gap-1.5 px-3 py-2 bg-purple-50 text-purple-700 rounded-lg text-sm font-medium hover:bg-purple-100 transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
            </svg>
            Calibrate
          </Link>
          {playCount > 0 && (
            <Link
              to="/play"
              className="flex items-center gap-1.5 px-3 py-2 bg-amber-50 text-amber-700 rounded-lg text-sm font-medium hover:bg-amber-100 transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {playCount} to review
            </Link>
          )}
        </div>
      </div>

      {/* Quick search */}
      <div className="mb-4">
        <div className="relative">
          <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Filter documents..."
            className="w-full pl-10 pr-4 py-2.5 bg-white border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-shadow"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
      </div>

      {/* Action buttons */}
      <div
        className="mb-6 border-2 border-dashed border-gray-200 rounded-xl p-6 text-center hover:border-primary-300 transition-colors"
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
      >
        <svg className="w-8 h-8 text-gray-300 mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
        </svg>
        <p className="text-sm text-gray-500 mb-3">
          Drag and drop files here, or use the buttons below
        </p>
        <div className="flex flex-wrap items-center justify-center gap-2">
          <label className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg text-sm font-medium hover:bg-primary-700 transition-colors cursor-pointer">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
            </svg>
            Upload Files
            <input
              type="file"
              accept="image/*,.pdf"
              multiple
              onChange={handleFileUpload}
              className="hidden"
            />
          </label>

          <button
            onClick={() => setShowCamera(true)}
            className="inline-flex items-center gap-2 px-4 py-2 bg-white border border-gray-300 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-50 transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            Camera
          </button>

          <Link
            to="/documents/upload"
            className="inline-flex items-center gap-2 px-4 py-2 bg-white border border-gray-300 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-50 transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            Google Photos
          </Link>
        </div>

        {uploadMutation.isPending && (
          <div className="mt-3 flex items-center justify-center gap-2 text-sm text-primary-600">
            <div className="w-4 h-4 border-2 border-primary-600 border-t-transparent rounded-full animate-spin" />
            Uploading...
          </div>
        )}
      </div>

      {/* Document list */}
      {isLoading ? (
        <div className="grid gap-3 sm:grid-cols-2">
          {[1, 2, 3, 4].map((i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
      ) : filteredDocs.length === 0 ? (
        <div className="text-center py-12">
          <svg className="w-12 h-12 text-gray-300 mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <p className="text-gray-500 text-sm">
            {searchQuery ? 'No documents match your search.' : 'No documents yet. Upload some handwritten pages to get started.'}
          </p>
        </div>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2">
          {filteredDocs.map((doc) => (
            <DocumentCard key={doc.id} doc={doc} onDelete={(id) => deleteMutation.mutate(id)} />
          ))}
        </div>
      )}

      {/* Camera capture modal */}
      {showCamera && (
        <CameraCapture
          onCapture={handleCameraCapture}
          onClose={() => setShowCamera(false)}
        />
      )}
    </div>
  );
}
