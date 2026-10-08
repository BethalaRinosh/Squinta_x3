import { useState, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { uploadDocuments, captureCamera, createPickerSession, pollPickerSession, importFromPicker } from '../api';
import { useToast } from '../hooks/useToast';
import CameraCapture from '../components/CameraCapture';

function GooglePhotosPicker({ onImport }) {
  const toast = useToast();
  const [status, setStatus] = useState('idle'); // idle | picking | polling | importing
  const [sessionId, setSessionId] = useState(null);

  const startPicker = async () => {
    // Open window SYNCHRONOUSLY from the click event to avoid popup blocker.
    // We'll set the URL once we have the picker URI.
    const pickerWindow = window.open('about:blank', '_blank', 'width=900,height=700');

    try {
      setStatus('picking');
      const session = await createPickerSession();
      setSessionId(session.session_id);

      if (pickerWindow) {
        pickerWindow.location.href = session.picker_uri;
      } else {
        // Popup was blocked — fall back to same-tab redirect
        window.location.href = session.picker_uri;
        return;
      }

      // Poll until the user finishes selecting.
      // The user signals "done" by closing the picker window — at that point
      // mediaItemsSet should be true. We must check ready BEFORE checking
      // window.closed, because closing IS the expected completion action.
      setStatus('polling');
      let windowClosedCount = 0;
      const poll = async () => {
        try {
          const result = await pollPickerSession(session.session_id);
          if (result.ready) {
            if (pickerWindow && !pickerWindow.closed) pickerWindow.close();
            setStatus('importing');
            const imported = await importFromPicker(session.session_id, 'Google Photos Import');
            toast.success(`Imported ${imported.imported_count} photos`);
            onImport(imported);
            setStatus('idle');
            setSessionId(null);
            return;
          }
        } catch {
          // API error — keep trying
        }

        // If the window is closed but not ready, give it a few more polls
        // (there can be a delay between closing the picker and the session updating)
        if (pickerWindow?.closed) {
          windowClosedCount++;
          if (windowClosedCount > 5) {
            // User closed without selecting
            toast.info('Photo picker closed without importing.');
            setStatus('idle');
            setSessionId(null);
            return;
          }
        }

        setTimeout(poll, 3000);
      };

      setTimeout(poll, 2000);
    } catch (err) {
      if (pickerWindow) pickerWindow.close();
      toast.error('Failed to open Google Photos picker. Try signing out and back in.');
      setStatus('idle');
    }
  };

  return (
    <div className="text-center py-8">
      <div className="w-16 h-16 bg-gray-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
        <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
      </div>

      {status === 'idle' && (
        <>
          <p className="text-gray-600 mb-1">Import from Google Photos</p>
          <p className="text-sm text-gray-400 mb-4">
            Opens Google's photo picker where you select the images to import
          </p>
          <button
            onClick={startPicker}
            className="px-6 py-2.5 bg-primary-600 text-white rounded-lg text-sm font-medium hover:bg-primary-700 transition-colors"
          >
            Open Google Photos
          </button>
        </>
      )}

      {status === 'picking' && (
        <div className="flex flex-col items-center gap-2">
          <div className="w-6 h-6 border-2 border-primary-600 border-t-transparent rounded-full animate-spin" />
          <p className="text-sm text-gray-600">Opening Google Photos picker...</p>
        </div>
      )}

      {status === 'polling' && (
        <div className="flex flex-col items-center gap-2">
          <div className="w-6 h-6 border-2 border-primary-600 border-t-transparent rounded-full animate-spin" />
          <p className="text-sm text-gray-600">Waiting for you to select photos...</p>
          <p className="text-xs text-gray-400">Select photos in the Google Photos window, then close it</p>
        </div>
      )}

      {status === 'importing' && (
        <div className="flex flex-col items-center gap-2">
          <div className="w-6 h-6 border-2 border-primary-600 border-t-transparent rounded-full animate-spin" />
          <p className="text-sm text-gray-600">Importing selected photos...</p>
        </div>
      )}
    </div>
  );
}

export default function Upload() {
  const navigate = useNavigate();
  const toast = useToast();
  const queryClient = useQueryClient();
  const fileInputRef = useRef(null);
  const [showCamera, setShowCamera] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [activeTab, setActiveTab] = useState('upload');

  const uploadMutation = useMutation({
    mutationFn: uploadDocuments,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      toast.success('Files uploaded successfully');
      if (data?.id) {
        navigate(`/documents/${data.id}`);
      } else {
        navigate('/documents');
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
      } else {
        navigate('/documents');
      }
    },
    onError: () => {
      toast.error('Failed to upload captured photo.');
    },
  });

  const handleFiles = useCallback(
    (files) => {
      if (files && files.length > 0) {
        uploadMutation.mutate(Array.from(files));
      }
    },
    [uploadMutation]
  );

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setIsDragging(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles]
  );

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  return (
    <div className="max-w-3xl mx-auto px-4 py-6">
      <div className="flex items-center gap-3 mb-6">
        <button
          onClick={() => navigate('/documents')}
          className="p-1.5 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100 transition-colors"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
        </button>
        <h1 className="text-xl font-bold text-gray-900">Add Documents</h1>
      </div>

      {/* Tab bar */}
      <div className="flex border-b border-gray-200 mb-6">
        {[
          { key: 'upload', label: 'Upload Files' },
          { key: 'camera', label: 'Camera' },
          { key: 'photos', label: 'Google Photos' },
        ].map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
              activeTab === tab.key
                ? 'border-primary-600 text-primary-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Upload tab */}
      {activeTab === 'upload' && (
        <div
          className={`border-2 border-dashed rounded-xl p-12 text-center transition-colors ${
            isDragging ? 'drop-zone-active border-primary-400' : 'border-gray-300 hover:border-gray-400'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <svg className="w-12 h-12 text-gray-300 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          <p className="text-gray-600 mb-1">
            {isDragging ? 'Drop files here' : 'Drag and drop files here'}
          </p>
          <p className="text-sm text-gray-400 mb-4">
            Supports images (JPG, PNG, HEIC) and PDFs
          </p>
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploadMutation.isPending}
            className="px-6 py-2.5 bg-primary-600 text-white rounded-lg text-sm font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors"
          >
            {uploadMutation.isPending ? (
              <span className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Uploading...
              </span>
            ) : (
              'Choose Files'
            )}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,.pdf"
            multiple
            onChange={(e) => handleFiles(e.target.files)}
            className="hidden"
          />
        </div>
      )}

      {/* Camera tab */}
      {activeTab === 'camera' && (
        <div className="text-center py-8">
          <div className="w-16 h-16 bg-gray-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </div>
          <p className="text-gray-600 mb-1">Capture a photo of handwritten text</p>
          <p className="text-sm text-gray-400 mb-4">
            Use your device's camera to take a picture of a page
          </p>
          <button
            onClick={() => setShowCamera(true)}
            className="px-6 py-2.5 bg-primary-600 text-white rounded-lg text-sm font-medium hover:bg-primary-700 transition-colors"
          >
            Open Camera
          </button>
        </div>
      )}

      {/* Google Photos tab */}
      {activeTab === 'photos' && (
        <GooglePhotosPicker
          onImport={(data) => {
            queryClient.invalidateQueries({ queryKey: ['documents'] });
            if (data?.document_id) {
              navigate(`/documents/${data.document_id}`);
            } else {
              navigate('/documents');
            }
          }}
        />
      )}

      {/* Camera modal */}
      {showCamera && (
        <CameraCapture
          onCapture={(base64) => cameraMutation.mutate(base64)}
          onClose={() => setShowCamera(false)}
        />
      )}
    </div>
  );
}
