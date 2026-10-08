import { useState, useCallback } from 'react';
import ConfidenceBadge from './ConfidenceBadge';

export default function OcrResultList({
  results = [],
  selectedResultId,
  onSelectResult,
  onCorrect,
}) {
  const [editingId, setEditingId] = useState(null);
  const [editText, setEditText] = useState('');

  const startEditing = useCallback((result) => {
    setEditingId(result.id);
    setEditText(result.text || '');
  }, []);

  const cancelEditing = useCallback(() => {
    setEditingId(null);
    setEditText('');
  }, []);

  const saveEdit = useCallback(
    (resultId) => {
      if (onCorrect && editText.trim()) {
        onCorrect(resultId, editText.trim());
      }
      setEditingId(null);
      setEditText('');
    },
    [editText, onCorrect]
  );

  if (results.length === 0) {
    return (
      <div className="text-center py-8 text-gray-400">
        <svg className="w-10 h-10 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        <p className="text-sm">No OCR results yet</p>
      </div>
    );
  }

  return (
    <div className="divide-y divide-gray-100">
      {results.map((result) => {
        const isSelected = result.id === selectedResultId;
        const isEditing = result.id === editingId;

        return (
          <div
            key={result.id}
            className={`px-3 py-2.5 cursor-pointer transition-colors ${
              isSelected ? 'bg-primary-50 border-l-2 border-primary-500' : 'hover:bg-gray-50 border-l-2 border-transparent'
            }`}
            onClick={() => onSelectResult?.(result.id)}
          >
            <div className="flex items-start justify-between gap-2">
              {isEditing ? (
                <div className="flex-1">
                  <input
                    type="text"
                    value={editText}
                    onChange={(e) => setEditText(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') saveEdit(result.id);
                      if (e.key === 'Escape') cancelEditing();
                    }}
                    autoFocus
                    className="w-full px-2 py-1 text-sm border border-primary-300 rounded focus:outline-none focus:ring-2 focus:ring-primary-500"
                    onClick={(e) => e.stopPropagation()}
                  />
                  <div className="flex gap-1 mt-1">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        saveEdit(result.id);
                      }}
                      className="text-xs text-primary-600 hover:text-primary-800 font-medium"
                    >
                      Save
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        cancelEditing();
                      }}
                      className="text-xs text-gray-500 hover:text-gray-700"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <div className="flex-1">
                  <p className="text-sm text-gray-800 leading-relaxed">
                    {result.text || <span className="italic text-gray-400">No text recognized</span>}
                  </p>
                  {result.translated_text && result.translated_text !== result.text && (
                    <p className="mt-1 text-sm text-emerald-700 leading-relaxed">
                      <span className="font-medium mr-1">English:</span>
                      {result.translated_text}
                    </p>
                  )}
                  {(result.language_name || result.language || result.script) && (
                    <div className="mt-1 flex flex-wrap items-center gap-1.5 text-[10px] text-gray-500">
                      {result.language_name && (
                        <span className="inline-flex items-center rounded-full bg-indigo-50 px-2 py-0.5 font-medium text-indigo-700">
                          {result.language_name}
                        </span>
                      )}
                      {result.script && (
                        <span className="inline-flex items-center rounded-full bg-slate-100 px-2 py-0.5 font-medium text-slate-600">
                          {result.script}
                        </span>
                      )}
                    </div>
                  )}
                </div>
              )}
              <div className="flex items-center gap-1.5 shrink-0">
                <ConfidenceBadge confidence={result.confidence ?? 0} />
                {!isEditing && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      startEditing(result);
                    }}
                    className="p-1 text-gray-400 hover:text-primary-600 transition-colors"
                    title="Edit"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                    </svg>
                  </button>
                )}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
