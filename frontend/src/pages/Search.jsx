import { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { search } from '../api';
import ConfidenceBadge from '../components/ConfidenceBadge';

function highlightMatch(text, query) {
  if (!query?.trim() || !text) return text;
  const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
  const parts = text.split(regex);
  return parts.map((part, i) =>
    regex.test(part) ? (
      <mark key={i} className="bg-amber-200 text-amber-900 rounded px-0.5">
        {part}
      </mark>
    ) : (
      part
    )
  );
}

function SearchResultCard({ result, query }) {
  return (
    <Link
      to={`/documents/${result.document_id}?page=${result.page_index ?? result.page_number ?? 0}`}
      className="block bg-white rounded-xl border border-gray-200 hover:border-primary-300 hover:shadow-md transition-all p-4"
    >
      <div className="flex gap-4">
        {/* Thumbnail */}
        {result.thumbnail_url && (
          <div className="shrink-0 w-16 h-20 rounded-lg overflow-hidden border border-gray-200 bg-gray-100">
            <img
              src={result.thumbnail_url}
              alt=""
              className="w-full h-full object-cover"
              loading="lazy"
            />
          </div>
        )}

        <div className="min-w-0 flex-1">
          {/* Document info */}
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-medium text-gray-500 truncate">
              {result.document_name || result.document_title || 'Document'}
            </span>
            {(result.page_index !== undefined || result.page_number !== undefined) && (
              <span className="text-xs text-gray-400">
                p. {(result.page_index ?? result.page_number ?? 0) + 1}
              </span>
            )}
            {result.confidence !== undefined && (
              <ConfidenceBadge confidence={result.confidence} />
            )}
          </div>

          {/* Matched text */}
          <p className="text-sm text-gray-800 leading-relaxed">
            {highlightMatch(result.text, query)}
          </p>

          {/* Context */}
          {result.context && (
            <p className="text-xs text-gray-400 mt-1 leading-relaxed line-clamp-2">
              {result.context}
            </p>
          )}
        </div>
      </div>
    </Link>
  );
}

export default function Search() {
  const [query, setQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const inputRef = useRef(null);

  // Debounce search query
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(query.trim());
    }, 300);
    return () => clearTimeout(timer);
  }, [query]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const { data: results, isLoading, isError } = useQuery({
    queryKey: ['search', debouncedQuery],
    queryFn: () => search(debouncedQuery),
    enabled: debouncedQuery.length >= 2,
  });

  const items = Array.isArray(results) ? results : results?.results || [];

  return (
    <div className="max-w-3xl mx-auto px-4 py-6">
      <h1 className="text-xl font-bold text-gray-900 mb-4">Search</h1>

      {/* Search input */}
      <div className="relative mb-6">
        <svg className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search handwritten text..."
          className="w-full pl-12 pr-10 py-3 bg-white border border-gray-200 rounded-xl text-base focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent shadow-sm transition-shadow"
        />
        {query && (
          <button
            onClick={() => {
              setQuery('');
              inputRef.current?.focus();
            }}
            className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>

      {/* Results */}
      {debouncedQuery.length < 2 && !query ? (
        <div className="text-center py-16">
          <svg className="w-16 h-16 text-gray-200 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <p className="text-gray-400">
            Search across all your recognized handwritten text
          </p>
          <p className="text-sm text-gray-300 mt-1">
            Type at least 2 characters to search
          </p>
        </div>
      ) : isLoading ? (
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="bg-white rounded-xl border border-gray-200 p-4">
              <div className="flex gap-4">
                <div className="skeleton w-16 h-20 shrink-0" />
                <div className="flex-1">
                  <div className="skeleton h-3 w-1/3 mb-2" />
                  <div className="skeleton h-4 w-full mb-1" />
                  <div className="skeleton h-4 w-4/5" />
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : isError ? (
        <div className="text-center py-12">
          <p className="text-red-500 text-sm">Search failed. Please try again.</p>
        </div>
      ) : items.length === 0 && debouncedQuery.length >= 2 ? (
        <div className="text-center py-12">
          <svg className="w-12 h-12 text-gray-300 mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <p className="text-gray-500 text-sm">
            No results found for "{debouncedQuery}"
          </p>
          <p className="text-gray-400 text-xs mt-1">
            Try different keywords or check your spelling
          </p>
        </div>
      ) : (
        <div>
          <p className="text-xs text-gray-400 mb-3">
            {items.length} result{items.length !== 1 ? 's' : ''} found
          </p>
          <div className="space-y-3">
            {items.map((result, i) => (
              <SearchResultCard key={result.id || i} result={result} query={debouncedQuery} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
