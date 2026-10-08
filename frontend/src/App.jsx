import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './hooks/useAuth';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Upload from './pages/Upload';
import DocumentView from './pages/DocumentView';
import Search from './pages/Search';
import Play from './pages/Play';
import Model from './pages/Model';
import Calibrate from './pages/Calibrate';

// Wait only for the local guest/Google session to initialize. Authentication
// is no longer a prerequisite for opening the app.
function AppRoute({ children }) {
  const { loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-primary-600 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <p className="text-sm text-gray-500">Loading Squinta...</p>
        </div>
      </div>
    );
  }

  return <Layout>{children}</Layout>;
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<AppRoute><Dashboard /></AppRoute>} />
      <Route path="/documents" element={<AppRoute><Dashboard /></AppRoute>} />
      <Route path="/documents/upload" element={<AppRoute><Upload /></AppRoute>} />
      <Route path="/documents/:id" element={<AppRoute><DocumentView /></AppRoute>} />
      <Route path="/search" element={<AppRoute><Search /></AppRoute>} />
      <Route path="/play" element={<AppRoute><Play /></AppRoute>} />
      <Route path="/model" element={<AppRoute><Model /></AppRoute>} />
      <Route path="/calibrate" element={<AppRoute><Calibrate /></AppRoute>} />

      {/* Unknown URLs, including old /login bookmarks, go to the dashboard. */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
