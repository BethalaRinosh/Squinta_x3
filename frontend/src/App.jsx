import { Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Upload from './pages/Upload';
import DocumentView from './pages/DocumentView';
import Search from './pages/Search';
import Play from './pages/Play';
import Model from './pages/Model';
import Calibrate from './pages/Calibrate';

function AppLayout({ children }) {
  return <Layout>{children}</Layout>;
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Navigate to="/documents" replace />} />
      <Route path="/auth/callback" element={<Navigate to="/documents" replace />} />
      <Route path="/" element={<AppLayout><Dashboard /></AppLayout>} />
      <Route path="/documents" element={<AppLayout><Dashboard /></AppLayout>} />
      <Route path="/documents/upload" element={<AppLayout><Upload /></AppLayout>} />
      <Route path="/documents/:id" element={<AppLayout><DocumentView /></AppLayout>} />
      <Route path="/search" element={<AppLayout><Search /></AppLayout>} />
      <Route path="/play" element={<AppLayout><Play /></AppLayout>} />
      <Route path="/model" element={<AppLayout><Model /></AppLayout>} />
      <Route path="/calibrate" element={<AppLayout><Calibrate /></AppLayout>} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
