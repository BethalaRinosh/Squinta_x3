import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Auth header interceptor
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor for 401 handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      // Only redirect if not already on login page
      if (window.location.pathname !== '/login') {
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

// ---- Auth ----

export async function login() {
  // Redirect to backend OAuth endpoint
  window.location.href = '/api/auth/login';
}

export async function getMe() {
  const { data } = await api.get('/auth/me');
  return data;
}

export async function logout() {
  const { data } = await api.post('/auth/logout');
  localStorage.removeItem('token');
  return data;
}

// ---- Documents ----

export async function listDocuments() {
  const { data } = await api.get('/documents');
  return data;
}

export async function uploadDocuments(files) {
  const formData = new FormData();
  for (const file of files) {
    formData.append('files', file);
  }
  const { data } = await api.post('/documents/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

export async function captureCamera(base64) {
  const { data } = await api.post('/documents/camera', { image: base64 });
  return data;
}

export async function getDocument(id) {
  const { data } = await api.get(`/documents/${id}`);
  return data;
}

export async function deleteDocument(id) {
  const { data } = await api.delete(`/documents/${id}`);
  return data;
}

export async function rotatePage(pageId, rotation) {
  const { data } = await api.post(`/documents/pages/${pageId}/rotate`, { rotation });
  return data;
}

// ---- OCR ----

export async function processPage(pageId) {
  const { data } = await api.post(`/ocr/process/${pageId}`);
  return data;
}

export async function processDocument(docId) {
  const { data } = await api.post(`/ocr/process-document/${docId}`);
  return data;
}

export async function getResults(pageId) {
  const { data } = await api.get(`/ocr/results/${pageId}`);
  return data;
}

export async function processBbox(pageId, bbox) {
  const { data } = await api.post(`/ocr/process-bbox/${pageId}`, {
    bbox_x: bbox.x,
    bbox_y: bbox.y,
    bbox_w: bbox.w,
    bbox_h: bbox.h,
  });
  return data;
}

export async function updateResultBbox(resultId, bbox) {
  const { data } = await api.put(`/ocr/result/${resultId}/bbox`, {
    bbox_x: bbox.x,
    bbox_y: bbox.y,
    bbox_w: bbox.w,
    bbox_h: bbox.h,
  });
  return data;
}

export async function setPageCrop(pageId, bbox) {
  const { data } = await api.post(`/documents/pages/${pageId}/crop`, {
    crop_x: bbox.x,
    crop_y: bbox.y,
    crop_w: bbox.w,
    crop_h: bbox.h,
  });
  return data;
}

export async function clearPageCrop(pageId) {
  const { data } = await api.post(`/documents/pages/${pageId}/crop/clear`);
  return data;
}

export async function autoCropPage(pageId) {
  const { data } = await api.post(`/documents/pages/${pageId}/crop/auto`);
  return data;
}

// ---- Corrections ----

export async function submitCorrection(ocrResultId, text) {
  const { data } = await api.post('/corrections', {
    ocr_result_id: ocrResultId,
    corrected_text: text,
  });
  return data;
}

export async function getPlayItems() {
  const { data } = await api.get('/corrections/play');
  return data;
}

export async function submitPlayCorrection(ocrResultId, text, correctedBbox = null) {
  const body = {
    ocr_result_id: ocrResultId,
    corrected_text: text,
  };
  if (correctedBbox) {
    body.corrected_bbox_x = correctedBbox.x;
    body.corrected_bbox_y = correctedBbox.y;
    body.corrected_bbox_w = correctedBbox.w;
    body.corrected_bbox_h = correctedBbox.h;
  }
  const { data } = await api.post('/corrections/play/submit', body);
  return data;
}

export async function getProcessingStatus() {
  const { data } = await api.get('/ocr/processing-status');
  return data;
}

// ---- Search ----

export async function search(query) {
  const { data } = await api.get('/search', { params: { q: query } });
  return data;
}

// ---- Google Photos Picker ----

export async function createPickerSession() {
  const { data } = await api.post('/photos/picker/session');
  return data;
}

export async function pollPickerSession(sessionId) {
  const { data } = await api.get(`/photos/picker/session/${sessionId}`);
  return data;
}

export async function importFromPicker(sessionId, documentName) {
  const { data } = await api.post('/photos/picker/import', {
    session_id: sessionId,
    document_name: documentName || 'Google Photos Import',
  });
  return data;
}

// ---- Model ----

export async function getModelStatus() {
  const { data } = await api.get('/model/status');
  return data;
}

export async function trainModel() {
  const { data } = await api.post('/model/train');
  return data;
}

export async function exportModel() {
  const response = await api.get('/model/export', { responseType: 'blob' });
  // Trigger download
  const url = window.URL.createObjectURL(new Blob([response.data]));
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', 'model.zip');
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
  return true;
}

// ---- Calibration ----

export async function submitCalibration(pageId, bbox, groundTruth) {
  const { data } = await api.post('/model/calibrate', {
    page_id: pageId,
    bbox_x: bbox.x,
    bbox_y: bbox.y,
    bbox_w: bbox.w,
    bbox_h: bbox.h,
    ground_truth: groundTruth,
  });
  return data;
}

export default api;
