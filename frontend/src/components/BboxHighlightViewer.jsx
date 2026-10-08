import { useState, useRef, useEffect, useCallback } from 'react';

export default function BboxHighlightViewer({ imageSrc, bbox, drawMode = false, onDrawBbox }) {
  const containerRef = useRef(null);
  const imgRef = useRef(null);
  const [imgDims, setImgDims] = useState({ width: 0, height: 0, naturalWidth: 0, naturalHeight: 0 });
  const [loaded, setLoaded] = useState(false);
  const [drawing, setDrawing] = useState(null); // { startX, startY, currentX, currentY } in natural coords

  // Reset loaded state when image source changes (e.g., after rotation)
  useEffect(() => {
    setLoaded(false);
  }, [imageSrc]);

  const updateDimensions = useCallback(() => {
    if (!imgRef.current) return;
    const img = imgRef.current;
    setImgDims({
      width: img.clientWidth,
      height: img.clientHeight,
      naturalWidth: img.naturalWidth,
      naturalHeight: img.naturalHeight,
    });
  }, []);

  useEffect(() => {
    if (!loaded) return;
    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, [loaded, updateDimensions]);

  const handleImageLoad = () => {
    setLoaded(true);
    updateDimensions();
  };

  const scaleX = imgDims.naturalWidth > 0 ? imgDims.width / imgDims.naturalWidth : 1;
  const scaleY = imgDims.naturalHeight > 0 ? imgDims.height / imgDims.naturalHeight : 1;

  const toNatural = useCallback((clientX, clientY) => {
    if (!imgRef.current) return { x: 0, y: 0 };
    const rect = imgRef.current.getBoundingClientRect();
    const x = Math.round((clientX - rect.left) / scaleX);
    const y = Math.round((clientY - rect.top) / scaleY);
    return {
      x: Math.max(0, Math.min(x, imgDims.naturalWidth)),
      y: Math.max(0, Math.min(y, imgDims.naturalHeight)),
    };
  }, [scaleX, scaleY, imgDims.naturalWidth, imgDims.naturalHeight]);

  // ── Mouse handlers ──
  const handleMouseDown = useCallback((e) => {
    if (!drawMode) return;
    e.preventDefault();
    const pos = toNatural(e.clientX, e.clientY);
    setDrawing({ startX: pos.x, startY: pos.y, currentX: pos.x, currentY: pos.y });
  }, [drawMode, toNatural]);

  const handleMouseMove = useCallback((e) => {
    if (!drawing) return;
    const pos = toNatural(e.clientX, e.clientY);
    setDrawing(prev => ({ ...prev, currentX: pos.x, currentY: pos.y }));
  }, [drawing, toNatural]);

  const handleMouseUp = useCallback(() => {
    if (!drawing) return;
    const x = Math.min(drawing.startX, drawing.currentX);
    const y = Math.min(drawing.startY, drawing.currentY);
    const w = Math.abs(drawing.currentX - drawing.startX);
    const h = Math.abs(drawing.currentY - drawing.startY);
    if (w > 5 && h > 5) {
      onDrawBbox?.({ x, y, w, h });
    }
    setDrawing(null);
  }, [drawing, onDrawBbox]);

  // ── Touch handlers (mobile draw support) ──
  const handleTouchStart = useCallback((e) => {
    if (!drawMode) return;
    e.preventDefault();
    const touch = e.touches[0];
    const pos = toNatural(touch.clientX, touch.clientY);
    setDrawing({ startX: pos.x, startY: pos.y, currentX: pos.x, currentY: pos.y });
  }, [drawMode, toNatural]);

  const handleTouchMove = useCallback((e) => {
    if (!drawing) return;
    e.preventDefault();
    const touch = e.touches[0];
    const pos = toNatural(touch.clientX, touch.clientY);
    setDrawing(prev => ({ ...prev, currentX: pos.x, currentY: pos.y }));
  }, [drawing, toNatural]);

  const handleTouchEnd = useCallback(() => {
    handleMouseUp(); // reuse same finalize logic
  }, [handleMouseUp]);

  // Compute draw rect in display coords
  const drawRect = drawing ? {
    left: Math.min(drawing.startX, drawing.currentX) * scaleX,
    top: Math.min(drawing.startY, drawing.currentY) * scaleY,
    width: Math.abs(drawing.currentX - drawing.startX) * scaleX,
    height: Math.abs(drawing.currentY - drawing.startY) * scaleY,
  } : null;

  // Bbox in display coords
  const bboxDisplay = bbox ? {
    left: bbox.x * scaleX,
    top: bbox.y * scaleY,
    width: bbox.w * scaleX,
    height: bbox.h * scaleY,
  } : null;

  // Zoom scale for the cropped view
  const zoomScale = bbox && bbox.w > 0 ? Math.min(400, bbox.w * 3) / bbox.w : 1;

  return (
    <div className="space-y-3">
      {/* Full page with highlight */}
      <div
        ref={containerRef}
        className={`relative inline-block w-full ${drawMode ? 'cursor-crosshair' : ''}`}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
      >
        {!loaded && (
          <div className="skeleton w-full" style={{ paddingBottom: '141.4%' }} />
        )}
        <img
          ref={imgRef}
          src={imageSrc}
          alt="Document page"
          loading="lazy"
          onLoad={handleImageLoad}
          className={`w-full h-auto block transition-opacity duration-300 ${loaded ? 'opacity-100' : 'opacity-0'}`}
          draggable={false}
        />
        {/* Dark overlay outside bbox */}
        {loaded && bboxDisplay && !drawMode && (
          <>
            {/* Top */}
            <div className="absolute inset-x-0 top-0 bg-black/40" style={{ height: `${bboxDisplay.top}px` }} />
            {/* Bottom */}
            <div className="absolute inset-x-0 bg-black/40" style={{ top: `${bboxDisplay.top + bboxDisplay.height}px`, bottom: 0 }} />
            {/* Left */}
            <div className="absolute bg-black/40" style={{ top: `${bboxDisplay.top}px`, left: 0, width: `${bboxDisplay.left}px`, height: `${bboxDisplay.height}px` }} />
            {/* Right */}
            <div className="absolute bg-black/40" style={{ top: `${bboxDisplay.top}px`, left: `${bboxDisplay.left + bboxDisplay.width}px`, right: 0, height: `${bboxDisplay.height}px` }} />
            {/* Yellow border */}
            <div
              className="absolute border-2 border-yellow-400 shadow-lg"
              style={{
                left: `${bboxDisplay.left}px`,
                top: `${bboxDisplay.top}px`,
                width: `${bboxDisplay.width}px`,
                height: `${bboxDisplay.height}px`,
              }}
            />
          </>
        )}
        {/* Drawing rectangle */}
        {loaded && drawRect && (
          <div
            className="absolute border-2 border-dashed border-blue-500 bg-blue-500/10"
            style={{
              left: `${drawRect.left}px`,
              top: `${drawRect.top}px`,
              width: `${drawRect.width}px`,
              height: `${drawRect.height}px`,
            }}
          />
        )}
      </div>

      {/* Zoomed crop of the bbox region */}
      {loaded && bbox && imgDims.naturalWidth > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-2 overflow-hidden">
          <div className="text-xs text-gray-400 mb-1 font-medium uppercase tracking-wide">Zoomed Region</div>
          <div className="overflow-hidden rounded" style={{ maxHeight: '120px' }}>
            <div
              style={{
                width: `${bbox.w * zoomScale}px`,
                height: `${bbox.h * zoomScale}px`,
                overflow: 'hidden',
                position: 'relative',
                maxWidth: '100%',
              }}
            >
              <img
                src={imageSrc}
                alt="Zoomed region"
                style={{
                  position: 'absolute',
                  width: `${imgDims.naturalWidth * zoomScale}px`,
                  left: `-${bbox.x * zoomScale}px`,
                  top: `-${bbox.y * zoomScale}px`,
                }}
                draggable={false}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
